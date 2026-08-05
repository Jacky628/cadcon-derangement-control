#!/usr/bin/env python3
"""Submission-stage audit for the isolated TAE v2 manuscript.

This script never mutates the frozen preregistration artifacts. It adds three
post-hoc checks that the workshop manuscript must disclose:

1. Collapse duplicated held-out ground truths to unique-program clusters
   before paired Wilcoxon tests.
2. Decompose detector choice from execution gating on the same generated
   outputs (regex/no gate, regex/execute gate, geometry/execute gate).
3. Remove wrong-header assignments whose only semantic change is the
   unscored MULTI_PART feature, then repeat the duplicate-aware wrong-vs-none
   and Test D analyses.

The v2.2 addendum adds two disclosures on the same frozen data:

4. A fixed-seed sign-flip permutation test (mean statistic, 200,000
   resamples) on the same cluster means, corroborating the asymptotic
   Pratt-Wilcoxon p-values at the small effective sample size.
5. Within-pair outcome agreement across the 40%-prefix stress-test cells,
   quantifying how close the adjacent duplicate entries are to exact
   evaluation replicas.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats


RELEASE = Path(__file__).resolve().parents[1]


OUT = Path(__file__).with_name("submission_audit.json")

FROZEN = RELEASE / "analysis" / "spike_frozen_lists.json"
SPIKE = RELEASE / "data" / "spike_results" / "geom_scores.jsonl"
F4 = RELEASE / "data" / "f4_results" / "geom_scores.jsonl"
TAGS = {"<CIRCLE>", "<NGON>", "<THIN>", "<TALL>"}
MULTI_PART = "<MULTI_PART>"
SEEDS = (0, 1, 2)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recall(intended: list[str], produced: list[str]) -> float | None:
    target = set(intended) & TAGS
    if not target:
        return None
    return len(target & set(produced)) / len(target)


def row_score(row: dict, detector: str = "geom", execution_gate: bool = True) -> float | None:
    if execution_gate and not row["executable"]:
        produced = []
    else:
        produced = row["geom_produced"] if detector == "geom" else row["produced"]
    return recall(row["intended"], produced)


def index_rows(rows: list[dict], n76: set[int]) -> dict[tuple[str, int, int], dict]:
    out = {}
    for row in rows:
        if row["idx"] in n76:
            out[(row["condition"], row["seed"], row["idx"])] = row
    return out


def cluster(values: dict[int, float], hashes: dict[int, str]) -> np.ndarray:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for idx, value in values.items():
        grouped[hashes[idx]].append(value)
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)


def cluster_size_counts(indices: set[int], hashes: dict[int, str]) -> dict[str, int]:
    counts = collections.Counter(hashes[idx] for idx in indices)
    return {
        str(size): count
        for size, count in sorted(collections.Counter(counts.values()).items())
    }


PERMUTATION_SEED = 20260804
PERMUTATION_RESAMPLES = 200_000


def perm_rng(label: str) -> np.random.Generator:
    """A generator whose stream depends only on the frozen seed and the test
    label, so adding or reordering tests cannot change earlier p-values."""
    import zlib

    return np.random.default_rng([PERMUTATION_SEED, zlib.crc32(label.encode())])


def signflip_p(values: np.ndarray, alternative: str, rng: np.random.Generator) -> float | None:
    """Sign-flip permutation p-value for the mean of paired cluster differences.

    Under the null of a symmetric difference distribution about zero, each
    cluster mean keeps its magnitude with a random sign. The add-one estimate
    keeps the p-value away from an impossible exact zero.
    """
    if not len(values) or np.all(values == 0):
        return None
    observed = float(np.mean(values))
    flips = rng.choice([-1.0, 1.0], size=(PERMUTATION_RESAMPLES, len(values)))
    null = (flips * np.abs(values)).mean(axis=1)
    if alternative == "less":
        exceed = int(np.sum(null <= observed))
    else:
        exceed = int(np.sum(null >= observed))
    return (exceed + 1) / (PERMUTATION_RESAMPLES + 1)


def wilcoxon(values: np.ndarray, alternative: str) -> float | None:
    if not len(values) or np.all(values == 0):
        return None
    return float(
        stats.wilcoxon(
            values,
            zero_method="pratt",
            alternative=alternative,
            method="asymptotic",
        ).pvalue
    )


def get_score(index: dict, condition: str, seed: int, idx: int) -> float:
    score = row_score(index[(condition, seed, idx)])
    assert score is not None
    return score


def paired_cluster_test(
    index: dict,
    left: str,
    right: str,
    seed: int,
    indices: set[int],
    hashes: dict[int, str],
    alternative: str,
    perm_label: str | None = None,
) -> dict:
    raw = {
        idx: get_score(index, left, seed, idx) - get_score(index, right, seed, idx)
        for idx in sorted(indices)
    }
    clustered = cluster(raw, hashes)
    result = {
        "n_entries": len(raw),
        "n_unique_programs": len(clustered),
        "mean": float(np.mean(clustered)),
        "p_pratt_asymptotic": wilcoxon(clustered, alternative),
        "nonzero_clusters": int(np.count_nonzero(clustered)),
    }
    if perm_label is not None:
        result["p_signflip_mean"] = signflip_p(clustered, alternative, perm_rng(perm_label))
    return result


def multipart_only_wrong_indices(
    index: dict[tuple[str, int, int], dict],
    condition: str,
    seed: int,
    indices: set[int],
) -> set[int]:
    """Wrong assignments that are identical on every scored primary tag."""
    excluded = set()
    for idx in sorted(indices):
        row = index[(condition, seed, idx)]
        delta = set(row["intended"]) ^ set(row["header_injected"])
        assert delta, f"wrong header unexpectedly equals intended header: {condition=} {seed=} {idx=}"
        if delta == {MULTI_PART}:
            excluded.add(idx)
    return excluded


def sensitivity_metadata(
    original: set[int],
    excluded: set[int],
    hashes: dict[int, str],
) -> dict:
    retained = original - excluded
    return {
        "n_entries_before": len(original),
        "n_excluded_multipart_only": len(excluded),
        "excluded_indices": sorted(excluded),
        "n_entries_after": len(retained),
        "n_unique_programs_after": len({hashes[idx] for idx in retained}),
        "retained_cluster_size_counts": cluster_size_counts(retained, hashes),
    }


def test_d_clustered(
    spike_index: dict,
    f4_index: dict,
    seed: int,
    indices: set[int],
    hashes: dict[int, str],
    perm_label: str | None = None,
) -> dict:
    drop_m = {}
    drop_r = {}
    for idx in sorted(indices):
        drop_m[idx] = get_score(spike_index, "spike_text_correct_p04", seed, idx) - get_score(
            spike_index, "spike_text_shuffled_p04", seed, idx
        )
        drop_r[idx] = get_score(f4_index, f"f4_R_correct_p04_s{seed}", seed, idx) - get_score(
            f4_index, f"f4_R_shuffled_p04_s{seed}", seed, idx
        )
    interaction = {
        idx: drop_m[idx] - drop_r[idx]
        for idx in sorted(indices)
    }
    clustered_m = cluster(drop_m, hashes)
    clustered_r = cluster(drop_r, hashes)
    clustered_interaction = cluster(interaction, hashes)
    np.testing.assert_allclose(clustered_interaction, clustered_m - clustered_r)
    mean_m = float(np.mean(clustered_m))
    mean_r = float(np.mean(clustered_r))
    result = {
        "n_entries": len(interaction),
        "n_unique_programs": len(clustered_interaction),
        # Stable aliases consumed directly by make_tae_figure.py.
        "drop_M_mean": mean_m,
        "drop_R_mean": mean_r,
        "m_correct_to_wrong_drop_mean": mean_m,
        "r_correct_to_wrong_drop_mean": mean_r,
        "mean": float(np.mean(clustered_interaction)),
        "p_pratt_asymptotic": wilcoxon(clustered_interaction, "greater"),
        "nonzero_clusters": int(np.count_nonzero(clustered_interaction)),
    }
    if perm_label is not None:
        result["p_signflip_mean"] = signflip_p(
            clustered_interaction, "greater", perm_rng(perm_label)
        )
    return result


def within_pair_agreement(
    rows: list[dict], n76: set[int], hashes: dict[int, str]
) -> dict:
    """How often the two entries of a duplicate pair receive identical
    outcomes across the 40%-prefix stress-test cells (primary subset)."""
    pair_of: dict[str, list[int]] = collections.defaultdict(list)
    for idx in sorted(n76):
        pair_of[hashes[idx]].append(idx)
    pairs = [tuple(v) for v in pair_of.values() if len(v) == 2]
    indexed = {(r["condition"], r["seed"], r["idx"]): r for r in rows}
    conditions = sorted({r["condition"] for r in rows if "_p04" in r["condition"]})
    total = same_exec = same_geom_score = 0
    for condition in conditions:
        seeds = sorted({r["seed"] for r in rows if r["condition"] == condition})
        for seed in seeds:
            for a, b in pairs:
                ra = indexed.get((condition, seed, a))
                rb = indexed.get((condition, seed, b))
                if ra is None or rb is None:
                    continue
                total += 1
                same_exec += ra["executable"] == rb["executable"]
                same_geom_score += row_score(ra) == row_score(rb)
    per_condition = {}
    for condition in conditions:
        seeds = sorted({r["seed"] for r in rows if r["condition"] == condition})
        cells = agree = 0
        for seed in seeds:
            for a, b in pairs:
                ra = indexed.get((condition, seed, a))
                rb = indexed.get((condition, seed, b))
                if ra is None or rb is None:
                    continue
                cells += 1
                agree += row_score(ra) == row_score(rb)
        per_condition[condition] = {
            "cells": cells,
            "same_gated_geometry_score": agree,
            "fraction": agree / cells,
        }
    return {
        "cells": "spike *_p04 conditions x seeds x 38 primary pairs",
        "n_pair_condition_seed": total,
        "same_executable": same_exec,
        "same_executable_fraction": same_exec / total,
        "same_gated_geometry_score": same_geom_score,
        "same_gated_geometry_score_fraction": same_geom_score / total,
        "per_condition": per_condition,
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, monotonicity enforced."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[name] = running
    return adjusted


def multiplicity(wrong: dict, test_d: dict) -> dict:
    """Holm adjustment over the nine carried-forward duplicate-aware tests."""
    raw = {}
    for mode in ("text", "token"):
        for seed in SEEDS:
            raw[f"{mode}{seed}"] = wrong[mode][str(seed)]["p_pratt_asymptotic"]
    for seed in SEEDS:
        raw[f"testD{seed}"] = test_d[str(seed)]["p_pratt_asymptotic"]
    families = {
        "text": [f"text{s}" for s in SEEDS],
        "token": [f"token{s}" for s in SEEDS],
        "test_d": [f"testD{s}" for s in SEEDS],
    }
    within = {}
    for keys in families.values():
        within.update(holm({k: raw[k] for k in keys}))
    return {
        "unadjusted": raw,
        "holm_within_family": within,
        "holm_global": holm(raw),
        "families": families,
    }


def cluster_representative_sensitivity(
    index: dict, n76: set[int], hashes: dict[int, str]
) -> dict:
    """Averaging within clusters versus keeping one arbitrary pair member."""
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for idx in sorted(n76):
        grouped[hashes[idx]].append(idx)
    members = {
        "first": {sorted(v)[0] for v in grouped.values()},
        "second": {sorted(v)[1] for v in grouped.values()},
    }
    result = {}
    for mode in ("text", "token"):
        result[mode] = {}
        for label in ("average", "first", "second"):
            per_seed = {}
            for seed in SEEDS:
                condition = f"spike_{mode}_shuffled_p04"
                if label == "average":
                    per_seed[str(seed)] = paired_cluster_test(
                        index, condition, "spike_baseline_p04", seed, n76, hashes, "less"
                    )["p_pratt_asymptotic"]
                else:
                    kept = members[label]
                    diffs = np.asarray(
                        [
                            get_score(index, condition, seed, idx)
                            - get_score(index, "spike_baseline_p04", seed, idx)
                            for idx in sorted(kept)
                        ]
                    )
                    per_seed[str(seed)] = wilcoxon(diffs, "less")
            result[mode][label] = {
                "p_pratt_asymptotic": per_seed,
                "n_significant_seeds": sum(
                    1 for p in per_seed.values() if p is not None and p < 0.05
                ),
            }
    return result


def detector_gate_decomposition(index: dict, n76: set[int]) -> dict:
    conditions = {
        "baseline": "spike_baseline_p04",
        "text_correct": "spike_text_correct_p04",
        "token_correct": "spike_token_correct_p04",
    }
    result = {}
    for label, condition in conditions.items():
        cells = {"regex_no_gate": [], "regex_exec_gate": [], "geom_exec_gate": []}
        for seed in SEEDS:
            for idx in sorted(n76):
                row = index[(condition, seed, idx)]
                cells["regex_no_gate"].append(row_score(row, "regex", False))
                cells["regex_exec_gate"].append(row_score(row, "regex", True))
                cells["geom_exec_gate"].append(row_score(row, "geom", True))
        result[label] = {key: float(np.mean(value)) for key, value in cells.items()}

    result["gain_over_baseline"] = {}
    for label in ("text_correct", "token_correct"):
        result["gain_over_baseline"][label] = {
            key: result[label][key] - result["baseline"][key]
            for key in ("regex_no_gate", "regex_exec_gate", "geom_exec_gate")
        }

    # Directly quantify produced-side detector disagreement on executable rows.
    executable = 0
    disagreements = 0
    for (condition, seed, idx), row in index.items():
        if not row["executable"]:
            continue
        executable += 1
        if row_score(row, "regex", True) != row_score(row, "geom", True):
            disagreements += 1
    result["executable_detector_comparison"] = {
        "n": executable,
        "different_scores": disagreements,
    }
    return result


def executability(rows: list[dict], condition_prefix: str, idx_filter: set[int] | None = None) -> float:
    selected = [
        row
        for row in rows
        if row["condition"].startswith(condition_prefix)
        and (idx_filter is None or row["idx"] in idx_filter)
    ]
    return float(np.mean([row["executable"] for row in selected]))


def main() -> None:
    frozen = json.loads(FROZEN.read_text())
    n76 = set(frozen["analysis_idx_n76"])
    hashes = {row["idx"]: row["gt_sha256"] for row in frozen["testset_gt_sha256"]}
    all_hash_counts = collections.Counter(hashes.values())
    primary_hash_counts = collections.Counter(hashes[idx] for idx in n76)

    spike_rows = load_jsonl(SPIKE)
    f4_rows = load_jsonl(F4)
    spike_index = index_rows(spike_rows, n76)
    f4_index = index_rows(f4_rows, n76)

    wrong = {"text": {}, "token": {}}
    wrong_multipart_sensitivity = {"text": {}, "token": {}}
    multipart_only_by_mode: dict[str, dict[int, set[int]]] = {
        "text": {},
        "token": {},
    }
    for mode in ("text", "token"):
        for seed in SEEDS:
            condition = f"spike_{mode}_shuffled_p04"
            wrong[mode][str(seed)] = paired_cluster_test(
                spike_index,
                condition,
                "spike_baseline_p04",
                seed,
                n76,
                hashes,
                "less",
                perm_label=f"wrong_vs_baseline/{mode}/{seed}",
            )

            excluded = multipart_only_wrong_indices(
                spike_index, condition, seed, n76
            )
            multipart_only_by_mode[mode][seed] = excluded
            retained = n76 - excluded
            sensitivity = paired_cluster_test(
                spike_index,
                condition,
                "spike_baseline_p04",
                seed,
                retained,
                hashes,
                "less",
            )
            sensitivity.update(sensitivity_metadata(n76, excluded, hashes))
            wrong_multipart_sensitivity[mode][str(seed)] = sensitivity

    test_d = {}
    test_d_multipart_sensitivity = {}
    for seed in SEEDS:
        excluded = multipart_only_by_mode["text"][seed]

        # Test D must use the same injected wrong header for M and R.
        for idx in sorted(n76):
            m_wrong = spike_index[("spike_text_shuffled_p04", seed, idx)]
            r_wrong = f4_index[(f"f4_R_shuffled_p04_s{seed}", seed, idx)]
            assert set(m_wrong["header_injected"]) == set(r_wrong["header_injected"])

        test_d[str(seed)] = test_d_clustered(
            spike_index, f4_index, seed, n76, hashes,
            perm_label=f"test_d/{seed}",
        )
        sensitivity = test_d_clustered(
            spike_index, f4_index, seed, n76 - excluded, hashes
        )
        sensitivity.update(sensitivity_metadata(n76, excluded, hashes))
        test_d_multipart_sensitivity[str(seed)] = sensitivity

    # Text and token variants share the seed-fixed wrong-header assignments.
    for seed in SEEDS:
        assert multipart_only_by_mode["text"][seed] == multipart_only_by_mode["token"][seed]

    figure2_drops = {
        seed: {
            "m_correct_to_wrong_drop": values["m_correct_to_wrong_drop_mean"],
            "r_correct_to_wrong_drop": values["r_correct_to_wrong_drop_mean"],
            "interaction": values["mean"],
            "test_d_p_pratt_asymptotic": values["p_pratt_asymptotic"],
        }
        for seed, values in test_d.items()
    }

    result = {
        "heldout_duplication": {
            "entries": len(hashes),
            "unique_programs": len(all_hash_counts),
            "all_cluster_sizes": sorted(set(all_hash_counts.values())),
            "primary_entries": len(n76),
            "primary_unique_programs": len(primary_hash_counts),
            "primary_cluster_sizes": sorted(set(primary_hash_counts.values())),
        },
        "within_pair_agreement": within_pair_agreement(spike_rows, n76, hashes),
        "multiplicity_holm": multiplicity(wrong, test_d),
        "cluster_representative_sensitivity": cluster_representative_sensitivity(
            spike_index, n76, hashes
        ),
        "detector_gate_decomposition": detector_gate_decomposition(spike_index, n76),
        "duplicate_aware_wrong_vs_baseline": wrong,
        "duplicate_aware_wrong_vs_baseline_excluding_multipart_only": wrong_multipart_sensitivity,
        "duplicate_aware_test_d": test_d,
        "duplicate_aware_test_d_excluding_multipart_only": test_d_multipart_sensitivity,
        "figure2_correct_to_wrong_drop": {
            "analysis": "duplicate-aware full primary subset",
            "per_seed": figure2_drops,
        },
        "multipart_only_sensitivity_definition": {
            "criterion": "set(intended) symmetric_difference set(header_injected) == {<MULTI_PART>}",
            "procedure": "exclude assignment-level rows, then average retained rows within ground-truth SHA-256 clusters",
            "test_d_m_and_r_wrong_assignment_identical": True,
            "text_and_token_excluded_indices_identical": True,
        },
        "executability_sensitivity": {
            "frozen_code_n76_R_correct_wrong_pooled": float(
                np.mean(
                    [
                        row["executable"]
                        for row in f4_rows
                        if row["idx"] in n76
                        and (
                            row["condition"].startswith("f4_R_correct_p04")
                            or row["condition"].startswith("f4_R_shuffled_p04")
                        )
                    ]
                )
            ),
            "frozen_code_n76_baseline": executability(spike_rows, "spike_baseline_p04", n76),
            "full100_R_correct_wrong_pooled": float(
                np.mean(
                    [
                        row["executable"]
                        for row in f4_rows
                        if row["condition"].startswith("f4_R_correct_p04")
                        or row["condition"].startswith("f4_R_shuffled_p04")
                    ]
                )
            ),
            "full100_baseline": executability(spike_rows, "spike_baseline_p04"),
        },
    }

    # Guard against silently reporting a different data version.
    assert result["heldout_duplication"] == {
        "entries": 100,
        "unique_programs": 50,
        "all_cluster_sizes": [2],
        "primary_entries": 76,
        "primary_unique_programs": 38,
        "primary_cluster_sizes": [2],
    }
    assert {
        str(seed): result["duplicate_aware_test_d_excluding_multipart_only"][str(seed)][
            "n_excluded_multipart_only"
        ]
        for seed in SEEDS
    } == {"0": 4, "1": 10, "2": 5}

    # Values printed in Appendix B must come from this run, not from a stale draft.
    published_holm = {
        "text0": (0.1272, 0.1272), "text1": (0.1272, 0.1272), "text2": (0.0683, 0.0911),
        "token0": (8.49e-4, 2.26e-3), "token1": (1.24e-3, 3.88e-3), "token2": (8.47e-3, 0.0423),
        "testD0": (0.0237, 0.0911), "testD1": (1.11e-3, 3.88e-3), "testD2": (4.63e-4, 1.39e-3),
    }
    holm_result = result["multiplicity_holm"]
    for key, (within_p, global_p) in published_holm.items():
        for label, expected in (("holm_within_family", within_p), ("holm_global", global_p)):
            actual = holm_result[label][key]
            assert abs(actual - expected) <= 5e-5 * max(1.0, expected / 1e-3), (
                f"Table B.5 {label}[{key}]: script {actual!r} vs manuscript {expected!r}"
            )
    published_significant = {
        ("text", "average"): 1, ("text", "first"): 2, ("text", "second"): 1,
        ("token", "average"): 3, ("token", "first"): 3, ("token", "second"): 3,
    }
    sensitivity = result["cluster_representative_sensitivity"]
    for (mode, label), expected in published_significant.items():
        actual = sensitivity[mode][label]["n_significant_seeds"]
        assert actual == expected, f"representative sensitivity {mode}/{label}: {actual} != {expected}"
    per_condition = result["within_pair_agreement"]["per_condition"]
    for condition, expected in (
        ("spike_text_shuffled_p04", 0.816), ("spike_token_shuffled_p04", 0.825),
    ):
        actual = per_condition[condition]["fraction"]
        assert abs(actual - expected) < 5e-4, f"{condition} agreement {actual} != {expected}"
    assert min(
        per_condition[c]["fraction"]
        for c in per_condition
        if "shuffled" not in c
    ) >= 0.99, "correct/masked/baseline within-pair agreement fell below the stated 99-100%"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
