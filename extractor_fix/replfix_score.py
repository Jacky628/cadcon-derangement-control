#!/usr/bin/env python
"""replfix_score.py — offline scoring of the repaired-extractor run on BOTH scoring surfaces.

FREEZE ARTIFACT. Pre-registration:
`_paper_artifacts_rc-fdedd0/提取器修复_独立预注册_20260812.md` §3, §3.1, §7.1, §8.

Input : merge-dir/adherence_samples.jsonl produced by replfix_generate.py (raw completions,
        no extraction applied).
Output: replfix_scores.jsonl — one row per (condition, seed, idx) carrying, for the SAME
        generation, the scored program and geometry verdict under
          * Rule A       (this pre-registration's surface), and
          * the legacy rule (the published surface),
        plus both `executable` definitions (§7.1) and the forensic fields (§8).

Two surfaces, one set of generations: the scoring rule is the only thing that varies, so the
comparison carries no generation noise. This is also what makes the §3 bit-identity self-check
possible — `replfix_selfcheck.py` compares the legacy columns here against the published
`geom_scores.jsonl`.

Execution budget: each cut is executed at most once and the verdict is reused for scoring, so
a row costs at most n_cuts subprocess launches, not 2 x n_cuts.

Usage:
  python replfix_score.py --in MERGE/adherence_samples.jsonl --out replfix_scores.jsonl \
                          [--workers 14]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import replfix_extractor as X
from geom_requirements import TAGS_PRIMARY, geom_features

# The primary metric scores the four-tag space; MULTI_PART is derived but not scored
# (mother pre-registration §3 / Cspike §2.1). Kept as a module constant so the analysis
# script cannot disagree with the scorer about what "the metric" is.
SCORED_TAGS = tuple(TAGS_PRIMARY)


def _adherence(intended: set, produced: set | None) -> float | None:
    """Mean per-sample positive-label recall over the scored tags.

    `produced is None` means the program did not execute; under execution-inclusive scoring
    that is a zero on every requirement, not a missing value. Returns None only when the
    sample has no scored intent at all (excluded from the mean, as in the mother protocol).
    """
    want = {t for t in intended if t in SCORED_TAGS}
    if not want:
        return None
    if produced is None:
        return 0.0
    return len(want & set(produced)) / len(want)


def prefix_key(gt_code: str, prefix_fraction: float) -> str:
    """Cluster key: the code prefix the model was actually shown, header excluded.

    The 400 programs collapse to 298 distinct prefixes, and the header-free arms cannot
    distinguish programs inside a cluster at all. The cluster is the pre-registered primary
    unit (§5.1). Slicing is byte-identical to the eval path in main.eval_validity.
    """
    if prefix_fraction <= 0.0:
        split = 0
    else:
        split = max(1, min(int(len(gt_code) * prefix_fraction), len(gt_code) - 1))
    pref = gt_code[:split]
    if pref == gt_code and len(gt_code) > 1:
        pref = gt_code[:len(gt_code) - 1]
    return hashlib.sha256(pref.encode()).hexdigest()[:16]


def score_row(row: dict) -> dict:
    """Score one generation under both extraction rules."""
    full = row["prompt"] + row["completion_at_eos"]

    # Cache keyed by program text: Rule A's cut search and the final scoring share verdicts.
    verdicts: dict[str, tuple] = {}

    def execute(program: str) -> bool:
        if program not in verdicts:
            verdicts[program] = geom_features(program)
        return verdicts[program][0] is not None

    res = X.extract_rule_a(full, executor=execute)

    def verdict_for(program: str) -> tuple:
        if program not in verdicts:
            verdicts[program] = geom_features(program)
        return verdicts[program]

    tags_a, status_a = verdict_for(res["program"])
    tags_l, status_l = verdict_for(res["program_legacy"])

    intended = set(row["intended"])
    out = {
        # identity + pairing keys
        "condition": row["condition"], "seed": row["seed"], "idx": row["idx"],
        "intended": sorted(intended),
        "header_source": row["header_source"],
        "header_injected": row.get("header_injected", []),
        "prefix_fraction": row.get("prefix_fraction"),
        "prefix_features": row.get("prefix_features", []),
        "prefix_key": prefix_key(row["gt_code"], row.get("prefix_fraction", 0.4)),

        # ── Rule A surface (this pre-registration) ──
        "executable": tags_a is not None,          # == exec_any_cut by construction
        "exec_status": status_a,
        "geom_produced": sorted(tags_a) if tags_a is not None else [],
        "recall": _adherence(intended, tags_a),

        # ── legacy surface (the published one), same generation ──
        "legacy_executable": tags_l is not None,   # == exec_first_block by construction
        "legacy_exec_status": status_l,
        "legacy_geom_produced": sorted(tags_l) if tags_l is not None else [],
        "legacy_recall": _adherence(intended, tags_l),

        # ── extraction diagnostics (§6 guardrail d, §7.1) ──
        "n_cuts": res["n_cuts"],
        "cut_index": res["cut_index"],
        "fell_back": res["fell_back"],
        "same_as_legacy": res["same_as_legacy"],
        "exec_first_block": res["exec_first_block"],
        "exec_any_cut": res["exec_any_cut"],
        "n_executions": len(verdicts),

        # ── forensics carried through (§8) ──
        "eos_emitted": row.get("eos_emitted"),
        "eos_position": row.get("eos_position"),
        "completion_tokens": row.get("completion_tokens"),
        "cap_hit": row.get("cap_hit"),
    }

    # Structural properties must hold on every real row, not just in the unit tests.
    assert out["exec_first_block"] == out["legacy_executable"], (
        f"{out['condition']} idx={out['idx']}: exec_first_block disagrees with the legacy "
        "verdict; the two must be the same execution of the same program")
    assert out["exec_any_cut"] == out["executable"]
    assert not (out["exec_first_block"] and not out["exec_any_cut"]), "property 1 violated"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="replfix_scores.jsonl")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.inp).read_text().splitlines() if l.strip()]
    missing = [i for i, r in enumerate(rows) if "completion_at_eos" not in r]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)}/{len(rows)} rows lack `completion_at_eos`. This input was "
            "not produced by replfix_generate.py; scoring it would compare different things.")

    keys = [(r["condition"], r["seed"], r["idx"]) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate (condition, seed, idx) rows in input"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        scored = list(ex.map(score_row, rows))

    with open(a.out, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    n = len(scored)
    multi = [s for s in scored if s["n_cuts"] >= 2]
    same_multi = sum(s["same_as_legacy"] for s in multi)
    lift = sum(s["exec_any_cut"] for s in scored) - sum(s["exec_first_block"] for s in scored)
    print(f"scored {n} rows -> {a.out}")
    print(f"  conditions            : {len(set(s['condition'] for s in scored))}")
    print(f"  executions performed  : {sum(s['n_executions'] for s in scored)}")
    print(f"  multi-cut rows        : {len(multi)}/{n}")
    print(f"  guardrail d (same_as_legacy | n_cuts>=2): "
          f"{same_multi}/{len(multi)} = {100*same_multi/max(len(multi),1):.1f}%  (fails if >50%)")
    print(f"  executability lift    : +{lift} rows ({100*lift/n:+.2f} pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
