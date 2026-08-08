#!/usr/bin/env python
"""replication_analysis.py — frozen analysis for the replication run.

Same decision chain as the C-spike (`spike_analysis.py`, whose `recall_4tag` and `wtest` are
imported rather than reimplemented, so the test specification cannot drift), with three
changes that follow from the new sample:

  * The unit is one unique transpiled program. The v1 held-out sample carried each program
    twice, so `submission_audit.py` had to average within ground-truth clusters before
    testing; here `build_replication_sample.py` guarantees uniqueness and this script ASSERTS
    it, so no clustering step exists to get wrong.
  * n is read from the frozen list instead of being hard-coded at 76.
  * The non-zero-pair guardrail scales with n (0.15n) instead of the absolute 15 that was
    calibrated for n=76.

Inputs:
  replication_frozen_lists.json            (from build_replication_sample.py, frozen pre-run)
  <dir>/geom_scores.jsonl                  (from geom_requirements.py over the eval outputs)

Usage: .venv/bin/python replication_analysis.py [--dir repl_results]
                                                [--frozen replication_frozen_lists.json]
                                                [--out repl_verdict.json]
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spike_analysis import recall_4tag, wtest, load_jsonl   # frozen test spec, reused verbatim

MODES = ("token", "text")
ALL_PREFIXES = ("p04", "p00")      # decision cell first
HEADERS = ("shuffled", "masked")
SEEDS = (0, 1, 2)
ALPHA = 0.05
PREFIX_LABEL = "repl"


def build_scores(geom_rows, analysis_idx):
    """{condition: {seed: {idx: execution-inclusive 4-tag geometric recall}}}."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in geom_rows:
        if r["idx"] not in analysis_idx:
            continue
        produced = set(r["geom_produced"]) if r["executable"] else set()
        rec = recall_4tag(r["intended"], produced)
        if rec is None:
            continue                      # empty 4-tag intent: excluded by construction here
        if not r["executable"]:
            rec = 0.0                     # execution-inclusive primary rule
        out[r["condition"]][r["seed"]][r["idx"]] = rec
    return out


def paired_diffs(cond_scores, base_scores, seed):
    ca, ba = cond_scores.get(seed, {}), base_scores.get(seed, {})
    idx = sorted(set(ca) & set(ba))
    return [ca[i] - ba[i] for i in idx], idx


def seed_stats(cond_scores, base_scores, min_nonzero):
    per_seed = {}
    for s in SEEDS:
        d, idx = paired_diffs(cond_scores, base_scores, s)
        if not d:
            per_seed[s] = {"n": 0, "mean": None, "p_less": None, "p_greater": None,
                           "nonzero": 0, "underpowered": True, "note": "missing arm"}
            continue
        pl, nl = wtest(d, "less")
        pg, ng = wtest(d, "greater")
        nz = int(sum(1 for x in d if x != 0))
        per_seed[s] = {
            "n": len(d), "mean": float(np.mean(d)),
            "p_less": None if math.isnan(pl) else pl, "p_less_note": nl,
            "p_greater": None if math.isnan(pg) else pg, "p_greater_note": ng,
            "nonzero": nz, "underpowered": nz < min_nonzero,
        }
    return per_seed


GUARDRAIL_MIN_PROFILE_N = 15    # prereg A-4: profiles below this are reported, not counted


def profile_effect(cond_scores, base_scores, profiles, pool_sizes=None):
    """Seed-averaged per-profile mean effect.

    Returns (guardrail_mean, per_profile, extras). The guardrail (criterion 4) averages only
    profiles with >= GUARDRAIL_MIN_PROFILE_N programs, equally weighted — a 6-program profile
    carrying the same weight as a 112-program one would let its noise flip the guardrail
    (prereg A-4). Profiles below the floor are still computed and reported.

    `extras` additionally carries the pool-proportional weighted mean (prereg A-2): the
    stratified draw over-samples rare profiles, so the unweighted analysis generalises to a
    profile-balanced population. Weighting each profile by its share of the clean pool
    estimates the same contrast for the natural DeepCAD distribution. Report-only.
    """
    per_prof = []
    for prof in profiles:
        pid = set(prof["idx"])
        effs = []
        for s in SEEDS:
            d, _ = paired_diffs(
                {s: {i: v for i, v in cond_scores.get(s, {}).items() if i in pid}},
                {s: {i: v for i, v in base_scores.get(s, {}).items() if i in pid}}, s)
            if d:
                effs.append(float(np.mean(d)))
        if effs:
            per_prof.append({"profile": prof["profile"], "n_idx": len(pid),
                             "effect": float(np.mean(effs)),
                             "counted_in_guardrail": len(pid) >= GUARDRAIL_MIN_PROFILE_N})

    counted = [p["effect"] for p in per_prof if p["counted_in_guardrail"]]
    overall = float(np.mean(counted)) if counted else None

    extras = {
        "guardrail_min_profile_n": GUARDRAIL_MIN_PROFILE_N,
        "n_profiles_counted": len(counted),
        "n_profiles_below_floor": sum(1 for p in per_prof if not p["counted_in_guardrail"]),
        "unweighted_mean_all_profiles":
            float(np.mean([p["effect"] for p in per_prof])) if per_prof else None,
    }
    if pool_sizes:
        num = den = 0.0
        for p in per_prof:
            w = pool_sizes.get(p["profile"], 0)
            num += w * p["effect"]
            den += w
        extras["pool_weighted_mean"] = (num / den) if den else None
        extras["pool_weight_source"] = "clean-pool profile sizes (natural distribution)"
    return overall, per_prof, extras


def _majority(flags, per_seed):
    """(count, count_excluding_underpowered_seeds) for a per-seed boolean predicate."""
    n_all = sum(1 for s in SEEDS if flags[s])
    n_powered = sum(1 for s in SEEDS if flags[s] and not per_seed[s]["underpowered"])
    return n_all, n_powered


def full_criterion(scores, prefix, header, profiles, min_nonzero, pool_sizes=None):
    """Prereg items 1-4 for (header vs baseline) at a prefix, plus the non-zero guardrail."""
    base = scores[f"{PREFIX_LABEL}_baseline_{prefix}"]
    seed_res = {m: seed_stats(scores[f"{PREFIX_LABEL}_{m}_{header}_{prefix}"], base, min_nonzero)
                for m in MODES}
    text, token = seed_res["text"], seed_res["token"]

    dir_text = {s: text[s]["mean"] is not None and text[s]["mean"] < 0 for s in SEEDS}
    sig_text = {s: text[s]["p_less"] is not None and text[s]["p_less"] < ALPHA for s in SEEDS}
    dir_token = {s: token[s]["mean"] is not None and token[s]["mean"] < 0 for s in SEEDS}

    n_dir_text, n_dir_text_powered = _majority(dir_text, text)
    n_sig_text, n_sig_text_powered = _majority(sig_text, text)
    n_dir_token, n_dir_token_powered = _majority(dir_token, token)

    veto = [(m, s) for m in MODES for s in SEEDS
            if seed_res[m][s].get("p_greater") is not None
            and seed_res[m][s]["p_greater"] < ALPHA]
    prof_overall, prof_detail, prof_extras = profile_effect(
        scores[f"{PREFIX_LABEL}_text_{header}_{prefix}"], base, profiles, pool_sizes)

    # An underpowered seed may not be the vote that creates a >=2/3 majority (F4 guardrail (c),
    # rescaled). Recorded explicitly rather than folded silently into the counts.
    c1_pass = n_dir_text >= 2 and n_sig_text >= 2
    c1_needs_underpowered = c1_pass and (n_dir_text_powered < 2 or n_sig_text_powered < 2)
    c2_pass = n_dir_token >= 2
    c2_needs_underpowered = c2_pass and n_dir_token_powered < 2

    res = {
        "prefix": prefix, "header": header,
        "per_seed": seed_res,
        "c1_text_direction": n_dir_text, "c1_text_significant": n_sig_text,
        "c1_text_direction_powered": n_dir_text_powered,
        "c1_text_significant_powered": n_sig_text_powered,
        "c1_pass": c1_pass, "c1_majority_needs_underpowered_seed": c1_needs_underpowered,
        "c2_token_direction": n_dir_token, "c2_token_direction_powered": n_dir_token_powered,
        "c2_pass": c2_pass, "c2_majority_needs_underpowered_seed": c2_needs_underpowered,
        "c3_veto_hits": veto, "c3_pass": not veto,
        "c4_profile_effect_text": prof_overall,
        "c4_pass": prof_overall is not None and prof_overall <= 0,
        "c4_profile_detail": prof_detail,
        "c4_extras": prof_extras,
        "underpowered_seeds": {m: [s for s in SEEDS if seed_res[m][s]["underpowered"]]
                               for m in MODES},
        "min_nonzero_required": min_nonzero,
    }
    res["PASS"] = bool(c1_pass and c2_pass and res["c3_pass"] and res["c4_pass"]
                       and not c1_needs_underpowered and not c2_needs_underpowered)
    return res


def executability(geom_rows, analysis_idx):
    per_cond = collections.defaultdict(list)
    for r in geom_rows:
        if r["idx"] in analysis_idx:
            per_cond[(r["condition"], r["seed"])].append(bool(r["executable"]))
    out = collections.defaultdict(dict)
    for (cond, seed), v in sorted(per_cond.items()):
        out[cond][str(seed)] = float(np.mean(v))
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="repl_results")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--out", default="repl_verdict.json")
    # The prefix-0 column is dropped by preregistered decision (v2 §1.1 / A-5): its effective
    # sample size is capped by the number of distinct headers (<=19) no matter how many
    # programs are drawn, so re-running it would buy no independent information.
    ap.add_argument("--prefixes", default="p04",
                    help="comma-separated subset of p04,p00 (default p04 only)")
    a = ap.parse_args()
    prefixes = tuple(x.strip() for x in a.prefixes.split(",") if x.strip())
    assert set(prefixes) <= set(ALL_PREFIXES), f"unknown prefix in {prefixes}"

    frozen = json.loads(Path(a.frozen).read_text())
    samples = frozen["samples"]
    analysis_idx = {r["idx"] for r in samples}
    n = frozen["analysis"]["n"]
    min_nonzero = frozen["analysis"]["nonzero_guardrail_count"]
    profiles = frozen["profiles_4tag"]
    pool_sizes = frozen["pool"]["profile_pool_sizes"]   # prereg A-2 weighted sensitivity

    # The whole point of the new sample: one idx == one unique program, so no clustering.
    assert len(analysis_idx) == n == len(samples), "frozen list size disagrees with n"
    assert len({r["program_sha256"] for r in samples}) == n, \
        "frozen sample contains duplicate programs — clustering would be required"
    assert all(r["profile"] for r in samples), "frozen sample contains empty 4-tag intent"
    assert sum(len(p["idx"]) for p in profiles) == n, "profile lists do not partition the sample"

    geom_rows = load_jsonl(Path(a.dir) / "geom_scores.jsonl")
    scores = build_scores(geom_rows, analysis_idx)

    missing = [c for prefix in prefixes
               for c in ([f"{PREFIX_LABEL}_baseline_{prefix}"]
                         + [f"{PREFIX_LABEL}_{m}_{h}_{prefix}" for m in MODES for h in
                            ("correct",) + HEADERS])
               if c not in scores]
    assert not missing, (
        f"{len(missing)} conditions absent from {a.dir}: {missing}. A missing arm would make "
        "the paired test silently drop programs; fix the run rather than analysing a hole.")

    verdict = {
        "protocol": frozen["protocol"],
        "n": n, "n_profiles": len(profiles),
        "unit": frozen["analysis"]["unit"],
        "min_nonzero_required": min_nonzero,
        "frozen_sampling": frozen["sampling"],
        "prefixes_analysed": list(prefixes),
        "cells": {},
    }
    for prefix in prefixes:
        for header in HEADERS:
            verdict["cells"][f"{header}_{prefix}"] = full_criterion(
                scores, prefix, header, profiles, min_nonzero, pool_sizes)

    wrong04 = verdict["cells"]["shuffled_p04"]["PASS"]
    verdict["C3_REPLICATES"] = wrong04
    verdict["h2_anchor_text_p04"] = verdict["cells"]["masked_p04"]["PASS"]
    if "p00" in prefixes:
        wrong00 = verdict["cells"]["shuffled_p00"]["PASS"]
        verdict["outcome_row"] = {(True, True): 1, (True, False): 2,
                                  (False, True): 3, (False, False): 4}[(wrong04, wrong00)]
    else:
        # Single-column design: the outcome table degenerates to the decision cell itself.
        # v1's rows 1/2 and 3/4 differed only in the prefix-0 column, which is not measured here.
        verdict["outcome_row"] = None
        verdict["outcome"] = "REPLICATED" if wrong04 else "NOT_REPLICATED"

    # Descriptive only, no decision role: correct-header cells and executability.
    verdict["descriptive_correct"] = {}
    for prefix in prefixes:
        base = scores[f"{PREFIX_LABEL}_baseline_{prefix}"]
        verdict["descriptive_correct"][prefix] = {
            m: seed_stats(scores[f"{PREFIX_LABEL}_{m}_correct_{prefix}"], base, min_nonzero)
            for m in MODES}
    verdict["executability"] = executability(geom_rows, analysis_idx)

    Path(a.out).write_text(json.dumps(verdict, indent=1, default=float))
    print(f"n={n} prefixes={','.join(prefixes)} "
          f"outcome={verdict.get('outcome') or 'row ' + str(verdict['outcome_row'])} "
          f"C3_REPLICATES={wrong04} h2_anchor={verdict['h2_anchor_text_p04']} -> {a.out}")


if __name__ == "__main__":
    main()
