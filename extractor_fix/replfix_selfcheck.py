#!/usr/bin/env python
"""replfix_selfcheck.py — the §3 bit-identity self-check. Run BEFORE any verdict is read.

FREEZE ARTIFACT. Pre-registration:
`_paper_artifacts_rc-fdedd0/提取器修复_独立预注册_20260812.md` §3.

The claim being tested
----------------------
Applying the LEGACY extraction rule to THIS run's generations must reproduce the published
geometry verdicts exactly. The whole chain is deterministic — same checkpoints, greedy
decoding, same eval_batch_size, same frozen 400-program sample, same max_new_tokens — so any
disagreement is environment drift, not a finding.

Per the pre-registration this is a gate, not a diagnostic: if it fails, the drift is located
and resolved first, and NOTHING from replfix_analysis.py may be read as a result until it
passes. The outcome is written to the run log either way.

What is compared, per (condition, seed, idx):
    executable      : legacy surface  vs  published `executable`
    geom_produced   : legacy surface  vs  published `geom_produced`

`geom_produced` is the tag set the primary metric is computed from, so agreement on it is
agreement on every published adherence number, cell by cell.

Usage:
  python replfix_selfcheck.py --scores replfix_scores.jsonl \
      --published repl_results/geom_scores.jsonl \
      --published testd_results/geom_scores.jsonl \
      [--out replfix_selfcheck.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# replfix_* condition -> the published condition it must reproduce.
CONDITION_MAP = {
    "replfix_baseline_p04":       "repl_baseline_p04",
    "replfix_text_correct_p04":   "repl_text_correct_p04",
    "replfix_text_shuffled_p04":  "repl_text_shuffled_p04",
    "replfix_text_masked_p04":    "repl_text_masked_p04",
    "replfix_token_correct_p04":  "repl_token_correct_p04",
    "replfix_token_shuffled_p04": "repl_token_shuffled_p04",
    "replfix_token_masked_p04":   "repl_token_masked_p04",
    "replfix_R_correct_p04":      "replD_R_correct_p04",
    "replfix_R_shuffled_p04":     "replD_R_shuffled_p04",
}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="replfix_scores.jsonl")
    ap.add_argument("--published", action="append", required=True,
                    help="published geom_scores.jsonl; repeat for repl_ and replD_ results")
    ap.add_argument("--out", default="replfix_selfcheck.json")
    a = ap.parse_args()

    new = load(a.scores)
    pub = {}
    for p in a.published:
        for r in load(p):
            pub[(r["condition"], r["seed"], r["idx"])] = r

    per_cond = collections.defaultdict(
        lambda: {"n": 0, "exec_match": 0, "tags_match": 0, "both_match": 0, "examples": []})
    missing = []

    for r in new:
        cond = r["condition"]
        target = CONDITION_MAP.get(cond)
        if target is None:
            raise SystemExit(f"FATAL: no published counterpart mapped for condition {cond}")
        key = (target, r["seed"], r["idx"])
        ref = pub.get(key)
        if ref is None:
            missing.append(key)
            continue
        st = per_cond[cond]
        st["n"] += 1
        e = bool(r["legacy_executable"]) == bool(ref["executable"])
        t = sorted(r["legacy_geom_produced"]) == sorted(ref["geom_produced"])
        st["exec_match"] += e
        st["tags_match"] += t
        st["both_match"] += (e and t)
        if not (e and t) and len(st["examples"]) < 5:
            st["examples"].append({
                "seed": r["seed"], "idx": r["idx"],
                "executable_new": bool(r["legacy_executable"]),
                "executable_published": bool(ref["executable"]),
                "tags_new": sorted(r["legacy_geom_produced"]),
                "tags_published": sorted(ref["geom_produced"]),
            })

    total = sum(s["n"] for s in per_cond.values())
    both = sum(s["both_match"] for s in per_cond.values())
    passed = (total > 0) and (both == total) and not missing

    report = {
        "gate": "bit-identity self-check (pre-registration §3)",
        "passed": bool(passed),
        "n_compared": total,
        "n_identical": both,
        "n_missing_counterpart": len(missing),
        "missing_examples": missing[:10],
        "per_condition": {c: {k: v for k, v in s.items()} for c, s in sorted(per_cond.items())},
    }
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"compared {total} rows against the published geometry verdicts")
    for c, s in sorted(per_cond.items()):
        flag = "OK " if s["both_match"] == s["n"] else "DIFF"
        print(f"  [{flag}] {c:28s} {s['both_match']}/{s['n']} identical "
              f"(executable {s['exec_match']}/{s['n']}, tags {s['tags_match']}/{s['n']})")
    if missing:
        print(f"  {len(missing)} rows had no published counterpart, e.g. {missing[:3]}")

    if passed:
        print("\nSELF-CHECK PASSED — the legacy rule reproduces the published surface exactly; "
              "differences on the Rule A surface are attributable to the extraction rule.")
        return 0
    print("\nSELF-CHECK FAILED — environment drift. Per §3 this must be located and resolved "
          "BEFORE any number from replfix_analysis.py is read as a result.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
