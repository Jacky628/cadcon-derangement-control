#!/usr/bin/env python
"""replfix_analysis.py — frozen decision analysis for the repaired-extractor re-run.

FREEZE ARTIFACT. Pre-registration:
`_paper_artifacts_rc-fdedd0/提取器修复_独立预注册_20260812.md` §5, §6, §7.1, §9.

The test specification is imported from `spike_analysis` (`wtest`), not reimplemented, so it
cannot drift from the published protocol: same-seed paired one-sided Wilcoxon, zero_method
"pratt", method "asymptotic", direction from the sign of mean(d), all-zero differences count
as non-significant.

What differs from the mother analysis, and only this:
  * PRIMARY UNIT is the prefix cluster (298 distinct model inputs), not the program. Paired
    differences are computed per program and then averaged within a cluster. The program unit
    is reported alongside but decides nothing (§5.1).
  * Every quantity is computed on BOTH scoring surfaces from the SAME generations: Rule A
    (`recall`) and the legacy rule (`legacy_recall`). The legacy columns are what the §3
    bit-identity self-check compares against the published numbers.
  * Outcome is three-valued: CONFIRMED / QUALIFIED / OVERTURNED (§9).

Inputs : replfix_scores.jsonl (replfix_score.py), replication_frozen_lists.json
Output : replfix_verdict.json + a human-readable summary on stdout

Usage: python replfix_analysis.py [--scores replfix_scores.jsonl]
                                  [--frozen replication_frozen_lists.json]
                                  [--out replfix_verdict.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spike_analysis import wtest          # frozen test spec, reused verbatim

SEEDS = (0, 1, 2)
ALPHA = 0.05
MODES = ("text", "token")
LABEL = "replfix"
GUARDRAIL_MIN_PROFILE_N = 15              # mother prereg A-4
GUARDRAIL_MIN_NONZERO = 60                # mother prereg §4 (0.15n), unchanged
GUARDRAIL_MAX_SAME_AS_LEGACY = 0.50       # this prereg §6 guardrail d

BASE = f"{LABEL}_baseline_p04"
def COND(mode, header): return f"{LABEL}_{mode}_{header}_p04"
R_CORRECT, R_WRONG = f"{LABEL}_R_correct_p04", f"{LABEL}_R_shuffled_p04"


# ── loading ──────────────────────────────────────────────────────────────────

def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_scores(rows, field):
    """{condition: {seed: {idx: adherence}}} for the given recall field."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        v = r.get(field)
        if v is None:
            continue                       # empty 4-tag intent, excluded by construction
        out[r["condition"]][r["seed"]][r["idx"]] = float(v)
    return out


def idx_to_cluster(rows):
    m = {}
    for r in rows:
        m.setdefault(r["idx"], r["prefix_key"])
    return m


# ── paired statistics on either unit ────────────────────────────────────────

def paired_diffs(cond, base, seed):
    ca, ba = cond.get(seed, {}), base.get(seed, {})
    idx = sorted(set(ca) & set(ba))
    return [ca[i] - ba[i] for i in idx], idx


def collapse_to_clusters(diffs, idx, i2c):
    """Average each program's paired difference within its prefix cluster (§5.1)."""
    acc = collections.defaultdict(list)
    for d, i in zip(diffs, idx):
        acc[i2c[i]].append(d)
    keys = sorted(acc)
    return [float(np.mean(acc[k])) for k in keys], keys


def seed_stats(cond, base, i2c=None, min_nonzero=GUARDRAIL_MIN_NONZERO):
    per_seed = {}
    for s in SEEDS:
        d, idx = paired_diffs(cond, base, s)
        if not d:
            per_seed[s] = {"n": 0, "mean": None, "p_less": None, "p_greater": None,
                           "nonzero": 0, "underpowered": True, "note": "missing arm"}
            continue
        if i2c is not None:
            d, _ = collapse_to_clusters(d, idx, i2c)
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


def criterion(per_seed, expect="less"):
    """Pre-registered rule: direction on >=2/3 seeds AND p<alpha on >=2/3, one reversal vetoes.

    `expect="less"` is the harm direction (wrong below baseline).
    """
    sign = -1.0 if expect == "less" else 1.0
    dirs, sigs, rev = {}, {}, {}
    for s in SEEDS:
        st = per_seed[s]
        m, p = st["mean"], st.get(f"p_{expect}")
        dirs[s] = (m is not None) and (m * sign > 0)
        sigs[s] = (p is not None) and (p < ALPHA)
        opp = st.get("p_greater" if expect == "less" else "p_less")
        rev[s] = (m is not None) and (m * sign < 0) and (opp is not None) and (opp < ALPHA)
    n_dir, n_sig = sum(dirs.values()), sum(sigs.values())
    reversed_any = any(rev.values())
    return {
        "direction_seeds": n_dir, "significant_seeds": n_sig,
        "reversal": reversed_any,
        "per_seed_direction": dirs, "per_seed_significant": sigs,
        "passed": (n_dir >= 2) and (n_sig >= 2) and not reversed_any,
    }


# ── outcome classification (§9) ─────────────────────────────────────────────

def classify(crit) -> str:
    if crit["reversal"]:
        return "OVERTURNED"
    if crit["direction_seeds"] < 2:
        return "OVERTURNED"          # sign did not hold on a majority of seeds
    if crit["significant_seeds"] >= 2:
        return "CONFIRMED"
    return "QUALIFIED"               # direction holds, significance does not


SEVERITY = {"CONFIRMED": 0, "QUALIFIED": 1, "OVERTURNED": 2}


# ── guardrails ──────────────────────────────────────────────────────────────

def guardrail_profiles(cond, base, profiles, i2c=None):
    per_prof, counted = [], []
    for prof in profiles:
        pid = set(prof["idx"])
        effs = []
        for s in SEEDS:
            d, idx = paired_diffs(
                {s: {i: v for i, v in cond.get(s, {}).items() if i in pid}},
                {s: {i: v for i, v in base.get(s, {}).items() if i in pid}}, s)
            if d:
                if i2c is not None:
                    d, _ = collapse_to_clusters(d, idx, i2c)
                effs.append(float(np.mean(d)))
        if effs:
            rec = {"profile": prof["profile"], "n_idx": len(pid),
                   "effect": float(np.mean(effs)),
                   "counted": len(pid) >= GUARDRAIL_MIN_PROFILE_N}
            per_prof.append(rec)
            if rec["counted"]:
                counted.append(rec["effect"])
    mean = float(np.mean(counted)) if counted else None
    return {"guardrail_mean": mean, "n_counted": len(counted), "per_profile": per_prof,
            "passed": (mean is not None) and (mean <= 0)}


def guardrail_extraction(rows):
    """Guardrail d: did the repair actually bite on multi-stage generations? (§6)"""
    by_cond = {}
    multi_all = [r for r in rows if r["n_cuts"] >= 2]
    same_all = sum(r["same_as_legacy"] for r in multi_all)
    for c in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == c and r["n_cuts"] >= 2]
        by_cond[c] = {
            "n_multi": len(sub),
            "same_as_legacy": sum(r["same_as_legacy"] for r in sub),
            "rate": (sum(r["same_as_legacy"] for r in sub) / len(sub)) if sub else None,
        }
    rate = same_all / len(multi_all) if multi_all else None
    return {"n_multi_cut_rows": len(multi_all), "same_as_legacy": same_all,
            "rate": rate, "threshold": GUARDRAIL_MAX_SAME_AS_LEGACY,
            "passed": (rate is not None) and (rate <= GUARDRAIL_MAX_SAME_AS_LEGACY),
            "per_condition": by_cond}


def executability(rows):
    """Both `executable` definitions per condition/seed, for §7.1's attribution table."""
    out = collections.defaultdict(dict)
    for c in sorted({r["condition"] for r in rows}):
        for s in SEEDS:
            sub = [r for r in rows if r["condition"] == c and r["seed"] == s]
            if not sub:
                continue
            out[c][s] = {
                "n": len(sub),
                "exec_first_block": sum(r["exec_first_block"] for r in sub) / len(sub),
                "exec_any_cut": sum(r["exec_any_cut"] for r in sub) / len(sub),
                "cap_hit": sum(bool(r.get("cap_hit")) for r in sub) / len(sub),
                "eos": sum(bool(r.get("eos_emitted")) for r in sub) / len(sub),
            }
    # Pooled, which is what §4.2's argument is stated at.
    pooled = {}
    for c in out:
        rs = [r for r in rows if r["condition"] == c]
        pooled[c] = {
            "exec_first_block": sum(r["exec_first_block"] for r in rs) / len(rs),
            "exec_any_cut": sum(r["exec_any_cut"] for r in rs) / len(rs),
        }
    return {"per_seed": dict(out), "pooled": pooled}


# ── main ────────────────────────────────────────────────────────────────────

def analyse_surface(rows, field, i2c, profiles):
    scores = build_scores(rows, field)
    base = scores[BASE]
    res = {}
    for mode in MODES:
        for header, expect in (("shuffled", "less"), ("masked", "less"), ("correct", "greater")):
            cond = scores.get(COND(mode, header))
            if not cond:
                continue
            cell = {}
            for unit, i2 in (("cluster", i2c), ("program", None)):
                ps = seed_stats(cond, base, i2)
                cr = criterion(ps, expect)
                cell[unit] = {"per_seed": ps, "criterion": cr}
            cell["profile_guardrail"] = guardrail_profiles(cond, base, profiles, i2c)
            cell["expect"] = expect
            res[f"{mode}_{header}"] = cell
    return res


def test_d(rows, field, i2c):
    """drop_M - drop_R interaction, judged exactly as the F4 mother document defines it."""
    scores = build_scores(rows, field)
    m_c, m_w = scores.get(COND("text", "correct")), scores.get(COND("text", "shuffled"))
    r_c, r_w = scores.get(R_CORRECT), scores.get(R_WRONG)
    if not all([m_c, m_w, r_c, r_w]):
        return {"note": "R or M arm missing", "available": False}

    per_seed = {}
    for s in SEEDS:
        idx = sorted(set(m_c.get(s, {})) & set(m_w.get(s, {}))
                     & set(r_c.get(s, {})) & set(r_w.get(s, {})))
        if not idx:
            per_seed[s] = {"n": 0, "mean": None, "note": "missing arm"}
            continue
        inter = [(m_c[s][i] - m_w[s][i]) - (r_c[s][i] - r_w[s][i]) for i in idx]
        if i2c is not None:
            inter, _ = collapse_to_clusters(inter, idx, i2c)
        pg, _ = wtest(inter, "greater")
        nz = int(sum(1 for x in inter if x != 0))
        per_seed[s] = {"n": len(inter), "mean": float(np.mean(inter)),
                       "p_greater": None if math.isnan(pg) else pg,
                       "nonzero": nz, "underpowered": nz < GUARDRAIL_MIN_NONZERO}

    # R competence gate (F4 mother doc, thresholds unchanged by this pre-registration).
    def arm_mean(sc):
        vals = [v for s in SEEDS for v in sc.get(s, {}).values()]
        return float(np.mean(vals)) if vals else None
    gate = {
        "R_correct_adherence": arm_mean(r_c),
        "R_wrong_adherence": arm_mean(r_w),
        "threshold": 0.30,
    }
    gate["passed"] = (gate["R_correct_adherence"] is not None
                      and gate["R_wrong_adherence"] is not None
                      and gate["R_correct_adherence"] >= 0.30
                      and gate["R_wrong_adherence"] >= 0.30)

    n_dir = sum(1 for s in SEEDS if (per_seed[s]["mean"] or 0) > 0)
    n_sig = sum(1 for s in SEEDS if (per_seed[s].get("p_greater") is not None
                                     and per_seed[s]["p_greater"] < ALPHA))
    return {"available": True, "per_seed": per_seed, "competence_gate": gate,
            "direction_seeds": n_dir, "significant_seeds": n_sig,
            "passed": gate["passed"] and n_dir >= 2 and n_sig >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="replfix_scores.jsonl")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--out", default="replfix_verdict.json")
    a = ap.parse_args()

    rows = load_rows(a.scores)
    frozen = json.loads(Path(a.frozen).read_text())
    n_expected = len(frozen["samples"])

    conds = sorted({r["condition"] for r in rows})
    assert len(conds) == 9, f"expected 9 conditions, found {len(conds)}: {conds}"
    for c in conds:
        for s in SEEDS:
            k = sum(1 for r in rows if r["condition"] == c and r["seed"] == s)
            assert k == n_expected, (
                f"{c} seed={s}: {k} rows but the frozen sample has {n_expected}; "
                "refusing to decide on an incomplete matrix (§12)")

    # Test D pairing: M and R must have received the same injected header on the same idx.
    m_w = {(r["seed"], r["idx"]): tuple(r["header_injected"])
           for r in rows if r["condition"] == COND("text", "shuffled")}
    r_w = {(r["seed"], r["idx"]): tuple(r["header_injected"])
           for r in rows if r["condition"] == R_WRONG}
    if m_w and r_w:
        assert m_w == r_w, ("M and R wrong-header arms received different header injections; "
                            "Test D would compare different treatments")

    i2c = idx_to_cluster(rows)
    n_clusters = len(set(i2c.values()))
    profiles = frozen.get("profiles_4tag") or frozen.get("analysis", {}).get("profiles") or []

    verdict = {
        "protocol": "extractor-fix independent pre-registration (2026-08-12)",
        "n_programs": n_expected,
        "n_clusters": n_clusters,
        "primary_unit": "prefix_cluster",
        "rule_a": analyse_surface(rows, "recall", i2c, profiles),
        "legacy": analyse_surface(rows, "legacy_recall", i2c, profiles),
        "test_d": {"rule_a": test_d(rows, "recall", i2c),
                   "legacy": test_d(rows, "legacy_recall", i2c)},
        "guardrail_extraction": guardrail_extraction(rows),
        "executability": executability(rows),
    }

    # ── Outcome (§9): primary comparison is wrong - baseline, cluster unit, Rule A ──
    outcomes = {}
    for mode in MODES:
        crit = verdict["rule_a"][f"{mode}_shuffled"]["cluster"]["criterion"]
        outcomes[mode] = classify(crit)
    worst = max(outcomes.values(), key=lambda o: SEVERITY[o])
    verdict["outcome"] = {"per_mode": outcomes, "overall": worst,
                          "rule": "text and token judged separately; the more conservative "
                                  "outcome sets the arXiv v5 obligation (§9)"}

    Path(a.out).write_text(json.dumps(verdict, indent=2))

    # ── human-readable summary ──
    print(f"n={n_expected} programs -> {n_clusters} prefix clusters (primary unit)")
    for surface in ("rule_a", "legacy"):
        print(f"\n=== {surface} surface ===")
        for mode in MODES:
            cell = verdict[surface].get(f"{mode}_shuffled")
            if not cell:
                continue
            for unit in ("cluster", "program"):
                ps, cr = cell[unit]["per_seed"], cell[unit]["criterion"]
                means = " / ".join("n/a" if ps[s]["mean"] is None else f"{ps[s]['mean']:+.4f}"
                                   for s in SEEDS)
                ps_ = " / ".join("n/a" if ps[s]["p_less"] is None else f"{ps[s]['p_less']:.3g}"
                                 for s in SEEDS)
                print(f"  {mode:5s} wrong-baseline [{unit:7s}] mean {means}  p {ps_}  "
                      f"dir {cr['direction_seeds']}/3 sig {cr['significant_seeds']}/3"
                      f"{'  REVERSAL' if cr['reversal'] else ''}")
    g = verdict["guardrail_extraction"]
    print(f"\nguardrail d: same_as_legacy on multi-cut rows "
          f"{g['same_as_legacy']}/{g['n_multi_cut_rows']} = "
          f"{(g['rate'] or 0)*100:.1f}%  -> {'PASS' if g['passed'] else 'FAIL'}")
    print("\nexecutability (pooled, exec_first_block -> exec_any_cut):")
    for c, v in verdict["executability"]["pooled"].items():
        print(f"  {c:28s} {v['exec_first_block']:.3f} -> {v['exec_any_cut']:.3f}")
    print(f"\nOUTCOME: {verdict['outcome']['per_mode']}  overall={verdict['outcome']['overall']}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
