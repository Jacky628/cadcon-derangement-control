"""f4_analysis.py — F4 FROZEN analysis (implements F4_预注册判据_20260707.md §4 verbatim).

Decides whether the C-spike's confirmed C3 harm (40%-prefix wrong-header < baseline) is
SEMANTIC (model reads header content) or a DISTRIBUTION-SHIFT/mismatch artifact, via a
dissociation between the real model M (trained on informative GT headers) and the control
model R (trained on shuffled-GT headers = same header marginal, correlation destroyed).

No new analytic degrees of freedom. drop_M is computed from the FROZEN C-spike
spike_results/geom_scores.jsonl using the IMPORTED spike_analysis recall function (finding #8);
a reproduction assert checks M means == the C-spike verdict (0.557 / 0.305).

Frozen dependencies (unchanged): geom_requirements.py (sha256 df380e…), spike_frozen_lists.json
(n76 / 6 profiles), spike_analysis.recall_4tag/build_scores/wtest/paired_diffs.

Order (prereg §4):  R-competence GATE  →  (if pass) Test D + guardrails  →  M1/M2 + F4b.
Usage: .venv/bin/python f4_analysis.py [--f4dir f4_results] [--spikedir spike_results] [--out f4_verdict.json]
"""
import argparse, collections, json, math
from pathlib import Path

import numpy as np
from scipy import stats

import spike_analysis as SA   # reuse frozen recall/test machinery (finding #8)

SEEDS = (0, 1, 2)
ALPHA = 0.05
TAGS_PRIMARY = SA.TAGS_PRIMARY
GATE_FLOOR = 0.30            # R arms must be non-trivially above the 40%-prefix floor
GATE_EXEC_MARGIN = 0.10     # R executability >= baseline_p04 exec - 0.10
MIN_NONZERO_PAIRS = 15      # per-seed underpowered flag (guardrail c)


def load_rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def build_scores_and_exec(rows, n76):
    """{cond:{seed:{idx: exec-inclusive 4-tag recall}}} and {cond:{seed:{idx: executable bool}}}."""
    sc = collections.defaultdict(lambda: collections.defaultdict(dict))
    ex = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["idx"] not in n76:
            continue
        rec = SA.recall_4tag(r["intended"], set(r["geom_produced"]) if r["executable"] else set())
        if rec is None:
            continue
        sc[r["condition"]][r["seed"]][r["idx"]] = 0.0 if not r["executable"] else rec
        ex[r["condition"]][r["seed"]][r["idx"]] = bool(r["executable"])
    return sc, ex


def arm(scores, name_fn):
    """Collect a logical arm as {seed: {idx: recall}} across seed-embedded or seed-agnostic naming."""
    return {s: scores.get(name_fn(s), {}).get(s, {}) for s in SEEDS}


def arm_exec_rate(execmap, name_fn):
    rates = []
    for s in SEEDS:
        d = execmap.get(name_fn(s), {}).get(s, {})
        if d:
            rates.append(sum(d.values()) / len(d))
    return float(np.mean(rates)) if rates else 0.0


def arm_mean(a):
    vals = [v for s in SEEDS for v in a.get(s, {}).values()]
    return float(np.mean(vals)) if vals else None


def paired(a_pos, a_neg, seed):
    """per-idx (a_pos - a_neg) at a seed, on the common idx (exec-inclusive => all n76 present)."""
    p, n = a_pos.get(seed, {}), a_neg.get(seed, {})
    idx = sorted(set(p) & set(n))
    return [p[i] - n[i] for i in idx], idx


def wil(d, alt):
    d = np.asarray(d, float)
    if d.size == 0 or np.all(d == 0):
        return float("nan")
    return float(stats.wilcoxon(d, zero_method="pratt", alternative=alt, method="asymptotic").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f4dir", default="f4_results")
    ap.add_argument("--spikedir", default="spike_results")
    ap.add_argument("--frozen", default="spike_frozen_lists.json")
    ap.add_argument("--out", default="f4_verdict.json")
    a = ap.parse_args()

    frozen = json.load(open(a.frozen))
    n76 = set(frozen["analysis_idx_n76"])
    profiles = frozen["profiles_4tag"]

    m_sc, m_ex = build_scores_and_exec(load_rows(Path(a.spikedir) / "geom_scores.jsonl"), n76)
    r_sc, r_ex = build_scores_and_exec(load_rows(Path(a.f4dir) / "geom_scores.jsonl"), n76)

    # ---- logical arms ----
    M_correct  = arm(m_sc, lambda s: "spike_text_correct_p04")
    M_shuffled = arm(m_sc, lambda s: "spike_text_shuffled_p04")
    base_p04   = arm(m_sc, lambda s: "spike_baseline_p04")
    base_p00   = arm(m_sc, lambda s: "spike_baseline_p00")
    R_correct  = arm(r_sc, lambda s: f"f4_R_correct_p04_s{s}")
    R_shuffled = arm(r_sc, lambda s: f"f4_R_shuffled_p04_s{s}")
    R_masked   = arm(r_sc, lambda s: f"f4_R_masked_p04_s{s}")
    R_corr_p00 = arm(r_sc, lambda s: f"f4_R_correct_p00_s{s}")
    M_corrupt  = arm(r_sc, lambda s: f"f4_M_corrupted_p04_s{s}")
    M_random   = arm(r_sc, lambda s: f"f4_M_random_p04_s{s}")

    V = {"n76": len(n76), "n_profiles": len(profiles)}

    # ---- reproduction assert (finding #8): M means must match the C-spike verdict ----
    mc, ms = arm_mean(M_correct), arm_mean(M_shuffled)
    V["repro"] = {"M_correct_mean": mc, "M_shuffled_mean": ms,
                  "ok": (mc is not None and ms is not None
                         and abs(mc - 0.557) <= 0.005 and abs(ms - 0.305) <= 0.005)}

    # ---- R-competence GATE (finding #1) ----
    base_exec = arm_exec_rate(m_ex, lambda s: "spike_baseline_p04")
    r_exec = np.mean([arm_exec_rate(r_ex, lambda s: f"f4_R_correct_p04_s{s}"),
                      arm_exec_rate(r_ex, lambda s: f"f4_R_shuffled_p04_s{s}")])
    rc_seed = [float(np.mean(list(R_correct[s].values()))) if R_correct[s] else None for s in SEEDS]
    rs_seed = [float(np.mean(list(R_shuffled[s].values()))) if R_shuffled[s] else None for s in SEEDS]
    n_rc_ok = sum(1 for v in rc_seed if v is not None and v >= GATE_FLOOR)
    n_rs_ok = sum(1 for v in rs_seed if v is not None and v >= GATE_FLOOR)
    gate_pass = (n_rc_ok >= 2 and n_rs_ok >= 2 and r_exec >= base_exec - GATE_EXEC_MARGIN)
    V["competence_gate"] = {
        "R_correct_per_seed": rc_seed, "R_shuffled_per_seed": rs_seed,
        "R_correct_mean": arm_mean(R_correct), "R_shuffled_mean": arm_mean(R_shuffled),
        "R_exec_rate": float(r_exec), "baseline_p04_exec_rate": float(base_exec),
        "floor": GATE_FLOOR, "n_R_correct>=floor": n_rc_ok, "n_R_shuffled>=floor": n_rs_ok,
        "PASS": bool(gate_pass),
    }

    # ---- Test D: interaction drop_M > drop_R (only meaningful if gate passes) ----
    perseed = {}
    for s in SEEDS:
        dM, _ = paired(M_correct, M_shuffled, s)
        dR, _ = paired(R_correct, R_shuffled, s)
        # pair on common idx
        pM, pMn = M_correct.get(s, {}), M_shuffled.get(s, {})
        pR, pRn = R_correct.get(s, {}), R_shuffled.get(s, {})
        idx = sorted(set(pM) & set(pMn) & set(pR) & set(pRn))
        diff = [(pM[i] - pMn[i]) - (pR[i] - pRn[i]) for i in idx]
        nz = sum(1 for x in diff if x != 0)
        perseed[s] = {"n": len(diff), "nonzero_pairs": nz,
                      "mean_diff": float(np.mean(diff)) if diff else None,
                      "drop_M_mean": float(np.mean(dM)) if dM else None,
                      "drop_R_mean": float(np.mean(dR)) if dR else None,
                      "p_greater": wil(diff, "greater"), "p_less": wil(diff, "less")}
    n_dir = sum(1 for s in SEEDS if (perseed[s]["mean_diff"] or 0) > 0)
    n_sig = sum(1 for s in SEEDS if perseed[s]["p_greater"] is not None
                and not math.isnan(perseed[s]["p_greater"]) and perseed[s]["p_greater"] < ALPHA)
    veto = [s for s in SEEDS if perseed[s]["p_less"] is not None
            and not math.isnan(perseed[s]["p_less"]) and perseed[s]["p_less"] < ALPHA]
    underpowered = [s for s in SEEDS if perseed[s]["nonzero_pairs"] < MIN_NONZERO_PAIRS]

    # guardrail (a): executable-intersection sensitivity (drop the exec-inclusive 0-fill)
    inter_diffs = []
    for s in SEEDS:
        eMc, eMs = m_ex.get("spike_text_correct_p04", {}).get(s, {}), m_ex.get("spike_text_shuffled_p04", {}).get(s, {})
        eRc = r_ex.get(f"f4_R_correct_p04_s{s}", {}).get(s, {})
        eRs = r_ex.get(f"f4_R_shuffled_p04_s{s}", {}).get(s, {})
        for i in n76:
            if eMc.get(i) and eMs.get(i) and eRc.get(i) and eRs.get(i):
                dm = M_correct[s][i] - M_shuffled[s][i]
                dr = R_correct[s][i] - R_shuffled[s][i]
                inter_diffs.append(dm - dr)
    inter_mean = float(np.mean(inter_diffs)) if inter_diffs else None

    # guardrail (b): profile-level (drop_M - drop_R) seed-avg, direction >= 0
    prof_eff = []
    for prof in profiles:
        pid = set(prof["idx"]) & n76
        effs = []
        for s in SEEDS:
            vals = []
            for i in pid:
                if i in M_correct[s] and i in R_correct[s]:
                    vals.append((M_correct[s][i] - M_shuffled[s][i]) - (R_correct[s][i] - R_shuffled[s][i]))
            if vals:
                effs.append(float(np.mean(vals)))
        if effs:
            prof_eff.append({"profile": prof["profile"], "effect": float(np.mean(effs))})
    prof_overall = float(np.mean([p["effect"] for p in prof_eff])) if prof_eff else None

    testD_pass = bool(n_dir >= 2 and n_sig >= 2 and not veto
                      and (inter_mean is not None and inter_mean > 0)
                      and (prof_overall is not None and prof_overall >= 0))
    V["test_D"] = {"per_seed": perseed, "n_direction>0": n_dir, "n_significant": n_sig,
                   "reverse_veto_seeds": veto, "underpowered_seeds": underpowered,
                   "guard_a_exec_intersection_mean": inter_mean,
                   "guard_b_profile_effect": prof_overall, "guard_b_detail": prof_eff,
                   "PASS": testD_pass}

    # ---- supporting M1 / M2 (report only) ----
    def eff_pair(ap_, an):  # per-seed mean(ap-an) + p
        out = {}
        for s in SEEDS:
            d, _ = paired(ap_, an, s)
            out[s] = {"mean": float(np.mean(d)) if d else None, "p_less": wil(d, "less")}
        return out
    V["M1_content"] = {"R_correct_vs_R_shuffled": eff_pair(R_correct, R_shuffled),
                       "note": "content-blindness evidence: expect ~0 (R ignores header content)"}
    V["M1_presence"] = {"R_correct_vs_R_masked": eff_pair(R_correct, R_masked),
                        "note": "PRESENCE contrast, NOT content evidence (per finding #7)"}
    V["M2_headeronly"] = {"R_correct_p00_mean": arm_mean(R_corr_p00),
                          "baseline_p00_mean": arm_mean(base_p00),
                          "M_correct_p00_ref": 0.776,
                          "note": "descriptive: R can't generate from header alone if it ignores headers"}

    # ---- F4b dose-response (supporting; same-seed paired, no seed pooling) ----
    def dose_step(hi, lo):
        n = 0
        detail = {}
        for s in SEEDS:
            d, _ = paired(hi, lo, s)
            m = float(np.mean(d)) if d else None
            p = wil(d, "greater")
            detail[s] = {"mean": m, "p": p}
            if m is not None and m > 0 and p is not None and not math.isnan(p) and p < ALPHA:
                n += 1
        return {"n_seed_hi>lo": n, "per_seed": detail}
    corr_gt_corrupt = dose_step(M_correct, M_corrupt)
    corrupt_gt_shuf = dose_step(M_corrupt, M_shuffled)
    V["F4b_dose"] = {
        "means": {"correct": arm_mean(M_correct), "corrupted": arm_mean(M_corrupt),
                  "shuffled": arm_mean(M_shuffled), "random": arm_mean(M_random)},
        "correct>corrupted": corr_gt_corrupt, "corrupted>shuffled": corrupt_gt_shuf,
        "monotone_confirmed": bool(corr_gt_corrupt["n_seed_hi>lo"] >= 2 and corrupt_gt_shuf["n_seed_hi>lo"] >= 2),
        "M_random_vs_M_shuffled_mean_diff": (arm_mean(M_random) - arm_mean(M_shuffled))
            if (arm_mean(M_random) is not None and arm_mean(M_shuffled) is not None) else None,
        "note": "SUPPORTING/exploratory. M_random is NON-EVIDENTIAL. non-monotone = not-confirmed (no spin).",
    }

    # ---- verdict ----
    if not V["competence_gate"]["PASS"]:
        ruling, headline = "UNINTERPRETABLE_control_invalid", "downgrade (per §0 ①): R near floor, dissociation not testable"
    elif testD_pass:
        ruling, headline = "DISSOCIATION_semantic", "strong: wrong DESIGN-INTENT hurts (distribution-shift excluded)"
    else:
        ruling, headline = "no_dissociation", "downgrade (per §0 ①): any mismatched header perturbs (distribution shift / not-clean control)"
    V["RULING"] = ruling
    V["headline"] = headline
    V["repro_ok"] = V["repro"]["ok"]

    Path(a.out).write_text(json.dumps(V, indent=1))
    print(f"GATE={V['competence_gate']['PASS']} TestD={testD_pass} -> RULING={ruling} | repro_ok={V['repro']['ok']} -> {a.out}")


if __name__ == "__main__":
    main()
