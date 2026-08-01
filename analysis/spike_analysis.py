"""spike_analysis.py — C-spike FROZEN analysis (implements prereg §2/§3 verbatim).

Implements ONLY the rules frozen in Cspike_预注册判据_20260706.md (sha256 99efe404…).
No new analytic degrees of freedom. Frozen by sha256 in cspike_frozen/sha256s.txt
BEFORE Step-3 scoring per Appendix A's standing commitment.

Inputs (produced by Step 2 + geom_requirements.py):
  spike_results/adherence_samples.jsonl   (regex-side rows incl. raw programs)
  spike_results/geom_scores.jsonl         (geometry-side rows, same keys)
  spike_frozen_lists.json                 (n76 idx, 6 profiles, repro reference)

Decision chain (prereg §3):
  d(seed s) = per-sample 4-tag geometric execution-inclusive recall difference
              (conditioned arm − baseline arm), same-seed pairing, idx ∈ n76.
  test     = scipy.stats.wilcoxon(d, zero_method="pratt", alternative="less",
             method="asymptotic"); degenerate all-zero d => NOT significant.
  direction= sign(mean(d)); mean==0 counts as inconsistent.
  full criterion at (mode-set, prefix):
    (1) text: >=2/3 seeds mean<0 AND >=2/3 seeds p<0.05
    (2) token: >=2/3 seeds mean<0
    (3) veto: ANY seed, ANY mode significant in reverse (alternative="greater" p<0.05)
    (4) profile guardrail: text profile-level mean effect (6 profiles, seed-avg) <= 0
  C3 survives <=> full criterion holds at prefix 0.4 (wrong vs baseline).
Usage: .venv/bin/python spike_analysis.py [--dir spike_results] [--out spike_verdict.json]
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np
from scipy import stats

TAGS_PRIMARY = ("<CIRCLE>", "<NGON>", "<THIN>", "<TALL>")
MODES = ("token", "text")
PREFIXES = ("p04", "p00")           # decision cell first
HEADERS = ("correct", "shuffled", "masked")
SEEDS = (0, 1, 2)
ALPHA = 0.05


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def recall_4tag(intended, produced_set):
    it = [t for t in intended if t in TAGS_PRIMARY]
    if not it:
        return None
    return len(set(it) & produced_set) / len(it)


def build_scores(geom_rows, n76):
    """{condition: {seed: {idx: exec-inclusive 4-tag geometric recall}}} over idx∈n76."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in geom_rows:
        if r["idx"] not in n76:
            continue
        produced = set(r["geom_produced"]) if r["executable"] else set()
        rec = recall_4tag(r["intended"], produced)
        if rec is None:                     # empty 4-tag intent — excluded by n76 anyway
            continue
        if not r["executable"]:
            rec = 0.0                       # execution-inclusive primary rule
        out[r["condition"]][r["seed"]][r["idx"]] = rec
    return out


def wtest(d, alternative):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return float("nan"), "degenerate_all_zero"
    res = stats.wilcoxon(d, zero_method="pratt", alternative=alternative,
                         method="asymptotic")
    return float(res.pvalue), "ok"


def paired_diffs(cond_scores, base_scores, seed):
    ca, ba = cond_scores.get(seed, {}), base_scores.get(seed, {})
    idx = sorted(set(ca) & set(ba))
    return [ca[i] - ba[i] for i in idx], idx


def seed_stats(cond_scores, base_scores):
    per_seed = {}
    for s in SEEDS:
        d, idx = paired_diffs(cond_scores, base_scores, s)
        if not d:
            per_seed[s] = {"n": 0, "mean": None, "p_less": None, "p_greater": None,
                           "note": "missing arm"}
            continue
        m = float(np.mean(d))
        pl, nl = wtest(d, "less")
        pg, ng = wtest(d, "greater")
        per_seed[s] = {"n": len(d), "mean": m,
                       "p_less": None if math.isnan(pl) else pl, "p_less_note": nl,
                       "p_greater": None if math.isnan(pg) else pg, "p_greater_note": ng}
    return per_seed


def profile_effect(cond_scores, base_scores, profiles):
    """Seed-averaged per-profile mean effect; returns (overall_mean, per_profile)."""
    per_prof = []
    for prof in profiles:
        pid = prof["idx"]
        effs = []
        for s in SEEDS:
            d, _ = paired_diffs(
                {s: {i: v for i, v in cond_scores.get(s, {}).items() if i in pid}},
                {s: {i: v for i, v in base_scores.get(s, {}).items() if i in pid}}, s)
            if d:
                effs.append(float(np.mean(d)))
        if effs:
            per_prof.append({"profile": prof["profile"], "n_idx": len(pid),
                             "effect": float(np.mean(effs))})
    overall = float(np.mean([p["effect"] for p in per_prof])) if per_prof else None
    return overall, per_prof


def full_criterion(scores, prefix, header, profiles):
    """Prereg §3 items 1-4 for (header vs baseline) at a prefix. Returns verdict dict."""
    base = scores[f"spike_baseline_{prefix}"]
    res = {"prefix": prefix, "header": header}
    seed_res = {}
    for m in MODES:
        seed_res[m] = seed_stats(scores[f"spike_{m}_{header}_{prefix}"], base)
    res["per_seed"] = seed_res

    text, token = seed_res["text"], seed_res["token"]
    n_dir_text = sum(1 for s in SEEDS if (text[s]["mean"] or 0) < 0 and text[s]["mean"] is not None)
    n_sig_text = sum(1 for s in SEEDS if text[s]["p_less"] is not None and text[s]["p_less"] < ALPHA)
    n_dir_token = sum(1 for s in SEEDS if token[s]["mean"] is not None and token[s]["mean"] < 0)
    veto = [(m, s) for m in MODES for s in SEEDS
            if seed_res[m][s].get("p_greater") is not None
            and seed_res[m][s]["p_greater"] < ALPHA]
    prof_overall, prof_detail = profile_effect(
        scores[f"spike_text_{header}_{prefix}"], base, profiles)

    res.update({
        "c1_text_direction": n_dir_text, "c1_text_significant": n_sig_text,
        "c1_pass": n_dir_text >= 2 and n_sig_text >= 2,
        "c2_token_direction": n_dir_token, "c2_pass": n_dir_token >= 2,
        "c3_veto_hits": veto, "c3_pass": not veto,
        "c4_profile_effect_text": prof_overall,
        "c4_pass": prof_overall is not None and prof_overall <= 0,
        "c4_profile_detail": prof_detail,
    })
    res["PASS"] = bool(res["c1_pass"] and res["c2_pass"] and res["c3_pass"] and res["c4_pass"])
    return res


def repro_check(merge_dir, frozen):
    """Prereg §4 Step 2: |Δrecall|<=0.01 and |Δvalidity|<=0.02 vs frozen old values."""
    new = json.load(open(Path(merge_dir) / "results.json"))
    out, ok = {}, True
    for cell, ref in frozen["repro_reference"].items():
        cell_new = new.get(cell)
        rows = {}
        for s, old in ref["per_seed"].items():
            nv = (cell_new or {}).get(s)
            if nv is None:
                rows[s] = {"status": "MISSING_NEW"}
                ok = False
                continue
            dr = abs((nv.get("adherence_recall") or 0) - (old.get("adherence_recall") or 0))
            dv = abs((nv.get("validity_rate") or 0) - (old.get("validity_rate") or 0))
            good = dr <= 0.01 and dv <= 0.02
            ok = ok and good
            rows[s] = {"d_recall": round(dr, 4), "d_validity": round(dv, 4),
                       "pass": good}
        out[cell] = {"old_condition": ref["old_condition"], "seeds": rows}
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="spike_results")
    ap.add_argument("--frozen", default="spike_frozen_lists.json")
    ap.add_argument("--out", default="spike_verdict.json")
    a = ap.parse_args()

    frozen = json.load(open(a.frozen))
    n76 = set(frozen["analysis_idx_n76"])
    profiles = frozen["profiles_4tag"]
    geom_rows = load_jsonl(Path(a.dir) / "geom_scores.jsonl")
    scores = build_scores(geom_rows, n76)

    verdict = {"n76": len(n76), "n_profiles": len(profiles)}

    # Reproduction gate (regex side) — reported first; failure = stop per prereg.
    verdict["repro_ok"], verdict["repro_detail"] = repro_check(a.dir, frozen)

    # Decision comparisons: wrong & masked vs baseline at both prefixes.
    verdict["cells"] = {}
    for prefix in PREFIXES:
        for header in ("shuffled", "masked"):
            verdict["cells"][f"{header}_{prefix}"] = full_criterion(
                scores, prefix, header, profiles)

    wrong04 = verdict["cells"]["shuffled_p04"]["PASS"]
    wrong00 = verdict["cells"]["shuffled_p00"]["PASS"]
    outcome = {(True, True): 1, (True, False): 2,
               (False, True): 3, (False, False): 4}[(wrong04, wrong00)]
    verdict["C3_SURVIVES"] = wrong04
    verdict["outcome_row"] = outcome
    verdict["ruling"] = "GO" if wrong04 else "NO-GO"
    verdict["h2_anchor_text_p04"] = verdict["cells"]["masked_p04"]["PASS"]

    # Descriptive: correct-header cells (steering reference, no decision role).
    verdict["descriptive_correct"] = {}
    for prefix in PREFIXES:
        base = scores[f"spike_baseline_{prefix}"]
        verdict["descriptive_correct"][prefix] = {
            m: seed_stats(scores[f"spike_{m}_correct_{prefix}"], base) for m in MODES}

    Path(a.out).write_text(json.dumps(verdict, indent=1))
    print(f"outcome_row={outcome} C3_SURVIVES={wrong04} ruling={verdict['ruling']} "
          f"repro_ok={verdict['repro_ok']} -> {a.out}")


if __name__ == "__main__":
    main()
