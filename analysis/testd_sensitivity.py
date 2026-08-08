#!/usr/bin/env python
"""testd_sensitivity.py — Test D 的描述性敏感性分析（判据已出，本脚本不改判）。

产出论文 Appendix B 需要的三块：
  (1) leave-one-profile-out：逐 profile 剔除后重算 Test D，检验 dissociation 是否依赖某一类几何
  (2) 可执行子集视图：去掉执行计入的零填充后交互如何变化，以及各臂的可执行子集水平
  (3) profile 加权对比：等权 vs 按样本量加权

检验规格沿用冻结实现（`spike_analysis.wtest`，pratt / greater / asymptotic），与
`testd_analysis.py` 一致；本脚本只做已冻结判据之外的描述性分解，不产生任何裁决。

用法：.venv/bin/python testd_sensitivity.py [--dir testd_analysis_input]
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_analysis as SA

SEEDS = (0, 1, 2)
M_C, M_W = "repl_text_correct_p04", "repl_text_shuffled_p04"
R_C, R_W = "replD_R_correct_p04", "replD_R_shuffled_p04"
BASE = "repl_baseline_p04"


def build(rows, idxs):
    sc = collections.defaultdict(lambda: collections.defaultdict(dict))
    ex = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["idx"] not in idxs:
            continue
        rec = SA.recall_4tag(r["intended"],
                             set(r["geom_produced"]) if r["executable"] else set())
        if rec is None:
            continue
        sc[r["condition"]][r["seed"]][r["idx"]] = 0.0 if not r["executable"] else rec
        ex[r["condition"]][r["seed"]][r["idx"]] = bool(r["executable"])
    return sc, ex


def testd_on(sc, keep):
    """在给定 idx 子集上重算 Test D，返回 per-seed (mean, p_greater, n)。"""
    out = []
    for s in SEEDS:
        common = sorted(set(sc[M_C][s]) & set(sc[M_W][s]) & set(sc[R_C][s]) & set(sc[R_W][s]) & keep)
        inter = [(sc[M_C][s][i] - sc[M_W][s][i]) - (sc[R_C][s][i] - sc[R_W][s][i]) for i in common]
        p, _ = SA.wtest(inter, "greater")
        out.append((float(np.mean(inter)) if inter else None,
                    None if math.isnan(p) else p, len(inter)))
    return out


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="testd_analysis_input")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--out", default="testd_sensitivity.json")
    a = ap.parse_args()

    frozen = json.loads(Path(a.frozen).read_text())
    idxs = {s["idx"] for s in frozen["samples"]}
    profiles = frozen["profiles_4tag"]
    rows = SA.load_jsonl(Path(a.dir) / "geom_scores.jsonl")
    sc, ex = build(rows, idxs)

    res = {"n": len(idxs), "note": "descriptive sensitivity; the frozen verdict is in testd_verdict.json"}

    # ── (1) leave-one-profile-out ───────────────────────────────────────────
    full = testd_on(sc, idxs)
    print(f"完整集 (n={full[0][2]}): "
          f"交互 {' / '.join(f'{m:+.3f}' for m,_,_ in full)}  "
          f"p {' / '.join(fmt_p(p) for _,p,_ in full)}")
    print(f"\n{'剔除的 profile':24s} {'剔除数':>6s} {'交互 s0/s1/s2':>26s} {'p s0/s1/s2':>28s}  方向/显著")
    loo = {}
    for prof in sorted(profiles, key=lambda p: -len(p["idx"])):
        rm = set(prof["idx"])
        keep = idxs - rm
        r = testd_on(sc, keep)
        ndir = sum(1 for m, _, _ in r if m is not None and m > 0)
        nsig = sum(1 for _, p, _ in r if p is not None and p < 0.05)
        loo[prof["profile"]] = {"n_removed": len(rm),
                                "per_seed": [{"mean": m, "p_greater": p, "n": n} for m, p, n in r],
                                "n_direction": ndir, "n_significant": nsig}
        print(f"{prof['profile']:24s} {len(rm):6d} "
              f"{' / '.join(f'{m:+.3f}' for m,_,_ in r):>26s} "
              f"{' / '.join(fmt_p(p) for _,p,_ in r):>28s}  {ndir}/{nsig}")

    # 含 circle 的全部样本
    circ = {i for p in profiles if "CIRCLE" in p["profile"] for i in p["idx"]}
    r = testd_on(sc, idxs - circ)
    ndir = sum(1 for m, _, _ in r if m is not None and m > 0)
    nsig = sum(1 for _, p, _ in r if p is not None and p < 0.05)
    loo["all circle-containing"] = {"n_removed": len(circ),
                                    "per_seed": [{"mean": m, "p_greater": p, "n": n} for m, p, n in r],
                                    "n_direction": ndir, "n_significant": nsig}
    print(f"{'all circle-containing':24s} {len(circ):6d} "
          f"{' / '.join(f'{m:+.3f}' for m,_,_ in r):>26s} "
          f"{' / '.join(fmt_p(p) for _,p,_ in r):>28s}  {ndir}/{nsig}")
    res["leave_one_profile_out"] = loo
    res["full_set"] = [{"mean": m, "p_greater": p, "n": n} for m, p, n in full]

    surviving = all(v["n_direction"] >= 2 and v["n_significant"] >= 2 for v in loo.values())
    res["survives_every_removal_2of3"] = surviving
    print(f"\n每次剔除后仍满足 ≥2/3 方向且 ≥2/3 显著: {'是' if surviving else '否'}")

    # ── (2) 可执行子集视图 ───────────────────────────────────────────────────
    print("\n=== 可执行子集（去掉执行计入的零填充）===")
    per_seed_exec = []
    for s in SEEDS:
        vals = []
        for i in sorted(idxs):
            if all(ex[c][s].get(i) for c in (M_C, M_W, R_C, R_W)):
                vals.append((sc[M_C][s][i] - sc[M_W][s][i]) - (sc[R_C][s][i] - sc[R_W][s][i]))
        per_seed_exec.append((float(np.mean(vals)) if vals else None, len(vals)))
    print(f"  四臂皆可执行交集上的交互: "
          f"{' / '.join(f'{m:+.3f}(N={n})' for m, n in per_seed_exec)}")
    arm_exec = {}
    for label, cond in (("baseline", BASE), ("M_correct", M_C), ("M_wrong", M_W),
                        ("R_correct", R_C), ("R_wrong", R_W)):
        v = [sc[cond][s][i] for s in SEEDS for i in sc[cond][s] if ex[cond][s].get(i)]
        arm_exec[label] = {"mean": float(np.mean(v)), "n": len(v)}
        print(f"  {label:10s} 可执行子集 adherence = {np.mean(v):.3f}  (n={len(v)})")
    res["executable_only"] = {"interaction_per_seed": per_seed_exec, "arm_means": arm_exec}

    # ── (3) profile 加权 ────────────────────────────────────────────────────
    effs, ws = [], []
    for prof in profiles:
        pid = set(prof["idx"])
        v = []
        for s in SEEDS:
            vv = [(sc[M_C][s][i] - sc[M_W][s][i]) - (sc[R_C][s][i] - sc[R_W][s][i])
                  for i in sorted(pid & set(sc[M_C][s]))]
            if vv:
                v.append(float(np.mean(vv)))
        if v:
            effs.append(float(np.mean(v)))
            ws.append(len(pid))
    unw = float(np.mean(effs))
    wtd = float(np.average(effs, weights=ws))
    res["profile_weighting"] = {"unweighted": unw, "sample_size_weighted": wtd,
                                "n_profiles": len(effs)}
    print(f"\n=== profile 加权 ===\n  等权 {unw:+.3f} | 按样本量加权 {wtd:+.3f} "
          f"（{len(effs)} 个 profile，全部计入）")

    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
