#!/usr/bin/env python
"""testd_analysis.py — Test D 在复制样本上的冻结分析。

实现 `复制实验_TestD增补预注册_20260806.md`，该增补只改「在哪批样本上测」，判定规则逐字
沿用 `F4_预注册判据_20260707.md` §4。检验机制从 `spike_analysis` import 复用，不重实现，
与 `f4_analysis.py` 的做法一致（母文档 finding #8）。

相对 F4 的差异，全部来自增补 §2，且仅此四项：
  * 分析集 = 复制实验的 400 个唯一程序（无需簇内平均；本脚本对唯一性设断言）
  * profile = 11 个，护栏 (b) 只计入 n>=15 的 10 个
  * 非零对护栏 (c) = 60 (=0.15n)，并采用严格读法：若 >=2/3 多数在剔除 underpowered seed
    后不再成立，判据不成立
  * drop_M 取自 repl_results，并核验 M 臂均值与 repl_verdict.json 一致（复现 assert）

执行顺序（母文档 §4）：R 胜任度门 → （通过才读）Test D + 护栏 → 支撑检查。

用法：.venv/bin/python testd_analysis.py [--dir repl_results] [--out testd_verdict.json]
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spike_analysis as SA          # 冻结的 recall / wtest，复用不重实现

SEEDS = (0, 1, 2)
ALPHA = 0.05
GATE_FLOOR = 0.30                    # 母文档阈值，不因换样本而调整
GATE_EXEC_MARGIN = 0.10              # 同上
GUARDRAIL_MIN_PROFILE_N = 15         # 增补 §5(b)

M_CORRECT = "repl_text_correct_p04"
M_WRONG = "repl_text_shuffled_p04"
R_CORRECT = "replD_R_correct_p04"
R_WRONG = "replD_R_shuffled_p04"
BASELINE = "repl_baseline_p04"


def load_rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def build(rows, idxs):
    """(scores, executable, header_injected)，均为 {cond: {seed: {idx: v}}}。"""
    sc = collections.defaultdict(lambda: collections.defaultdict(dict))
    ex = collections.defaultdict(lambda: collections.defaultdict(dict))
    hdr = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["idx"] not in idxs:
            continue
        rec = SA.recall_4tag(r["intended"],
                             set(r["geom_produced"]) if r["executable"] else set())
        if rec is None:
            continue
        sc[r["condition"]][r["seed"]][r["idx"]] = 0.0 if not r["executable"] else rec
        ex[r["condition"]][r["seed"]][r["idx"]] = bool(r["executable"])
        if "header_injected" in r:
            hdr[r["condition"]][r["seed"]][r["idx"]] = tuple(sorted(r["header_injected"]))
    return sc, ex, hdr


def arm_mean(sc, cond):
    per = [float(np.mean(list(sc[cond][s].values()))) if sc[cond].get(s) else None for s in SEEDS]
    return per


def exec_rate(ex, cond):
    return [float(np.mean(list(ex[cond][s].values()))) if ex[cond].get(s) else None for s in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="repl_results")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--repl-verdict", default="repl_verdict.json")
    ap.add_argument("--out", default="testd_verdict.json")
    a = ap.parse_args()

    frozen = json.loads(Path(a.frozen).read_text())
    samples = frozen["samples"]
    idxs = {s["idx"] for s in samples}
    n = frozen["analysis"]["n"]
    min_nonzero = frozen["analysis"]["nonzero_guardrail_count"]
    profiles = frozen["profiles_4tag"]

    assert len({s["program_sha256"] for s in samples}) == n, \
        "冻结样本含重复程序——Test D 的配对单位将不成立"

    rows = load_rows(Path(a.dir) / "geom_scores.jsonl")
    sc, ex, hdr = build(rows, idxs)

    missing = [c for c in (M_CORRECT, M_WRONG, R_CORRECT, R_WRONG, BASELINE) if c not in sc]
    assert not missing, f"缺少条件: {missing}"

    # ── 复现 assert：M 臂必须与已冻结的复制实验判据一致 ────────────────────
    rv = json.loads(Path(a.repl_verdict).read_text())
    for s in SEEDS:
        d_here = [sc[M_WRONG][s][i] - sc[BASELINE][s][i]
                  for i in sorted(set(sc[M_WRONG][s]) & set(sc[BASELINE][s]))]
        m_here = float(np.mean(d_here))
        m_frozen = rv["cells"]["shuffled_p04"]["per_seed"]["text"][str(s)]["mean"]
        assert abs(m_here - m_frozen) < 1e-9, \
            f"seed{s}: M 臂 wrong-vs-baseline 均值 {m_here} != 冻结判据的 {m_frozen}"

    # ── M / R 必须注入同一个 wrong header（增补 §4）─────────────────────────
    mism = [(s, i) for s in SEEDS for i in sorted(idxs)
            if hdr[M_WRONG].get(s, {}).get(i) != hdr[R_WRONG].get(s, {}).get(i)]
    assert not mism, (f"M 与 R 的 wrong header 注入不一致，共 {len(mism)} 处，"
                      f"首例 seed={mism[0][0]} idx={mism[0][1]}；Test D 的交互无意义")

    V = {"n": n, "min_nonzero_required": min_nonzero,
         "prereg": "复制实验_TestD增补预注册_20260806.md",
         "assert_M_matches_frozen_verdict": True,
         "assert_same_wrong_header_M_R": True}

    # ── R 胜任度门（前置）────────────────────────────────────────────────────
    rc, rs = arm_mean(sc, R_CORRECT), arm_mean(sc, R_WRONG)
    r_exec = float(np.mean([v for v in exec_rate(ex, R_CORRECT) + exec_rate(ex, R_WRONG)
                            if v is not None]))
    base_exec = float(np.mean([v for v in exec_rate(ex, BASELINE) if v is not None]))
    n_rc = sum(1 for v in rc if v is not None and v >= GATE_FLOOR)
    n_rs = sum(1 for v in rs if v is not None and v >= GATE_FLOOR)
    gate = (n_rc >= 2 and n_rs >= 2 and r_exec >= base_exec - GATE_EXEC_MARGIN)
    V["competence_gate"] = {
        "R_correct_per_seed": rc, "R_shuffled_per_seed": rs,
        "R_exec_rate": r_exec, "baseline_p04_exec_rate": base_exec,
        "floor": GATE_FLOOR, "exec_margin": GATE_EXEC_MARGIN,
        "n_R_correct_above_floor": n_rc, "n_R_shuffled_above_floor": n_rs,
        "PASS": bool(gate),
    }

    # ── Test D（门通过后才有意义，但仍计算并落盘）──────────────────────────
    per_seed = {}
    for s in SEEDS:
        common = sorted(set(sc[M_CORRECT][s]) & set(sc[M_WRONG][s])
                        & set(sc[R_CORRECT][s]) & set(sc[R_WRONG][s]))
        dM = [sc[M_CORRECT][s][i] - sc[M_WRONG][s][i] for i in common]
        dR = [sc[R_CORRECT][s][i] - sc[R_WRONG][s][i] for i in common]
        inter = [m - r for m, r in zip(dM, dR)]
        nz = int(sum(1 for x in inter if x != 0))
        pg, _ = SA.wtest(inter, "greater")
        pl, _ = SA.wtest(inter, "less")
        per_seed[str(s)] = {
            "n": len(inter), "nonzero_pairs": nz, "underpowered": nz < min_nonzero,
            "drop_M_mean": float(np.mean(dM)), "drop_R_mean": float(np.mean(dR)),
            "interaction_mean": float(np.mean(inter)),
            "p_greater": None if math.isnan(pg) else pg,
            "p_less": None if math.isnan(pl) else pl,
        }

    dir_ok = {s: per_seed[str(s)]["interaction_mean"] > 0 for s in SEEDS}
    sig_ok = {s: (per_seed[str(s)]["p_greater"] is not None
                  and per_seed[str(s)]["p_greater"] < ALPHA) for s in SEEDS}
    n_dir = sum(dir_ok.values())
    n_sig = sum(sig_ok.values())
    n_dir_pow = sum(1 for s in SEEDS if dir_ok[s] and not per_seed[str(s)]["underpowered"])
    n_sig_pow = sum(1 for s in SEEDS if sig_ok[s] and not per_seed[str(s)]["underpowered"])
    veto = [s for s in SEEDS if per_seed[str(s)]["p_less"] is not None
            and per_seed[str(s)]["p_less"] < ALPHA]

    # 护栏 (a) 可执行交集
    inter_exec = []
    for s in SEEDS:
        for i in sorted(idxs):
            if all(ex[c][s].get(i) for c in (M_CORRECT, M_WRONG, R_CORRECT, R_WRONG)):
                inter_exec.append((sc[M_CORRECT][s][i] - sc[M_WRONG][s][i])
                                  - (sc[R_CORRECT][s][i] - sc[R_WRONG][s][i]))
    guard_a = float(np.mean(inter_exec)) if inter_exec else None

    # 护栏 (b) profile 级方向，只计 n>=15
    prof_detail, counted = [], []
    for prof in profiles:
        pid = set(prof["idx"])
        effs = []
        for s in SEEDS:
            vals = [(sc[M_CORRECT][s][i] - sc[M_WRONG][s][i])
                    - (sc[R_CORRECT][s][i] - sc[R_WRONG][s][i])
                    for i in sorted(pid & set(sc[M_CORRECT][s]) & set(sc[R_CORRECT][s]))]
            if vals:
                effs.append(float(np.mean(vals)))
        if effs:
            e = float(np.mean(effs))
            inc = len(pid) >= GUARDRAIL_MIN_PROFILE_N
            prof_detail.append({"profile": prof["profile"], "n_idx": len(pid),
                                "effect": e, "counted_in_guardrail": inc})
            if inc:
                counted.append(e)
    guard_b = float(np.mean(counted)) if counted else None

    strict_ok = not (n_dir >= 2 and n_dir_pow < 2) and not (n_sig >= 2 and n_sig_pow < 2)
    testD_pass = bool(n_dir >= 2 and n_sig >= 2 and not veto and strict_ok
                      and guard_a is not None and guard_a > 0
                      and guard_b is not None and guard_b >= 0)

    V["test_D"] = {
        "per_seed": per_seed,
        "n_direction_positive": n_dir, "n_significant": n_sig,
        "n_direction_positive_powered": n_dir_pow, "n_significant_powered": n_sig_pow,
        "majority_survives_underpowered_removal": strict_ok,
        "reverse_veto_seeds": veto,
        "underpowered_seeds": [s for s in SEEDS if per_seed[str(s)]["underpowered"]],
        "guard_a_exec_intersection_mean": guard_a,
        "guard_b_profile_mean": guard_b,
        "guard_b_n_counted": len(counted),
        "guard_b_detail": prof_detail,
        "PASS": testD_pass,
    }

    if not gate:
        V["ruling"] = "UNINTERPRETABLE"
        V["headline"] = "对照无效（R 贴地板/退化）→ 回落母文档 §0 降级 framing；不得写 dissociation confirmed"
    elif testD_pass:
        V["ruling"] = "DISSOCIATION_SEMANTIC"
        V["headline"] = "语义驱动确认：强版 wrong design-intent hurts；明写排除分布漂移"
    else:
        V["ruling"] = "NO_DISSOCIATION"
        V["headline"] = "分布漂移/失配未被排除 → 母文档 §0 降级 framing"

    Path(a.out).write_text(json.dumps(V, indent=1, default=float))
    g = V["competence_gate"]
    print(f"R 胜任度门: R_correct={[round(x,3) if x else x for x in rc]} "
          f"R_shuffled={[round(x,3) if x else x for x in rs]}")
    print(f"           R_exec={r_exec:.3f} vs baseline_exec={base_exec:.3f} "
          f"(门槛 {base_exec-GATE_EXEC_MARGIN:.3f}) → PASS={g['PASS']}")
    if gate:
        for s in SEEDS:
            d = per_seed[str(s)]
            print(f"  Test D seed{s}: drop_M={d['drop_M_mean']:+.4f} drop_R={d['drop_R_mean']:+.4f} "
                  f"interaction={d['interaction_mean']:+.4f} p_greater={d['p_greater']:.4g} "
                  f"非零={d['nonzero_pairs']}")
        print(f"  方向 {n_dir}/3  显著 {n_sig}/3  否决 {veto or '无'}  "
              f"护栏a={guard_a:+.4f}  护栏b={guard_b:+.4f} ({len(counted)} profiles)")
    print(f"\n裁决: {V['ruling']}\n{V['headline']}")


if __name__ == "__main__":
    main()
