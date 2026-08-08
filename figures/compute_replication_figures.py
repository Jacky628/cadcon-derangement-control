#!/usr/bin/env python
"""compute_replication_figures.py — 复制实验与 Test D 重测的图表/表格数字。

与 `release/figures/compute_all.py` 同构：从冻结原始分数重算每一个将出现在论文里的数字，
用 import 的 `spike_analysis` 机制（不重实现 recall/检验），并对已冻结判据中的对应值设
复现断言。产出 `replication_figures_data.json`。

覆盖：
  A. 七臂 adherence（几何 vs 训练 regex）—— Figure 2
  B. Test D per-seed drop 与交互 —— Figure 3
  C. output identity（correct/wrong 下产出几何是否相同）—— Figure 4
  D. correct-header 增益（两次运行对照，含 token 臂的符号翻转）
  E. per-profile 全表（两个对比 + 五个臂绝对值）—— Appendix C

用法：.venv/bin/python compute_replication_figures.py
"""
import collections, json, math, sys
from pathlib import Path

import numpy as np

# --- released copy -----------------------------------------------------------
# The frozen original runs in a flat directory. Only the path bindings marked
# below differ from the version whose hash is in replication_frozen/sha256s.txt;
# the computation is untouched.
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"          # path binding, released copy only
DATA = HERE.parent / "data"                  # path binding, released copy only
sys.path.insert(0, str(ANALYSIS))
import spike_analysis as SA

SEEDS = (0, 1, 2)
TAGS4 = set(SA.TAGS_PRIMARY)
INP = DATA / "testd_analysis_input" / "geom_scores.jsonl"        # path binding
FROZEN = json.loads((ANALYSIS / "replication_frozen_lists.json").read_text())  # path binding
IDXS = {s["idx"] for s in FROZEN["samples"]}
PROFILES = FROZEN["profiles_4tag"]

ARMS = {
    "baseline":      "repl_baseline_p04",
    "text_correct":  "repl_text_correct_p04",
    "text_wrong":    "repl_text_shuffled_p04",
    "text_masked":   "repl_text_masked_p04",
    "token_correct": "repl_token_correct_p04",
    "token_wrong":   "repl_token_shuffled_p04",
    "token_masked":  "repl_token_masked_p04",
    "R_correct":     "replD_R_correct_p04",
    "R_wrong":       "replD_R_shuffled_p04",
}

report = []
def check(name, got, expected, tol=5e-4):
    ok = abs(got - expected) <= tol if isinstance(expected, float) else got == expected
    report.append((name, ok, expected, got))
    return ok


rows = [json.loads(l) for l in INP.read_text().splitlines() if l.strip()]
geom = collections.defaultdict(lambda: collections.defaultdict(dict))   # 几何侧
rgx = collections.defaultdict(lambda: collections.defaultdict(dict))    # 训练 regex 侧
exe = collections.defaultdict(lambda: collections.defaultdict(dict))
prod = collections.defaultdict(lambda: collections.defaultdict(dict))   # 产出的几何特征集
for r in rows:
    if r["idx"] not in IDXS:
        continue
    g = SA.recall_4tag(r["intended"], set(r["geom_produced"]) if r["executable"] else set())
    if g is None:
        continue
    c, s, i = r["condition"], r["seed"], r["idx"]
    geom[c][s][i] = 0.0 if not r["executable"] else g
    # regex 侧：论文的对照口径是「不加执行门」的训练度量
    rq = SA.recall_4tag(r["intended"], set(r.get("produced", [])))
    rgx[c][s][i] = rq if rq is not None else 0.0
    exe[c][s][i] = bool(r["executable"])
    prod[c][s][i] = frozenset(set(r["geom_produced"]) & TAGS4) if r["executable"] else None

D = {"n": len(IDXS), "source": str(INP.name)}

# ── A. 七臂 adherence ───────────────────────────────────────────────────────
D["arms"] = {}
for label, cond in ARMS.items():
    gm = float(np.mean([geom[cond][s][i] for s in SEEDS for i in geom[cond][s]]))
    rm = float(np.mean([rgx[cond][s][i] for s in SEEDS for i in rgx[cond][s]]))
    ex = float(np.mean([exe[cond][s][i] for s in SEEDS for i in exe[cond][s]]))
    per_seed = [float(np.mean(list(geom[cond][s].values()))) for s in SEEDS]
    D["arms"][label] = {"geom": gm, "regex": rm, "executability": ex, "geom_per_seed": per_seed}

base = D["arms"]["baseline"]["geom"]
check("baseline geom == 0.406", round(base, 3), 0.406)

# ── B. Test D per-seed ──────────────────────────────────────────────────────
verdict = json.loads((DATA / "testd_verdict.json").read_text())   # path binding
D["test_d"] = {"per_seed": []}
for s in SEEDS:
    common = sorted(set(geom[ARMS["text_correct"]][s]) & set(geom[ARMS["text_wrong"]][s])
                    & set(geom[ARMS["R_correct"]][s]) & set(geom[ARMS["R_wrong"]][s]))
    dM = [geom[ARMS["text_correct"]][s][i] - geom[ARMS["text_wrong"]][s][i] for i in common]
    dR = [geom[ARMS["R_correct"]][s][i] - geom[ARMS["R_wrong"]][s][i] for i in common]
    inter = [a - b for a, b in zip(dM, dR)]
    p, _ = SA.wtest(inter, "greater")
    D["test_d"]["per_seed"].append({
        "seed": s, "drop_M": float(np.mean(dM)), "drop_R": float(np.mean(dR)),
        "interaction": float(np.mean(inter)), "p_greater": None if math.isnan(p) else p})
    fv = verdict["test_D"]["per_seed"][str(s)]
    check(f"drop_M seed{s} 复现冻结判据", round(float(np.mean(dM)), 4), round(fv["drop_M_mean"], 4))
    check(f"交互 seed{s} 复现冻结判据", round(float(np.mean(inter)), 4), round(fv["interaction_mean"], 4))

# ── C. output identity ──────────────────────────────────────────────────────
D["output_identity"] = {}
for label, (cc, cw) in (("M", (ARMS["text_correct"], ARMS["text_wrong"])),
                        ("R", (ARMS["R_correct"], ARMS["R_wrong"]))):
    per_seed = []
    for s in SEEDS:
        both = [i for i in sorted(IDXS)
                if exe[cc][s].get(i) and exe[cw][s].get(i)]
        same = sum(1 for i in both if prod[cc][s][i] == prod[cw][s][i])
        per_seed.append({"seed": s, "n_jointly_executable": len(both),
                         "identical": same, "fraction": same / len(both) if both else None})
    D["output_identity"][label] = per_seed

# ── D. correct-header 增益（两次运行对照）────────────────────────────────────
D["correct_header_gain"] = {}
for mode in ("text", "token"):
    g = D["arms"][f"{mode}_correct"]["geom"] - base
    r = D["arms"][f"{mode}_correct"]["regex"] - D["arms"]["baseline"]["regex"]
    D["correct_header_gain"][mode] = {"geom": g, "regex": r}
check("token correct 几何增益为负", D["correct_header_gain"]["token"]["geom"] < 0, True)

# ── E. per-profile 全表 ─────────────────────────────────────────────────────
D["per_profile"] = []
for prof in PROFILES:
    pid = set(prof["idx"])
    row = {"profile": prof["profile"], "n": len(pid)}
    for label, cond in ARMS.items():
        v = [geom[cond][s][i] for s in SEEDS for i in sorted(pid & set(geom[cond][s]))]
        row[label] = float(np.mean(v)) if v else None
    row["wrong_minus_baseline"] = row["text_wrong"] - row["baseline"]
    effs = []
    for s in SEEDS:
        vv = [(geom[ARMS["text_correct"]][s][i] - geom[ARMS["text_wrong"]][s][i])
              - (geom[ARMS["R_correct"]][s][i] - geom[ARMS["R_wrong"]][s][i])
              for i in sorted(pid & set(geom[ARMS["text_correct"]][s]))]
        if vv:
            effs.append(float(np.mean(vv)))
    row["interaction"] = float(np.mean(effs)) if effs else None
    D["per_profile"].append(row)
D["per_profile"].sort(key=lambda r: -r["n"])

out = HERE / "replication_figures_data.json"
out.write_text(json.dumps(D, indent=1, default=float))

nf = sum(1 for _, ok, _, _ in report if not ok)
for name, ok, exp, got in report:
    if not ok:
        print(f"  FAIL {name}: 期望 {exp} 实得 {got}")
print(f"== {len(report)-nf}/{len(report)} 项复现断言通过 -> {out.name} ==\n")

print("七臂 adherence（几何 / regex / executability）:")
for k, v in D["arms"].items():
    print(f"  {k:14s} geom={v['geom']:.3f}  regex={v['regex']:.3f}  exec={v['executability']:.3f}")
print("\ncorrect-header 增益（相对 baseline）:")
for m, v in D["correct_header_gain"].items():
    print(f"  {m:6s} 几何 {v['geom']:+.3f}   regex {v['regex']:+.3f}")
print("\noutput identity（correct 与 wrong 产出几何相同的比例）:")
for lab, ps in D["output_identity"].items():
    print(f"  {lab}: " + " / ".join(f"{p['fraction']:.2f}(N={p['n_jointly_executable']})" for p in ps))
