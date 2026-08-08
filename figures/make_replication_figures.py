#!/usr/bin/env python
"""make_replication_figures.py — Figures 2-4 重制于复制实验样本（n=400）。

不改 `make_figures.py`：本脚本把复制实验的产物
(`replication_figures_data.json` + `testd_verdict.json` + `testd_sensitivity.json`)
适配成 `make_figures.py` 期望的 schema，写出 `figures_data_replication.json`，
再以 exec() 复用同一份绘图代码，因此视觉语言与 v1 的图逐像素同源。

输出文件名加 `_repl` 后缀，v1 的三张图原封不动保留——论文需要并置两次运行，
不能把旧图覆盖掉。

用法：<sandbox>/.venv/bin/python make_replication_figures.py
"""
import json, re, sys
from pathlib import Path

# --- released copy -----------------------------------------------------------
# The frozen original reads its three inputs from an absolute sandbox path. Only
# those three bindings differ from the version whose hash is in
# replication_frozen/sha256s.txt; the plotting code is untouched.
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"                  # path binding, released copy only

R = json.loads((HERE / "replication_figures_data.json").read_text())
V = json.loads((DATA / "testd_verdict.json").read_text())
S = json.loads((DATA / "testd_sensitivity.json").read_text())
SEEDS = ("0", "1", "2")

# ── Figure 2：七臂 geom vs regex ─────────────────────────────────────────────
# v1 的键含 p00 列；复制实验按预注册只跑 p04，故只填 p04 键（绘图只读 p04）。
arms = R["arms"]
t2g = {"baseline_p04": arms["baseline"]["geom"]}
t2r = {"baseline_p04": arms["baseline"]["regex"]}
for tok in ("text", "token"):
    for src, dst in (("correct", "correct"), ("wrong", "wrong"), ("masked", "masked")):
        t2g[f"{tok}_{dst}_p04"] = arms[f"{tok}_{src}"]["geom"]
        t2r[f"{tok}_{dst}_p04"] = arms[f"{tok}_{src}"]["regex"]

# ── Figure 3：dissociation ──────────────────────────────────────────────────
ps = V["test_D"]["per_seed"]
testd = {
    "drop_M": [ps[s]["drop_M_mean"] for s in SEEDS],
    "drop_R": [ps[s]["drop_R_mean"] for s in SEEDS],
    "inter_mean": [ps[s]["interaction_mean"] for s in SEEDS],
    "p": [ps[s]["p_greater"] for s in SEEDS],
    "nonzero": [ps[s]["nonzero_pairs"] for s in SEEDS],
    "guard_a_mean": V["test_D"]["guard_a_exec_intersection_mean"],
    "guard_b_unweighted": S["profile_weighting"]["unweighted"],
    "guard_b_weighted": S["profile_weighting"]["sample_size_weighted"],
}

# ── Figure 4：output identity（分子/分母对）──────────────────────────────────
ident = {
    lab: [[d["identical"], d["n_jointly_executable"]] for d in R["output_identity"][lab]]
    for lab in ("M", "R")
}

D_OUT = {"table2": {"geom": t2g, "regex": t2r}, "testd": testd, "identity": ident,
         "gate": {"R_c_seed": V["competence_gate"]["R_correct_per_seed"],
                  "R_w_seed": V["competence_gate"]["R_shuffled_per_seed"],
                  "R_exec_pooled": V["competence_gate"]["R_exec_rate"],
                  "baseline_exec": V["competence_gate"]["baseline_p04_exec_rate"]},
         "exec_only": {lab: S["executable_only"]["arm_means"][key]["mean"]
                       for lab, key in (("R", "R_correct"), ("B", "baseline"), ("M", "M_correct"))},
         "_provenance": "replication n=400 (repl_verdict/testd_verdict/testd_sensitivity)"}
(HERE / "figures_data_replication.json").write_text(json.dumps(D_OUT, indent=1))

# ── 复用 make_figures.py 的绘图代码，只替换数据源与输出名 ─────────────────────
src = (HERE / "make_figures.py").read_text()
src = src.replace("D = json.load(open(HERE / 'figures_data.json'))",
                  "D = json.load(open(HERE / 'figures_data_replication.json'))")
for fig in ("fig2_seven_arms", "fig3_dissociation", "fig4_output_identity"):
    src = src.replace(f"HERE / '{fig}'", f"HERE / '{fig}_repl'")
# v1 图的 y 轴上限 0.80 是按 v1 的臂高定的；复制样本的 regex 臂更高，需放宽
hi = max(max(t2g.values()), max(t2r.values()))
src = src.replace("ax.set_ylim(0, 0.80)", f"ax.set_ylim(0, {max(0.80, hi*1.18):.2f})")
src = src.replace("ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])",
                  "ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])")
for anchor in ("0.755", "0.769", "0.700", "0.714"):
    src = src.replace(f"({anchor}", f"({float(anchor)*1.18:.3f}")
    src = src.replace(f" {anchor},", f" {float(anchor)*1.18:.3f},")

sys.path.insert(0, str(HERE))
exec(compile(src, str(HERE / "make_figures.py"), "exec"),
     {"__name__": "__main__", "__file__": str(HERE / "make_figures.py")})

print("\n复制实验版图表:")
for f in sorted(HERE.glob("*_repl.*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
print("\nv1 原图保持不变:")
for f in sorted(HERE.glob("fig[234]_*")):
    if "_repl" not in f.name:
        print(f"  {f.name}")
