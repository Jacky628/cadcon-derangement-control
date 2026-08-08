#!/usr/bin/env python
"""blinded_power_check.py — 盲态功效检查，决定是否触发扩样。

**在 replication_analysis.py 之前运行，且其输出不含任何方向信息。**

预注册增补 §13 的执行工具。它只计算一个 nuisance parameter——配对差的非零计数——
据此决定是否需要以更大样本重做。非零计数与效应方向无关，因此依它调整样本量属于
盲态样本量重估（blinded sample size re-estimation），不膨胀 I 类错误率；而看过
方向或 p 值之后再决定扩样是 optional stopping，会使名义 α 失效。

本脚本**故意不实现** mean、符号、Wilcoxon 或任何比较方向的代码路径：盲态由工具
保证，而不是靠执行者的自律。输出里出现的每一个数都是计数。

用法：.venv/bin/python blinded_power_check.py [--dir repl_results]
                                              [--frozen replication_frozen_lists.json]
                                              [--out blinded_power_check.json]
退出码：0 = 未触发（按 n 定稿），10 = 触发扩样，1 = 数据不完整无法判定
"""
import argparse, collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spike_analysis import recall_4tag, load_jsonl

MODES = ("token", "text")
DECISION_CELLS = ("shuffled",)      # 触发依据只看主判据格；masked 只报告
REPORT_CELLS = ("masked",)
PREFIX = "p04"
LABEL = "repl"
SEEDS = (0, 1, 2)
MIN_SEEDS_UNDERPOWERED_TO_TRIGGER = 2      # >=2/3 seeds 失守则该模式的多数不可靠


def gated_recall(row):
    produced = set(row["geom_produced"]) if row["executable"] else set()
    r = recall_4tag(row["intended"], produced)
    if r is None:
        return None
    return r if row["executable"] else 0.0


def nonzero_count(scores, cond, base, seed, analysis_idx):
    """配对差中非零的个数。只返回计数——不返回差值本身，也不返回任何符号。"""
    a, b = scores.get(cond, {}).get(seed, {}), scores.get(base, {}).get(seed, {})
    idx = sorted((set(a) & set(b)) & analysis_idx)
    n_pairs = len(idx)
    nz = sum(1 for i in idx if a[i] != b[i])
    return n_pairs, nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="repl_results")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--out", default="blinded_power_check.json")
    a = ap.parse_args()

    frozen = json.loads(Path(a.frozen).read_text())
    analysis_idx = {s["idx"] for s in frozen["samples"]}
    n = frozen["analysis"]["n"]
    threshold = frozen["analysis"]["nonzero_guardrail_count"]

    rows = load_jsonl(Path(a.dir) / "geom_scores.jsonl")
    scores = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["idx"] not in analysis_idx:
            continue
        v = gated_recall(r)
        if v is not None:
            scores[r["condition"]][r["seed"]][r["idx"]] = v

    base = f"{LABEL}_baseline_{PREFIX}"
    result = {"n": n, "threshold": threshold, "prefix": PREFIX,
              "rule": "trigger if >=%d of 3 seeds fall below the threshold in EITHER mode of "
                      "the decision cell" % MIN_SEEDS_UNDERPOWERED_TO_TRIGGER,
              "decision_cells": {}, "report_only_cells": {}, "incomplete": []}

    triggered = False
    for header in DECISION_CELLS + REPORT_CELLS:
        bucket = "decision_cells" if header in DECISION_CELLS else "report_only_cells"
        cell = {}
        for m in MODES:
            cond = f"{LABEL}_{m}_{header}_{PREFIX}"
            per_seed, under = {}, 0
            for s in SEEDS:
                n_pairs, nz = nonzero_count(scores, cond, base, s, analysis_idx)
                if n_pairs == 0:
                    result["incomplete"].append(f"{cond} seed{s}")
                per_seed[str(s)] = {"n_pairs": n_pairs, "nonzero": nz,
                                    "below_threshold": nz < threshold}
                under += int(nz < threshold)
            cell[m] = {"per_seed": per_seed, "n_seeds_below_threshold": under}
            if header in DECISION_CELLS and under >= MIN_SEEDS_UNDERPOWERED_TO_TRIGGER:
                triggered = True
        result[bucket][header] = cell

    if result["incomplete"]:
        result["verdict"] = "INCOMPLETE"
        print(f"INCOMPLETE — 缺少臂: {result['incomplete']}", flush=True)
        Path(a.out).write_text(json.dumps(result, indent=1))
        return 1

    result["verdict"] = "TRIGGER_EXPANSION" if triggered else "NO_TRIGGER"
    Path(a.out).write_text(json.dumps(result, indent=1))

    print(f"盲态功效检查 (n={n}, 阈值={threshold})")
    for header in DECISION_CELLS + REPORT_CELLS:
        tag = "判据格" if header in DECISION_CELLS else "仅报告"
        bucket = "decision_cells" if header in DECISION_CELLS else "report_only_cells"
        for m in MODES:
            c = result[bucket][header][m]
            counts = [c["per_seed"][str(s)]["nonzero"] for s in SEEDS]
            print(f"  [{tag}] {header:9s} {m:5s} 非零计数 seed0/1/2 = {counts} "
                  f"→ 低于阈值的 seed 数 {c['n_seeds_below_threshold']}")
    print(f"\n裁定：{result['verdict']}")
    if triggered:
        print("  → 触发扩样。按增补 §13：本次 n=400 的结果以「功效不足、不定论」报告，")
        print("     扩样为独立的新预注册实验，不与本次 pooling。")
    else:
        print("  → 不触发。判据按 n=400 定稿；此后不得因结果不理想而扩样。")
    print("\n（本脚本不计算也不输出任何方向、均值或 p 值。）")
    return 10 if triggered else 0


if __name__ == "__main__":
    sys.exit(main())
