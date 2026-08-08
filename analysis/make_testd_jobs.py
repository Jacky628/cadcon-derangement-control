#!/usr/bin/env python
"""make_testd_jobs.py — 生成并 dry-check Test D 重测的 6 个 eval 作业。

依 `复制实验_TestD增补预注册_20260806.md` §4：R 臂在复制样本（400 个冻结程序）上重测
correct / shuffled 两个 header 条件 × 3 seed；M 臂复用复制实验已产出的
`repl_text_correct_p04` / `repl_text_shuffled_p04`，不重跑。

R 臂 checkpoint 已在盘（f4_text_shufhdr_seed{0,1,2}），无需任何训练。

dry-check（全部致命，先于 GPU）：
  * 6 个作业、(condition, seed) 无重复、每 condition 恰好 3 seed
  * 全部 prefix 0.4、eval_subset_size == 冻结样本的 n
  * 无 header_source != correct 与 eval_mode == none 的静默降级组合
  * train_mode 恒为 text（R 是 text-header 训练的 derangement 控制）
  * 引用的 checkpoint 全部存在
  * condition 标签带 replD_ 前缀，与复制实验的 repl_ 命名空间隔离，
    避免误并入 repl_results 的 skip 键

用法：.venv/bin/python make_testd_jobs.py [--out replD_jobs.json]
"""
import argparse, hashlib, json
from pathlib import Path

SEEDS = (0, 1, 2)
PREFIX = 0.4
CKPT = "f4_text_shufhdr_seed{s}"
ARMS = (("correct", "correct"), ("shuffled", "shuffled"))


def build(n: int) -> list:
    jobs = []
    for label, header_source in ARMS:
        for s in SEEDS:
            jobs.append({
                "kind": "eval",
                "condition": f"replD_R_{label}_p04",
                "ckpt": CKPT.format(s=s),
                "seed": s,
                "eval_mode": "text",     # R 是 text-header 训练的控制模型
                "train_mode": "text",
                "prefix_fraction": PREFIX,
                "eval_subset_size": n,
                "header_source": header_source,
            })
    return jobs


def dry_check(jobs, n, sandbox: Path):
    assert len(jobs) == 6, f"{len(jobs)} jobs, expected 6"
    keys = [(j["condition"], j["seed"]) for j in jobs]
    assert len(set(keys)) == len(keys), "duplicate (condition, seed)"
    counts = {}
    for j in jobs:
        counts[j["condition"]] = counts.get(j["condition"], 0) + 1
        assert j["prefix_fraction"] == PREFIX, f"{j['condition']}: prefix != {PREFIX}"
        assert j["eval_subset_size"] == n, f"{j['condition']}: eval_subset_size != {n}"
        assert j["train_mode"] == "text", f"{j['condition']}: R must be text-trained"
        assert j["eval_mode"] == "text", f"{j['condition']}: R arms keep the header present"
        assert not (j["header_source"] != "correct" and j["eval_mode"] == "none"), \
            f"{j['condition']}: wrong-header request silently downgraded to masked"
        assert j["condition"].startswith("replD_"), f"stale label: {j['condition']}"
        ck = sandbox / "checkpoints" / j["ckpt"]
        assert (ck / "adapter_config.json").exists(), f"checkpoint missing: {ck}"
    assert set(counts.values()) == {3}, f"expected 3 seeds per condition: {counts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="replD_jobs.json")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    a = ap.parse_args()
    sandbox = Path(__file__).resolve().parent
    n = json.loads((sandbox / a.frozen).read_text())["analysis"]["n"]

    jobs = build(n)
    dry_check(jobs, n, sandbox)
    Path(a.out).write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{a.out:22s} {len(jobs)} jobs (n={n})  "
          f"sha256={hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}")
    for j in jobs:
        print(f"  {j['condition']:24s} seed={j['seed']} ckpt={j['ckpt']} "
              f"header_source={j['header_source']}")


if __name__ == "__main__":
    main()
