# 提取器修复重跑 —— RUN LOG

预注册：`_paper_artifacts_rc-fdedd0/提取器修复_独立预注册_20260812.md`（冻结件 sha256 见
`sha256s.txt` 末尾「提取器修复重跑」段）。

---

## 2026-08-12 01:51 UTC —— 第一次发车，27/27 作业快速失败，零数据产出

**现象**：全部作业 rc=1，均在模型加载前退出。

**根因**：`replfix_generate.py` 未把 `--frozen-list` 解析为绝对路径。worker 以 per-job 目录为
cwd 运行，相对路径 `replication_frozen_lists.json` 因此解析失败。
`run_dualgpu.run_orchestrate` 对 `--cleantest-cache` 做了 `.resolve()`（其注释：
"MUST be absolute: workers run with cwd=per-job dir"），但对 `--frozen-list` 未做同样处理
——既有的不对称，此前的调用方碰巧传了可解析的路径故未暴露。

**处置**：修在 `replfix_generate.py`，`run_dualgpu.py` **未修改**（哈希不变）。
不涉及判据、度量、阈值、护栏、规则 A、`eval_batch_size`、`max_new_tokens`、结局映射或
§10 的写法，属预注册 §12 允许的管线修复。修订后 sha256 已追加入册。

**数据影响**：无。失败发生在模型加载前，`replfix_results/` 已整目录删除后重建。

---

## 2026-08-12 01:54:19 UTC —— 正式发车

- 编排 pid 6081，27 作业，双卡（GPU 0/1）
- `frozen-sample binding OK (400 programs)` —— 冻结样本绑定检查通过
- 作业清单 sha256 `79da79140fb4b56eb91911311f0660d5d5b662ca94a3bfcfb79f72674a3fbc96`

### 02:04 UTC 实测吞吐与时间盒偏差（如实记录，不改预注册）

| 项 | 值 |
|---|---|
| 实测 | 0.136 样本/秒/卡 → **51 分钟/作业**（含约 110 s 模型加载） |
| 母预注册实测校验 | 21 作业 8.71 h → **50 分钟/作业** |
| 27 作业双卡 makespan | **≈ 11.5 小时** |
| 预计完成 | 13:22 UTC / 21:22 北京时间 |

**与预注册 §12「预计 ≤10.6 h」的偏差及其原因**：10.6 h 由 `8.71 h + 1.9 h` 相加得到，
但这两个数各自都是**双卡并行后的 makespan**（分别为 21 作业与 6 作业两批），相加等于假设
两批串行执行。27 作业合为一批时正确的外推是 `8.71 × 27/21 ≈ 11.2 h`。

每作业耗时与母预注册实测（51 vs 50 分钟）高度一致，**说明生成路径无性能异常**，偏差纯粹
来自当初的外推方法。

**协议影响：无。**§12 的硬上限为 36 小时，11.5 « 36。预计值不是判据量。预注册为冻结件，
**不因此修改**；偏差记录于此。

---

## 跑完后的三步（顺序不可换，见预注册 §3、§5）

1. `replfix_selfcheck.py` —— legacy 规则在本次生成上必须逐位复现已发表的
   `repl_results/geom_scores.jsonl` 与 `testd_results/geom_scores.jsonl`。
   **不通过即判环境漂移，须先定位解决；在此之前 `replfix_analysis.py` 的任何数字不得读作结果。**
2. `replfix_score.py` —— 双面离线评分（GT 探测外推约 2.3 h，14 线程）。
3. `replfix_analysis.py` —— 簇单位（298）主判据 + 护栏 a–d + Test D + 结局分类。
