# C-spike 预注册判据与执行 spec（冻结于评测开跑之前）

> 日期：2026-07-06 ｜ 状态：**v2（对抗审查后修订版），待附录 A 填毕即冻结**
> v1→v2：经独立对抗审查（17 条发现，NEEDS-REVISION），全部必修项已落实；修订对照见附录 B。审查引用的承重数字已由主会话独立复核（42/100 MP-intent、10 个纯 MP、GT 几何真 MP 6/26、生成侧 0/62、STOP_AT import 时绑定）。
> 依据：`Phase2_审计备忘_第二轮复核_20260703.md` + `Phase2_红队盲复现报告_20260703.md` + 2026-07-06 地面侦察（`scratchpad/cspike_recon.json`）
> 用户拍板（2026-07-06）：①发车；②成功定义=诚实 TMLR/workshop 发表算成功；③C3 塌→全线终止（无 B' 打捞）；④spike 期间并行做 NWCAD-aware 重写挂 arXiv。

## 0. 目的（一句话）

在**独立于训练 regex 的几何度量**下、在**去前缀（prefix-0%）与原协议（prefix-40%）两种设定**下，重测 CADCON 的头条主张 C3（wrong-content header 使 adherence 跌破无条件基线）与 H2 锚点（masked < baseline），判定 Phase-2 是否有可存档的科学内核。**本文档在任何评测运行之前冻结全部判定规则；开跑后不得修改判据。**

## 1. 评测矩阵（冻结）

**条件模型**（全部为既有 checkpoint，纯推理，零训练）：

| 角色 | checkpoint | seeds |
|---|---|---|
| 条件化-token | `sft_constraint_tokens_seed{0,1,2}` | 3 |
| 条件化-text | `sft_constraint_text_seed{0,1,2}` | 3 |
| 无条件基线 | `sft_no_constraints_seed{0,1,2}` | 3 |

**格子**：每个条件化模型 × {prefix 0.0, 0.4} × {header: correct, shuffled, masked}=六格；基线 × {prefix 0.0, 0.4}（无 header）。
共 **42 个 eval job**（6 格 × 2 模式 × 3 seed + 2 × 3 基线），每 job 100 样本（固定 dedup 干净测试集前 100，与论文同集同序，idx 跨条件配对）。

- `shuffled` = `_build_shuffle_plan`（main.py:584-597）：取另一真实样本的 well-formed header（错误内容），rng seed 固定 → 与论文 Experiment A 同构造、完全确定。
- `masked` = `eval_mode="none"`（无 header），非 `header_source` 维度（结构性如此）。
- **不含** corrupted 剂量组（GO 后 Phase-2 主体再做）；不扩 eval 集（F1 修复属 Phase-2 主体，非 spike）。
- 解码：贪心（do_sample=False）、bf16、batch 8、`calibrate_generation` 自适应 cap——与论文完全同路径。
- **发车命令（冻结，含全部旗标）**：
  `.venv/bin/python run_dualgpu.py --jobs spike_sixcell.json --merge-dir spike_results --cleantest-cache _cleantest_cache --gpus 0,1`
  （新 condition 标签一律 `spike_*` 前缀；`spike_sixcell.json` 的 sha256 与测试集 idx→gt_code 哈希清单冻结于附录 A。）

## 2. 度量（冻结）

### 2.1 主度量：4-tag 几何侧 recall（执行计入式）

对每个生成程序：在隔离子进程执行（timeout 10s）→ 对执行成功的 B-rep 实体跑几何断言（`geom_requirements.py`，采纳 CADTestBench 的 check()/requirement-group 协议，MIT）。**下表为规范定义——恰好这 4 条单断言决定 tag 归属，requirement-group 仅为封装**：

| tag | 几何断言（在执行后实体上） |
|---|---|
| CIRCLE | 存在 geomType()=="CIRCLE" 的边 |
| NGON | 存在含 ≥5 条 LINE 边的面 |
| THIN | bbox 纵横比 z/max(x,y) < 0.3 |
| TALL | 同比 > 2.0 |

- **MULTI_PART 从主度量剔除**（预注册依据，冻结前书面证据：06-28 审计显示 GT 侧 regex-MP 26 个可执行样本中几何真 MP 仅 6、生成侧 0/62——重叠 extrude 融合为单实体使该 tag 几何不可测；42/100 样本含 MP-intent、其中 10 个纯 MP-intent 会制造结构性零差值削检验功效）。5-tag 版本降为敏感性分析 §2.2(c')。
- **执行计入式（primary）**：执行失败（定义 = `geom_requirements.py` 自身子进程执行失败或 10s 超时，**不是** adherence 行里 oracle 的 valid 位）的程序，该样本几何 recall 记 0。
- **分析集 n=76（冻结）**：intent（4-tag，regex(GT)）非空的 76 个样本，idx 清单见附录 A；14 个 5-tag 空 intent + 10 个纯 MP-intent 样本不进 recall，只进 executability 表（executability 率按全 100 报）。
- intended 侧 = regex(GT)（header 编码的即是它，这是接口定义而非度量循环；produced 侧的独立性才是 F3 的要害）。
- **判定只用 recall**。precision/F1 只入附录表（分母空时按现行 None-剔除约定），**不得在摘要/引言/结论中作为 C3 任一方向的证据引用**（#17）。
- `geom_requirements.py` 的 sha256 在 **Step 1 冒烟之前**记入附录 A 并随归档 commit；此后任何改动只能走钢铁条款 (i) 的 bug 例外并留痕。

### 2.2 副度量（预注册的敏感性分析，全部只报不改判）

(a) 可执行子集 recall + 每格 executability rate（必报，暴露选择效应）；(b) 配对比较用两格可执行 idx 交集；(c') 5-tag 版本（含 MULTI_PART）；(d) intended 改用 geom(GT)（仅 GT 可执行子集）；(e) 训练 regex 度量照算（旧数复现对照，见 §4 Step 2 判定阈值）；**(f) regex 度量 × 执行门（invalid→0）——补全「探测器 × 执行门」2×2 分解，四格全报**（审计已证可执行子集上 geom 与 regex 逐一相等，主度量的增量主要来自执行门；此分解用于归因，不预设结论）。

**措辞规则（冻结）**：若 primary 通过但副度量 (b)（可执行交集）的 mean 效应在 ≥2/3 seed 上 ≥0，则论文主张必须表述为「wrong intent degrades **usable-output** adherence（validity-mediated）」，禁止表述为 conditional-on-valid 的 adherence 下降。

### 2.3 已知度量局限（如实带入论文，不在 spike 内修）

THIN/TALL 阈值 0.3/2.0 与训练 regex 共享（度量实现独立、阈值选择不独立）；THIN/TALL 轴锁 z；可执行子集上几何探测器与 regex 高度一致（62/62 逐一相等）——主度量的独立增量主要在执行门与直接打分（非仅验证阳性）两处。

## 3. 判定规则（冻结——按格预注册主张改写表）

**配对（#7）**：一切 Δ 均为**同 seed 配对**：Δ(s) = 条件臂(seed s) − baseline(seed s)，按 idx 逐样本配对。决策链路中任何位置禁止使用 seed 平均的 baseline。

**冻结检验规格（#1）**：显著性检验 = `scipy.stats.wilcoxon(d, zero_method="pratt", alternative="less", method="asymptotic")`，d = 同 seed 下（条件臂 − baseline）的 per-sample 4-tag 几何 recall 配对差（n=76）；p 取原始值不四舍五入；检验退化（全零差）计为不显著；scipy 版本记入附录 A。**方向 = mean(d) 的符号**（不用 median——重零时常无定义）；mean(d)=0 计为方向不一致。其他 zero_method/侧别/exact 组合作为敏感性分析只报不改判。

**完整判据（对给定模式 m、给定 prefix p 的「wrong<baseline」比较）**：
1. text 模式：3 seed 中 mean(d)<0 的 ≥2/3，且 ≥2/3 个 seed 冻结检验 p<0.05；
2. token 模式：3 seed 中 mean(d)<0 的 ≥2/3（方向要求，不要求显著）；
3. **一票否决（#5）**：任一 seed、任一模式在冻结检验的反向检验（alternative="greater"）下 p<0.05 → 该判据不成立；
4. **profile 级方向护栏（#8）**：text@该 prefix 的 profile 级 mean 效应点估计 ≤0（profile = 4-tag intent 集合的去重分组，非空 profile 清单见附录 A；per-profile recall 均值，seed 平均，仅方向要求非显著性要求——防 Simpson 反转，不让 n≈12 的低功效检验变成误杀扳机）。

**profile 级稳健性检验（只报不改判，#8）**：profile 级配对 Wilcoxon，zero_method="pratt"、双侧、单位 = 非空 profile 的 per-profile recall 均值、seed 平均（与 `paired_power.py` 同规格，tag 集换 4-tag）。

**表格判定符（#6）**：下表 ✔ ⇔ 该 prefix 下**完整判据（1-4 全部）成立**；✘ = 其余一切情形。✔/✘ 是二值判据结果，不是点估计符号。两列同规格。

**C3 存活判定（用户拍板"终止"开关）**：C3 存活 ⇔ Δ_wrong(0.4) 列 = ✔。

| # | Δ_wrong(0.4) | Δ_wrong(0.0) | 裁决 | 头条改写 |
|---|---|---|---|---|
| 1 | ✔ | ✔ | **GO** | "wrong intent hurts more than no intent"（CAD 首次隔离，NWCAD 词汇），跨补全/生成两设定稳健——最强 |
| 2 | ✔ | ✘ | **GO** | 收窄为条件-上下文冲突设定（=neutral regression 的 CAD 实例）；**附带条款（#9）：正文与摘要必须写明 prefix-0 未检出效应、0.4 协议存在已记录的前缀泄漏（F2）耦合** |
| 3 | ✘ | ✔ | **NO-GO** | **理由（#9 改写）**：发表主张所在格（0.4）直接重测失败；0.0-only 效应是需要新实验支撑的另一个主张，超出 spike 授权范围——按用户拍板 ③ 不豁免终止。该观察如实写入更正型负结果并标注为 salvageable |
| 4 | ✘ | ✘ | **NO-GO** | C3 彻底死亡；更正型负结果写作 |

**归因措辞（#4）**：结局 3/4 的死因**不得预设为「度量假象」**——归因交由 §2.2(f) 的 2×2 分解报告；结局 1/2 的存活若主要由执行门驱动（见 §2.2 措辞规则），按 validity-mediated 表述。

**H2 锚点副判定（#16）**：锚定格 = **text 模式 @ prefix 0.4**：Δ_none(0.4) 按完整判据（1-4，token 条款相应替换为 token-masked 方向）评定；不成立 → 论文"习得有害 header 依赖"段落降级为 regex 度量限定的观察。其余 Δ_none 格描述性报告。

**钢铁条款**：(i) 开跑后不得增删格子/换度量/改阈值/改判据；发现管线 bug 可修复重跑，但须在报告中记录；(ii) 结局 2 不得在摘要或正文中伪装成结局 1；(iii) 无论结局如何，本文档与结果一并归档。

## 4. 执行计划

**Step 0 — 代码 glue（开跑前，改动清单全列）**：
1. `eval_validity` 增 ~4-8 LOC：每样本落盘 `generated_raw / program_extracted / prompt / gt_idx`（main.py:1135-1143 的 append 记录处）。改前备份 `main.py.bak-cspike-<ts>`，改后 **stage-10/experiment 双向同步**（产物同步规则）。
2. 生成 `spike_sixcell.json`（42 job，`spike_*` 标签）+ **dry-check 脚本**：校验无 `header_source=shuffled ∧ eval_mode=none` 静默降级组合、无旧 condition 标签、prefix∈{0.0,0.4}。
3. **预算分支：验证而非改动（#13）**——STOP_AT 在模块 import 时绑定（70h×0.80），每 job 独立进程 10-31 分钟，该分支结构性死；冒烟断言 worker.log 无 "scaling down eval" 且 job_result total_count=8；全量后断言 42 个 job_result total_count=100。零代码改动。
4. `geom_requirements.py`：§2.1 规范断言 + check() 协议 + per-requirement 汇报（借 CADTestBench 输出 schema），进程池化（~1.9s/程序 × ~4200 次 ≈ 8 核池 ~20min）。
5. **冻结（#10）**：本文档 + `geom_requirements.py` + `spike_sixcell.json` 的 sha256、n=76 idx 清单、非空 profile 清单、scipy 版本、复现对照格清单**全部在 Step 1 冒烟之前**记入附录 A 并 git commit（归档目录）。

**Step 1 — 冒烟（8 样本 × 3 job，#12）**：
- `token-correct@0.4 seed2`（管线 + 394-tensor 轻量格式 checkpoint 的 vocab/embed 尺寸核对，一石二鸟）；
- `text-shuffled@0.0 seed0`（史上唯一没跑过的组合类型；断言 worker.log 打印 `HEADER_SOURCE: shuffled differs-from-correct on X/n` 且 X>0）；
- `baseline@0.0 seed0`。
验证 raw 落盘字段齐全、断言管线端到端、无 "scaling down eval"。冒烟结果丢弃、不进分析。

**Step 2 — 全量 42 job**：冻结命令（§1），双卡 ~5-8h。**regex 旧数复现对照（#11）**：复现对象 = results.json 已有对应旧数的 cell×seed（清单见附录 A）；通过 ⇔ 每格 regex adherence_recall 绝对偏差 ≤0.01 且 validity_rate 偏差 ≤0.02；任一超差 → 停跑、根因、留痕、重跑受影响 job；复现结果**永不用于重新解释几何侧结论**。

**Step 3 — 几何打分（CPU）+ 预注册分析 + 报告**。

**时间盒（#15）**：计时起点 = Step 2 发车时间戳（落日志）；自起点**硬上限 5 天**。超时 → **NO-GO(budget)**：按拍板 ③ 终止，但报告必须写明 **C3 既未证实也未证伪**，不得从不完整矩阵出任何方向的裁决。边界情形（现在写死）：若 42 job 已全部完成而打分/分析未完，允许 **+48h 仅打分与分析延期**（不重跑、不补跑）。

## 5. 风险与预案（侦察实测）

| 风险 | 预案 |
|---|---|
| token_seed2 轻量格式（394 vs 398 tensors） | 冒烟即用 seed2 核对 vocab/embed；不符则该 seed 标注并降级为 2-seed 判据（方向 2/2 + 显著 2/2，一票否决照旧） |
| shuffled 使 executability 大变 → 主度量被 validity 淹没 | 主度量本就执行计入（合并效应=真实部署效应）；§2.2(a)(b)(f) 三路拆解 + 措辞规则强制 validity-mediated 表述 |
| GT 程序 22% 不可执行 | 不影响 produced 侧打分；intended 主定义 regex(GT) 不受影响 |
| 双卡争用/中断 | job 级断点续跑（orchestrator 原生）；`spike_results` 新 merge-dir 隔离旧 skip 键 |
| 旧数不复现 | §4 Step 2 阈值化处置（≤0.01/≤0.02），不再要求"逐字节" |
| 全零差退化 | 冻结规则：计为不显著（§3） |

## 6. 与三份裁决文档的对应

- 六格矩阵 + 微型化：红队盲复现报告「建议吸收 2 项」之 (a)；
- 独立度量 = CADTests 风格：round-2 审计 §2.2（采纳 harness 与断言风格而非成品套件——侦察证实成品套件绑定其自有样本，不适用）；
- C3 塌→终止：用户拍板 ③（V2 立场）；
- 预注册+时间盒：红队三方共同收敛点；
- v2 全部修订：对抗审查（2026-07-06，17 条发现），见附录 B。

---

## 附录 A：冻结记录（已于 Step 1 冒烟前填写，2026-07-06 09:35:35 +0800）

冻结件归档于 `cspike_frozen/`（同目录），哈希清单 `cspike_frozen/sha256s.txt`：

- `geom_requirements.py`：`df380e8178e0523a92682fb4584dd1792013ece36937b6401fe0c9afe394cb7a`
- `spike_sixcell.json`（42 job）：`1d5c8810c6280975aef798d09b623087361281ca4e976d5bab73c228583f8347`
- `spike_smoke.json`（3 job 冒烟）：`3eb1ecf5f4ab040ea795321ab020e7396cf74cf91431f28de7b00b11d52d22cc`
- `spike_frozen_lists.json`（含以下全部清单）：`32296d0842b5dd1c27dfcc5cec72e307effe5b254f1431c8c40260a5ad5c5f33`
  - **n=76 分析集 idx 清单**（`analysis_idx_n76`）；14 个空 5-tag intent idx（`empty_5tag_idx`）
  - **非空 4-tag profile 清单**：**6 个 profile**（`profiles_4tag`；注意 4-tag 下 profile 数为 6，非 5-tag 的 12——§3 判据 4 的方向护栏与稳健性检验均按此 6 profile 执行）
  - 测试集前 100 样本 idx→gt_code sha256（`testset_gt_sha256`；与 06-28 审计 `holdout_samples.jsonl` 的 regex_intended 交叉验证 **0 失配**）
  - **regex 复现对照 12 格**（`repro_reference`，含旧 condition 名与逐 seed 旧值；shuffled@0.0 两格无旧数不在对照内）
- 补丁后 `main.py`（沙盒=stage-10 字节一致）：`37d66c4d36465c091fca212af1d0307f334ee2caa21e77f7a924cde4c7c32d70`（改动=append_adherence_sample 增 4 字段落盘 + 收编 stage-10 的 dedup-on-write；备份 `main.py.bak-cspike-20260706-092859`）
- scipy 版本：**1.17.1**
- 本文档（v2 终稿）sha256：见 `cspike_frozen/sha256s.txt` 追加行（文档自身哈希在本行写入后计算，故存于外部清单）
- 冻结时间：**2026-07-06 09:35 +0800**；对抗审查（17 条）与主会话独立复核记录见附录 B 及会话档案
- 附加冻结承诺：分析脚本 `spike_analysis.py` 将在 Step 3 打分前完成并把 sha256 追加至 `cspike_frozen/sha256s.txt`（只实现 §2/§3 已冻结规则，不引入新自由度）

## 附录 B：对抗审查修订对照（17 条）

| # | 发现 | 处置 |
|---|---|---|
| 1 | Wilcoxon 未定规格（项目烧伤史） | ✅ 冻结 pratt/less/asymptotic + mean 方向 + 退化=不显著（§3） |
| 2 | 14 空 intent 样本使 n=100 自相矛盾 | ✅ 主分析集冻结为 4-tag 非空 n=76（§2.1；含 #3 连带的 10 个纯 MP 剔除） |
| 3 | MULTI_PART 焊进 kill switch | ✅ 选项 (i)：4-tag primary，5-tag 降敏感性（§2.1，附复核数字） |
| 4 | 可执行子集上 geom≡regex，主检验实为执行门 | ✅ 增 2×2 分解 (f) + validity-mediated 措辞规则 + 结局 3/4 归因不预设（§2.2/§3） |
| 5 | 显著反向异议 seed 可通过 | ✅ 一票否决条款（§3 判据 3） |
| 6 | 0.0 列无判定标准 | ✅ ✔=完整判据成立，两列同规格（§3） |
| 7 | baseline 配对未定 | ✅ 同 seed 配对，禁 seed 平均（§3） |
| 8 | profile 级检验未指定+存活方向漏洞 | ✅ 冻结 paired_power 规格 + text 点估计 ≤0 方向护栏（§3 判据 4） |
| 9 | 结局 3 理由失实、结局 2 缺附带条款 | ✅ 理由改写为绑定版本 + 结局 2 正文条款（§3 表） |
| 10 | 冻结时点矛盾 + 断言规范歧义 + 执行失败双定义 | ✅ 统一冒烟前冻结 + 规范定义声明 + 执行失败钉死为 geom 子进程（§2.1/§4） |
| 11 | 复现对照无阈值 | ✅ ≤0.01/≤0.02 阈值 + 对象清单 + 永不反向解释（§4 Step 2） |
| 12 | 冒烟测不到 seed2/shuffled | ✅ 冒烟改 3 job：seed2-correct@0.4 / text-shuffled@0.0 / baseline@0.0（§4 Step 1） |
| 13 | 预算旋钮是运行时 no-op | ✅ 改为验证而非改动（已核实 STOP_AT import 绑定）（§4 Step 0.3） |
| 14 | 发车命令缺旗标 | ✅ 冻结全命令 + job JSON/测试集哈希入附录（§1） |
| 15 | 超时与证据 NO-GO 混同 | ✅ NO-GO(budget) 区分 + 计时起点 + +48h 仅打分延期（§4） |
| 16 | H2 降级未锚定 | ✅ 锚定 text@0.4（§3） |
| 17 | precision 无角色 | ✅ 判定只用 recall，precision/F1 附录且禁作证据（§2.1） |
