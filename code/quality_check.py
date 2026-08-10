"""逐条件质量复查(共享:main.py 每条件完成后自动调 / pause_status.py 手动 dump)。

分级:
  - WARN = 软信号(只报告 + 落盘),如 validity 退化、adherence 异常、跨臂方向反。
    这些多半是「结果」而非 bug,自动中断会误杀,留给人 + 最后的 paired 分析判断。
  - FAIL = 结构性故障(需干预),如 checkpoint 没存好、核心指标缺失。下游复用会崩/白烧。

纯 stdlib,无副作用,易单测。CANARY(EOS abort)是另一层,不在此。
"""
import math

GRPO_CONDS = {
    "grpo_proxy_reward_classifier",
    "grpo_hidden_state_only_proxy",
    "grpo_oracle_reward_reference",
}
# 「条件化」臂(与 no_constraints 比方向)
CONDITIONED_ARMS = {
    "qwen25coder_sft_constraint_tokens",
    "qwen25coder_sft_constraint_text",
}
DIRECTION_TOL = 0.15  # 低于 base 超过这个幅度才提示方向反


def _is_nan(v):
    return isinstance(v, float) and math.isnan(v)


def _find(records, cond, seed):
    for r in records or []:
        if r.get("condition") == cond and r.get("seed") == seed:
            return r
    return None


def check_condition(record, prior_records=None, ckpt_ok=None):
    """返回 (status, flags)。status ∈ {PASS, WARN, FAIL};flags = 原因列表。"""
    flags = []
    hard = False
    cond = record.get("condition", "")
    seed = record.get("seed")

    # ── 结构性 FAIL(需干预)──
    if ckpt_ok is False:
        flags.append("checkpoint 未保存(结构性:下游复用会崩/静默重训)")
        hard = True
    if "validity_rate" not in record:
        flags.append("validity_rate 缺失(条件未正常产出)")
        hard = True

    # ── 软信号 WARN(只报告)──
    vr = record.get("validity_rate")
    if vr == 0.0:
        flags.append("validity_rate==0(可疑退化)")
    elif vr == 1.0:
        flags.append("validity_rate==1.0(可疑满分)")

    f1 = record.get("adherence_f1")
    if _is_nan(f1):
        flags.append("adherence_f1 是 nan")
    for k in ("adherence_recall", "adherence_precision"):
        v = record.get(k)
        if v is not None and not _is_nan(v) and not (0.0 <= v <= 1.0):
            flags.append(f"{k}={v} ∉ [0,1]")

    if cond in GRPO_CONDS:
        ag = record.get("proxy_oracle_agreement_rate")
        if ag in (0.0, 1.0):  # None=未测量,不算退化
            flags.append(f"proxy_oracle_agreement={ag}(退化)")

    # 跨臂方向 sanity(同 seed,条件化臂 adherence 明显低于 no_constraints)
    if cond in CONDITIONED_ARMS and f1 is not None and not _is_nan(f1):
        base = _find(prior_records, "qwen25coder_sft_no_constraints", seed)
        if base is not None:
            bf1 = base.get("adherence_f1")
            if bf1 is not None and not _is_nan(bf1) and f1 < bf1 - DIRECTION_TOL:
                flags.append(
                    "adherence 明显低于 no_constraints(H2 方向反,仅提示)")

    if hard:
        return "FAIL", flags
    if flags:
        return "WARN", flags
    return "PASS", flags


def is_dependency_safe(source_cond, seed, prior_records):
    """下游条件复用 source_cond 的 checkpoint 前调用。
    只在源条件 **结构性 FAIL** 时阻断(软 WARN 如 validity=0 不阻断——可能是真差结果)。
    返回 (ok, reason)。"""
    rec = _find(prior_records, source_cond, seed)
    if rec is None:
        return True, "源条件结果未记录(交由 checkpoint 复用逻辑处理)"
    status, flags = check_condition(rec)
    if status == "FAIL":
        return False, f"源条件 {source_cond} seed={seed} 质量 FAIL: {flags}"
    return True, "ok"


def format_flag_line(record, prior_records=None, ckpt_ok=None):
    """给 main.py 自动打日志用的一行 QUALITY_CHECK 串。"""
    status, flags = check_condition(record, prior_records, ckpt_ok)
    cond = record.get("condition", "?")
    seed = record.get("seed", "?")
    reason = "; ".join(flags) if flags else "ok"
    return f"QUALITY_CHECK: condition={cond} seed={seed} status={status} :: {reason}"
