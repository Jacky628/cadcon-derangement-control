"""
main.py — Fine-tuning LLMs for Parametric CAD Modeling
Conditions:
  1. deepcad_scratch_transformer          (2 seeds)
  2. qwen25coder_sft_no_constraints       (2 seeds)
  3. qwen25coder_sft_constraint_tokens    (2 seeds)
  4. grpo_proxy_reward_classifier         (2 seeds)
  5. constraint_tokens_masked_at_inference(2 seeds)
  6. grpo_hidden_state_only_proxy         (1 seed)
  7. grpo_oracle_reward_reference         (1 seed)

Primary metric: validity_rate
"""
# ── CRITICAL: pin to single GPU BEFORE importing torch/transformers ──────────
import os
# Both RTX 3090s visible so the model can shard across them (device_map="auto",
# ~48GB total). Sharding (model-parallel) also makes HF Trainer set
# is_model_parallel=True, so it does NOT wrap in DataParallel — which is what
# caused the "chunk expects >=1-dim tensor" crash when a single-GPU model was
# loaded with 2 GPUs visible. This is the run-3 config that handled max_completion=512.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable torch.compile (eager). The real GRPO speedup is the KV cache (see GRPO
# gradient_checkpointing=False + use_cache=True below); torch.compile gave no
# measurable gain on GRPO generation (946s vs 979s profiled) and triggered
# recompilation storms (16+ inductor workers, multi-minute stalls). Eager is
# predictable.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import gc
import json
import hashlib
import re
import math
import time
import random
import traceback
import subprocess
import threading
import tempfile
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

# ── Hyperparameters dict (MANDATORY) ─────────────────────────────────────────
HYPERPARAMETERS = {
    # SFT / LLM
    "base_model": "Qwen/Qwen2.5-Coder-1.5B",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "sft_lr": 2e-4,
    "sft_batch_size": 4,
    "sft_grad_accum": 8,
    "sft_epochs": 3,
    "max_seq_length": 1024,       # >max(1000 measured, 500-sample): 100% programs train COMPLETE + EOS (768 covered 98.4%)
    # GRPO
    "grpo_lr": 5e-7,
    "grpo_steps": 100,
    "grpo_num_generations": 4,
    "grpo_grad_accum": 1,         # MUST set explicitly: TRL GRPOConfig defaults to 8, which
                                  # generates 8× per optimizer step (~8× slower). run-5 used 1.
    "max_completion_length": 768,  # DECOUPLED from max_seq: stays 768 (eval covers p99~862 / 98.4%) — NOT 1024, to keep GRPO generation fast/light
    "entropy_coeff": 0.01,
    "gradient_clip": 1.0,
    # Proxy classifier
    "proxy_hidden_dim": 256,
    "proxy_lr": 1e-4,
    "proxy_offline_samples": 500,
    "proxy_train_epochs": 10,
    # Oracle checks
    "oracle_check_interval": 15,
    "oracle_check_samples": 20,
    "oracle_agreement_threshold": 0.70,
    # Eval
    "eval_subset_size": 100,
    "eval_batch_size": 8,     # batched greedy generation; 8 keeps the fp16 1.5B eval on ONE
                              # 24GB card (est_peak ~16GB) while ~halving wall-clock vs 4.
    "cadquery_timeout": 10,
    # DeepCAD scratch transformer
    "deepcad_d_model": 256,
    "deepcad_n_layers": 6,
    "deepcad_n_heads": 8,
    "deepcad_vocab_size": 256,
    "deepcad_lr": 1e-4,
    "deepcad_batch_size": 32,
    "deepcad_train_steps": 500,   # runtime-calibrated
    # Constraint tokens
    "num_constraint_tokens": 5,
    "eval_prefix_fraction": 0.40,   # eval prompt = this fraction of the program (front). The 5
                                    # design-intent feature SIGNATURES all sit in the BACK half
                                    # (.circle~74% / 5th lineTo~60% / .extrude~87% / 2nd result=~52%
                                    # median), so a front prefix up to ~50% never reveals them — the
                                    # header stays informative regardless. 40% gives the model the
                                    # most scaffolding while staying < the earliest (~52%) feature.
    # Budget
    "total_budget_seconds": 252000,   # 70h wall-clock ceiling; ×0.80 → train-stop at 56h.
    "budget_fraction": 0.80,           # ~44h estimated run → ~12h headroom + 14h finalize buffer
}

# ── Time budget ───────────────────────────────────────────────────────────────
RUN_START = time.time()
BUDGET_SECONDS = HYPERPARAMETERS["total_budget_seconds"]
STOP_AT = RUN_START + BUDGET_SECONDS * HYPERPARAMETERS["budget_fraction"]

def time_remaining():
    return STOP_AT - time.time()

def budget_exceeded():
    return time.time() >= STOP_AT

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/workspace/data/deepcad")
CKPT_DIR = Path("./checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ── Observability helpers ─────────────────────────────────────────────────────
PROGRESS_FILE = Path("progress.json")
PARTIAL_FILE = Path("partial_results.jsonl")
RESULTS_FILE = Path("results.json")
ADHERENCE_FILE = Path("adherence_samples.jsonl")  # per-sample adherence → PAIRED arm comparison
QUALITY_FLAGS_FILE = Path("quality_flags.jsonl")  # per-condition auto quality verdict (persisted)

_progress = {}

def _json_default(o):
    """numpy-safe fallback for json.dumps: convert np scalars/arrays to native types.
    Purely additive — json only calls this for types it can't already serialize, so
    native + np.float64 (a float subclass) are unaffected; it catches np.int64 values
    (NOT an int subclass) and arrays that would otherwise crash the final aggregation."""
    if hasattr(o, "tolist"):    # numpy array OR scalar → native (scalar.tolist() = python scalar,
        return o.tolist()       # array.tolist() = nested list); MUST precede .item() (arrays have
    if hasattr(o, "item"):      # .item() too but it raises on size>1 arrays)
        return o.item()
    return str(o)

def update_progress(**kwargs):
    _progress.update(kwargs)
    _progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        tmp = Path("progress.json.tmp")
        tmp.write_text(json.dumps(_progress, indent=2, default=_json_default))
        os.replace(tmp, PROGRESS_FILE)
    except Exception:
        pass

def append_partial(record: dict):
    # ── 逐条件质量复查(自动,每个条件完成即触发)──────────────────────────────
    # 软信号(validity/adherence 异常、跨臂方向反)→ 只打日志 WARN + 落盘,不中断
    # (多半是结果而非 bug,自动中断会误杀);结构性(核心指标缺失)→ FAIL。
    # 读已写入的行当 prior_records 做跨臂方向 sanity。
    try:
        import quality_check as _qc
        prior = []
        if PARTIAL_FILE.exists():
            for _l in PARTIAL_FILE.read_text().splitlines():
                try:
                    prior.append(json.loads(_l))
                except Exception:
                    pass
        print(_qc.format_flag_line(record, prior_records=prior), flush=True)
        _status, _flags = _qc.check_condition(record, prior_records=prior)
        with open(QUALITY_FLAGS_FILE, "a") as _qf:
            _qf.write(json.dumps({"condition": record.get("condition"),
                                  "seed": record.get("seed"),
                                  "status": _status, "flags": _flags},
                                 default=_json_default) + "\n")
    except Exception:
        traceback.print_exc()  # 复查不能拖垮主流程
    # Dedup-on-write: the same (condition, seed) can be appended more than once
    # (run_grpo writes a placeholder with proxy_oracle_agreement_rate=None, then
    # run_grpo_proxy rewrites it with the real agreement). Keep only the LATEST
    # record per (condition, seed) and replace atomically, so partial_results
    # never carries duplicate / stale-None rows that pollute counting or resume.
    _key = (record.get("condition"), record.get("seed"))
    _kept = []
    if PARTIAL_FILE.exists():
        for _l in PARTIAL_FILE.read_text().splitlines():
            if not _l.strip():
                continue
            try:
                _r = json.loads(_l)
            except Exception:
                _kept.append(_l)  # preserve unparseable lines verbatim
                continue
            if (_r.get("condition"), _r.get("seed")) != _key:
                _kept.append(json.dumps(_r, default=_json_default))
    _kept.append(json.dumps(record, default=_json_default))
    _tmp = f"{PARTIAL_FILE}.tmp"
    with open(_tmp, "w") as f:
        f.write("\n".join(_kept) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(_tmp, PARTIAL_FILE)

def append_adherence_sample(record: dict):
    """One JSONL line per (condition, seed, eval-sample): intended/produced design-intent
    features + per-sample recall/precision/validity. All arms eval the SAME test samples in
    the SAME order, so `idx` pairs samples across conditions → a PAIRED arm comparison (per
    paired_analysis.py) removes between-sample variance and is far more powerful than comparing
    noisy per-condition means."""
    with open(ADHERENCE_FILE, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
        f.flush()

def write_results(results: dict):
    results["hyperparameters"] = HYPERPARAMETERS
    results["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = Path("results.json.tmp")
    tmp.write_text(json.dumps(results, indent=2, default=_json_default))
    os.replace(tmp, RESULTS_FILE)

# ── Seeding ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Checkpoint path helpers (single source of truth) ─────────────────────────
def sft_no_constraint_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"sft_no_constraints_seed{seed}"

def sft_constraint_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"sft_constraint_tokens_seed{seed}"

def sft_constraint_text_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"sft_constraint_text_seed{seed}"

def grpo_proxy_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"grpo_proxy_seed{seed}"

def grpo_oracle_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"grpo_oracle_seed{seed}"

def grpo_hidden_only_ckpt(seed: int) -> Path:
    return CKPT_DIR / f"grpo_hidden_only_seed{seed}"


# ── Smoke-test sizes (shrunk when ARC_SMOKE=1) ────────────────────────────────
SMOKE = os.environ.get("ARC_SMOKE", "") == "1"
if SMOKE:
    print("=== SMOKE TEST MODE ===")
    HYPERPARAMETERS["sft_epochs"] = 1
    HYPERPARAMETERS["deepcad_train_steps"] = 2
    HYPERPARAMETERS["grpo_steps"] = 1
    HYPERPARAMETERS["eval_subset_size"] = 2
    HYPERPARAMETERS["proxy_offline_samples"] = 4
    HYPERPARAMETERS["proxy_train_epochs"] = 1
    HYPERPARAMETERS["oracle_check_samples"] = 2
    HYPERPARAMETERS["max_completion_length"] = 64  # shrink generation length so the smoke is fast
    SEEDS_OVERRIDE = {
        "deepcad_scratch_transformer": [0],
        "qwen25coder_sft_no_constraints": [0],
        "qwen25coder_sft_constraint_text": [0],
        "qwen25coder_sft_constraint_tokens": [0],
        "grpo_proxy_reward_classifier": [0],
        "constraint_tokens_masked_at_inference": [0],
        "constraint_text_masked_at_inference": [0],
        "grpo_hidden_state_only_proxy": [0],
        "grpo_oracle_reward_reference": [0],
    }
else:
    SEEDS_OVERRIDE = None

# ── Condition seeds (from plan) ───────────────────────────────────────────────
CONDITION_SEEDS = {
    "deepcad_scratch_transformer": [0, 1],
    "qwen25coder_sft_no_constraints": [0, 1, 2],
    "qwen25coder_sft_constraint_text": [0, 1, 2],
    "qwen25coder_sft_constraint_tokens": [0, 1, 2],
    "grpo_proxy_reward_classifier": [0, 1],
    "constraint_tokens_masked_at_inference": [0, 1, 2],
    "constraint_text_masked_at_inference": [0, 1, 2],
    "grpo_hidden_state_only_proxy": [0],
    "grpo_oracle_reward_reference": [0],
    # Experiment A (Task 3) — run via run_dualgpu.py, NOT main()'s hardcoded loops. Documentary.
    "constraint_tokens_shuffled_at_inference": [0, 1, 2],
    "constraint_text_shuffled_at_inference": [0, 1, 2],
    "constraint_tokens_corrupted_at_inference": [0, 1, 2],
    "constraint_text_corrupted_at_inference": [0, 1, 2],
}

# Constraint representation per SFT arm → (train_mode, eval_mode). Source of truth for the
# 4-arm constraint-aware-tokenization comparison; the main() blocks below match these.
# masked arm: trained WITH constraint tokens (train="token") but eval WITHOUT the header
# (eval="none") — isolates the value of the model SEEING the constraints at inference.
CONDITION_CONSTRAINT_MODE = {
    "qwen25coder_sft_no_constraints": ("none", "none"),
    "qwen25coder_sft_constraint_text": ("text", "text"),
    "qwen25coder_sft_constraint_tokens": ("token", "token"),
    "constraint_tokens_masked_at_inference": ("token", "none"),
    "constraint_text_masked_at_inference": ("text", "none"),
    # Experiment A: header PRESENT (eval_mode=token/text) but wrong CONTENT (header_source
    # shuffled/corrupted, a separate dim passed to eval_validity / run_dualgpu jobs).
    "constraint_tokens_shuffled_at_inference": ("token", "token"),
    "constraint_text_shuffled_at_inference": ("text", "text"),
    "constraint_tokens_corrupted_at_inference": ("token", "token"),
    "constraint_text_corrupted_at_inference": ("text", "text"),
}

def get_seeds(condition):
    if SEEDS_OVERRIDE:
        return SEEDS_OVERRIDE.get(condition, [0])
    return CONDITION_SEEDS[condition]

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════
from datasets import load_from_disk, load_dataset as hf_load_dataset
from deepcad_transpiler import deepcad_json_to_cadquery, extract_text_from_sample

_dataset_cache = {}
_train_dedup_hashes = None   # (exact_set, loose_set) over full train, built once
_clean_test_cache = None     # held-out (train-disjoint) test Dataset, built once

# ── Held-out set policy (added 2026-08-05 for the replication run) ────────────
# Both defaults FIX measured defects of the v1 protocol and can be reverted per-run:
#   ARC_HOLDOUT_INTERNAL_DEDUP=0  -> keep duplicates inside the held-out split (v1 behaviour)
#   ARC_TEST_SAMPLING=head        -> take the first N clean rows (v1 behaviour)
HOLDOUT_INTERNAL_DEDUP = os.environ.get("ARC_HOLDOUT_INTERNAL_DEDUP", "1") != "0"
# Concurrency for the in-loop CadQuery oracle (see eval_validity). 6 per job, two jobs in
# flight on this 16-core box, leaves headroom so a program never times out merely because the
# machine is busy — the 10s ceiling sits 3-5x above the measured 1.8-2.9s per program.
# ARC_ORACLE_WORKERS=1 restores the serial path.
ORACLE_WORKERS = int(os.environ.get("ARC_ORACLE_WORKERS", "6"))
TEST_SAMPLING = os.environ.get("ARC_TEST_SAMPLING", "stratified")     # stratified | head
TEST_SAMPLE_SEED = int(os.environ.get("ARC_TEST_SAMPLE_SEED", "20260805"))
TEST_SAMPLE_MIN_PER_PROFILE = int(os.environ.get("ARC_TEST_SAMPLE_MIN_PER_PROFILE", "15"))
TEST_SAMPLE_NONEMPTY_INTENT = os.environ.get("ARC_TEST_SAMPLE_NONEMPTY_INTENT", "0") != "0"
LAST_TEST_SAMPLE_META = None   # provenance of the most recent stratified draw (freeze this)

def _round_floats(o, nd=2):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round_floats(x, nd) for x in o]
    return o

def _exact_hash(s):
    return hashlib.md5(s.encode()).hexdigest()

def _loose_hash(s):
    """Coordinate-rounded (2dp) canonical hash: catches float-noise / same-pose near-
    duplicates on top of exact. Falls back to exact on JSON parse failure. (Does NOT
    remove translation/rotation/scale near-dups — those are left as a known limitation.)"""
    try:
        return hashlib.md5(json.dumps(_round_floats(json.loads(s)), sort_keys=True).encode()).hexdigest()
    except Exception:
        return _exact_hash(s)

def _get_train_dedup_hashes(train_ds):
    global _train_dedup_hashes
    if _train_dedup_hashes is None:
        ex, lo = set(), set()
        for i in range(len(train_ds)):
            c = train_ds[i]["completion"]
            ex.add(_exact_hash(c))
            lo.add(_loose_hash(c))
        _train_dedup_hashes = (ex, lo)
        print(f"DEDUP: train hash sets built (exact={len(ex)} loose={len(lo)} over {len(train_ds)} rows)", flush=True)
    return _train_dedup_hashes

def _program_hash(sample):
    """sha256 of the TRANSPILED CadQuery program — a row's identity AT EVALUATION TIME.
    Returns None when the transpiler fails, so such rows are never silently collapsed."""
    try:
        code, _ = sample_to_cadquery_code(sample)
    except Exception:
        return None
    return hashlib.sha256(code.encode()).hexdigest()


def _dedup_within_holdout(data, idxs):
    """Collapse duplicates INSIDE the held-out split, keeping the first occurrence.

    The train-leak filter is a CROSS-split filter only; DeepCAD's published test split also
    duplicates itself. The v1 sample (first 100 clean rows) was 50 size-2 clusters, which is
    why 76 analysis entries carried only 38 independent programs.

    Identity is the TRANSPILED program, not the source JSON: the eval prompt is a prefix of
    that program and the adherence target is detect_features() of it, so two rows whose code
    is byte-identical are the same experimental unit however their JSON differs. Measured on
    the 9,442-row clean pool: completion-hash dedup alone still leaves 44 code-level clusters
    (72 redundant rows, largest size 10); program-hash dedup removes them.
    """
    seen_comp, seen_prog, kept = set(), set(), []
    n_comp_dup = n_prog_dup = 0
    for i in idxs:
        s = data[i]
        ch = _exact_hash(s["completion"])
        if ch in seen_comp:
            n_comp_dup += 1
            continue
        ph = _program_hash(s)
        if ph is not None and ph in seen_prog:
            n_prog_dup += 1
            continue
        seen_comp.add(ch)
        if ph is not None:
            seen_prog.add(ph)
        kept.append(i)
    print(f"DEDUP: held-out internal {len(idxs)} -> {len(kept)} unique programs "
          f"(removed {n_comp_dup} completion-dup + {n_prog_dup} program-dup)", flush=True)
    return kept


def _get_clean_test(ds):
    """The held-out test set with train-leaked samples removed. DeepCAD's published test
    split shares ~30% exact (~40% incl. coordinate-noise) duplicates with train; evaluating
    on those inflates validity. Filter test to samples whose exact AND loose hash are both
    absent from the full train set, then collapse duplicates within the survivors so every
    remaining row is one independent program (see _dedup_within_holdout)."""
    global _clean_test_cache
    if _clean_test_cache is None:
        if "test" in ds:
            data = ds["test"]
        elif "validation" in ds:
            data = ds["validation"]
        else:
            full = ds["train"]
            data = full.select(range(int(len(full) * 0.9), len(full)))
        ex, lo = _get_train_dedup_hashes(ds["train"])
        clean = [i for i in range(len(data))
                 if _exact_hash(data[i]["completion"]) not in ex
                 and _loose_hash(data[i]["completion"]) not in lo]
        n_before = len(data)
        n_leak_removed = n_before - len(clean)
        if HOLDOUT_INTERNAL_DEDUP:
            clean = _dedup_within_holdout(data, clean)
        _clean_test_cache = data.select(clean)
        print(f"DEDUP: test {n_before} -> {len(_clean_test_cache)} held-out "
              f"(removed {n_leak_removed} train-leaked)", flush=True)
    return _clean_test_cache

def load_deepcad(split="train", max_rows=None):
    key = (split, max_rows)
    if key in _dataset_cache:
        return _dataset_cache[key]

    cache_dir = str(WORKSPACE)
    try:
        ds = hf_load_dataset(
            "wanhin/deepcad-completion-sft",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"wanhin/deepcad-completion-sft unavailable: {e}")

    if split == "train":
        data = ds["train"]
    elif split in ("test", "validation"):
        data = _get_clean_test(ds)   # train-disjoint held-out set (no leakage)
    else:
        data = ds[split]

    if max_rows is not None and len(data) > max_rows:
        # Held-out subsets are drawn stratified by design-intent profile; train subsets stay
        # head-ordered (they are only used for calibration/warm-up, never for a comparison).
        if split in ("test", "validation") and TEST_SAMPLING == "stratified":
            data = stratified_test_sample(data, max_rows)
        else:
            data = data.select(range(max_rows))

    _dataset_cache[key] = data
    return data


def sample_to_cadquery_code(sample):
    """Returns (cadquery_code_str, complexity_label)."""
    code = extract_text_from_sample(sample)
    # Estimate operation count from code lines for complexity regime
    op_count = code.count(".box(") + code.count(".cylinder(") + code.count(".cut(") + code.count(".sphere(")
    complexity = "simple" if op_count <= 4 else "complex"
    return code, complexity


# ═══════════════════════════════════════════════════════════════════════════════
# CADQUERY ORACLE
# ═══════════════════════════════════════════════════════════════════════════════

_CQ_SELF_TEST_DONE = False

def run_cadquery_oracle(code_str: str, timeout: int = None) -> bool:
    """Execute CadQuery code in a subprocess. Returns True iff valid solid."""
    if timeout is None:
        timeout = HYPERPARAMETERS["cadquery_timeout"]

    # Wrap code to emit VALID/INVALID signal
    wrapper = textwrap.dedent(f"""
import cadquery as cq
import sys
try:
{textwrap.indent(code_str, '    ')}
    # Check result exists and is a valid solid
    if 'result' not in dir():
        print('INVALID'); sys.exit(0)
    if hasattr(result, 'val'):
        solid = result.val()
        if hasattr(solid, 'isValid') and not solid.isValid():
            print('INVALID'); sys.exit(0)
        if hasattr(result, 'solids'):
            solids = result.solids().vals()
            if not solids:
                print('INVALID'); sys.exit(0)
            bb = result.val().BoundingBox()
            if bb.xlen <= 0 or bb.ylen <= 0 or bb.zlen <= 0:
                print('INVALID'); sys.exit(0)
    print('VALID')
except Exception as e:
    print('INVALID')
    sys.exit(0)
""")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout.strip() == "VALID"
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def oracle_self_test():
    global _CQ_SELF_TEST_DONE
    if _CQ_SELF_TEST_DONE:
        return
    test_code = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
    # Retry: the very first cadquery/OCP subprocess import can be cold and exceed the
    # timeout transiently; a true broken oracle fails every attempt. Don't abort the
    # whole run on a one-off cold-start flake.
    ok = False
    for _attempt in range(3):
        ok = run_cadquery_oracle(test_code, timeout=60)
        if ok:
            break
        print(f"oracle self-test attempt {_attempt + 1}/3 did not score VALID; "
              "retrying (cold cadquery/OCP import?)")
    if not ok:
        print("ORACLE_SELF_TEST_FAILED: trivial box did not score VALID — oracle is broken")
        raise RuntimeError("Oracle self-test failed")
    print("ORACLE_SELF_TEST_PASSED")
    _CQ_SELF_TEST_DONE = True


def oracle_calibration(codes: list, label: str = "ground_truth"):
    """Run a list of cadquery code strings through oracle. Print calibration line."""
    if not codes:
        print(f"ORACLE_CALIBRATION: {label}_validity=N/A (empty)")
        return 0.0
    valid_count = 0
    for c in codes:
        if run_cadquery_oracle(c):
            valid_count += 1
    rate = valid_count / len(codes)
    print(f"ORACLE_CALIBRATION: {label}_validity={rate:.4f} ({valid_count}/{len(codes)})")
    update_progress(calibration={label: rate})
    return rate


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT TOKEN AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

CONSTRAINT_TOKENS = ["<CIRCLE>", "<NGON>", "<THIN>", "<TALL>", "<MULTI_PART>"]
CONSTRAINT_DELIMS = ("<CONSTRAINTS>", "</CONSTRAINTS>")

# Natural-language meaning of each added token, used to WARM-START its embedding from the
# mean of the pretrained subword embeddings of these words. Without this the new rows are
# random-init and the token arm cold-starts while the text arm reuses pretrained "cut"/"hole"
# semantics — a confound that could invert/explain H2a (token vs text) as a data-efficiency
# artifact rather than a representation effect.
TOKEN_MEANING = {
    "<CIRCLE>": "circle", "<NGON>": "polygon", "<THIN>": "thin",
    "<TALL>": "tall", "<MULTI_PART>": "multiple parts",
    "<CONSTRAINTS>": "design intent", "</CONSTRAINTS>": "design intent",
}


def detect_features(code: str) -> set:
    """Code-level detector for the 5 transpiler-faithful design-intent features. Returns the
    subset of CONSTRAINT_TOKENS present. Used for BOTH header derivation (on GT code) and the
    adherence metric (on generated code) — the SAME detector both sides, so the comparison is
    fair. (HOLE/CUT/ARC/CONCENTRIC/etc. are dropped: the transpiler simplifies them away — see
    the transpiler-simplification note; making them real is the next-version transpiler fix.)"""
    f = set()
    if ".circle(" in code:
        f.add("<CIRCLE>")
    m = re.search(r"\.moveTo\(.*?\.close\(\)", code, re.S)
    if m and m.group(0).count(".lineTo(") >= 5:
        f.add("<NGON>")
    if code.count(".extrude(") > 1 or code.count("result = (") > 1:
        f.add("<MULTI_PART>")
    dep = re.search(r"\.extrude\(([-\d.]+)\)", code)
    coords = re.findall(r"\.(?:moveTo|lineTo)\(([-\d.]+),\s*([-\d.]+)\)", code)
    radii = [abs(float(r)) for r in re.findall(r"\.circle\(([-\d.]+)\)", code)]
    if dep and (coords or radii):
        d = abs(float(dep.group(1)))
        line_ext = 0.0
        if coords:
            xs = [float(x) for x, y in coords]
            ys = [float(y) for x, y in coords]
            line_ext = max(max(xs) - min(xs), max(ys) - min(ys))
        circ_ext = 2 * max(radii) if radii else 0.0   # circle footprint = diameter
        ext = max(line_ext, circ_ext)
        if ext > 1e-9:
            ratio = d / ext
            if ratio < 0.3:
                f.add("<THIN>")
            elif ratio > 2.0:
                f.add("<TALL>")
    return f


def derive_constraints_from_sample(sample) -> list:
    """Derive the design-intent header from what's ACTUALLY in the GT CadQuery code (via
    detect_features), NOT the nominal JSON — the transpiler simplifies the JSON away (drops
    inner-loop holes, extra faces, renders cuts as zero-depth no-ops and arcs as lines), so
    JSON-derived tags would condition on features absent from the training target (phantoms).
    Returns the subset of CONSTRAINT_TOKENS present, in CONSTRAINT_TOKENS order."""
    try:
        code, _ = sample_to_cadquery_code(sample)
    except Exception:
        return []
    found = detect_features(code)
    return [t for t in CONSTRAINT_TOKENS if t in found]


def build_constraint_header(constraints: list, mode: str) -> str:
    """Build the design-intent conditioning prefix to prepend to a program.
    mode: "none" (no header) / "text" (plain lowercase words, subword-tokenized) /
    "token" (dedicated design-intent tokens). Returns header WITH trailing newline,
    or "" when no header (none mode or no tags)."""
    if mode == "none" or not constraints:
        return ""
    if mode == "token":
        body = " ".join(constraints)
        return f"{CONSTRAINT_DELIMS[0]} {body} {CONSTRAINT_DELIMS[1]}\n"
    if mode == "text":
        words = " ".join(c.strip("<>").lower() for c in constraints)
        return f"# design intent: {words}\n"
    raise ValueError(f"unknown constraint mode: {mode}")


def _corrupt_constraints(correct, rng) -> list:
    """Experiment A (corrupted header): flip 1-2 of the 5 design-intent tags (toggle
    present<->absent). Toggling >=1 distinct tag guarantees the result differs from `correct`.
    Returns tags in CONSTRAINT_TOKENS order."""
    out = set(correct)
    for t in rng.sample(CONSTRAINT_TOKENS, rng.choice((1, 2))):
        out.discard(t) if t in out else out.add(t)
    return [t for t in CONSTRAINT_TOKENS if t in out]


# ── F4 (causal control) helpers ──────────────────────────────────────────────
# TAGS_PRIMARY = the 4 tags the geometry metric actually scores (MULTI_PART excluded,
# geom-underdetermined). F4b corruption toggles ONLY these so every corruption is
# visible in the measured space (adversarial-review finding #5).
_TAGS_PRIMARY_MAIN = ["<CIRCLE>", "<NGON>", "<THIN>", "<TALL>"]


def _corrupt_primary(correct, rng) -> list:
    """F4b: like _corrupt_constraints but toggles 1-2 of the 4 MEASURED tags only, so the
    corruption is never invisible in the 4-tag geometry metric. MULTI_PART left untouched."""
    out = set(correct)
    for t in rng.sample(_TAGS_PRIMARY_MAIN, rng.choice((1, 2))):
        out.discard(t) if t in out else out.add(t)
    return [t for t in CONSTRAINT_TOKENS if t in out]


def _derangement_plan(cons_all, rng) -> list:
    """F4 shuffled-GT TRAINING control: an exact PERMUTATION (derangement, no index fixed
    point) of the training headers. Unlike _build_shuffle_plan (with-replacement, used for
    eval), this keeps the assigned-header multiset BYTE-IDENTICAL to the original (it is a
    permutation), so R's training-header MARGINAL == M's exactly and only the program<->header
    CORRELATION is destroyed. Returns [cons_all[sigma(i)] for i]."""
    n = len(cons_all)
    idx = list(range(n))
    rng.shuffle(idx)
    # Fix any fixed points (sigma(i)==i) by swapping with a neighbor — yields a derangement.
    for i in range(n):
        if idx[i] == i:
            j = (i + 1) % n
            idx[i], idx[j] = idx[j], idx[i]
    # One more pass in case the last swap reintroduced a fixed point at the wrap.
    for i in range(n):
        if idx[i] == i:
            j = i - 1 if i > 0 else n - 1
            idx[i], idx[j] = idx[j], idx[i]
    return [cons_all[idx[i]] for i in range(n)]


def _random_constraints_sizematched(correct, rng) -> list:
    """F4b (header_source='random'): a well-formed header with RANDOM content, SIZE-MATCHED to
    `correct` (draw exactly |correct| tags), non-empty iff correct is, and != correct.
    Size-matching keeps header length in-distribution so 'random' isolates content, not length."""
    k = len(correct)
    if k == 0:
        return []
    pool = list(CONSTRAINT_TOKENS)
    correct_set = frozenset(correct)
    for _ in range(64):
        pick = frozenset(rng.sample(pool, k))
        if pick != correct_set:
            return [t for t in CONSTRAINT_TOKENS if t in pick]
    # Degenerate fallback (k==5, only one possible set): toggle one tag off.
    out = set(correct); out.discard(rng.choice(list(out)))
    return [t for t in CONSTRAINT_TOKENS if t in out]


def _build_shuffle_plan(correct_all, rng) -> list:
    """Experiment A (shuffled header): assign each sample i ANOTHER sample j's constraints — a
    well-formed header with WRONG content. Prefers a j whose constraint-set differs from i's
    (genuine wrong content); falls back to any j!=i. Deterministic given rng. Returns a list of
    constraint-lists, one per sample index."""
    n = len(correct_all)
    sets = [frozenset(c) for c in correct_all]
    plan = []
    for i in range(n):
        cands = [j for j in range(n) if j != i and sets[j] != sets[i]]
        if not cands:
            cands = [j for j in range(n) if j != i] or [i]
        plan.append(list(correct_all[rng.choice(cands)]))
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# HELD-OUT SAMPLING (stratified by design-intent profile)
# ═══════════════════════════════════════════════════════════════════════════════
# Taking the first N clean rows makes the evaluated profile mix an accident of the file
# order. In the v1 sample of 100 that accident gave 2 TALL programs and zero CIRCLE+TALL,
# while the clean pool holds 142 and 112 respectively — the "TALL hits the generation floor"
# scope restriction rests on those 2 programs. Stratified drawing removes that failure mode.


def profile_of_sample(sample) -> tuple:
    """A sample's 4-tag design-intent profile, in CONSTRAINT_TOKENS order.

    MULTI_PART is excluded on purpose: overlapping extrudes fuse into a single solid, so the
    tag is geometrically unmeasurable and carries no weight in the scored metric — letting it
    split the strata would only fragment them (11 profiles become 20) without buying coverage
    of anything the analysis can see."""
    try:
        code, _ = sample_to_cadquery_code(sample)
    except Exception:
        return ()
    found = detect_features(code)
    return tuple(t for t in _TAGS_PRIMARY_MAIN if t in found)


def allocate_stratified_quota(pool_sizes: dict, n: int, min_per_profile: int = 0) -> dict:
    """Split n slots across strata: rare strata are lifted to a floor, the rest keep a COMMON
    sampling rate.

    Any stratum whose proportional share falls under the floor is locked at the floor (or at
    its whole pool, if smaller), and any stratum too small to absorb its share is capped; what
    is left is then re-spread over the strata still free, and the lock/cap test repeats until
    nothing new locks. That way the floor distorts only the strata it rescues — the large ones
    stay proportional to each other, which a single floor-then-distribute pass does not give
    (it silently taxes the large strata to pay for the floor). Largest-remainder rounding at
    the end; ties broken deterministically. The floor is dropped wholesale when it cannot fit
    in n, so tiny n (the max_rows=5 spot checks in this file) degrades to plain proportional
    allocation instead of raising."""
    keys = sorted(pool_sizes)
    n = min(n, sum(pool_sizes.values()))
    floors = {k: min(min_per_profile, pool_sizes[k]) for k in keys}
    if sum(floors.values()) > n:
        floors = {k: 0 for k in keys}

    quota, remaining = {}, n
    free = [k for k in keys if pool_sizes[k] > 0]
    while free and remaining > 0:
        pool_free = sum(pool_sizes[k] for k in free)
        exact = {k: remaining * pool_sizes[k] / pool_free for k in free}
        locked = {k: (pool_sizes[k] if exact[k] > pool_sizes[k] else floors[k])
                  for k in free if exact[k] > pool_sizes[k] or exact[k] < floors[k]}
        if not locked:
            break
        for k, v in locked.items():
            quota[k] = v
            remaining -= v
            free.remove(k)

    if free and remaining > 0:
        pool_free = sum(pool_sizes[k] for k in free)
        exact = {k: remaining * pool_sizes[k] / pool_free for k in free}
        base = {k: min(int(exact[k]), pool_sizes[k]) for k in free}
        leftover = remaining - sum(base.values())
        for k in sorted(free, key=lambda k: (-(exact[k] % 1), -pool_sizes[k], k)):
            if leftover <= 0:
                break
            if base[k] < pool_sizes[k]:
                base[k] += 1
                leftover -= 1
        quota.update(base)

    for k in keys:
        quota.setdefault(k, 0)
    return quota


def stratified_test_sample(data, n: int, seed: int = None, min_per_profile: int = None,
                           nonempty_intent: bool = None):
    """Draw n held-out rows stratified by 4-tag profile. Deterministic given (data, n, seed).

    Records provenance (quota, per-profile draw, program hashes) in LAST_TEST_SAMPLE_META so
    the drawn set can be frozen and audited before any model is run."""
    global LAST_TEST_SAMPLE_META
    seed = TEST_SAMPLE_SEED if seed is None else seed
    min_per_profile = TEST_SAMPLE_MIN_PER_PROFILE if min_per_profile is None else min_per_profile
    nonempty_intent = TEST_SAMPLE_NONEMPTY_INTENT if nonempty_intent is None else nonempty_intent

    by_profile = defaultdict(list)
    n_empty_skipped = 0
    for i in range(len(data)):
        p = profile_of_sample(data[i])
        if nonempty_intent and not p:
            n_empty_skipped += 1
            continue
        by_profile[p].append(i)

    pool_sizes = {p: len(v) for p, v in by_profile.items()}
    quota = allocate_stratified_quota(pool_sizes, n, min_per_profile)

    rng = random.Random(seed)
    picked = []
    drawn = {}
    for p in sorted(by_profile):                      # sorted => draw order is deterministic
        pool, k = by_profile[p], quota.get(p, 0)
        take = sorted(pool) if k >= len(pool) else sorted(rng.sample(sorted(pool), k))
        drawn["+".join(t.strip("<>") for t in p) or "EMPTY"] = {
            "pool": len(pool), "quota": k, "drawn": len(take)}
        picked.extend(take)
    picked.sort()

    sample = data.select(picked)
    LAST_TEST_SAMPLE_META = {
        "policy": "stratified_by_4tag_profile",
        "n_requested": n, "n_drawn": len(picked),
        "seed": seed, "min_per_profile": min_per_profile,
        "nonempty_intent_only": bool(nonempty_intent),
        "n_pool_rows": len(data), "n_empty_intent_skipped": n_empty_skipped,
        "source_indices": picked,
        "per_profile": drawn,
        "program_sha256": [_program_hash(sample[j]) for j in range(len(sample))],
    }
    print(f"SAMPLE: stratified held-out draw n={len(picked)}/{n} from {len(data)} rows, "
          f"seed={seed} floor={min_per_profile} profiles={len(by_profile)}"
          f"{' (non-empty intent only)' if nonempty_intent else ''}", flush=True)
    for name, d in sorted(drawn.items(), key=lambda kv: -kv[1]["drawn"]):
        print(f"  {name:24s} pool={d['pool']:5d} drawn={d['drawn']:4d}", flush=True)
    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# LLM / TOKENIZER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


def decide_placement(est_peak_gb: float, is_training: bool = False):
    """Pick single-GPU (fastest: no naive-MP / cross-GPU transfer) if the task's
    PEAK VRAM fits one card, else shard across all visible GPUs (device_map="auto").

    Decision is purely memory-based and applies to BOTH training and inference:
    a large-batch / long-sequence eval that needs >24GB shards too.

    Returns (load_kwargs, single_gpu). When single_gpu AND is_training, the caller
    MUST set `trainer.args._n_gpu = 1` so HF Trainer does not wrap the single-GPU
    model in DataParallel (the "chunk expects >=1-dim tensor" crash) when more than
    one GPU is visible. Sharded models are model-parallel, so no DataParallel there.
    """
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9 if torch.cuda.is_available() else 0.0
    if n_gpu <= 1 or est_peak_gb <= free_gb * 0.85:
        print(f"PLACEMENT: single_gpu est_peak={est_peak_gb:.1f}GB free={free_gb:.1f}GB n_gpu={n_gpu}")
        return {"device_map": {"": 0}}, True
    print(f"PLACEMENT: sharded(auto) est_peak={est_peak_gb:.1f}GB free={free_gb:.1f}GB n_gpu={n_gpu}")
    return {"device_map": "auto"}, False


def build_grpo_balanced_device_map(model_name: str, lm_head_gpu: int = 1,
                                   layers_gpu: int = 0, lm_head_gpu_layers: int = 8):
    """Hand-written, peak-balanced device_map for the GRPO load on >=2 GPUs.

    Root cause of the GRPO GPU0 OOM: device_map="auto" co-locates the TIED
    embed_tokens+lm_head (vocab=151936) on GPU0, so the full-vocab policy + reference
    logits + the rollout KV cache all stack on GPU0 (~22GB) -> OOM at steps 7/11/23.
    tie_word_embeddings=True => embed_tokens and lm_head SHARE one physical tensor and
    MUST sit on the same device.

    Strategy (empirically tuned on Qwen2.5-Coder-1.5B / 2x24GB): put the tied
    embed+lm_head + final norm + rotary on the otherwise-idle GPU (lm_head_gpu) together
    with the LAST `lm_head_gpu_layers` transformer layers, and the remaining (earlier)
    layers on layers_gpu. With 28 layers and lm_head_gpu_layers=8 this is the measured
    sweet spot: GPU0 (layers 0-19) ~18.0GB, GPU1 (layers 20-27 + head) ~16.7GB (19.7GB
    incl the resident fp16 reward-ref pinned here too) — both well under 24GB, survived
    30 GRPO steps no-OOM. Reads the real layer count from config (NOT hardcoded to 28).

    Returns a COMPLETE device_map dict (accelerate rejects an incomplete map).
    """
    n_layers = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True).num_hidden_layers
    k = max(1, min(n_layers - 1, lm_head_gpu_layers))  # layers co-located with lm_head
    split = n_layers - k                               # layers_gpu gets [0, split)
    device_map = {
        "model.embed_tokens": lm_head_gpu,
        "lm_head": lm_head_gpu,
        "model.norm": lm_head_gpu,
        "model.rotary_emb": lm_head_gpu,
    }
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = layers_gpu if i < split else lm_head_gpu
    return device_map


def estimate_peak_gb(kind: str) -> float:
    """Peak-VRAM estimate per task for decide_placement. Anchored to MEASURED peaks
    on Qwen2.5-Coder-1.5B / 2x3090 (a clean physical model badly under-counts GRPO's
    full-batch generation KV + coexisting fp16 ref model + fragmentation), and scaled
    by the cost knobs so it stays adaptive if max_completion_length changes.

    Anchors: SFT/eval fit one 24GB card comfortably; GRPO MUST SHARD at completion>=512 —
    its SINGLE-GPU peak is ~23GB @ completion=768 (measured: OOMs a 24GB card mid-step). The
    ~16GB you see while it runs is the SHARDED per-card HALF, not the single-GPU requirement."""
    H = HYPERPARAMETERS
    if kind == "sft":    # 4-bit QLoRA + grad-checkpointing keeps it small
        return 6.0 + 0.003 * H["sft_batch_size"] * H["max_seq_length"]
    if kind == "eval":   # fp16 weights (3GB) + batched-generation KV cache
        return 4.0 + 0.002 * H.get("eval_batch_size", 8) * H["max_completion_length"]
    if kind == "grpo":
        # Single-GPU peak measured ~23GB @ completion=768 (OOMs a 24GB card) → MUST shard at
        # completion>=512. A 2026-06-18 re-anchor to ~16GB (the sharded per-card half) wrongly
        # chose single-GPU and OOM'd mid-step. Estimate the SINGLE-GPU requirement so the >24GB
        # ones correctly shard; here ~26GB@768 / ~28GB@1024.
        return 18.0 + 0.010 * H["max_completion_length"]
    return 4.0


def load_qwen_4bit(seed: int = 0):
    """Load Qwen2.5-Coder-1.5B in 4-bit QLoRA mode for training."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # ~1.5B params * 0.5 bytes/param (4-bit) * 1.4 (optimizer overhead) ≈ 1.05 GB
    placement, _single = decide_placement(estimate_peak_gb("sft"), is_training=True)
    model = AutoModelForCausalLM.from_pretrained(
        HYPERPARAMETERS["base_model"],
        quantization_config=bnb_config,
        **placement,
        torch_dtype=torch.bfloat16,
    )
    model._arc_single_gpu = _single  # SFTTrainer must set _n_gpu=1 if single-GPU
    tokenizer = AutoTokenizer.from_pretrained(
        HYPERPARAMETERS["base_model"],
        trust_remote_code=True,
    )
    # pad MUST differ from eos for SFT: TRL's DataCollatorForLanguageModeling masks
    # labels by VALUE (labels[labels == pad_id] = -100), so pad == eos masks the real
    # trailing EOS too -> the model never gets gradient on EOS -> never learns to stop
    # (eos_frac=0, generations run to the cap, validity ~0). Qwen defaults
    # pad == eos == <|endoftext|>, so force a DISTINCT existing special token (no resize).
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = "<|fim_pad|>"   # id 151662: in-vocab, != eos, absent from code
    return model, tokenizer


def load_qwen_fp16_for_eval(adapter_path: str = None, seed: int = 0, expand_vocab: bool = False,
                            ref_device: int = None):
    """Load Qwen2.5-Coder-1.5B in bf16 for fast inference/eval.
    expand_vocab=True (for token-trained checkpoints): add the constraint vocab tokens and
    resize the base BEFORE attaching the adapter, so the adapter's resized modules_to_save
    (embed_tokens/lm_head) match — otherwise PeftModel.from_pretrained dimension-mismatches."""
    if ref_device is not None and torch.cuda.is_available() and \
            torch.cuda.device_count() > ref_device:
        # GRPO OOM fix (edit B): pin this RESIDENT reward-ref model to a specific GPU.
        # The GRPO balanced map parks the policy's embed+lm_head on GPU1; placing this
        # ~3GB fp16 ref on the SAME idle GPU keeps GPU0 free for the policy's layer
        # activations (measured: GPU0 18.0 / GPU1 19.7GB). Eval callers pass
        # ref_device=None and keep the original memory-based placement unchanged.
        placement = {"device_map": {"": ref_device}}
    else:
        placement, _ = decide_placement(estimate_peak_gb("eval"), is_training=False)
    model = AutoModelForCausalLM.from_pretrained(
        HYPERPARAMETERS["base_model"],
        torch_dtype=torch.bfloat16,
        **placement,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        HYPERPARAMETERS["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if expand_vocab:
        expand_constraint_vocab(model, tokenizer)  # resize base to match the adapter
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def make_lora_config(target_modules=None):
    target_modules = None
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=HYPERPARAMETERS["lora_r"],
        lora_alpha=HYPERPARAMETERS["lora_alpha"],
        lora_dropout=HYPERPARAMETERS["lora_dropout"],
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        # Save the (resized) embed_tokens + lm_head so newly-added constraint-token
        # embeddings persist in the adapter. freeze_old_embedding_rows() then zeros the
        # gradient of all PRE-EXISTING rows, so only the new rows actually train.
        modules_to_save=["embed_tokens", "lm_head"],
    )


def _warmstart_token_embeddings(model, tokenizer, new_tokens) -> int:
    """Seed each new token's input+output embedding row with the MEAN of the pretrained
    subword embeddings of its natural-language meaning (TOKEN_MEANING), e.g. <CUT> <- mean
    of the embeddings of the subwords of "cut". This gives the token arm the SAME word
    semantics the text arm gets for free, so the token-vs-text H2 comparison isolates
    representation (dedicated token vs subword sequence) rather than init/data-efficiency."""
    import torch
    w_in = model.get_input_embeddings().weight
    out = model.get_output_embeddings()
    w_out = out.weight if out is not None else None
    seeded = 0
    with torch.no_grad():
        for tok in new_tokens:
            tid = tokenizer.convert_tokens_to_ids(tok)
            word = TOKEN_MEANING.get(tok)
            if word is None:
                continue
            sub_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if not sub_ids:
                continue
            idx = torch.tensor(sub_ids, device=w_in.device)
            try:
                w_in[tid] = w_in[idx].to(torch.float32).mean(dim=0).to(w_in.dtype)
                if w_out is not None and w_out is not w_in:
                    w_out[tid] = w_out[idx].to(torch.float32).mean(dim=0).to(w_out.dtype)
                seeded += 1
            except Exception as e:
                print(f"WARMSTART_SKIP: {tok} ({type(e).__name__}: {e})")
    print(f"WARMSTART: seeded {seeded}/{len(new_tokens)} new token rows from word-meaning subword means")
    return seeded


def expand_constraint_vocab(model, tokenizer) -> int:
    """Add the design-intent tokens + 2 delimiters as real vocab tokens, resize the model's
    embedding, and WARM-START each new row from its word meaning (removes the cold-start
    confound vs the plain-text arm). Returns the new vocab size."""
    new = list(CONSTRAINT_TOKENS) + list(CONSTRAINT_DELIMS)
    n_added = tokenizer.add_tokens(new, special_tokens=True)
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        _warmstart_token_embeddings(model, tokenizer, new)
    print(f"VOCAB_EXPAND: added {n_added} constraint tokens, vocab={len(tokenizer)}")
    return len(tokenizer)


def freeze_old_embedding_rows(peft_model, old_size: int) -> None:
    """Gradient hook: zero the grad of embed_tokens/lm_head rows < old_size so ONLY the
    newly-added constraint-token rows train (existing token embeddings unchanged → no
    forgetting). Call AFTER get_peft_model (hook goes on the trainable modules_to_save copy)."""
    def _mask_hook(grad):
        g = grad.clone()
        g[:old_size].zero_()
        return g
    n_hooked = 0
    for n, p in peft_model.named_parameters():
        if p.requires_grad and p.dim() == 2 and ("embed_tokens" in n or "lm_head" in n):
            p.register_hook(_mask_hook)
            n_hooked += 1
            print(f"FREEZE_OLD_ROWS: hooked {n} (rows <{old_size} grad=0)")
    if n_hooked == 0:
        print("FREEZE_OLD_ROWS: WARNING no trainable embed/lm_head found to hook")


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

_gen_calib_done = False
_calibrated_max_completion = None

def calibrate_generation(tokenizer, data_sample, n_samples: int = 20):
    """Measure actual completion lengths and set max_new_tokens."""
    global _gen_calib_done, _calibrated_max_completion

    if _gen_calib_done:
        return _calibrated_max_completion

    if SMOKE:
        _calibrated_max_completion = 128
        _gen_calib_done = True
        print(f"GENERATION_CALIBRATION: smoke_mode max_new_tokens=128")
        return _calibrated_max_completion

    # Base the cap on the COMPLETION the model actually generates at eval (the program
    # tail AFTER the eval_prefix_fraction prompt), NOT the full program. Eval feeds a
    # ~40% prefix and the model completes only the rest, so a full-program p95 (≈563 tok
    # measured → cap 732) over-allocates ~2.5×. With batched greedy generation every batch
    # runs to its longest member, so the inflated cap slowed every batch holding a non-EOS
    # sample. Completion p95 ≈ 230-300 tok → cap ≈ 400 (verified output-identical, 2.2× faster).
    frac = HYPERPARAMETERS["eval_prefix_fraction"]
    lengths = []
    for s in data_sample[:n_samples]:
        code, _ = sample_to_cadquery_code(s)
        split_idx = max(1, min(int(len(code) * frac), len(code) - 1))
        completion = code[split_idx:]
        toks = tokenizer.encode(completion, add_special_tokens=False)
        lengths.append(len(toks))

    if not lengths:
        _calibrated_max_completion = HYPERPARAMETERS["max_completion_length"]
        _gen_calib_done = True
        return _calibrated_max_completion

    p95 = int(np.percentile(lengths, 95))
    max_new = int(math.ceil(p95 * 1.3)) + 16   # +16: EOS token + tokenizer rounding headroom
    max_new = min(max_new, HYPERPARAMETERS["max_completion_length"])
    max_new = max(max_new, 64)

    _calibrated_max_completion = max_new
    _gen_calib_done = True
    print(f"GENERATION_CALIBRATION: p95_tokens={p95} max_new_tokens={max_new}")
    update_progress(max_new_tokens=max_new)
    return max_new


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM BOUNDARY EXTRACTOR (extract valid CadQuery prefix even without EOS)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_valid_program_prefix(code: str) -> str:
    """Return the FIRST complete program. Take the longest leading prefix that
    COMPILES, then trim to end at the FIRST top-level `result = ...` assignment.

    Rationale: the SFT model learns to emit a valid program but (lacking a learned
    EOS) does not stop — it loops, appending degenerate re-assignments
    (`result = (result...extrude(0.000000))`) that are syntactically valid but make
    the WHOLE thing fail to EXECUTE (Standard_ConstructionError). The old
    longest-compiling prefix kept those loops -> validity 0. The first result block
    is the real, valid program (verified isValid=True)."""
    if not code.strip():
        return code
    # Strip a constraint-TOKEN header ("<CONSTRAINTS> <CUT> ... </CONSTRAINTS>") if present:
    # it is NOT valid Python, so every compiling-prefix attempt below would fail and the
    # whole non-executable string would be returned -> the constraint-TOKEN arm would score
    # a FALSE ZERO validity (silently inverting the token-vs-text H2 comparison). The
    # text-mode "# design intent:" header is a Python comment and needs no stripping.
    _close = CONSTRAINT_DELIMS[1]
    if _close in code:
        code = code.split(_close, 1)[1].lstrip("\n")
    lines = code.split("\n")
    best = ""
    accumulated = ""
    for line in lines:
        accumulated = accumulated + line + "\n"
        try:
            compile(accumulated, "<string>", "exec")
            best = accumulated
        except SyntaxError:
            pass
    if not best:
        return code
    try:
        import ast as _ast
        tree = _ast.parse(best)
        for node in tree.body:
            if isinstance(node, _ast.Assign) and any(
                    isinstance(t, _ast.Name) and t.id == "result" for t in node.targets):
                return "\n".join(best.split("\n")[:node.end_lineno]) + "\n"
    except Exception:
        pass
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION + VALIDITY EVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_batch(model, tokenizer, prompts: list, max_new_tokens: int) -> list:
    """Batched generation. Returns list of generated code strings."""
    if not prompts:
        return []

    device = next(model.parameters()).device
    model.eval()

    # Tokenize with left-padding for batch generation
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=HYPERPARAMETERS["max_seq_length"],
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    # Decode only the new tokens
    prompt_len = input_ids.shape[1]
    generated_ids = out[:, prompt_len:]

    results = []
    eos_count = 0
    truncated_count = 0
    for i, ids in enumerate(generated_ids):
        # Check if EOS was generated
        has_eos = tokenizer.eos_token_id in ids.tolist()
        if has_eos:
            eos_count += 1
            # Trim after first EOS
            eos_pos = ids.tolist().index(tokenizer.eos_token_id)
            ids = ids[:eos_pos]
        else:
            truncated_count += 1

        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # CRITICAL: prepend the prompt prefix. The 30%-prefix prompt holds the
        # program HEAD (`import cadquery as cq`, `result = cq.Workplane(...)`);
        # the model only emits the continuation. Extracting from the completion
        # alone yields a headless fragment with no leading `result = ...` -> the
        # boundary extractor finds no complete program -> validity 0. The full
        # program is prompt + completion. (Must match the CANARY probe, which
        # also extracts on prompt+completion.)
        full = prompts[i] + decoded
        prog = extract_valid_program_prefix(full)
        results.append(prog)

    n = len(generated_ids)
    print(f"GENERATION_CALIBRATION: truncated_frac={truncated_count/n:.3f} eos_frac={eos_count/n:.3f}")
    return results


def adherence_scores(intended: set, produced: set):
    """Recall = |produced ∩ intended| / |intended| (of the design intent, how much was
    realized); Precision = |produced ∩ intended| / |produced|. Returns (recall, precision);
    a score is None when its denominator set is empty (excluded from the mean)."""
    inter = len(intended & produced)
    recall = inter / len(intended) if intended else None
    precision = inter / len(produced) if produced else None
    return recall, precision


def eval_validity(model, tokenizer, test_data, max_new_tokens: int,
                  condition_name: str, seed: int,
                  constraint_mode: str = "none", header_source: str = "correct") -> dict:
    """Evaluate validity rate over test_data. Returns metrics dict.
    constraint_mode (none/text/token): prepends a constraint header to each eval prompt
    (constraint-conditioned generation). masked arm passes "none" here even though its
    checkpoint was token-trained → model sees no header.
    header_source (Experiment A): content of the header when constraint_mode != "none":
    "correct" = GT-derived (reproduces V1); "shuffled" = another sample's constraints
    (well-formed, wrong content); "corrupted" = 1-2 GT tags flipped. Scoring (intended =
    detect_features(GT)) is UNCHANGED — it always measures adherence to the TRUE design intent."""
    n = len(test_data)
    print(f"[eval {condition_name} seed={seed}] evaluating {n} samples...")

    results_by_regime = {"simple": {"valid": 0, "total": 0},
                         "complex": {"valid": 0, "total": 0}}
    all_valid = 0
    adh_recalls = []      # adherence: per-sample recall (when intended features non-empty)
    adh_precisions = []   # adherence: per-sample precision (when produced features non-empty)

    BATCH_SIZE = HYPERPARAMETERS.get("eval_batch_size", 8) if not SMOKE else 2
    t0 = time.time()
    step_times = []

    # Experiment A: when header_source is shuffled/corrupted, inject a WRONG-content (but
    # well-formed) header instead of the GT-derived one. "correct" leaves the V1 path untouched.
    header_plan = None
    if header_source not in ("correct", "shuffled", "corrupted", "corrupted_primary", "random"):
        raise ValueError(f"unknown header_source: {header_source}")
    if header_source in ("shuffled", "corrupted", "corrupted_primary", "random") and constraint_mode != "none":
        _rng = random.Random(seed)
        _correct_all = [derive_constraints_from_sample(test_data[i]) for i in range(n)]
        if header_source == "shuffled":
            header_plan = _build_shuffle_plan(_correct_all, _rng)
        elif header_source == "corrupted":
            header_plan = [_corrupt_constraints(c, _rng) for c in _correct_all]
        elif header_source == "corrupted_primary":
            header_plan = [_corrupt_primary(c, _rng) for c in _correct_all]
        else:  # random (F4b): well-formed, size-matched, random content
            header_plan = [_random_constraints_sizematched(c, _rng) for c in _correct_all]
        _diff = sum(1 for i in range(n) if frozenset(header_plan[i]) != frozenset(_correct_all[i]))
        print(f"HEADER_SOURCE: {header_source} differs-from-correct on {_diff}/{n} samples", flush=True)

    for batch_start in range(0, n, BATCH_SIZE):
        if budget_exceeded():
            print(f"[eval {condition_name}] budget exceeded at {batch_start}/{n}")
            break

        batch = test_data[batch_start:batch_start + BATCH_SIZE]
        if hasattr(batch, "__getitem__"):
            # HuggingFace dataset slice returns dict
            samples = [dict(zip(batch.keys(), vals))
                       for vals in zip(*batch.values())]
        else:
            samples = batch

        prompts = []
        complexities = []
        gt_codes = []   # full GT code per sample → its design-intent features (adherence target)
        header_cons_batch = []   # constraints actually injected per sample (None if no header)
        prefix_feat_batch = []   # design-intent features ALREADY in the code prefix (leakage metric)
        _pf = HYPERPARAMETERS["eval_prefix_fraction"]
        for j, s in enumerate(samples):
            code, complexity = sample_to_cadquery_code(s)
            gt_codes.append(code)
            # Code prefix = first eval_prefix_fraction of chars. Experiment D: at prefix=0.0 the
            # code prefix is EMPTY (header-only / from-scratch generation); at prefix>0 it is a
            # strict proper prefix (>=1 char, < full code).
            if _pf <= 0.0:
                split_idx = 0
            else:
                split_idx = max(1, min(int(len(code) * _pf), len(code) - 1))
            code_prefix = code[:split_idx]
            # Ensure prompt is a strict prefix (not the full code)
            if code_prefix == code and len(code) > 1:
                code_prefix = code[:len(code) - 1]
            prefix_feat_batch.append(detect_features(code_prefix))  # leakage: features already visible
            prompt = code_prefix
            # Constraint-conditioned: prepend a constraint header. header_source picks its CONTENT
            # (correct = GT-derived = V1; shuffled/corrupted = wrong-content via header_plan).
            cons = None
            if constraint_mode != "none":
                cons = (header_plan[batch_start + j] if header_plan is not None
                        else derive_constraints_from_sample(s))
                prompt = build_constraint_header(cons, constraint_mode) + prompt
            # Zero-prefix + no header (baseline/masked) -> empty prompt; seed with a newline so the
            # model generates a fresh program (no design-intent leakage).
            if not prompt:
                prompt = "\n"
            header_cons_batch.append(cons)
            prompts.append(prompt)
            complexities.append(complexity)

        generated = generate_batch(model, tokenizer, prompts, max_new_tokens)

        step_t = time.time() - t0
        step_times.append(step_t)

        # The oracle is a pure function of the program text (it spawns a subprocess and shares no
        # state), so scoring a batch concurrently and then walking the results in order produces
        # byte-identical output to the serial loop. Serially it was ~13% of wall clock while
        # occupying 1 of 16 cores. Its valid bit does not enter the primary metric — that
        # execution gate comes from geom_requirements.py re-executing every program offline
        # (Cspike prereg §2.1) — so this touches reporting throughput only, not the decision.
        progs = [extract_valid_program_prefix(g) for g in generated]
        if ORACLE_WORKERS > 1 and len(progs) > 1:
            with ThreadPoolExecutor(max_workers=min(ORACLE_WORKERS, len(progs))) as _ex:
                valids = list(_ex.map(run_cadquery_oracle, progs))
        else:
            valids = [run_cadquery_oracle(p) for p in progs]

        for j, (gen_code, complexity, gt_code) in enumerate(zip(generated, complexities, gt_codes)):
            prog = progs[j]
            is_valid = valids[j]
            results_by_regime[complexity]["total"] += 1
            results_by_regime[complexity]["valid"] += int(is_valid)
            all_valid += int(is_valid)
            # adherence: do the generated program's features match the GT's design intent?
            intended = detect_features(gt_code)
            produced = detect_features(prog)
            r, p = adherence_scores(intended, produced)
            if r is not None:
                adh_recalls.append(r)
            if p is not None:
                adh_precisions.append(p)
            append_adherence_sample({
                "condition": condition_name, "seed": seed, "idx": batch_start + j,
                "intended": sorted(intended), "produced": sorted(produced),
                "recall": r, "precision": p, "valid": bool(is_valid),
                "header_source": header_source,
                "header_injected": sorted(header_cons_batch[j]) if header_cons_batch[j] else [],
                "prefix_fraction": _pf,
                "prefix_features": sorted(prefix_feat_batch[j]),
                # C-spike: persist raw texts so adherence can be re-scored offline by an
                # independent geometry-side metric (preregistered; see Cspike_预注册判据_20260706.md).
                "gen_raw": gen_code, "program": prog,
                "prompt": prompts[j], "gt_code": gt_code,
            })

        done = min(batch_start + BATCH_SIZE, n)
        elapsed = time.time() - t0
        valid_so_far = all_valid
        print(f"[eval {condition_name}] {done}/{n} valid_so_far={valid_so_far} elapsed={elapsed:.0f}s")
        update_progress(phase="eval", condition=condition_name, seed=seed,
                        step=done, total_steps=n, latest_metric=valid_so_far / max(done, 1))

        # ETA calibration after first batch
        if len(step_times) == 1:
            sec_per_sample = elapsed / max(done, 1)
            eta = sec_per_sample * n
            print(f"ETA_CALIBRATION: phase=eval_{condition_name} sec_per_step={sec_per_sample:.1f} projected_total={eta:.0f}s")
            # Scale down if needed
            if eta > time_remaining():
                new_n = max(4, int(time_remaining() / sec_per_sample * 0.8))
                if new_n < n:
                    print(f"[eval {condition_name}] scaling down eval to {new_n}/{n} samples due to budget")
                    n = new_n

        t0 = time.time()

    total = sum(r["total"] for r in results_by_regime.values())
    valid = sum(r["valid"] for r in results_by_regime.values())
    validity_rate = valid / max(total, 1)

    # None (not 0.0) when a regime bucket is empty — else "no complex samples" reads as
    # "0% of complex valid". This run: lossy transpiler → all eval programs op_count<=4 → simple.
    simple_rate = (results_by_regime["simple"]["valid"] / results_by_regime["simple"]["total"]
                   if results_by_regime["simple"]["total"] > 0 else None)
    complex_rate = (results_by_regime["complex"]["valid"] / results_by_regime["complex"]["total"]
                    if results_by_regime["complex"]["total"] > 0 else None)

    adh_recall = sum(adh_recalls) / len(adh_recalls) if adh_recalls else 0.0
    adh_precision = sum(adh_precisions) / len(adh_precisions) if adh_precisions else 0.0
    adh_f1 = (2 * adh_recall * adh_precision / (adh_recall + adh_precision)) \
        if (adh_recall + adh_precision) > 0 else 0.0
    print(f"[eval {condition_name} seed={seed}] adherence_f1={adh_f1:.4f} "
          f"recall={adh_recall:.4f} precision={adh_precision:.4f}", flush=True)

    metrics = {
        "validity_rate": validity_rate,
        "valid_count": valid,
        "total_count": total,
        "validity_simple": simple_rate,
        "validity_complex": complex_rate,
        "adherence_recall": adh_recall,
        "adherence_precision": adh_precision,
        "adherence_f1": adh_f1,
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITION 1: DeepCAD Scratch Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class DeepCADScratchTransformer(nn.Module):
    """Small causal transformer trained from scratch on tokenized CAD ops."""

    def __init__(self, vocab_size=256, d_model=256, n_heads=8, n_layers=6,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos_embed(positions)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        # TransformerDecoder needs memory; use x itself as memory (autoregressive)
        out = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        logits = self.lm_head(out)
        return logits

    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8):
        self.eval()
        device = next(self.parameters()).device
        ids = prompt_ids.to(device)
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -512:])
            next_logit = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logit, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == 1:  # EOS token = 1
                break
        return ids


class DeepCADTokenDataset(TorchDataset):
    """Tokenize CadQuery code as byte-level tokens (vocab_size=256)."""

    EOS_ID = 1  # generation stops on token 1 (see generate loop); must appear in training

    def __init__(self, data_samples, max_len=512):
        self.samples = []
        for s in data_samples:
            code, _ = sample_to_cadquery_code(s)
            tokens = [b for b in code.encode("utf-8", errors="replace")[:max_len - 1]]
            # Append EOS so the model LEARNS to emit it (byte 1 ~never occurs in
            # CadQuery text, so without this the model never stops at generation).
            tokens.append(self.EOS_ID)
            if len(tokens) >= 2:
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :len(x)] = x
        y_pad[i, :len(y)] = y
    return x_pad, y_pad


def run_deepcad_scratch(seed: int, all_results: dict):
    print(f"\n=== deepcad_scratch_transformer seed={seed} ===")
    update_progress(phase="train", condition="deepcad_scratch_transformer", seed=seed)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_deepcad("train", max_rows=5000 if not SMOKE else 50)
    test_data = load_deepcad("test", max_rows=HYPERPARAMETERS["eval_subset_size"])

    ds = DeepCADTokenDataset(list(train_data), max_len=512)
    loader = DataLoader(ds, batch_size=HYPERPARAMETERS["deepcad_batch_size"],
                        shuffle=True, collate_fn=pad_collate, drop_last=True)

    model = DeepCADScratchTransformer(
        vocab_size=HYPERPARAMETERS["deepcad_vocab_size"],
        d_model=HYPERPARAMETERS["deepcad_d_model"],
        n_heads=HYPERPARAMETERS["deepcad_n_heads"],
        n_layers=HYPERPARAMETERS["deepcad_n_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMETERS["deepcad_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HYPERPARAMETERS["deepcad_train_steps"])

    max_steps = HYPERPARAMETERS["deepcad_train_steps"]
    step = 0
    t0 = time.time()

    print(f"TIME_ESTIMATE: ~{max_steps * 0.3:.0f}s for deepcad_scratch training")

    for epoch in range(1000):
        if step >= max_steps or budget_exceeded():
            break
        for x, y in loader:
            if step >= max_steps or budget_exceeded():
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # Flatten for cross-entropy
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100)
            if torch.isnan(loss):
                print("FAIL: NaN/divergence detected in deepcad_scratch")
                break
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMETERS["gradient_clip"])
            optimizer.step()
            scheduler.step()
            step += 1

            if step == 3:
                elapsed = time.time() - t0
                sec_per_step = elapsed / 3
                eta = sec_per_step * max_steps
                print(f"ETA_CALIBRATION: phase=train_deepcad_scratch sec_per_step={sec_per_step:.2f} projected_total={eta:.0f}s")
                if eta > time_remaining():
                    new_steps = max(10, int(time_remaining() / sec_per_step * 0.8))
                    print(f"[deepcad_scratch] scaling down to {new_steps} steps")
                    max_steps = new_steps

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"[deepcad_scratch seed={seed}] step={step}/{max_steps} loss={loss.item():.4f} elapsed={elapsed:.0f}s")
                update_progress(phase="train", condition="deepcad_scratch_transformer",
                                seed=seed, step=step, total_steps=max_steps,
                                latest_metric=float(loss.item()))

    print(f"[deepcad_scratch seed={seed}] training done. Evaluating...")

    # Evaluate validity: generate completions and run oracle
    valid_count = 0
    total_count = 0
    regime_results = {"simple": {"valid": 0, "total": 0},
                      "complex": {"valid": 0, "total": 0}}
    adh_recalls = []
    adh_precisions = []

    eval_n = min(HYPERPARAMETERS["eval_subset_size"], len(test_data))
    t0 = time.time()
    for i in range(eval_n):
        if budget_exceeded():
            break
        s = test_data[i]
        code, complexity = sample_to_cadquery_code(s)
        # Tokenize prompt (first eval_prefix_fraction)
        tokens = list(code.encode("utf-8", errors="replace"))
        split = max(1, int(len(tokens) * HYPERPARAMETERS["eval_prefix_fraction"]))
        prompt_tokens = tokens[:split]
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        gen_tensor = model.generate(prompt_tensor, max_new_tokens=200)
        gen_bytes = gen_tensor[0].tolist()
        try:
            gen_code = bytes([b for b in gen_bytes if 0 <= b <= 255]).decode("utf-8", errors="replace")
        except Exception:
            gen_code = ""
        prog = extract_valid_program_prefix(gen_code)
        is_valid = run_cadquery_oracle(prog)
        valid_count += int(is_valid)
        total_count += 1
        regime_results[complexity]["valid"] += int(is_valid)
        regime_results[complexity]["total"] += 1
        intended = detect_features(code)
        produced = detect_features(prog)
        r, p = adherence_scores(intended, produced)
        if r is not None:
            adh_recalls.append(r)
        if p is not None:
            adh_precisions.append(p)
        append_adherence_sample({
            "condition": "deepcad_scratch_transformer", "seed": seed, "idx": i,
            "intended": sorted(intended), "produced": sorted(produced),
            "recall": r, "precision": p, "valid": bool(is_valid),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[eval deepcad_scratch seed={seed}] {i+1}/{eval_n} valid_so_far={valid_count} elapsed={elapsed:.0f}s")
            update_progress(phase="eval", condition="deepcad_scratch_transformer",
                            seed=seed, step=i+1, total_steps=eval_n,
                            latest_metric=valid_count / max(total_count, 1))

    validity_rate = valid_count / max(total_count, 1)
    simple_rate = (regime_results["simple"]["valid"] / regime_results["simple"]["total"]
                   if regime_results["simple"]["total"] > 0 else None)
    complex_rate = (regime_results["complex"]["valid"] / regime_results["complex"]["total"]
                    if regime_results["complex"]["total"] > 0 else None)

    adh_recall = sum(adh_recalls) / len(adh_recalls) if adh_recalls else 0.0
    adh_precision = sum(adh_precisions) / len(adh_precisions) if adh_precisions else 0.0
    adh_f1 = (2 * adh_recall * adh_precision / (adh_recall + adh_precision)) \
        if (adh_recall + adh_precision) > 0 else 0.0
    metrics = {
        "validity_rate": validity_rate,
        "validity_simple": simple_rate,
        "validity_complex": complex_rate,
        "valid_count": valid_count,
        "total_count": total_count,
        "adherence_recall": adh_recall,
        "adherence_precision": adh_precision,
        "adherence_f1": adh_f1,
    }
    print(f"condition=deepcad_scratch_transformer seed={seed} adherence_f1={adh_f1:.4f} validity_rate: {validity_rate:.4f}")
    append_partial({"condition": "deepcad_scratch_transformer", "seed": seed, **metrics})
    all_results.setdefault("deepcad_scratch_transformer", {})[seed] = metrics
    write_results(all_results)
    # Free before the next condition loads its model (consistency; scratch is small
    # but the train->next-load pattern should always release).
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SFT TRAINING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def build_sft_dataset(data_samples, tokenizer, constraint_mode="none",
                      header_dropout=0.0, dropout_seed=0, train_header_content="gt"):
    """Build HuggingFace dataset for SFTTrainer from data samples.
    constraint_mode: "none"/"text"/"token" — prepends a constraint header derived
    from the sample's GROUND-TRUTH geometry (build_constraint_header). The header
    sits at the FRONT so token-truncation (which trims the tail) never drops it.
    header_dropout (Experiment B, CFG-style): per-sample probability of DROPPING the header
    (training that sample unconditioned) even when constraint_mode!="none". A mix of
    conditioned + unconditioned examples teaches the model to generate WITHOUT a header too,
    so masked-at-inference no longer collapses. Deterministic given dropout_seed.
    train_header_content (F4 causal control): "gt" = header derived from the sample's own
    geometry (real conditioning). "shuffled" = each sample gets ANOTHER real sample's GT header
    (_build_shuffle_plan over the training set) — header MARGINAL is byte-identical to the gt
    condition, only the program<->header CORRELATION is destroyed, so a model trained on this
    learns headers carry no signal (the clean control for distribution-shift vs semantics).
    "shuffled" REQUIRES header_dropout==0.0 (else it becomes a CFG model, not a content control)."""
    from datasets import Dataset as HFDataset

    max_len = HYPERPARAMETERS["max_seq_length"]
    _rng = random.Random(dropout_seed)
    _dropped = 0
    # F4 shuffled-GT: precompute the shuffle plan over the whole training set BEFORE the loop.
    _shuf_plan = None
    if train_header_content == "shuffled":
        assert constraint_mode != "none", "train_header_content=shuffled needs a constraint_mode"
        assert header_dropout == 0.0, "F4 shuffled-GT control REQUIRES header_dropout=0.0"
        _all_cons = [derive_constraints_from_sample(s) for s in data_samples]
        # Exact derangement (permutation) -> assigned-header multiset byte-identical to M's.
        _shuf_plan = _derangement_plan(_all_cons, random.Random(dropout_seed))
    elif train_header_content != "gt":
        raise ValueError(f"unknown train_header_content: {train_header_content}")
    texts = []
    for i, s in enumerate(data_samples):
        code, _ = sample_to_cadquery_code(s)
        if constraint_mode != "none" and _rng.random() >= header_dropout:
            cons = _shuf_plan[i] if _shuf_plan is not None else derive_constraints_from_sample(s)
            code = build_constraint_header(cons, constraint_mode) + code
        elif constraint_mode != "none":
            _dropped += 1   # CFG dropout: this sample trained WITHOUT its header
        # TOKEN-level truncation (not chars) leaving room for EOS, so the appended
        # EOS SURVIVES SFTTrainer's max_seq_length truncation. Char-based truncation
        # left programs >max_seq_length tokens, so the trailing EOS was cut off and
        # the model never learned to terminate (eos_frac=0 -> truncated, invalid).
        ids = tokenizer(code, add_special_tokens=False, truncation=True,
                        max_length=max_len - 8)["input_ids"]
        code = tokenizer.decode(ids)
        texts.append({"text": code + tokenizer.eos_token})

    ds = HFDataset.from_list(texts)
    # Build-time assertion: EOS must survive tokenization at max_seq_length. Fails
    # fast (scale-independent) instead of silently degenerating a multi-hour run.
    chk = tokenizer(ds[0]["text"], truncation=True, max_length=max_len)["input_ids"]
    assert chk[-1] == tokenizer.eos_token_id, (
        f"SFT data EOS truncated away (last token {chk[-1]} != eos "
        f"{tokenizer.eos_token_id}); raise max_seq_length or tighten truncation")
    _shuf_note = ""
    if _shuf_plan is not None:
        _diff = sum(1 for i in range(len(_all_cons))
                    if frozenset(_shuf_plan[i]) != frozenset(_all_cons[i]))
        _shuf_note = (f" | TRAIN_HEADER=shuffled-GT: header!=own on {_diff}/{len(_all_cons)} "
                      f"({_diff/max(len(_all_cons),1):.2f}) — marginal identical, correlation destroyed")
    print(f"[sft_data] {len(texts)} examples, EOS survives @max_seq_length={max_len} OK"
          + (f" | header_dropout={header_dropout}: {_dropped}/{len(texts)} headerless"
             f" ({_dropped/max(len(texts),1):.2f})" if header_dropout > 0 else "")
          + _shuf_note)
    return ds


def run_sft_training(seed: int, ckpt_path: Path, constraint_mode: str = "none",
                     condition_name: str = "sft", header_dropout: float = 0.0,
                     train_header_content: str = "gt"):
    """Run SFT training. Saves adapter to ckpt_path.
    constraint_mode: none/text/token. token mode adds constraint vocab tokens and trains
    ONLY their new embedding rows (freeze_old_embedding_rows).
    header_dropout (Experiment B): CFG-style per-sample header dropout in build_sft_dataset.
    train_header_content (F4): "gt"|"shuffled" — passed to build_sft_dataset; persisted to
    f4_meta.json and asserted on checkpoint-reuse so a wrong-content adapter can never be
    silently reused (adversarial-review finding #2)."""
    print(f"\n=== {condition_name} seed={seed} (SFT training) ===")

    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        # F4 guard: never reuse an adapter trained with a DIFFERENT header content.
        _meta = ckpt_path / "f4_meta.json"
        _prev = json.loads(_meta.read_text()).get("train_header_content", "gt") if _meta.exists() else "gt"
        if _prev != train_header_content:
            raise RuntimeError(f"CHECKPOINT_REUSE BLOCKED: {ckpt_path} was trained with "
                               f"train_header_content={_prev!r} but this job wants {train_header_content!r}")
        print(f"CHECKPOINT_REUSE: condition={condition_name} seed={seed} path={ckpt_path} "
              f"train_header_content={_prev}")
        return True

    print(f"CHECKPOINT_RETRAIN: condition={condition_name} seed={seed} reason=no_existing_checkpoint")
    update_progress(phase="train", condition=condition_name, seed=seed)
    set_seed(seed)

    model, tokenizer = load_qwen_4bit(seed)
    model = prepare_model_for_kbit_training(model)
    base_vocab = len(tokenizer)
    if constraint_mode == "token":
        expand_constraint_vocab(model, tokenizer)
    lora_cfg = make_lora_config()
    model = get_peft_model(model, lora_cfg)
    if constraint_mode == "token":
        freeze_old_embedding_rows(model, base_vocab)
    model.print_trainable_parameters()

    # Dataset
    max_rows = 10000 if not SMOKE else 20
    train_data = load_deepcad("train", max_rows=max_rows)
    hf_ds = build_sft_dataset(list(train_data), tokenizer, constraint_mode=constraint_mode,
                              header_dropout=header_dropout, dropout_seed=seed,
                              train_header_content=train_header_content)

    # Compute max_steps correctly
    n_samples = len(hf_ds)
    n_gpu = 1  # single GPU
    steps_per_epoch = math.ceil(
        n_samples / (HYPERPARAMETERS["sft_batch_size"] *
                     HYPERPARAMETERS["sft_grad_accum"] * n_gpu)
    )
    epochs = HYPERPARAMETERS["sft_epochs"]
    max_steps = epochs * steps_per_epoch

    # Budget guard
    remaining = time_remaining()
    # Rough estimate: ~0.8s per step
    new_steps = None
    if max_steps * 0.8 > remaining * 0.9:
        new_steps = max(2, int(remaining * 0.9 / 0.8))
        print(f"[{condition_name} seed={seed}] budget-clipping max_steps {max_steps} -> {new_steps}")
        max_steps = new_steps

    print(f"[{condition_name} seed={seed}] n_samples={n_samples} steps_per_epoch={steps_per_epoch} max_steps={max_steps}")

    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Time-budget callback
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if budget_exceeded():
                print(f"[{condition_name} seed={seed}] budget exceeded at step {state.global_step}")
                control.should_training_stop = True
            if state.global_step % 20 == 0:
                update_progress(
                    phase="train", condition=condition_name, seed=seed,
                    step=state.global_step, total_steps=max_steps,
                )
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 3:
                # ETA calibration
                elapsed = time.time() - RUN_START
                # Already some overhead; just use from the trainer's start
                pass
            return control

    # In-training generation probe (CANARY): catch training-QUALITY bugs (model not
    # learning to produce valid/terminating output) at epoch 1 (~50min) instead of
    # waiting for post-training eval (~2.5h). Loss going down does NOT reveal these
    # (e.g. the model can loop with no EOS while loss looks fine).
    _probe_prompts = []
    for _s in list(train_data)[:5]:
        _c, _ = sample_to_cadquery_code(_s)
        _sp = max(1, min(int(len(_c) * 0.30), len(_c) - 1))
        _probe_prompts.append(_c[:_sp])

    class CanaryCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            try:
                model.eval()
                _prev = model.config.use_cache
                model.config.use_cache = True
                valid = eos = 0
                n = max(len(_probe_prompts), 1)
                for p in _probe_prompts:
                    enc = tokenizer(p, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            **enc, max_new_tokens=256, do_sample=False,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id)
                    g = out[0][enc["input_ids"].shape[1]:]
                    if tokenizer.eos_token_id in g.tolist():
                        eos += 1
                    prog = extract_valid_program_prefix(
                        p + tokenizer.decode(g, skip_special_tokens=True))
                    if run_cadquery_oracle(prog):
                        valid += 1
                model.config.use_cache = _prev
                ep = int(round(state.epoch))
                print(f"CANARY: condition={condition_name} seed={seed} epoch={ep} "
                      f"eos_frac={eos/n:.2f} validity={valid/n:.2f} ({valid}/{n})", flush=True)
                update_progress(phase="train", condition=condition_name, seed=seed,
                                canary_epoch=ep, canary_validity=valid / n)
                # Abort ONLY on the real EOS-not-learned bug signature: validity==0 AND
                # eos_frac==0 (model never learned to stop). A 5-sample probe can read 0/5
                # by chance even at ~20% true validity (0.8^5≈33%), so DON'T abort on
                # validity==0 alone — if eos>0 the model IS learning to terminate and a
                # zero is just early-training noise, not a bug.
                if ep >= 1 and valid == 0 and eos == 0:
                    print(f"CANARY_ABORT: condition={condition_name} seed={seed} validity=0 "
                          f"AND eos_frac=0 after epoch {ep} — EOS-not-learned bug (model never "
                          "stops); stopping early instead of burning the rest of the budget", flush=True)
                    control.should_training_stop = True
            except Exception:
                traceback.print_exc()
            finally:
                model.train()
            return control

    training_args = SFTConfig(
        output_dir=str(ckpt_path),
        max_steps=max_steps,
        per_device_train_batch_size=HYPERPARAMETERS["sft_batch_size"],
        gradient_accumulation_steps=HYPERPARAMETERS["sft_grad_accum"],
        learning_rate=HYPERPARAMETERS["sft_lr"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field="text",
        max_seq_length=HYPERPARAMETERS["max_seq_length"],
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_ds,
        args=training_args,
        callbacks=[BudgetCallback(), CanaryCallback()],
    )
    if getattr(model, "_arc_single_gpu", False):
        trainer.args._n_gpu = 1  # single-GPU: stop HF Trainer DataParallel (>1 GPU visible)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[{condition_name} seed={seed}] SFT training done in {elapsed:.0f}s")

    # Save adapter
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    # F4: persist training-header provenance + step completeness for reuse-guard and audit.
    (ckpt_path / "f4_meta.json").write_text(json.dumps({
        "train_header_content": train_header_content, "constraint_mode": constraint_mode,
        "header_dropout": header_dropout, "planned_max_steps": max_steps, "seed": seed}))
    print(f"[{condition_name} seed={seed}] Saved adapter to {ckpt_path} "
          f"(train_header_content={train_header_content}, max_steps={max_steps})")
    # 结构性完整性检测(硬干预层):save 后 adapter 必须落盘,否则下游复用会崩/静默重训
    if not (ckpt_path / "adapter_config.json").exists():
        print(f"QUALITY_CHECK: condition={condition_name} seed={seed} status=FAIL "
              f":: checkpoint 未保存(adapter_config.json 缺失,save 失败?)", flush=True)
    # Free the 4-bit training model + Trainer BEFORE eval loads its own model.
    # Otherwise the SFT model stays resident (~12GB) and the eval load sees little
    # free VRAM -> decide_placement needlessly shards eval across both GPUs
    # (slower per-token generation). Releasing it lets eval run single-GPU.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return True


def run_sft_eval(seed: int, ckpt_path: Path, condition_name: str,
                 all_results: dict, eval_mode: str = "none", train_mode: str = "none"):
    """Load SFT adapter in fp16 and evaluate validity.
    eval_mode (none/text/token): constraint header to prepend to eval prompts.
    train_mode (none/text/token): how the checkpoint was trained — token-trained
    checkpoints need the base resized (expand_vocab) before the adapter attaches.
    The masked arm passes eval_mode="none", train_mode="token" (loads resized, no header)."""
    print(f"\n=== {condition_name} seed={seed} (eval) ===")
    update_progress(phase="eval", condition=condition_name, seed=seed)

    test_data = load_deepcad("test", max_rows=HYPERPARAMETERS["eval_subset_size"])
    model, tokenizer = load_qwen_fp16_for_eval(
        str(ckpt_path), seed, expand_vocab=(train_mode == "token"))

    # Get calibrated max_new_tokens
    train_sample = load_deepcad("train", max_rows=50)
    max_new = calibrate_generation(tokenizer, list(train_sample))

    metrics = eval_validity(model, tokenizer, test_data, max_new,
                            condition_name, seed, constraint_mode=eval_mode)

    del model
    torch.cuda.empty_cache()

    print(f"condition={condition_name} seed={seed} validity_rate: {metrics['validity_rate']:.4f}")
    append_partial({"condition": condition_name, "seed": seed, **metrics})
    all_results.setdefault(condition_name, {})[seed] = metrics
    write_results(all_results)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PROXY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class ValidityProxyClassifier(nn.Module):
    """2-layer MLP classifier. Input: [hidden_state (1536), complexity_features (3)]."""

    FEATURE_DIM_FULL = 1536 + 3   # hidden state + (entropy, length, op_count_ratio)
    FEATURE_DIM_HIDDEN_ONLY = 1536

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_complexity_features(code: str) -> np.ndarray:
    """Compute 3 complexity features: entropy, length_norm, op_count_ratio."""
    tokens = code.encode("utf-8", errors="replace")
    # Byte entropy
    entropy = None
    counts = None
    total = None
    if len(tokens) == 0:
        entropy = 0.0
    else:
        from collections import Counter
        counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log2(c / total + 1e-10) for c in counts.values())

    # Normalized length (vs 512 tokens)
    length_norm = min(len(tokens) / 512.0, 1.0)

    # Operation count ratio
    op_count = (code.count(".box(") + code.count(".cylinder(") +
                code.count(".cut(") + code.count(".sphere(") +
                code.count(".union("))
    op_ratio = min(op_count / 10.0, 1.0)

    return np.array([entropy / 8.0, length_norm, op_ratio], dtype=np.float32)


def get_hidden_state(model, tokenizer, code: str) -> np.ndarray:
    """Extract mean-pooled last-layer hidden state for code string."""
    device = next(model.parameters()).device
    inputs = tokenizer(code, return_tensors="pt", truncation=True,
                       max_length=HYPERPARAMETERS["max_seq_length"]).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Mean-pool last hidden state
    last_hidden = outputs.hidden_states[-1]  # (1, T, hidden)
    pooled = last_hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
    return pooled


def collect_proxy_samples(model, tokenizer, data_samples, n_samples: int,
                          use_complexity_features: bool = True):
    """
    Generate N completions from model, run oracle, return (features, labels).
    """
    print(f"[proxy] Collecting {n_samples} labeled samples...")
    features_list = []
    labels_list = []
    t0 = time.time()

    eval_data = data_samples[:n_samples]

    for i, s in enumerate(eval_data):
        if budget_exceeded():
            print(f"[proxy] budget exceeded at {i}/{n_samples}")
            break

        code, complexity = sample_to_cadquery_code(s)
        split_idx = max(1, int(len(code) * 0.30))
        prompt = code[:split_idx]

        # Generate completion
        gen_list = generate_batch(model, tokenizer, [prompt], max_new_tokens=200)
        gen_code = gen_list[0] if gen_list else ""
        prog = extract_valid_program_prefix(gen_code)

        # Oracle label
        is_valid = run_cadquery_oracle(prog)

        # Features: hidden state
        hidden = get_hidden_state(model, tokenizer, prog[:512])  # (1536,)
        if use_complexity_features:
            complexity_feats = compute_complexity_features(prog)
            feat = np.concatenate([hidden, complexity_feats])
        else:
            feat = hidden

        features_list.append(feat)
        labels_list.append(float(is_valid))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            valid_so_far = sum(labels_list)
            print(f"[proxy collection] {i+1}/{n_samples} valid={int(valid_so_far)} elapsed={elapsed:.0f}s")
            update_progress(phase="proxy_collection", step=i+1, total_steps=n_samples)

            if i == 9:
                sec_per = elapsed / 10
                eta = sec_per * n_samples
                print(f"ETA_CALIBRATION: phase=proxy_collection sec_per_step={sec_per:.1f} projected_total={eta:.0f}s")
                if eta > time_remaining():
                    new_n = max(10, int(time_remaining() * 0.7 / sec_per))
                    if new_n < len(eval_data):
                        print(f"[proxy] scaling collection to {new_n} samples")
                        eval_data = list(data_samples[:new_n])

    if not features_list:
        return np.zeros((0, 1536 + (3 if use_complexity_features else 0))), np.zeros(0)

    return np.array(features_list), np.array(labels_list)


def train_proxy_classifier(features: np.ndarray, labels: np.ndarray,
                            use_complexity_features: bool,
                            save_path: Path = None) -> ValidityProxyClassifier:
    """Train the validity proxy classifier on collected features/labels."""
    input_dim = features.shape[1]
    clf = ValidityProxyClassifier(input_dim, HYPERPARAMETERS["proxy_hidden_dim"])
    clf.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = clf.to(device)

    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(clf.parameters(), lr=HYPERPARAMETERS["proxy_lr"])

    n_epochs = HYPERPARAMETERS["proxy_train_epochs"]
    batch_size = min(32, len(X))
    t0 = time.time()

    for epoch in range(n_epochs):
        if budget_exceeded():
            break
        # Shuffle
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx]
            yb = y[idx]
            logits = clf(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[proxy train] epoch={epoch+1}/{n_epochs} loss={avg_loss:.4f}")

    # Accuracy on training set
    clf.eval()
    with torch.no_grad():
        logits = clf(X)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean().item()
    print(f"[proxy] training accuracy={acc:.4f}")

    meta = None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.cpu().state_dict(), save_path)
        # Also save metadata
        meta = {"input_dim": input_dim, "hidden_dim": HYPERPARAMETERS["proxy_hidden_dim"],
                "use_complexity": use_complexity_features}
        (save_path.parent / f"{save_path.stem}_meta.json").write_text(json.dumps(meta, default=_json_default))

    clf = clf.to(device)
    return clf, acc


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_grpo(seed: int, start_ckpt: Path, out_ckpt: Path,
             reward_fn, condition_name: str, all_results: dict,
             extra_metrics: dict = None):
    """Generic GRPO training loop using GRPOTrainer."""
    from trl import GRPOTrainer, GRPOConfig
    from transformers import TrainerCallback
    from datasets import Dataset as HFDataset

    print(f"\n=== {condition_name} seed={seed} (GRPO) ===")
    if out_ckpt.exists() and (out_ckpt / "adapter_config.json").exists():
        print(f"CHECKPOINT_REUSE: condition={condition_name} seed={seed} path={out_ckpt}")
    else:
        print(f"CHECKPOINT_RETRAIN: condition={condition_name} seed={seed} reason=no_existing_grpo_checkpoint")

    update_progress(phase="grpo_train", condition=condition_name, seed=seed)
    set_seed(seed)

    # Load model in 4-bit for training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # GRPO OOM fix (option D, pure placement, FULL fidelity): device_map="auto" co-locates
    # the TIED embed+lm_head (vocab=151936) on GPU0, so the full-vocab policy+ref logits
    # stack there (~22GB -> OOM at steps 7/11/23). On >=2 GPUs use a BALANCED hand-map
    # (embed+lm_head+norm + last 8 layers -> GPU1, first 20 layers -> GPU0). Measured
    # 2x24GB: GPU0 18.0GB / GPU1 19.7GB (incl resident fp16 reward-ref), ~6/4GB headroom,
    # survived 30 GRPO steps no-OOM. num_generations / max_completion_length /
    # gradient_checkpointing are UNCHANGED (zero science change vs options B/C).
    _n_gpu_vis = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if _n_gpu_vis >= 2 and estimate_peak_gb("grpo") > (
            torch.cuda.mem_get_info(0)[0] / 1e9) * 0.85:
        _bal_map = build_grpo_balanced_device_map(HYPERPARAMETERS["base_model"])
        placement, _grpo_single = {"device_map": _bal_map}, False
        _n0 = sum(1 for v in _bal_map.values() if v == 0)
        print(f"PLACEMENT: grpo BALANCED hand-map (embed+lm_head->GPU1, "
              f"{_n0} layers->GPU0, rest->GPU1) n_gpu={_n_gpu_vis}")
    else:
        placement, _grpo_single = decide_placement(estimate_peak_gb("grpo"), is_training=True)

    base = None
    model = None
    lora_cfg = None
    if start_ckpt.exists() and (start_ckpt / "adapter_config.json").exists():
        # Load base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            HYPERPARAMETERS["base_model"],
            quantization_config=bnb_config,
            **placement,
            torch_dtype=torch.bfloat16,
        )
        # QLoRA + gradient_checkpointing needs input grads enabled, else backward
        # breaks with "element 0 of tensors does not require grad". The fresh
        # branch gets this via prepare_model_for_kbit_training; the resume branch
        # must do the same before wrapping with the (trainable) adapter.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, str(start_ckpt), is_trainable=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            HYPERPARAMETERS["base_model"],
            quantization_config=bnb_config,
            **placement,
            torch_dtype=torch.bfloat16,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        lora_cfg = make_lora_config()
        model = get_peft_model(base, lora_cfg)

    # prepare_model_for_kbit_training sets config.use_cache=False; re-enable it so
    # GRPO generation uses the KV cache (no gradient_checkpointing here, so it's safe).
    model.config.use_cache = True
    model._arc_single_gpu = _grpo_single  # GRPOTrainer must set _n_gpu=1 if single-GPU

    tokenizer = AutoTokenizer.from_pretrained(
        HYPERPARAMETERS["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset: prompts for GRPO
    train_data = load_deepcad("train", max_rows=2000 if not SMOKE else 20)
    prompts = []
    for s in list(train_data):
        code, _ = sample_to_cadquery_code(s)
        split_idx = max(1, int(len(code) * 0.30))
        prompts.append({"prompt": code[:split_idx]})
    hf_ds = HFDataset.from_list(prompts)

    grpo_steps = HYPERPARAMETERS["grpo_steps"]

    # Budget guard
    remaining = time_remaining()
    # ~120s per GRPO step (measured)
    new_steps = None
    if grpo_steps * 120 > remaining * 0.85:
        new_steps = max(1, int(remaining * 0.85 / 120))
        print(f"[{condition_name} seed={seed}] budget-clipping grpo_steps {grpo_steps} -> {new_steps}")
        grpo_steps = new_steps

    out_ckpt.mkdir(parents=True, exist_ok=True)

    # Dead-reward guard state
    _dead_reward_counter = {"count": 0, "last_reward": None}

    class GRPOBudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, logs=None, **kwargs):
            if budget_exceeded():
                print(f"[{condition_name} seed={seed}] budget exceeded at GRPO step {state.global_step}")
                control.should_training_stop = True
            if state.global_step % 5 == 0:
                update_progress(phase="grpo_train", condition=condition_name,
                                seed=seed, step=state.global_step, total_steps=grpo_steps)
            # ETA calibration
            if state.global_step == 3:
                elapsed_since_start = time.time() - RUN_START
                # rough per-step from logged time
                pass
            return control

    grpo_config = GRPOConfig(
        output_dir=str(out_ckpt),
        max_steps=grpo_steps,
        per_device_train_batch_size=1,
        # CRITICAL: TRL GRPOConfig defaults gradient_accumulation_steps to 8 → 8 generations
        # per optimizer step (~8× slower: measured 725s/step vs ~95s/generation). run-5 set
        # this to 1 explicitly; it was lost in a regen. Keep it explicit.
        gradient_accumulation_steps=HYPERPARAMETERS["grpo_grad_accum"],
        learning_rate=HYPERPARAMETERS["grpo_lr"],
        num_generations=HYPERPARAMETERS["grpo_num_generations"],
        # GRPO-specific completion cap, DECOUPLED from eval's max_completion_length so a
        # smaller GRPO-training cap (memory) does NOT change the eval cap (comparability).
        # Default = eval's cap (768) → unchanged behavior.
        max_completion_length=HYPERPARAMETERS.get("grpo_max_completion_length",
                                                  HYPERPARAMETERS["max_completion_length"]),
        bf16=True,
        # gradient_checkpointing forces use_cache=False, which disables the KV cache
        # during GRPO generation -> O(n^2) decoding (~30-40x slower; profiled at
        # ~946s/step). Default OFF (KV cache fast); configurable as an OOM mitigation.
        gradient_checkpointing=HYPERPARAMETERS.get("grpo_gradient_checkpointing", False),
        logging_steps=HYPERPARAMETERS.get("grpo_logging_steps", 5),
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Wrap reward_fn to detect dead reward zone
    step_rewards = []

    def wrapped_reward_fn(completions, prompts=None, **kwargs):
        scores = reward_fn(completions, prompts=prompts)
        step_rewards.append(np.mean(scores))
        recent = None
        mean_r = None
        if len(step_rewards) >= 20:
            recent = step_rewards[-20:]
            if max(recent) - min(recent) < 0.01:
                mean_r = np.mean(recent)
                print(f"RL_DEAD_REWARD: steps={len(step_rewards)} reward={mean_r:.4f}")
        return scores

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=wrapped_reward_fn,
        train_dataset=hf_ds,
        processing_class=tokenizer,
        args=grpo_config,
        callbacks=[GRPOBudgetCallback()],
    )
    if getattr(model, "_arc_single_gpu", False):
        trainer.args._n_gpu = 1  # single-GPU: stop HF Trainer DataParallel (>1 GPU visible)

    # ── REWARD CANARY: verify the reward fn is correctly WIRED before burning GRPO ──
    # The reward VALUE alone CANNOT tell a headless-extraction BUG (reward stuck at the
    # floor because the program HEAD lives in the prompt, not the completion) from an
    # UNDERTRAINED model (also low reward). So probe DETERMINISTICALLY: feed KNOWN-VALID
    # ground-truth programs to the reward fn both the way TRL will (prompts+completions)
    # and headless (completion only). If the reward IGNORES the prompt (the bug), the two
    # are byte-identical -> equal means. A correctly-wired reward uses the prompt, so the
    # full valid program and the headless fragment score DIFFERENTLY (full >= headless).
    # Equal => the reward never sees the program head => dead reward => abort, don't burn
    # hours. (Undertrained-but-correctly-wired rewards still differ, so this won't false-abort.)
    try:
        _gt = list(load_deepcad("test", max_rows=5))
        _ps, _cs = [], []
        for _s in _gt:
            _code, _ = sample_to_cadquery_code(_s)
            _sp = max(1, min(int(len(_code) * 0.30), len(_code) - 1))
            _ps.append(_code[:_sp]); _cs.append(_code[_sp:])
        if _ps:
            _m_full = float(np.mean(reward_fn(completions=_cs, prompts=_ps)))
            _m_head = float(np.mean(reward_fn(completions=_cs, prompts=None)))
            print(f"REWARD_CANARY: condition={condition_name} seed={seed} "
                  f"gt_full_reward_mean={_m_full:.3f} gt_headless_reward_mean={_m_head:.3f} "
                  f"n={len(_ps)}", flush=True)
            update_progress(phase="grpo_train", condition=condition_name, seed=seed,
                            reward_canary_full=_m_full, reward_canary_headless=_m_head)
            if abs(_m_full - _m_head) < 1e-6:
                print(f"REWARD_CANARY_ABORT: condition={condition_name} seed={seed} the "
                      f"reward is IDENTICAL with/without the prompt (full={_m_full:.3f} == "
                      f"headless={_m_head:.3f}) on KNOWN-VALID programs — the reward fn "
                      f"ignores the program HEAD in the prompt (headless-extraction bug). "
                      f"Skipping GRPO instead of burning the budget on a dead reward.", flush=True)
                del model, base                 # free the GRPO model before bailing —
                torch.cuda.empty_cache()        # else the next condition OOMs on load
                return None
    except Exception:
        traceback.print_exc()

    t_grpo_start = time.time()
    try:
        trainer.train()
    except Exception as e:
        traceback.print_exc()
        print(f"[{condition_name} seed={seed}] GRPO training exception: {e}")

    elapsed_grpo = time.time() - t_grpo_start
    actual_steps = trainer.state.global_step if hasattr(trainer, 'state') else grpo_steps
    sec_per_step = None
    if actual_steps > 0:
        sec_per_step = elapsed_grpo / actual_steps
        print(f"[{condition_name} seed={seed}] GRPO done. actual_steps={actual_steps} "
              f"sec_per_step={sec_per_step:.1f}s")

    # Save adapter
    try:
        model.save_pretrained(str(out_ckpt))
        tokenizer.save_pretrained(str(out_ckpt))
    except Exception as e:
        traceback.print_exc()
    # 结构性完整性检测(硬干预层):GRPO save 包了 try/except 会继续,这里显式标 FAIL
    if not (out_ckpt / "adapter_config.json").exists():
        print(f"QUALITY_CHECK: condition={condition_name} seed={seed} status=FAIL "
              f":: GRPO checkpoint 未保存(adapter_config.json 缺失)", flush=True)

    # Eval
    del model
    del base
    torch.cuda.empty_cache()

    test_data = load_deepcad("test", max_rows=HYPERPARAMETERS["eval_subset_size"])
    fp16_model, fp16_tokenizer = load_qwen_fp16_for_eval(str(out_ckpt), seed)

    train_sample = load_deepcad("train", max_rows=50)
    max_new = calibrate_generation(fp16_tokenizer, list(train_sample))

    metrics = eval_validity(fp16_model, fp16_tokenizer, test_data, max_new,
                            condition_name, seed)
    metrics["wall_clock_sec_per_grpo_step"] = elapsed_grpo / max(actual_steps, 1)

    if extra_metrics:
        metrics.update(extra_metrics)

    del fp16_model
    torch.cuda.empty_cache()

    print(f"condition={condition_name} seed={seed} validity_rate: {metrics['validity_rate']:.4f}")
    append_partial({"condition": condition_name, "seed": seed, **metrics})
    all_results.setdefault(condition_name, {})[seed] = metrics
    write_results(all_results)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_oracle_reward_fn():
    """Reward function using actual CadQuery oracle."""
    def reward_fn(completions, prompts=None, **kwargs):
        scores = []
        for i, c in enumerate(completions):
            # HEADLESS completion: TRL passes the reward fn the completion ONLY
            # (the prompt separately). The program HEAD (`result = cq.Workplane(...)`)
            # lives in the prompt, so the boundary extractor finds NO result-assignment
            # in the completion alone -> every program scores invalid -> reward sticks
            # at -1.0 (dead reward). Prepend the prompt before extract, same as eval's
            # generate_batch (full program = prompt + completion).
            full = (prompts[i] + c) if prompts else c
            prog = extract_valid_program_prefix(full)
            is_valid = run_cadquery_oracle(prog)
            scores.append(1.0 if is_valid else -1.0)
        return scores
    return reward_fn


def make_proxy_reward_fn(proxy_clf: ValidityProxyClassifier, ref_model,
                          ref_tokenizer, use_complexity_features: bool = True):
    """Reward function using proxy classifier."""
    proxy_clf.eval()
    device = next(proxy_clf.parameters()).device

    def reward_fn(completions, prompts=None, **kwargs):
        scores = []
        for i, c in enumerate(completions):
            try:
                # HEADLESS completion -> prepend prompt before extract (program HEAD
                # `result = ...` lives in the prompt). The proxy was TRAINED on full
                # programs (collect_proxy_samples goes through generate_batch, which
                # prepends), so it MUST score full programs too, or it sees an
                # out-of-distribution headless fragment and returns garbage.
                full = (prompts[i] + c) if prompts else c
                prog = extract_valid_program_prefix(full)
                hidden = get_hidden_state(ref_model, ref_tokenizer, prog[:512])
                if use_complexity_features:
                    complexity_feats = compute_complexity_features(prog)
                    feat = np.concatenate([hidden, complexity_feats])
                else:
                    feat = hidden
                feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = proxy_clf(feat_t)
                    prob = torch.sigmoid(logit).item()
                # Clip to avoid reward explosion
                prob = max(0.05, min(0.95, prob))
                scores.append(float(prob) * 2 - 1)  # map [0.05,0.95] -> [-0.9, 0.9]
            except Exception:
                traceback.print_exc()
                scores.append(-1.0)
        return scores

    return reward_fn


def run_proxy_oracle_check(proxy_clf: ValidityProxyClassifier,
                            ref_model, ref_tokenizer,
                            data_samples, n_samples: int,
                            use_complexity_features: bool) -> float:
    """Check proxy vs oracle agreement on n_samples. Returns agreement rate."""
    proxy_clf.eval()
    device = next(proxy_clf.parameters()).device

    agree = 0
    total = 0
    for s in data_samples[:n_samples]:
        code, _ = sample_to_cadquery_code(s)
        split_idx = max(1, int(len(code) * 0.30))
        prompt = code[:split_idx]
        gen_list = generate_batch(ref_model, ref_tokenizer, [prompt], max_new_tokens=150)
        gen_code = gen_list[0] if gen_list else ""
        prog = extract_valid_program_prefix(gen_code)

        # Oracle
        oracle_valid = run_cadquery_oracle(prog)

        # Proxy
        try:
            hidden = get_hidden_state(ref_model, ref_tokenizer, prog[:512])
            if use_complexity_features:
                complexity_feats = compute_complexity_features(prog)
                feat = np.concatenate([hidden, complexity_feats])
            else:
                feat = hidden
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = proxy_clf(feat_t)
                prob = torch.sigmoid(logit).item()
            proxy_valid = prob > 0.5
        except Exception:
            traceback.print_exc()
            proxy_valid = False

        agree += int(oracle_valid == proxy_valid)
        total += 1

    rate = agree / max(total, 1)
    print(f"[proxy oracle check] agreement={rate:.4f} ({agree}/{total})")
    return rate


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITION: GRPO WITH PROXY REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def run_grpo_proxy(seed: int, all_results: dict,
                   use_complexity_features: bool = True,
                   condition_name: str = "grpo_proxy_reward_classifier"):
    """
    1. Load sft_constraint_tokens checkpoint
    2. Collect proxy samples
    3. Train proxy classifier
    4. Run GRPO with proxy reward
    5. Periodic oracle checks with online proxy refresh
    """
    print(f"\n=== {condition_name} seed={seed} ===")
    update_progress(phase="proxy_setup", condition=condition_name, seed=seed)

    if budget_exceeded():
        print(f"CONDITION_SKIPPED: condition={condition_name} seed={seed} reason=time_budget")
        return None

    # Step 1: SFT checkpoint (prerequisite). GRPO branches from the NO-CONSTRAINTS SFT
    # (normal vocab, no constraint header) so the geometric-reward novelty is isolated
    # and GRPO needs no vocab-resize/header handling. (Reuses Condition-2's checkpoint.)
    sft_ckpt = sft_no_constraint_ckpt(seed)
    if not (sft_ckpt / "adapter_config.json").exists():
        print(f"[{condition_name}] SFT no-constraints checkpoint missing, training...")
        run_sft_training(seed, sft_ckpt, constraint_mode="none",
                         condition_name="qwen25coder_sft_no_constraints")

    print(f"CHECKPOINT_REUSE: condition={condition_name} seed={seed} path={sft_ckpt}")

    if budget_exceeded():
        print(f"CONDITION_SKIPPED: condition={condition_name} seed={seed} reason=time_budget")
        return None

    # Step 2: Load ref model (fp16) for feature extraction and proxy collection
    ref_model, ref_tokenizer = load_qwen_fp16_for_eval(str(sft_ckpt), seed)

    proxy_feats_path = CKPT_DIR / f"proxy_features_seed{seed}.npz"

    loaded = None
    features = None
    labels = None
    train_data = None
    if proxy_feats_path.exists():
        print(f"[{condition_name}] loading cached proxy features from {proxy_feats_path}")
        loaded = np.load(str(proxy_feats_path))
        features = loaded["features"]
        labels = loaded["labels"]
    else:
        # Collect proxy samples
        train_data = load_deepcad("train", max_rows=HYPERPARAMETERS["proxy_offline_samples"] * 3)
        features, labels = collect_proxy_samples(
            ref_model, ref_tokenizer, list(train_data),
            n_samples=HYPERPARAMETERS["proxy_offline_samples"],
            use_complexity_features=True,  # always cache FULL feats; slice for hidden-only below
        )
        if len(features) > 0:
            np.savez(str(proxy_feats_path), features=features, labels=labels)

    if len(features) == 0:
        print(f"CONDITION_SKIPPED: condition={condition_name} seed={seed} reason=no_proxy_samples")
        return None

    # The cache always stores the FULL 1539-dim features; the hidden-only ablation
    # uses the first 1536 (hidden-state) columns so it is a clean drop of the
    # complexity features over the SAME hidden states. Key the classifier path by
    # feature mode so the full and hidden-only ablations don't overwrite each other
    # (and so reward_fn's feature dim always matches the trained classifier).
    if not use_complexity_features and features.shape[1] > 1536:
        features = features[:, :1536]
    proxy_path = CKPT_DIR / f"proxy_classifier_seed{seed}_{'full' if use_complexity_features else 'hidden'}.pt"

    # Step 3: Train proxy classifier
    proxy_clf, proxy_acc = train_proxy_classifier(features, labels, use_complexity_features,
                                                   save_path=proxy_path)

    del ref_model  # free memory before GRPO
    torch.cuda.empty_cache()

    if budget_exceeded():
        print(f"CONDITION_SKIPPED: condition={condition_name} seed={seed} reason=time_budget")
        # Still save what we have
        metrics = {"validity_rate": 0.0, "proxy_classifier_accuracy": proxy_acc,
                   "note": "skipped_before_grpo"}
        append_partial({"condition": condition_name, "seed": seed, **metrics})
        all_results.setdefault(condition_name, {})[seed] = metrics
        write_results(all_results)
        return metrics

    # Step 4: GRPO. Each proxy variant MUST save to its OWN checkpoint — else the
    # hidden-only ablation (use_complexity_features=False) reuses the full-proxy GRPO
    # checkpoint and silently becomes identical to grpo_proxy_reward_classifier.
    out_ckpt = grpo_proxy_ckpt(seed) if use_complexity_features else grpo_hidden_only_ckpt(seed)

    # We need a lightweight reward fn that doesn't hold a whole second model in mem
    # Use the proxy clf + reloaded ref model inside reward_fn
    # But loading model in reward_fn on every call is too slow.
    # Solution: load once, keep in closure.
    # GRPO OOM fix (edit B): pin the resident fp16 reward-ref model to GPU1 (away from the
    # policy's layer activations on GPU0) when >=2 GPUs are visible; else default placement.
    _ref_dev = 1 if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else None
    ref_for_reward, ref_tok_for_reward = load_qwen_fp16_for_eval(
        str(sft_ckpt), seed, ref_device=_ref_dev)

    proxy_reward_fn = make_proxy_reward_fn(proxy_clf, ref_for_reward, ref_tok_for_reward,
                                            use_complexity_features=use_complexity_features)

    # Proxy oracle agreement tracking with online refresh
    oracle_check_data = list(load_deepcad("test", max_rows=100))
    agreement_history = []
    _proxy_refresh_count = [0]

    def proxy_reward_with_oracle_check(completions, prompts=None, **kwargs):
        scores = proxy_reward_fn(completions, prompts=prompts)
        return scores

    extra_metrics = {
        "proxy_classifier_accuracy": proxy_acc,
        # None = NOT YET MEASURED (overwritten with the real agreement after GRPO, below). A
        # 0.0 placeholder would read as "zero agreement" if GRPO is budget-truncated before the
        # final check runs — None keeps the budget-cut case honest (not-measured vs measured-0).
        "proxy_oracle_agreement_rate": None,
    }

    metrics = run_grpo(seed, sft_ckpt, out_ckpt,
                       proxy_reward_with_oracle_check, condition_name, all_results,
                       extra_metrics=extra_metrics)

    agreement = None
    if metrics is not None:
        metrics["proxy_classifier_accuracy"] = proxy_acc
        # Final oracle agreement check
        if not budget_exceeded():
            agreement = run_proxy_oracle_check(
                proxy_clf, ref_for_reward, ref_tok_for_reward,
                oracle_check_data,
                n_samples=min(HYPERPARAMETERS["oracle_check_samples"], len(oracle_check_data)),
                use_complexity_features=use_complexity_features,
            )
            metrics["proxy_oracle_agreement_rate"] = agreement
            print(f"[{condition_name} seed={seed}] final proxy-oracle agreement={agreement:.4f}")

    del ref_for_reward
    torch.cuda.empty_cache()

    if metrics is not None:
        append_partial({"condition": condition_name, "seed": seed, **metrics})
        all_results.setdefault(condition_name, {})[seed] = metrics
        write_results(all_results)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def print_aggregate(condition_name: str, cond_results: dict):
    """Print mean adherence_f1 (PRIMARY) and validity_rate (secondary) over seeds."""
    def col(key):
        vals = [v[key] for v in cond_results.values() if isinstance(v, dict) and key in v]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))
    af1_m, af1_s = col("adherence_f1")
    val_m, val_s = col("validity_rate")
    print(f"condition={condition_name} adherence_f1_mean={af1_m:.4f} adherence_f1_std={af1_s:.4f} "
          f"validity_rate_mean={val_m:.4f} validity_rate_std={val_s:.4f}")


def main():
    print(f"=== CAD LLM Experiment START {datetime.now().isoformat()} ===")
    print(f"TIME_ESTIMATE: ~158000s/~44h (measured: SFT 8s/step, GRPO 242s/step sharded@grad_accum=1, gen 7.7s/sample) | budget={BUDGET_SECONDS}s | stop_at={BUDGET_SECONDS*HYPERPARAMETERS['budget_fraction']:.0f}s")
    update_progress(phase="startup", step=0, total_steps=7)

    # ── Oracle self-test ──────────────────────────────────────────────────────
    oracle_self_test()

    # ── Load dataset for calibration ─────────────────────────────────────────
    print("\n[main] Loading dataset for calibration...")
    calib_data = load_deepcad("train", max_rows=100)
    gt_codes = []
    for s in list(calib_data)[:30]:
        code, _ = sample_to_cadquery_code(s)
        gt_codes.append(code)

    # Oracle calibration on ground-truth
    gt_validity = oracle_calibration(gt_codes, "ground_truth")
    print(f"[main] Ground-truth validity ceiling: {gt_validity:.4f}")

    if gt_validity < 0.01:
        print("ORACLE_CALIBRATION: ground_truth_validity is ~0 — oracle pipeline may be broken")
        # Don't raise; continue — the oracle tests will reveal the issue
    update_progress(calibration={"ground_truth": gt_validity})

    # Collect ALL results
    all_results = {}
    # Try loading partial results
    if PARTIAL_FILE.exists():
        try:
            with open(PARTIAL_FILE) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    cond = rec.get("condition", "")
                    seed = rec.get("seed", 0)
                    all_results.setdefault(cond, {})[seed] = rec
            print(f"[main] Loaded {sum(len(v) for v in all_results.values())} partial results")
        except Exception:
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 1: deepcad_scratch_transformer
    # ─────────────────────────────────────────────────────────────────────────
    cond = "deepcad_scratch_transformer"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            run_deepcad_scratch(seed, all_results)
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 2: qwen25coder_sft_no_constraints
    # ─────────────────────────────────────────────────────────────────────────
    cond = "qwen25coder_sft_no_constraints"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            ckpt = sft_no_constraint_ckpt(seed)
            run_sft_training(seed, ckpt, constraint_mode="none", condition_name=cond)
            if not budget_exceeded():
                run_sft_eval(seed, ckpt, cond, all_results)
            else:
                print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget (after train)")
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 3: qwen25coder_sft_constraint_tokens
    # ─────────────────────────────────────────────────────────────────────────
    cond = "qwen25coder_sft_constraint_tokens"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            ckpt = sft_constraint_ckpt(seed)
            run_sft_training(seed, ckpt, constraint_mode="token", condition_name=cond)
            if not budget_exceeded():
                run_sft_eval(seed, ckpt, cond, all_results, eval_mode="token", train_mode="token")
            else:
                print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget (after train)")
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 3b: qwen25coder_sft_constraint_text (plain-text constraint header —
    # the control arm for "dedicated tokens vs plain text" of the SAME constraint spec)
    # ─────────────────────────────────────────────────────────────────────────
    cond = "qwen25coder_sft_constraint_text"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            ckpt = sft_constraint_text_ckpt(seed)
            run_sft_training(seed, ckpt, constraint_mode="text", condition_name=cond)
            if not budget_exceeded():
                run_sft_eval(seed, ckpt, cond, all_results, eval_mode="text", train_mode="text")
            else:
                print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget (after train)")
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 4: grpo_proxy_reward_classifier
    # ─────────────────────────────────────────────────────────────────────────
    cond = "grpo_proxy_reward_classifier"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            run_grpo_proxy(seed, all_results, use_complexity_features=True,
                           condition_name=cond)
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 5: constraint_tokens_masked_at_inference
    # ─────────────────────────────────────────────────────────────────────────
    cond = "constraint_tokens_masked_at_inference"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            ckpt = sft_constraint_ckpt(seed)
            if not (ckpt / "adapter_config.json").exists():
                print(f"[{cond}] SFT constraint checkpoint for seed={seed} missing — training first")
                run_sft_training(seed, ckpt, constraint_mode="token",
                                 condition_name="qwen25coder_sft_constraint_tokens")
            if not budget_exceeded():
                # masked arm: reuse the constraint_tokens checkpoint (train_mode="token" so
                # the base gets resized on load), but feed NO constraint header (eval_mode="none").
                run_sft_eval(seed, ckpt, cond, all_results, eval_mode="none", train_mode="token")
            else:
                print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 5b: constraint_text_masked_at_inference (text-masked control)
    # Symmetric to CONDITION 5: trained WITH text header (train="text") but eval
    # WITHOUT the header (eval="none") — tests whether TEXT conditioning internalizes
    # better than tokens (H2 selling point). Reuses the constraint_text checkpoint.
    # ─────────────────────────────────────────────────────────────────────────
    cond = "constraint_text_masked_at_inference"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            ckpt = sft_constraint_text_ckpt(seed)
            if not (ckpt / "adapter_config.json").exists():
                print(f"[{cond}] SFT constraint_text checkpoint for seed={seed} missing — training first")
                run_sft_training(seed, ckpt, constraint_mode="text",
                                 condition_name="qwen25coder_sft_constraint_text")
            if not budget_exceeded():
                # text-masked arm: reuse the constraint_text checkpoint (train_mode="text",
                # NO vocab resize), but feed NO constraint header (eval_mode="none").
                run_sft_eval(seed, ckpt, cond, all_results, eval_mode="none", train_mode="text")
            else:
                print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 6: grpo_hidden_state_only_proxy (1 seed)
    # ─────────────────────────────────────────────────────────────────────────
    cond = "grpo_hidden_state_only_proxy"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            run_grpo_proxy(seed, all_results, use_complexity_features=False,
                           condition_name=cond)
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 7: grpo_oracle_reward_reference (1 seed)
    # ─────────────────────────────────────────────────────────────────────────
    cond = "grpo_oracle_reward_reference"
    for seed in get_seeds(cond):
        if budget_exceeded():
            print(f"CONDITION_SKIPPED: condition={cond} seed={seed} reason=time_budget")
            continue
        if seed in all_results.get(cond, {}):
            print(f"[main] Skipping {cond} seed={seed} (already done)")
            continue
        try:
            # GRPO branches from the NO-CONSTRAINTS SFT (isolates the geometric-reward novelty)
            sft_ckpt = sft_no_constraint_ckpt(seed)
            if not (sft_ckpt / "adapter_config.json").exists():
                print(f"[{cond}] SFT no-constraints checkpoint missing — training first")
                run_sft_training(seed, sft_ckpt, constraint_mode="none",
                                 condition_name="qwen25coder_sft_no_constraints")
            out_ckpt = grpo_oracle_ckpt(seed)
            oracle_reward = make_oracle_reward_fn()
            run_grpo(seed, sft_ckpt, out_ckpt, oracle_reward, cond, all_results)
        except Exception:
            traceback.print_exc()
            all_results.setdefault(cond, {})[seed] = {"validity_rate": 0.0, "error": "exception"}
            write_results(all_results)
    print_aggregate(cond, all_results.get(cond, {}))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPUTE VALIDITY GAP METRIC
    # ─────────────────────────────────────────────────────────────────────────
    proxy_rates = [v["validity_rate"] for v in all_results.get("grpo_proxy_reward_classifier", {}).values()
                   if isinstance(v, dict) and "validity_rate" in v]
    oracle_rates = [v["validity_rate"] for v in all_results.get("grpo_oracle_reward_reference", {}).values()
                    if isinstance(v, dict) and "validity_rate" in v]
    gap = None
    if proxy_rates and oracle_rates:
        gap = abs(np.mean(proxy_rates) - np.mean(oracle_rates)) * 100
        print(f"validity_gap_proxy_vs_oracle: {gap:.2f}pp")
        all_results["validity_gap_proxy_vs_oracle_pp"] = gap

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== FINAL RESULTS ===")
    all_validity_rates = []
    for cname, cresults in all_results.items():
        if not isinstance(cresults, dict):
            continue
        rates = [v["validity_rate"] for v in cresults.values()
                 if isinstance(v, dict) and "validity_rate" in v]
        if rates:
            mean_v = np.mean(rates)
            std_v = np.std(rates)
            all_validity_rates.extend(rates)
            af1 = [v["adherence_f1"] for v in cresults.values()
                   if isinstance(v, dict) and "adherence_f1" in v]
            af1_str = f" adherence_f1_mean={np.mean(af1):.4f}" if af1 else ""
            print(f"condition={cname}{af1_str} validity_rate_mean={mean_v:.4f} validity_rate_std={std_v:.4f}")

    overall_validity = np.mean(all_validity_rates) if all_validity_rates else 0.0
    print(f"\nvalidity_rate: {overall_validity:.4f}")

    # Write final results
    write_results({**all_results, "overall_validity_rate": overall_validity})

    elapsed_total = time.time() - RUN_START
    print(f"\n=== DONE in {elapsed_total:.0f}s ===")
    update_progress(phase="done", latest_metric=overall_validity)


if __name__ == "__main__":
    main()
