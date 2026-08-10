#!/usr/bin/env python
"""build_replication_sample.py — draw and FREEZE the replication held-out sample.

Produces, from the published DeepCAD test split:
  1. `_cleantest_dedup/`  — the train-disjoint held-out pool with internal duplicates
                            collapsed (one row = one unique transpiled program)
  2. `_repl_sample_N/`    — the stratified draw the 42 eval jobs will run on, saved as a
                            dataset so every job sees byte-identical rows in one order
  3. `replication_frozen_lists.json` — the audit record: per-idx program sha256 and profile,
                            per-profile index lists (criterion 4), quota, seed, pool census,
                            library versions and the sha256 of every script involved

Nothing here touches a model; it is pure CPU bookkeeping meant to run BEFORE the run is
frozen. Re-running with the same seed reproduces the same draw bit-for-bit.

Usage:
  .venv/bin/python build_replication_sample.py [--n 400] [--seed 20260805]
                                               [--min-per-profile 15] [--out-prefix _repl_sample]
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent
sys.path.insert(0, str(SANDBOX))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import main as M
from datasets import load_from_disk


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def tag_name(profile) -> str:
    return "+".join(t.strip("<>") for t in profile) or "EMPTY"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--min-per-profile", type=int, default=15)
    ap.add_argument("--out-prefix", default="_repl_sample")
    ap.add_argument("--pool-cache", default="_cleantest_dedup")
    ap.add_argument("--frozen", default="replication_frozen_lists.json")
    ap.add_argument("--legacy-cache", default="_cleantest_cache",
                    help="v1 clean-test cache, used only as an independent cross-check")
    a = ap.parse_args()

    assert M.HOLDOUT_INTERNAL_DEDUP, "ARC_HOLDOUT_INTERNAL_DEDUP=0 — refusing to freeze a sample "
    assert M.TEST_SAMPLING == "stratified", "ARC_TEST_SAMPLING=head — refusing to freeze a sample"

    # ── 1. clean, internally-deduped held-out pool ───────────────────────────
    t0 = time.time()
    raw = M.hf_load_dataset("wanhin/deepcad-completion-sft", cache_dir=str(M.WORKSPACE),
                            trust_remote_code=True)
    n_test_split = len(raw["test"])
    pool = M.load_deepcad("test", max_rows=None)
    print(f"[pool] {n_test_split} published test rows -> {len(pool)} unique held-out programs "
          f"({time.time()-t0:.0f}s)", flush=True)

    pool_hashes = [M._program_hash(pool[i]) for i in range(len(pool))]
    assert len(set(pool_hashes)) == len(pool_hashes), "pool still contains duplicate programs"

    # Cross-check against the frozen v1 cache: deduping THAT must give the same pool. Catches a
    # silently changed upstream dataset or transpiler as well as a bug in the dedup itself.
    legacy = Path(a.legacy_cache)
    if legacy.exists():
        old = load_from_disk(str(legacy))
        kept = M._dedup_within_holdout(old, list(range(len(old))))
        old_hashes = [M._program_hash(old[i]) for i in kept]
        assert old_hashes == pool_hashes, (
            f"pool mismatch vs {a.legacy_cache}: {len(old_hashes)} vs {len(pool_hashes)} programs")
        print(f"[pool] cross-check against {a.legacy_cache}: identical ({len(kept)} programs)",
              flush=True)

    profiles = [M.profile_of_sample(pool[i]) for i in range(len(pool))]
    pool_hist = Counter(profiles)
    n_nonempty = sum(c for p, c in pool_hist.items() if p)

    pool_dir = Path(a.pool_cache)
    if not pool_dir.exists():
        pool.save_to_disk(str(pool_dir))
        print(f"[pool] saved -> {pool_dir}", flush=True)

    # ── 2. stratified draw ───────────────────────────────────────────────────
    sample = M.stratified_test_sample(pool, a.n, seed=a.seed,
                                      min_per_profile=a.min_per_profile,
                                      nonempty_intent=True)
    meta = M.LAST_TEST_SAMPLE_META
    assert len(sample) == a.n, f"drew {len(sample)} rows, expected {a.n}"
    assert len(set(meta["program_sha256"])) == a.n, "drawn sample contains duplicate programs"

    out_dir = Path(f"{a.out_prefix}_{a.n}")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists — refusing to overwrite a frozen sample")
    sample.save_to_disk(str(out_dir))
    print(f"[sample] saved -> {out_dir}", flush=True)

    # ── 3. freeze record ─────────────────────────────────────────────────────
    rows, by_profile = [], defaultdict(list)
    for i in range(len(sample)):
        s = sample[i]
        code, _ = M.sample_to_cadquery_code(s)
        feats = M.detect_features(code)
        prof = tuple(t for t in M._TAGS_PRIMARY_MAIN if t in feats)
        assert prof, f"idx {i} drawn with empty 4-tag intent"
        rows.append({
            "idx": i,
            "pool_row": meta["source_indices"][i],
            "program_sha256": meta["program_sha256"][i],
            "profile": list(prof),
            "intent_5tag": sorted(feats),
            "program_chars": len(code),
        })
        by_profile[tag_name(prof)].append(i)

    import numpy, scipy, datasets as _datasets
    frozen = {
        "protocol": "replication-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool": {
            "published_test_rows": n_test_split,
            "train_disjoint_rows": len(load_from_disk(str(legacy))) if legacy.exists() else None,
            "unique_programs": len(pool),
            "nonempty_4tag_intent": n_nonempty,
            "profile_pool_sizes": {tag_name(p): c for p, c in
                                   sorted(pool_hist.items(), key=lambda kv: -kv[1]) if p},
            "empty_4tag_intent": pool_hist.get((), 0),
        },
        "sampling": {
            "policy": meta["policy"],
            "n": a.n, "seed": a.seed, "min_per_profile": a.min_per_profile,
            "nonempty_intent_only": True,
            "per_profile": meta["per_profile"],
            "dataset_dir": str(out_dir),
            "pool_dir": str(pool_dir),
        },
        "samples": rows,
        "profiles_4tag": [{"profile": name, "idx": idxs}
                          for name, idxs in sorted(by_profile.items())],
        "analysis": {
            "unit": "one unique transpiled program per idx (no clustering required)",
            "n": a.n,
            "alpha": 0.05,
            "nonzero_guardrail_fraction": 0.15,
            "nonzero_guardrail_count": int(round(0.15 * a.n)),
            "wilcoxon": {"zero_method": "pratt", "method": "asymptotic",
                         "alternative_primary": "less", "alternative_veto": "greater"},
            "pairing": "same seed, paired by idx",
        },
        "environment": {
            "python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "datasets": _datasets.__version__,
        },
        "code_sha256": {name: sha256_file(SANDBOX / name) for name in
                        ("main.py", "run_dualgpu.py", "geom_requirements.py",
                         "build_replication_sample.py")},
    }
    Path(a.frozen).write_text(json.dumps(frozen, indent=1, ensure_ascii=False) + "\n")

    print(f"\n[freeze] {a.frozen}")
    print(f"  pool             : {len(pool)} unique programs "
          f"({n_nonempty} with non-empty 4-tag intent)")
    print(f"  drawn            : {a.n} programs across {len(by_profile)} profiles")
    print(f"  nonzero guardrail: >= {frozen['analysis']['nonzero_guardrail_count']} per seed")
    for name, idxs in sorted(by_profile.items(), key=lambda kv: -len(kv[1])):
        print(f"    {name:24s} {len(idxs):4d}")
    print(f"\n  sha256({a.frozen}) = {sha256_file(Path(a.frozen))}")
    print(f"  sha256(sample dir listing) = "
          f"{hashlib.sha256(''.join(meta['program_sha256']).encode()).hexdigest()}")


if __name__ == "__main__":
    main()
