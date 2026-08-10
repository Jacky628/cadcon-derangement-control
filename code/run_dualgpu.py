#!/usr/bin/env python
"""run_dualgpu.py — dual-GPU phase-split runner for ResearchClaw CAD-LLM eval jobs (#3 enabler).

Launches independent single-(condition,seed) EVAL jobs across CUDA_VISIBLE_DEVICES=0 and =1
concurrently (one job per GPU, max 2 in flight). Each worker runs in an ISOLATED per-job dir
(its own CWD, with `checkpoints` symlinked to the sandbox) so main.py's CWD-relative writes
(partial_results.jsonl / adherence_samples.jsonl / results.json) never collide. A single
orchestrator process then MERGES each finished job's outputs into the target dir (ADD-only:
append partial + adherence rows, update results.json per-condition). Restart-safe: a job whose
(condition,seed) already has a partial row in the merge dir is skipped (resume from last finished
job). main.py is NOT modified — workers import it and call its functions.

GRPO is NOT routed here (it shards both cards via the OOM balanced map); this is SFT/eval only.

Modes:
  orchestrate (default): --jobs jobspec.json --merge-dir DIR [--runs-dir DIR] [--gpus 0,1]
                         [--cleantest-cache PATH] [--sandbox DIR]
  --worker --job job.json [--cleantest-cache PATH]      (internal; one eval job)
  --precompute-cleantest --cleantest-cache PATH         (internal; build dedup'd test cache once)

Job schema (one object per eval job in the --jobs list):
  {
    "condition": "constraint_text_prefix0.0",  # UNIQUE output label (skip key = condition+seed)
    "ckpt":      "sft_constraint_text_seed0",   # dir name under checkpoints/ (read-only)
    "seed":      0,
    "eval_mode": "text",      # header at eval -> eval_validity constraint_mode: none|text|token
    "train_mode":"text",      # checkpoint's train mode (token -> expand_vocab on load): none|text|token
    "prefix_fraction": 0.4,   # eval_prefix_fraction override
    "eval_subset_size": 100
  }
"""
import os, sys, json, time, argparse, subprocess, shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# WORKER: run ONE eval job in an isolated CWD (set by the orchestrator via Popen cwd)
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(job_path: str, sandbox: str, cleantest_cache: str | None,
               frozen_list: str | None = None):
    sys.path.insert(0, sandbox)   # so `import main` (+ deepcad_transpiler, quality_check) resolves
    job = json.loads(Path(job_path).read_text())
    import main as M

    # ── TRAIN job (Experiment B): SFT a NEW checkpoint (header-dropout) and exit ──
    # ckpt_path is relative -> resolves through the per-job `checkpoints` symlink to the real
    # sandbox checkpoints, so the new adapter persists in $P/checkpoints (ADD-only).
    if job.get("kind") == "train":
        import torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        ckpt_path = Path("checkpoints") / job["ckpt"]
        print(f"[worker:train] cond={job['condition']} seed={job['seed']} CVD={cvd} "
              f"ckpt={job['ckpt']} train_mode={job['train_mode']} "
              f"header_dropout={job.get('header_dropout',0.0)}", flush=True)
        t0 = time.time()
        M.run_sft_training(job["seed"], ckpt_path, constraint_mode=job["train_mode"],
                           condition_name=job["condition"],
                           header_dropout=job.get("header_dropout", 0.0),
                           train_header_content=job.get("train_header_content", "gt"))
        trained = (ckpt_path / "adapter_config.json").exists()
        result = {"job": job, "kind": "train", "trained": bool(trained),
                  "ckpt": str(ckpt_path.resolve()), "wall_sec": time.time() - t0,
                  "cuda_visible_devices": cvd}
        Path("job_result.json").write_text(json.dumps(result, indent=2))
        print(f"[worker:train] DONE cond={job['condition']} seed={job['seed']} "
              f"trained={trained} in {result['wall_sec']:.0f}s", flush=True)
        return 0 if trained else 1

    # Per-job HYPERPARAMETER overrides (do NOT mutate main.py source).
    M.HYPERPARAMETERS["eval_prefix_fraction"] = job["prefix_fraction"]
    M.HYPERPARAMETERS["eval_subset_size"] = job["eval_subset_size"]
    M._gen_calib_done = False                    # recalibrate the cap for THIS prefix
    M._calibrated_max_completion = None

    # Optional: skip the 329k-row train dedup by injecting the precomputed clean test set.
    if cleantest_cache and Path(cleantest_cache).exists():
        from datasets import load_from_disk
        M._clean_test_cache = load_from_disk(cleantest_cache)
        hashes = [M._program_hash(M._clean_test_cache[i])
                  for i in range(len(M._clean_test_cache))]
        # Injecting the cache bypasses _get_clean_test, so ITS internal dedup never runs on
        # these rows. A cache built before that fix silently carries the duplicate entries
        # that left the v1 sample of 100 with only 50 independent programs — check the cache
        # instead of trusting its filename.
        if M.HOLDOUT_INTERNAL_DEDUP:
            n_dup = len(hashes) - len(set(hashes))
            assert n_dup == 0, (
                f"clean-test cache {cleantest_cache} holds {n_dup} duplicate programs; rebuild "
                "it with --precompute-cleantest, or set ARC_HOLDOUT_INTERNAL_DEDUP=0 to "
                "reproduce the v1 protocol on purpose")
        # Being duplicate-free is not the same as being the RIGHT set. Nothing else in this
        # path ties the injected rows to the frozen sample: pointing --cleantest-cache at the
        # 4,641-row pool instead of the 400-row draw would run happily and produce a verdict
        # over the wrong programs. Bind the two explicitly when a frozen list is supplied.
        if frozen_list:
            frozen = json.loads(Path(frozen_list).read_text())
            want = [s["program_sha256"] for s in frozen["samples"]]
            assert hashes == want, (
                f"clean-test cache {cleantest_cache} does not match {frozen_list}: "
                f"{len(hashes)} rows vs {len(want)} frozen, "
                f"first mismatch at position "
                f"{next((i for i, (a, b) in enumerate(zip(hashes, want)) if a != b), 'n/a')}")
            assert job["eval_subset_size"] == len(want), (
                f"job eval_subset_size={job['eval_subset_size']} != frozen n={len(want)}; "
                "a smaller value would re-sample inside the worker and break idx-level pairing")
            print(f"[worker] frozen-sample binding OK ({len(want)} programs match "
                  f"{frozen_list})", flush=True)
        print(f"[worker] clean-test cache HIT ({len(M._clean_test_cache)} rows) — skipping dedup", flush=True)

    # The oracle self-test never runs in this path (main() is not entered), so a cold or
    # broken CadQuery/OCP state would surface as validity 0.000 — a plausible-looking result
    # rather than a crash. Cheap to rule out once per job.
    M.oracle_self_test()

    import torch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    ckpt_path = Path("checkpoints") / job["ckpt"]
    assert ckpt_path.exists(), f"checkpoint missing: {ckpt_path.resolve()}"
    print(f"[worker] cond={job['condition']} seed={job['seed']} CVD={cvd} ckpt={job['ckpt']} "
          f"prefix={job['prefix_fraction']} eval_mode={job['eval_mode']} "
          f"header_source={job.get('header_source','correct')}", flush=True)

    t0 = time.time()
    model, tok = M.load_qwen_fp16_for_eval(str(ckpt_path), job["seed"],
                                           expand_vocab=(job["train_mode"] == "token"))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[worker] model loaded in {time.time()-t0:.1f}s; GPU free={free/2**30:.1f}GB/"
              f"{total/2**30:.1f}GB (single-card headroom check)", flush=True)

    train_sample = M.load_deepcad("train", max_rows=50)
    max_new = M.calibrate_generation(tok, list(train_sample))
    test_data = M.load_deepcad("test", max_rows=job["eval_subset_size"])

    metrics = M.eval_validity(model, tok, test_data, max_new,
                              job["condition"], job["seed"], constraint_mode=job["eval_mode"],
                              header_source=job.get("header_source", "correct"))

    result = {"job": job, "metrics": metrics, "max_new_tokens": max_new,
              "wall_sec": time.time() - t0, "cuda_visible_devices": cvd}
    Path("job_result.json").write_text(json.dumps(result, indent=2, default=M._json_default))
    print(f"[worker] DONE cond={job['condition']} seed={job['seed']} "
          f"validity={metrics['validity_rate']:.3f} f1={metrics['adherence_f1']:.4f} "
          f"in {result['wall_sec']:.0f}s", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE: build the dedup'd held-out test set ONCE and save_to_disk (CPU-only)
# ─────────────────────────────────────────────────────────────────────────────
def run_precompute(sandbox: str, cleantest_cache: str):
    sys.path.insert(0, sandbox)
    import main as M
    t0 = time.time()
    M.load_deepcad("test", max_rows=None)         # triggers the train-dedup, fills _clean_test_cache
    n = len(M._clean_test_cache)
    if Path(cleantest_cache).exists():
        shutil.rmtree(cleantest_cache)
    M._clean_test_cache.save_to_disk(cleantest_cache)
    print(f"[precompute] clean-test ({n} rows) saved to {cleantest_cache} in {time.time()-t0:.0f}s", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR: schedule jobs onto GPU slots, merge results single-writer
# ─────────────────────────────────────────────────────────────────────────────
def _done_keys(merge_dir: Path):
    """(condition, seed) pairs already present in merge_dir/partial_results.jsonl."""
    done = set()
    pf = merge_dir / "partial_results.jsonl"
    if pf.exists():
        for line in pf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r.get("condition"), r.get("seed")))
            except Exception:
                pass
    return done


def _merge_job(outdir: Path, merge_dir: Path):
    """Single-writer merge of one finished job's outputs into merge_dir (ADD-only).
    Returns (cond, seed, metrics-or-None). TRAIN jobs have no metrics → nothing to merge."""
    res = json.loads((outdir / "job_result.json").read_text())
    job = res["job"]
    cond, seed = job["condition"], job["seed"]
    if res.get("kind") == "train" or "metrics" not in res:
        return cond, seed, None   # train job: ckpt persisted in $P/checkpoints; nothing to merge
    metrics = res["metrics"]

    # Write order matters for restartability. `partial_results.jsonl` is the resume marker
    # (_done_keys reads it), so it goes LAST: a kill between the data and the marker costs one
    # re-run whose appended rows are byte-identical (greedy decoding is deterministic at a
    # fixed batch size — measured: two runs at batch 8 agreed on all 64 programs), and
    # build_scores keys by (condition, seed, idx) so duplicates collapse harmlessly. The
    # reverse order would mark a cell done with missing or truncated data, which no downstream
    # assertion can detect — the analysis only checks that a condition is present, not that it
    # carries all n rows.

    # 1) append this job's adherence rows, then verify the count before claiming the cell
    adh = outdir / "adherence_samples.jsonl"
    n_rows = 0
    if adh.exists():
        payload = adh.read_text()
        n_rows = sum(1 for line in payload.splitlines() if line.strip())
        with open(merge_dir / "adherence_samples.jsonl", "a") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
    expected = metrics.get("total_count")
    assert expected is None or n_rows == expected, (
        f"{cond} seed={seed}: merged {n_rows} adherence rows but metrics say {expected}; "
        "refusing to mark the cell done with incomplete data")

    # 2) update results.json for THIS condition/seed only (preserve everything else)
    rj = merge_dir / "results.json"
    allr = json.loads(rj.read_text()) if rj.exists() else {}
    allr.setdefault(cond, {})[str(seed)] = metrics
    tmp = merge_dir / "results.json.tmp"
    tmp.write_text(json.dumps(allr, indent=2))
    os.replace(tmp, rj)

    # 3) resume marker LAST
    with open(merge_dir / "partial_results.jsonl", "a") as f:
        f.write(json.dumps({"condition": cond, "seed": seed, **metrics}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return cond, seed, metrics


def run_orchestrate(jobs_path, merge_dir, runs_dir, sandbox, gpus, cleantest_cache,
                    frozen_list=None):
    sandbox = Path(sandbox).resolve()
    merge_dir = Path(merge_dir).resolve(); merge_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(runs_dir).resolve(); runs_dir.mkdir(parents=True, exist_ok=True)
    # MUST be absolute: workers run with cwd=per-job dir, so a relative cache path would
    # resolve against the wrong CWD -> silent cache miss -> every worker re-runs the 91s dedup.
    if cleantest_cache:
        cleantest_cache = str(Path(cleantest_cache).resolve())
    ckpt_src = sandbox / "checkpoints"
    assert ckpt_src.exists(), f"sandbox checkpoints missing: {ckpt_src}"
    jobs = json.loads(Path(jobs_path).read_text())
    assert isinstance(jobs, list), "jobspec must be a JSON list of job objects"

    # Optional one-time clean-test precompute (skips per-worker dedup).
    if cleantest_cache and not Path(cleantest_cache).exists():
        print(f"[orch] precomputing clean-test cache -> {cleantest_cache}", flush=True)
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--precompute-cleantest", "--sandbox", str(sandbox),
                            "--cleantest-cache", cleantest_cache],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert p.returncode == 0 and Path(cleantest_cache).exists(), "clean-test precompute failed"

    done = _done_keys(merge_dir)
    pending = [j for j in jobs if (j["condition"], j["seed"]) not in done]
    skipped = len(jobs) - len(pending)
    print(f"[orch] {len(jobs)} jobs | {skipped} already-done (skip) | {len(pending)} to run | gpus={gpus}", flush=True)

    slots = {g: None for g in gpus}          # gpu -> dict(proc, job, outdir, start) or None
    idx = 0
    completed, failed = [], []
    while idx < len(pending) or any(slots.values()):
        # launch into free GPU slots
        for g in gpus:
            if slots[g] is None and idx < len(pending):
                job = pending[idx]; idx += 1
                jid = f"{job['condition']}__seed{job['seed']}".replace("/", "_")
                outdir = runs_dir / jid
                if outdir.exists():
                    shutil.rmtree(outdir)
                outdir.mkdir(parents=True)
                (outdir / "checkpoints").symlink_to(ckpt_src)
                (outdir / "job.json").write_text(json.dumps(job))
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--job", str(outdir / "job.json"), "--sandbox", str(sandbox)]
                if cleantest_cache:
                    cmd += ["--cleantest-cache", cleantest_cache]
                if frozen_list:
                    cmd += ["--frozen-list", frozen_list]
                logf = open(outdir / "worker.log", "w")
                proc = subprocess.Popen(cmd, cwd=str(outdir), stdout=logf, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
                slots[g] = {"proc": proc, "job": job, "outdir": outdir, "start": time.time(), "log": logf}
                print(f"[orch] launch GPU{g} <- {job['condition']} seed={job['seed']} (job {idx}/{len(pending)})", flush=True)

        # poll running slots
        for g in gpus:
            s = slots[g]
            if s is None:
                continue
            rc = s["proc"].poll()
            if rc is None:
                continue
            s["log"].close()
            jobdesc = f"{s['job']['condition']} seed={s['job']['seed']}"
            if rc == 0 and (s["outdir"] / "job_result.json").exists():
                cond, seed, m = _merge_job(s["outdir"], merge_dir)
                completed.append((cond, seed))
                detail = (f"validity={m['validity_rate']:.3f} f1={m['adherence_f1']:.4f}"
                          if m is not None else "TRAINED (ckpt saved)")
                print(f"[orch] [OK] GPU{g} {jobdesc} {detail} ({time.time()-s['start']:.0f}s)", flush=True)
            else:
                failed.append((s["job"]["condition"], s["job"]["seed"]))
                print(f"[orch] [FAIL] GPU{g} {jobdesc} rc={rc} (see {s['outdir']}/worker.log)", flush=True)
            slots[g] = None
        time.sleep(2)

    print(f"\n[orch] SUMMARY: completed={len(completed)} failed={len(failed)} skipped={skipped}", flush=True)
    if failed:
        print(f"[orch] FAILED jobs: {failed}", flush=True)
    return 0 if not failed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--precompute-cleantest", action="store_true")
    ap.add_argument("--job")
    ap.add_argument("--jobs")
    ap.add_argument("--merge-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--sandbox", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--cleantest-cache")
    ap.add_argument("--frozen-list",
                    help="replication_frozen_lists.json; binds the injected cache to the frozen sample")
    a = ap.parse_args()

    if a.worker:
        sys.exit(run_worker(a.job, a.sandbox, a.cleantest_cache, a.frozen_list))
    if a.precompute_cleantest:
        sys.exit(run_precompute(a.sandbox, a.cleantest_cache))
    assert a.jobs and a.merge_dir, "orchestrate needs --jobs and --merge-dir"
    runs_dir = a.runs_dir or str(Path(a.sandbox) / "_dualgpu_runs")
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    frozen_list = str(Path(a.frozen_list).resolve()) if a.frozen_list else None
    sys.exit(run_orchestrate(a.jobs, a.merge_dir, runs_dir, a.sandbox, gpus, a.cleantest_cache,
                             frozen_list))


if __name__ == "__main__":
    main()
