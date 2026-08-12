# CADCON: derangement-controlled audit of design-intent conditioning in CAD program completion

Frozen artifacts for a study of what happens when a fine-tuned CAD code model is given a
*wrong* design-intent header, and of how much of a correct header's apparent benefit is an
artifact of scoring generated text without requiring it to execute.

> **Note for reviewers.** This copy is served anonymously and the citation block is withheld
> for the duration of double-blind review. Section and table references throughout follow the
> submitted manuscript. Nothing else is removed: the artifacts, scripts and raw scores below are
> the complete set.

Contents: four pre-registration documents (and the public anchor of a fifth, whose
experiment is still running), the frozen held-out lists and decision rules, the
complete raw per-sample geometry scores for every arm and seed of every evaluation, the frozen
analysis scripts with their hash manifests, and the figure and table scripts.

## Two evaluation campaigns, never pooled

Everything here belongs to one of two campaigns, and the distinction matters more than any
individual file.

**The initial campaign** (2026-07) evaluated a `{0%, 40%}`-prefix × `{correct, wrong, masked}`
matrix on a 100-entry held-out list, plus a derangement control. A submission-stage audit then
found that those 100 entries comprise 50 paired unique programs — 76 primary entries over 38
unique programs after the four-feature filter — which is why that round's inference is
duplicate-aware over ground-truth clusters. `analysis/submission_audit.py` reproduces that
correction.

**The replication campaign** (2026-08) is what carries every headline claim in the current
paper. It re-runs the 40%-prefix column on a pre-registered sample of **400 deduplicated**
held-out programs stratified over all eleven four-tag intent profiles, and re-runs the
derangement control on those same 400 programs, at the same indices, with byte-identical wrong
headers. It overlaps the earlier analysis subset on 2 of 400 programs.

The two campaigns differ by construction in intent-profile composition (eleven stratified
profiles against six) and in generation-truncation regime, so **their absolute levels are not
commensurable and are never pooled or placed on a common scale**. Files are named for their
campaign throughout; anything without a `repl`/`testd` prefix belongs to the initial one.

## What is recomputable from this release

Every pre-registered decision, and the figure and table values behind them, are recomputable
from this repository alone; four descriptive tables are not. Both halves are listed below,
because a release that overstates its own coverage is worse than one that bounds it.

Verified by running the scripts below against nothing but this repository:

| Script | Output | Reproduces |
|---|---|---|
| `analysis/replication_analysis.py` | `REPLICATED` | the primary verdict, §4.1 |
| `analysis/testd_analysis.py` | `DISSOCIATION_SEMANTIC` | the causal verdict, §4.2 |
| `analysis/blinded_power_check.py` | non-zero counts only | the blinded check, §3.6 |
| `figures/compute_replication_figures.py` | `replication_figures_data.json` | the replication campaign's figure and table values |
| `figures/compute_all.py` | 105/105 PASS | the initial campaign's entry-level statistics |
| `analysis/submission_audit.py` | assertions pass | the duplicate-aware re-analysis of the initial campaign |

The first four were each checked against their frozen outputs and agree value for value.

Beyond the scored results, this release also carries the model's actual outputs. Each campaign's
`adherence_samples.jsonl` holds, per scored row, the prompt the model saw, the model's output after
one pass of the program extractor (`gen_raw` — *not* the untrimmed completion; see Appendix A.6),
the program that was scored, and the ground-truth program — so generation-level claims
can be checked directly rather than taken on the scores alone. The unique-generation counts the
paper reports for the earlier 0%-prefix column (§4.5, §6) are one such claim, and they recompute
from `data/spike_results/adherence_samples.jsonl`: counting distinct `gen_raw` per condition per
seed over the 100 nominal samples gives one for the unconditioned baseline, one or two for the
masked arms, and ten to sixteen across the four header-bearing arms.

**Not reproducible from the release**, and stated here rather than left for a reader to
discover: Appendix B's Table 11 (the five treatments of the tie-heavy difference distribution)
and Table 12 (the diff-mass decomposition) have no producing script here, and of Table 13's
four leave-one-*feature*-out rows only `all CIRCLE-containing` is in
`data/testd_sensitivity.json` — the other three were computed with the same frozen mechanics
but the script that produced them is not part of this release. The eleven leave-one-*profile*-out
rows of Table 13, which carry the pre-registered guardrail, are reproducible.

## An experiment still in progress

`preregistration/extractor-fix-2026-08-12-ANCHOR.md` records the decision rules, thresholds and
reporting obligations of a re-run that was launched on 2026-08-12 and had produced no result
when the anchor was deposited. It addresses a defect this release already documents: the
generation extractor truncates at the first top-level `result =` assignment while the scoring
target is the untruncated ground-truth program, so a model that continues into a second
modelling stage is not credited for it — and 202 of the 400 reference programs are multi-stage.

The anchor fixes in advance what each possible outcome obliges, including the two that require
limiting or withdrawing a published claim. Scoring code, verdict and run log will be added when
the run completes, whatever it shows.

## Layout

```
code/               The training and generation harness, byte-identical to what ran. This is
                    the import closure of main.py, computed by walking the import graph.
  main.py                     defines the extractor φ (`detect_features`), header construction,
                              held-out loading and train-leak deduplication, batched greedy
                              generation, the first-`result` generation extractor, and LoRA
                              fine-tuning. sha256 e2571030… matches replication_frozen/sha256s.txt.
                              It is the project's whole pipeline file and also contains the GRPO
                              study and CFG variant that Appendix D reports as supporting no
                              claim; released whole because a trimmed copy would not hash to what
                              ran.
  deepcad_transpiler.py       THE TRANSPILER: DeepCAD JSON history → CadQuery program. main.py
                              imports `deepcad_json_to_cadquery` / `extract_text_from_sample`
                              from here. Every program in the corpus came out of this file.
  build_replication_sample.py builds the deduplicated train-disjoint pool and draws the frozen
                              stratified 400-program sample — the provenance of the dataset below
  run_dualgpu.py              distributes evaluation jobs across the two GPUs
  geom_requirements.py        identical to analysis/geom_requirements.py; here so code/ stands alone
  quality_check.py            run-time condition checks imported by main.py; in the closure, but
                              no claim rests on it
  SHA256SUMS.txt              hashes of the six files above

env/                The environment the runs actually used
  ENVIRONMENT.md              Python 3.11.15, torch 2.5.1+cu121, transformers 4.46.3, peft 0.13.2,
                              cadquery 2.7.0 on OCP 7.8.1.1, 2×RTX 3090. Also records that
                              re-executing 400 frozen generations under cadquery 2.8.0 / OCP
                              7.9.3 reproduces the executability bit and realized feature set
                              on 400/400.
  requirements-frozen-full.txt  complete pip freeze of that interpreter

analysis/           Frozen analysis code and evaluation manifests
  ── initial campaign
  spike_analysis.py           frozen six-cell decision rules (recall, Wilcoxon machinery)
  f4_analysis.py              frozen control protocol: competence gate + Test D  (sha256 c506ab4e…)
  spike_frozen_lists.json     frozen 100-program held-out list; 76-program analysis subset;
                              the six geometry-profile definitions
  spike_sixcell.json          the 42 six-cell evaluation jobs
  f4_eval_jobs.json           the 18 control-evaluation jobs
  f4_train_jobs.json          the 3 derangement-control training jobs
  submission_audit.py/.json   duplicate-aware re-analysis of that campaign, and its output
  sha256_manifest_frozen.txt  SHA-256 manifest recorded at freeze time
  ── replication campaign
  replication_analysis.py     frozen primary decision rules on the 400-program sample
  testd_analysis.py           frozen retest of the control on that same sample
  blinded_power_check.py      counts non-zero pairs; contains no mean, direction or Wilcoxon path
  testd_sensitivity.py        leave-one-profile-out, the executable-only view and profile
                              weighting (App. B). It does NOT compute Table 11's five
                              zero-handling treatments; nothing here does.
  replication_frozen_lists.json  the 400 deduplicated programs, eleven four-tag profile
                              definitions, per-profile stratification quotas
  repl_p04.json               the 21 replication evaluation jobs
  replD_jobs.json             the 6 control-retest jobs
  make_testd_jobs.py          builds replD_jobs.json from the frozen sample
  bench/                      outputs of the batch-invariance measurements cited in §3.6
                              (the measurement scripts need the harness; see below)

  geom_requirements.py        execution-level geometric assertions (the independent metric),
                              byte-identical across both campaigns

preregistration/    Four documents, each frozen BEFORE the corresponding data existed
  Cspike_…20260706_FROZEN.md  initial six-cell protocol
  F4_…20260707_FROZEN.md      derangement-control protocol
  复制实验_预注册判据_20260806_FROZEN.md   replication protocol, incl. the blinded expansion rule
  复制实验_TestD增补预注册_20260806.md      retest increment: changes the analysis sample, nothing else
  (kept byte-identical to their freeze-time state so the manifest hashes verify; the originals
   are written in Chinese — the decision rules they freeze are restated in English in the
   paper, §3.5–3.6 and Appendix A)

replication_frozen/
  sha256s.txt                 the authoritative record of the code actually used at launch.
                              Supersedes the `code_sha256` field embedded in
                              replication_frozen_lists.json, which is a stale snapshot taken at
                              sampling time — see Appendix A. The frozen list was not edited,
                              since editing it would invalidate its own hash.

data/               Complete raw per-sample geometry scores, and the generations they score
  spike_results/geom_scores.jsonl        4,200 rows: six-cell matrix, all arms × seeds
  f4_results/geom_scores.jsonl           1,800 rows: initial control evaluations
  repl_results/geom_scores.jsonl         8,400 rows: 21 replication jobs × 400 programs
  testd_results/geom_scores.jsonl        2,400 rows: 6 control-retest jobs × 400 programs
  testd_analysis_input/geom_scores.jsonl 10,800 rows: the two above, concatenated. The retest
                              verdict is computed from this and from neither results directory
                              on its own.
  {spike,f4,repl,testd}_results/adherence_samples.jsonl   the same rows with the text: the
                              prompt shown to the model (`prompt`), the output of one pass
                              of the extractor (`gen_raw` — not the untrimmed completion, see
                              Appendix A.6), the scored program (`program`), and the
                              ground-truth program (`gt_code`), alongside the regex-side
                              adherence fields. 4,200 / 1,800 / 8,400 / 2,400 rows, 26 MB in
                              total, and the only files here from which generation-level claims
                              — unique-generation counts, what a given arm actually wrote —
                              can be checked. These are also the input end of the scoring chain
                              `geom_requirements.py` implements: `score` turns them into
                              `geom_scores.jsonl` and `summary` turns those into
                              `evaluation_summary.json`, so the geometry scores can be
                              regenerated rather than trusted. That path needs a CAD kernel
                              (`cadquery`), which the reproduce section's three packages do not
                              install; every verdict and figure script reads the scores directly
                              and needs no kernel.
  {spike,f4,repl,testd}_results/results.json, partial_results.jsonl, evaluation_summary.json
                              per-condition validity and adherence roll-ups written by the
                              evaluation harness. `results.json` is load-bearing rather than a
                              convenience: `figures/compute_all.py` reads the initial campaign's
                              copy for the regex-side values it asserts, and
                              `analysis/spike_analysis.py` reads it for the pre-registered
                              reproduction check against the frozen reference cells.
  repl_results/blinded_power_check.json  output of the blinded check, produced before any
                              effect direction or p-value was computed
  spike_verdict.json, f4_verdict.json, repl_verdict.json, testd_verdict.json
  testd_sensitivity.json      output of the above: Test D robustness decompositions (App. B)
  f4_meta_seed{0,1,2}.json    control-training metadata (derangement plan, 0 header-dropout)
  sha256s_raw.txt             manifest of the initial campaign's data snapshot
  replication_sample_400.jsonl  the 400 held-out programs materialised in full: idx, pool_row,
                              program_sha256, four-tag profile, five-tag intent, program length,
                              and the CadQuery source itself. Every program_sha256 is the sha256
                              of that row's gt_code and matches the frozen sample list entry for
                              the same idx — checked at generation time, 400/400. Per-profile
                              counts 112/70/58/40/39/15/15/15/15/15/6, which is paper Table 1.

figures/
  compute_all.py              recomputes the initial campaign and asserts 105 values
  compute_replication_figures.py   recomputes the replication campaign's figure/table data
  make_fig1_v4.py, make_replication_figures.py     current figure scripts (paper Figures 1-4;
                              make_replication_figures.py keeps the earlier round's stems and
                              numbering — see the correspondence below)
  make_fig1.py, make_figures.py, fig_style.py      the initial campaign's figure scripts
  figures_data.json, replication_figures_data.json  outputs of the two compute scripts
  verification_report.txt     the 105 PASS lines
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
```

Frozen at python 3.11.15 / numpy 2.4.4 / scipy 1.17.1; the replication verdicts were
reproduced on exactly those versions.

```bash
# --- replication campaign (carries every headline claim) ---

# blinded power check — the protocol requires this to run before the verdict, and forbids
# enlarging the sample after it returns
.venv/bin/python analysis/blinded_power_check.py \
    --dir data/repl_results --frozen analysis/replication_frozen_lists.json

# primary verdict
.venv/bin/python analysis/replication_analysis.py \
    --dir data/repl_results --frozen analysis/replication_frozen_lists.json \
    --prefixes p04 --out repl_verdict.json            # -> REPLICATED

# causal verdict; note the concatenated input
.venv/bin/python analysis/testd_analysis.py \
    --dir data/testd_analysis_input --frozen analysis/replication_frozen_lists.json \
    --repl-verdict repl_verdict.json --out testd_verdict.json   # -> DISSOCIATION_SEMANTIC

# figure and table data, then the figures themselves
cd figures
../.venv/bin/python compute_replication_figures.py    # -> replication_figures_data.json
../.venv/bin/python make_replication_figures.py       # -> paper Figures 2-4, written under
                                                      #    the earlier round's file stems
../.venv/bin/python make_fig1_v4.py                   # -> Figure 1
cd ..

# --- initial campaign ---
cd figures && ../.venv/bin/python compute_all.py      # -> 105/105 PASS
cd ../analysis && ../.venv/bin/python submission_audit.py
```

## What is not in this release, and why

- ~~The training and evaluation harness and the materialised 400-program dataset.~~ **Both are
  now released** — see `code/` and `data/replication_sample_400.jsonl` below. The pinned
  environment is in `env/`.
- **The pre-launch verification scripts** tabulated in Appendix A, the post-fix regression
  check, and the two batch-invariance measurement scripts. Each of them reads the live sandbox
  or loads a fine-tuned checkpoint, so none can run against this release; shipping them would
  present as reproducible something that is not. Their *outputs* are here — `analysis/bench/`
  holds the batch-invariance measurements the paper cites — and Appendix A states what each
  check returned.
- **LoRA adapter checkpoints** for all reported conditions, which will be released upon
  acceptance.

## Released copies that differ from their frozen hash

`replication_frozen/sha256s.txt` records the code as it stood at launch. Three of its entries
cover files whose copy here is not byte-identical, and it is worth being exact about why:

| File | Difference |
|---|---|
| `figures/make_fig1_v4.py` | Redrawn after the freeze — layout, and a font-embedding fix required by arXiv. The copy here is the one that produced Figure 1 as published. |
| `figures/compute_replication_figures.py` | Four path bindings, marked in the source, point at `analysis/` and `data/` instead of a flat directory. The computation is untouched. |
| `figures/make_replication_figures.py` | Three path bindings, marked in the source, replace an absolute sandbox path. The plotting code is untouched. |

`make_replication_figures.py` writes the three panels under the stems the earlier round used,
which are not the names or the numbers the paper carries. The contents match; only the labels
differ:

| stem written by the script, here | number in the paper | stem used in the paper's own tree |
|---|---|---|
| `fig3_dissociation_repl` | Figure 2 | `fig2_dissociation_v4` |
| `fig4_output_identity_repl` | Figure 3 | `fig3_output_identity_v4` |
| `fig2_seven_arms_repl` | Figure 4 | `fig4_metric_comparison_v4` |

Checked by rendering both at 110 dpi and comparing: same data, same labels, same values, with
only sub-pixel layout differences. `make_fig1_v4.py` writes Figure 1 under its own name.
This repository ships the figure *scripts*, not the figure files; the third column names the
stems the typeset paper uses, so that a reader who runs the scripts can tell which output is
which.

Every other manifest entry covering a released file verifies: 26 of 26.

The `code_sha256` field embedded inside `analysis/replication_frozen_lists.json` is a separate
matter — it is a stale snapshot taken at sampling time, superseded by
`replication_frozen/sha256s.txt`, and the frozen list was deliberately not edited because
editing it would invalidate its own hash. Appendix A records this.
