# CADCON: derangement-controlled audit of design-intent conditioning in CAD program completion

Frozen artifacts for a study of what happens when a fine-tuned CAD code model is given a
*wrong* design-intent header, and of how much of a correct header's apparent benefit is an
artifact of scoring generated text without requiring it to execute.

**Paper:** *Wrong Design Intent Is Worse Than None: A Derangement-Control Diagnosis of Header
Conditioning in CAD Program Completion* — Yang Xiao, Sichuan University
(`yangx64519@gmail.com`), [arXiv:2607.23191](https://arxiv.org/abs/2607.23191).

Contents: four pre-registration documents, the frozen held-out lists and decision rules, the
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

## Every reported number is recomputable from this release

Verified by running the scripts below against nothing but this repository:

| Script | Output | Reproduces |
|---|---|---|
| `analysis/replication_analysis.py` | `REPLICATED` | the primary verdict, §4.1 |
| `analysis/testd_analysis.py` | `DISSOCIATION_SEMANTIC` | the causal verdict, §4.2 |
| `analysis/blinded_power_check.py` | non-zero counts only | the blinded check, §3.6 |
| `figures/compute_replication_figures.py` | `replication_figures_data.json` | Figures 2–4 and the replication tables |
| `figures/compute_all.py` | 105/105 PASS | the initial campaign's entry-level statistics |
| `analysis/submission_audit.py` | assertions pass | the duplicate-aware re-analysis of the initial campaign |

The first four were each checked against their frozen outputs and agree value for value.

## Layout

```
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
  testd_sensitivity.py        five treatments of the tie-heavy difference distribution (App. B)
  replication_frozen_lists.json  the 400 deduplicated programs, eleven four-tag profile
                              definitions, per-profile stratification quotas
  repl_p04.json               the 21 replication evaluation jobs
  replD_jobs.json             the 6 control-retest jobs
  make_testd_jobs.py          builds replD_jobs.json from the frozen sample
  verify_*.py                 the pre-launch checks tabulated in Appendix A
  regress_after_0806.py       post-fix regression of the evaluation path
  bench/                      the batch-invariance measurements cited in §3.6

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

data/               Complete raw per-sample geometry scores
  spike_results/geom_scores.jsonl        4,200 rows: six-cell matrix, all arms × seeds
  f4_results/geom_scores.jsonl           1,800 rows: initial control evaluations
  repl_results/geom_scores.jsonl         8,400 rows: 21 replication jobs × 400 programs
  testd_results/geom_scores.jsonl        2,400 rows: 6 control-retest jobs × 400 programs
  testd_analysis_input/geom_scores.jsonl 10,800 rows: the two above, concatenated. The retest
                              verdict is computed from this and from neither results directory
                              on its own.
  repl_results/blinded_power_check.json  output of the blinded check, produced before any
                              effect direction or p-value was computed
  spike_verdict.json, f4_verdict.json, repl_verdict.json, testd_verdict.json
  testd_sensitivity.json      zero-handling robustness (Appendix B)
  f4_meta_seed{0,1,2}.json    control-training metadata (derangement plan, 0 header-dropout)
  sha256s_raw.txt             manifest of the initial campaign's data snapshot

figures/
  compute_all.py              recomputes the initial campaign and asserts 105 values
  compute_replication_figures.py   recomputes the replication campaign's figure/table data
  make_fig1_v4.py, make_replication_figures.py     current figure scripts (paper Figures 1–4)
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

# figure and table data
.venv/bin/python figures/compute_replication_figures.py \
    --repl-dir data/repl_results --testd-dir data/testd_analysis_input

# --- initial campaign ---
cd figures && ../.venv/bin/python compute_all.py     # -> 105/105 PASS
cd ../analysis && ../.venv/bin/python submission_audit.py
```

## What is not in this release, and why

- **The training and evaluation harness** (`main.py`, `run_dualgpu.py`, the DeepCAD
  transpiler). These appear in the hash manifests, so the code that produced the data is
  identified and verifiable by hash, but the harness itself is not distributed here. What is
  released is everything needed to recompute the reported numbers from the raw per-sample
  scores.
- **The materialised 400-program dataset.** The programs are identified in
  `replication_frozen_lists.json`, which carries the sample identities, the profile
  definitions and the draw quotas.
- **Throughput benchmarks and smoke tests**, which support no reported number.
- **LoRA adapter checkpoints** for all reported conditions, which will be released upon
  acceptance.

One manifest entry is superseded rather than stale: `make_fig1_v4.py` was redrawn after the
freeze (layout, and a font-embedding fix for arXiv), so its hash in `replication_frozen/
sha256s.txt` records the pre-redraw version. The script in this repository is the one that
produced Figure 1 as published. Every other entry covering a released file verifies.
