# CADCON: derangement-controlled audit of design-intent conditioning in CAD program completion

Frozen artifacts for a study of what happens when a fine-tuned CAD code model is given a
*wrong* design-intent header, and of how much of a correct header's apparent benefit is an
artifact of scoring generated text without requiring it to execute.

Contents: the pre-registration documents, the frozen held-out lists and decision rules, the
complete raw per-sample geometry scores for every arm and seed, and the analysis scripts.

> **Note for reviewers.** This copy is served anonymously; the citation block and paper
> identifiers are masked by the anonymization service. A submission-stage audit of these
> artifacts found that the 100 held-out entries comprise 50 paired unique programs (76 primary
> entries / 38 unique programs), so the submitted manuscript reports duplicate-aware statistics
> over ground-truth clusters. `analysis/submission_audit.py` reproduces that correction.

Every reported number is recomputable from this release. Two verification scripts cover the
two analysis generations. `figures/compute_all.py` recomputes the original entry-level
figure/table statistics and asserts 105 values. `analysis/submission_audit.py` supersedes the
inferential part of that pass: it rebuilds unique-program clusters from the frozen ground-truth
hashes and recomputes the duplicate-aware tests, the detector/execution-gate decomposition, the
permutation and Holm checks, the cluster-representative sensitivity, and the within-pair
agreement statistics. It reads the frozen artifacts read-only and asserts the values printed in
the submitted manuscript.

## Layout

```
analysis/           Frozen analysis code and evaluation manifests
  spike_analysis.py           frozen six-cell decision rules (recall, Wilcoxon machinery)
  f4_analysis.py              frozen control protocol: competence gate + Test D  (sha256 c506ab4e…)
  geom_requirements.py        execution-level geometric assertions (the independent metric)
  spike_frozen_lists.json     frozen 100-program held-out list; 76-program analysis subset;
                              the six geometry-profile definitions
  spike_sixcell.json          the 42 six-cell evaluation jobs
  f4_eval_jobs.json           the 18 control-evaluation jobs
  f4_train_jobs.json          the 3 derangement-control training jobs
  submission_audit.py         duplicate-aware re-analysis (post-hoc; see the note above)
  submission_audit.json       its output
  sha256_manifest_frozen.txt  SHA-256 manifest recorded at freeze time

preregistration/    Pre-registration documents, frozen BEFORE the corresponding data existed
  (kept byte-identical to their freeze-time state so the manifest hashes verify;
   original documents are written in Chinese — the decision rules they freeze are
   restated in English in the paper, §3.6–3.7)

data/               Complete raw per-sample geometry scores
  spike_results/geom_scores.jsonl   4,200 rows: six-cell matrix, all arms × seeds
  f4_results/geom_scores.jsonl      1,800 rows: control evaluations
  spike_verdict.json, f4_verdict.json   frozen decision outputs
  f4_meta_seed{0,1,2}.json          control-training metadata (derangement plan, 0 header-dropout)
  sha256s_raw.txt                   manifest of this data snapshot

figures/            Figure/table generation = the verification pass
  compute_all.py              recomputes EVERYTHING from data/ and asserts 105 paper values
  make_figures.py, make_fig1.py, fig_style.py    figure scripts (read figures_data.json only)
  figures_data.json           output of compute_all.py
  verification_report.txt     the 105 PASS lines
```

## Reproduce the paper's numbers

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
cd figures
../.venv/bin/python compute_all.py     # -> 105/105 checks PASS (all paths are relative)
../.venv/bin/python make_figures.py    # -> Figures 2-4
../.venv/bin/python make_fig1.py       # -> Figure 1
cd ../analysis
../.venv/bin/python submission_audit.py  # -> duplicate-aware re-analysis, assertions pass
```

LoRA adapter checkpoints for all reported conditions will be released upon acceptance.
