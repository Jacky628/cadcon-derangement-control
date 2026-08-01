# Release: Wrong Design Intent Is Worse Than None

Artifacts for the paper *"Wrong Design Intent Is Worse Than None: A Derangement-Control
Diagnosis of Header Conditioning in CAD Program Completion"* (Yang Xiao, Sichuan University),
available at **https://arxiv.org/abs/2607.23191**.

```bibtex
@misc{xiao2026wrongintent,
  title  = {Wrong Design Intent Is Worse Than None: A Derangement-Control Diagnosis
            of Header Conditioning in CAD Program Completion},
  author = {Yang Xiao},
  year   = {2026},
  eprint = {2607.23191},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2607.23191}
}
```

Every number in the paper is recomputable from this release. The verification script
(`figures/compute_all.py`) recomputes all figure/table/appendix statistics from the raw
per-sample scores and asserts them against the values printed in the paper (105 checks).

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
```

LoRA adapter checkpoints for all reported conditions will be released upon acceptance.
