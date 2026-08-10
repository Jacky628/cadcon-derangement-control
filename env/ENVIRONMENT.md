# Environment of the runs reported in the paper

Recovered from the interpreter the campaign actually ran under
(`artifacts/rc-20260605-140117-fdedd0/stage-12/runs/sandbox/_project_1/.venv`), which is still
on disk. The four entries the frozen pre-registration also records — Python, NumPy, SciPy,
`datasets` — agree with it, so this is the environment of record and not a reconstruction.

## Core

| Component | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.5.1+cu121 |
| transformers | 4.46.3 |
| peft | 0.13.2 |
| trl | 0.14.0 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.1.1 |
| datasets | 3.1.0 |
| tokenizers | 0.20.3 |
| safetensors | 0.7.0 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| **cadquery** | **2.7.0** |
| **cadquery-ocp** | **7.8.1.1.post1** |
| CUDA runtime (torch wheel) | 12.1 |
| GPU | 2 $\times$ NVIDIA GeForce RTX 3090 (24 GB), driver 591.86 |

`requirements-frozen-full.txt` is the complete `pip freeze` (203 packages). Two entries in it,
`numpy` and `scipy`, are conda-built wheels and appear as local `file://` URLs; the versions in the
table above are what those wheels contain, and PyPI wheels of the same versions are equivalent for
everything this code does.

## Geometry stack: two versions, both verified

The geometric assertions of `geom_requirements.py` execute each program in a CadQuery subprocess,
so the CadQuery/OCP version is the one library whose behaviour could plausibly change a reported
number. It does not, across the one version step we can test:

| | Runs of record | Independent re-execution (2026-08-10) |
|---|---|---|
| cadquery | 2.7.0 | 2.8.0 |
| cadquery-ocp | 7.8.1.1.post1 | 7.9.3.1.1 |
| Result | — | executability bit and realized feature set identical on **400 / 400** generations drawn across all arms and seeds |

The re-execution was done on a machine with no attempt to match the original environment. It is
evidence that the produced-side metric is robust to this version step; it is not a claim that the
metric is version-independent in general.

## What this does *not* pin

The `main.py` in `code/` hardcodes one data root, `WORKSPACE = Path("/workspace/data/deepcad")`,
which is where the DeepCAD JSON histories were staged. Point it at your own copy of DeepCAD; nothing
else in the file assumes a particular location. Checkpoints are written to `./checkpoints`
relative to the working directory.
