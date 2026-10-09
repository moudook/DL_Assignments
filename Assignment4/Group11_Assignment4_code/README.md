# Assignment 4 — MNIST dimension reduction, denoising and weight analysis

Group 11. CS601T, IIT Delhi.

Submission is three artefacts: `Group11_Assignment4_code.zip`, `Group11_Assignment4_report.pdf`
(built from `report/report.tex`), and this repository's code. The PDF is uploaded to Moodle
manually — it is gitignored here so the two never drift apart.

---

## One entry point

```bash
python run_all.py --outdir results_final
```

That is the whole submission, in order, about 22 minutes on a GPU. Tasks are genuinely
dependent, not merely convenient to run in sequence — Task-5a's bottleneck comes from
Task-3's winner and Task-6a is "best compressed representation in one hidden layer
autoencoder" — so the pipeline is sequential by requirement.

| Flag | Effect |
|---|---|
| `--outdir DIR` | where every artefact is written (default `results`) |
| `--only 1 2 3` | run selected tasks; dependencies are pulled in automatically |
| `--quick` | tiny epoch budget — a smoke test, meaningless numbers |
| `--max-epochs N` | override the 10,000-epoch AE budget |
| `--dims 32 64` | override the bottleneck grid |
| `--archs 3L_A 5L_A` | override the classifier grid |
| `--retrain-aes` | force the autoencoders to retrain instead of reusing Task-2's |

A `Makefile` wraps the two useful invocations:

```
make quick      # ~1 min, verifies the pipeline works, results are meaningless
make train      # the real thing, ~22 min, writes results_final/
make verify     # re-derive the headline numbers from the run's own artefacts
make report     # compile report/report.tex
```

Modules other than `run_all.py` are stages, not entry points. There is no `v2`, no
`final`, and no second copy of any script.

## Layout

```
run_all.py           the only orchestrator; everything else is a stage
data.py              dataset loading and the index/digit label mapping
pca.py               PCA fit, explained variance, transform
models.py            architectures, fixed seeds, hyperparameters
train.py             training loops, stopping rules, degeneracy detection
run_tracker.py       per-run checkpointing and progress reporting
evaluate.py          test evaluation, confusion matrices, per-class metrics
monitor.py           live run monitor (progress lines and GPU state)
pilot.py             the sweep runner that runs one config end to end
plots.py             all plotting primitives and the save + tidy helper
dataset_figures.py   the seven dataset-structure figures
summary_figures.py   cross-task comparison figures
task{1..6}_*.py      one file per assignment task
DECISIONS.md         every methodological decision and the evidence for it
report/              LaTeX source (report.tex) and template (main.tex)
```

## Results layout

`results_final/` holds one JSON per task plus `summary.json`, `selection.json`,
`run_config.json` and 66 PyTorch checkpoints. Figures live under `plots/`, numbered by
task and then by family so the tree reads in the order a reader wants it:

```
plots/
  00_dataset/                      7   dataset structure, before any model runs
  00_summary/                     7   cross-task comparisons
  task1/01_training_curves/       32   one per (dimension, architecture)
  task1/02_accuracy/               32
  task1/03_confusion_matrices/     20
  task1/07_representations/         2
  task1/08_comparisons/           10
  task2/01_training_curves/        16   ...
  task2/04_reconstructions/        32
  task2/05_reconstruction_error/   16
  task2/08_comparisons/             3
  task3/                          same taxonomy as task1
  task4/                          same taxonomy as task1
  task5/01_training_curves/        10
  task5/04_reconstructions/        12
  task6/06_weights_and_activations/ 15   5 figures x 3 variants
```

The figure set is **exploratory**, generated in full so nothing has to be re-run to check
an inkling; the report selects from it. Nothing in `plots/` is hand-made, and every figure
is reproducible from `run_all.py`.

## Headline results

| Task | Best configuration | Test accuracy | vs A3 |
|---|---|---|---|
| 1 | PCA k=32 (3L_B) | 98.2345% | −0.53 pp |
| 2 | AE k=256 | lowest reconstruction MSE | — |
| 3 | 1-hidden AE k=32 (3L_A) | 98.3399% | −0.42 pp |
| 4 | 3-hidden AE k=64 (4L_A) | 98.4980% | −0.26 pp |
| 5 | denoising AE 20% k=32 (3L_A) | 98.4980% | — |

**Every width is chosen on the test set**, as the assignment asks, which makes each
headline a maximum over four test numbers and biases it upward by +0.17 to +0.47 pp. ADR-007
records the effect size and what a validation-based choice would have given instead.

## Conventions that will bite you

**Labels are indices, not digit values.** `data["classes"] = ['0','4','5','6','7']`, so a
confusion-matrix row is a *position*: index 1 is the digit 4, index 2 is 5, index 3 is 6,
index 4 is 7. `cm[3][4]` is `6<->7`, not `5<->6`. Use `_digit(data, i)` when printing.
ADR-011 records the hour this cost.

**Autoencoders train once, in Task-2, and are reused.** That is a pure efficiency choice —
an autoencoder is deterministic given seed 42, so Task-3's "present training data to each of
the encoders built" is satisfied by the encoder Task-2 already built. `--retrain-aes` forces
retraining if you want to check.

**Task-1 accuracy falls as k rises.** 98.23% at k=32 down to 97.23% at k=256, while retained
variance rises from 0.759 to 0.979. The representations are not deteriorating; the first layer
is becoming ill-conditioned as its input dimension grows while its width stays put. Because
the four classifier architectures are held fixed across widths (ADR-008), the width axis
carries two variables at once and a difference along it is never attributable to width
alone — which is why comparisons are reported at matched k (ADR-009).

## Where the reasoning lives

`DECISIONS.md` is an architecture-decision record: sixteen entries, one per methodological
choice, each with the problem, the evidence, the fix and how it was verified. It is where
the code's *why* lives. The modules themselves are deliberately terse — per ADR-015, a
long comment is the comment most likely to contradict the code that follows it.

Highlights of what is recorded there:

- **ADR-001** why one stopping tolerance cannot serve both a cross-entropy and an MSE, and
  how a 5L_A classifier can appear converged while its loss rises.
- **ADR-003** why degeneracy is an 85% absolute floor rather than a multiple of chance —
  six runs sat between 59% and 79%, roughly four times chance, and a chance test cannot see
  any of them.
- **ADR-004** why `finish()` writes final weights: checkpoints were being compared against
  reported numbers and disagreeing by up to 0.9 pp with nothing in the artefacts to explain it.
- **ADR-011** that prototype proximity does not explain the confusions (Spearman ρ = −0.22),
  including two bugs in the code that originally measured it.
- **ADR-016** the two bugs that only surfaced once the comments were stripped, one of which
  had left the pipeline unable to run at all.

## Reproducing

```bash
make quick                                   # sanity check first
python run_all.py --outdir results_final      # the real run, ~22 min
cd report && pdflatex report.tex && pdflatex report.tex
```

All numbers in the report come from `results_final/`. The figures are regenerated every run,
so a figure and a reported number can never disagree unless the run itself changed.
