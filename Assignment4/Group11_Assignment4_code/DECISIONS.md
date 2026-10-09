# Architecture Decision Record

Every decision here was forced by a specific, measured problem. Each entry names the
problem, the evidence, the fix, and how it was verified — so a reader can check the
reasoning rather than take it on trust. Where an earlier decision turned out to be
wrong, that is recorded in the entry that superseded it.

This document is the reason the code can be terse. The `#` and `"""` text inside the
modules is limited to what is needed to read them; the *why* lives here.

---

## ADR-001 — Stopping criteria: one absolute tolerance cannot serve two loss scales

**Status:** Accepted, superseding the inherited rule.

**Problem.** Assignment 3's rule — stop when the single-epoch change in training loss
falls below `1e-4` — was carried over unchanged. Applied to the autoencoders it
stopped them far short of convergence. Replaying full traces from the shipped
checkpoints: at `1hidden k=256` it fired at epoch 186 with test MSE `0.0181`, while a
relative threshold reached `0.0023` at epoch 1030 and continuing to 12,000 epochs
reached `0.00195`.

The cause is the loss scales. A `1e-4` delta on a mean squared error of order `1e-3`
is a large relative step; the same delta on a cross-entropy of order `1.6` is `6e-5`
of it and is numerically indistinguishable from optimiser noise. One tolerance cannot
be tight enough for one and loose enough for the other.

**A second, subtler failure.** A single-epoch difference also cannot distinguish
settlement from oscillation. On this dataset the 5L_A classifier stalls at loss
`1.6089` at epoch 13 and then rises monotonically through `1.6091, 1.6096, 1.6102,
1.6108, 1.6111`. Adam's momentum still produces one-epoch `|dL|` values around `1e-5`
inside that rise, all under tolerance. Verified in `task1_pca32_5L_A.pt`, epochs
13-19, `|dL| = 1.2e-5` at epoch 19. The naive rule certifies convergence while the
model is going backwards.

**Decision.**
- Classifiers keep the mandated absolute `1e-4` rule, but require it to hold for
  `tau = 15` *consecutive* epochs. A genuine minimum sustains the difference; an
  oscillation does not.
- Autoencoders use a **relative plateau** rule instead: stop when the monotone best
  loss fails to improve by more than `1e-3` in relative terms across a window of
  `50` epochs. The window is immune to the epoch-to-epoch jitter of the masked
  corruption, which is what makes it usable for the Task-5 denoisers.
- A budget of `10,000` epochs bounds every run.

**Verification.** All ten autoencoder and denoising runs stop on the plateau rule
between epochs 1,946 and 7,835, none hitting the cap. Replaying `_ConvergenceWatch`
verbatim on each checkpoint's stored history reproduces the reported stopping epoch
for all 66 runs. See ADR-002 for what this rule does *not* guarantee.

---

## ADR-002 — "Converged" is a weaker claim than the word suggests, and must be labelled

**Problem.** The plateau rule of ADR-001 fires when the best loss *so far* has stopped
improving. It says nothing about the current loss, which may still be rising. The
one-hidden `k=32` run reaches its best training loss of `0.0085233` near epoch 3,459
and stops at epoch 3,470 with `0.0085397` — still climbing when the rule fires.

**Decision.** The report states the rule's actual guarantee, and reserves the word
"converged" only where the evidence supports it. The claim recorded is "the best
reconstruction attained has been static for 50 epochs", not "training has converged".

---

## ADR-003 — Degeneracy must be judged by an absolute floor, not by chance

**Problem.** The original guard flagged a run at or below `1.25 x chance`, which for
five classes is 25%. An independent audit found the band this misses: six runs of the
5L_A architecture settled at 58.81, 59.63, 77.87, 77.87, 78.34 and 79.45 percent
validation accuracy — roughly three to four times chance — and all six were recorded
as `"converged"`. A chance-level test cannot see them because none is at chance.

**Decision.** `DEGENERATE_BELOW = 0.85` — an absolute validation-accuracy floor. It is
calibrated to the two clusters actually observed on this dataset: genuine results at
97-99 percent, 5L_A failures at 58.8-79.5 percent. A fixed multiple of chance cannot
separate them because 79% is already nearly four times chance.

**Cost of the choice.** The threshold is dataset-specific rather than principled. It is
stated as such in the report. `MIN_EPOCHS_BEFORE_STOP = 200` and `CHANCE_ACC = 0.25`
were removed as dead code once the floor was in place.

**Verification.** Flagged runs all carry `terminal_state: "degenerate"`, never
`"converged"`. No run above 85% is flagged.

---

## ADR-004 — Checkpoints must be verifiable against the numbers they describe

**Problem.** Checkpoints were written every `CHECKPOINT_EVERY = 10` epochs and never
again at the end of a run. The final model was therefore 0-9 epochs ahead of anything
on disk. An independent audit that reloaded `task4_k64_4L_A.pt` and re-evaluated it
got 98.52% where `task4.json` reported 98.50% — with nothing in the artifacts to
explain the gap. For a marker trying to verify the work, the checkpoint is the only
handle available, and it disagreed with the reported result.

**Decision.** `RunTracker.finish()` writes the final weights, with model, optimiser,
run identity metadata and convergence-watch state, before closing. The checkpoint on
disk is now exactly the model the results JSON describes.

**Verification.** All 66 checkpoints reproduce their reported metric and epoch count:
Task-4 accuracy to `0.00e+00`, autoencoder MSE to `1.8e-10`. `ckpt["epoch"] + 1 ==
epochs_run` for every run.

---

## ADR-005 — The convergence-watch state is part of a checkpoint, not decoration

**Problem.** `_ConvergenceWatch` serialised only its patience streak. A run resumed
from a checkpoint therefore rebuilt the watch with the *constructor's* defaults —
`mode="abs"`, `window=15` — and an empty best-loss history. Two consequences: the
plateau test was silently disabled, because an empty deque never satisfies
`len(hist) > window`, and if it had run, the wrong rule would have been applied. This
was a live correctness bug, not a latent one: it fires on any resume.

**Decision.** `state_dict()`/`load_state_dict()` now round-trip `mode`, `window`,
`min_epochs`, `triggered` and the windowed best-loss history. `load_state_dict`
prefers the saved values over the constructor's, so a resumed run keeps the criterion
it was actually trained under.

**Verification.** Resuming into a watch built with deliberately wrong defaults
restores `plateau`/`50`/51-entry history.

---

## ADR-006 — Denoising must be measured against the copy baseline

**Problem.** A denoising autoencoder's clean-input reconstruction error is a fine
number and measures the wrong thing. The objective's input is corrupted and its target
is clean, so the quantity that demonstrates denoising is the model's error when *fed
the corrupted input* — compared against the error of simply copying that corrupted
input through, which is what a model that learned nothing would achieve.

**Decision.** Both are computed and recorded: `corrupted_input_error` and
`copy_baseline`. The report quotes the margin rather than the absolute error.

**Result.** At 20% corruption the model reaches `0.013577` against a baseline of
`0.022842`, i.e. 59% of the error of copying. At 40%: `0.017821` against `0.045832`,
39%. Both margins are substantial. A control check confirms the margin comes from the
denoising objective: the *plain* autoencoder fed a corrupted input beats the copy
baseline by only 12.4% at 20% and 1.2% at 40%.

---

## ADR-007 — A maximum over test scores must be labelled, not quietly reported

**Problem.** Assignment 4 asks which reduced representation classifies best, and the
natural reading is the highest test accuracy. That makes each headline a maximum over
four test numbers, which is optimistically biased by construction: even if all four
widths were equivalent, the luckiest would still win.

**Decision.** Report the bias and the alternatives beside the headline: the mean across
widths, and what a validation-based choice would have given.

**Result.** The bias is +0.17 to +0.47 pp — the same order as the effects the report
discusses. Two details matter more than the numbers. Validation agrees with the
test-based choice in Tasks 1 and 4. Task 3 is the exception: validation picks `k=64`
(test 98.08%), so the reported `k=32` at 98.34% is 0.26 pp above what a selection rule
that never looked at the test set would have found.

---

## ADR-008 — Architecture widths are held fixed, which makes the width axis two variables

**Problem.** A4's central question is which reduced representation classifies best.
If each width `k` were given architectures rescaled to it, a difference between
widths would be confounded with a difference in classifier capacity.

**Decision.** The four architectures are held fixed across all input dimensions, so
every width faces identical capacity.

**Consequence, recorded because it is easy to misread.** Task 1's accuracy *falls* as
`k` rises — 98.23% at `k=32` to 97.23% at `k=256` — while retained variance rises from
0.759 to 0.979. The representations are not deteriorating; the first layer is becoming
ill-conditioned as its input dimension grows while its width stays put. The width axis
of every figure in the submission carries two variables at once, so a difference along
it is never attributable to width alone. This is why ADR-009 exists.

---

## ADR-009 — Report differences at matched k, not differences between maxima

**Problem.** Comparing each task's *best* confounds two things: representation quality
and the capacity artefact of ADR-008. Since both curves fall with `k`, and the linear
one falls faster, plotting raw accuracies makes the nonlinear code look better the
wider it gets — when most of the widening gap is PCA degrading.

**Decision.** Where representation quality is the claim, plot the *difference at
matched k* (`summary_width_vs_gain.png`). Where maxima are compared, state that both
are falling and attribute the change in gap accordingly.

**Result.** At matched k the one-hidden code beats PCA at all four widths, but by very
uneven margins: +0.11 pp at 32, +0.05 at 64, +0.50 at 128, +0.95 at 256. The
comparison that isolates representation quality is at small k — 0.05 to 0.11 pp. The
honest summary is that the nonlinear code is better but not much better.

---

## ADR-010 — The deeper encoder helps; the deeper *classifier* is what fails

**Status:** Accepted, reversing the report's original claim.

**Problem.** The first draft of the report claimed that depth did not help and that the
deeper encoder family was worse. The data say otherwise, and the original claim came
from reading the maxima rather than the matched widths.

**Evidence.** The three-hidden code beats the one-hidden code at three of four widths:
+0.08 pp at k=32, +0.42 at k=64, +0.08 at k=128, losing only at k=256 (−0.37 pp). Its
best configuration, 98.50% at k=64, is the highest measured anywhere in the assignment.
At k=64 and k=128 the three-hidden encoder beats the one-hidden encoder at *every*
classifier depth including 5L_A — 97.92 vs 77.87 and 98.08 vs 59.63 — so the deeper
encoder not only does not aggravate the deep-classifier failure, it rescues two runs
the one-hidden encoder loses entirely.

All six degenerate runs are 5L_A *classifier* failures. Not one is an encoder failure.
The 5L_A architecture is healthy at k=32 (98.00%) and k=128 (98.08%) on the same
three-hidden encoder that collapses at k=256 (78.34%).

**Decision.** The report attributes the failures to the deepest classifier and records
that the deeper encoder helps while the bottleneck stays narrow.

---

## ADR-011 — Prototype proximity does not explain the confusion structure

**Status:** Accepted, replacing an unsupported claim.

**Problem.** The report explained the dominant 5<->6 confusion geometrically — "a 5 and
a 6 differ mainly in the curvature of the lower bowl". That is a plausible story with
no measurement behind it. It is the kind of explanation that sounds like evidence and
is not.

**Evidence.** Task-1's best test model (k=32, 3L_B) against the train-set class
prototypes. Spearman rank correlation between prototype proximity and observed
confusion mass:

| pair | prototype dist | rank (1=closest) | confusions | rank (1=most confused) |
|---|---|---|---|---|
| 4<->7 | 3.909 | 1 | 12 | 2 |
| 4<->5 | 4.016 | 2 | 2 | 9.5 |
| 4<->6 | 4.200 | 3 | 5 | 6.5 |
| 5<->6 | 4.287 | 4 | 16 | 1 |
| 5<->7 | 4.365 | 5 | 8 | 3.5 |
| 0<->5 | 4.486 | 6 | 5 | 6.5 |
| 6<->7 | 5.574 | 7 | 3 | 8 |
| 0<->6 | 5.587 | 8 | 8 | 3.5 |
| 0<->4 | 6.159 | 9 | 2 | 9.5 |
| 0<->7 | 6.274 | 10 | 6 | 5 |

**Spearman rho = -0.22** (tie-aware, average ranks). The sign is the opposite of what
proximity would predict: the closest prototypes are confused *less*, not more. The
nearest pair, 4<->7, is only the second-most-confused; the most-confused pair, 5<->6,
is only fourth-nearest; and 0<->6 — the eighth-nearest pair — is tied for third-most
confused. Of 67 off-diagonal errors, 5<->6 accounts for 16 (23.9%).

Only Task-1 records a confusion matrix, so this is a single-model statistic, not an
average over representations. It is nonetheless the most conservative case: Task-1 is
the weakest representation, so if the ordering were a property of the data rather than
of model capacity it would show up here.

**Correction recorded.** An earlier draft of this document reported rho = +0.33 from a
mean across the three classification tasks. That number is not reproducible — Tasks 3,
4 and 5 do not store confusion matrices at all, so no such mean can be computed. It is
replaced here by the verified single-model figure.

**Decision.** The report states that the dominant confusion is not explained by
prototype proximity, and that the surviving claim is the weaker one: 5<->6 dominates,
and no representation moves error mass onto a pair the raw-input classifier did not
already confuse. Whether the ordering reflects structure a linear prototype cannot see
is an open question, not an established mechanism.

**Bug found while verifying.** The figure that displays this statistic originally
computed Spearman on *ordinal* ranks produced by `enumerate` over a sorted list. With
three pairs tied on confusion mass, that assigned them arbitrary distinct ranks and
made rho depend on sort order rather than on the data — it printed +0.14, and the tie
break alone was enough to flip the sign. The fix passes the raw values to the tie-aware
`_spearman` and reserves the ordinal ranks for the scatter plot, where arbitrary
tie-breaking is harmless.

A second bug in the same expression: `obs[i, j] + obs[j, i]` was used for the pair's
confusion mass, but `_observed_confusion_masses` already symmetrises, so every pair was
counted twice — 134 instead of 67. Uniform doubling leaves the ranks and therefore
`rho` unchanged, which is why the bug was invisible in the headline statistic and only
showed up in the absolute counts. Fixed to use `obs[i, j]`.

There is a third lesson here about the labels, and it is the kind that costs hours.
`data["classes"] = ['0','4','5','6','7']` means the confusion matrix rows are
**ImageFolder indices, not digit values**: index 1 is the digit 4, index 2 is 5, index 3
is 6, index 4 is 7. Reading `cm[3][4]` as "5<->6" gives 3 errors when the true answer is
`cm[2][3] = 16`. Every figure that prints digit labels must map through the index, and
`_digit(data, i)` exists for exactly this reason.

---

## ADR-012 — Two figures were silently broken for two different reasons

**Status:** Fixed and verified visually.

**Problem.** `summary_accuracy_vs_dimension.png` drew all three curves flat at about
1 on a 0-100 axis. `summary_accuracy_bars.png` drew bars of height 32, 64, 128, 256.

**Causes.**
1. The curves were passed as fractions (`0.9823`) into a figure whose axis is labelled
   "accuracy (%)" and whose reference line is `100 * A3_TEST_ACC`. The x100 was missing
   at every call site.
2. `dimension_bars` received a `{dimension: accuracy}` mapping, but matplotlib uses a
   dict's *keys* as bar heights when handed one directly — so the bars were drawn at
   32/64/128/256 units.

**Decision.** Convert to percent at the point where the figure is built, once, with a
comment naming the axis it feeds. Harden `dimension_bars` to normalise mappings itself
and to raise on a length mismatch, so no caller can reproduce the bug by accident.

**Verification.** Both figures were inspected as images after regeneration, not just
checked for successful execution.

---

## ADR-013 — The Task-5 "trade-off" does not exist at the levels tested

**Status:** Accepted, reversing the report's original claim.

**Problem.** The report stated that denoising "trades discriminative accuracy for
robustness", and that a large corruption fraction "eventually pays for that robustness
with discriminative accuracy".

**Evidence.** Both denoising models *gain* accuracy over the plain one-hidden code at
the same width: +0.16 pp at 20% corruption, +0.08 pp at 40%. The margin shrinks in the
expected direction but never turns negative at either level tested. What does degrade,
and steeply, is clean-input reconstruction: 0.009346 (plain) to 0.015053 (20%) to
0.028403 (40%).

**Decision.** The report states that the apparent trade-off is between two *evaluation
conditions* — clean input versus corrupted input — not between robustness and
discriminative power, and that no accuracy cost was observed at the levels tested.

---

## ADR-014 — One entry point per artefact; the template is not a second report

**Problem.** Best-practice guidance is explicit that `v2`, `final` and duplicate
artefacts are noise. The project had `main.tex` (with placeholder macros) and
`report.tex` (filled), which is exactly that smell: a reader cannot tell at a glance
which is the deliverable.

**Decision.** `report.tex` is the single source of truth and compiles to
`report.pdf`. The `.RESULT` macro definitions stay in the preamble so a future run can
be dropped back in, and the README records how. `main.tex` is kept only as the
untouched template and is named in the README as such.

---

## ADR-015 — Comments explain why; docstrings explain the interface

**Problem.** The codebase carried roughly 6,500 lines with ~318 hash-comments and ~239
docstring lines, much of it narrating what the next few lines obviously do.

**Decision.**
- **Docstrings** follow PEP 257 / the Google style: a one-line imperative summary, a
  blank line, then Args/Returns only where the signature does not make them obvious,
  and a note on non-obvious invariants.
- **Comments** are reserved for *why* — measured evidence, non-obvious constraints,
  decisions that would look like bugs otherwise. They are capped at a few lines and
  point at the ADR that carries the full reasoning.
- **Narrative** that records a problem and its fix moves into this file, where it can
  be read without reading code.

PEP 8 states the reason plainly: comments that contradict the code are worse than no
comments. Long comments are the ones most likely to drift, so the code keeps as few as
possible and this file keeps the history.

---

## ADR-016 — Stripping the comments surfaced two latent bugs the comments were hiding

**Status:** Completed, with two corrective fixes.

**Context.** With ADR-015 as the record, roughly 626 comment lines were removed from
6,448 lines of source, leaving 5,822. The removal is done with `tokenize`, not regex,
so a `#` inside a docstring, f-string or string literal is never mistaken for a comment;
docstrings are string tokens and survive untouched, as do the CLI usage examples inside
the module docstrings of `run_all.py`, `monitor.py` and `pilot.py`. All 17 modules were
then re-imported and the whole pipeline re-run end to end.

**Two genuine bugs surfaced, both introduced by the figures pass:**

1. **`run_all.py` would not run at all.** The `try/except` wrapping the summary-figure
   stage had its `except` at 7 spaces instead of 8, and the second `try` one level too
   deep. Python rejected the file with `IndentationError: unindent does not match any
   outer indentation level`. The earlier successful run predated the edit, so the
   artifacts on disk were fine and nothing appeared wrong — but the pipeline was dead
   and would have failed at grading.

2. **`dataset_figures.class_mean_distances` raised on every call with confusion data.**
   The panel guard read `if obs:` where `obs` is a NumPy array, which is always
   ambiguous. This failure was invisible because `run_all.py` wraps the redraw in
   `try/except` and only prints `[WARN]` — so a broken figure looked like a skipped
   optimisation rather than an error.

**Decision.** Fix both, and keep the `[WARN]` swallow only where loss is genuinely
tolerable. The confusion-panel redraw is not such a case: it silently degrades a
submitted figure while reporting success.

**Lesson recorded.** A pipeline that runs and a pipeline that works are different
things. Exit code 0 with a swallowed exception is indistinguishable from success to
anyone reading the log, so verification has to read the artifacts rather than the exit
status.

---

## ADR-017 — A verifier that cannot fail is worse than no verifier

**Status:** Accepted.

**Problem.** `scripts/verify.py` was written to restate six claims from this document
against a finished run. All six passed on the first execution. Two of them passed because
they were reading the wrong nesting level of the result JSON and could not have failed:
the degeneracy check looked for `terminal_state` at the container level, where only
`degenerate_archs` exists, and the epoch-cap check looked for `epochs_run` and
`max_epochs` on the same containers, where neither appears. Both found zero records and
correctly concluded that nothing violated the rule — a vacuous pass.

The real structure is one level deeper: `by_bottleneck["256"]["architectures"]["5L_A"]["terminal_state"]`,
and the AE runs live in `task2.json["by_model"][...]["epochs_run"]` with the cap in
`task2.json["max_epochs"]`.

**Decision.** Fix the paths, and make the checks fail loudly when they find nothing to
test. Each now asserts a minimum count — a degeneracy check that finds zero degenerate runs
reports failure rather than success, because zero could mean "a clean run" or "the check
is reading the wrong field", and those are not distinguishable from inside.

**Result.** 42 runs labelled converged, 6 labelled degenerate (all 5L_A, all between
58.81% and 79.45% validation accuracy), 8 autoencoder runs all stopped by a recorded rule
and none at the epoch budget. These numbers reproduce ADR-003 exactly.

**Lesson recorded.** A test whose first success is indistinguishable from a test that never
ran is not evidence. The check that passes is only meaningful if it has been seen to fail
at least once on data that should fail it.

---

## ADR-018 — The redraw said it drew the panels and did not

**Status:** Fixed.

**Problem.** The `run_all.py` redraw at the end of the pipeline called
`class_mean_distances(data, args.outdir)` and then unconditionally printed
"redrew ... with confusion panels". The `confusions_from` argument was never passed, so the
function defaulted to reading no `task1.json`, drew the placeholder panels, and the log
reported success. Three smoke runs in a row printed the reassuring message while the figure
they referred to had blank panels.

The bug was introduced in the same edit that added the redraw, and it survived because the
message was a `print` placed after the call rather than a statement about what the call
achieved. This is the same failure mode as ADR-017 one layer lower: a message that describes
an intention rather than a result.

**Decision.** Pass `confusions_from` explicitly, and make the message conditional on the
condition that makes it true — the report is only printed when `task1.json` actually exists.

**Verification.** A `--quick` run now produces a figure whose confusion and rank panels are
populated, not placeholder text. The check was done by reading the image, not by reading
the log: `scripts/verify.py` also asserts the figure exceeds the byte size of the
placeholder version.
