# Assignment 4 — Gap & Issues Report

> Deadline: **Sunday, October 11, 2026, 10:00 PM**
> Reviewed files: `task1_pca.py`, `task2_autoencoder.py`, `task3_ae_classify.py` (covers Task 3 & 4), `task5_denoising.py`, `task6_weights.py`, `models.py`, `train.py`, `data.py`, `pca.py`, `evaluate.py`, `run_all.py`

---

## 🐛 Bugs

### BUG-1 — `ae_state` key mismatch: Task-2 states will NEVER be reused by Tasks 3/4 (HIGH PRIORITY)

**File:** [run_all.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/run_all.py#L172-L188) / [task3_ae_classify.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task3_ae_classify.py#L105-L107)

`run_all.py` builds `ae_states` keyed by strings like `"1hidden_32"`, `"3hidden_64"`:

```python
ae_states = {k: v["model_state"] for k, v in r["by_model"].items()}
# → keys are "1hidden_32", "1hidden_64", "3hidden_32", etc.
```

But `task3_ae_classify.py` looks up by the **integer** bottleneck `b`:

```python
if ae_state and b in ae_state:   # b is an int like 32
```

`32 in {"1hidden_32": ...}` is always `False`. Every lookup misses, and Tasks 3/4 silently retrain all 8 autoencoders from scratch instead of reusing Task-2's weights.

**Fix:** pass kind-filtered sub-dicts keyed by bottleneck int:
```python
# Task-3 call
ae_state={b: ae_states.get(f"1hidden_{b}") for b in dims}

# Task-4 call
ae_state={b: ae_states.get(f"3hidden_{b}") for b in dims}
```

---

### BUG-2 — `train_acc` is always `None` for Task-5 classifier

**File:** [task5_denoising.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task5_denoising.py#L212)

```python
"train_acc": test_res and None,   # ← always None
```

`test_res` is a non-empty dict (always truthy), so this expression always evaluates to `None`. Fix: call `evaluate_classifier(clf_trained, red, split="train")` and use its accuracy, or drop the field entirely since A4 doesn't require it.

---

### BUG-3 — Variance curve has a single data point instead of a curve

**File:** [task1_pca.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task1_pca.py#L99)

```python
curve = wide.explained_variance_curve([64]).cpu().tolist()
```

Passing `[64]` produces one point at k=64. The figure title says "784 components" but only one point is plotted. Fix:

```python
curve = wide.explained_variance_curve().cpu().tolist()  # all 784 points
```

---

## ⚠️ Methodology / Apples-to-Apples Comparison Issues

### METH-1 — A3 comparison caveat in the code is incomplete; the report must cover both differences

**File:** [task1_pca.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task1_pca.py#L267-L270)

The stored caveat flags only the batch-size difference. The report comparison tables must also state:

- A3 used **NAG**; A4 uses **Adam**. Different optimizers confound the comparison independently of batch size.
- A3 operated on **784-d raw input**; A4 operates on **reduced representations**. The whole point of Tasks 1/3/4 is that this is the intentional variable — state it explicitly rather than leaving it implied.

---

### METH-2 — Task-5 reconstruction error is measured clean-in/clean-out; the report must say so

**File:** [train.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/train.py#L519-L522)

`reconstruct_error` feeds **clean** images through the trained denoiser and measures the reconstruction against the clean target. This is the correct inference-time metric, but it is not the same quantity as the training loss (which feeds noisy input). The A4 wording "average reconstruction errors" is silent on which regime. The report must specify: *post-training, clean-input reconstruction error* — otherwise a reader comparing the two numbers will be confused by the discrepancy.

---

### METH-3 — 1-hidden / 3-hidden / 2-hidden naming must be fixed once in the report and not repeated

**File:** [task2_autoencoder.py](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task2_autoencoder.py#L15-L19)

The assignment itself uses two different names for the same model: "3 hidden layer" in Task-2a and "2-hidden layer autoencoder" in Task-4. The code handles both labels correctly. The report must resolve this once — a footnote or parenthetical on first use stating that both names refer to `784 → 400 → k → 400 → 784` — and then use one name consistently for the rest of the document.

---

## 📋 Missing Coverage / Output Gaps

### GAP-1 — Tasks 1b/3b/4b: full val-accuracy tables (4 architectures × 4 dimensions) must appear in the report

**A4 requirement (1b-ii, 3b-ii, 4b-ii):** *"Present the classification accuracy on the validation set for the different architectures of FCNN classification model."*

This means 16 cells per task (4 archs × 4 dims/bottlenecks), not just the winner per dimension. The code computes and stores all values. Confirm the report tables expose the full grid.

---

### GAP-2 — Task-5d: both validation AND test accuracy must appear in the report

The code reports both `val_acc` and `test_acc` for the Task-3-selected architecture. The A4 requirement (5d-ii) names both explicitly. Confirm both numbers are in the report, not just test accuracy.

---

## ✅ Things That Are Correct (Potential Concerns Verified OK)

| Concern | Status |
|---|---|
| PCA eigenvectors fitted on TRAIN only, val/test projected with TRAIN mean | ✅ Correct — `PCA.fit(X_train)` then `project_all` |
| Best architecture selected on **validation**, never test | ✅ Correct — all tasks use `max(..., key=val_acc)` |
| Adam optimizer for autoencoders (A4-mandated) | ✅ Correct |
| Bottleneck layer is always **linear** (no activation) | ✅ Correct — `nn.Linear` only for bottleneck |
| Sigmoid on remaining hidden layers, used consistently | ✅ Correct — logistic sigmoid throughout |
| 3-hidden AE outer layers are 400 neurons each (A4 mandate) | ✅ Correct — `AE_OUTER = 400` |
| Task-5 bottleneck comes from Task-3's best result (not hardcoded) | ✅ Correct — reads `selection.json["3"]["best_bottleneck"]` |
| Task-5 classifier uses the SAME architecture as Task-3's best | ✅ Correct — reads `task3_arch` from `selection.json` |
| Task-6 uses Task-3's winning bottleneck for the plain AE | ✅ Correct |
| All four classifier architectures used consistently across Tasks 1/3/4 | ✅ Correct — `CLASSIFIER_ARCHS` shared constant |
| Denoising AE: corrupted INPUT, clean TARGET during training | ✅ Correct |
| Reconstruction error computed AFTER training (not during) | ✅ Correct |
| One image per class from **each** of train/val/test for recon grids | ✅ Correct — `one_per_class` per split |
| Task-6 plots encoder weight vectors (28×28) AND max-activating inputs — superset of A4 requirement | ✅ Correct — report just needs to label which plot answers which question |

---

## ⚡ Potential Optimizations

> **Context:** The pipeline already ran in **3.3 min** wall-clock on the RTX 2050 (per `constraints.md`). These suggestions are future-proofing for re-runs with tighter time budgets, additional architectures, or CPU-only machines. **None touch methodology, selection rules, or any reported metric.**

---

### OPT-1 — Fix BUG-1 first: free ~50% of Tasks 3/4 autoencoder time

As noted in BUG-1, the `ae_state` key mismatch means Tasks 3 & 4 silently retrain all 8 autoencoders that Task-2 already built. This is simultaneously a bug fix and the highest-value optimization — no methodology change.

**Estimated saving:** eliminates 8 full autoencoder retraining runs (4 bottlenecks × 2 kinds).

---

### OPT-2 — Eliminate the double PCA fit per dimension (one-liner)

**File:** [task1_pca.py L85–89](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/task1_pca.py#L85-L89)

Two separate `PCA.fit()` calls run on the same data for each `k`. Each `eigh` on the 784×784 covariance matrix costs ~1–2 s on CPU. For 4 dimensions that is 4 wasted fits.

**Fix:**
```python
pca = PCA(k=k).fit(data["X_train"])
_representation_cache[k] = pca.project_all(data)
red = _representation_cache[k]   # no second fit
```

---

### OPT-3 — Separate early-stopping patience for autoencoders vs classifiers

**File:** [train.py L64](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/train.py#L64)

`STOP_PATIENCE = 15` was calibrated against the classifier's oscillating plateau problem. Autoencoders have smooth MSE loss — that pathology does not apply. A patience of 8 fires at the same converged point and saves ~7 epochs of unnecessary training per AE run.

```python
AE_STOP_PATIENCE = 8   # pass only to train_autoencoder calls
```

Reported metrics are unaffected: early stopping fires after convergence, and all metrics are evaluated on the final weights after stopping.

---

### OPT-4 — `torch.compile` for autoencoder forward/backward (flag already exists)

**File:** [constraints.md L385–391](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/constraints.md#L385-L391)

`torch.compile` fuses the Linear/Sigmoid/MSELoss kernel sequence on Ampere (sm86), giving **20–40% throughput gain** for these sequential models. The `ENABLE_COMPILE` flag is already wired up with a `try/except` fallback — flip it to `True` and run `--quick` first to confirm no interaction with `weights_only=False` checkpoint loading.

Only worthwhile if running >500 epochs per model; JIT overhead per architecture shape is 20–40 s.

---

### OPT-5 — AMP fp16 for autoencoder training (flag already exists, test carefully)

**File:** [constraints.md L352–365](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/constraints.md#L352-L365)

The 3-hidden AE's large Linear layers (`784↔400`) benefit from Tensor Core fp16 GEMM on the RTX 2050. Expected speedup: **1.5–2×** for the training loop.

**Constraints that must hold:**
- PCA, mean subtraction, reconstruction error reporting, and weight-viz argmax stay fp32 — these are already outside the training loop and unaffected.
- Use `GradScaler` + `isfinite()` guard — the RTX 2050 has logged fp16 Adam spikes on wide architectures.
- `ENABLE_AMP = False` by default; enable only after verifying reconstruction error numbers are stable.

All reported metrics (accuracy, reconstruction error, confusion matrix) are evaluated in fp32 after training and are unaffected by AMP.

---

### OPT-6 — Raise `CHECKPOINT_EVERY` for large autoencoders

**File:** [train.py L69](file:///home/anshuman139/Deep-Learning/DL_Assignments/Assignment4/Group11_Assignment4_code/train.py#L69)

At `CHECKPOINT_EVERY=10`, the heaviest 3-hidden AE writes ~4 MB every 10 epochs. Over a 10,000-epoch run that is ~4 GB of cumulative SSD I/O. Pass `checkpoint_every=50` to the large AE training calls: this bounds crash-loss to 50 epochs while cutting I/O by 5×. The `constraints.md` document explicitly anticipates this per-run adjustment.

---

### Optimization Priority Summary

| # | Optimization | Effort | Benefit | Safe to apply? |
|---|---|---|---|---|
| OPT-1 | Fix BUG-1 (ae_state keys) | **Trivial** | Eliminates 8 AE retraining runs | ✅ Yes — bug fix |
| OPT-2 | Eliminate double PCA fit | **Trivial** | ~4–8 s per full run | ✅ Yes — one-liner |
| OPT-3 | `AE_STOP_PATIENCE = 8` | Low | ~7 epochs saved per AE | ✅ Yes — no metric change |
| OPT-4 | `torch.compile` (flag exists) | Low | 20–40% per AE model | ✅ Test with `--quick` first |
| OPT-5 | AMP fp16 (flag exists) | Medium | 1.5–2× AE training | ⚠️ Verify reconstruction error parity |
| OPT-6 | Raise `CHECKPOINT_EVERY` for AEs | Low | ~4 GB less SSD I/O | ✅ Yes |

> **Bottom line:** OPT-1 and OPT-2 are the only ones worth applying before the submission deadline — both are one-line changes with zero risk. OPT-3 through OPT-6 are relevant for future re-runs with larger epoch budgets.

---

## Summary: Action Items Before Submission

| Priority | Item | File |
|---|---|---|
| 🔴 HIGH | Fix ae_state key mismatch (BUG-1) — autoencoders retrain unnecessarily | `run_all.py` |
| 🟡 MED | Fix variance curve to plot all 784 points (BUG-3) | `task1_pca.py` |
| 🟡 MED | Fix `train_acc: None` in Task-5 results (BUG-2) | `task5_denoising.py` |
| 🟢 LOW | Report: full val-accuracy tables (4 archs × 4 dims) for Tasks 1/3/4 (GAP-1) | Report |
| 🟢 LOW | Report: show both val and test accuracy for Task-5d (GAP-2) | Report |
| 🟢 LOW | Report: state clean-input reconstruction metric for Task-5 (METH-2) | Report |
| 🟢 LOW | Report: state optimizer and batch-size differences when comparing to A3 (METH-1) | Report |
| 🟢 LOW | Report: resolve 1-hidden/3-hidden/2-hidden naming once and use consistently (METH-3) | Report |
| 🟢 LOW | Clean up double PCA fit in Task-1 (OPT-2) | `task1_pca.py` |
