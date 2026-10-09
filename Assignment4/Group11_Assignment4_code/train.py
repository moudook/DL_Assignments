"""
train.py — training loops for Assignment-4.

Two loop families:
    train_classifier  — FCNN, cross-entropy, for Tasks 1, 3, 4, 5
    train_autoencoder — AE / DenoisingAE, MSE, for Tasks 2, 5

Shared contract
---------------
Every loop takes an optional RunTracker and obeys the same rules:
  - full-batch gradient step (locked decision; A4 is silent on batch size)
  - early stop on |L_t - L_{t-1}| < TOL, the A3 stopping rule A4 inherits by
    convention. TOL is NOT an A4 requirement but is our stated convention.
  - checkpoint every CHECKPOINT_EVERY epochs, atomically, with model AND
    optimizer state
  - resume from an existing checkpoint instead of restarting at epoch 0
  - full-batch val/test evaluation, matching A3 so numbers stay comparable
  - non-finite loss / gradient guards

Why full batch was validated rather than assumed
-----------------------------------------------
The pilot measured it: at full batch with lr=0.001 the classifier sits at 20.00%
(chance) after 30 epochs, reproducing A3's Batch-GD failure. At lr=0.01 it
reaches 72.62% and still climbing. Classifier LR is therefore 0.01. Autoencoders
use Adam at 0.001, where pilot showed AE loss falling 0.195 -> 0.070.

Early-stopping caveat we must not repeat from A3
-------------------------------------------------
A3's Batch GD satisfied the 1e-4 rule in 2 epochs while accuracy was still at
chance, i.e. the rule certified a useless model as "converged". Here `finish`
reports the terminal state and the final metric, and the caller records whether
the model actually beat chance. A run that stops at chance is visible as
stopped_early with a poor metric, not silently reported as a success.
"""

import collections
import os

import torch
import torch.nn as nn

from models import make_noise
from run_tracker import atomic_save

TOL = 1e-4
# Early stopping requires the tolerance to hold over a WINDOW of consecutive
# epochs, not on a single epoch pair.
#
# Why a window is necessary (measured on this dataset): the 5L_A classifier
# [64,32,16,8,4] stalls with loss 1.6089 at epoch 13 and then OSCILLATES upward
# (1.6091, 1.6096, 1.6102, 1.6108, 1.6111). Adam's momentum swinging in a local
# plateau produced a single-epoch |dL| of 1.2e-05, under the 1e-4 tolerance, so a
# one-epoch test declared convergence while validation accuracy was still exactly
# 20.00% (chance). The loss was flat AND RISING - the opposite of convergence.
#
# A single consecutive-epoch difference cannot distinguish:
#   - genuine convergence (loss settled, small change persists), from
#   - a plateau the optimizer is oscillating inside of (small change transiently).
# Requiring the criterion to hold for PATIENCE consecutive epochs distinguishes
# them: real convergence sustains the small delta, an oscillation does not.
#
# This is the same defect that made Assignment-3's Batch GD "converge" in 2
# epochs at 20% accuracy. It is inherited behaviour that needed correcting, not
# reproduced deliberately.
STOP_PATIENCE = 15

# Minimum epochs before the stopping rule may fire at all.
#
# The absolute criterion |dL| < 1e-4 is scale-blind, and that bites hardest on
# the autoencoders. Measured on 3hidden_k=256: the MSE is 0.067171 at epoch 29
# and still falling by ~9e-5 per epoch. 9e-5 is BELOW the 1e-4 tolerance, so the
# rule fired at epoch 39 with test reconstruction error 0.0668 - against 0.0313
# for k=128 and 0.0185 for the 1-hidden k=256. The extra capacity was stopped
# before it was used, and because Task-4 consumes this encoder, three of four
# classifiers on that representation then sat at 20% (chance).
#
# On an MSE of 0.067 a 9e-5 step is 0.13% per epoch - meaningful progress, not a
# plateau. The identical delta on a cross-entropy of 1.6 would be 0.006% and
# genuinely converged. A single absolute tolerance cannot serve both loss scales.
#
# The assignment mandates the absolute 1e-4 criterion, so it is kept verbatim.
# An earlier revision added an epoch FLOOR before the rule could fire, on the
# theory that it would suppress the premature stops. Both call sites now pass
# min_epochs=0, because it emerged that the floor was binding on runs whose
# classifier trained properly (stopping them early for no benefit) and was
# unnecessary once the AEs were moved onto the plateau rule. It was therefore
# dead code and has been removed rather than left to describe behaviour the
# code does not have.

CLASSIFIER_LR = 0.01     # pilot-validated; 0.001 sits at chance at full batch
AUTOENCODER_LR = 0.001   # A4 mandates Adam; pilot-validated
# Autoencoders use a RELATIVE stopping tolerance; classifiers keep A3's absolute
# one. Measured on real 1500-epoch traces, the absolute rule stops the AEs 8x
# short of their achievable reconstruction error because MSE and cross-entropy
# differ in scale by ~100x. See _ConvergenceWatch.__init__ for the traces.
AE_REL_TOL = 1e-3
# Window, in epochs, over which "best loss so far" must stop improving.
#
# Why a WINDOW and not an epoch-to-epoch delta:
#   - an epoch-to-epoch delta cannot tell a plateau from steady slow descent.
#     Measured: under relative tolerance the 1-hidden autoencoders were still
#     improving ~0.1% per epoch when it fired, and replaying the identical run to
#     12,000 epochs reached 1.83x lower reconstruction error.
#   - the denoising AEs resample their corruption mask every epoch, so the
#     per-epoch delta jitters far above the threshold. Neither denoiser ever
#     completed the required streak: both ran the full 10,000 epochs. The
#     criterion was silently disabled exactly where A4 requires it most.
#
# "Best loss has not improved over the last WINDOW epochs" is immune to both:
# it is scale-free (a ratio), it is immune to per-epoch jitter (it tracks the
# monotone best, not the noise), and under geometric decay L_t = L_0 g^t the
# windowed improvement is (1 - g^W), which shrinks to zero only when the decay
# genuinely stops.
AE_PLATEAU_WINDOW = 50
MAX_EPOCHS = 10000       # measured affordable: ~4.6 min worst-case AE
CHECKPOINT_EVERY = 10
LOG_EVERY = 100

CLASSIFIER_OPT = torch.optim.Adam
AE_OPT = torch.optim.Adam   # A4 Task-2b mandates Adam for autoencoders


def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


CHANCE = 1.0 / 5


# A run below this validation accuracy is not a converged model, it is a failed
# one. Calibrated against the runs actually observed on this dataset: legitimate
# results sit at 97-99% and the 5L_A architecture's failures sit at 58.8%, 59.6%
# and 77.9%. 85% sits cleanly between those two groups - far enough above every
# failure to catch them all, far enough below every real result to never risk
# mislabelling a success. It is specific to this dataset rather than a general
# rule, which is deliberate: a fixed multiple of chance cannot separate them,
# since 77.9% is nearly 4x chance.
DEGENERATE_BELOW = 0.85


def flag_degenerate(final_acc, chance=CHANCE):
    """
    True when a run learned too little to be called converged.

    Needed because the A3 stopping rule CANNOT distinguish "converged to a good
    solution" from "stopped moving because the gradient vanished". A deep sigmoid
    stack satisfies |dL| < 1e-4 within a few epochs simply because nothing is
    changing, which the rule scores as success.

    Observed for real in this pipeline: the 5L_A architecture [64,32,16,8,4]
    stopped at exactly 20.00% val accuracy after 20 epochs, reported as
    converged. Five stacked sigmoid layers attenuate the signal multiplicatively
    and the gradient underflows, so training never begins.

    The original threshold was 1.25x chance, which only caught the 20% case. An
    independent audit then found the wider band this still misses: 5L_A runs
    settling at 58.8%, 59.6% and 77.9% val accuracy were all still recorded as
    "converged" with degenerate=False. They are severely underfit - they learned
    *something*, so they escaped the chance test, but nowhere near enough to be a
    usable classifier, and calling them successes is the same error one notch up.
    Hence the absolute DEGENERATE_BELOW threshold documented above.

    Every finished run therefore carries an explicit `degenerate` flag, which task
    modules surface rather than leaving a failed model to be read as a valid
    result.
    """
    return bool(final_acc < DEGENERATE_BELOW)


def selection_bias_report(by_key, get_test, get_val):
    """
    Quantify the optimism in "pick the representation with the best TEST score".

    A4 asks which reduced representation classifies best, and reads that off the
    test scores. Doing so means the headline is a maximum over several test
    numbers, so it is biased upward: even if every representation were equally
    good, the luckiest one would still score above the truth. An independent
    audit measured the size of that gap here as +0.17 to +0.47 pp.

    Reporting only the maximum hides it. This returns the same number alongside
    the mean across representations, the maximum-minus-mean gap, and what a
    validation-selected choice would have scored - so the report can state the
    bias instead of quietly benefiting from it.

    Architecture selection is val-only and is not affected; only this
    across-representation choice is.
    """
    tests = {k: get_test(v) for k, v in by_key.items()}
    vals = {k: get_val(v) for k, v in by_key.items()}
    best_test_key = max(tests, key=lambda k: tests[k])
    best_val_key = max(vals, key=lambda k: vals[k])
    mean = sum(tests.values()) / len(tests)
    return {
        "criterion": "best test accuracy (as mandated by A4)",
        "criterion_is_optimistically_biased": True,
        "bias_note": (
            "The best representation is chosen by test accuracy, so this is a "
            "maximum over several test scores and is biased upward. The honest "
            "unbiased estimate is the mean across representations, and the "
            "validation-selected choice is reported alongside."),
        "selected_by_test": int(best_test_key) if isinstance(best_test_key, int)
                            else best_test_key,
        "selected_by_test_accuracy": tests[best_test_key],
        "mean_across_representations": mean,
        "max_minus_mean_pp": 100 * (tests[best_test_key] - mean),
        "selected_by_validation": int(best_val_key) if isinstance(best_val_key, int)
                                   else best_val_key,
        "selected_by_validation_accuracy": vals[best_val_key],
        "per_representation_test": tests,
        "per_representation_val": vals,
    }


def _guard_finite(name, tensor, epoch):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name}: non-finite value at epoch {epoch + 1}")


class _ConvergenceWatch:
    """
    Patience-window early-stopping criterion.

    Counts CONSECUTIVE epochs where |L_t - L_{t-1}| < tol, and only reports
    convergence once that count reaches `patience`. A one-epoch test is not
    sufficient - see the STOP_PATIENCE comment for the measured failure.

    Also tracks the best loss seen, because a run whose loss is still RISING when
    the criterion fires has plateaued, not converged. `improved` reports whether
    any real progress happened at all, which distinguishes "converged" from
    "never started learning".

    State is checkpointed (state_dict/load_state) because the patience streak is
    part of the run's state: without it, every resume restarts the counter from
    zero, so a crash mid-run can make training stop EARLIER than the criterion
    intends. Verified: a relaunch after convergence resumed from epoch 242 and
    ran 16 further epochs before the criterion was met again.
    """

    def __init__(self, tol=TOL, patience=STOP_PATIENCE,
                 min_epochs=0, mode="abs", window=None):
        self.tol = tol
        self.patience = patience
        # Epochs that must elapse before the criterion is even considered. Zero
        # by default and zero at every call site - the MIN_EPOCHS_BEFORE_STOP
        # floor it once refused to work below was removed as dead code.
        self.min_epochs = min_epochs
        # "abs"     -> |L_t - L_{t-1}| < tol                   (A3's literal rule)
        # "rel"     -> |L_t - L_{t-1}| < tol * |best loss|     (scale-free delta)
        # "plateau" -> best loss has not improved by more than tol (relative)
        #              over a window of `window` epochs. Immune to per-epoch
        #              jitter, so it works with the stochastic denoising objective.
        #
        # The three are NOT interchangeable. Replaying real 1500-epoch traces:
        #   1hidden k=256  abs1e-4 stops ep186 loss 0.0181 | rel1e-3 stops ep1030
        #                   loss 0.0023; but continuing to 12,000 epochs reaches
        #                   0.00195, so "rel" also stops short.
        # For the autoencoders only "plateau" certifies convergence.
        self.mode = mode
        self.window = window or patience
        # Monotone best-loss history, for the plateau test.
        self._best_hist = collections.deque(maxlen=self.window + 1)
        self.streak = 0
        self.prev_loss = None
        self.best_loss = float("inf")
        self.improved = False
        self.triggered = False
        self.trigger_epoch = None

    def state_dict(self):
        return {"streak": self.streak, "prev_loss": self.prev_loss,
                "best_loss": self.best_loss, "improved": self.improved,
                # The plateau window IS the criterion's state, not decoration.
                # Without it a resumed run falls back to `window=patience` (or
                # whatever the constructor defaults to) and re-evaluates the
                # history from a truncated deque, so it can stop far earlier
                # than the rule intends. `_best_hist` is likewise part of the
                # state: after a resume it is empty, and an empty deque silently
                # disables the plateau test.
                "mode": self.mode, "window": self.window,
                "min_epochs": self.min_epochs, "triggered": self.triggered,
                "best_hist": list(self._best_hist)}

    def load_state_dict(self, sd):
        self.streak = sd.get("streak", 0)
        self.prev_loss = sd.get("prev_loss")
        self.best_loss = sd.get("best_loss", float("inf"))
        self.improved = sd.get("improved", False)
        # Prefer the saved rule over the constructor's default, so a run resumed
        # from a checkpoint keeps the criterion it was actually trained under.
        self.mode = sd.get("mode", self.mode)
        self.window = sd.get("window", self.window)
        self.min_epochs = sd.get("min_epochs", self.min_epochs)
        self.triggered = sd.get("triggered", False)
        hist = sd.get("best_hist")
        if hist:
            self._best_hist = collections.deque(list(hist),
                                                maxlen=self.window + 1)

    def update(self, loss, epoch):
        """Feed one epoch's loss. Returns True once the criterion is satisfied."""
        if self.best_loss - loss > self.tol:
            self.improved = True
        if loss < self.best_loss:
            self.best_loss = loss
        self._best_hist.append(self.best_loss)

        if self.mode == "plateau":
            # Fire when the monotone best loss has not improved by more than
            # `tol` (relative) across the whole window. Immune to jitter.
            if len(self._best_hist) > self.window:
                old = self._best_hist[0]
                gain = (old - self.best_loss) / max(abs(old), 1e-12)
                if gain < self.tol and epoch + 1 >= self.min_epochs:
                    self.triggered = True
                    self.trigger_epoch = epoch
            return self.triggered

        if self.prev_loss is not None:
            thr = self.tol if self.mode == "abs" else self.tol * abs(self.best_loss)
            if abs(loss - self.prev_loss) < thr:
                self.streak += 1
            else:
                self.streak = 0
        self.prev_loss = loss

        # The min_epochs floor: below it the streak is tracked but cannot trigger.
        if epoch + 1 >= self.min_epochs and self.streak >= self.patience:
            self.triggered = True
            self.trigger_epoch = epoch
        return self.triggered


def _save_checkpoint(tracker, model, optimizer, epoch, device, extra=None,
                     watch=None):
    """
    Atomic checkpoint with model + optimizer state.

    Routed through tracker.check so cadence and atomicity live in one place.
    The convergence watcher's state travels with the checkpoint via `extra`:
    RunTracker.check() builds its OWN payload dict and only merges `extra`, so
    passing watch_state as a separate argument is silently discarded - which is
    exactly what happened, leaving every checkpoint without it.
    """
    if tracker is None:
        return
    merged = dict(extra or {})
    if watch is not None:
        merged["watch_state"] = watch.state_dict()
    tracker.check(epoch, model=model, optimizer=optimizer, extra=merged)


def _split_tensors(data):
    """
    Pull X/y for train and val from whichever representation `data` holds.

    Interface decision (option A): a REDUCED representation is itself a drop-in
    data dict. PCA.project_all() and encode_all() emit exactly the same key
    names as load_splits() (X_train, y_train, X_val, y_val, X_test, y_test), so
    train_classifier has ONE code path whether the input is raw 784-d images or a
    32/64/128/256-d projection or bottleneck.

    Consequences, which is why this is simpler than any alternative:
      - no input_dim parameter to thread through and get wrong
      - no branching on dimensionality
      - no way to train an input_dim=32 model against raw 784-d data, because
        the caller physically cannot put 784 features in a reduced dict
      - a shape mismatch is a plain nn.Linear matmul error, which is clear
        enough on its own and needs no bespoke guard
    """
    return (data["X_train"], data["y_train"],
            data["X_val"], data["y_val"])


def train_classifier(model, data, arch_name="arch", tracker=None,
                     lr=CLASSIFIER_LR, max_epochs=MAX_EPOCHS, tol=TOL,
                     device=None, checkpoint_every=CHECKPOINT_EVERY,
                     resume=True, patience=STOP_PATIENCE):
    """
    Train an FCNN classifier with full-batch cross-entropy.

    data: a representation dict. Raw images from load_splits(), or a reduced one
    from PCA.project_all() / encode_all() - both use identical key names, so this
    function is the same either way.
    Returns history dict with train_loss, train_acc, val_acc, and terminal state.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    X_tr, y_tr, X_va, y_va = _split_tensors(data)

    optimizer = CLASSIFIER_OPT(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    # Resume in place when a checkpoint exists. The epoch counter, Adam moments,
    # and elapsed clock all come from the checkpoint, so a resumed run continues
    # the same trajectory instead of restarting Adam's bias correction.
    if resume and tracker is not None and tracker.has_checkpoint():
        ckpt = torch.load(tracker.ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if tracker.history:
                tracker.first_epoch_time = ckpt.get("start_time", tracker.first_epoch_time)
            # Restore the RUN HISTORY, not just the model. A fresh RunTracker
            # starts with an empty list, so a resumed run would otherwise report
            # "history": [] and every figure built from it would be empty (and
            # loss_and_accuracy, which indexes history[0], would crash). The
            # checkpoint carries the full curve for exactly this reason.
            if ckpt.get("history"):
                tracker.history = ckpt["history"]
            tracker.log(f"RESUME {arch_name} from epoch {start_epoch} "
                        f"({len(tracker.history)} epochs of history restored)")

    # A checkpoint at or past max_epochs means there is nothing left to do. The
    # loop below would not execute, leaving `epoch` unbound - hence this branch.
    # This is the normal path when re-running a finished job, not an edge case.
    if start_epoch >= max_epochs:
        final_val = _accuracy(model.eval()(X_va).detach(), y_va)
        tracker.log(f"SKIP {arch_name}: already at epoch {start_epoch} "
                    f">= max_epochs {max_epochs}")
        tracker.finish(status="skipped", val_acc=final_val, lr=lr,
                       epochs_run=0)
        return {
            "arch_name": arch_name,
            "lr": lr,
            "train_loss": tracker.history,
            "final_val_acc": final_val,
            "stopped_early": False,
            "epochs_run": 0,
            "skipped": True,
            "model_state": {k: v.detach().clone()
                            for k, v in model.state_dict().items()},
        }

    stopped_early = False
    # Classifiers: A3-mandated ABSOLUTE tolerance. Cross-entropy starts near 1.6,
    # so 1e-4 is a genuine 0.006% - the literal inherited rule is appropriate.
    watch = _ConvergenceWatch(tol=tol, patience=patience, mode="abs",
                             min_epochs=0)
    # Restore the patience streak from the checkpoint. Without this, a resumed run
    # restarts the counter at zero and can stop EARLIER than the criterion
    # intends, which silently changes results depending on where a crash fell.
    if resume and tracker is not None and tracker.has_checkpoint():
        _ck = torch.load(tracker.ckpt_path, map_location="cpu",
                         weights_only=False)
        if _ck.get("watch_state"):
            watch.load_state_dict(_ck["watch_state"])

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        _guard_finite(f"{arch_name} loss", loss, epoch)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                _guard_finite(f"{arch_name} grad[{name}]", p.grad, epoch)
        optimizer.step()

        train_acc = _accuracy(logits.detach(), y_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_acc = _accuracy(val_logits, y_va)

        loss_val = loss.item()

        if tracker is not None:
            tracker.tick(epoch, loss=loss_val, val_acc=val_acc,
                         train_acc=train_acc)

        if epoch % LOG_EVERY == 0 and tracker is not None:
            tracker.log(f"  {arch_name} ep={epoch} loss={loss_val:.5f} "
                        f"val_acc={val_acc:.4f}")

        # Checkpoint on cadence and always on the final epoch.
        if tracker is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch == max_epochs - 1
        ):
            _save_checkpoint(tracker, model, optimizer, epoch, device,
                             extra={"arch_name": arch_name, "lr": lr},
                             watch=watch)

        # Early stop, but only once |dL| < tol has held for `patience` CONSECUTIVE
        # epochs - see _ConvergenceWatch for the measured failure this prevents.
        if watch.update(loss_val, epoch):
            stopped_early = True
            if tracker is not None:
                tracker.log(
                    f"CONVERGED {arch_name} at epoch {epoch + 1} "
                    f"(|dL| < {tol:g} for {patience} consecutive epochs)")
            break

    final_val = _accuracy(model.eval()(X_va).detach(), y_va)
    degenerate = flag_degenerate(final_val)
    # A collapsed run is NOT reported as success. Keeping "converged" as the
    # status would let a chance-level model read as a valid result in the tables.
    if degenerate:
        state = "degenerate"
    else:
        state = "converged" if stopped_early else "max_epochs"

    # Log the warning BEFORE finish(), because finish() closes the log file and
    # a subsequent log() raises ValueError on the closed handle. That ordering
    # bug crashed the run after training had already finished successfully.
    if tracker is not None and degenerate:
        tracker.log(
            f"WARNING {arch_name}: finished at {100 * final_val:.2f}% "
            f"val accuracy (chance = 20%). Training collapsed - see the "
            f"vanishing-gradient note in train.flag_degenerate. Excluded "
            f"from best-architecture selection.")
    if tracker is not None:
        # model/optimizer are passed so finish() writes the FINAL weights: without
        # this the checkpoint trails the reported metrics by up to CHECKPOINT_EVERY-1
        # epochs and cannot be used to verify the results.
        # `extra` carries the identity metadata the periodic save wrote - arch
        # name, lr and the convergence watch state. Without it the final
        # checkpoint loses run_name/lr/watch_state that the periodic saves had,
        # so a resume after the last save would lose the plateau window and the
        # convergence streak.
        tracker.finish(status=state, model=model, optimizer=optimizer,
                        extra={"arch_name": arch_name, "lr": lr},
                        watch=watch,
                        val_acc=final_val,
                        epochs_run=epoch + 1 - start_epoch)

    return {
        "arch_name": arch_name,
        "lr": lr,
        "train_loss": tracker.history if tracker else [],
        "final_val_acc": final_val,
        "stopped_early": stopped_early,
        "degenerate": degenerate,
        "terminal_state": state,
        "epochs_run": epoch + 1 - start_epoch,
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def reconstruct_error(model, X, batch_size=4096):
    """
    Mean per-sample reconstruction error, in fp32.

    MSE averaged over samples and pixels: mean((x - x_hat)^2). Computed in
    chunks so a large split does not spike VRAM; batches are independent
    (no cross-sample terms), so chunking does not change the value.
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i + batch_size]
            rec = model(xb)
            total += ((xb - rec) ** 2).mean(dim=1).sum().item()
            n += xb.size(0)
    return total / max(1, n)


@torch.no_grad()
def denoising_error(model, X, noise_level, batch_size=4096, seed=1234,
                    copy_baseline=False):
    """
    Reconstruction error when the model is fed CORRUPTED input, against the CLEAN
    target - the metric that actually measures denoising.

    `reconstruct_error` feeds clean images, which for a denoising autoencoder is
    the wrong quantity: it reports how good a plain autoencoder the denoising
    objective happened to produce, not how much noise the model removes.

    `copy_baseline=True` additionally returns the error of simply passing the
    corrupted image through unchanged. Without that reference, "did it denoise?"
    is unanswerable, because a low error could equally mean the input was
    barely corrupted. Measured on the shipped models it beats the copy baseline
    by ~40.6% at 20% noise and ~61.1% at 40%, which is the claim the report needs.
    (An earlier draft of this docstring said ~43%/~63%; those figures were wrong.
    The percentages are computed from denoise_error["test"] in the result JSONs
    rather than quoted by hand, so they can always be re-derived from the
    artifacts.)

    seeded so the reported figure is reproducible.
    """
    model.eval()
    total, n = 0.0, 0
    base_total = 0.0
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i + batch_size]
        # A fixed generator per call batch keeps the corruption identical across
        # calls for a given split.
        g = torch.Generator(device=X.device).manual_seed(seed + i)
        noisy = make_noise(xb, noise_level, g)
        total += ((model(noisy) - xb) ** 2).mean(dim=1).sum().item()
        if copy_baseline:
            base_total += ((noisy - xb) ** 2).mean(dim=1).sum().item()
        n += xb.size(0)
    err = total / max(1, n)
    if not copy_baseline:
        return err
    return err, base_total / max(1, n)


@torch.no_grad()
def encode_all(model, data, batch_size=4096):
    """
    Bottleneck representation for every split — what Tasks 3, 4, 5 and 6 consume.

    fp32, model.eval() so no training-time behaviour leaks in. Chunked to bound
    peak VRAM.

    Returns a DROP-IN representation dict: same key names as load_splits()
    (X_train, y_train, X_val, y_val, X_test, y_test), so it passes straight to
    train_classifier in place of the raw dict. y_* are carried over unchanged.
    """
    model.eval()
    out = {}
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        out[f"X_{split}"] = torch.cat(
            [model.encode(X[i:i + batch_size])
             for i in range(0, X.size(0), batch_size)]
        )
        out[f"y_{split}"] = data[f"y_{split}"]
    out["representation"] = getattr(model, "representation", "autoencoder")
    return out


def train_autoencoder(model, data, run_name="ae", tracker=None,
                      lr=AUTOENCODER_LR, max_epochs=MAX_EPOCHS, tol=TOL,
                      device=None, checkpoint_every=CHECKPOINT_EVERY,
                      noise_level=0.0, noise_generator=None, resume=True,
                      patience=STOP_PATIENCE):
    """
    Train an autoencoder with full-batch MSE. Handles the denoising case too.

    noise_level > 0 makes this a DENOISING autoencoder (Task-5): the forward
    pass sees a corrupted input while the loss is computed against the CLEAN
    image. That asymmetry is what forces the bottleneck to discard noise. At
    noise_level == 0 it is a plain autoencoder (Task-2).

    autoencoder LR 0.001 per A4's Adam mandate and the pilot.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    X_tr = data["X_train"]
    optimizer = AE_OPT(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    if resume and tracker is not None and tracker.has_checkpoint():
        ckpt = torch.load(tracker.ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if tracker.history:
                tracker.first_epoch_time = ckpt.get("start_time", tracker.first_epoch_time)
            # Restore the run history - see the identical note in
            # train_classifier. Without it a resumed run reports an empty curve
            # and the plots built from it are blank or crash.
            if ckpt.get("history"):
                tracker.history = ckpt["history"]
            tracker.log(f"RESUME {run_name} from epoch {start_epoch} "
                        f"({len(tracker.history)} epochs of history restored)")

    # Already trained past the budget: nothing to do. Without this the loop body
    # never runs and `epoch` is left unbound below.
    if start_epoch >= max_epochs:
        recon = {
            split: reconstruct_error(model, data[f"X_{split}"])
            for split in ("train", "val", "test")
        }
        tracker.log(f"SKIP {run_name}: already at epoch {start_epoch} "
                    f">= max_epochs {max_epochs}")
        tracker.finish(status="skipped",
                       train_recon=recon["train"], val_recon=recon["val"],
                       test_recon=recon["test"], lr=lr, noise=noise_level,
                       epochs_run=0)
        return {
            "run_name": run_name,
            "lr": lr,
            "noise_level": noise_level,
            "history": tracker.history,
            "recon_error": recon,
            "stopped_early": False,
            "epochs_run": 0,
            "skipped": True,
            "model_state": {k: v.detach().clone()
                            for k, v in model.state_dict().items()},
        }

    from models import make_noise

    stopped_early = False
    # Autoencoders: RELATIVE tolerance (1e-3 of the best loss seen). The absolute
    # rule measurably fires ~8x too early on MSE — see _ConvergenceWatch.__init__
    # for the replayed 1500-epoch traces that justify this.
    watch = _ConvergenceWatch(tol=AE_REL_TOL, patience=patience, mode="plateau",
                             window=AE_PLATEAU_WINDOW, min_epochs=0)
    # Restore the patience streak; see train_classifier for why.
    if resume and tracker is not None and tracker.has_checkpoint():
        _ck = torch.load(tracker.ckpt_path, map_location="cpu",
                         weights_only=False)
        if _ck.get("watch_state"):
            watch.load_state_dict(_ck["watch_state"])

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        x_in = make_noise(X_tr, noise_level, noise_generator) if noise_level > 0 else X_tr
        rec = model(x_in)
        # Target is always the CLEAN image, even in the denoising case.
        loss = criterion(rec, X_tr)
        _guard_finite(f"{run_name} loss", loss, epoch)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if tracker is not None:
            tracker.tick(epoch, loss=loss_val)

        if epoch % LOG_EVERY == 0 and tracker is not None:
            tracker.log(f"  {run_name} ep={epoch} loss={loss_val:.6f}")

        if tracker is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch == max_epochs - 1
        ):
            _save_checkpoint(tracker, model, optimizer, epoch, device,
                             extra={"run_name": run_name, "lr": lr,
                                    "noise_level": noise_level},
                             watch=watch)

        # Patience-window early stopping; see _ConvergenceWatch. Single-epoch
        # testing is unreliable because an oscillating plateau can dip below tol
        # transiently, exactly as the 5L_A classifier did at chance accuracy.
        if watch.update(loss_val, epoch):
            stopped_early = True
            if tracker is not None:
                # Report the rule that ACTUALLY ran. Previously the AE logs and
                # every AE figure printed the module-level absolute TOL while the
                # autoencoders were using a relative threshold ~10x tighter, so
                # the convergence claim understated the strictness by an order of
                # magnitude.
                if watch.mode == "plateau":
                    rule = (f"best loss improved < {100 * watch.tol:.3g}% "
                            f"over {watch.window} epochs")
                elif watch.mode == "rel":
                    rule = f"|dL| < {watch.tol:g} * |best L|"
                else:
                    rule = f"|dL| < {watch.tol:g}"
                tracker.log(f"CONVERGED {run_name} at epoch {epoch + 1} "
                            f"({rule})")
            break

    # Post-training reconstruction error on all three splits, as A4 Task-2c
    # requires (computed AFTER training, not during).
    recon = {
        split: reconstruct_error(model, data[f"X_{split}"])
        for split in ("train", "val", "test")
    }
    # For the DENOISING AE only, also report the corrupted-input error and the
    # copy-the-input baseline. The clean-input number above is what A4's generic
    # "average reconstruction errors" asks for and is what the tables show, but on
    # its own it does not measure denoising - it measures how good a plain
    # autoencoder the denoising objective happened to produce.
    denoise = None
    if noise_level > 0:
        denoise = {
            split: dict(zip(
                ("corrupted_input_error", "copy_baseline"),
                denoising_error(model, data[f"X_{split}"], noise_level,
                                copy_baseline=True)))
            for split in ("train", "val", "test")
        }
    state = "converged" if stopped_early else "max_epochs"
    if tracker is not None:
        metrics = {"train_recon": recon["train"], "val_recon": recon["val"],
                   "test_recon": recon["test"], "lr": lr,
                   "noise": noise_level,
                   "epochs_run": epoch + 1 - start_epoch}
        if denoise is not None:
            metrics["test_denoise_err"] = \
                denoise["test"]["corrupted_input_error"]
            metrics["test_copy_baseline"] = denoise["test"]["copy_baseline"]
        # As for the classifier: save the FINAL weights so the checkpoint on disk
        # is exactly the model these metrics describe, carrying the same identity
        # metadata (run name, lr, noise, watch state) the periodic save wrote.
        tracker.finish(status=state, model=model, optimizer=optimizer,
                       extra={"run_name": run_name, "lr": lr,
                              "noise_level": noise_level},
                       watch=watch, **metrics)

    return {
        "run_name": run_name,
        "lr": lr,
        "noise_level": noise_level,
        "history": tracker.history if tracker else [],
        "recon_error": recon,
        "denoise_error": denoise,
        "stopped_early": stopped_early,
        "epochs_run": epoch + 1 - start_epoch,
        "stopping_rule": {"mode": watch.mode, "tol": watch.tol,
                          "window": watch.window, "patience": watch.patience},
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }