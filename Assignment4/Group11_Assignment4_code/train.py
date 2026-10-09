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
STOP_PATIENCE = 15


CLASSIFIER_LR = 0.01
AUTOENCODER_LR = 0.001
AE_REL_TOL = 1e-3
AE_PLATEAU_WINDOW = 50
MAX_EPOCHS = 10000
CHECKPOINT_EVERY = 10
LOG_EVERY = 100

CLASSIFIER_OPT = torch.optim.Adam
AE_OPT = torch.optim.Adam


def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


CHANCE = 1.0 / 5


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
        self.min_epochs = min_epochs
        self.mode = mode
        self.window = window or patience
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
                "mode": self.mode, "window": self.window,
                "min_epochs": self.min_epochs, "triggered": self.triggered,
                "best_hist": list(self._best_hist)}

    def load_state_dict(self, sd):
        self.streak = sd.get("streak", 0)
        self.prev_loss = sd.get("prev_loss")
        self.best_loss = sd.get("best_loss", float("inf"))
        self.improved = sd.get("improved", False)
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
    if resume and tracker is not None and tracker.has_checkpoint():
        ckpt = torch.load(tracker.ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if tracker.history:
                tracker.first_epoch_time = ckpt.get("start_time", tracker.first_epoch_time)
            if ckpt.get("history"):
                tracker.history = ckpt["history"]
            tracker.log(f"RESUME {arch_name} from epoch {start_epoch} "
                        f"({len(tracker.history)} epochs of history restored)")

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
    watch = _ConvergenceWatch(tol=tol, patience=patience, mode="abs",
                             min_epochs=0)
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

        if tracker is not None and (
            (epoch + 1) % checkpoint_every == 0 or epoch == max_epochs - 1
        ):
            _save_checkpoint(tracker, model, optimizer, epoch, device,
                             extra={"arch_name": arch_name, "lr": lr},
                             watch=watch)

        if watch.update(loss_val, epoch):
            stopped_early = True
            if tracker is not None:
                tracker.log(
                    f"CONVERGED {arch_name} at epoch {epoch + 1} "
                    f"(|dL| < {tol:g} for {patience} consecutive epochs)")
            break

    final_val = _accuracy(model.eval()(X_va).detach(), y_va)
    degenerate = flag_degenerate(final_val)
    if degenerate:
        state = "degenerate"
    else:
        state = "converged" if stopped_early else "max_epochs"

    if tracker is not None and degenerate:
        tracker.log(
            f"WARNING {arch_name}: finished at {100 * final_val:.2f}% "
            f"val accuracy (chance = 20%). Training collapsed - see the "
            f"vanishing-gradient note in train.flag_degenerate. Excluded "
            f"from best-architecture selection.")
    if tracker is not None:
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
            if ckpt.get("history"):
                tracker.history = ckpt["history"]
            tracker.log(f"RESUME {run_name} from epoch {start_epoch} "
                        f"({len(tracker.history)} epochs of history restored)")

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
    watch = _ConvergenceWatch(tol=AE_REL_TOL, patience=patience, mode="plateau",
                             window=AE_PLATEAU_WINDOW, min_epochs=0)
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

        if watch.update(loss_val, epoch):
            stopped_early = True
            if tracker is not None:
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

    recon = {
        split: reconstruct_error(model, data[f"X_{split}"])
        for split in ("train", "val", "test")
    }
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
