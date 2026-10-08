"""
run_tracker.py — long-run progress tracking and atomic checkpointing.

Why this exists
---------------
Assignment-4 runs ~40 training jobs (4 PCA dims x 4 classifier archs, plus
autoencoders, denoisers, and their classifiers) at up to 10,000 epochs each on
a laptop RTX 2050. That is a multi-day workload. Two things must survive an
unplanned stop:

  1. The training state. A3 only checkpointed at convergence, so a crash at
     hour 50 discarded everything. Here we checkpoint every CHECKPOINT_EVERY
     epochs using an atomic write (temp file + os.replace), because a crash
     *during* a plain torch.save() leaves a truncated, unloadable file — which
     is strictly worse than having no checkpoint at all.

  2. The progress. Every run emits a machine-readable PROGRESS line to its log
     so monitor.py can reconstruct state without attaching to the process.

File layout produced under --outdir:
    <outdir>/logs/<run_id>.log        human-readable log (what you'll tail)
    <outdir>/checkpoints/<run_id>.pt  latest resumable state
    <outdir>/status/<run_id>.json     last progress snapshot (atomic)

ETA method
----------
Naive mean-epoch-rate ETAs are wrong on this machine: a laptop GPU under
sustained load thermally throttles, so later epochs are slower than early ones.
A mean-rate ETA therefore reports a confidently incorrect number that only
drifts more optimistic over time. We use a ROLLING window of recent epochs, so
throttling shows up as the ETA lengthening — the honest direction.
"""

import json
import os
import time
from collections import deque

import torch

# Checkpoint cadence. 10 epochs is frequent enough that a crash costs at most
# 10 epochs of work, and cheap enough to be irrelevant next to a 10k-epoch run.
# Deliberately NOT every epoch: a checkpoint for the heavy autoencoders
# (784 -> 400 -> k -> 400 -> 784) is megabytes, so per-epoch saving means
# hundreds of MB of pointless writes over a long run. Raise this per-run if
# epochs are slow enough that 10 is still a meaningful loss of work.
CHECKPOINT_EVERY = 10

# Status-JSON refresh cadence, in epochs. The status file is a tiny overwrite,
# so it is cheap - but there is no reason to touch disk every epoch on ~40 runs.
# Kept aligned with CHECKPOINT_EVERY so status and checkpoint never disagree
# about where a run had reached when it was interrupted.
STATUS_EVERY = 10

# Rolling window for the rate estimate, in epochs.
# Long enough to smooth per-epoch jitter, short enough to react to throttling
# within a few minutes.
RATE_WINDOW = 50


def ensure_dirs(outdir):
    """Create the logs/, checkpoints/, and status/ subdirectories."""
    for sub in ("logs", "checkpoints", "status"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)


def atomic_save(obj, path):
    """
    torch.save to a temp file, then os.replace onto the destination.

    os.replace is atomic on Windows and POSIX (same filesystem), so a reader
    either sees the previous complete file or the new complete file — never a
    half-written one. Writing directly to `path` risks a corrupt checkpoint if
    the process dies mid-write.
    """
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(text, path):
    """Same atomicity guarantee as atomic_save, for small text/JSON files."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class RunTracker:
    """
    Per-run progress reporter and checkpointer.

    Usage:
        tracker = RunTracker("pca_128_3L_A", outdir)
        for epoch in range(max_epochs):
            ...
            tracker.tick(epoch, loss, val_metric)
            # tracker.check() saves automatically every CHECKPOINT_EVERY epochs
        tracker.finish(status="converged", loss=loss)

    The PROGRESS lines written here are what monitor.py parses. Keep the format
    stable — monitor.py keys off the leading 'PROGRESS' token and k=v pairs.
    """

    def __init__(self, run_id, outdir, total_epochs=None,
                 checkpoint_every=CHECKPOINT_EVERY, status_every=STATUS_EVERY):
        self.run_id = run_id
        self.outdir = outdir
        ensure_dirs(outdir)

        # Per-run cadence override. Defaults are module-level, but a slow run
        # (heavy autoencoder, throttled GPU) may want a wider checkpoint interval
        # because there the write cost is dominated by epoch time, not disk I/O.
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.status_every = max(1, int(status_every))

        self.log_path = os.path.join(outdir, "logs", f"{run_id}.log")
        self.ckpt_path = os.path.join(outdir, "checkpoints", f"{run_id}.pt")
        self.status_path = os.path.join(outdir, "status", f"{run_id}.json")

        self.total_epochs = total_epochs

        # Reused across resume: a resumed run must not reset its clock, or the
        # ETA restarts from zero every time the process is restarted.
        self.start_time = time.time()
        self.first_epoch_time = self.start_time

        self.recent = deque(maxlen=RATE_WINDOW)  # (timestamp, epoch) pairs
        self.history = []
        self.done = False
        self.final_status = None

        self.log_fh = open(self.log_path, "a", encoding="utf-8", buffering=1)

    # ── logging ──────────────────────────────────────────────────────────

    def log(self, msg):
        """Human-readable line, timestamped, flushed immediately."""
        self.log_fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def resume_from(self, checkpoint_path):
        """
        Restore epoch counter and loss history from a previous run.

        Preserves the ORIGINAL start_time (mtime of the checkpoint) so that
        elapsed time and ETA remain meaningful across a resume. Without this,
        a run restarted three times would report near-zero elapsed and a wildly
        optimistic ETA.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.history = ckpt.get("history", [])
        if self.history:
            self.first_epoch_time = ckpt.get("start_time", self.start_time)
            last_epoch = self.history[-1]["epoch"]
            self.log(
                f"RESUMED from {os.path.basename(checkpoint_path)} "
                f"at epoch {last_epoch} (elapsed so far: "
                f"{time.time() - self.first_epoch_time:.0f}s)"
            )
            return ckpt
        return None

    # ── progress ─────────────────────────────────────────────────────────

    def _rate(self):
        """
        Epochs per second over the rolling window.

        Returns None until the window has at least 2 samples — a rate from one
        interval is just noise.
        """
        if len(self.recent) < 2:
            return None
        (t0, e0), (t1, e1) = self.recent[0], self.recent[-1]
        dt = t1 - t0
        de = e1 - e0
        if dt <= 0 or de <= 0:
            return None
        return de / dt

    def tick(self, epoch, **metrics):
        """
        Record one epoch's progress, emit a PROGRESS line, and checkpoint
        every CHECKPOINT_EVERY epochs.

        metrics: loss, and optionally val_acc / val_loss / recon_error.
        """
        now = time.time()
        self.recent.append((now, epoch))

        entry = {"epoch": epoch, "wall": now - self.first_epoch_time}
        entry.update(metrics)
        self.history.append(entry)

        rate = self._rate()
        eta = None
        if rate and rate > 0 and self.total_epochs:
            remaining = max(0, self.total_epochs - epoch)
            eta = remaining / rate

        parts = [f"epoch={epoch}", f"wall={entry['wall']:.1f}"]
        for k, v in metrics.items():
            if v is not None:
                parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
        if rate:
            parts.append(f"rate={rate:.3f}ep/s")
        if eta is not None:
            parts.append(f"eta={eta:.0f}s")
        if self.total_epochs:
            parts.append(f"pct={100.0 * epoch / self.total_epochs:.1f}")

        self.log("PROGRESS " + " ".join(parts))

        # Status JSON is throttled to STATUS_EVERY epochs, not written every
        # epoch. The PROGRESS line above is still per-epoch because it is a
        # cheap append to a text file, whereas the JSON is an overwrite that we
        # have no reason to touch every iteration across ~40 concurrent runs.
        if epoch % self.status_every == 0 or epoch == 0:
            self._write_status(epoch, metrics, rate, eta, "running")

        if epoch > 0 and epoch % self.checkpoint_every == 0:
            self.check(epoch)

    def _write_status(self, epoch, metrics, rate, eta, state):
        """Atomically refresh the JSON snapshot that monitor.py reads."""
        snap = {
            "run_id": self.run_id,
            "state": state,
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "elapsed_s": time.time() - self.first_epoch_time,
            "rate_ep_s": rate,
            "eta_s": eta,
            "pid": os.getpid(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
        }
        atomic_write_text(json.dumps(snap, indent=2), self.status_path)

    # ── checkpointing ────────────────────────────────────────────────────

    def check(self, epoch, model=None, optimizer=None, extra=None):
        """
        Atomically save resumable state.

        Only writes when model/optimizer are supplied — otherwise this is a
        no-op, so tick() can call it unconditionally without knowing whether
        the caller wants to checkpoint weights.
        """
        if model is None:
            return False
        payload = {
            "run_id": self.run_id,
            "epoch": epoch,
            "start_time": self.first_epoch_time,
            "history": self.history,
        }
        if model is not None:
            payload["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        atomic_save(payload, self.ckpt_path)
        self.log(f"CHECKPOINT epoch={epoch} -> {os.path.basename(self.ckpt_path)}")
        return True

    def has_checkpoint(self):
        return os.path.exists(self.ckpt_path)

    def finish(self, status, **metrics):
        """
        Mark the run terminal and record its outcome, then close the log.

        Any log() must happen BEFORE calling this: finish() closes the handle,
        and RunTracker.log() deliberately does not reopen it. Reopening silently
        would hide ordering bugs; leaving it to raise does the opposite, which is
        how a warning logged after finish() was caught.
        """
        self.done = True
        self.final_status = status
        self.log(f"FINISH status={status} " + " ".join(f"{k}={v}" for k, v in metrics.items()))
        self._write_status(
            self.history[-1]["epoch"] if self.history else 0,
            metrics, None, None, status,
        )
        self.close()

    def close(self):
        if not self.log_fh.closed:
            self.log_fh.close()


def thin_history(history, max_points=200):
    """
    Reduce a per-epoch history to at most max_points entries, always keeping the
    first and last.

    Per-epoch histories at 10,000 epochs make the result JSONs large enough to be
    unreadable (Task-5 hit 350 KB, Task-6 several MB) while adding no
    information: a convergence curve is fully characterised by its endpoints and
    its shape, and the UNTHINNED history is preserved in every checkpoint. So
    thin for JSON, keep full fidelity on disk.
    """
    if not history or len(history) <= max_points:
        return history
    step = len(history) / float(max_points - 1)
    idx = sorted({0, len(history) - 1} | {int(i * step) for i in range(max_points)})
    return [history[i] for i in idx[:max_points]]


def format_duration(seconds):
    """Seconds -> '2d 4h', '3h 12m', or '45s'. Compact, for tables."""
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600:02d}h"