"""
monitor.py — live progress monitor for long-running Assignment-4 training jobs.

Read-only: this script never touches the training process. It reconstructs
state by reading the JSON snapshots in <outdir>/status/, so you can run it,
Ctrl-C it, or run it from a different terminal with zero effect on training.

Usage:
    python monitor.py                      # default outdir
    python monitor.py --outdir results
    python monitor.py --watch               # refresh every 5s
    python monitor.py --watch --interval 2
    python monitor.py --run pca_128_3L_A   # single run detail
    python monitor.py --gpu                # also show GPU state

Why a rolling-window ETA matters here
-------------------------------------
On a laptop RTX 2050, sustained load causes thermal throttling, so later epochs
run slower than early ones. An ETA computed from the mean epoch rate over the
whole run reports a number that only ever gets more optimistic, which is
exactly backwards. This monitor shows the current-window rate, so throttling
surfaces as a lengthening ETA and higher temperature — you see it happening
while you can still react to it.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tracker import format_duration

DEFAULT_OUTDIR = "results"

# Ordered by pipeline position so the display reads in execution order.
STATE_ORDER = {"running": 0, "checkpointed": 1, "converged": 2,
               "max_epochs": 3, "failed": 4, "skipped": 5}


def read_statuses(outdir):
    """Load every status snapshot. Returns (statuses, unreadable_count)."""
    status_dir = os.path.join(outdir, "status")
    if not os.path.isdir(status_dir):
        return [], 0
    statuses, bad = [], 0
    for name in os.listdir(status_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(status_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                statuses.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            # A .json.tmp mid-replace can land here; atomic writes mean it is
            # transient, so just count it rather than erroring out.
            bad += 1
    return statuses, bad


def gpu_state():
    """GPU temperature / utilization / memory via nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        t, util, mem_used, mem_total, power = [
            x.strip() for x in out.stdout.strip().split(",")
        ]
        return {
            "temp": int(t),
            "util": int(util),
            "mem_used": int(mem_used),
            "mem_total": int(mem_total),
            "power": power,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def proc_alive(pid):
    """Is the training process still running? Best-effort, Windows-safe."""
    if not pid:
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def fmt_row(s):
    """One status snapshot -> a fixed-width table row."""
    state = s.get("state", "?")
    epoch, total = s.get("epoch"), s.get("total_epochs")
    prog = f"{epoch}/{total}" if total else str(epoch)
    rate = s.get("rate_ep_s")
    rate_s = f"{rate:.3f}" if rate else "-"
    alive = proc_alive(s.get("pid"))
    # A snapshot says "running" but the process is gone => crashed or killed
    # since the last epoch. Surfacing this is the monitor's main value-add.
    mark = "" if alive is None or alive else "  <-- PROCESS GONE"
    m = s.get("metrics") or {}
    extra = ""
    for key in ("val_acc", "val_loss", "recon_error"):
        if key in m and m[key] is not None:
            extra = f"{key}={m[key]:.4f}"
            break
    return (f"{s.get('run_id', '?'):<26} {state:<12} {prog:<14} "
            f"{format_duration(s.get('elapsed_s')):>9} "
            f"{format_duration(s.get('eta_s')):>9} {rate_s:>7}  {extra}{mark}")


HEADER = (f"{'RUN':<26} {'STATE':<12} {'EPOCH':<14} {'ELAPSED':>9} "
          f"{'ETA':>9} {'EP/S':>7}  METRIC")


def render(outdir, show_gpu, verbose):
    statuses, bad = read_statuses(outdir)
    if not statuses:
        print(f"No status files under {os.path.join(outdir, 'status')}")
        print("Either nothing has started yet, or --outdir is wrong.")
        return

    if verbose:
        statuses.sort(key=lambda s: s.get("epoch", 0))
    else:
        statuses.sort(key=lambda s: (
            STATE_ORDER.get(s.get("state"), 9),
            -(s.get("rate_ep_s") or 0),
        ))

    running = [s for s in statuses if s.get("state") == "running"]
    done = [s for s in statuses if s.get("state") in
            ("converged", "max_epochs", "failed", "skipped")]

    print(HEADER)
    print("-" * len(HEADER))
    for s in statuses:
        print(fmt_row(s))
    if bad:
        print(f"\n({bad} snapshot(s) mid-write; retry in a moment)")

    print(f"\n{len(running)} running, {len(done)} finished, {len(statuses)} total")

    if show_gpu:
        g = gpu_state()
        if g:
            mem_pct = 100.0 * g["mem_used"] / max(1, g["mem_total"])
            print(f"GPU: {g['temp']}C  util {g['util']}%  "
                  f"mem {g['mem_used']}/{g['mem_total']} MiB ({mem_pct:.0f}%)  "
                  f"power {g['power']} W")
            # Throttling is the failure mode that silently inflates a long run.
            if g["temp"] >= 85:
                print("  WARNING: GPU at/above 85C — expect throttling. "
                      "ETAs will lengthen. Consider cleaning vents.")
        else:
            print("GPU: nvidia-smi unavailable")


def watch(outdir, show_gpu, interval):
    try:
        while True:
            # Clear screen when attached to a TTY; plain scroll when piped to
            # a file, so redirecting this to a log stays readable.
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            render(outdir, show_gpu, verbose=False)
            print(f"\nRefreshing every {interval}s — Ctrl-C to exit.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training continues unaffected.")


def detail(outdir, run_id):
    """Single-run view, including the tail of its log."""
    path = os.path.join(outdir, "status", f"{run_id}.json")
    if not os.path.exists(path):
        print(f"No status for '{run_id}' at {path}")
        return
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
    print(f"Run:      {s.get('run_id')}")
    print(f"State:    {s.get('state')}")
    print(f"Epoch:    {s.get('epoch')} / {s.get('total_epochs')}")
    print(f"Elapsed:  {format_duration(s.get('elapsed_s'))}")
    print(f"ETA:      {format_duration(s.get('eta_s'))}")
    print(f"Rate:     {s.get('rate_ep_s')} ep/s")
    print(f"PID:      {s.get('pid')} (alive: {proc_alive(s.get('pid'))})")
    print(f"Metrics:  {s.get('metrics')}")
    print(f"Updated:  {s.get('updated')}")

    log_path = os.path.join(outdir, "logs", f"{run_id}.log")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as fh:
            tail = fh.readlines()[-25:]
        print(f"\n--- last {len(tail)} log lines ({os.path.basename(log_path)}) ---")
        for line in tail:
            print("  " + line.rstrip())


def main():
    ap = argparse.ArgumentParser(
        description="Live progress monitor for Assignment-4 training runs.")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help=f"run output directory (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--watch", action="store_true",
                    help="refresh continuously until Ctrl-C")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between refreshes (default: 5)")
    ap.add_argument("--run", metavar="RUN_ID",
                    help="show detail for one run and exit")
    ap.add_argument("--gpu", action="store_true",
                    help="include GPU temperature/utilization/memory")
    ap.add_argument("--verbose", action="store_true",
                    help="sort by epoch instead of by state")
    args = ap.parse_args()

    if args.run:
        detail(args.outdir, args.run)
    elif args.watch:
        watch(args.outdir, args.gpu, args.interval)
    else:
        render(args.outdir, args.gpu, args.verbose)
        print("\nTip: --watch for live, --run <id> for detail, --gpu for temps.")


if __name__ == "__main__":
    main()