"""
pilot.py — measure real cost and convergence behaviour BEFORE writing the main pipeline.

Purpose
-------
Two unknowns make guessing MAX_EPOCHS dangerous on this machine:

  1. Epoch cost. A laptop RTX 2050 thermally throttles under sustained load, so
     the first epoch is not representative. A cap extrapolated from a cold start
     underestimates total runtime.

  2. Convergence under full-batch. A3 recorded Batch GD (full batch, lr=1e-3)
     flatlining at ~20% accuracy - random chance - while satisfying the 1e-4
     stopping criterion in 2 epochs. Full-batch + lr=1e-3 + tol=1e-4 may be exactly
     the same pathology. If so, a 10,000-epoch full-batch sweep would burn days
     and produce garbage. That must be caught here, not after the sweep.

What this measures
------------------
  - Step time and peak VRAM for the heaviest model (3-hidden AE, 784-400-256-400-784,
    ~832K params) AND a representative classifier, at full batch and at bs=256.
  - Real learning: does loss actually fall, and does val accuracy rise?
  - Learning-rate sensitivity: whether a larger LR fixes full-batch convergence.
  - Cold vs. warm epoch rate, to estimate throttling headroom.

Output: a printed report plus pilot_findings.json, consumed when fixing
MAX_EPOCHS. Nothing here is part of the final submission.

Usage:
    python pilot.py                      # default sweep
    python pilot.py --epochs 30 --bs full,256
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from data import load_splits, CLASS_NAMES
from models import CLASSIFIER_ARCHS, build_classifier, build_autoencoder
from run_tracker import atomic_write_text

# Same constants the main pipeline will use, so the pilot measures the real thing.
INIT_SEED = 42
TOL = 1e-4
AE_BOTTLENECK = 256  # heaviest required bottleneck
CLASSIFIER_ARCH = "5L_A"  # widest classifier in the 4-arch set (5 hidden layers)


def _sync(device):
    """Force pending GPU work to finish, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mem_mb(device):
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**2, 1)


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def time_forward_backward(model, X, y, device, opt_fn, epochs=12):
    """
    Warm-up pass, then timed epochs. Returns (cold_s, warm_s, loss_trace).

    The cold pass is excluded from the average on purpose: cuDNN autotuning and
    lazy CUDA context setup inflate it, and using it would overstate cost.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    # opt_fn is a FACTORY (torch.optim.Adam), not an instantiated optimizer:
    # passing an already-built optimizer here re-binds the loop variable to
    # None on the next call and then calls methods on None.
    opt = opt_fn(model.parameters(), lr=LR_GRID[0])

    # Warm-up: first call pays context setup + kernel selection.
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = criterion(model(X), y)
    loss.backward()
    opt.step()
    _sync(device)

    times, losses = [], []
    for _ in range(epochs):
        _sync(device)
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    return times[0], sum(times[1:]) / len(times[1:]), losses


def time_ae_step(model, X, opt_fn, epochs=12):
    """Same timing discipline for the autoencoder (MSE loss)."""
    criterion = nn.MSELoss()
    model.to(device)
    opt = opt_fn(model.parameters(), lr=AE_LR)

    model.train()
    opt.zero_grad(set_to_none=True)
    criterion(model(X), X).backward()
    opt.step()
    _sync(device)

    times, losses = [], []
    for _ in range(epochs):
        _sync(device)
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(X), X)
        loss.backward()
        opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    return times[0], sum(times[1:]) / len(times[1:]), losses


def convergence_probe(build_fn, X_tr, y_tr, X_va, y_va, device, lrs, epochs, tag):
    """
    Does it actually learn at each LR? Reports val accuracy reached, and whether
    the 1e-4 stopping rule would fire early while accuracy is still at chance -
    the A3 Batch GD failure mode.
    """
    criterion = nn.CrossEntropyLoss()
    results = []
    for lr in lrs:
        torch.manual_seed(INIT_SEED)
        model = build_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        losses, accs, stopped_early = [], [], None
        for ep in range(epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(X_tr), y_tr)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                acc = (model(X_va).argmax(1) == y_va).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)

            # Replicate the assignment's stopping rule exactly.
            if ep > 0 and abs(losses[-1] - losses[-2]) < TOL:
                stopped_early = ep + 1
                break

        results.append({
            "tag": tag,
            "lr": lr,
            "epochs_run": len(losses),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "val_acc_first": accs[0],
            "val_acc_last": accs[-1],
            "val_acc_max": max(accs),
            "stopped_early_at": stopped_early,
            "chance_level": round(100.0 / len(CLASS_NAMES), 4),
            # "Learned" means beat chance by a real margin. Comparing an accuracy
            # (0-1) against 1.5 x chance_PERCENT (30.0) is always False, which
            # is why the first pilot run wrongly reported that nothing learned.
            "learned": max(accs) > 3.0 * (1.0 / len(CLASS_NAMES)),
        })
        print(f"    lr={lr:<8} ep={len(losses):<4} "
              f"loss {losses[0]:.4f}->{losses[-1]:.4f}  "
              f"val {100*accs[0]:.2f}%->{100*accs[-1]:.2f}%  "
              f"max {100*max(accs):.2f}%  "
              f"early_stop@{stopped_early if stopped_early else '-'}")
    return results


def main():
    global device, LR_GRID, AE_LR

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Group_11")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", default="full,256")
    ap.add_argument("--lrs", default="0.001,0.01,0.05")
    ap.add_argument("--out", default="pilot_findings.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR_GRID = [float(x) for x in args.lrs.split(",")]
    AE_LR = LR_GRID[0]

    print("=" * 72)
    print("PILOT - measuring real cost before fixing MAX_EPOCHS")
    print("=" * 72)
    print(f"device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"torch  : {torch.__version__}")
    print(f"tolerance: {TOL}   pilot epochs: {args.epochs}")

    print("\n[1/4] loading data ...")
    t0 = time.perf_counter()
    data = load_splits(args.data, device=device)
    print(f"  train {tuple(data['X_train'].shape)}  "
          f"val {tuple(data['X_val'].shape)}  test {tuple(data['X_test'].shape)}")
    print(f"  classes {CLASS_NAMES}  loaded in {time.perf_counter()-t0:.1f}s")
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]

    findings = {"device": str(device), "classes": CLASS_NAMES}

    # ── 2. memory + step time at full batch ──────────────────────────────
    print("\n[2/4] memory + step time (full batch, 11,385 samples)")
    _reset_peak(device)
    torch.manual_seed(INIT_SEED)
    ae = build_autoencoder("3hidden", AE_BOTTLENECK).to(device)
    ae_params = sum(p.numel() for p in ae.parameters())
    cold, warm, ae_losses = time_ae_step(ae, X_tr, torch.optim.Adam)
    ae_mem = _peak_mem_mb(device)
    print(f"  3-hidden AE (784-400-{AE_BOTTLENECK}-400-784), {ae_params:,} params")
    print(f"    cold {cold*1000:.1f} ms/epoch, warm {warm*1000:.1f} ms/epoch "
          f"({1/warm:.1f} ep/s)")
    print(f"    loss {ae_losses[0]:.5f} -> {ae_losses[-1]:.5f}")
    print(f"    peak VRAM: {ae_mem} MB")
    findings["ae_3hidden"] = {
        "params": ae_params, "cold_ms": cold*1000, "warm_ms": warm*1000,
        "ep_per_s": 1/warm, "peak_vram_mb": ae_mem,
        "loss_first": ae_losses[0], "loss_last": ae_losses[-1],
    }

    _reset_peak(device)
    torch.manual_seed(INIT_SEED)
    clf = build_classifier(CLASSIFIER_ARCH, input_dim=784).to(device)
    clf_params = sum(p.numel() for p in clf.parameters())
    c_cold, c_warm, _ = time_forward_backward(clf, X_tr, y_tr, device,
                                             torch.optim.Adam)
    clf_mem = _peak_mem_mb(device)
    print(f"  classifier {CLASSIFIER_ARCH}, {clf_params:,} params")
    print(f"    cold {c_cold*1000:.1f} ms/epoch, warm {c_warm*1000:.1f} ms/epoch "
          f"({1/c_warm:.1f} ep/s)")
    print(f"    peak VRAM: {clf_mem} MB")
    findings["classifier_widest"] = {
        "arch": CLASSIFIER_ARCH, "params": clf_params,
        "cold_ms": c_cold*1000, "warm_ms": c_warm*1000,
        "ep_per_s": 1/c_warm, "peak_vram_mb": clf_mem,
    }

    # ── 3. batch-size comparison ─────────────────────────────────────────
    print("\n[3/4] batch-size comparison (does bs change step cost much?)")
    bs_results = []
    for bs in [int(b) for b in args.bs.split(",") if b != "full"]:
        sub = torch.randperm(X_tr.size(0), device=device)[:bs]
        Xs, ys = X_tr[sub], y_tr[sub]
        _reset_peak(device)
        torch.manual_seed(INIT_SEED)
        m = build_classifier("3L_A", input_dim=784).to(device)
        _, warm_s, _ = time_forward_backward(m, Xs, ys, device, torch.optim.Adam)
        steps_per_epoch = (X_tr.size(0) + bs - 1) // bs
        print(f"  bs={bs:<6} warm {warm_s*1000:6.2f} ms/step  "
              f"{steps_per_epoch} steps/epoch  "
              f"=> {warm_s*steps_per_epoch:.2f} s/epoch")
        bs_results.append({"bs": bs, "ms_per_step": warm_s*1000,
                           "steps_per_epoch": steps_per_epoch,
                           "s_per_epoch": warm_s*steps_per_epoch})
    findings["batch_size"] = bs_results

    # ── 4. does full-batch + lr actually learn? ──────────────────────────
    print(f"\n[4/4] convergence check over {args.epochs} epochs "
          f"(the A3 Batch-GD failure mode)")
    print("  chance level = %.2f%%" % (100.0 / len(CLASS_NAMES)))
    conv = []
    print("  full batch:")
    conv += convergence_probe(
        lambda: build_classifier(CLASSIFIER_ARCH, input_dim=784),
        X_tr, y_tr, X_va, y_va, device, LR_GRID, args.epochs, "full")
    print("  bs=256:")
    sub = torch.randperm(X_tr.size(0), device=device)[:256]
    conv += convergence_probe(
        lambda: build_classifier("3L_A", input_dim=784),
        X_tr[sub], y_tr[sub], X_va, y_va, device, LR_GRID, args.epochs, "bs256")
    findings["convergence"] = conv

    # ── verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    ae_rate = findings["ae_3hidden"]["ep_per_s"]
    cl_rate = findings["classifier_widest"]["ep_per_s"]
    any_learned = any(c["learned"] for c in conv)
    for cap in (1000, 2000, 5000, 10000):
        print(f"  {cap:>6} epochs  ->  3-hidden AE {cap/ae_rate/60:6.1f} min   "
              f"widest classifier {cap/cl_rate/60:6.1f} min")
    print(f"\n  anything learned above chance? {'YES' if any_learned else 'NO'}")
    if not any_learned:
        print("  WARNING: nothing beat chance. Full-batch + these LRs does not")
        print("  converge. Reconsider batch size or LR before the main sweep.")
    findings["verdict"] = {"ae_ep_per_s": ae_rate, "clf_ep_per_s": cl_rate,
                           "anything_learned": any_learned}
    atomic_write_text(json.dumps(findings, indent=2), args.out)
    print(f"\nwrote {args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()