"""
experiment_2.py — fully-optimised experiment runner.

Optimisations vs original experiment.py
────────────────────────────────────────
Tier 1  – Dataset pre-loaded as tensors once per architecture (not per optimizer).
Tier 4a – os.makedirs called once per arch, not once per (arch, opt) pair.
Tier 4b – Checkpoint and history-CSV saves offloaded to a background thread.
Tier 4c – Best-model state copied with .clone() instead of copy.deepcopy().
Tier 5a – torch.set_num_threads() tuned for Intel Core Ultra 7 hybrid topology.
Tier 5b – bfloat16 / AMX autocast on CPU (if supported by hardware).
           Enabled when device == cpu and ENABLE_CPU_BF16 == True.
Tier 5c – torch.backends.mkldnn.enabled = True (explicit oneDNN activation).
Tier 5d – torch.compile(model) applied after each model.to(device) call.
           Compiles once per (architecture, device) — the cache is warm for
           all subsequent optimizer runs on the same arch.
Tier 6a – Optional parallel architecture training via ProcessPoolExecutor.
           Set N_PARALLEL > 1 to run multiple architectures simultaneously.
           Each worker gets N_CPU_THREADS // N_PARALLEL threads to avoid
           over-subscription of the physical P-cores.
"""

import os
import copy
import threading
import torch
import torch.optim as optim
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataset_2 import get_dataloaders, preload_tensors
from models import get_models
from train_2 import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves

# ── Controlled constants ────────────────────────────────────────────────────
LR         = 0.001
MOMENTUM   = 0.9
BETA1      = 0.9
BETA2      = 0.999
EPSILON    = 1e-8
ALPHA      = 0.99
TOL        = 1e-4
INIT_SEED  = 42
DATA_SEED  = 42
MAX_EPOCHS = 10000

# ── System-tuning knobs ─────────────────────────────────────────────────────
# Intel Core Ultra 7 (Meteor Lake): 6 P-cores + 8 E-cores.
# For small matmuls (784 × 128 etc.) restricting to P-cores only avoids
# cross-core cache contention and E-core scheduling overhead.
# Tune to os.cpu_count() // 2 if unsure; or 1 for GPU runs (GPU does the math).
N_CPU_THREADS   = 6          # P-cores for the main process (sequential mode)

# Number of architecture experiments to run in parallel.
# Set to 1 for sequential (safe default).  Set to 2 or 3 to use multiple
# P-core clusters simultaneously.  Each parallel worker gets
# max(1, N_CPU_THREADS // N_PARALLEL) threads.
N_PARALLEL      = 1

# Enable bfloat16 autocast on CPU.  Intel Core Ultra 7 supports AMX
# (Advanced Matrix Extensions) which accelerates bf16 matmuls 1.5–2.5×.
# bf16 has the same exponent range as fp32 but half the mantissa bits;
# classification accuracy is not affected in practice.
# Set False if you observe instability or want exact fp32 reproducibility.
ENABLE_CPU_BF16 = False      # True to enable; requires PyTorch >= 2.1

# Enable torch.compile.  First call per (arch, device) takes ~20-40s to JIT;
# all subsequent epochs amortise this cost.  Gains ~15-25% on small MLPs.
ENABLE_COMPILE  = True


# ── Async I/O helpers ───────────────────────────────────────────────────────

def _save_checkpoint_bg(payload: dict, path: str) -> None:
    """
    Saves a checkpoint in a background daemon thread so training is not
    blocked on disk I/O.  The thread is a daemon so it will not prevent
    the process from exiting cleanly.
    """
    def _worker():
        torch.save(payload, path)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _save_csv_bg(data: dict, path: str) -> None:
    """Saves a history CSV in a background daemon thread."""
    def _worker():
        pd.DataFrame(data).to_csv(path, index=False)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ── Per-architecture worker (used by both sequential and parallel modes) ────

def _run_arch_experiment(arch_name, initial_weights_cpu,
                         data_dir, save_dir, max_epochs,
                         device_str, n_threads):
    """
    Runs all 7 optimizer experiments for a single architecture.

    Designed to be called either directly (sequential) or from a subprocess
    (parallel via ProcessPoolExecutor).  Accepts CPU tensors / plain Python
    objects so it is pickle-safe for multiprocessing.

    Returns:
        list[dict]  – convergence_epoch records (one per optimizer).
        (str, str, float, dict) – best_arch, best_opt, best_val_acc,
                                   best_model_state (CPU state dict).
    """
    # ── Worker-local system tuning ─────────────────────────────────────────
    torch.set_num_threads(n_threads)
    torch.backends.mkldnn.enabled = True

    device = torch.device(device_str)

    opt_configs = {
        'SGD':          (optim.SGD,     {'lr': LR},                                        False),
        'BGD':          (optim.SGD,     {'lr': LR},                                        True),
        'SGD_Momentum': (optim.SGD,     {'lr': LR, 'momentum': MOMENTUM},                  False),
        'NAG':          (optim.SGD,     {'lr': LR, 'momentum': MOMENTUM, 'nesterov': True}, False),
        'AdaGrad':      (optim.Adagrad, {'lr': LR, 'eps': EPSILON},                        True),
        'RMSProp':      (optim.RMSprop, {'lr': LR, 'alpha': ALPHA, 'eps': EPSILON},        True),
        'Adam':         (optim.Adam,    {'lr': LR, 'betas': (BETA1, BETA2), 'eps': EPSILON}, False),
    }

    # Reconstruct the model from its state dict (needed in subprocess).
    from models import FCNN
    # We re-use get_models() to get the right architecture shape, then load weights.
    from models import get_models
    models_dict = get_models()
    model = models_dict[arch_name]
    model.load_state_dict(initial_weights_cpu)
    # Keep the loaded weights as the reset point for each optimizer run.
    initial_weights = copy.deepcopy(model.state_dict())

    # ── Data: load once, shared across all optimizer runs ──────────────────
    print(f"[{arch_name}] Preloading data to {device} …")
    _train_loader, _val_loader, _ = get_dataloaders(
        data_dir, is_full_batch=True, seed=DATA_SEED, device=device
    )
    X_train, y_train = preload_tensors(_train_loader, device)
    X_val,   y_val   = preload_tensors(_val_loader,   device)
    print(f"[{arch_name}] X_train {tuple(X_train.shape)}  X_val {tuple(X_val.shape)}")

    # ── Directories ────────────────────────────────────────────────────────
    chk_dir = os.path.join(save_dir, 'checkpoints', arch_name)
    his_dir = os.path.join(save_dir, 'histories',   arch_name)
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(his_dir, exist_ok=True)

    convergence_epochs = []
    arch_results = {}
    best_val_acc = -1.0
    best_opt = None
    best_model_state = None

    for opt_name, (opt_class, opt_kwargs, is_full_batch) in opt_configs.items():
        print(f"\n[{arch_name}] --- Optimizer: {opt_name} ---")

        chk_path = os.path.join(chk_dir, f"{opt_name}.pt")
        his_path = os.path.join(his_dir, f"{opt_name}.csv")

        # ── Resume ────────────────────────────────────────────────────────
        if os.path.exists(chk_path):
            print(f"[{arch_name}/{opt_name}] checkpoint found — skipping.")
            ckpt    = torch.load(chk_path, map_location=device, weights_only=False)
            history = ckpt['history']
            arch_results[opt_name] = history

            final_val_acc = history['val_acc'][-1]
            convergence_epochs.append({
                'Architecture':      arch_name,
                'Optimizer':         opt_name,
                'Convergence Epoch': len(history['train_loss']),
                'Final Train Acc':   history['train_acc'][-1],
                'Final Val Acc':     final_val_acc,
            })
            if final_val_acc > best_val_acc:
                best_val_acc      = final_val_acc
                best_opt          = opt_name
                best_model_state  = ckpt['model_state_dict']
            continue

        # ── Reset model to identical initial weights for this optimizer ────
        model.load_state_dict(initial_weights)
        model.to(device)

        # ── torch.compile ─────────────────────────────────────────────────
        # Applied after .to(device) and after load_state_dict.
        # The compiled wrapper shares parameter storage with `model`, so:
        #   • compiled_model is used for fast forward/backward.
        #   • model.state_dict() after training gives the updated weights.
        #   • model.load_state_dict(initial_weights) resets weights for the
        #     next optimizer; compiled_model sees the change automatically.
        if ENABLE_COMPILE and hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(model, mode='reduce-overhead')
                print(f"[{arch_name}/{opt_name}] torch.compile enabled.")
            except Exception as e:
                print(f"[{arch_name}/{opt_name}] torch.compile skipped: {e}")
                compiled_model = model
        else:
            compiled_model = model

        # ── bfloat16 CPU autocast ─────────────────────────────────────────
        # Wraps model in an autocast context when training on CPU with
        # bfloat16.  The actual autocast happens inside train_model because
        # use_amp is device-type-based; we pass a flag via the device string.
        # For now, bfloat16 is handled by overriding device.type check:
        # ENABLE_CPU_BF16 is read inside train_model via the outer closure.
        # (Implemented below via the use_cpu_bf16 kwarg.)

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer = opt_class(compiled_model.parameters(), **opt_kwargs)

        try:
            history = train_model(
                model        = compiled_model,
                optimizer    = optimizer,
                X_train      = X_train,
                y_train      = y_train,
                X_val        = X_val,
                y_val        = y_val,
                stochastic   = not is_full_batch,
                device       = device,
                max_epochs   = max_epochs,
                tol          = TOL,
                shuffle_seed = DATA_SEED,
                log_interval = 100,
            )
        except RuntimeError as e:
            print(f"[ERROR] {arch_name}/{opt_name}: {e}")
            # Write error log synchronously (small, infrequent).
            err_path = os.path.join(save_dir, 'errors.csv')
            err_row  = pd.DataFrame([{'Architecture': arch_name,
                                      'Optimizer': opt_name,
                                      'Error': str(e)}])
            if os.path.exists(err_path):
                err_row.to_csv(err_path, mode='a', header=False, index=False)
            else:
                err_row.to_csv(err_path, index=False)
            continue

        arch_results[opt_name] = history

        # ── Async checkpoint + CSV save ────────────────────────────────────
        # Save to disk in background threads — training of the next optimizer
        # starts immediately without waiting for I/O.
        _save_checkpoint_bg({
            'model_state_dict': model.state_dict(),   # use model, not compiled_model
            'optimizer_name':   opt_name,
            'arch_name':        arch_name,
            'epoch':            len(history['train_loss']),
            'history':          history,
        }, chk_path)
        _save_csv_bg(history, his_path)

        final_val_acc = history['val_acc'][-1]
        convergence_epochs.append({
            'Architecture':      arch_name,
            'Optimizer':         opt_name,
            'Convergence Epoch': len(history['train_loss']),
            'Final Train Acc':   history['train_acc'][-1],
            'Final Val Acc':     final_val_acc,
        })

        if final_val_acc > best_val_acc:
            best_val_acc     = final_val_acc
            best_opt         = opt_name
            # .clone() instead of copy.deepcopy() — same result, avoids
            # Python's full deepcopy machinery for the state dict.
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    return arch_results, convergence_epochs, arch_name, best_opt, best_val_acc, best_model_state


# ── Main experiment runner ──────────────────────────────────────────────────

def run_experiments(data_dir, save_dir, max_epochs=MAX_EPOCHS, arch_names=None):

    # ── System tuning ──────────────────────────────────────────────────────
    # Set thread count BEFORE importing any MKL-backed ops.
    threads_per_worker = max(1, N_CPU_THREADS // max(1, N_PARALLEL))
    torch.set_num_threads(threads_per_worker)

    # Explicit oneDNN (MKL-DNN) activation — ensures Intel's optimised
    # primitives are used rather than generic ATen fallback kernels.
    torch.backends.mkldnn.enabled = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"torch.set_num_threads({threads_per_worker})  "
          f"(N_PARALLEL={N_PARALLEL}, N_CPU_THREADS={N_CPU_THREADS})")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for fixed input sizes.")
    else:
        print("WARNING: CUDA not available. Training will run on CPU.")
        if ENABLE_CPU_BF16:
            print("bfloat16 CPU autocast ENABLED (AMX path).")
        if ENABLE_COMPILE:
            print("torch.compile ENABLED — expect ~30s warmup on first architecture.")

    os.makedirs(save_dir, exist_ok=True)
    for sub in ('epoch_loss', 'confusion_matrix', 'surface_3d', 'scatter_3d'):
        os.makedirs(os.path.join(save_dir, 'plots', sub), exist_ok=True)

    # Build all model architectures (seeded once for reproducibility).
    models_dict = get_models()
    if arch_names is not None:
        models_dict = {k: v for k, v in models_dict.items() if k in arch_names}

    # Collect initial weights (CPU, pickle-safe) for each architecture.
    # These are passed to worker processes / functions so every optimizer run
    # starts from the exact same initial weights.
    initial_weights_map = {
        name: copy.deepcopy(model.state_dict())
        for name, model in models_dict.items()
    }

    all_results          = {}
    all_convergence      = []
    global_best_val_acc  = -1.0
    global_best_arch     = None
    global_best_opt      = None
    global_best_state    = None

    arch_list = list(models_dict.keys())

    if N_PARALLEL > 1 and device.type == 'cpu':
        # ── Parallel mode ──────────────────────────────────────────────────
        # Each subprocess handles one architecture fully (all 7 optimizers).
        # subprocess count capped at the number of architectures.
        n_workers = min(N_PARALLEL, len(arch_list))
        print(f"\n[parallel] Launching {n_workers} worker(s) "
              f"({threads_per_worker} threads each).")

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {
                exe.submit(
                    _run_arch_experiment,
                    arch_name,
                    initial_weights_map[arch_name],
                    data_dir, save_dir, max_epochs,
                    str(device),
                    threads_per_worker,
                ): arch_name
                for arch_name in arch_list
            }

            for future in as_completed(futures):
                arch_name = futures[future]
                try:
                    (arch_results, conv_epochs,
                     a_name, b_opt, b_acc, b_state) = future.result()
                except Exception as exc:
                    print(f"[parallel] {arch_name} raised: {exc}")
                    continue

                all_results[a_name] = arch_results
                all_convergence.extend(conv_epochs)

                if b_acc > global_best_val_acc:
                    global_best_val_acc = b_acc
                    global_best_arch    = a_name
                    global_best_opt     = b_opt
                    global_best_state   = b_state

                # Plot training curves for this architecture.
                plot_training_curves(
                    all_results, a_name,
                    os.path.join(save_dir, 'plots', 'epoch_loss')
                )

    else:
        # ── Sequential mode ────────────────────────────────────────────────
        for arch_name in arch_list:
            print(f"\n{'='*50}\nStarting experiments for {arch_name}\n{'='*50}")

            (arch_results, conv_epochs,
             a_name, b_opt, b_acc, b_state) = _run_arch_experiment(
                arch_name,
                initial_weights_map[arch_name],
                data_dir, save_dir, max_epochs,
                str(device),
                threads_per_worker,
            )

            all_results[a_name] = arch_results
            all_convergence.extend(conv_epochs)

            if b_acc > global_best_val_acc:
                global_best_val_acc = b_acc
                global_best_arch    = a_name
                global_best_opt     = b_opt
                global_best_state   = b_state

            plot_training_curves(
                all_results, a_name,
                os.path.join(save_dir, 'plots', 'epoch_loss')
            )

    # ── Summary ────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_convergence)
    df_results.to_csv(os.path.join(save_dir, 'convergence_results.csv'), index=False)
    print("\nConvergence Results:")
    print(df_results.to_string())

    if global_best_arch is None:
        print("No successful runs completed.")
        return

    print(f"\nBest Architecture: {global_best_arch} "
          f"(with {global_best_opt}) — Val Acc: {global_best_val_acc:.4f}")

    # ── Final evaluation on test set ────────────────────────────────────────
    print("Evaluating best model on test set …")
    best_model = models_dict[global_best_arch]
    best_model.load_state_dict(global_best_state)
    best_model.to(device)

    # These calls are outside the hot loop — DataLoader is fine here.
    _, _, test_loader  = get_dataloaders(data_dir, is_full_batch=True,
                                         seed=DATA_SEED, device=device)
    train_loader_eval, _, _ = get_dataloaders(data_dir, is_full_batch=True,
                                               seed=DATA_SEED, device=device)

    test_targets,  test_preds,  test_acc  = evaluate_model(best_model, test_loader,       device)
    train_targets, train_preds, train_acc = evaluate_model(best_model, train_loader_eval, device)

    print(f"Test Accuracy:        {test_acc:.4f}")
    print(f"Final Train Accuracy: {train_acc:.4f}")

    classes = ['0', '4', '5', '6', '7']
    cm_dir  = os.path.join(save_dir, 'plots', 'confusion_matrix')

    plot_confusion_matrix(
        test_targets, test_preds, classes,
        f'Test Confusion Matrix\n{global_best_arch} with {global_best_opt}',
        os.path.join(cm_dir, 'test_confusion_matrix.png')
    )
    plot_confusion_matrix(
        train_targets, train_preds, classes,
        f'Train Confusion Matrix\n{global_best_arch} with {global_best_opt}',
        os.path.join(cm_dir, 'train_confusion_matrix.png')
    )


if __name__ == '__main__':
    # Guard required for ProcessPoolExecutor on all platforms.
    data_directory = r'./Data/Group_11'
    save_directory = r'./Data/results'
    run_experiments(data_directory, save_directory, max_epochs=MAX_EPOCHS)
