import torch
import torch.optim as optim
import os
import copy
import pandas as pd

from dataset_2 import get_dataloaders, preload_tensors
from models import get_models
from train_2 import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves

# ---------------------------------------------------------------------------
# Controlled constants — no magic numbers, all values explicit and documented
# ---------------------------------------------------------------------------
LR = 0.001
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
ALPHA = 0.99
TOL = 1e-4
INIT_SEED = 42
DATA_SEED = 42
MAX_EPOCHS = 10000


def run_experiments(data_dir, save_dir, max_epochs=MAX_EPOCHS, arch_names=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for fixed input sizes.")
    else:
        print("WARNING: CUDA not available. Training will run on CPU and be much slower.")
        print("Check: Runtime > Change runtime type > Hardware accelerator > GPU")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'epoch_loss'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'surface_3d'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'scatter_3d'), exist_ok=True)

    models_dict = get_models()

    if arch_names is not None:
        models_dict = {k: v for k, v in models_dict.items() if k in arch_names}

    opt_configs = {
        'SGD':          (optim.SGD,     {'lr': LR},                                       False),
        'BGD':          (optim.SGD,     {'lr': LR},                                       True),
        'SGD_Momentum': (optim.SGD,     {'lr': LR, 'momentum': MOMENTUM},                 False),
        'NAG':          (optim.SGD,     {'lr': LR, 'momentum': MOMENTUM, 'nesterov': True}, False),
        'AdaGrad':      (optim.Adagrad, {'lr': LR, 'eps': EPSILON},                       True),
        'RMSProp':      (optim.RMSprop, {'lr': LR, 'alpha': ALPHA, 'eps': EPSILON},       True),
        'Adam':         (optim.Adam,    {'lr': LR, 'betas': (BETA1, BETA2), 'eps': EPSILON}, False),
    }

    results = {}
    convergence_epochs = []
    errors = []

    best_arch = None
    best_opt = None
    best_val_acc = -1
    best_model_state = None

    for arch_name, model in models_dict.items():
        print(f"\n{'='*50}\nStarting experiments for {arch_name}\n{'='*50}")
        results[arch_name] = {}

        # Save initial random weights once per architecture
        initial_weights = copy.deepcopy(model.state_dict())

        # ── CHANGE 1: Load data ONCE per architecture, not once per optimizer ──
        # We call get_dataloaders with is_full_batch=True so the entire dataset
        # is read through the transform pipeline in one pass.  preload_tensors()
        # then concatenates the chunks into a single (N, 784) device tensor.
        # This call replaces the 7 separate get_dataloaders() calls that the
        # original experiment.py made (one per optimizer) for this architecture.
        print(f"[preload] Loading and pinning dataset to {device} …")
        _full_train_loader, _full_val_loader, test_loader = get_dataloaders(
            data_dir,
            is_full_batch=True,
            seed=DATA_SEED,
            device=device
        )

        # ── CHANGE 2: Drain DataLoaders into raw tensors ────────────────────
        # After this point, PIL / torchvision transforms / collate never run
        # again for this architecture — all optimizer runs share these tensors.
        X_train, y_train = preload_tensors(_full_train_loader, device)
        X_val,   y_val   = preload_tensors(_full_val_loader,   device)
        print(f"[preload] X_train {tuple(X_train.shape)}, "
              f"X_val {tuple(X_val.shape)}  — on {X_train.device}")

        os.makedirs(os.path.join(save_dir, 'checkpoints', arch_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'histories',   arch_name), exist_ok=True)

        for opt_name, (opt_class, opt_kwargs, is_full_batch) in opt_configs.items():
            print(f"\n--- Optimizer: {opt_name} ---")

            checkpoint_dir  = os.path.join(save_dir, 'checkpoints', arch_name)
            history_dir     = os.path.join(save_dir, 'histories',   arch_name)
            checkpoint_path = os.path.join(checkpoint_dir, f"{opt_name}.pt")
            history_path    = os.path.join(history_dir,    f"{opt_name}.csv")

            # Resume from checkpoint if available
            if os.path.exists(checkpoint_path):
                print(f"[checkpoint] Skipping {arch_name}/{opt_name}, checkpoint already exists.")
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                history = ckpt['history']
                results[arch_name][opt_name] = history

                conv_epoch      = len(history['train_loss'])
                final_train_acc = history['train_acc'][-1]
                final_val_acc   = history['val_acc'][-1]

                convergence_epochs.append({
                    'Architecture':    arch_name,
                    'Optimizer':       opt_name,
                    'Convergence Epoch': conv_epoch,
                    'Final Train Acc': final_train_acc,
                    'Final Val Acc':   final_val_acc
                })

                if final_val_acc > best_val_acc:
                    best_val_acc        = final_val_acc
                    best_arch           = arch_name
                    best_opt            = opt_name
                    best_model_state    = ckpt['model_state_dict']
                continue

            # Reload exact same initial weights for this optimizer run
            model.load_state_dict(initial_weights)
            model.to(device)

            # Build optimizer AFTER model is on the target device
            optimizer = opt_class(model.parameters(), **opt_kwargs)

            # ── CHANGE 3: Pass pre-loaded tensors to train_model ────────────
            # stochastic=True  → sample-by-sample loop (batch_size=1 SGD etc.)
            # stochastic=False → single full forward/backward (BGD, AdaGrad, RMSProp)
            # No DataLoader is passed; train_model_2 uses the tensor fast-path.
            try:
                history = train_model(
                    model        = model,
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
                )
            except RuntimeError as e:
                msg = f"[ERROR] {arch_name}/{opt_name}: {e}"
                print(msg)
                errors.append({'Architecture': arch_name, 'Optimizer': opt_name, 'Error': str(e)})
                pd.DataFrame(errors).to_csv(os.path.join(save_dir, 'errors.csv'), index=False)
                continue

            results[arch_name][opt_name] = history

            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_name':   opt_name,
                'arch_name':        arch_name,
                'epoch':            len(history['train_loss']),
                'history':          history
            }, checkpoint_path)

            # Save history CSV
            pd.DataFrame(history).to_csv(history_path, index=False)

            conv_epoch      = len(history['train_loss'])
            final_train_acc = history['train_acc'][-1]
            final_val_acc   = history['val_acc'][-1]

            convergence_epochs.append({
                'Architecture':      arch_name,
                'Optimizer':         opt_name,
                'Convergence Epoch': conv_epoch,
                'Final Train Acc':   final_train_acc,
                'Final Val Acc':     final_val_acc
            })

            if final_val_acc > best_val_acc:
                best_val_acc     = final_val_acc
                best_arch        = arch_name
                best_opt         = opt_name
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Plot training curves for this architecture
        plot_training_curves(results, arch_name, os.path.join(save_dir, 'plots', 'epoch_loss'))

    df_results = pd.DataFrame(convergence_epochs)
    df_results.to_csv(os.path.join(save_dir, 'convergence_results.csv'), index=False)
    print("\nConvergence Results:")
    print(df_results.to_string())

    if best_arch is None:
        print("No successful runs completed.")
        return

    print(f"\nBest Architecture: {best_arch} (with {best_opt}) - Val Acc: {best_val_acc:.4f}")

    print("Evaluating best model on test set...")
    best_model = models_dict[best_arch]
    best_model.load_state_dict(best_model_state)
    best_model.to(device)

    # Final evaluation uses DataLoaders (called once, outside the hot loop —
    # no need to optimise these calls).
    _, _, test_loader_eval = get_dataloaders(data_dir, is_full_batch=True, seed=DATA_SEED, device=device)
    test_targets, test_preds, test_acc = evaluate_model(best_model, test_loader_eval, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    train_loader_eval, _, _ = get_dataloaders(data_dir, is_full_batch=True, seed=DATA_SEED, device=device)
    train_targets, train_preds, train_acc = evaluate_model(best_model, train_loader_eval, device)
    print(f"Final Train Accuracy for Best Model: {train_acc:.4f}")

    classes = ['0', '4', '5', '6', '7']

    plot_confusion_matrix(
        test_targets, test_preds, classes,
        f'Test Confusion Matrix\n{best_arch} with {best_opt}',
        os.path.join(save_dir, 'plots', 'confusion_matrix', 'test_confusion_matrix.png')
    )

    plot_confusion_matrix(
        train_targets, train_preds, classes,
        f'Train Confusion Matrix\n{best_arch} with {best_opt}',
        os.path.join(save_dir, 'plots', 'confusion_matrix', 'train_confusion_matrix.png')
    )


if __name__ == '__main__':
    data_directory = r'./Data/Group_11'
    save_directory = r'./Data/results'
    run_experiments(data_directory, save_directory, max_epochs=10000)
