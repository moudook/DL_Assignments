import torch
import torch.optim as optim
import os
import copy
import pandas as pd

from dataset import get_dataloaders
from models import get_models
from train import train_model
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


def run_experiments(data_dir, save_dir, max_epochs=MAX_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'epoch_loss'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'surface_3d'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots', 'scatter_3d'), exist_ok=True)

    models_dict = get_models()

    opt_configs = {
        'SGD': (optim.SGD, {'lr': LR}, False),
        'BGD': (optim.SGD, {'lr': LR}, True),
        'SGD_Momentum': (optim.SGD, {'lr': LR, 'momentum': MOMENTUM}, False),
        'NAG': (optim.SGD, {'lr': LR, 'momentum': MOMENTUM, 'nesterov': True}, False),
        'AdaGrad': (optim.Adagrad, {'lr': LR, 'eps': EPSILON}, True),
        'RMSProp': (optim.RMSprop, {'lr': LR, 'alpha': ALPHA, 'eps': EPSILON}, True),
        'Adam': (optim.Adam, {'lr': LR, 'betas': (BETA1, BETA2), 'eps': EPSILON}, False)
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

        for opt_name, (opt_class, opt_kwargs, is_full_batch) in opt_configs.items():
            print(f"\n--- Optimizer: {opt_name} ---")

            checkpoint_dir = os.path.join(save_dir, 'checkpoints', arch_name)
            history_dir = os.path.join(save_dir, 'histories', arch_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(history_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{opt_name}.pt")
            history_path = os.path.join(history_dir, f"{opt_name}.csv")

            # Resume from checkpoint if available
            if os.path.exists(checkpoint_path):
                print(f"[checkpoint] Skipping {arch_name}/{opt_name}, checkpoint already exists.")
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                history = ckpt['history']
                results[arch_name][opt_name] = history

                conv_epoch = len(history['train_loss'])
                final_train_acc = history['train_acc'][-1]
                final_val_acc = history['val_acc'][-1]

                convergence_epochs.append({
                    'Architecture': arch_name,
                    'Optimizer': opt_name,
                    'Convergence Epoch': conv_epoch,
                    'Final Train Acc': final_train_acc,
                    'Final Val Acc': final_val_acc
                })

                if final_val_acc > best_val_acc:
                    best_val_acc = final_val_acc
                    best_arch = arch_name
                    best_opt = opt_name
                    best_model_state = ckpt['model_state_dict']
                continue

            # Reload exact same initial weights for this optimizer run
            model.load_state_dict(initial_weights)
            model.to(device)

            # Get dataloaders with deterministic generator and device-aware pin_memory
            train_loader, val_loader, test_loader = get_dataloaders(
                data_dir,
                batch_size=1 if not is_full_batch else None,
                is_full_batch=is_full_batch,
                seed=DATA_SEED,
                device=device
            )

            # Build optimizer AFTER model is on the target device
            optimizer = opt_class(model.parameters(), **opt_kwargs)

            try:
                history = train_model(
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    max_epochs=max_epochs,
                    tol=TOL
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
                'optimizer_name': opt_name,
                'arch_name': arch_name,
                'epoch': len(history['train_loss']),
                'history': history
            }, checkpoint_path)

            # Save history CSV
            pd.DataFrame(history).to_csv(history_path, index=False)

            conv_epoch = len(history['train_loss'])
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]

            convergence_epochs.append({
                'Architecture': arch_name,
                'Optimizer': opt_name,
                'Convergence Epoch': conv_epoch,
                'Final Train Acc': final_train_acc,
                'Final Val Acc': final_val_acc
            })

            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_arch = arch_name
                best_opt = opt_name
                best_model_state = copy.deepcopy(model.state_dict())

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

    # Needs a dataloader for evaluation. Batch size can be anything, full batch is fast.
    _, _, test_loader = get_dataloaders(data_dir, is_full_batch=True, seed=DATA_SEED, device=device)
    test_targets, test_preds, test_acc = evaluate_model(best_model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.4f}")

    # Train targets/preds for confusion matrix
    train_loader, _, _ = get_dataloaders(data_dir, is_full_batch=True, seed=DATA_SEED, device=device)
    train_targets, train_preds, train_acc = evaluate_model(best_model, train_loader, device)

    print(f"Final Train Accuracy for Best Model: {train_acc:.4f}")

    classes = ['0', '4', '5', '6', '7']

    plot_confusion_matrix(test_targets, test_preds, classes,
                          f'Test Confusion Matrix\n{best_arch} with {best_opt}',
                          os.path.join(save_dir, 'plots', 'confusion_matrix', 'test_confusion_matrix.png'))

    plot_confusion_matrix(train_targets, train_preds, classes,
                          f'Train Confusion Matrix\n{best_arch} with {best_opt}',
                          os.path.join(save_dir, 'plots', 'confusion_matrix', 'train_confusion_matrix.png'))


if __name__ == '__main__':
    data_directory = r'd:\DeepLearning\DL_Assignments\Assignment3\Group_11\Group_11'
    save_directory = r'd:\DeepLearning\DL_Assignments\Assignment3\Group11_Assignment3_code\results'
    run_experiments(data_directory, save_directory, max_epochs=10000)
