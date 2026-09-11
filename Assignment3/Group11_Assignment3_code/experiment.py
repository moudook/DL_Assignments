import torch
import torch.optim as optim
import os
import copy
import pandas as pd

from dataset import get_dataloaders
from models import get_models
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves

def run_experiments(data_dir, save_dir, max_epochs=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    models_dict = get_models()
    
    opt_configs = {
        'SGD': (optim.SGD, {'lr': 0.001}, False), # batch_size=1
        'BGD': (optim.SGD, {'lr': 0.001}, True), # batch_size=total
        'SGD_Momentum': (optim.SGD, {'lr': 0.001, 'momentum': 0.9}, False),
        'NAG': (optim.SGD, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True}, False),
        'AdaGrad': (optim.Adagrad, {'lr': 0.001}, True),
        'RMSProp': (optim.RMSprop, {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-8}, True),
        'Adam': (optim.Adam, {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}, False)
    }
    
    results = {}
    convergence_epochs = []
    
    best_arch = None
    best_opt = None
    best_val_acc = -1
    best_model_state = None
    
    for arch_name, model in models_dict.items():
        print(f"\n{'='*50}\nStarting experiments for {arch_name}\n{'='*50}")
        results[arch_name] = {}
        
        # Save initial random weights
        initial_weights = copy.deepcopy(model.state_dict())
        
        for opt_name, (opt_class, opt_kwargs, is_full_batch) in opt_configs.items():
            print(f"\n--- Optimizer: {opt_name} ---")
            
            # Load initial weights
            model.load_state_dict(initial_weights)
            
            # Get dataloaders
            train_loader, val_loader, test_loader = get_dataloaders(
                data_dir, 
                batch_size=1 if not is_full_batch else None, 
                is_full_batch=is_full_batch
            )
            
            optimizer = opt_class(model.parameters(), **opt_kwargs)
            
            history = train_model(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=max_epochs,
                tol=1e-4
            )
            
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
                best_model_state = copy.deepcopy(model.state_dict())
            
        # Plot training curves for this architecture
        plot_training_curves(results, arch_name, save_dir)
        
    df_results = pd.DataFrame(convergence_epochs)
    df_results.to_csv(os.path.join(save_dir, 'convergence_results.csv'), index=False)
    print("\nConvergence Results:")
    print(df_results.to_string())
    
    print(f"\nBest Architecture: {best_arch} (with {best_opt}) - Val Acc: {best_val_acc:.4f}")
    
    print("Evaluating best model on test set...")
    best_model = models_dict[best_arch]
    best_model.load_state_dict(best_model_state)
    
    # Needs a dataloader for evaluation. Batch size can be anything, full batch is fast.
    _, _, test_loader = get_dataloaders(data_dir, is_full_batch=True)
    test_targets, test_preds, test_acc = evaluate_model(best_model, test_loader, device)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Train targets/preds for confusion matrix
    train_loader, _, _ = get_dataloaders(data_dir, is_full_batch=True)
    train_targets, train_preds, train_acc = evaluate_model(best_model, train_loader, device)
    
    print(f"Final Train Accuracy for Best Model: {train_acc:.4f}")
    
    classes = ['0', '4', '5', '6', '7']
    
    plot_confusion_matrix(test_targets, test_preds, classes, 
                          f'Test Confusion Matrix\n{best_arch} with {best_opt}', 
                          os.path.join(save_dir, 'test_confusion_matrix.png'))
                          
    plot_confusion_matrix(train_targets, train_preds, classes, 
                          f'Train Confusion Matrix\n{best_arch} with {best_opt}', 
                          os.path.join(save_dir, 'train_confusion_matrix.png'))

if __name__ == '__main__':
    data_directory = r'd:\DeepLearning\DL_Assignments\Assignment3\Group_11\Group_11'
    save_directory = r'd:\DeepLearning\DL_Assignments\Assignment3\Group11_Assignment3_code\results'
    run_experiments(data_directory, save_directory, max_epochs=10000)
