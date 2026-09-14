import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_model(model, optimizer, train_loader, val_loader, device, max_epochs=10000, tol=1e-4):
    """
    Trains a model using a specified optimizer and dataloaders.
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: 'cpu' or 'cuda'
        max_epochs: Maximum number of epochs to train
        tol: Tolerance for early stopping (absolute difference between successive epoch losses)
        
    Returns:
        history: dictionary containing lists of metrics over epochs
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    prev_loss = float('inf')
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast(device.type, dtype=torch.float16):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss {loss.item()} before backward/update"
                )
            
            if use_amp:
                scaler.scale(loss).backward()
                
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError(
                            f"Non-finite gradient detected before optimizer.step()"
                        )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RuntimeError(
                            f"Non-finite gradient detected before optimizer.step()"
                        )
                
                optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(batch_y.cpu().numpy())
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(all_train_targets, all_train_preds)
        
        # Validation
        model.eval()
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(batch_y.cpu().numpy())
                
        epoch_val_acc = accuracy_score(all_val_targets, all_val_preds)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Check stopping criteria
        loss_diff = abs(epoch_train_loss - prev_loss)
        if loss_diff < tol:
            print(f"Convergence reached at epoch {epoch+1}. Loss diff: {loss_diff:.6f} < {tol}")
            break
            
        prev_loss = epoch_train_loss
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")
            
    return history
