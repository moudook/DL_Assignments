import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train_model(model, optimizer,
                train_loader=None, val_loader=None,
                X_train=None, y_train=None,
                X_val=None, y_val=None,
                stochastic=True,
                device=None, max_epochs=10000, tol=1e-4,
                shuffle_seed=42):
    """
    Trains a model using a specified optimizer.

    Supports two data-supply modes (selected automatically):

    Tensor mode (fast — preferred):
        Pass X_train, y_train, X_val, y_val as pre-loaded device tensors.
        • stochastic=True  → iterates one sample at a time (batch_size=1 SGD)
          using torch.randperm for shuffling — produces the same shuffle order
          as the original DataLoader with the same seed.
        • stochastic=False → single full-batch forward/backward pass (BGD,
          AdaGrad, RMSProp).
        In both cases no DataLoader machinery runs inside the epoch loop, which
        eliminates PIL-decode / transform / collate / .to(device) overhead that
        was previously paid per sample per epoch.

    DataLoader mode (fallback — backward-compatible):
        Pass train_loader and val_loader as before.  Behaviour is identical to
        the original train.py.  Used when X_train / X_val are not supplied.

    Args:
        model            : PyTorch model to train.
        optimizer        : PyTorch optimizer instance.
        train_loader     : DataLoader for training  (fallback mode only).
        val_loader       : DataLoader for validation (fallback mode only).
        X_train (Tensor) : shape (N, 784), float32, on device  (tensor mode).
        y_train (Tensor) : shape (N,),     int64,   on device  (tensor mode).
        X_val   (Tensor) : shape (M, 784), float32, on device  (tensor mode).
        y_val   (Tensor) : shape (M,),     int64,   on device  (tensor mode).
        stochastic (bool): True  → sample-by-sample (batch_size=1).
                           False → full-batch in one forward/backward call.
        device           : torch.device for the model.
        max_epochs (int) : Maximum number of training epochs.
        tol (float)      : Early-stopping tolerance on consecutive epoch loss.
        shuffle_seed(int): Seed for torch.Generator used in randperm shuffle.
                           Must match DATA_SEED used when the dataset was loaded
                           to guarantee reproducibility.

    Returns:
        history (dict): {'train_loss': [...], 'train_acc': [...], 'val_acc': [...]}
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

    # ── Tensor mode: decide which fast path to take ────────────────────────
    use_tensor_mode = (X_train is not None) and (X_val is not None)

    if use_tensor_mode:
        N = X_train.shape[0]

        if stochastic:
            # One torch.Generator per training run; its state advances each
            # epoch via randperm, exactly replicating how DataLoader(shuffle=True,
            # generator=Generator().manual_seed(seed)) would advance its RNG.
            rng = torch.Generator()
            rng.manual_seed(shuffle_seed)

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        # ── TENSOR MODE ────────────────────────────────────────────────────
        if use_tensor_mode:

            if stochastic:
                # ── Stochastic path (batch_size=1) ─────────────────────────
                # randperm with the persistent generator reproduces the same
                # per-epoch shuffle as the original DataLoader.  No PIL, no
                # collate, no .to(device) — pure tensor index.
                perm = torch.randperm(N, generator=rng)

                for i in range(N):
                    idx = perm[i]
                    batch_X = X_train[idx].unsqueeze(0)   # (1, 784) — no copy
                    batch_y = y_train[idx].unsqueeze(0)   # (1,)

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
                        # NOTE: gradients here are still SCALED by scaler.get_scale()
                        # (a large factor, e.g. 65536x) to prevent fp16 underflow
                        # during backward. Seeing inf/nan at this exact point is a
                        # NORMAL, expected part of AMP — not a sign of diverged
                        # training — and is especially common early on with
                        # batch_size=1, where per-sample gradient variance is high.
                        # scaler.step() unscales the gradients itself, detects any
                        # non-finite values, and SKIPS the optimizer step for just
                        # this batch; scaler.update() then shrinks the scale factor
                        # so subsequent batches are less likely to overflow. This
                        # self-corrects automatically, so we must not hard-fail on
                        # it here (doing so used to kill the run on the first
                        # unlucky batch).
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()

                        for p in model.parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                raise RuntimeError(
                                    "Non-finite gradient detected before optimizer.step()"
                                )

                        optimizer.step()

                    running_loss += loss.item() * batch_X.size(0)

                    _, preds = torch.max(outputs, 1)
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_targets.extend(batch_y.cpu().numpy())

            else:
                # ── Full-batch path ─────────────────────────────────────────
                # Single forward + backward over the entire training set.
                # No loop needed — the tensor IS the batch.
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        outputs = model(X_train)
                        loss = criterion(outputs, y_train)
                else:
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss {loss.item()} before backward/update"
                    )

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError(
                                "Non-finite gradient detected before optimizer.step()"
                            )

                    optimizer.step()

                running_loss += loss.item() * X_train.size(0)

                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(y_train.cpu().numpy())

        # ── DATALOADER MODE (fallback — identical to original train.py) ────
        else:
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

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
                    # See note in tensor-mode stochastic path above.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError(
                                "Non-finite gradient detected before optimizer.step()"
                            )

                    optimizer.step()

                running_loss += loss.item() * batch_X.size(0)

                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(batch_y.cpu().numpy())

        # ── Per-epoch metrics ───────────────────────────────────────────────
        n_train = N if use_tensor_mode else len(train_loader.dataset)
        epoch_train_loss = running_loss / n_train
        epoch_train_acc  = accuracy_score(all_train_targets, all_train_preds)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        all_val_preds   = []
        all_val_targets = []

        with torch.no_grad():
            if use_tensor_mode:
                # Single forward pass — X_val is already on device, no loop needed.
                val_outputs = model(X_val)
                _, val_preds = torch.max(val_outputs, 1)
                all_val_preds   = val_preds.cpu().numpy().tolist()
                all_val_targets = y_val.cpu().numpy().tolist()
            else:
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    outputs = model(batch_X)
                    _, preds = torch.max(outputs, 1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_targets.extend(batch_y.cpu().numpy())

        epoch_val_acc = accuracy_score(all_val_targets, all_val_preds)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # ── Early stopping ──────────────────────────────────────────────────
        loss_diff = abs(epoch_train_loss - prev_loss)
        if loss_diff < tol:
            print(f"Convergence reached at epoch {epoch+1}. "
                  f"Loss diff: {loss_diff:.6f} < {tol}")
            break

        prev_loss = epoch_train_loss

        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Restore train mode after eval block (done at top of next epoch too,
        # but explicit here for clarity when reading the flow).
        model.train()

    return history
