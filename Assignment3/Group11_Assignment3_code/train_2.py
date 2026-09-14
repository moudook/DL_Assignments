import gc
import torch
import torch.nn as nn


def train_model(model, optimizer,
                train_loader=None, val_loader=None,
                X_train=None, y_train=None,
                X_val=None, y_val=None,
                stochastic=True,
                device=None,
                max_epochs=10000,
                tol=1e-4,
                shuffle_seed=42,
                log_interval=100):
    """
    Trains a model using a specified optimizer.

    Supports two data-supply modes (selected automatically):

    Tensor mode (fast — preferred):
        Pass X_train, y_train, X_val, y_val as pre-loaded device tensors.
        • stochastic=True  → iterates one sample at a time (batch_size=1 SGD)
          using torch.randperm for shuffling — identical shuffle order to the
          original DataLoader with the same seed.
        • stochastic=False → single full-batch forward/backward (BGD/AdaGrad/RMSProp).
        No DataLoader machinery, PIL decoding, or .to(device) runs inside the
        epoch loop.

    DataLoader mode (fallback — backward-compatible):
        Pass train_loader and val_loader.  Behaviour is identical to original
        train.py.  Used when X_train / X_val are not supplied.

    Optimizations applied vs original train.py
    ───────────────────────────────────────────
    • Tier 1  – Dataset pre-loaded as tensors; direct index, zero allocation.
    • Tier 2a – running_loss accumulated as a tensor; .item() called once/epoch.
    • Tier 2b – preds_buf pre-allocated ONCE before the epoch loop; filled in-
                place per sample — no list.append(), no .cpu().numpy() per step.
    • Tier 2c – Pure-torch accuracy (one tensor op); no sklearn, no numpy.
    • Tier 2d – Fused gradient-norm check via clip_grad_norm_(inf) — one kernel
                instead of a Python loop over parameter tensors.
    • Tier 2e – torch.inference_mode() for validation (faster than no_grad).
    • Tier 3  – Throttled logging: print every log_interval epochs (default 100).
                History and model weights are unaffected — this is cosmetic only.
    • Tier 5f – Python GC disabled during the epoch loop to eliminate GC pauses.
                Manual gc.collect() every 50 epochs; re-enabled in finally block.

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
        device           : torch.device ('cpu' or 'cuda').
        max_epochs (int) : Hard cap on training epochs.
        tol (float)      : Early-stopping tolerance on consecutive epoch loss.
                           Training stops when abs(epoch_loss - prev_loss) < tol.
                           This is the original criterion, unchanged.
        shuffle_seed(int): Seed for torch.Generator — must match DATA_SEED used
                           during preload_tensors() to guarantee reproducibility.
        log_interval(int): Print every N epochs (cosmetic only — does not affect
                           history or model weights). Default 100.

    Returns:
        history (dict): {'train_loss': [...], 'train_acc': [...], 'val_acc': [...]}
                        One entry per epoch up to the stopping epoch.
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

    # ── Convergence tracking ────────────────────────────────────────────────
    # Original criterion: abs(epoch_loss - prev_loss) < tol.
    # This is unchanged from the assignment specification.
    prev_loss = float('inf')

    # ── Pre-compute tensor-mode constants ───────────────────────────────────
    use_tensor_mode = (X_train is not None) and (X_val is not None)

    N   = None
    rng = None
    # preds_buf and running_loss_t are pre-allocated OUTSIDE the epoch loop
    # so we pay zero allocation cost per epoch.
    preds_buf       = None
    running_loss_t  = None

    if use_tensor_mode:
        N = X_train.shape[0]
        # Persistent loss accumulator — zeroed at the start of each epoch
        # via zero_() rather than re-allocation.
        running_loss_t = torch.zeros(1, device=device)

        if stochastic:
            # One Generator per training run; state advances via randperm each
            # epoch, replicating DataLoader(shuffle=True, generator=...) exactly.
            rng = torch.Generator()
            rng.manual_seed(shuffle_seed)
            # Pre-allocated prediction buffer — N entries, filled in-place.
            # Declared ONCE here; avoids re-allocation every epoch.
            preds_buf = torch.empty(N, dtype=torch.long, device=device)

    # ── Disable Python GC for the hot loop ──────────────────────────────────
    # The epoch loop is Python-intensive (especially for batch_size=1).
    # Periodic GC during tight loops can cause unpredictable pauses.
    # We re-enable it in the finally block, and manually collect every 50 epochs.
    gc.disable()

    try:
        for epoch in range(max_epochs):
            model.train()

            # Zero the persistent accumulators (in-place, no re-allocation).
            if running_loss_t is not None:
                running_loss_t.zero_()

            # ── TENSOR MODE ──────────────────────────────────────────────────
            if use_tensor_mode:

                if stochastic:
                    # ── Stochastic path (batch_size=1) ───────────────────────
                    # randperm with the persistent RNG reproduces the same
                    # per-epoch shuffle as the original DataLoader.  No PIL,
                    # no collate, no .to(device) — pure tensor index.
                    perm = torch.randperm(N, generator=rng)

                    for i in range(N):
                        idx    = perm[i]
                        batch_X = X_train[idx].unsqueeze(0)  # view, no copy
                        batch_y = y_train[idx].unsqueeze(0)  # view, no copy

                        optimizer.zero_grad(set_to_none=True)

                        if use_amp:
                            with torch.amp.autocast(device.type, dtype=torch.float16):
                                outputs = model(batch_X)
                                loss    = criterion(outputs, batch_y)
                        else:
                            outputs = model(batch_X)
                            loss    = criterion(outputs, batch_y)

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

                            # Fused gradient-norm check: single kernel over all params
                            # instead of a Python loop calling .all() per tensor.
                            # clip_grad_norm_(inf) returns the global L2 norm without
                            # clipping anything.
                            total_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=float('inf')
                            )
                            if not torch.isfinite(total_norm):
                                raise RuntimeError(
                                    "Non-finite gradient detected before optimizer.step()"
                                )

                            optimizer.step()

                        # Accumulate WITHOUT .item() — zero Python/device sync.
                        # batch_size=1: loss IS the per-sample loss directly.
                        running_loss_t.add_(loss.detach())

                        # In-place write to pre-allocated buffer — zero allocation.
                        preds_buf[i] = outputs.argmax(1)

                    # ── Epoch-level metrics (stochastic) ─────────────────────
                    # .item() called exactly ONCE per epoch.
                    epoch_train_loss = running_loss_t.item() / N
                    # preds_buf[i] = prediction for sample perm[i].
                    # y_train[perm] = corresponding ground-truth labels.
                    # One tensor comparison + .mean() — no Python loop, no numpy.
                    epoch_train_acc  = (preds_buf == y_train[perm]).float().mean().item()

                else:
                    # ── Full-batch path ───────────────────────────────────────
                    # Single forward + backward over the entire training set.
                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            outputs = model(X_train)
                            loss    = criterion(outputs, y_train)
                    else:
                        outputs = model(X_train)
                        loss    = criterion(outputs, y_train)

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

                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float('inf')
                        )
                        if not torch.isfinite(total_norm):
                            raise RuntimeError(
                                "Non-finite gradient detected before optimizer.step()"
                            )

                        optimizer.step()

                    running_loss_t.add_(loss.detach() * X_train.size(0))

                    epoch_train_loss = running_loss_t.item() / N
                    epoch_train_acc  = (outputs.argmax(1) == y_train).float().mean().item()

            # ── DATALOADER MODE (fallback — backward-compatible) ────────────
            else:
                running_loss_dl = torch.zeros(1, device=device)
                batch_preds_list  = []
                batch_targets_list = []

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            outputs = model(batch_X)
                            loss    = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_X)
                        loss    = criterion(outputs, batch_y)

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

                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float('inf')
                        )
                        if not torch.isfinite(total_norm):
                            raise RuntimeError(
                                "Non-finite gradient detected before optimizer.step()"
                            )

                        optimizer.step()

                    running_loss_dl.add_(loss.detach() * batch_X.size(0))
                    batch_preds_list.append(outputs.argmax(1))
                    batch_targets_list.append(batch_y)

                all_preds_t   = torch.cat(batch_preds_list)
                all_targets_t = torch.cat(batch_targets_list)
                n_dl = len(train_loader.dataset)
                epoch_train_loss = running_loss_dl.item() / n_dl
                epoch_train_acc  = (all_preds_t == all_targets_t).float().mean().item()

            # ── Validation (every epoch — required for correct history curves) ──
            model.eval()
            # torch.inference_mode is faster than no_grad: it additionally
            # skips version-counter increments and view-tracking bookkeeping.
            with torch.inference_mode():
                if use_tensor_mode:
                    val_out       = model(X_val)
                    epoch_val_acc = (val_out.argmax(1) == y_val).float().mean().item()
                else:
                    vp, vt = [], []
                    for bX, bY in val_loader:
                        bX = bX.to(device, non_blocking=True)
                        bY = bY.to(device, non_blocking=True)
                        vp.append(model(bX).argmax(1))
                        vt.append(bY)
                    epoch_val_acc = (torch.cat(vp) == torch.cat(vt)).float().mean().item()
            model.train()

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)

            # ── Early stopping (original criterion — unchanged) ───────────────
            # abs(epoch_loss - prev_loss) < tol, exactly as specified.
            loss_diff = abs(epoch_train_loss - prev_loss)
            if loss_diff < tol:
                print(f"Convergence reached at epoch {epoch + 1}. "
                      f"Loss diff: {loss_diff:.6f} < {tol}")
                break

            prev_loss = epoch_train_loss

            # ── Throttled logging ─────────────────────────────────────────────
            if epoch % log_interval == 0:
                print(f"Epoch [{epoch + 1:>5}/{max_epochs}]  "
                      f"loss={epoch_train_loss:.4f}  "
                      f"train_acc={epoch_train_acc:.4f}  "
                      f"val_acc={epoch_val_acc:.4f}")

            # Manual GC every 50 epochs — prevents unbounded heap growth while
            # keeping pause frequency low (once per ~50-epoch burst, not per epoch).
            if epoch % 50 == 49:
                gc.collect()

    finally:
        # Always restore GC — critical if we exit via exception.
        gc.enable()
        gc.collect()

    return history
