import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sentiment_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MAE + 0.5 × MSE — standard loss for MOSEI regression.
    MAE keeps training stable, MSE penalises big mistakes harder.
    """
    return nn.L1Loss()(preds, targets) + 0.5 * nn.MSELoss()(preds, targets)


def run_one_epoch(model, loader, optimizer=None, is_train: bool = True):
    """
    One full pass over the dataset.

    is_train=True  → updates model weights
    is_train=False → just computes predictions (val / test)

    Returns: avg_loss, mae, all_preds, all_labels
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio          = batch['audio'].to(device)
            vision         = batch['vision'].to(device)
            labels         = batch['label'].to(device)

            preds = model(input_ids, attention_mask, audio, vision)
            loss  = sentiment_loss(preds, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents weights exploding during training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    mae       = np.mean(np.abs(preds_np - labels_np))

    return avg_loss, mae, preds_np, labels_np


def train(model, train_loader, val_loader, cfg) -> dict:
    """
    Full training loop.
    Saves best model checkpoint to config['model_save_path'].
    Returns history dict for plotting.
    """

    PATIENCE    = 5      # stop if no improvement for 5 epochs
    no_improve  = 0

    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg['learning_rate'],
        weight_decay = cfg['weight_decay']
    )
    # Learning rate gently decreases to ~0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs']
    )

    save_path = Path(cfg['model_save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [], 'train_mae': [],
        'val_loss':   [], 'val_mae':   [], 'val_corr': []
    }
    best_val_mae = float('inf')

    print(f'Training on: {device}')
    print(f'Epochs: {cfg["num_epochs"]}  |  Batch size: {cfg["batch_size"]}\n')

    for epoch in range(cfg['num_epochs']):

        train_loss, train_mae, _, _ = run_one_epoch(
            model, train_loader, optimizer, is_train=True
        )
        val_loss, val_mae, val_preds, val_labels = run_one_epoch(
            model, val_loader, is_train=False
        )
        val_corr = pearsonr(val_preds, val_labels)[0]
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)

        print(
            f'Epoch {epoch+1:02d}/{cfg["num_epochs"]}  |  '
            f'Train Loss: {train_loss:.4f}  MAE: {train_mae:.4f}  |  '
            f'Val Loss: {val_loss:.4f}  MAE: {val_mae:.4f}  Corr: {val_corr:.4f}'
        )

        # if val_mae < best_val_mae:
        #     best_val_mae = val_mae
        #     torch.save(model.state_dict(), save_path)
        #     print(f'  ✅ Best model saved  (Val MAE = {best_val_mae:.4f})')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save(model.state_dict(), save_path)
            print(f'  ✅ Best model saved (Val MAE = {best_val_mae:.4f})')
        else:
            no_improve += 1
            print(f'  No improvement for {no_improve}/{PATIENCE} epochs')

            if no_improve >= PATIENCE:
                print(f'\n🛑 Early stopping at epoch {epoch+1}')
                break

    print('\nTraining complete!')
    return history