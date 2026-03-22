import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import mlflow
import mlflow.pytorch

import subprocess

def get_dvc_data_version():
    """Returns the MD5 hash of current data version."""
    try:
        result = subprocess.run(
            ['dvc', 'status', 'data/mosei_dataset.h5.dvc'],
            capture_output=True, text=True
        )
        # Read hash from .dvc file
        with open('data/mosei_dataset.h5.dvc') as f:
            import yaml
            dvc_info = yaml.safe_load(f)
            return dvc_info['outs'][0]['md5']
    except:
        return "unknown"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def sentiment_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     """
#     MAE + 0.5 × MSE — standard loss for MOSEI regression.
#     MAE keeps training stable, MSE penalises big mistakes harder.
#     """
#     return nn.L1Loss()(preds, targets) + 0.5 * nn.MSELoss()(preds, targets)

def sentiment_loss(preds, targets):
    mae  = nn.L1Loss()(preds, targets)
    mse  = nn.MSELoss()(preds, targets)

    # Pearson correlation loss
    preds_c   = preds   - preds.mean()
    targets_c = targets - targets.mean()
    corr_loss = -(
        (preds_c * targets_c).mean() /
        (preds.std() * targets.std() + 1e-8)
    )

    return mae + 0.5 * mse + 0.3 * corr_loss

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

    mlflow.set_experiment("mosei-multimodal-sentiment")

    with mlflow.start_run(run_name=f"transformer_fusion"):

        # ── Log all config as parameters ─────────────────────────
        # This records exactly what settings produced this result
        mlflow.log_params({
            "data_version": get_dvc_data_version(), 
            "audio_dim":      cfg['audio_dim'],
            "vision_dim":     cfg['vision_dim'],
            "d_model":        cfg['d_model'],
            "n_heads":        cfg['n_heads'],
            "enc_layers":     cfg['enc_layers'],
            "fuse_layers":    cfg['fuse_layers'],
            "dropout":        cfg['dropout'],
            "batch_size":     cfg['batch_size'],
            "learning_rate":  cfg['learning_rate'],
            "weight_decay":   cfg['weight_decay'],
            "num_epochs":     cfg['num_epochs'],
            "seq_len":        cfg['seq_len'],
        })

        optimizer = optim.AdamW([
            {'params': model.distilbert.parameters(),     'lr': 1e-5},
            {'params': model.audio_encoder.parameters(),  'lr': cfg['learning_rate']},
            {'params': model.vision_encoder.parameters(), 'lr': cfg['learning_rate']},
            {'params': model.text_encoder.parameters(),   'lr': cfg['learning_rate']},
            {'params': model.fusion.parameters(),         'lr': cfg['learning_rate']},
            {'params': model.regressor.parameters(),      'lr': cfg['learning_rate']},
        ], weight_decay=cfg['weight_decay'])

        scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['num_epochs']
        )

        save_path    = Path(cfg['model_save_path'])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        best_val_mae = float('inf')
        history      = {
            'train_loss': [], 'train_mae': [],
            'val_loss':   [], 'val_mae':   [], 'val_corr': []
        }
        PATIENCE   = 5
        no_improve = 0

    # PATIENCE    = 5      # stop if no improvement for 5 epochs
    # no_improve  = 0

        model = model.to(device)

    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr           = cfg['learning_rate'],
    #     weight_decay = cfg['weight_decay']
    # )

#     optimizer = optim.AdamW([
#     {'params': model.distilbert.parameters(),     'lr': 1e-5},  # tiny for BERT
#     {'params': model.audio_encoder.parameters(),  'lr': cfg['learning_rate']},
#     {'params': model.vision_encoder.parameters(), 'lr': cfg['learning_rate']},
#     {'params': model.text_encoder.parameters(),   'lr': cfg['learning_rate']},
#     {'params': model.fusion.parameters(),         'lr': cfg['learning_rate']},
#     {'params': model.regressor.parameters(),      'lr': cfg['learning_rate']},
# ], weight_decay=cfg['weight_decay'])
    
#     # Learning rate gently decreases to ~0 over training
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=cfg['num_epochs']
#     )

#     save_path = Path(cfg['model_save_path'])
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     history = {
#         'train_loss': [], 'train_mae': [],
#         'val_loss':   [], 'val_mae':   [], 'val_corr': []
#     }
#     best_val_mae = float('inf')

#     print(f'Training on: {device}')
#     print(f'Epochs: {cfg["num_epochs"]}  |  Batch size: {cfg["batch_size"]}\n')

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
       
        # ── After training: log final metrics ────────────────────
        mlflow.log_metrics({
            "best_val_mae":  best_val_mae,
            "best_val_corr": max(history['val_corr']),
        })

        # ── Log the best model file as an artifact ────────────────
        # Artifact = any file you want to save alongside the run
        mlflow.log_artifact(str(save_path))        # saves best_model.pt
        mlflow.log_artifact("config/config.json")  # saves config used

        # ── Register model in MLflow Model Registry ───────────────
        # This gives the model a version number and lifecycle stage
        mlflow.pytorch.log_model(
            pytorch_model = model,
            artifact_path = "model",
            registered_model_name = "mosei-sentiment-transformer"
        )

        print(f'\n✅ MLflow run complete')
        print(f'   Best Val MAE : {best_val_mae:.4f}')
        print(f'   Best Val Corr: {max(history["val_corr"]):.4f}')

    print('\nTraining complete!')
    return history