import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import mlflow
import mlflow.pytorch
import subprocess
import math
import json
import random
from tqdm import tqdm

_DEBUG_LOG = Path("/kaggle/working/debug-e745fb.log")

MELD_LABEL_VALUES   = np.array([-1., 0., 1.], dtype=np.float32)
MOSEI_LABEL_VALUES  = np.array([
    -3., -2.6666667, -2.3333333, -2., -1.6666666, -1.3333334,
    -1., -0.6666667, -0.5, -0.33333334, -0.16666667, 0.,
    0.16666667, 0.33333334, 0.5, 0.6666667, 0.8333333, 1.,
    1.1666666, 1.3333334, 1.5, 1.6666666, 1.8333334, 2.,
    2.3333333, 2.6666667, 3.
], dtype=np.float32)

def snap_to_valid(preds: np.ndarray, dataset: str = 'mosei') -> np.ndarray:
    label_vals = MELD_LABEL_VALUES if dataset == 'meld' else MOSEI_LABEL_VALUES
    diffs = np.abs(preds[:, None] - label_vals[None, :])
    return label_vals[diffs.argmin(axis=1)]

def get_dvc_data_version():
    dvc_file_paths = [
        "data/mosei_dataset.h5.dvc",
        "/kaggle/input/datasets/dhananjayapaliwal/multimodal-github/data/mosei_dataset.h5.dvc",
    ]
    for path in dvc_file_paths:
        try:
            import yaml
            with open(path) as f:
                dvc_info = yaml.safe_load(f)
                return dvc_info['outs'][0]['md5']
        except Exception:
            continue
    return "unknown — dvc file not found"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def sentiment_loss(preds, targets):
#     mae     = nn.L1Loss()(preds, targets)
#     mse     = nn.MSELoss()(preds, targets)
#     preds_c = preds   - preds.mean()
#     tgt_c   = targets - targets.mean()
#     corr    = -((preds_c * tgt_c).mean() /
#                 (preds.std().clamp(min=1e-8) * targets.std().clamp(min=1e-8)))
#     return mae + 0.5 * mse + 0.3 * corr

def sentiment_loss(preds, targets):
    # Class weights to handle MELD imbalance
    # neutral=47% → weight 0.6, negative=29% → weight 1.0, positive=23% → weight 1.2
    weights = torch.ones_like(targets)
    weights[targets == -1.0] = 1.0
    weights[targets ==  0.0] = 0.6
    weights[targets ==  1.0] = 1.2
    weights = weights / weights.mean()  # normalize

    mae     = (torch.abs(preds - targets) * weights).mean()
    mse     = ((preds - targets)**2 * weights).mean()
    preds_c = preds   - preds.mean()
    tgt_c   = targets - targets.mean()
    corr    = -((preds_c * tgt_c).mean() /
                (preds.std().clamp(min=1e-8) * targets.std().clamp(min=1e-8)))
    return mae + 0.5 * mse + 0.3 * corr


def apply_modality_dropout(audio, vision, input_ids, attention_mask,
                           audio_drop_prob=0.15,
                           vision_drop_prob=0.15,
                           text_drop_prob=0.05):
    """
    Randomly zero out entire modalities during training.
    Forces model to work with any combination of modalities.
    """
    if random.random() < audio_drop_prob:
        audio = torch.zeros_like(audio)
    if random.random() < vision_drop_prob:
        vision = torch.zeros_like(vision)
    if random.random() < text_drop_prob:
        input_ids      = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(attention_mask)  # ones to avoid transformer crash
    return audio, vision, input_ids, attention_mask


def run_one_epoch(model, loader, optimizer=None, is_train: bool = True,
                  use_modality_dropout: bool = False,
                  audio_drop_prob: float = 0.15,
                  vision_drop_prob: float = 0.15,
                  text_drop_prob: float = 0.05):
    """
    One full pass over the dataset.
    Returns: avg_loss, mae, all_preds, all_labels
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=f'{"Train" if is_train else "Val  "}',
                          leave=False, unit='batch'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio          = batch['audio'].to(device)
            vision         = batch['vision'].to(device)
            labels         = batch['label'].to(device)

            # Apply modality dropout during training only
            if is_train and use_modality_dropout:
                audio, vision, input_ids, attention_mask = apply_modality_dropout(
                    audio, vision, input_ids, attention_mask,
                    audio_drop_prob=audio_drop_prob,
                    vision_drop_prob=vision_drop_prob,
                    text_drop_prob=text_drop_prob,
                )

            preds = model(input_ids, attention_mask, audio, vision)
            loss  = sentiment_loss(preds, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader.dataset)
    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    mae       = np.mean(np.abs(preds_np - labels_np))

    return avg_loss, mae, preds_np, labels_np


def train(model, train_loader, val_loader, cfg, resume_from=None) -> dict:
    """
    Full training loop with resume/checkpoint + modality dropout.

    resume_from: explicit path to checkpoint.pt.
                 If None — auto-detects checkpoint.pt next to model_save_path.
    """

    mlflow.set_experiment("mosei-multimodal-sentiment")

    with mlflow.start_run(run_name="transformer_fusion"):

        mlflow.log_params({
            "data_version":     get_dvc_data_version(),
            "audio_dim":        cfg['audio_dim'],
            "vision_dim":       cfg['vision_dim'],
            "d_model":          cfg['d_model'],
            "n_heads":          cfg['n_heads'],
            "enc_layers":       cfg['enc_layers'],
            "fuse_layers":      cfg['fuse_layers'],
            "dropout":          cfg['dropout'],
            "batch_size":       cfg['batch_size'],
            "learning_rate":    cfg['learning_rate'],
            "weight_decay":     cfg['weight_decay'],
            "num_epochs":       cfg['num_epochs'],
            "seq_len":          cfg['seq_len'],
            "modality_dropout": cfg.get('modality_dropout', True),
            "audio_drop_prob":  cfg.get('audio_drop_prob',  0.15),
            "vision_drop_prob": cfg.get('vision_drop_prob', 0.15),
            "text_drop_prob":   cfg.get('text_drop_prob',   0.05),
        })

        model = model.to(device)

        optimizer = optim.AdamW([
            {'params': [p for p in model.roberta.parameters() if p.requires_grad], 'lr': 2e-5},
            {'params': model.audio_encoder.parameters(),  'lr': cfg['learning_rate']},
            {'params': model.vision_encoder.parameters(), 'lr': cfg['learning_rate']},
            {'params': model.text_encoder.parameters(),   'lr': cfg['learning_rate']},
            {'params': model.fusion.parameters(),         'lr': cfg['learning_rate']},
            {'params': model.regressor.parameters(),      'lr': cfg['learning_rate']},
        ], weight_decay=cfg['weight_decay'])

        def warmup_cosine(epoch):
            warmup_epochs = 2
            num_epochs    = cfg['num_epochs']
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
            cosine_span = max(num_epochs - warmup_epochs, 1)
            progress    = min((epoch - warmup_epochs) / cosine_span, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler  = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
        save_path  = Path(cfg['model_save_path'])
        ckpt_path  = save_path.parent / 'checkpoint.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Resume logic ──────────────────────────────────────────
        start_epoch      = 0
        best_val_mae_raw = float('inf')
        no_improve       = 0
        history          = {
            'train_loss': [], 'train_mae': [],
            'val_loss':   [], 'val_mae':   [], 'val_corr': [],
            'val_mae_snap': [], 'val_corr_snap': [],
        }

        resume_path = Path(resume_from) if resume_from else ckpt_path
        if resume_path.exists():
            print(f'🔄 Resuming from {resume_path}...')
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch      = ckpt['epoch'] + 1
            best_val_mae_raw = ckpt['best_val_mae_raw']
            no_improve       = ckpt['no_improve']
            history          = ckpt['history']
            print(f'  ✅ Resumed from epoch {start_epoch} | '
                  f'best_val_mae={best_val_mae_raw:.4f} | '
                  f'no_improve={no_improve}/{cfg.get("patience", 8)}')
        else:
            print('🆕 Starting fresh training...')

        PATIENCE         = cfg.get('patience', 8)
        use_mod_dropout  = cfg.get('modality_dropout', True)
        audio_drop_prob  = cfg.get('audio_drop_prob',  0.15)
        vision_drop_prob = cfg.get('vision_drop_prob', 0.15)
        text_drop_prob   = cfg.get('text_drop_prob',   0.05)

        print(f'\nTraining on : {device}')
        print(f'Epochs      : {start_epoch} → {cfg["num_epochs"]}')
        print(f'Batch size  : {cfg["batch_size"]}')
        print(f'Mod dropout : {use_mod_dropout} '
              f'(audio={audio_drop_prob} vision={vision_drop_prob} text={text_drop_prob})\n')

        for epoch in range(start_epoch, cfg['num_epochs']):

            train_loss, train_mae, _, _ = run_one_epoch(
                model, train_loader, optimizer,
                is_train=True,
                use_modality_dropout=use_mod_dropout,
                audio_drop_prob=audio_drop_prob,
                vision_drop_prob=vision_drop_prob,
                text_drop_prob=text_drop_prob,
            )
            val_loss, _, val_preds, val_labels = run_one_epoch(
                model, val_loader, is_train=False,
                use_modality_dropout=False,  # never drop during val
            )

            # ── Debug prints ──────────────────────────────────────
            print(f'Raw preds — min:{val_preds.min():.4f} max:{val_preds.max():.4f} std:{val_preds.std():.4f}')
            val_preds_snapped = snap_to_valid(val_preds, dataset=cfg.get('dataset', 'mosei'))
            print(f'Snapped preds unique: {np.unique(val_preds_snapped, return_counts=True)}')
            print(f'Val labels   unique: {np.unique(val_labels, return_counts=True)}')

            vp           = np.asarray(val_preds).ravel()
            vl           = np.asarray(val_labels).ravel()
            val_mae_snap = np.mean(np.abs(val_preds_snapped - val_labels))
            val_mae_raw  = np.mean(np.abs(vp - vl))
            val_corr_snap = (
                pearsonr(val_preds_snapped, val_labels)[0]
                if val_preds_snapped.std() > 0 and val_labels.std() > 0
                else 0.0
            )
            val_corr_raw = (
                pearsonr(vp, vl)[0]
                if vp.std() > 1e-8 and vl.std() > 1e-8
                else 0.0
            )

            # ── Debug log ─────────────────────────────────────────
            try:
                import time as _time
                _payload = {
                    "sessionId":    "e745fb",
                    "runId":        "train",
                    "hypothesisId": "H1_metrics",
                    "location":     "training/trainer.py:train.loop",
                    "message":      "val continuous vs snapped",
                    "data": {
                        "epoch":        int(epoch),
                        "val_mae_raw":  float(val_mae_raw),
                        "val_mae_snap": float(val_mae_snap),
                        "val_corr_raw": float(val_corr_raw),
                        "pred_min":     float(vp.min()),
                        "pred_max":     float(vp.max()),
                        "pred_std":     float(vp.std()),
                    },
                    "timestamp": int(_time.time() * 1000),
                }
                with open(_DEBUG_LOG, "a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps(_payload) + "\n")
            except Exception:
                pass

            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae_raw)
            history['val_corr'].append(val_corr_raw)
            history['val_mae_snap'].append(val_mae_snap)
            history['val_corr_snap'].append(val_corr_snap)

            mlflow.log_metrics({
                "train_loss":    train_loss,
                "train_mae":     train_mae,
                "val_loss":      val_loss,
                "val_mae_raw":   val_mae_raw,
                "val_mae_snap":  val_mae_snap,
                "val_corr_raw":  val_corr_raw,
                "val_corr_snap": val_corr_snap,
            }, step=epoch)

            print(
                f'Epoch {epoch+1:02d}/{cfg["num_epochs"]}  |  '
                f'Train Loss: {train_loss:.4f}  MAE: {train_mae:.4f}  |  '
                f'Val Loss: {val_loss:.4f}  MAE(raw): {val_mae_raw:.4f}  '
                f'MAE(snap): {val_mae_snap:.4f}  '
                f'Corr(raw): {val_corr_raw:.4f}  Corr(snap): {val_corr_snap:.4f}'
            )

            # ── Save best model ───────────────────────────────────
            if val_mae_raw < best_val_mae_raw:
                best_val_mae_raw = val_mae_raw
                no_improve       = 0
                torch.save(model.state_dict(), save_path)
                print(f'  ✅ Best model saved (Val MAE raw={best_val_mae_raw:.4f} snap={val_mae_snap:.4f})')
            else:
                no_improve += 1
                print(f'  No improvement for {no_improve}/{PATIENCE} epochs')
                if no_improve >= PATIENCE:
                    print(f'\n🛑 Early stopping at epoch {epoch+1}')
                    break

            # ── Save full checkpoint every epoch ──────────────────
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_mae_raw': best_val_mae_raw,
                'no_improve':      no_improve,
                'history':         history,
            }, ckpt_path)
            print(f'  💾 Checkpoint saved (epoch {epoch+1})')

        # ── Final MLflow logging ──────────────────────────────────
        mlflow.log_metrics({
            "best_val_mae_raw":  best_val_mae_raw,
            "best_val_corr_raw": max(history['val_corr']),
        })

        mlflow.log_artifact(str(save_path))

        config_save_path = "/kaggle/working/config.json"
        with open(config_save_path, "w") as f:
            json.dump(cfg, f, indent=4)
        mlflow.log_artifact(config_save_path)

        mlflow.pytorch.log_model(
            pytorch_model         = model,
            artifact_path         = "model",
            registered_model_name = "mosei-sentiment-transformer"
        )

        print(f'\n✅ MLflow run complete')
        print(f'   Best Val MAE (raw): {best_val_mae_raw:.4f}')
        print(f'   Best Val Corr (raw): {max(history["val_corr"]):.4f}')

    print('\nTraining complete!')
    return history