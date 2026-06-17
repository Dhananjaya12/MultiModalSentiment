import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from scipy.stats import pearsonr
from pathlib import Path
import mlflow
import mlflow.pytorch
import subprocess
import math
import json
import random
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time

CLASS_NAMES = ['negative', 'neutral', 'positive']  # class idx 0,1,2 <-> label -1,0,1

_DEBUG_LOG = Path("/content/drive/MyDrive/UNT OneDrive Backup/Backup folder/Backup folder/Projects/MultiModalSentimentGithub/output/debug-e745fb.log")

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

def sentiment_loss(logits, class_idx, class_weights=None):
    """Cross-entropy over the 3 sentiment classes (negative/neutral/positive)."""
    return F.cross_entropy(logits, class_idx, weight=class_weights)


def labels_to_class_idx(labels: torch.Tensor) -> torch.Tensor:
    """MELD labels are exactly -1./0./1. -> class indices 0/1/2."""
    return (labels + 1.0).round().long()


def compute_class_weights(loader, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights computed from a dataloader's underlying labels."""
    from torch.utils.data import Subset
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        indices = sorted(dataset.indices)
        dataset = dataset.dataset
    else:
        indices = sorted(dataset.indices)
    with h5py.File(dataset.hdf5_path, 'r') as f:
        labels = f['labels'][indices]
    class_idx = np.clip(np.round(labels + 1.0), 0, num_classes - 1).astype(int)
    counts  = np.bincount(class_idx, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> dict:
    """Accuracy, per-class precision/recall/F1, macro F1, and confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        'accuracy':         float(np.trace(cm) / cm.sum()),
        'precision':        precisions,
        'recall':           recalls,
        'f1':               f1s,
        'f1_macro':         float(np.mean(f1s)),
        'confusion_matrix': cm,
    }


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
                  text_drop_prob: float = 0.05,
                  scaler=None,
                  class_weights=None):
    """
    One full pass over the dataset.
    Returns: avg_loss, accuracy, all_preds (class idx), all_labels (class idx)
    """
    # t_model_mode_start = time.time()
    model.train() if is_train else model.eval()
    # t_model_mode_end = time.time()
    # print(f'Model set to {"train" if is_train else "eval"} mode in {t_model_mode_end - t_model_mode_start:.2f} seconds\n')

    total_loss = 0.0
    all_preds, all_labels = [], []
    # import time
    # data_time  = 0.0
    # model_time = 0.0
    batch_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=f'{"Train" if is_train else "Val  "}',
                          leave=False, unit='batch'):

            # t0 = time.time()
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio          = batch['audio'].to(device)
            vision         = batch['vision'].to(device)
            labels         = batch['label'].to(device)
            class_idx      = labels_to_class_idx(labels)
            # data_time += time.time() - t0

            # t1 = time.time()
            if is_train and use_modality_dropout:
                # t_mod_dropout_start = time.time()
                audio, vision, input_ids, attention_mask = apply_modality_dropout(
                    audio, vision, input_ids, attention_mask,
                    audio_drop_prob=audio_drop_prob,
                    vision_drop_prob=vision_drop_prob,
                    text_drop_prob=text_drop_prob,
                )
                # t_mod_dropout_end = time.time()
                # print(f'  Modality dropout applied in {t_mod_dropout_end - t_mod_dropout_start:.2f} seconds')
            
            with autocast():
                # t_model_preds_start = time.time()
                preds = model(input_ids, attention_mask, audio, vision)
                # t_model_preds_end = time.time()
                # print(f'  Model predictions time: {t_model_preds_end - t_model_preds_start:.2f} seconds')
                # t_loss_start = time.time()
                loss  = sentiment_loss(preds, class_idx, class_weights)
                # t_loss_end = time.time()
                # print(f'  Loss computation time: {t_loss_end - t_loss_start:.2f} seconds')

            if is_train:
                # t_optimizer_start = time.time()
                optimizer.zero_grad()
                # t_optimizer_end = time.time()
                # print(f'  Optimizer zero_grad time: {t_optimizer_end - t_optimizer_start:.2f} seconds')
                # t_backward_start = time.time()
                scaler.scale(loss).backward()
                # t_backward_end = time.time()
                # print(f'  Backward pass time: {t_backward_end - t_backward_start:.2f} seconds')
                # t_optimizer_step_start = time.time()
                scaler.unscale_(optimizer)
                # t_optimizer_step_end = time.time()
                # print(f'  Optimizer step preparation time: {t_optimizer_step_end - t_optimizer_step_start:.2f} seconds')
                # t_clip_start = time.time()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # t_clip_end = time.time()
                # print(f'  Gradient clipping time: {t_clip_end - t_clip_start:.2f} seconds')
                # t_optimizer_step_start = time.time()
                scaler.step(optimizer)
                # t_optimizer_step_end = time.time()
                # print(f'  Optimizer step time: {t_optimizer_step_end - t_optimizer_step_start:.2f} seconds')
                scaler.update()
            # model_time += time.time() - t1

            total_loss += loss.item() * len(labels)
            pred_classes = preds.detach().argmax(dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(class_idx.cpu().numpy())

            batch_count += 1
            if batch_count == 3:
                print(f'\n  [Timing after 3 batches]')
                # print(f'  Data loading : {data_time:.2f}s  ({data_time/batch_count:.2f}s/batch)')
                # print(f'  Model forward: {model_time:.2f}s  ({model_time/batch_count:.2f}s/batch)')

    avg_loss  = total_loss / len(loader.dataset)
    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    accuracy  = np.mean(preds_np == labels_np)

    return avg_loss, accuracy, preds_np, labels_np


def train(model, train_loader, val_loader, cfg, resume_from=None) -> dict:
    """
    Full training loop with resume/checkpoint + modality dropout.

    resume_from: explicit path to checkpoint.pt.
                 If None — auto-detects checkpoint.pt next to model_save_path.
    """
    mlflow.set_tracking_uri(cfg['mlflow_uri'])
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
        
        # t_model_loading_start = time.time()
        model = model.to(device)
        # t_model_loading_end = time.time()
        # print(f'Model loading time: {t_model_loading_end - t_model_loading_start:.2f} seconds\n')


        # t_optimizer_init_start = time.time()
        head_lr = cfg.get('head_lr', cfg['learning_rate'] * 20)
        optimizer = optim.AdamW([
            {'params': [p for p in model.roberta.parameters() if p.requires_grad], 'lr': 2e-5},
            {'params': model.audio_encoder.parameters(),  'lr': cfg['learning_rate']},
            {'params': model.vision_encoder.parameters(), 'lr': cfg['learning_rate']},
            {'params': model.text_encoder.parameters(),   'lr': cfg['learning_rate']},
            {'params': model.fusion.parameters(),         'lr': cfg['learning_rate']},
            {'params': model.regressor.parameters(),      'lr': head_lr},
        ], weight_decay=cfg['weight_decay'])
        print(f'LR — encoders/fusion: {cfg["learning_rate"]}  |  head: {head_lr}')
        # t_optimizer_init_end = time.time()
        # print(f'Optimizer initialization time: {t_optimizer_init_end - t_optimizer_init_start:.2f} seconds\n')

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

        # ── Class weights (inverse frequency, from train split) ────
        class_weights = compute_class_weights(train_loader).to(device)
        print(f'Class weights [negative, neutral, positive]: {class_weights.tolist()}')

        # ── Resume logic ──────────────────────────────────────────
        start_epoch   = 0
        best_val_f1   = -1.0
        no_improve    = 0
        history       = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [], 'val_f1_macro': [],
        }

        resume_path = Path(resume_from) if resume_from else ckpt_path
        if resume_path.exists():
            print(f'🔄 Resuming from {resume_path}...')
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_val_f1 = ckpt['best_val_f1']
            no_improve  = ckpt['no_improve']
            history     = ckpt['history']
            print(f'  ✅ Resumed from epoch {start_epoch} | '
                  f'best_val_f1_macro={best_val_f1:.4f} | '
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

        scaler = GradScaler()

        for epoch in range(start_epoch, cfg['num_epochs']):
            # t_epoch_training_start = time.time()
            train_loss, train_acc, _, _ = run_one_epoch(
                model, train_loader, optimizer,
                is_train=True,
                use_modality_dropout=use_mod_dropout,
                audio_drop_prob=audio_drop_prob,
                vision_drop_prob=vision_drop_prob,
                text_drop_prob=text_drop_prob,
                scaler=scaler,
                class_weights=class_weights,
            )
            # t_epoch_training_end = time.time()
            # print(f'Epoch {epoch+1} training time: {t_epoch_training_end - t_epoch_training_start:.2f} seconds\n')
            # t_epoch_val_start = time.time()
            val_loss, val_acc, val_preds, val_labels = run_one_epoch(
                model, val_loader, is_train=False,
                use_modality_dropout=False,  # never drop during val
                scaler=None,
                class_weights=class_weights,
            )
            # t_epoch_val_end = time.time()
            # print(f'Epoch {epoch+1} validation time: {t_epoch_val_end - t_epoch_val_start:.2f} seconds\n')

            # ── Debug prints ──────────────────────────────────────
            # t_debug_start = time.time()
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
                        "epoch":        int(epoch + 1),
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
            # t_debug_end = time.time()
            # print(f'Debug logging time: {t_debug_end - t_debug_start:.2f} seconds\n')

            scheduler.step()
            # t_epoch_mlflow_start = time.time()

            cls_metrics = compute_classification_metrics(val_preds, val_labels, num_classes=3)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(cls_metrics['accuracy'])
            history['val_f1_macro'].append(cls_metrics['f1_macro'])

            mlflow.log_metrics({
                "train_loss":      train_loss,
                "train_acc":       train_acc,
                "val_loss":        val_loss,
                "val_acc":         cls_metrics['accuracy'],
                "val_f1_macro":    cls_metrics['f1_macro'],
                "val_f1_negative": cls_metrics['f1'][0],
                "val_f1_neutral":  cls_metrics['f1'][1],
                "val_f1_positive": cls_metrics['f1'][2],
            }, step=epoch)

            print(
                f'Epoch {epoch+1:02d}/{cfg["num_epochs"]}  |  '
                f'Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  '
                f'Val Loss: {val_loss:.4f}  Acc: {cls_metrics["accuracy"]:.4f}  '
                f'F1(macro): {cls_metrics["f1_macro"]:.4f}'
            )
            print(
                f'  F1 — neg: {cls_metrics["f1"][0]:.4f}  '
                f'neu: {cls_metrics["f1"][1]:.4f}  '
                f'pos: {cls_metrics["f1"][2]:.4f}'
            )
            print(f'  Confusion matrix [neg/neu/pos rows=true, cols=pred]:')
            for i, row in enumerate(cls_metrics['confusion_matrix']):
                print(f'    {CLASS_NAMES[i]:8s} {row}')

            # ── Save best model ───────────────────────────────────
            if cls_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = cls_metrics['f1_macro']
                no_improve  = 0
                torch.save(model.state_dict(), save_path)
                print(f'  ✅ Best model saved (Val F1 macro={best_val_f1:.4f}  Acc={cls_metrics["accuracy"]:.4f})')
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
                'best_val_f1':     best_val_f1,
                'no_improve':      no_improve,
                'history':         history,
            }, ckpt_path)
            print(f'  💾 Checkpoint saved (epoch {epoch+1})')
            # t_epoch_mlflow_end = time.time()
            # print(f'Epoch {epoch+1} MLflow logging time: {t_epoch_mlflow_end - t_epoch_mlflow_start:.2f} seconds\n')

        # ── Final MLflow logging ──────────────────────────────────
        mlflow.log_metrics({
            "best_val_f1_macro": best_val_f1,
            "best_val_acc":      max(history['val_acc']),
        })

        mlflow.log_artifact(str(save_path))

        config_save_path = str(Path(cfg['model_save_path']).parent / 'config.json')
        with open(config_save_path, "w") as f:
            json.dump(cfg, f, indent=4)
        mlflow.log_artifact(config_save_path)

        mlflow.pytorch.log_model(
            pytorch_model         = model,
            artifact_path         = "model",
            registered_model_name = "mosei-sentiment-transformer"
        )

        print(f'\n✅ MLflow run complete')
        print(f'   Best Val F1 (macro): {best_val_f1:.4f}')
        print(f'   Best Val Acc: {max(history["val_acc"]):.4f}')

    print('\nTraining complete!')
    return history