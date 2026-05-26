import torch
import numpy as np
import json
from data.dataloader import get_dataloaders
from model.model import TransformerFusionModel
from training.trainer import train
from evaluation.evaluator import evaluate
from utils.plotting import plot_training_history, plot_predictions

import time

def main():

    with open("config.json") as f:
        cfg = json.load(f)

    # Reproducibility
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Step 1: Load data ─────────────────────────────────────────
    print('Loading data from HDF5...')
    t_data_loader_start = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    t_data_loader_end = time.time()
    print(f'Data loading time: {t_data_loader_end - t_data_loader_start:.2f} seconds\n')

    # ── Step 2: Build model ───────────────────────────────────────
    t_model_build_start = time.time()
    print('\nBuilding model...')
    model = TransformerFusionModel(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,}  (RoBERTa frozen except last 2 layers)\n')
    t_model_build_end = time.time()
    print(f'Model building time: {t_model_build_end - t_model_build_start:.2f} seconds\n')

    # ── Step 3: Train ─────────────────────────────────────────────
    t_train_start = time.time()
    history = train(model, train_loader, val_loader, cfg)
    t_train_end = time.time()
    print(f'Training time: {t_train_end - t_train_start:.2f} seconds\n')

    # ── Step 4: Evaluate ──────────────────────────────────────────
    t_eval_start = time.time()
    print('\nEvaluating on test set...')
    metrics = evaluate(model, test_loader, cfg)
    t_eval_end = time.time()
    print(f'Evaluation time: {t_eval_end - t_eval_start:.2f} seconds\n')

    # ── Step 5: Plot ──────────────────────────────────────────────
    t_plot_start = time.time()
    print('\nSaving plots...')
    plot_training_history(history, cfg)
    plot_predictions(metrics, cfg)
    t_plot_end = time.time()
    print(f'Plotting time: {t_plot_end - t_plot_start:.2f} seconds\n')

    print('\n✅ All done.')


if __name__ == '__main__':
    main()