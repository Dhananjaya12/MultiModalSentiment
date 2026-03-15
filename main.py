import torch
import numpy as np
import json
from data.dataloader import get_dataloaders
from model.model import TransformerFusionModel
from training.trainer import train
from evaluation.evaluator import evaluate
from utils.plotting import plot_training_history, plot_predictions

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
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Step 2: Build model ───────────────────────────────────────
    print('\nBuilding model...')
    model = TransformerFusionModel(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,}  (DistilBERT frozen)\n')

    # ── Step 3: Train ─────────────────────────────────────────────
    history = train(model, train_loader, val_loader, cfg)

    # ── Step 4: Evaluate ──────────────────────────────────────────
    print('\nEvaluating on test set...')
    metrics = evaluate(model, test_loader, cfg)

    # ── Step 5: Plot ──────────────────────────────────────────────
    print('\nSaving plots...')
    plot_training_history(history, cfg)
    plot_predictions(metrics, cfg)

    print('\n✅ All done.')


if __name__ == '__main__':
    main()