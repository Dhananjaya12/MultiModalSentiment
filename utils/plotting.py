import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history: dict, cfg):
    """Plots loss, MAE, and Pearson correlation curves."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val',   color='coral')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_mae'], label='Train', color='steelblue')
    axes[1].plot(history['val_mae'],   label='Val',   color='coral')
    axes[1].set_title('MAE per Epoch  (lower = better)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()

    axes[2].plot(history['val_corr'], color='green')
    axes[2].set_title('Val Pearson Corr  (higher = better)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Correlation')

    plt.suptitle('Transformer Fusion — Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = Path(cfg['plots_save_path']) / 'training_history.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved → {out}')


def plot_predictions(metrics: dict, cfg):
    """Scatter plot of predictions vs ground truth + error distribution."""

    test_preds  = metrics['preds']
    test_labels = metrics['labels']
    test_corr   = metrics['corr']
    test_mae    = metrics['mae']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: predicted vs actual
    axes[0].scatter(test_labels, test_preds, alpha=0.3, s=10, color='steelblue')
    axes[0].plot([-3, 3], [-3, 3], 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel('True Sentiment Score')
    axes[0].set_ylabel('Predicted Sentiment Score')
    axes[0].set_title(f'Predictions vs Ground Truth\nCorr={test_corr:.3f}  |  MAE={test_mae:.3f}')
    axes[0].legend()

    # Error distribution
    errors = test_preds - test_labels
    axes[1].hist(errors, bins=40, color='steelblue', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero error')
    axes[1].set_xlabel('Prediction Error  (pred − true)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].legend()

    plt.tight_layout()

    out = Path(cfg['plots_save_path']) / 'predictions.png'
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved → {out}')