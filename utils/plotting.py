import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = ['negative', 'neutral', 'positive']


def plot_training_history(history: dict, cfg):
    """Loss, accuracy, and macro F1 curves."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val',   color='coral')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train', color='steelblue')
    axes[1].plot(history['val_acc'],   label='Val',   color='coral')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(history['val_f1_macro'], color='green')
    axes[2].set_title('Val Macro F1  (higher = better)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].set_ylim(0, 1)

    plt.suptitle('Transformer Fusion — Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = Path(cfg['plots_save_path']) / 'training_history.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved → {out}')


def plot_predictions(metrics: dict, cfg):
    """Confusion matrix heatmap + per-class F1 bar chart."""

    cm = metrics['confusion_matrix']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion matrix
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(len(CLASS_NAMES)))
    axes[0].set_yticks(range(len(CLASS_NAMES)))
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_yticklabels(CLASS_NAMES)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'Confusion Matrix  (Acc={metrics["accuracy"]*100:.2f}%)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, ax=axes[0])

    # Per-class F1
    axes[1].bar(CLASS_NAMES, metrics['f1'], color=['#e05c5c', '#5c8ae0', '#5cba6a'])
    axes[1].set_title(f'Per-class F1  (macro={metrics["f1_macro"]:.3f})')
    axes[1].set_ylabel('F1 score')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(metrics['f1']):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)

    plt.tight_layout()

    out = Path(cfg['plots_save_path']) / 'predictions.png'
    plt.savefig(out, dpi=150)
    plt.show()
    print(f'Saved → {out}')
