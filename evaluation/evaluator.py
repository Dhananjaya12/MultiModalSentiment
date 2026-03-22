import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from training.trainer import run_one_epoch
import mlflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, test_loader, cfg) -> dict:
    """
    Loads the best saved checkpoint and evaluates on the test set.

    Returns a dict of all standard MOSEI metrics:
      MAE, Pearson Corr, Acc-2, F1, Acc-7
    """
    # Load best checkpoint
    save_path = Path(cfg['model_save_path'])
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)

    _, test_mae, test_preds, test_labels = run_one_epoch(
        model, test_loader, is_train=False
    )

    # Pearson correlation
    test_corr = pearsonr(test_preds, test_labels)[0]

    # Binary accuracy: positive (>0) vs non-positive (≤0)
    bin_preds  = (test_preds  > 0).astype(int)
    bin_labels = (test_labels > 0).astype(int)
    bin_acc    = accuracy_score(bin_labels, bin_preds)
    bin_f1     = f1_score(bin_labels, bin_preds, average='weighted')

    # 7-class accuracy: round to nearest integer in [-3, 3]
    preds7  = np.clip(np.round(test_preds),  -3, 3)
    labels7 = np.clip(np.round(test_labels), -3, 3)
    acc7    = accuracy_score(labels7, preds7)

    metrics = {
        'mae':    test_mae,
        'corr':   test_corr,
        'acc2':   bin_acc,
        'f1':     bin_f1,
        'acc7':   acc7,
        'preds':  test_preds,
        'labels': test_labels,
    }

    # ── Log test metrics to the SAME MLflow run ───────────────────
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metrics({
            "test_mae":  test_mae,
            "test_corr": test_corr,
            "test_acc2": bin_acc,
            "test_f1":   bin_f1,
            "test_acc7": acc7,
        })

    # Print results
    print('═' * 52)
    print('           TEST SET RESULTS')
    print('═' * 52)
    print(f'  MAE            {test_mae:.4f}   (lower is better)')
    print(f'  Pearson Corr   {test_corr:.4f}   (higher is better)')
    print(f'  Accuracy-2     {bin_acc:.4f}   (pos vs non-pos)')
    print(f'  F1 Score       {bin_f1:.4f}')
    print(f'  Accuracy-7     {acc7:.4f}   (7-class)')
    print('═' * 52)

    return metrics