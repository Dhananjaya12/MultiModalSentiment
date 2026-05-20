import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from training.trainer import run_one_epoch
# from data.dataloader import idx_to_label
import mlflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def evaluate(model, test_loader, cfg) -> dict:
    """
    Loads the best saved checkpoint and evaluates on the test set.

    Returns a dict of meaningful metrics:
      MAE, Weighted MAE, Pearson, Spearman, Within-0.5, Within-1.0, MAE per region
    """
    # Load best checkpoint
    save_path = Path(cfg['model_save_path'])
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)

    _, _, test_preds, test_labels = run_one_epoch(
        model, test_loader, is_train=False
    )
    test_preds = snap_to_valid(test_preds, dataset='meld')

    # pred_floats  = np.array([idx_to_label(i) for i in test_preds])
    # label_floats = np.array([idx_to_label(i) for i in test_labels])

    pred_floats  = test_preds
    label_floats = test_labels

    # ── Core metrics ──────────────────────────────────────────────
    mae      = np.mean(np.abs(pred_floats - label_floats))
    pearson  = pearsonr(pred_floats,  label_floats)[0]
    spearman = spearmanr(pred_floats, label_floats)[0]

    # ── Within-N accuracy ─────────────────────────────────────────
    within_half = np.mean(np.abs(pred_floats - label_floats) <= 0.5)
    within_one  = np.mean(np.abs(pred_floats - label_floats) <= 1.0)

    # # ── Weighted MAE ──────────────────────────────────────────────
    # label_counts   = np.bincount(test_labels.astype(int), minlength=27).astype(float)
    # sample_weights = 1.0 / (label_counts[test_labels.astype(int)] + 1e-8)
    # sample_weights = sample_weights / sample_weights.sum()
    # weighted_mae   = np.average(np.abs(pred_floats - label_floats),
    #                             weights=sample_weights)

    # Map float labels to 27-class indices for weighting
    unique_labels, counts = np.unique(label_floats, return_counts=True)
    weight_map     = {l: 1.0/c for l, c in zip(unique_labels, counts)}
    sample_weights = np.array([weight_map[l] for l in label_floats])
    sample_weights = sample_weights / sample_weights.sum()
    weighted_mae   = np.average(np.abs(pred_floats - label_floats), weights=sample_weights)

    # ── MAE per sentiment region ──────────────────────────────────
    regions = {
        "strongly_negative": label_floats <= -2,
        "mildly_negative"  : (label_floats > -2)  & (label_floats < 0),
        "neutral"          : (label_floats >= 0)   & (label_floats <= 0.5),
        "mildly_positive"  : (label_floats > 0.5)  & (label_floats <= 2),
        "strongly_positive": label_floats > 2,
    }
    region_mae = {}
    for name, mask in regions.items():
        if mask.sum() > 0:
            region_mae[name] = float(np.abs(
                pred_floats[mask] - label_floats[mask]
            ).mean())
        else:
            region_mae[name] = None

    metrics = {
        'mae':          mae,
        'weighted_mae': weighted_mae,
        'pearson':      pearson,
        'corr':         pearson,
        'spearman':     spearman,
        'within_half':  within_half,
        'within_one':   within_one,
        'region_mae':   region_mae,
        'preds':        pred_floats,
        'labels':       label_floats,
    }

    # ── Log to MLflow ─────────────────────────────────────────────
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metrics({
            "test_mae":          mae,
            "test_weighted_mae": weighted_mae,
            "test_pearson":      pearson,
            "test_spearman":     spearman,
            "test_within_half":  within_half,
            "test_within_one":   within_one,
        })
        for name, val in region_mae.items():
            if val is not None:
                mlflow.log_metric(f"test_mae_{name}", val)

    # ── Print results ─────────────────────────────────────────────
    print('\n' + '═' * 52)
    print('           TEST SET RESULTS')
    print('═' * 52)
    print(f'  MAE                  {mae:.4f}   (lower is better)')
    print(f'  Weighted MAE         {weighted_mae:.4f}   (lower is better)')
    print(f'  Pearson  Corr        {pearson:.4f}   (higher is better)')
    print(f'  Spearman Corr        {spearman:.4f}   (higher is better)')
    print(f'  Within-0.5 Acc       {within_half*100:.2f}%')
    print(f'  Within-1.0 Acc       {within_one*100:.2f}%')
    print('\n  MAE per sentiment region:')
    for name, val in region_mae.items():
        if val is not None:
            print(f'    {name:22s}  {val:.4f}')
        else:
            print(f'    {name:22s}  no samples')
    print('═' * 52)

    return metrics