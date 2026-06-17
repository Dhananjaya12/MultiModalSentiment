import torch
import numpy as np
from pathlib import Path
from training.trainer import run_one_epoch, compute_class_weights, compute_classification_metrics, CLASS_NAMES
import mlflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, test_loader, cfg) -> dict:
    """
    Loads best checkpoint and evaluates on the test set.
    Returns accuracy, macro F1, per-class metrics, and confusion matrix.
    """
    save_path = Path(cfg['model_save_path'])
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)

    class_weights = compute_class_weights(test_loader).to(device)
    test_loss, test_acc, test_preds, test_labels = run_one_epoch(
        model, test_loader, is_train=False, class_weights=class_weights
    )

    cls = compute_classification_metrics(test_preds, test_labels, num_classes=3)

    metrics = {
        'loss':             test_loss,
        'accuracy':         cls['accuracy'],
        'f1_macro':         cls['f1_macro'],
        'precision':        cls['precision'],
        'recall':           cls['recall'],
        'f1':               cls['f1'],
        'confusion_matrix': cls['confusion_matrix'],
        'preds':            test_preds,
        'labels':           test_labels,
    }

    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metrics({
            "test_loss":     test_loss,
            "test_accuracy": cls['accuracy'],
            "test_f1_macro": cls['f1_macro'],
        })
        for i, name in enumerate(CLASS_NAMES):
            mlflow.log_metric(f"test_precision_{name}", cls['precision'][i])
            mlflow.log_metric(f"test_recall_{name}",    cls['recall'][i])
            mlflow.log_metric(f"test_f1_{name}",        cls['f1'][i])

    print('\n' + '═' * 56)
    print('              TEST SET RESULTS')
    print('═' * 56)
    print(f'  Loss                 {test_loss:.4f}')
    print(f'  Accuracy             {cls["accuracy"]*100:.2f}%')
    print(f'  F1 (macro)           {cls["f1_macro"]:.4f}')
    print('\n  Per-class metrics:')
    header = f'  {"":10s}  {"Precision":>10s}  {"Recall":>10s}  {"F1":>10s}'
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:10s}  {cls["precision"][i]:>10.4f}  '
              f'{cls["recall"][i]:>10.4f}  {cls["f1"][i]:>10.4f}')
    print('\n  Confusion matrix (rows=true, cols=pred):')
    print(f'  {"":10s}  ' + '  '.join(f'{n:>10s}' for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row_str = '  '.join(f'{v:>10d}' for v in cls['confusion_matrix'][i])
        print(f'  {name:10s}  {row_str}')
    print('═' * 56)

    return metrics
