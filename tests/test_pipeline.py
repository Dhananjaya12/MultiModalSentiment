import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr


class TestDataLoader:
    """Tests that DataLoader works correctly."""

    def test_dataloader_loads(self, train_loader):
        """DataLoader must initialize without errors."""
        # from data.dataloader import get_dataloaders
        # train_loader, val_loader, test_loader = get_dataloaders()
        assert train_loader is not None
        # assert val_loader   is not None
        # assert test_loader  is not None

    def test_batch_shapes(self, train_loader, cfg):
        """Every batch must have correct shapes."""
        # from data.dataloader import get_dataloaders
        # train_loader, _, _ = get_dataloaders()

        batch = next(iter(train_loader))

        B = cfg['batch_size']
        assert batch['audio'].shape  == (B, cfg['seq_len'], cfg['audio_dim']),  \
            f"Wrong audio shape:  {batch['audio'].shape}"
        assert batch['vision'].shape == (B, cfg['seq_len'], cfg['vision_dim']), \
            f"Wrong vision shape: {batch['vision'].shape}"
        assert batch['input_ids'].shape      == (B, cfg['max_text_len']), \
            f"Wrong input_ids shape: {batch['input_ids'].shape}"
        assert batch['attention_mask'].shape == (B, cfg['max_text_len']), \
            f"Wrong attention_mask shape"
        assert batch['label'].shape == (B,), \
            f"Wrong label shape: {batch['label'].shape}"

    def test_no_nan_in_batches(self, train_loader):
        """First 5 batches must not contain NaN."""
        # from data.dataloader import get_dataloaders
        # train_loader, _, _ = get_dataloaders()

        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            assert not torch.isnan(batch['audio']).any(),  \
                f"NaN in audio at batch {i}"
            assert not torch.isnan(batch['vision']).any(), \
                f"NaN in vision at batch {i}"
            assert not torch.isnan(batch['label']).any(),  \
                f"NaN in labels at batch {i}"

    def test_labels_in_range(self, train_loader):
        """Labels must be within [-3, 3]."""
        # from data.dataloader import get_dataloaders
        # train_loader, _, _ = get_dataloaders()

        batch = next(iter(train_loader))
        labels = batch['label']

        assert labels.min() >= -3.0, f"Label below -3: {labels.min()}"
        assert labels.max() <=  3.0, f"Label above +3: {labels.max()}"


class TestMetrics:
#     """Tests evaluation metrics are computed correctly."""

#     def test_mae_calculation(self):
#         """MAE should be computed correctly."""
#         preds  = np.array([1.0, 2.0, 3.0])
#         labels = np.array([1.5, 2.5, 3.5])
#         mae    = np.mean(np.abs(preds - labels))
#         assert abs(mae - 0.5) < 1e-6, f"MAE calculation wrong: {mae}"

#     def test_pearson_correlation(self):
#         """Pearson correlation of perfect predictions should be 1.0."""
#         preds  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#         labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#         corr   = pearsonr(preds, labels)[0]
#         assert abs(corr - 1.0) < 1e-6, f"Pearson corr wrong: {corr}"

#     def test_binary_accuracy(self):
#         """Binary accuracy should be 1.0 for correct predictions."""
#         preds  = np.array([ 1.0, -1.0,  2.0, -2.0])
#         labels = np.array([ 0.5, -0.5,  1.5, -1.5])
#         bin_p  = (preds  > 0).astype(int)
#         bin_l  = (labels > 0).astype(int)
#         acc    = accuracy_score(bin_l, bin_p)
#         assert acc == 1.0, f"Binary accuracy wrong: {acc}"

    def test_trained_model_beats_random(self, model, test_loader, device):
        """Trained model must beat random baseline MAE of ~1.5."""

        all_preds, all_labels = [], []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 10:  # test on 10 batches
                    break
                preds = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['audio'].to(device),
                    batch['vision'].to(device)
                )
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        mae = np.mean(np.abs(
            np.array(all_preds) - np.array(all_labels)
        ))

        # Random baseline MAE on MOSEI is ~1.4
        assert mae < 1.4, \
            f"Model MAE {mae:.4f} doesn't beat random baseline 1.4"