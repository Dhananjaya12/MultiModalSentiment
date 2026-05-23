import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import TransformerFusionModel
# from data.dataloader import label_to_idx, LABEL_VALUES

@pytest.fixture(scope="function")   # fresh model for EACH training test
def untrained_model(device, cfg):
    """
    Fresh untrained model for training tests.
    Cannot share with other tests because training modifies weights.
    scope="function" means a new one per test — intentional here.
    """
    model = TransformerFusionModel(cfg).to(device)
    model.train()
    return model

class TestTrainingStep:
    """Tests that a single training step works correctly."""

    def test_loss_is_finite(self, untrained_model, sample_batch):
        """Loss must be a finite number after one forward pass."""

        output = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        # labels_idx = torch.tensor(
        #     [label_to_idx(l.item()) for l in sample_batch['label']], dtype=torch.long
        # )
        # loss = nn.CrossEntropyLoss()(output, labels_idx)

        loss = nn.L1Loss()(output, sample_batch['label'])

        assert torch.isfinite(loss), \
            f"Loss is not finite: {loss.item()}"

    def test_gradients_flow(self, untrained_model, sample_batch):
        """Gradients must flow back through the model."""
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        output = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        # labels_idx = torch.tensor(
        #     [label_to_idx(l.item()) for l in sample_batch['label']], dtype=torch.long
        # )
        # loss = nn.CrossEntropyLoss()(output, labels_idx)

        loss = nn.L1Loss()(output, sample_batch['label'])
        loss.backward()

        # Check at least some gradients are non-zero
        has_grad = False
        for param in untrained_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients flowing — backprop is broken"

    def test_loss_decreases_after_step(self, untrained_model, sample_batch):
        """Loss should decrease after one optimizer step."""
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)

        # Loss before step
        output_before = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        # labels_idx  = torch.tensor(
        #     [label_to_idx(l.item()) for l in sample_batch['label']], dtype=torch.long
        # )
        # loss_before = nn.CrossEntropyLoss()(output_before, labels_idx)
        loss_before = nn.L1Loss()(output_before, sample_batch['label'])
        # Take one step
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Loss after step
        with torch.no_grad():
            output_after = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        loss_after = nn.L1Loss()(output_after, sample_batch['label'])

        assert loss_after < loss_before, \
            f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"

    def test_model_weights_change_after_step(self, untrained_model, sample_batch):
        """Model weights must actually update after optimizer step."""
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)

        # Save weights before
        weights_before = untrained_model.audio_encoder.projection[0].weight.clone()

        # One training step
        # model.train()
        optimizer.zero_grad()
        output = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        # labels_idx = torch.tensor(
        #     [label_to_idx(l.item()) for l in sample_batch['label']], dtype=torch.long
        # )
        # loss = nn.CrossEntropyLoss()(output, labels_idx)

        loss = nn.L1Loss()(output, sample_batch['label'])
        loss.backward()
        optimizer.step()

        # Weights after
        weights_after = untrained_model.audio_encoder.projection[0].weight

        assert not torch.allclose(weights_before, weights_after), \
            "Weights did not change after optimizer step"

# ADD at the bottom of test_training.py

class TestModalityDropout:
    """Tests that modality dropout works correctly."""

    def test_audio_dropout_produces_different_output(self, untrained_model, sample_batch):
        """Zeroing audio should change model output."""
        untrained_model.eval()
        with torch.no_grad():
            out_full = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
            audio_zeros = torch.zeros_like(sample_batch['audio'])
            out_no_audio = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                audio_zeros,
                sample_batch['vision']
            )
        assert not torch.allclose(out_full, out_no_audio), \
            "Zeroing audio had no effect — audio encoder is not contributing"

    def test_vision_dropout_produces_different_output(self, untrained_model, sample_batch):
        """Zeroing vision should change model output."""
        untrained_model.eval()
        with torch.no_grad():
            out_full = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
            vision_zeros = torch.zeros_like(sample_batch['vision'])
            out_no_vision = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                vision_zeros
            )
        assert not torch.allclose(out_full, out_no_vision), \
            "Zeroing vision had no effect — vision encoder is not contributing"

    def test_text_only_produces_valid_output(self, untrained_model, sample_batch):
        """Model must not crash with only text (zeros for audio+vision)."""
        untrained_model.eval()
        with torch.no_grad():
            output = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                torch.zeros_like(sample_batch['audio']),
                torch.zeros_like(sample_batch['vision'])
            )
        assert not torch.isnan(output).any(),  "NaN with text-only input"
        assert not torch.isinf(output).any(),  "Inf with text-only input"
        assert output.shape == (4,),           "Wrong output shape with text-only"


class TestCheckpointResume:
    """Tests that checkpoint saving and resuming works."""

    def test_checkpoint_saves_correctly(self, untrained_model, tmp_path, cfg):
        """Checkpoint must contain all required keys."""
        import torch
        ckpt_path = tmp_path / 'checkpoint.pt'
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)

        torch.save({
            'epoch':            0,
            'model_state':      untrained_model.state_dict(),
            'optimizer_state':  optimizer.state_dict(),
            'scheduler_state':  scheduler.state_dict(),
            'best_val_mae_raw': 0.5,
            'no_improve':       0,
            'history':          {'train_loss': [], 'train_mae': [],
                                 'val_loss': [], 'val_mae': [], 'val_corr': [],
                                 'val_mae_snap': [], 'val_corr_snap': []},
        }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        for key in ['epoch', 'model_state', 'optimizer_state',
                    'scheduler_state', 'best_val_mae_raw', 'no_improve', 'history']:
            assert key in ckpt, f"Missing key in checkpoint: {key}"

    def test_checkpoint_loads_correctly(self, untrained_model, tmp_path, cfg):
        """Model loaded from checkpoint must produce same output as original."""
        from model.model import TransformerFusionModel
        import torch

        ckpt_path = tmp_path / 'checkpoint.pt'
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)

        torch.save({
            'epoch':            0,
            'model_state':      untrained_model.state_dict(),
            'optimizer_state':  optimizer.state_dict(),
            'scheduler_state':  scheduler.state_dict(),
            'best_val_mae_raw': 0.5,
            'no_improve':       0,
            'history':          {'train_loss': [], 'train_mae': [],
                                 'val_loss': [], 'val_mae': [], 'val_corr': [],
                                 'val_mae_snap': [], 'val_corr_snap': []},
        }, ckpt_path)

        # Load into new model
        model2 = TransformerFusionModel(cfg)
        ckpt   = torch.load(ckpt_path, map_location='cpu')
        model2.load_state_dict(ckpt['model_state'])

        # Both models must produce identical output
        untrained_model.eval()
        model2.eval()
        x = torch.randint(0, 100, (2, cfg['max_text_len']))
        m = torch.ones(2, cfg['max_text_len'], dtype=torch.long)
        a = torch.randn(2, cfg['seq_len'], cfg['audio_dim'])
        v = torch.randn(2, cfg['seq_len'], cfg['vision_dim'])

        with torch.no_grad():
            out1 = untrained_model(x, m, a, v)
            out2 = model2(x, m, a, v)

        assert torch.allclose(out1, out2, atol=1e-5), \
            "Loaded model produces different output than original"