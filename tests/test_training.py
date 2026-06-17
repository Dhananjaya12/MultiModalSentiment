import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import TransformerFusionModel


def _ce_loss(output, sample_batch):
    """CrossEntropy loss helper matching the real training setup."""
    class_idx = (sample_batch['label'] + 1.0).round().long()
    return nn.CrossEntropyLoss()(output, class_idx)


@pytest.fixture(scope="function")
def untrained_model(device, cfg):
    """Fresh untrained model per test — training modifies weights."""
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
        loss = _ce_loss(output, sample_batch)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

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
        loss = _ce_loss(output, sample_batch)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in untrained_model.parameters()
        )
        assert has_grad, "No gradients flowing — backprop is broken"

    def test_loss_decreases_after_step(self, untrained_model, sample_batch):
        """Loss should decrease after one optimizer step."""
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)

        output_before = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        loss_before = _ce_loss(output_before, sample_batch)

        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        with torch.no_grad():
            output_after = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        loss_after = _ce_loss(output_after, sample_batch)

        assert loss_after < loss_before, \
            f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"

    def test_model_weights_change_after_step(self, untrained_model, sample_batch):
        """Model weights must actually update after optimizer step."""
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)
        weights_before = untrained_model.audio_encoder.projection[0].weight.clone()

        optimizer.zero_grad()
        output = untrained_model(
            sample_batch['input_ids'],
            sample_batch['attention_mask'],
            sample_batch['audio'],
            sample_batch['vision']
        )
        loss = _ce_loss(output, sample_batch)
        loss.backward()
        optimizer.step()

        weights_after = untrained_model.audio_encoder.projection[0].weight
        assert not torch.allclose(weights_before, weights_after), \
            "Weights did not change after optimizer step"


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
            out_no_audio = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                torch.zeros_like(sample_batch['audio']),
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
            out_no_vision = untrained_model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                torch.zeros_like(sample_batch['vision'])
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
        assert not torch.isnan(output).any(), "NaN with text-only input"
        assert not torch.isinf(output).any(), "Inf with text-only input"
        assert output.shape == (4, 3),        "Wrong output shape with text-only"


class TestCheckpointResume:
    """Tests that checkpoint saving and resuming works."""

    def test_checkpoint_saves_correctly(self, untrained_model, tmp_path, cfg):
        """Checkpoint must contain all required keys."""
        ckpt_path = tmp_path / 'checkpoint.pt'
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)

        torch.save({
            'epoch':           0,
            'model_state':     untrained_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_f1':     -1.0,
            'no_improve':      0,
            'history':         {'train_loss': [], 'train_acc': [],
                                'val_loss': [], 'val_acc': [], 'val_f1_macro': []},
        }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        for key in ['epoch', 'model_state', 'optimizer_state',
                    'scheduler_state', 'best_val_f1', 'no_improve', 'history']:
            assert key in ckpt, f"Missing key in checkpoint: {key}"

    def test_checkpoint_loads_correctly(self, untrained_model, tmp_path, cfg):
        """Model loaded from checkpoint must produce same output as original."""
        ckpt_path = tmp_path / 'checkpoint.pt'
        optimizer = optim.Adam(untrained_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0)

        torch.save({
            'epoch':           0,
            'model_state':     untrained_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_f1':     -1.0,
            'no_improve':      0,
            'history':         {'train_loss': [], 'train_acc': [],
                                'val_loss': [], 'val_acc': [], 'val_f1_macro': []},
        }, ckpt_path)

        model2 = TransformerFusionModel(cfg)
        ckpt   = torch.load(ckpt_path, map_location='cpu')
        model2.load_state_dict(ckpt['model_state'])

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
