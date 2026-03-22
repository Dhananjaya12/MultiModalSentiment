import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import TransformerFusionModel

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
        loss = nn.L1Loss()(output, sample_batch['label'])
        loss.backward()
        optimizer.step()

        # Weights after
        weights_after = untrained_model.audio_encoder.projection[0].weight

        assert not torch.allclose(weights_before, weights_after), \
            "Weights did not change after optimizer step"