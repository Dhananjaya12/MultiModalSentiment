import pytest
import torch
import numpy as np
from model.model import TransformerFusionModel


class TestModelArchitecture:
    """Tests model structure and initialization."""

    def test_model_instantiates(self, model):
        """Model should instantiate without errors."""
        assert model is not None

    def test_model_has_required_components(self, model):
        """Model must have all required submodules."""
        assert hasattr(model, 'roberta'),        "Missing roberta"
        assert hasattr(model, 'audio_encoder'),  "Missing audio_encoder"
        assert hasattr(model, 'vision_encoder'), "Missing vision_encoder"
        assert hasattr(model, 'text_encoder'),   "Missing text_encoder"
        assert hasattr(model, 'fusion'),         "Missing fusion"
        assert hasattr(model, 'regressor'),      "Missing regressor"

    def test_roberta_frozen(self, model):
        """RoBERTa must be frozen except last 2 layers (10 & 11)."""
        for name, param in model.roberta.named_parameters():
            if 'encoder.layer.11' not in name and \
               'encoder.layer.10' not in name:
                assert not param.requires_grad, \
                    f"RoBERTa param {name} should be frozen"

    def test_trainable_parameter_count(self, model):
        """Model should have reasonable number of trainable params."""
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 1_000_000,  f"Too few trainable params: {total:,}"
        assert total < 50_000_000, f"Too many trainable params: {total:,}"


class TestModelForward:
    """Tests model forward pass."""

    def test_output_shape(self, model, sample_batch):
        """Output must be (batch_size, 3) — one logit per class per sample."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        assert output.shape == (4, 3), \
            f"Expected output shape (4, 3), got {output.shape}"

    def test_output_is_logits_per_sample(self, model, sample_batch):
        """Output must be 2D (batch, num_classes)."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        assert output.ndim == 2, \
            f"Output should be 2D (batch, 3), got {output.ndim}D"
        assert output.shape[1] == 3, \
            f"Output should have 3 classes, got {output.shape[1]}"

    def test_no_nan_in_output(self, model, sample_batch):
        """Model must not produce NaN predictions."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        assert not torch.isnan(output).any(), \
            f"Model produced NaN outputs: {output}"

    def test_no_inf_in_output(self, model, sample_batch):
        """Model must not produce Inf predictions."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        assert not torch.isinf(output).any(), \
            f"Model produced Inf outputs: {output}"

    def test_output_in_reasonable_range(self, model, sample_batch):
        """Logits from an initialised model should be within [-50, 50]."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
        assert output.abs().max() < 50.0, \
            f"Logits out of reasonable range: {output}"

    def test_different_inputs_give_different_outputs(self, model, sample_batch):
        """Model must not output same value for all inputs."""
        audio2  = torch.randn_like(sample_batch['audio'])  * 5
        vision2 = torch.randn_like(sample_batch['vision']) * 5

        with torch.no_grad():
            out1 = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )
            out2 = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                audio2, vision2
            )

        assert not torch.allclose(out1, out2), \
            "Model gives identical output for different inputs — something is wrong"


class TestSavedModel:
    """Tests that the saved best_model.pt loads and works correctly."""

    def test_saved_model_loads(self, cfg, device):
        """Saved model must load without errors."""
        from pathlib import Path
        path = Path(cfg['model_save_path'])

        if not path.exists():
            pytest.skip("No saved model found — skipping")

        model = TransformerFusionModel(cfg)
        model.load_state_dict(torch.load(path, map_location=device))
        assert model is not None

    def test_saved_model_produces_valid_output(self, model, sample_batch):
        """Saved model must produce valid predictions."""
        with torch.no_grad():
            output = model(
                sample_batch['input_ids'],
                sample_batch['attention_mask'],
                sample_batch['audio'],
                sample_batch['vision']
            )

        assert not torch.isnan(output).any(), "Saved model produces NaN"
        assert not torch.isinf(output).any(), "Saved model produces Inf"
        assert output.shape == (4, 3),        "Saved model wrong output shape"
