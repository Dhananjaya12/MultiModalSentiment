import pytest
import torch
import numpy as np
import h5py
import json
from pathlib import Path
from transformers import DistilBertTokenizer

from config.config_loader import load_config
from model.model import TransformerFusionModel


@pytest.fixture(scope="session")
def cfg():
    """Load config once for all tests."""
    return load_config()


@pytest.fixture(scope="session")
def device():
    """Device to run tests on."""
    return torch.device('cpu')  # always CPU for tests — faster, consistent


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once for all tests."""
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


@pytest.fixture(scope="session")
def model(cfg, device):
    """
    Load the trained model once for all tests.
    Uses the saved best_model.pt.
    """
    m = TransformerFusionModel()
    model_path = Path(cfg['model_save_path'])

    if model_path.exists():
        m.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ Model loaded from {model_path}")
    else:
        print(f"\n⚠️  No saved model found at {model_path} — using random weights")

    m.eval()
    return m.to(device)


@pytest.fixture(scope="session")
def sample_batch(cfg, device, tokenizer):
    """
    Creates one small fake batch that looks exactly like real data.
    Used by model tests — no need to load real HDF5 for basic tests.
    """
    batch_size = 4

    # Fake audio (500, 74) per sample
    audio = torch.randn(batch_size, cfg['seq_len'], cfg['audio_dim']).to(device)

    # Fake vision (500, 713) per sample
    vision = torch.randn(batch_size, cfg['seq_len'], cfg['vision_dim']).to(device)

    # Fake text
    texts = ["I really loved this movie"] * batch_size
    enc   = tokenizer(
        texts,
        max_length     = cfg['max_text_len'],
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    return {
        'input_ids':      enc['input_ids'].to(device),
        'attention_mask': enc['attention_mask'].to(device),
        'audio':          audio,
        'vision':         vision,
        'label':          torch.randn(batch_size).to(device),
    }


@pytest.fixture(scope="session")
def hdf5_sample(cfg):
    """
    Loads first 10 samples from real HDF5 file.
    Used for data quality tests.
    """
    with h5py.File(cfg['hdf5_path'], 'r') as f:
        return {
            'audio':  f['audio'][:10],    # (10, 500, 74)
            'vision': f['vision'][:10],   # (10, 500, 713)
            'labels': f['labels'][:10],   # (10,)
            'texts':  f['texts'][:10],    # (10,)
            'total':  f['audio'].shape[0] # total samples
        }
    
@pytest.fixture(scope="session")
def test_loader(cfg):
    """
    Test dataloader — loaded once for all pipeline tests.
    scope="session" so it doesn't reload for every test.
    """
    from data.dataloader import get_dataloaders
    _, _, test_loader = get_dataloaders()
    return test_loader

@pytest.fixture(scope="session")
def train_loader(cfg):
    from data.dataloader import get_dataloaders
    train_loader, _, _ = get_dataloaders()
    return train_loader