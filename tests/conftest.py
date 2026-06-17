import pytest
import torch
import numpy as np
import h5py
import json
from pathlib import Path
from model.model import TransformerFusionModel


@pytest.fixture(scope="session")
def cfg():
    """Load config once for all tests."""
    with open("config.json") as f:
        cfg = json.load(f)
    return cfg


@pytest.fixture(scope="session")
def device():
    """Device to run tests on."""
    return torch.device('cpu')  # always CPU for tests — faster, consistent


# @pytest.fixture(scope="session")
# def tokenizer():
#     """Load tokenizer once for all tests."""
#     return RobertaTokenizer.from_pretrained('roberta-large')


@pytest.fixture(scope="session")
def model(cfg, device):
    """
    Load the trained model once for all tests.
    Uses the saved best_model.pt.
    """
    m = TransformerFusionModel(cfg)
    model_path = Path(cfg['model_save_path'])

    if model_path.exists():
        m.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ Model loaded from {model_path}")
    else:
        print(f"\n⚠️  No saved model found at {model_path} — using random weights")

    m.eval()
    return m.to(device)


@pytest.fixture(scope="session")
def sample_batch(cfg, device):   # ← remove tokenizer from args
    batch_size = 4
    audio  = torch.randn(batch_size, cfg['seq_len'], cfg['audio_dim']).to(device)
    vision = torch.randn(batch_size, cfg['seq_len'], cfg['vision_dim']).to(device)

    # Create fake pre-tokenized tensors — no tokenizer needed
    input_ids      = torch.randint(0, 50265, (batch_size, cfg['max_text_len'])).to(device)
    attention_mask = torch.ones(batch_size, cfg['max_text_len'], dtype=torch.long).to(device)

    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'audio':          audio,
        'vision':         vision,
        'label':          torch.tensor([-1., 0., 1., 0.]).to(device),  # valid MELD labels
    }


@pytest.fixture(scope="session")
def hdf5_sample(cfg):
    with h5py.File(Path(cfg['data_folder']) / cfg.get('hdf5_file', 'meld_dataset_v2.h5'), 'r') as f:
        return {
            'audio':          f['audio'][:10],
            'vision':         f['vision'][:10],
            'labels':         f['labels'][:10],
            'texts':          f['texts'][:10],
            'input_ids':      f['input_ids'][:10],       # ← ADD
            'attention_mask': f['attention_mask'][:10],  # ← ADD
            'total':          f['audio'].shape[0]
        }
    
@pytest.fixture(scope="session")
def test_loader(cfg):
    """
    Test dataloader — loaded once for all pipeline tests.
    scope="session" so it doesn't reload for every test.
    """
    from data.dataloader import get_dataloaders
    _, _, test_loader = get_dataloaders(cfg)
    return test_loader

@pytest.fixture(scope="session")
def train_loader(cfg):
    from data.dataloader import get_dataloaders
    train_loader, _, _ = get_dataloaders(cfg)
    return train_loader