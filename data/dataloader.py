import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Dataset Class
# Reads directly from HDF5 — loads one sample at a time,
# not everything into RAM at once.
# ─────────────────────────────────────────────────────────────────

class MOSEIDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list, tokenizer, cfg):
        """
        hdf5_path : path to mosei_dataset.h5
        indices   : which row indices belong to this split
                    e.g. [0, 1, 2, ..., 14000] for train
        tokenizer : DistilBERT tokenizer
        """
        self.path      = hdf5_path
        self.indices   = indices
        self.tokenizer = tokenizer
        self.file      = None   # opened lazily per worker
        self.cfg = cfg

    def __len__(self):
        return len(self.indices)

    def _get_file(self):
        # Open once per worker — important for multi-worker DataLoader
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        return self.file

    def __getitem__(self, idx):
        real_idx = self.indices[idx]   # map split index → actual HDF5 row
        f = self._get_file()

        # Load just this one sample — no need to load everything
        audio  = torch.tensor(f['audio'] [real_idx], dtype=torch.float)  # (500, 74)
        vision = torch.tensor(f['vision'][real_idx], dtype=torch.float)  # (500, 713)
        label  = torch.tensor(f['labels'][real_idx], dtype=torch.float)  # scalar
        text   = f['texts'][real_idx]

        if isinstance(text, bytes):
            text = text.decode('utf-8')

        # Tokenize text
        enc = self.tokenizer(
            text,
            max_length  = self.cfg['max_text_len'],
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt'
        )

        return {
            'input_ids':      enc['input_ids'].squeeze(),       # (128,)
            'attention_mask': enc['attention_mask'].squeeze(),  # (128,)
            'audio':          audio,                            # (500, 74)
            'vision':         vision,                           # (500, 713)
            'label':          label,                            # scalar
        }


# ─────────────────────────────────────────────────────────────────
# Build DataLoaders
# ─────────────────────────────────────────────────────────────────

def get_dataloaders(cfg):
    """
    Reads total sample count from HDF5,
    splits indices into train/val/test,
    returns three DataLoaders.
    """
    hdf5_path = Path(cfg['data_folder']) / "mosei_dataset.h5"

    # Get total number of samples
    with h5py.File(hdf5_path, 'r') as f:
        N = f['audio'].shape[0]
        # N = 10

    print(f"Total samples in HDF5: {N}")

    # Split indices — never shuffle the actual data, just the indices
    np.random.seed(cfg['seed'])
    all_indices = np.random.permutation(N).tolist()

    train_end = int(cfg['train_ratio'] * N)
    val_end   = int((cfg['train_ratio'] + cfg['val_ratio']) * N)

    train_idx = all_indices[:train_end]
    val_idx   = all_indices[train_end:val_end]
    test_idx  = all_indices[val_end:]

    print(f"  Train : {len(train_idx)} samples")
    print(f"  Val   : {len(val_idx)}   samples")
    print(f"  Test  : {len(test_idx)}  samples")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = MOSEIDataset(hdf5_path, train_idx, tokenizer, cfg)
    val_dataset   = MOSEIDataset(hdf5_path, val_idx,   tokenizer, cfg)
    test_dataset  = MOSEIDataset(hdf5_path, test_idx,  tokenizer, cfg)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = 2,
        pin_memory  = pin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = 2,
        pin_memory  = pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = 2,
        pin_memory  = pin
    )

    return train_loader, val_loader, test_loader