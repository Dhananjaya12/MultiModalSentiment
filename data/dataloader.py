import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertTokenizer
# from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path

import numpy as np

# LABEL_VALUES = np.array([
#     -3., -2.6666667, -2.3333333, -2., -1.6666666, -1.3333334,
#     -1., -0.6666667, -0.5, -0.33333334, -0.16666667, 0.,
#     0.16666667, 0.33333334, 0.5, 0.6666667, 0.8333333, 1.,
#     1.1666666, 1.3333334, 1.5, 1.6666666, 1.8333334, 2.,
#     2.3333333, 2.6666667, 3.
# ], dtype=np.float32)

# def label_to_idx(label):
#     return int(np.argmin(np.abs(LABEL_VALUES - label)))

# def idx_to_label(idx):
#     return float(LABEL_VALUES[idx])


# ─────────────────────────────────────────────────────────────────
# Dataset Class
# Reads directly from HDF5 — loads one sample at a time,
# not everything into RAM at once.
# ─────────────────────────────────────────────────────────────────

class MOSEIDataset(Dataset):
    def __init__(self, hdf5_path: str, indices: list, cfg):
        """
        hdf5_path : path to mosei_dataset.h5
        indices   : which row indices belong to this split
                    e.g. [0, 1, 2, ..., 14000] for train
        tokenizer : DistilBERT tokenizer
        """
        self.path      = hdf5_path
        self.indices   = indices
        # self.tokenizer = tokenizer
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
        audio  = torch.tensor(f['audio'] [real_idx], dtype=torch.float)  # (300, 768)
        vision = torch.tensor(f['vision'][real_idx], dtype=torch.float)  # (300, 512)
        audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
        vision = (vision - vision.mean()) / (vision.std() + 1e-8)
        label  = torch.tensor(f['labels'][real_idx], dtype=torch.float)  # scalar
        # label  = torch.tensor(label_to_idx(f['labels'][real_idx]), dtype=torch.long)
        # text   = f['texts'][real_idx]

        # if isinstance(text, bytes):
        #     text = text.decode('utf-8')

        # # Tokenize text
        # enc = self.tokenizer(
        #     text,
        #     max_length  = self.cfg['max_text_len'],
        #     padding     = 'max_length',
        #     truncation  = True,
        #     return_tensors = 'pt'
        # )

        input_ids      = torch.tensor(f['input_ids'][real_idx], dtype=torch.long)
        attention_mask = torch.tensor(f['attention_mask'][real_idx], dtype=torch.long)  

        return {
            'input_ids':      input_ids,       # (128,)
            'attention_mask': attention_mask,  # (128,)
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
    # hdf5_path = Path(cfg['data_folder']) / "mosei_dataset.h5"
    hdf5_path = Path(cfg['data_folder']) / cfg.get('hdf5_file', 'meld_dataset.h5')

    # Get total number of samples
    with h5py.File(hdf5_path, 'r') as f:
        N = f['audio'].shape[0]
        # N = 10

        print(f"Total samples in HDF5: {N}")

    # # Split indices — never shuffle the actual data, just the indices
    # np.random.seed(cfg['seed'])
    # all_indices = np.random.permutation(N).tolist()

    # train_end = int(cfg['train_ratio'] * N)
    # val_end   = int((cfg['train_ratio'] + cfg['val_ratio']) * N)

    # train_idx = all_indices[:train_end]
    # val_idx   = all_indices[train_end:val_end]
    # test_idx  = all_indices[val_end:]

        if 'n_train' in f.attrs:
            n_train = int(f.attrs['n_train'])
            n_dev   = int(f.attrs['n_dev'])
            n_test  = int(f.attrs['n_test'])
            train_idx = list(range(0, n_train))
            val_idx   = list(range(n_train, n_train + n_dev))
            test_idx  = list(range(n_train + n_dev, n_train + n_dev + n_test))
            print(f"  Using fixed splits from HDF5 attrs")
        else:
            # Original random split for MOSEI
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

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    train_dataset = MOSEIDataset(hdf5_path, train_idx, cfg)
    val_dataset   = MOSEIDataset(hdf5_path, val_idx, cfg)
    test_dataset  = MOSEIDataset(hdf5_path, test_idx, cfg)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = pin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = 4,
        pin_memory  = pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = 4,
        pin_memory  = pin
    )

    return train_loader, val_loader, test_loader