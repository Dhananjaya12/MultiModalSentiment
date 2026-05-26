import h5py
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time


class MOSEIDataset(Dataset):
    # def __init__(self, hdf5_path: str, indices: list, cfg):
    #     """
    #     Loads entire split into RAM at init time.
    #     After loading, __getitem__ serves from RAM — no disk reads during training.
    #     """
    #     self.indices = indices
    #     self.cfg     = cfg

    #     print(f'  Loading {len(indices)} samples into RAM...')
    #     t = time.time()

    #     sorted_idx = sorted(indices)

    #     with h5py.File(hdf5_path, 'r') as f:
    #         self.audio          = f['audio']         [sorted_idx]
    #         self.vision         = f['vision']        [sorted_idx]
    #         self.labels         = f['labels']        [sorted_idx]
    #         self.input_ids      = f['input_ids']     [sorted_idx]
    #         self.attention_mask = f['attention_mask'][sorted_idx]

        
    #     # Map original index → position in loaded arrays
    #     self.idx_map = {orig: new for new, orig in enumerate(sorted_idx)}
    #     print(f'  ✅ Loaded into RAM in {time.time()-t:.1f}s  '
    #           f'({self.audio.nbytes/1024**3:.1f}GB audio + '
    #           f'{self.vision.nbytes/1024**3:.1f}GB vision)')

    # def __len__(self):
    #     return len(self.indices)

    # def __getitem__(self, idx):
    #     i      = self.idx_map[self.indices[idx]]
    #     audio  = torch.tensor(self.audio[i],  dtype=torch.float32)
    #     vision = torch.tensor(self.vision[i], dtype=torch.float32)

    #     # Normalize per sample — same as before
    #     audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
    #     vision = (vision - vision.mean()) / (vision.std() + 1e-8)

    #     return {
    #         'input_ids':      torch.tensor(self.input_ids[i],      dtype=torch.long),
    #         'attention_mask': torch.tensor(self.attention_mask[i], dtype=torch.long),
    #         'audio':          audio,
    #         'vision':         vision,
    #         'label':          torch.tensor(self.labels[i],         dtype=torch.float32),
    #     }

    def __init__(self, hdf5_path: str, indices: list, cfg):
        """
        Stores indices only. Reads from HDF5 on demand in __getitem__.
        One HDF5 file handle opened per worker (lazy init).
        """
        self.hdf5_path = str(hdf5_path)
        self.indices   = indices
        self.cfg       = cfg
        self._file     = None
        print(f'  Dataset ready: {len(indices)} samples (HDF5 on-demand loading)')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')

        i      = self.indices[idx]
        audio  = torch.from_numpy(self._file['audio'][i].copy()).float()
        vision = torch.from_numpy(self._file['vision'][i].copy()).float()

        # Normalize per sample — same as before
        audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
        vision = (vision - vision.mean()) / (vision.std() + 1e-8)

        return {
            'input_ids':      torch.from_numpy(self._file['input_ids'][i].copy()).long(),
            'attention_mask': torch.from_numpy(self._file['attention_mask'][i].copy()).long(),
            'audio':          audio,
            'vision':         vision,
            'label':          torch.tensor(float(self._file['labels'][i]), dtype=torch.float32),
        }


def get_dataloaders(cfg):
    hdf5_path = Path(cfg['data_folder']) / cfg.get('hdf5_file', 'meld_dataset.h5')

    with h5py.File(hdf5_path, 'r') as f:
        N = f['audio'].shape[0]
        print(f'Total samples in HDF5: {N}')

        if 'n_train' in f.attrs:
            n_train   = int(f.attrs['n_train'])
            n_dev     = int(f.attrs['n_dev'])
            n_test    = int(f.attrs['n_test'])
            train_idx = list(range(0, n_train))
            val_idx   = list(range(n_train, n_train + n_dev))
            test_idx  = list(range(n_train + n_dev, n_train + n_dev + n_test))
            print(f'  Using fixed splits from HDF5 attrs')
        else:
            np.random.seed(cfg['seed'])
            all_indices = np.random.permutation(N).tolist()
            train_end   = int(cfg['train_ratio'] * N)
            val_end     = int((cfg['train_ratio'] + cfg['val_ratio']) * N)
            train_idx   = all_indices[:train_end]
            val_idx     = all_indices[train_end:val_end]
            test_idx    = all_indices[val_end:]

    print(f'  Train : {len(train_idx)} samples')
    print(f'  Val   : {len(val_idx)} samples')
    print(f'  Test  : {len(test_idx)} samples')


    t_train_dataset_init_start = time.time()
    train_dataset = MOSEIDataset(hdf5_path, train_idx, cfg)
    t_train_dataset_init_end = time.time()
    print(f'Train dataset initialization time: {t_train_dataset_init_end - t_train_dataset_init_start:.2f} seconds\n')
    t_val_dataset_init_start = time.time()
    val_dataset   = MOSEIDataset(hdf5_path, val_idx,   cfg)
    t_val_dataset_init_end = time.time()
    print(f'Validation dataset initialization time: {t_val_dataset_init_end - t_val_dataset_init_start:.2f} seconds\n')
    t_test_dataset_init_start = time.time()
    test_dataset  = MOSEIDataset(hdf5_path, test_idx,  cfg)
    t_test_dataset_init_end = time.time()
    print(f'Test dataset initialization time: {t_test_dataset_init_end - t_test_dataset_init_start:.2f} seconds\n')

    pin = torch.cuda.is_available()

    t_train_loader_init_start = time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size = cfg['batch_size'],
        shuffle = True,
        num_workers = 4,
        pin_memory = pin,
        persistent_workers = True,

    )
    t_train_loader_init_end = time.time()
    print(f'Train DataLoader initialization time: {t_train_loader_init_end - t_train_loader_init_start:.2f} seconds\n')
    t_val_loader_init_start = time.time()
    val_loader = DataLoader(
        val_dataset,
        batch_size = cfg['batch_size'],
        shuffle = False,
        num_workers = 4,
        pin_memory = pin,
        persistent_workers = True,

    )
    t_val_loader_init_end = time.time()
    print(f'Validation DataLoader initialization time: {t_val_loader_init_end - t_val_loader_init_start:.2f} seconds\n')
    t_test_loader_init_start = time.time()
    test_loader = DataLoader(
        test_dataset,
        batch_size = cfg['batch_size'],
        shuffle = False,
        num_workers = 4,
        pin_memory = pin,
        persistent_workers = True,

    )
    t_test_loader_init_end = time.time()
    print(f'Test DataLoader initialization time: {t_test_loader_init_end - t_test_loader_init_start:.2f} seconds\n')

    return train_loader, val_loader, test_loader