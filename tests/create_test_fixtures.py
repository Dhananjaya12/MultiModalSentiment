"""
Creates minimal fake data fixtures for CI testing.
Runs on GitHub Actions where real HDF5 is not available.
"""
import os
import h5py
import numpy as np
import torch
from pathlib import Path

Path("tests/fixtures").mkdir(exist_ok=True)

print("Creating fake HDF5...")
N = 20

with h5py.File("tests/fixtures/meld_dataset.h5", "w") as f:
    f.create_dataset("audio",  data=np.random.randn(N, 300, 768).astype(np.float32))
    f.create_dataset("vision", data=np.random.randn(N, 300, 512).astype(np.float32))
    f.create_dataset("labels", data=np.random.choice([-1., 0., 1.], N).astype(np.float32))


    dt    = h5py.string_dtype(encoding='utf-8')
    texts = np.array([f"test sentence {i}" for i in range(N)], dtype=object)
    f.create_dataset("texts", data=texts, dtype=dt)

    f.attrs['n_train']    = 16
    f.attrs['n_dev']      = 2
    f.attrs['n_test']     = 2
    f.attrs['dataset']    = 'meld'
    f.attrs['seq_len']    = 300
    f.attrs['audio_dim']  = 768
    f.attrs['vision_dim'] = 512

print(f"✅ Fake HDF5 created: {N} samples")

print("Creating fake model checkpoint...")
from model.model import TransformerFusionModel
import json

with open("config.json") as f:
    cfg = json.load(f)

model = TransformerFusionModel(cfg)
torch.save(model.state_dict(), "tests/fixtures/best_model.pt")
print("✅ Fake model checkpoint created")