"""
Creates minimal fake data fixtures for CI testing.
Runs on GitHub Actions where real HDF5 is not available.
"""
import os
import h5py
import numpy as np
import torch
from pathlib import Path

# Create fixtures folder
Path("tests/fixtures").mkdir(exist_ok=True)

# ── Create tiny fake HDF5 ─────────────────────────────────────────
print("Creating fake HDF5...")
N = 20   # just 20 samples — enough to test shapes and logic

with h5py.File("tests/fixtures/test_data.h5", "w") as f:
    f.create_dataset("audio",  data=np.random.randn(N, 500, 74).astype(np.float32))
    f.create_dataset("vision", data=np.random.randn(N, 500, 713).astype(np.float32))
    f.create_dataset("labels", data=np.random.uniform(-3, 3, N).astype(np.float32))

    dt = h5py.special_dtype(vlen=str)
    texts = np.array([f"test sentence {i}" for i in range(N)], dtype=object)
    f.create_dataset("texts", data=texts, dtype=dt)

print(f"✅ Fake HDF5 created: {N} samples")

# ── Create fake saved model ───────────────────────────────────────
print("Creating fake model checkpoint...")
from model.model import TransformerFusionModel

model = TransformerFusionModel()
torch.save(model.state_dict(), "tests/fixtures/best_model.pt")
print("✅ Fake model checkpoint created")