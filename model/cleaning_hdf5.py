
import h5py
from pathlib import Path
import numpy as np
import json

with open("config.json") as f:
    cfg = json.load(f)

HDF5_PATH = Path(cfg['data_folder']) / "mosei_dataset.h5"

print("Opening file...")

with h5py.File(HDF5_PATH, 'a') as f:   # 'a' = open for editing

    # ── Fix Audio ─────────────────────────────────────────────────
    print("Checking audio...")
    audio = f['audio'][:]                          # load into RAM
    nan_count = np.isnan(audio).sum()
    print(f"  Nan values found in audio: {nan_count}")

    if nan_count > 0:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        f['audio'][:] = audio                      # write back to file
        print(f"  ✅ Audio fixed and saved")

    del audio    

    video = f['vision'][:]                          # load into RAM
    nan_count = np.isnan(video).sum()
    print(f"  Nan values found in video: {nan_count}")

    if nan_count > 0:
        video = np.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
        f['vision'][:] = video                      # write back to file
        print(f"  ✅ video fixed and saved")

    del video

    # ── Verify ────────────────────────────────────────────────────
    print("\nVerifying fix...")
    print(f"  Audio  has nan: {np.isnan(f['audio'][:]).any()}")   # should be False
    print(f"  Video  has nan: {np.isnan(f['vision'][:]).any()}")   # should be False

print("\n✅ Done. File cleaned in place.")