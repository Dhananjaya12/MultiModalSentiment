from mmsdk import mmdatasdk
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import json
from pathlib import Path
import time
from hdfs import flush_to_hdf5

with open("config.json") as f:
    config = json.load(f)

DATA_PATH = Path(config["data_folder"])
HDF5_PATH = DATA_PATH / "mosei_dataset.h5"
TARGET_LEN = config["features_target_length"]  # every sample will be resampled to this length

recipe = {
    "labels": str(DATA_PATH / "CMU_MOSEI_Labels_filtered.csd"),
    "text":   str(DATA_PATH / "CMU_MOSEI_TimestampedWords_filtered.csd"),
    "audio":  str(DATA_PATH / "CMU_MOSEI_COVAREP_filtered.csd"),
    "vision": str(DATA_PATH / "CMU_MOSEI_VisualOpenFace2_filtered.csd"),
}

dataset = mmdatasdk.mmdataset(recipe)


# ── Helper: get overlapping interval indices ──────────────────────
def get_interval_overlap(intervals, start, end):
    idx = []
    for i, (s, e) in enumerate(intervals):
        if e >= start and s <= end:
            idx.append(i)
    return idx


# ── Helper: resample (T, D) → (n, D) using timestamps ────────────
def resample_to_n(features, intervals, n=TARGET_LEN):
    """
    Resamples a variable-length sequence to exactly n timesteps.
    
    Args:
        features:  np.array of shape (T, D)
        intervals: np.array of shape (T, 2) — [start, end] per row
        n:         target number of timesteps
    
    Returns:
        np.array of shape (n, D)
    """
    T = len(features)

    # Edge case: if we have fewer points than needed,
    # nearest-neighbor is safer than linear interpolation
    kind = 'linear' if T >= 4 else 'nearest'

    midpoints    = intervals.mean(axis=1)          # (T,) timestamps
    target_times = np.linspace(midpoints[0], midpoints[-1], n)

    interp_fn = interp1d(
        midpoints, features,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(features[0], features[-1])
    )

    return interp_fn(target_times).astype(np.float32)  # (n, D)


# # ── Main ingestion loop ───────────────────────────────────────────
# conn   = get_connection()
# cursor = conn.cursor()

# insert_query = """
# INSERT INTO mosei_samples
#     (id, text, audio_features, vision_features, sentiment_label)
# VALUES (%s, %s, %s, %s, %s)
# ON CONFLICT (id)
# DO UPDATE SET
#     text = EXCLUDED.text,
#     audio_features = EXCLUDED.audio_features,
#     vision_features = EXCLUDED.vision_features,
#     sentiment_label = EXCLUDED.sentiment_label;
# """

batch_audio  = []
batch_vision = []
batch_texts  = []
batch_labels = []
batch_ids    = []
keys  = list(dataset['labels'].data.keys())

skipped = 0
total   = 0

for vid in tqdm(keys):
    label_intervals  = dataset['labels'].data[vid]['intervals'][:]
    label_features   = dataset['labels'].data[vid]['features'][:]

    text_intervals   = dataset['text'].data[vid]['intervals'][:]
    text_features    = dataset['text'].data[vid]['features'][:]

    audio_intervals  = dataset['audio'].data[vid]['intervals'][:]
    audio_features   = dataset['audio'].data[vid]['features'][:]

    vision_intervals = dataset['vision'].data[vid]['intervals'][:]
    vision_features  = dataset['vision'].data[vid]['features'][:]

    for i, (start, end) in enumerate(label_intervals):
        sample_id = f"{vid}_{start}_{end}"

        sentiment = float(label_features[i][0])

        # ── Text (unchanged — words become a sentence) ────────
        text_idx  = get_interval_overlap(text_intervals, start, end)
        words = []
        for j in text_idx:
            word = text_features[j][0]
            if isinstance(word, bytes):
                word = word.decode()

            word = word.strip()          # remove extra spaces
            if word.lower() != "sp" and word != "":
                words.append(word)

        text_sentence = " ".join(words)
        
        # ── Audio → resample to (500, 74) ─────────────────────
        audio_idx = get_interval_overlap(audio_intervals, start, end)
        if len(audio_idx) < 2:   # need at least 2 points to interpolate
            skipped += 1
            continue

        audio_seq = resample_to_n(
            audio_features[audio_idx],    # (T_a, 74)
            audio_intervals[audio_idx],   # (T_a, 2)
            n=TARGET_LEN
        )  # → (500, 74)

        # ── Vision → resample to (500, 713) ───────────────────
        vision_idx = get_interval_overlap(vision_intervals, start, end)
        if len(vision_idx) < 2:
            skipped += 1
            continue

        vision_seq = resample_to_n(
            vision_features[vision_idx],   # (T_v, 713)
            vision_intervals[vision_idx],  # (T_v, 2)
            n=TARGET_LEN
        )  # → (500, 713)

        # ── Verify shapes before inserting ────────────────────
        assert audio_seq.shape  == (TARGET_LEN, 74),  f"Bad audio shape:  {audio_seq.shape}"
        assert vision_seq.shape == (TARGET_LEN, 713), f"Bad vision shape: {vision_seq.shape}"

        batch_audio.append(audio_seq)
        batch_vision.append(vision_seq)
        batch_texts.append(text_sentence)
        batch_labels.append(sentiment)
        batch_ids.append(sample_id)
        total += 1
        
        if len(batch_labels) >= 100:
            start_time = time.time()
            print(f'Flushing batch to HDF5... (total so far: {total})')

            flush_to_hdf5(HDF5_PATH, batch_audio, batch_vision,
                          batch_texts, batch_labels, batch_ids, TARGET_LEN)

            # Clear all lists — same as your batch.clear()
            batch_audio.clear()
            batch_vision.clear()
            batch_texts.clear()
            batch_labels.clear()
            batch_ids.clear()

            print(f"Flush took {time.time() - start_time:.4f} seconds")

# ── Final flush (same as your final execute_batch) ────────────────
if batch_labels:
    print('Flushing final batch...')
    flush_to_hdf5(HDF5_PATH, batch_audio, batch_vision,
                  batch_texts, batch_labels, batch_ids, TARGET_LEN)

print(f"\n✅ Done.")
print(f"   Total saved : {total}")
print(f"   Skipped     : {skipped}")

print(f"✅ Done. Skipped {skipped} segments (too short to resample).")