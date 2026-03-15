import h5py
import numpy as np

# ── HDF5 flush helper ─────────────────────────────────────────────
def flush_to_hdf5(hdf5_path, batch_audio, batch_vision, batch_texts,
                  batch_labels, batch_ids, target_length):
    """
    Appends one batch of data to the HDF5 file.
    If the file doesn't exist yet, it creates it with resizable datasets.
    If it already exists, it appends to the existing datasets.

    This is exactly like your execute_batch → conn.commit() but for HDF5.
    """
    n = len(batch_labels)   # how many samples in this batch

    with h5py.File(hdf5_path, 'a') as f:  # 'a' = append (create if not exists)

        if 'audio' not in f:
            # ── First batch: CREATE the datasets ─────────────
            # maxshape=(None, ...) means "can grow along axis 0"
            f.create_dataset(
                'audio',
                data     = np.stack(batch_audio),          # (n, 500, 74)
                maxshape = (None, target_length, 74),
                chunks   = (1, target_length, 74),            # one chunk per sample
                compression = 'gzip'
            )
            f.create_dataset(
                'vision',
                data     = np.stack(batch_vision),         # (n, 500, 713)
                maxshape = (None, target_length, 713),
                chunks   = (1, target_length, 713),
                compression = 'gzip'
            )
            f.create_dataset(
                'labels',
                data     = np.array(batch_labels, dtype='float32'),
                maxshape = (None,)
            )
            # Text and IDs stored as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('texts', data=np.array(batch_texts, dtype=object),
                             dtype=dt, maxshape=(None,))
            f.create_dataset('ids',   data=np.array(batch_ids,   dtype=object),
                             dtype=dt, maxshape=(None,))

        else:
            # ── Subsequent batches: RESIZE then APPEND ────────
            # This is like doing INSERT INTO — extends the dataset
            current = f['audio'].shape[0]   # how many rows exist already
            new_total = current + n

            # Resize each dataset to make room
            f['audio'].resize(new_total,  axis=0)
            f['vision'].resize(new_total, axis=0)
            f['labels'].resize(new_total, axis=0)
            f['texts'].resize(new_total,  axis=0)
            f['ids'].resize(new_total,    axis=0)

            # Write new data into the new slots
            f['audio'] [current:new_total] = np.stack(batch_audio)
            f['vision'][current:new_total] = np.stack(batch_vision)
            f['labels'][current:new_total] = np.array(batch_labels, dtype='float32')
            f['texts'] [current:new_total] = np.array(batch_texts,  dtype=object)
            f['ids']   [current:new_total] = np.array(batch_ids,    dtype=object)