import pytest
import numpy as np
import h5py
from pathlib import Path


class TestHDF5File:
    """Tests that your HDF5 data file is correct and clean."""

    def test_file_exists(self, cfg):
        """HDF5 file must exist at the configured path."""
        path = Path(cfg['hdf5_path'])
        assert path.exists(), \
            f"HDF5 file not found at {path}"

    def test_file_has_required_datasets(self, hdf5_sample):
        """HDF5 must contain audio, vision, labels, texts."""
        # with h5py.File(cfg['hdf5_path'], 'r') as f:
        #     required = ['audio', 'vision', 'labels', 'texts']
        #     for key in required:
        #         assert key in f, \
        #             f"Missing dataset '{key}' in HDF5 file"
        required = ['audio', 'vision', 'labels', 'texts']
        for key in required:
            assert key in hdf5_sample.keys(), \
                f"Missing dataset '{key}' in HDF5 file"


    def test_audio_shape(self, hdf5_sample, cfg):
        """Audio must be (N, 500, 74)."""
        audio = hdf5_sample['audio']
        assert audio.ndim == 3, \
            f"Audio should be 3D, got {audio.ndim}D"
        assert audio.shape[1] == cfg['seq_len'], \
            f"Audio timesteps should be {cfg['seq_len']}, got {audio.shape[1]}"
        assert audio.shape[2] == cfg['audio_dim'], \
            f"Audio features should be {cfg['audio_dim']}, got {audio.shape[2]}"

    def test_vision_shape(self, hdf5_sample, cfg):
        """Vision must be (N, 500, 713)."""
        vision = hdf5_sample['vision']
        assert vision.ndim == 3, \
            f"Vision should be 3D, got {vision.ndim}D"
        assert vision.shape[1] == cfg['seq_len'], \
            f"Vision timesteps should be {cfg['seq_len']}, got {vision.shape[1]}"
        assert vision.shape[2] == cfg['vision_dim'], \
            f"Vision features should be {cfg['vision_dim']}, got {vision.shape[2]}"

    def test_labels_range(self, hdf5_sample):
        """Labels must be within [-3, 3] — MOSEI range."""
        labels = hdf5_sample['labels']
        assert labels.min() >= -3.0, \
            f"Label below -3: {labels.min()}"
        assert labels.max() <= 3.0, \
            f"Label above +3: {labels.max()}"

    def test_no_nan_in_audio(self, hdf5_sample):
        """Audio must not contain any NaN values."""
        nan_count = np.isnan(hdf5_sample['audio']).sum()
        assert nan_count == 0, \
            f"Found {nan_count} NaN values in audio"

    def test_no_nan_in_vision(self, hdf5_sample):
        """Vision must not contain any NaN values."""
        nan_count = np.isnan(hdf5_sample['vision']).sum()
        assert nan_count == 0, \
            f"Found {nan_count} NaN values in vision"

    def test_no_nan_in_labels(self, hdf5_sample):
        """Labels must not contain any NaN values."""
        nan_count = np.isnan(hdf5_sample['labels']).sum()
        assert nan_count == 0, \
            f"Found {nan_count} NaN values in labels"

    def test_no_inf_in_audio(self, hdf5_sample):
        """Audio must not contain infinity values."""
        inf_count = np.isinf(hdf5_sample['audio']).sum()
        assert inf_count == 0, \
            f"Found {inf_count} Inf values in audio"

    def test_no_inf_in_vision(self, hdf5_sample):
        """Vision must not contain infinity values."""
        inf_count = np.isinf(hdf5_sample['vision']).sum()
        assert inf_count == 0, \
            f"Found {inf_count} Inf values in vision"

    def test_texts_not_empty(self, hdf5_sample):
        """Text entries must not be empty strings."""
        texts = hdf5_sample['texts']
        for i, text in enumerate(texts):
            if isinstance(text, bytes):
                text = text.decode()
            assert len(text.strip()) > 0, \
                f"Empty text at index {i}"

    def test_minimum_sample_count(self, hdf5_sample):
        """Dataset must have at least 1000 samples."""
        assert hdf5_sample['total'] >= 1000, \
            f"Too few samples: {hdf5_sample['total']}"