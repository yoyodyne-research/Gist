"""Feature collection utility for latent layer training.

Collects feature vectors from various sources:
- Live Bela streaming (USB serial or network)
- Pre-recorded audio files processed through CARFAC simulation
- Existing NPZ feature logs

Data is subsampled and stored in NPZ format for training.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Union
import struct
import time


# Feature vector dimensions (must match LatentInput in carfac_frontend.h)
FEATURE_DIMS = 150
NUM_BANDS = 71
SUMMARY_DIMS = 8

# Subsampling settings
SUBSAMPLE_FACTOR = 4  # Log every Nth control-rate frame (~60 Hz effective from 230 Hz)


class FeatureCollector:
    """Collects and stores feature vectors for training."""

    def __init__(
        self,
        output_path: Union[str, Path],
        subsample_factor: int = SUBSAMPLE_FACTOR,
        max_samples: int = 500_000
    ):
        """
        Args:
            output_path: Path to write NPZ file
            subsample_factor: Keep every Nth sample
            max_samples: Maximum samples to collect
        """
        self.output_path = Path(output_path)
        self.subsample_factor = subsample_factor
        self.max_samples = max_samples

        self.features: list = []
        self.timestamps: list = []
        self.sample_count = 0
        self.kept_count = 0

    def add_sample(self, feature_vec: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Add a feature vector (with subsampling).

        Args:
            feature_vec: [FEATURE_DIMS] float32 array
            timestamp: Optional timestamp in seconds

        Returns:
            True if sample was kept
        """
        if len(self.features) >= self.max_samples:
            return False

        self.sample_count += 1

        # Subsample
        if self.sample_count % self.subsample_factor != 0:
            return False

        if len(feature_vec) != FEATURE_DIMS:
            raise ValueError(f"Expected {FEATURE_DIMS} dims, got {len(feature_vec)}")

        self.features.append(feature_vec.astype(np.float32))
        self.timestamps.append(timestamp if timestamp else time.time())
        self.kept_count += 1

        return True

    def save(self) -> None:
        """Save collected features to NPZ file."""
        if not self.features:
            print("No features to save")
            return

        X = np.stack(self.features, axis=0)
        T = np.array(self.timestamps, dtype=np.float64)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output_path,
            features=X,
            timestamps=T,
            subsample_factor=self.subsample_factor
        )

        print(f"Saved {len(X)} samples to {self.output_path}")
        print(f"  Total received: {self.sample_count}")
        print(f"  Kept (1/{self.subsample_factor}): {self.kept_count}")
        print(f"  Shape: {X.shape}")
        print(f"  Duration: {T[-1] - T[0]:.1f}s")

    def __len__(self) -> int:
        return len(self.features)


def load_features(path: Union[str, Path]) -> np.ndarray:
    """
    Load features from NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        X: [n_samples, FEATURE_DIMS] feature matrix
    """
    data = np.load(path)
    return data['features']


def load_multiple(paths: list) -> np.ndarray:
    """
    Load and concatenate features from multiple NPZ files.

    Args:
        paths: List of paths to NPZ files

    Returns:
        X: Concatenated feature matrix
    """
    arrays = [load_features(p) for p in paths]
    return np.concatenate(arrays, axis=0)


def iterate_batches(
    X: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True
) -> Iterator[np.ndarray]:
    """
    Iterate over features in batches.

    Args:
        X: [n_samples, n_features] feature matrix
        batch_size: Batch size
        shuffle: Shuffle before iterating

    Yields:
        Batches of shape [batch_size, n_features]
    """
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[indices[start:end]]


def parse_binary_packet(data: bytes) -> Optional[np.ndarray]:
    """
    Parse a binary feature packet from Bela serial output.

    Expected format:
    - 4 bytes: magic header (0xFEAT)
    - 4 bytes: uint32 sequence number
    - 150 * 4 bytes: float32 features
    - 4 bytes: checksum

    Args:
        data: Raw bytes

    Returns:
        Feature vector or None if invalid
    """
    HEADER_MAGIC = 0x54414546  # "FEAT" in little-endian
    PACKET_SIZE = 4 + 4 + FEATURE_DIMS * 4 + 4  # 612 bytes

    if len(data) != PACKET_SIZE:
        return None

    # Parse header
    magic, seq = struct.unpack('<II', data[:8])
    if magic != HEADER_MAGIC:
        return None

    # Parse features
    features = np.frombuffer(data[8:8 + FEATURE_DIMS * 4], dtype=np.float32)

    # TODO: Verify checksum if needed

    return features.copy()


def analyze_features(X: np.ndarray) -> dict:
    """
    Analyze collected feature statistics.

    Args:
        X: [n_samples, n_features] feature matrix

    Returns:
        Dictionary with statistics
    """
    return {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'mean': X.mean(axis=0).tolist(),
        'std': X.std(axis=0).tolist(),
        'min': X.min(axis=0).tolist(),
        'max': X.max(axis=0).tolist(),
        # Per-band energy (first 71 features are env_fast)
        'mean_energy': float(X[:, :NUM_BANDS].mean()),
        'total_energy_range': (
            float(X[:, 142].min()),
            float(X[:, 142].max())
        )
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature collection utility')
    parser.add_argument('command', choices=['analyze', 'info'])
    parser.add_argument('path', help='Path to NPZ file')
    args = parser.parse_args()

    if args.command == 'info':
        data = np.load(args.path)
        print(f"File: {args.path}")
        print(f"Keys: {list(data.keys())}")
        X = data['features']
        print(f"Shape: {X.shape}")
        if 'timestamps' in data:
            T = data['timestamps']
            print(f"Duration: {T[-1] - T[0]:.1f}s")
        if 'subsample_factor' in data:
            print(f"Subsample factor: {data['subsample_factor']}")

    elif args.command == 'analyze':
        X = load_features(args.path)
        stats = analyze_features(X)
        print(f"Samples: {stats['n_samples']}")
        print(f"Features: {stats['n_features']}")
        print(f"Mean energy: {stats['mean_energy']:.6f}")
        print(f"Total energy range: {stats['total_energy_range']}")
