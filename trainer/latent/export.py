"""Export trained PCA + GMM models to JSON for C++ runtime."""

import json
import numpy as np
from pathlib import Path
from typing import Union, Optional

from .pca import LatentPCA
from .gmm import LatentGMM


def export_model(
    pca: LatentPCA,
    gmm: LatentGMM,
    output_path: Union[str, Path],
    smoothing_ms: float = 30.0,
    metadata: Optional[dict] = None
) -> None:
    """
    Export trained PCA + GMM models to a single JSON file.

    Args:
        pca: Fitted LatentPCA instance
        gmm: Fitted LatentGMM instance
        output_path: Path to write JSON file
        smoothing_ms: Responsibility smoothing time constant (ms)
        metadata: Optional metadata to include
    """
    pca_params = pca.get_params()
    gmm_params = gmm.get_params()

    model = {
        'pca': {
            'mean': pca_params['mean'],
            'std': pca_params['std'],
            'components': pca_params['components'],
            'rotation_matrix': pca_params['rotation_matrix'],
            'explained_variance_ratio': pca_params['explained_variance_ratio']
        },
        'gmm': {
            'n_components': gmm_params['n_components'],
            'means': gmm_params['means'],
            'covariances': gmm_params['covariances'],
            'weights': gmm_params['weights']
        },
        'config': {
            'input_dims': len(pca_params['mean']),
            'pca_dims': pca_params['n_components'],
            'gmm_components': gmm_params['n_components'],
            'smoothing_ms': smoothing_ms
        }
    }

    if metadata:
        model['metadata'] = metadata

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"Exported model to: {output_path}")
    print(f"  Input dims: {model['config']['input_dims']}")
    print(f"  PCA dims: {model['config']['pca_dims']}")
    print(f"  GMM components: {model['config']['gmm_components']}")


def load_model(path: Union[str, Path]) -> dict:
    """
    Load a model JSON file.

    Args:
        path: Path to model JSON

    Returns:
        Model dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def validate_model(model: dict) -> bool:
    """
    Validate model structure and dimensions.

    Args:
        model: Loaded model dictionary

    Returns:
        True if valid
    """
    try:
        pca = model['pca']
        gmm = model['gmm']
        config = model['config']

        input_dims = config['input_dims']
        pca_dims = config['pca_dims']
        gmm_components = config['gmm_components']

        # Check PCA dimensions
        assert len(pca['mean']) == input_dims
        assert len(pca['std']) == input_dims
        assert len(pca['components']) == pca_dims
        assert len(pca['components'][0]) == input_dims

        # Check GMM dimensions
        assert len(gmm['means']) == gmm_components
        assert len(gmm['means'][0]) == pca_dims
        assert len(gmm['covariances']) == gmm_components
        assert len(gmm['covariances'][0]) == pca_dims
        assert len(gmm['weights']) == gmm_components

        # Check weights sum to ~1
        weight_sum = sum(gmm['weights'])
        assert abs(weight_sum - 1.0) < 1e-5

        return True

    except (KeyError, AssertionError, IndexError) as e:
        print(f"Validation failed: {e}")
        return False


def export_reference_outputs(
    pca: LatentPCA,
    gmm: LatentGMM,
    X: np.ndarray,
    output_path: Union[str, Path],
    n_samples: int = 100
) -> None:
    """
    Export reference inputs/outputs for C++ validation.

    Saves a subset of inputs with their expected outputs for unit testing
    the C++ implementation against the Python reference.

    Args:
        pca: Fitted LatentPCA instance
        gmm: Fitted LatentGMM instance
        X: [n_samples, n_features] input features
        output_path: Path to write NPZ file
        n_samples: Number of samples to export
    """
    # Select random samples
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_test = X[indices].astype(np.float32)

    # Compute reference outputs
    Z_test = pca.transform(X_test)
    R_test = gmm.predict_proba(Z_test)
    log_prob_test = gmm.score_samples(Z_test)

    # Compute derived signals
    mode_id = np.argmax(R_test, axis=1)
    mode_strength = np.max(R_test, axis=1)

    # Entropy: -sum(r * log(r + eps))
    eps = 1e-10
    entropy = -np.sum(R_test * np.log(R_test + eps), axis=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        X=X_test,
        Z=Z_test,
        R=R_test,
        log_prob=log_prob_test,
        mode_id=mode_id,
        mode_strength=mode_strength,
        entropy=entropy
    )

    print(f"Exported {len(X_test)} reference samples to: {output_path}")


if __name__ == '__main__':
    # Test with dummy data
    from .pca import train_pca
    from .gmm import train_gmm

    np.random.seed(42)
    X = np.random.randn(1000, 150).astype(np.float32)

    pca = train_pca(X, n_components=8)
    Z = pca.transform(X)
    gmm = train_gmm(Z, n_components=14)

    # Export model
    export_model(pca, gmm, '/tmp/test_latent_model.json')

    # Validate
    model = load_model('/tmp/test_latent_model.json')
    print(f"Model valid: {validate_model(model)}")

    # Export reference outputs
    export_reference_outputs(pca, gmm, X, '/tmp/test_reference.npz')
