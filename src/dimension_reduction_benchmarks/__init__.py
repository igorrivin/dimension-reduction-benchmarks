"""Experimental dimension-estimation and reduction benchmarks."""

from .autoencoder import (
    AutoencoderDimensionResult,
    ReconstructionElbowResult,
    autoencoder_intrinsic_dimension,
    reconstruction_elbow,
)

__version__ = "0.1.0"

__all__ = [
    "AutoencoderDimensionResult",
    "ReconstructionElbowResult",
    "autoencoder_intrinsic_dimension",
    "reconstruction_elbow",
]
