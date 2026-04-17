"""CUDA-accelerated operations for Living Memory Dynamics.

This module provides GPU-optimized implementations of key LMD operations
using Triton for maximum performance.

Requirements:
    pip install triton>=2.0.0

Usage:
    from lmd.cuda import (
        batch_cosine_similarity,
        batch_coupling,
        density_estimation,
        memory_step_kernel,
    )
"""

import torch

# Check for Triton availability
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return torch.cuda.is_available() and TRITON_AVAILABLE


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and is_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Import accelerated functions if available
if TRITON_AVAILABLE:
    from .batch_ops import (
        BatchCouplingComputer,
        BatchDensityEstimator,
        BatchMemoryStepper,
    )
    from .kernels import (
        batch_cosine_similarity,
        batch_coupling,
        density_estimation,
        memory_step_fused,
        pairwise_distances,
        void_probe_density,
    )
else:
    # Fallback to pure PyTorch implementations
    from .fallback import (
        BatchCouplingComputer,
        BatchDensityEstimator,
        BatchMemoryStepper,
        batch_cosine_similarity,
        batch_coupling,
        density_estimation,
        memory_step_fused,
        pairwise_distances,
        void_probe_density,
    )


__all__ = [
    "is_cuda_available",
    "get_device",
    "TRITON_AVAILABLE",
    "batch_cosine_similarity",
    "batch_coupling",
    "density_estimation",
    "pairwise_distances",
    "void_probe_density",
    "memory_step_fused",
    "BatchCouplingComputer",
    "BatchDensityEstimator",
    "BatchMemoryStepper",
]
