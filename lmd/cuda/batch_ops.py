"""High-level batch operations for LMD with automatic GPU/CPU dispatch.

These classes wrap the low-level kernels and provide:
- Automatic device selection
- Memory-efficient chunking for large batches
- Caching of intermediate results
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import get_device, is_cuda_available

# Import kernels based on availability
if is_cuda_available():
    from .kernels import (
        batch_coupling,
        density_estimation,
        memory_step_fused,
        void_probe_density,
    )
else:
    from .fallback import (
        batch_coupling,
        density_estimation,
        memory_step_fused,
        void_probe_density,
    )


@dataclass
class CouplingResult:
    """Result of coupling computation."""
    matrix: torch.Tensor
    gradient: torch.Tensor
    max_coupling: float
    mean_coupling: float


class BatchCouplingComputer:
    """Computes coupling matrices efficiently in batches.

    Handles large memory sets by chunking to avoid OOM.
    """

    def __init__(
        self,
        coupling_strength: float = 0.1,
        chunk_size: int = 1024,
        device: Optional[torch.device] = None
    ):
        self.coupling_strength = coupling_strength
        self.chunk_size = chunk_size
        self.device = device or get_device()

        # Cache
        self._last_matrix: Optional[torch.Tensor] = None

    def compute(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor,
        return_gradient: bool = True
    ) -> CouplingResult:
        """Compute full coupling matrix with optional gradient.

        Args:
            embeddings: (N, K) memory embeddings
            energies: (N,) memory energies
            valences: (N,) memory valences
            return_gradient: Whether to compute gradient

        Returns:
            CouplingResult with matrix, gradient, and statistics
        """
        N = embeddings.shape[0]

        # Move to device
        embeddings = embeddings.to(self.device)
        energies = energies.to(self.device)
        valences = valences.to(self.device)

        if N <= self.chunk_size:
            # Single batch
            matrix = batch_coupling(
                embeddings, energies, valences,
                self.coupling_strength
            )
        else:
            # Chunked computation
            matrix = self._compute_chunked(embeddings, energies, valences)

        self._last_matrix = matrix

        # Compute gradient if requested
        gradient = None
        if return_gradient:
            gradient = self._compute_gradient(embeddings, matrix)

        return CouplingResult(
            matrix=matrix,
            gradient=gradient,
            max_coupling=matrix.abs().max().item(),
            mean_coupling=matrix.abs().mean().item()
        )

    def _compute_chunked(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor
    ) -> torch.Tensor:
        """Compute coupling in chunks for large N."""
        N = embeddings.shape[0]
        matrix = torch.zeros(N, N, device=self.device, dtype=torch.float32)

        for i in range(0, N, self.chunk_size):
            i_end = min(i + self.chunk_size, N)
            for j in range(0, N, self.chunk_size):
                j_end = min(j + self.chunk_size, N)

                # Compute chunk
                chunk = batch_coupling(
                    embeddings[i:i_end],
                    energies[i:i_end],
                    valences[i:i_end],
                    self.coupling_strength
                )

                # If i != j, need to also compute cross terms
                if i != j:
                    # This is a simplification - full cross-coupling needs more work
                    pass

                matrix[i:i_end, j:j_end] = chunk[:, :j_end-j]

        return matrix

    def _compute_gradient(
        self,
        embeddings: torch.Tensor,
        coupling_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of embeddings from coupling."""
        N, K = embeddings.shape

        # grad[i] = sum_j coupling[i,j] * (emb[j] - emb[i])
        emb_i = embeddings.unsqueeze(1)
        emb_j = embeddings.unsqueeze(0)
        direction = emb_j - emb_i
        weighted = coupling_matrix.unsqueeze(2) * direction
        gradient = weighted.sum(dim=1)

        return gradient


class BatchDensityEstimator:
    """Estimates density in embedding space with GPU acceleration."""

    def __init__(
        self,
        bandwidth: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.bandwidth = bandwidth
        self.device = device or get_device()

    def estimate(
        self,
        queries: torch.Tensor,
        points: torch.Tensor
    ) -> torch.Tensor:
        """Estimate density at query points.

        Args:
            queries: (N_q, K) query points
            points: (N_p, K) reference points

        Returns:
            (N_q,) density estimates
        """
        queries = queries.to(self.device)
        points = points.to(self.device)

        return density_estimation(queries, points, self.bandwidth)

    def find_voids(
        self,
        known_points: torch.Tensor,
        n_probes: int = 100,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find void regions (low density areas).

        Args:
            known_points: (N, K) known embeddings
            n_probes: Number of random probes
            top_k: Number of voids to return

        Returns:
            (void_centers, void_densities) - top_k void locations and their densities
        """
        K = known_points.shape[1]
        known_points = known_points.to(self.device)

        # Generate random probes on unit sphere
        probes = torch.randn(n_probes, K, device=self.device)
        probes = F.normalize(probes, p=2, dim=1)

        # Compute void density (low = void-like)
        density = void_probe_density(probes, known_points)

        # Get lowest density probes
        values, indices = density.topk(top_k, largest=False)

        return probes[indices], values

    def find_frontiers(
        self,
        known_points: torch.Tensor,
        n_directions: int = 20,
        margin: float = 0.3
    ) -> torch.Tensor:
        """Find frontier regions at edges of known space.

        Args:
            known_points: (N, K) known embeddings
            n_directions: Number of directions to explore
            margin: How far beyond known space to project

        Returns:
            (n_directions, K) frontier points
        """
        known_points = known_points.to(self.device)
        K = known_points.shape[1]

        centroid = known_points.mean(dim=0)

        # Random directions
        directions = torch.randn(n_directions, K, device=self.device)
        directions = F.normalize(directions, p=2, dim=1)

        frontiers = []
        for direction in directions:
            # Project all points onto this direction
            projections = (known_points - centroid) @ direction
            max_proj = projections.max().item()

            # Frontier is beyond the furthest known point
            frontier = centroid + direction * (max_proj * (1 + margin))
            frontiers.append(frontier)

        return torch.stack(frontiers)


class BatchMemoryStepper:
    """Performs batch memory evolution steps with GPU acceleration."""

    def __init__(
        self,
        coupling_strength: float = 0.1,
        noise_scale: float = 0.01,
        energy_decay: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.coupling_computer = BatchCouplingComputer(
            coupling_strength=coupling_strength,
            device=device
        )
        self.noise_scale = noise_scale
        self.energy_decay = energy_decay
        self.device = device or get_device()

    def step(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor,
        phases: torch.Tensor,
        dt: float = 0.01
    ) -> CouplingResult:
        """Perform one evolution step (in-place).

        Args:
            embeddings: (N, K) memory embeddings (modified in-place)
            energies: (N,) memory energies (modified in-place)
            valences: (N,) memory valences (read-only)
            phases: (N,) narrative phases (modified in-place)
            dt: Time step

        Returns:
            CouplingResult with coupling statistics
        """
        # Move to device (if not already)
        embeddings = embeddings.to(self.device)
        energies = energies.to(self.device)
        valences = valences.to(self.device)
        phases = phases.to(self.device)

        # Compute coupling and gradient
        result = self.coupling_computer.compute(
            embeddings, energies, valences,
            return_gradient=True
        )

        # Apply fused step
        memory_step_fused(
            embeddings, energies, phases, result.gradient,
            dt=dt,
            noise_scale=self.noise_scale,
            energy_decay=self.energy_decay
        )

        return result

    def step_batch(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor,
        phases: torch.Tensor,
        n_steps: int,
        dt: float = 0.01
    ) -> List[CouplingResult]:
        """Perform multiple evolution steps.

        Args:
            embeddings: (N, K) memory embeddings (modified in-place)
            energies: (N,) memory energies (modified in-place)
            valences: (N,) memory valences (read-only)
            phases: (N,) narrative phases (modified in-place)
            n_steps: Number of steps
            dt: Time step per step

        Returns:
            List of CouplingResult for each step
        """
        results = []
        for _ in range(n_steps):
            result = self.step(embeddings, energies, valences, phases, dt)
            results.append(result)

        return results
