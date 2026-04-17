"""Pure PyTorch fallback implementations for CUDA operations.

These are used when Triton is not available. They provide the same API
but run on CPU or use standard PyTorch CUDA operations.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def batch_cosine_similarity(A: torch.Tensor, B: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute pairwise cosine similarity between embeddings.

    Args:
        A: (M, K) tensor of embeddings
        B: (N, K) tensor of embeddings (if None, B=A)

    Returns:
        (M, N) tensor of cosine similarities
    """
    if B is None:
        B = A

    # Normalize
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)

    # Cosine similarity via matrix multiplication
    return torch.mm(A_norm, B_norm.t())


def batch_coupling(
    embeddings: torch.Tensor,
    energies: torch.Tensor,
    valences: torch.Tensor,
    coupling_strength: float = 0.1
) -> torch.Tensor:
    """Compute pairwise memory coupling matrix.

    Args:
        embeddings: (N, K) tensor of memory embeddings
        energies: (N,) tensor of memory energies
        valences: (N,) tensor of memory valences
        coupling_strength: Base coupling strength

    Returns:
        (N, N) coupling matrix
    """
    # Cosine similarity
    cos_sim = batch_cosine_similarity(embeddings)

    # Energy product
    energy_product = energies.unsqueeze(1) * energies.unsqueeze(0)

    # Valence resonance (similar emotions couple more strongly)
    valence_diff = torch.abs(valences.unsqueeze(1) - valences.unsqueeze(0))
    valence_resonance = 1.0 - 0.5 * valence_diff

    # Combined coupling
    coupling = coupling_strength * cos_sim * energy_product * valence_resonance

    return coupling


def density_estimation(
    queries: torch.Tensor,
    points: torch.Tensor,
    bandwidth: float = 0.5
) -> torch.Tensor:
    """Estimate density at query points using Gaussian kernel.

    Args:
        queries: (N_q, K) query points
        points: (N_p, K) reference points
        bandwidth: Gaussian kernel bandwidth

    Returns:
        (N_q,) density at each query point
    """
    # Compute squared distances
    sq_dist = torch.cdist(queries, points, p=2).pow(2)

    # Gaussian kernel
    kernel = torch.exp(-sq_dist / (2 * bandwidth ** 2))

    # Sum and normalize
    density = kernel.mean(dim=1)

    return density


def pairwise_distances(A: torch.Tensor, B: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute pairwise L2 distances.

    Args:
        A: (M, K) tensor
        B: (N, K) tensor (if None, B=A)

    Returns:
        (M, N) distance matrix
    """
    if B is None:
        B = A

    return torch.cdist(A, B, p=2)


def void_probe_density(
    probes: torch.Tensor,
    known_points: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """Compute void density for probes (lower = more void-like).

    Args:
        probes: (N_p, K) probe points
        known_points: (N_k, K) known memory embeddings
        k: Number of nearest neighbors

    Returns:
        (N_p,) density scores (lower = more void-like)
    """
    # Compute distances
    distances = pairwise_distances(probes, known_points)

    # Get k-nearest distances
    k = min(k, distances.shape[1])
    knn_dists, _ = distances.topk(k, dim=1, largest=False)

    # Mean kNN distance as density proxy (inverted)
    density = 1.0 / (knn_dists.mean(dim=1) + 0.1)

    return density


def memory_step_fused(
    embeddings: torch.Tensor,
    energies: torch.Tensor,
    phases: torch.Tensor,
    coupling_grad: torch.Tensor,
    dt: float = 0.01,
    noise_scale: float = 0.01,
    energy_decay: float = 0.01
) -> None:
    """Fused memory evolution step (in-place).

    Args:
        embeddings: (N, K) memory embeddings (modified in-place)
        energies: (N,) memory energies (modified in-place)
        phases: (N,) narrative phases (modified in-place)
        coupling_grad: (N, K) gradient from coupling field
        dt: Time step
        noise_scale: Noise magnitude
        energy_decay: Energy decay rate
    """
    # Add noise
    noise = torch.randn_like(embeddings) * noise_scale

    # Update embeddings
    embeddings.add_(dt * (coupling_grad + noise))

    # Normalize
    embeddings.div_(embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8))

    # Update energy
    activity = coupling_grad.abs().mean(dim=1)
    energies.sub_(dt * energy_decay * (1.0 - activity))
    energies.clamp_(min=0.0)

    # Update phase
    phase_velocity = 0.1 * (1.0 + energies)
    phases.add_(dt * phase_velocity)


# ============================================================================
# Batch Operation Classes
# ============================================================================

class BatchCouplingComputer:
    """Computes coupling matrices efficiently in batches."""

    def __init__(self, coupling_strength: float = 0.1):
        self.coupling_strength = coupling_strength

    def compute(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor
    ) -> torch.Tensor:
        """Compute full coupling matrix."""
        return batch_coupling(
            embeddings, energies, valences,
            self.coupling_strength
        )

    def compute_gradient(
        self,
        embeddings: torch.Tensor,
        coupling_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of embeddings from coupling.

        grad[i] = sum_j coupling[i,j] * (emb[j] - emb[i])
        """
        N, K = embeddings.shape

        # Expand for broadcasting
        emb_i = embeddings.unsqueeze(1)  # (N, 1, K)
        emb_j = embeddings.unsqueeze(0)  # (1, N, K)

        # Direction vectors
        direction = emb_j - emb_i  # (N, N, K)

        # Weight by coupling
        weighted = coupling_matrix.unsqueeze(2) * direction  # (N, N, K)

        # Sum over j
        gradient = weighted.sum(dim=1)  # (N, K)

        return gradient


class BatchDensityEstimator:
    """Estimates density in embedding space."""

    def __init__(self, bandwidth: float = 0.5):
        self.bandwidth = bandwidth

    def estimate(self, queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Estimate density at query points."""
        return density_estimation(queries, points, self.bandwidth)

    def find_voids(
        self,
        known_points: torch.Tensor,
        n_probes: int = 100,
        top_k: int = 10
    ) -> torch.Tensor:
        """Find void regions (low density areas).

        Returns:
            (top_k, K) tensor of void centers
        """
        K = known_points.shape[1]

        # Generate random probes
        probes = torch.randn(n_probes, K, device=known_points.device)
        probes = F.normalize(probes, p=2, dim=1)

        # Compute density
        density = void_probe_density(probes, known_points)

        # Get lowest density probes (most void-like)
        _, indices = density.topk(top_k, largest=False)

        return probes[indices]


class BatchMemoryStepper:
    """Performs batch memory evolution steps."""

    def __init__(
        self,
        coupling_strength: float = 0.1,
        noise_scale: float = 0.01,
        energy_decay: float = 0.01
    ):
        self.coupling_computer = BatchCouplingComputer(coupling_strength)
        self.noise_scale = noise_scale
        self.energy_decay = energy_decay

    def step(
        self,
        embeddings: torch.Tensor,
        energies: torch.Tensor,
        valences: torch.Tensor,
        phases: torch.Tensor,
        dt: float = 0.01
    ) -> None:
        """Perform one evolution step (in-place)."""
        # Compute coupling
        coupling_matrix = self.coupling_computer.compute(
            embeddings, energies, valences
        )

        # Compute gradient
        gradient = self.coupling_computer.compute_gradient(
            embeddings, coupling_matrix
        )

        # Fused step
        memory_step_fused(
            embeddings, energies, phases, gradient,
            dt=dt,
            noise_scale=self.noise_scale,
            energy_decay=self.energy_decay
        )
