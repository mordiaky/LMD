"""Advanced Divergence Operators for Creative Leaps.

These operators enable human-like creative jumps: wild analogies, inventions,
and unexpected combinations - all internally without LLM decoding.

The key insight: Treat embedding space as a rich manifold to navigate creatively.
Add operators that mimic human associative jumps.

Invented by Joshua R. Thomas, January 2026.
"""

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .safeguards import EPS, safe_normalize


class LeapType(Enum):
    """Types of creative leaps."""
    ANALOGICAL = auto()      # Transfer patterns between distant domains
    DIFFUSION = auto()       # Walk through manifold via noise/denoise
    ORTHOGONAL = auto()      # Perpendicular concept merges
    EXTRAPOLATION = auto()   # Ray-trace into voids
    GRAFT = auto()           # Component substitution


@dataclass
class CreativeLeap:
    """Result of a creative leap operation."""
    embedding: torch.Tensor
    leap_type: LeapType
    source_embeddings: List[torch.Tensor]
    leap_distance: float  # How "far" the leap was
    novelty_score: float  # Estimated novelty
    components: Optional[Dict[str, torch.Tensor]] = None  # For hierarchical
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate embedding
        if torch.isnan(self.embedding).any() or torch.isinf(self.embedding).any():
            self.embedding = safe_normalize(torch.randn_like(self.embedding))

    @property
    def novelty(self) -> float:
        """Alias for novelty_score — matches the public API shown in README examples."""
        return self.novelty_score

    @property
    def coherence(self) -> float:
        """Coherence derived as the complement of raw novelty, clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.novelty_score))


class LeapOperator(ABC):
    """Base class for creative leap operators."""

    @abstractmethod
    def leap(
        self,
        sources: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> CreativeLeap:
        """Perform a creative leap from source embeddings."""
        pass

    @property
    @abstractmethod
    def leap_type(self) -> LeapType:
        """Return the type of leap this operator produces."""
        pass


class AnalogicalTransfer(LeapOperator):
    """Transfer patterns between distant domains.

    Algorithm:
    1. Retrieve two distant memory clusters (low similarity)
    2. Force structural blend:
       new_idea = A + (B - centroid_B) + projection onto A's principal directions

    This "transplants" patterns from unrelated domains -> emergent analogies.
    Example: "dragon fire" pattern applied to "underwater glass" -> refracted light "fire"
    """

    def __init__(
        self,
        content_dim: int = 32,
        min_cluster_distance: float = 0.5,
        blend_strength: float = 0.7,
        projection_weight: float = 0.3
    ):
        self.content_dim = content_dim
        self.min_cluster_distance = min_cluster_distance
        self.blend_strength = blend_strength
        self.projection_weight = projection_weight

    @property
    def leap_type(self) -> LeapType:
        return LeapType.ANALOGICAL

    def find_distant_clusters(
        self,
        embeddings: List[torch.Tensor],
        n_clusters: int = 2
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Find two maximally distant clusters using simple k-means-like approach."""
        if len(embeddings) < 4:
            # Not enough for clustering, split arbitrarily
            mid = len(embeddings) // 2
            return embeddings[:mid], embeddings[mid:]

        stacked = torch.stack(embeddings)

        # Start with two most distant points as centroids
        dists = torch.cdist(stacked, stacked)
        max_idx = dists.argmax()
        i, j = max_idx // len(embeddings), max_idx % len(embeddings)

        centroid_a = stacked[i]
        centroid_b = stacked[j]

        # Assign points to nearest centroid
        cluster_a = []
        cluster_b = []

        for emb in embeddings:
            dist_a = (emb - centroid_a).norm()
            dist_b = (emb - centroid_b).norm()
            if dist_a < dist_b:
                cluster_a.append(emb)
            else:
                cluster_b.append(emb)

        # Ensure non-empty
        if not cluster_a:
            cluster_a = [embeddings[0]]
        if not cluster_b:
            cluster_b = [embeddings[-1]]

        return cluster_a, cluster_b

    def compute_principal_directions(
        self,
        cluster: List[torch.Tensor],
        n_directions: int = 3
    ) -> torch.Tensor:
        """Compute principal directions of a cluster via SVD."""
        if len(cluster) < 2:
            # Not enough for PCA, return random orthogonal basis
            basis = torch.eye(min(n_directions, self.content_dim))
            if self.content_dim > n_directions:
                basis = F.pad(basis, (0, self.content_dim - n_directions))
            return basis

        stacked = torch.stack(cluster)
        centered = stacked - stacked.mean(dim=0, keepdim=True)

        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            # Return top principal directions
            return Vh[:min(n_directions, Vh.shape[0])]
        except RuntimeError:
            # SVD failed, return identity
            return torch.eye(min(n_directions, self.content_dim), self.content_dim)

    def project_onto_directions(
        self,
        vector: torch.Tensor,
        directions: torch.Tensor
    ) -> torch.Tensor:
        """Project vector onto principal directions and reconstruct."""
        # directions: [n_dirs, content_dim]
        # vector: [content_dim]
        coeffs = directions @ vector  # [n_dirs]
        projection = coeffs @ directions  # [content_dim]
        return projection

    def leap(
        self,
        sources: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> CreativeLeap:
        """Perform analogical transfer between distant clusters."""
        if len(sources) < 2:
            # Need at least 2 sources
            base = sources[0] if sources else torch.randn(self.content_dim)
            return CreativeLeap(
                embedding=safe_normalize(base + 0.1 * torch.randn_like(base)),
                leap_type=self.leap_type,
                source_embeddings=sources,
                leap_distance=0.1,
                novelty_score=0.1
            )

        # Find distant clusters
        cluster_a, cluster_b = self.find_distant_clusters(sources)

        # Compute centroids
        centroid_a = torch.stack(cluster_a).mean(dim=0)
        centroid_b = torch.stack(cluster_b).mean(dim=0)

        # Check distance
        cluster_distance = (centroid_a - centroid_b).norm().item()

        # Get principal directions of cluster A (the "target" domain)
        principal_dirs = self.compute_principal_directions(cluster_a)

        # Select a random point from each cluster
        idx_a = torch.randint(len(cluster_a), (1,)).item()
        idx_b = torch.randint(len(cluster_b), (1,)).item()
        A = cluster_a[idx_a]
        B = cluster_b[idx_b]

        # Analogical transfer formula:
        # new_idea = A + blend_strength * (B - centroid_B)
        #            + projection_weight * project(B - centroid_B, principal_dirs_A)

        pattern_from_B = B - centroid_b
        pattern_from_B = safe_normalize(pattern_from_B) * pattern_from_B.norm().clamp(max=2.0)

        # Project pattern onto A's principal directions (domain adaptation)
        adapted_pattern = self.project_onto_directions(pattern_from_B, principal_dirs)

        # Combine
        new_idea = (
            A
            + self.blend_strength * intensity * pattern_from_B
            + self.projection_weight * intensity * adapted_pattern
        )

        # Add small noise for variation
        new_idea = new_idea + 0.05 * torch.randn_like(new_idea)
        new_idea = safe_normalize(new_idea)

        # Compute leap distance (how different from both sources)
        dist_from_a = (new_idea - A).norm().item()
        dist_from_b = (new_idea - B).norm().item()
        leap_distance = min(dist_from_a, dist_from_b)

        # Novelty: distance from both cluster centroids
        dist_from_centroid_a = (new_idea - centroid_a).norm().item()
        dist_from_centroid_b = (new_idea - centroid_b).norm().item()
        novelty_score = min(dist_from_centroid_a, dist_from_centroid_b) / (cluster_distance + EPS)
        novelty_score = min(1.0, novelty_score)

        return CreativeLeap(
            embedding=new_idea,
            leap_type=self.leap_type,
            source_embeddings=[A, B],
            leap_distance=leap_distance,
            novelty_score=novelty_score,
            metadata={
                "cluster_distance": cluster_distance,
                "blend_strength": self.blend_strength * intensity,
                "projection_weight": self.projection_weight * intensity
            }
        )


class ManifoldWalker(LeapOperator):
    """Walk through embedding manifold via diffusion steps.

    Algorithm (simplified denoising diffusion):
    1. Start from current embedding W
    2. Add graduated noise over T steps (forward diffusion)
    3. "Denoise" by stepping toward high-density rewarding regions

    Creates paths through unexplored interpolations.
    """

    def __init__(
        self,
        content_dim: int = 32,
        n_diffusion_steps: int = 10,
        noise_schedule: str = "linear",  # "linear", "cosine", "exponential"
        density_weight: float = 0.5,
        temperature: float = 1.0,
        n_steps: Optional[int] = None,  # alias for n_diffusion_steps
    ):
        self.content_dim = content_dim
        self.n_diffusion_steps = n_steps if n_steps is not None else n_diffusion_steps
        self.noise_schedule = noise_schedule
        self.density_weight = density_weight
        self.temperature = temperature

        # Precompute noise schedule
        self._betas = self._compute_schedule()

    @property
    def n_steps(self) -> int:
        """Alias for n_diffusion_steps."""
        return self.n_diffusion_steps

    @property
    def leap_type(self) -> LeapType:
        return LeapType.DIFFUSION

    def _compute_schedule(self) -> torch.Tensor:
        """Compute noise schedule (betas)."""
        t = torch.linspace(0, 1, self.n_diffusion_steps)

        if self.noise_schedule == "linear":
            # Linear: beta increases linearly
            betas = 0.0001 + 0.02 * t
        elif self.noise_schedule == "cosine":
            # Cosine: smoother schedule
            alphas = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.cat([torch.tensor([0.0001]), betas.clamp(max=0.999)])
        else:  # exponential
            betas = 0.0001 * torch.exp(5 * t)
            betas = betas.clamp(max=0.999)

        return betas

    def estimate_density(
        self,
        point: torch.Tensor,
        reference_points: List[torch.Tensor],
        bandwidth: float = 0.5
    ) -> float:
        """Estimate local density using Gaussian kernel."""
        if not reference_points:
            return 0.0

        stacked = torch.stack(reference_points)
        dists = (stacked - point.unsqueeze(0)).norm(dim=1)

        # Gaussian kernel density estimate
        kernel_values = torch.exp(-0.5 * (dists / bandwidth) ** 2)
        density = kernel_values.mean().item()

        return density

    def compute_density_gradient(
        self,
        point: torch.Tensor,
        reference_points: List[torch.Tensor],
        bandwidth: float = 0.5
    ) -> torch.Tensor:
        """Compute gradient of density (points toward higher density)."""
        if not reference_points:
            return torch.zeros_like(point)

        stacked = torch.stack(reference_points)
        diffs = stacked - point.unsqueeze(0)  # [N, dim]
        dists = diffs.norm(dim=1, keepdim=True)  # [N, 1]

        # Gaussian kernel weights
        weights = torch.exp(-0.5 * (dists / bandwidth) ** 2)  # [N, 1]

        # Gradient direction (weighted sum of directions to points)
        gradient = (weights * diffs / (dists + EPS)).sum(dim=0)

        return safe_normalize(gradient)

    def leap(
        self,
        sources: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> CreativeLeap:
        """Perform manifold walk via diffusion."""
        if not sources:
            return CreativeLeap(
                embedding=safe_normalize(torch.randn(self.content_dim)),
                leap_type=self.leap_type,
                source_embeddings=[],
                leap_distance=1.0,
                novelty_score=1.0
            )

        # Start from a random source
        start_idx = torch.randint(len(sources), (1,)).item()
        current = sources[start_idx].clone()
        original = current.clone()

        path = [current.clone()]

        # Forward diffusion: gradually add noise
        for t in range(self.n_diffusion_steps // 2):
            beta = self._betas[t] * intensity * self.temperature
            noise = torch.randn_like(current)
            current = math.sqrt(1 - beta) * current + math.sqrt(beta) * noise
            path.append(current.clone())

        # Reverse diffusion: denoise toward high-density regions
        for t in range(self.n_diffusion_steps // 2, self.n_diffusion_steps):
            # Compute density gradient (toward high-density)
            density_grad = self.compute_density_gradient(current, sources)

            # Denoise step: move toward density + add smaller noise
            beta = self._betas[self.n_diffusion_steps - 1 - t] * intensity
            denoise_direction = self.density_weight * density_grad
            noise = torch.randn_like(current) * math.sqrt(beta) * 0.5

            current = current + denoise_direction * 0.1 + noise
            current = safe_normalize(current)
            path.append(current.clone())

        # Final position
        final = safe_normalize(current)

        # Compute leap metrics
        leap_distance = (final - original).norm().item()

        # Novelty: minimum distance to any source
        min_dist = min((final - s).norm().item() for s in sources)
        novelty_score = min(1.0, min_dist / 2.0)

        return CreativeLeap(
            embedding=final,
            leap_type=self.leap_type,
            source_embeddings=[original],
            leap_distance=leap_distance,
            novelty_score=novelty_score,
            metadata={
                "n_steps": self.n_diffusion_steps,
                "path_length": len(path),
                "temperature": self.temperature * intensity
            }
        )


class OrthogonalComposer(LeapOperator):
    """Create novel combinations via orthogonal decomposition.

    Algorithm (Gram-Schmidt based):
    1. Decompose memories into orthogonal basis vectors
    2. Recombine novel orthogonal combos
    3. Forces perpendicular (conceptually unrelated) merges

    Example: "underwater" + "glass" + "fire" -> "superheated steam propulsion"
    """

    def __init__(
        self,
        content_dim: int = 32,
        n_components: int = 4,
        orthogonal_weight: float = 0.6,
        residual_weight: float = 0.3
    ):
        self.content_dim = content_dim
        self.n_components = n_components
        self.orthogonal_weight = orthogonal_weight
        self.residual_weight = residual_weight

    @property
    def leap_type(self) -> LeapType:
        return LeapType.ORTHOGONAL

    def gram_schmidt(self, vectors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gram-Schmidt orthogonalization.

        Returns:
            orthogonal_basis: [n_vectors, dim] orthonormal vectors
            coefficients: [n_vectors, n_vectors] projection coefficients
        """
        n = len(vectors)
        if n == 0:
            return torch.zeros(0, self.content_dim), torch.zeros(0, 0)

        stacked = torch.stack(vectors)  # [n, dim]

        # Gram-Schmidt
        orthogonal = []
        coefficients = torch.zeros(n, n)

        for i in range(n):
            v = stacked[i].clone()

            # Subtract projections onto previous orthogonal vectors
            for j, u in enumerate(orthogonal):
                coeff = (v @ u) / (u @ u + EPS)
                coefficients[i, j] = coeff
                v = v - coeff * u

            # Normalize
            norm = v.norm()
            if norm > EPS:
                orthogonal.append(v / norm)
                coefficients[i, i] = norm
            else:
                # Vector was linearly dependent, add noise
                orthogonal.append(safe_normalize(torch.randn(self.content_dim)))
                coefficients[i, i] = 0.1

        return torch.stack(orthogonal), coefficients

    def leap(
        self,
        sources: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> CreativeLeap:
        """Create orthogonal composition."""
        if len(sources) < 2:
            base = sources[0] if sources else torch.randn(self.content_dim)
            # Add orthogonal noise
            noise = torch.randn_like(base)
            orthogonal_noise = noise - (noise @ base / (base @ base + EPS)) * base
            result = safe_normalize(base + 0.5 * intensity * orthogonal_noise)

            return CreativeLeap(
                embedding=result,
                leap_type=self.leap_type,
                source_embeddings=sources,
                leap_distance=0.5 * intensity,
                novelty_score=0.3
            )

        # Select subset of sources for decomposition
        n_select = min(self.n_components, len(sources))
        indices = torch.randperm(len(sources))[:n_select]
        selected = [sources[i] for i in indices]

        # Orthogonalize
        orthogonal_basis, coefficients = self.gram_schmidt(selected)

        # Generate novel combination
        # Strategy: shuffle coefficients and recombine orthogonal vectors
        n_basis = orthogonal_basis.shape[0]

        # Create novel coefficient combination
        # Mix original coefficients with random weights
        original_coeffs = coefficients.diagonal()
        random_coeffs = torch.randn(n_basis) * self.orthogonal_weight * intensity

        # Shuffle: assign coefficients to different basis vectors
        perm = torch.randperm(n_basis)
        shuffled_coeffs = original_coeffs[perm]

        # Final coefficients: blend of shuffled original + random
        final_coeffs = (1 - intensity) * original_coeffs + intensity * (
            0.5 * shuffled_coeffs + 0.5 * random_coeffs
        )

        # Reconstruct from orthogonal basis
        novel_combination = (final_coeffs.unsqueeze(1) * orthogonal_basis).sum(dim=0)

        # Add residual from sources (maintains some coherence)
        if self.residual_weight > 0:
            source_mean = torch.stack(selected).mean(dim=0)
            novel_combination = (
                (1 - self.residual_weight) * novel_combination
                + self.residual_weight * source_mean
            )

        result = safe_normalize(novel_combination)

        # Compute metrics
        leap_distance = min((result - s).norm().item() for s in selected)

        # Orthogonality score: how orthogonal is result to sources?
        orthogonality = sum(
            1 - abs(F.cosine_similarity(result.unsqueeze(0), s.unsqueeze(0)).item())
            for s in selected
        ) / len(selected)

        novelty_score = orthogonality * 0.5 + leap_distance * 0.5
        novelty_score = min(1.0, novelty_score)

        return CreativeLeap(
            embedding=result,
            leap_type=self.leap_type,
            source_embeddings=selected,
            leap_distance=leap_distance,
            novelty_score=novelty_score,
            metadata={
                "n_basis_vectors": n_basis,
                "orthogonality": orthogonality,
                "final_coefficients": final_coeffs.tolist()
            }
        )


class VoidExtrapolator(LeapOperator):
    """Ray-trace from dense clusters into unexplored voids.

    Algorithm:
    1. Identify "voids" (low-density regions) via RepulsionField gaps
    2. Ray-trace from dense clusters toward voids
    3. Linear extension + controlled noise

    This turns curiosity into targeted frontier exploration.
    """

    def __init__(
        self,
        content_dim: int = 32,
        n_void_probes: int = 50,
        extrapolation_strength: float = 1.5,
        noise_scale: float = 0.2,
        n_probes: Optional[int] = None,  # alias for n_void_probes
    ):
        self.content_dim = content_dim
        # Accept either `n_probes` or `n_void_probes`; `n_probes` wins when provided.
        self.n_void_probes = n_probes if n_probes is not None else n_void_probes
        self.extrapolation_strength = extrapolation_strength
        self.noise_scale = noise_scale

    @property
    def n_probes(self) -> int:
        """Alias for n_void_probes."""
        return self.n_void_probes

    @property
    def leap_type(self) -> LeapType:
        return LeapType.EXTRAPOLATION

    def find_voids(
        self,
        sources: List[torch.Tensor],
        n_probes: int = 50
    ) -> List[torch.Tensor]:
        """Find low-density regions (voids) in embedding space."""
        if not sources:
            return [safe_normalize(torch.randn(self.content_dim))]

        stacked = torch.stack(sources)

        # Generate random probes
        probes = torch.randn(n_probes, self.content_dim)
        probes = F.normalize(probes, dim=1)

        # Scale probes to be in similar range as sources
        source_norms = stacked.norm(dim=1)
        mean_norm = source_norms.mean()
        probes = probes * mean_norm

        # Compute distance to nearest source for each probe
        dists = torch.cdist(probes, stacked)
        min_dists = dists.min(dim=1).values

        # Return probes that are far from all sources (voids)
        k = min(5, n_probes)
        _, void_indices = min_dists.topk(k)

        return [probes[i] for i in void_indices]

    def leap(
        self,
        sources: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> CreativeLeap:
        """Extrapolate toward a void region."""
        if not sources:
            return CreativeLeap(
                embedding=safe_normalize(torch.randn(self.content_dim)),
                leap_type=self.leap_type,
                source_embeddings=[],
                leap_distance=1.0,
                novelty_score=1.0
            )

        # Find void regions
        voids = self.find_voids(sources, self.n_void_probes)

        if not voids:
            # Fallback: random direction
            void_target = safe_normalize(torch.randn(self.content_dim))
        else:
            # Select a random void
            void_idx = torch.randint(len(voids), (1,)).item()
            void_target = voids[void_idx]

        # Find the densest cluster (centroid of sources)
        stacked = torch.stack(sources)
        centroid = stacked.mean(dim=0)

        # Direction from centroid to void
        direction = void_target - centroid
        direction = safe_normalize(direction)

        # Extrapolate: go past the void
        extrapolation_distance = (void_target - centroid).norm()

        new_point = (
            centroid
            + direction * extrapolation_distance * self.extrapolation_strength * intensity
        )

        # Add controlled noise perpendicular to direction
        noise = torch.randn_like(new_point)
        perpendicular_noise = noise - (noise @ direction) * direction
        perpendicular_noise = safe_normalize(perpendicular_noise) * self.noise_scale * intensity

        new_point = new_point + perpendicular_noise
        result = safe_normalize(new_point)

        # Compute metrics
        leap_distance = (result - centroid).norm().item()

        # Novelty: distance from all sources
        min_dist = min((result - s).norm().item() for s in sources)
        novelty_score = min(1.0, min_dist / 2.0)

        return CreativeLeap(
            embedding=result,
            leap_type=self.leap_type,
            source_embeddings=[centroid],
            leap_distance=leap_distance,
            novelty_score=novelty_score,
            metadata={
                "void_distance": extrapolation_distance.item(),
                "extrapolation_strength": self.extrapolation_strength * intensity,
                "direction": direction.tolist()[:5]  # First 5 dims for debugging
            }
        )


@dataclass
class CreativeLeapConfig:
    """Configuration for creative leap system."""
    content_dim: int = 32

    # Operator weights (probability of selecting each)
    analogical_weight: float = 0.3
    diffusion_weight: float = 0.25
    orthogonal_weight: float = 0.25
    extrapolation_weight: float = 0.2

    # Operator-specific settings
    min_cluster_distance: float = 0.5
    n_diffusion_steps: int = 10
    n_orthogonal_components: int = 4
    n_void_probes: int = 50

    # Global settings
    default_intensity: float = 1.0
    temperature: float = 1.0

    @classmethod
    def conservative(cls) -> "CreativeLeapConfig":
        """Conservative config: smaller leaps, more coherent."""
        return cls(
            analogical_weight=0.2,
            diffusion_weight=0.4,
            orthogonal_weight=0.2,
            extrapolation_weight=0.2,
            default_intensity=0.5,
            temperature=0.5
        )

    @classmethod
    def wild(cls) -> "CreativeLeapConfig":
        """Wild config: larger leaps, more novel."""
        return cls(
            analogical_weight=0.35,
            diffusion_weight=0.2,
            orthogonal_weight=0.25,
            extrapolation_weight=0.2,
            default_intensity=1.5,
            temperature=1.5
        )


class CreativeLeapEngine:
    """Engine that orchestrates creative leap operators.

    Features:
    - Multiple operator types with configurable weights
    - Valence-driven operator selection (dopamine modulation)
    - Tracks leap history for learning
    - Thread-safe operation
    """

    def __init__(self, config: Optional[CreativeLeapConfig] = None):
        self.config = config or CreativeLeapConfig()

        # Initialize operators
        self.operators: Dict[LeapType, LeapOperator] = {
            LeapType.ANALOGICAL: AnalogicalTransfer(
                content_dim=self.config.content_dim,
                min_cluster_distance=self.config.min_cluster_distance
            ),
            LeapType.DIFFUSION: ManifoldWalker(
                content_dim=self.config.content_dim,
                n_diffusion_steps=self.config.n_diffusion_steps,
                temperature=self.config.temperature
            ),
            LeapType.ORTHOGONAL: OrthogonalComposer(
                content_dim=self.config.content_dim,
                n_components=self.config.n_orthogonal_components
            ),
            LeapType.EXTRAPOLATION: VoidExtrapolator(
                content_dim=self.config.content_dim,
                n_void_probes=self.config.n_void_probes
            )
        }

        # Operator weights (normalized to sum to 1)
        self.operator_weights = {
            LeapType.ANALOGICAL: self.config.analogical_weight,
            LeapType.DIFFUSION: self.config.diffusion_weight,
            LeapType.ORTHOGONAL: self.config.orthogonal_weight,
            LeapType.EXTRAPOLATION: self.config.extrapolation_weight
        }
        self._normalize_weights()

        # Leap history for learning
        self.leap_history: List[Tuple[CreativeLeap, float]] = []  # (leap, quality)
        self.max_history = 1000

        # Thread safety
        self._lock = threading.RLock()

    def _normalize_weights(self):
        """Normalize operator weights to sum to 1."""
        total = sum(self.operator_weights.values())
        if total > EPS:
            for k in self.operator_weights:
                self.operator_weights[k] /= total

    def select_operator(
        self,
        dopamine: float = 0.5,
        force_type: Optional[LeapType] = None
    ) -> LeapOperator:
        """Select an operator based on weights and dopamine modulation.

        Args:
            dopamine: 0-1 normalized dopamine level
            force_type: Force a specific operator type

        Returns:
            Selected operator
        """
        if force_type is not None:
            return self.operators[force_type]

        # Modulate weights based on dopamine
        # High dopamine -> more radical operators (analogical, orthogonal)
        # Low dopamine -> more conservative (diffusion)
        modulated_weights = {}

        for leap_type, base_weight in self.operator_weights.items():
            if leap_type in [LeapType.ANALOGICAL, LeapType.ORTHOGONAL]:
                # Radical operators boosted by high dopamine
                modulator = 0.5 + dopamine
            elif leap_type == LeapType.EXTRAPOLATION:
                # Exploration boosted moderately
                modulator = 0.7 + 0.6 * dopamine
            else:  # DIFFUSION
                # Conservative operators favored at low dopamine
                modulator = 1.5 - dopamine

            modulated_weights[leap_type] = base_weight * modulator

        # Normalize
        total = sum(modulated_weights.values())

        # Sample operator
        r = torch.rand(1).item() * total
        cumsum = 0.0
        for leap_type, weight in modulated_weights.items():
            cumsum += weight
            if r <= cumsum:
                return self.operators[leap_type]

        # Fallback
        return self.operators[LeapType.DIFFUSION]

    def leap(
        self,
        sources: List[torch.Tensor],
        dopamine: float = 0.5,
        intensity: Optional[float] = None,
        force_type: Optional[LeapType] = None,
        context: Optional[torch.Tensor] = None
    ) -> CreativeLeap:
        """Perform a creative leap.

        Args:
            sources: Source embeddings to leap from
            dopamine: 0-1 normalized dopamine (affects operator selection)
            intensity: Leap intensity (None = use default)
            force_type: Force specific operator type
            context: Optional context embedding

        Returns:
            Creative leap result
        """
        with self._lock:
            operator = self.select_operator(dopamine, force_type)

            intensity = intensity if intensity is not None else self.config.default_intensity

            # Dopamine also modulates intensity
            # High dopamine = more intense leaps
            modulated_intensity = intensity * (0.5 + dopamine)

            leap = operator.leap(sources, context, modulated_intensity)

            return leap

    def batch_leap(
        self,
        sources: List[torch.Tensor],
        n_leaps: int = 5,
        dopamine: float = 0.5,
        diversity_bonus: float = 0.3
    ) -> List[CreativeLeap]:
        """Generate multiple diverse creative leaps.

        Args:
            sources: Source embeddings
            n_leaps: Number of leaps to generate
            dopamine: Dopamine level
            diversity_bonus: Bonus for leaps different from previous

        Returns:
            List of creative leaps
        """
        leaps = []
        used_types = set()

        for i in range(n_leaps):
            # Vary intensity across batch
            intensity_variation = 0.7 + 0.6 * (i / n_leaps)

            # Try to use different operator types for diversity
            force_type = None
            if diversity_bonus > 0 and i > 0:
                unused_types = set(self.operators.keys()) - used_types
                if unused_types:
                    force_type = list(unused_types)[torch.randint(len(unused_types), (1,)).item()]

            leap = self.leap(
                sources,
                dopamine=dopamine,
                intensity=self.config.default_intensity * intensity_variation,
                force_type=force_type
            )

            leaps.append(leap)
            used_types.add(leap.leap_type)

        return leaps

    def record_quality(self, leap: CreativeLeap, quality: float) -> None:
        """Record quality of a leap for learning.

        Args:
            leap: The creative leap
            quality: Quality score (0-1)
        """
        with self._lock:
            self.leap_history.append((leap, quality))

            # Prune history
            if len(self.leap_history) > self.max_history:
                self.leap_history = self.leap_history[-self.max_history:]

    def adapt_weights(self) -> None:
        """Adapt operator weights based on historical quality."""
        with self._lock:
            if len(self.leap_history) < 10:
                return  # Not enough data

            # Compute average quality per operator type
            type_qualities: Dict[LeapType, List[float]] = {t: [] for t in LeapType}

            for leap, quality in self.leap_history:
                type_qualities[leap.leap_type].append(quality)

            # Update weights based on average quality
            for leap_type in self.operator_weights:
                qualities = type_qualities.get(leap_type, [])
                if qualities:
                    avg_quality = sum(qualities) / len(qualities)
                    # Increase weight for high-quality operators
                    self.operator_weights[leap_type] *= (0.9 + 0.2 * avg_quality)

            self._normalize_weights()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about leap performance."""
        with self._lock:
            stats = {
                "total_leaps": len(self.leap_history),
                "operator_weights": dict(self.operator_weights),
                "per_type_stats": {}
            }

            for leap_type in LeapType:
                type_leaps = [(leap, q) for leap, q in self.leap_history if leap.leap_type == leap_type]
                if type_leaps:
                    qualities = [q for _, q in type_leaps]
                    novelties = [leap.novelty_score for leap, _ in type_leaps]
                    distances = [leap.leap_distance for leap, _ in type_leaps]

                    stats["per_type_stats"][leap_type.name] = {
                        "count": len(type_leaps),
                        "avg_quality": sum(qualities) / len(qualities),
                        "avg_novelty": sum(novelties) / len(novelties),
                        "avg_distance": sum(distances) / len(distances)
                    }

            return stats
