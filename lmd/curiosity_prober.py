"""Active Curiosity Prober: Targeted Frontier Exploration.

Instead of random noise for creativity, actively probe low-density regions:
- Use RepulsionField to identify "voids" (gaps in explored space)
- Direct internal curiosity toward voids with weighted probing
- Ray-trace from dense clusters into unexplored territory
- Speculative extrapolation beyond known boundaries

This turns curiosity into systematic discovery of novel basins.

Invented by Joshua R. Thomas, January 2026.
"""

import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch

from .safeguards import EPS, RepulsionField, safe_normalize


class ProbeStrategy(Enum):
    """Strategies for curiosity probing."""
    VOID_SEEK = auto()        # Seek low-density regions
    FRONTIER = auto()         # Explore edges of known space
    EXTRAPOLATE = auto()      # Ray-trace beyond clusters
    INTERPOLATE = auto()      # Find gaps between clusters
    ORTHOGONAL = auto()       # Explore perpendicular to known


@dataclass
class ProbeResult:
    """Result of a curiosity probe."""
    target: torch.Tensor          # The target point to explore
    strategy: ProbeStrategy       # Which strategy found it
    novelty_estimate: float       # Expected novelty at target
    density_estimate: float       # Estimated local density
    confidence: float             # How confident in this probe
    path: Optional[List[torch.Tensor]] = None  # Path to target if applicable


@dataclass
class VoidRegion:
    """A detected void (low-density region) in embedding space."""
    center: torch.Tensor
    radius: float
    density: float
    neighbors: int  # How many known points nearby


class ActiveCuriosityProber:
    """Actively probes embedding space to find interesting unexplored regions.

    Key innovations:
    1. Void detection: Find gaps in explored space
    2. Frontier mapping: Identify edges of knowledge
    3. Ray-tracing: Extrapolate beyond known territory
    4. Interpolation gaps: Find holes between clusters
    """

    def __init__(
        self,
        content_dim: int = 32,
        n_probes: int = 100,
        void_threshold: float = 0.3,  # Density below this = void
        frontier_margin: float = 0.2,  # How far beyond edge to probe
        repulsion_field: Optional[RepulsionField] = None
    ):
        self.content_dim = content_dim
        self.n_probes = n_probes
        self.void_threshold = void_threshold
        self.frontier_margin = frontier_margin
        self.repulsion_field = repulsion_field

        # Cache for efficiency
        self._probe_cache: List[torch.Tensor] = []
        self._density_cache: Dict[str, float] = {}
        self._known_voids: List[VoidRegion] = []
        self.max_cached_voids = 50

        # Strategy weights (dynamically adjusted)
        self.strategy_weights = {
            ProbeStrategy.VOID_SEEK: 0.3,
            ProbeStrategy.FRONTIER: 0.25,
            ProbeStrategy.EXTRAPOLATE: 0.2,
            ProbeStrategy.INTERPOLATE: 0.15,
            ProbeStrategy.ORTHOGONAL: 0.1
        }

        # Thread safety
        self._lock = threading.RLock()

    def estimate_density(
        self,
        point: torch.Tensor,
        known_points: List[torch.Tensor],
        bandwidth: float = 0.5
    ) -> float:
        """Estimate local density using Gaussian kernel."""
        if not known_points:
            return 0.0

        stacked = torch.stack(known_points)
        dists = (stacked - point.unsqueeze(0)).norm(dim=1)

        # Gaussian kernel
        kernel = torch.exp(-0.5 * (dists / bandwidth) ** 2)
        density = kernel.mean().item()

        return density

    def find_voids(
        self,
        known_points: List[torch.Tensor],
        n_samples: int = 100
    ) -> List[VoidRegion]:
        """Find void regions (low-density areas) in embedding space.

        Algorithm:
        1. Generate uniform random probes
        2. Compute density at each probe
        3. Return probes with density below threshold
        """
        if not known_points:
            # Everything is void
            center = safe_normalize(torch.randn(self.content_dim))
            return [VoidRegion(center=center, radius=1.0, density=0.0, neighbors=0)]

        stacked = torch.stack(known_points)

        # Compute bounding statistics
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0).clamp(min=0.1)

        # Generate probes in and around known space
        probes = []

        # In-space probes (normal distribution around mean)
        in_space = torch.randn(n_samples // 2, self.content_dim) * std * 1.5 + mean
        probes.append(in_space)

        # Out-space probes (uniform on larger sphere)
        radius = stacked.norm(dim=1).max() * 1.5
        out_space = safe_normalize(torch.randn(n_samples // 2, self.content_dim)) * radius
        probes.append(out_space)

        all_probes = torch.cat(probes, dim=0)

        # Compute density at each probe
        voids = []
        for probe in all_probes:
            density = self.estimate_density(probe, known_points)

            if density < self.void_threshold:
                # Count nearby neighbors
                dists = (stacked - probe.unsqueeze(0)).norm(dim=1)
                neighbors = (dists < 1.0).sum().item()

                # Estimate void radius (distance to nearest point)
                min_dist = dists.min().item()

                voids.append(VoidRegion(
                    center=probe,
                    radius=min_dist,
                    density=density,
                    neighbors=int(neighbors)
                ))

        # Sort by lowest density (most void-like)
        voids.sort(key=lambda v: v.density)

        return voids[:self.max_cached_voids]

    def find_frontier(
        self,
        known_points: List[torch.Tensor],
        n_directions: int = 20
    ) -> List[torch.Tensor]:
        """Find frontier points at the edge of known space.

        Algorithm:
        1. Compute convex hull direction (pointing outward)
        2. Find outermost points in each direction
        3. Extend slightly beyond
        """
        if not known_points:
            return [safe_normalize(torch.randn(self.content_dim))]

        stacked = torch.stack(known_points)
        centroid = stacked.mean(dim=0)

        # Generate random directions
        directions = safe_normalize(torch.randn(n_directions, self.content_dim))

        frontier_points = []

        for direction in directions:
            # Project all points onto direction
            projections = (stacked - centroid) @ direction

            # Find furthest point in this direction
            max_idx = projections.argmax()
            stacked[max_idx]
            max_proj = projections[max_idx].item()

            # Extend beyond by frontier_margin
            frontier = centroid + direction * (max_proj * (1 + self.frontier_margin))
            frontier_points.append(frontier)

        return frontier_points

    def find_interpolation_gaps(
        self,
        known_points: List[torch.Tensor],
        n_pairs: int = 20
    ) -> List[torch.Tensor]:
        """Find gaps between distant clusters via midpoint analysis.

        Algorithm:
        1. Find pairs of distant points
        2. Sample midpoints between them
        3. Return midpoints with low local density
        """
        if len(known_points) < 2:
            return []

        stacked = torch.stack(known_points)
        n_points = len(known_points)

        # Find distant pairs
        dists = torch.cdist(stacked, stacked)

        # Get top-k distant pairs
        flat_dists = dists.view(-1)
        k = min(n_pairs * 2, n_points * n_points)
        _, top_indices = flat_dists.topk(k)

        gaps = []
        used_pairs = set()

        for idx in top_indices:
            i, j = idx.item() // n_points, idx.item() % n_points
            if i >= j:  # Skip diagonal and duplicates
                continue
            if (i, j) in used_pairs:
                continue
            used_pairs.add((i, j))

            # Midpoint between distant points
            midpoint = 0.5 * (stacked[i] + stacked[j])

            # Check if midpoint is low density
            density = self.estimate_density(midpoint, known_points)
            if density < self.void_threshold * 2:  # Slightly relaxed threshold
                gaps.append(midpoint)

            if len(gaps) >= n_pairs:
                break

        return gaps

    def extrapolate_ray(
        self,
        known_points: List[torch.Tensor],
        n_rays: int = 10,
        extrapolation_factor: float = 1.5
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Ray-trace from dense center outward into unexplored territory.

        Returns: List of (origin, target) pairs
        """
        if not known_points:
            origin = torch.zeros(self.content_dim)
            target = safe_normalize(torch.randn(self.content_dim))
            return [(origin, target)]

        stacked = torch.stack(known_points)
        centroid = stacked.mean(dim=0)

        rays = []

        # Method 1: Ray from centroid through random known point
        indices = torch.randperm(len(known_points))[:n_rays // 2]
        for idx in indices:
            point = stacked[idx]
            direction = safe_normalize(point - centroid)
            distance = (point - centroid).norm()

            # Extend beyond
            target = centroid + direction * (distance * extrapolation_factor)
            rays.append((point, target))

        # Method 2: Ray from centroid in random directions
        random_dirs = safe_normalize(torch.randn(n_rays // 2, self.content_dim))
        max_radius = (stacked - centroid).norm(dim=1).max()

        for direction in random_dirs:
            target = centroid + direction * (max_radius * extrapolation_factor)
            rays.append((centroid, target))

        return rays

    def generate_orthogonal_probes(
        self,
        known_points: List[torch.Tensor],
        n_probes: int = 10
    ) -> List[torch.Tensor]:
        """Generate probes orthogonal to the main variation directions.

        These explore "perpendicular" concepts.
        """
        if len(known_points) < 2:
            return [safe_normalize(torch.randn(self.content_dim))]

        stacked = torch.stack(known_points)
        centered = stacked - stacked.mean(dim=0, keepdim=True)

        # SVD to find principal directions
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            principal_dirs = Vh[:min(3, Vh.shape[0])]
        except RuntimeError:
            principal_dirs = torch.eye(min(3, self.content_dim), self.content_dim)

        probes = []

        # Generate directions orthogonal to all principal directions
        for _ in range(n_probes):
            candidate = torch.randn(self.content_dim)

            # Gram-Schmidt: subtract projection onto principal directions
            for pdir in principal_dirs:
                proj = (candidate @ pdir) * pdir
                candidate = candidate - proj

            if candidate.norm() > EPS:
                candidate = safe_normalize(candidate)
                # Place probe at mean + orthogonal direction
                probe = stacked.mean(dim=0) + candidate * stacked.std()
                probes.append(probe)

        return probes

    def probe(
        self,
        known_points: List[torch.Tensor],
        strategy: Optional[ProbeStrategy] = None,
        n_results: int = 5
    ) -> List[ProbeResult]:
        """Perform curiosity probing to find interesting exploration targets.

        Args:
            known_points: Currently known points in embedding space
            strategy: Specific strategy to use (None = sample based on weights)
            n_results: Number of probe results to return

        Returns:
            List of probe results sorted by novelty estimate
        """
        with self._lock:
            results = []

            if strategy:
                strategies = [strategy]
            else:
                # Sample strategies based on weights
                strategies = list(self.strategy_weights.keys())

            for strat in strategies:
                if strat == ProbeStrategy.VOID_SEEK:
                    voids = self.find_voids(known_points, n_samples=self.n_probes)
                    for void in voids[:n_results]:
                        results.append(ProbeResult(
                            target=void.center,
                            strategy=strat,
                            novelty_estimate=1.0 - void.density,
                            density_estimate=void.density,
                            confidence=0.8
                        ))

                elif strat == ProbeStrategy.FRONTIER:
                    frontiers = self.find_frontier(known_points)
                    for frontier in frontiers[:n_results]:
                        density = self.estimate_density(frontier, known_points)
                        results.append(ProbeResult(
                            target=frontier,
                            strategy=strat,
                            novelty_estimate=0.7,
                            density_estimate=density,
                            confidence=0.7
                        ))

                elif strat == ProbeStrategy.EXTRAPOLATE:
                    rays = self.extrapolate_ray(known_points)
                    for origin, target in rays[:n_results]:
                        density = self.estimate_density(target, known_points)
                        results.append(ProbeResult(
                            target=target,
                            strategy=strat,
                            novelty_estimate=0.8,
                            density_estimate=density,
                            confidence=0.6,
                            path=[origin, target]
                        ))

                elif strat == ProbeStrategy.INTERPOLATE:
                    gaps = self.find_interpolation_gaps(known_points)
                    for gap in gaps[:n_results]:
                        density = self.estimate_density(gap, known_points)
                        results.append(ProbeResult(
                            target=gap,
                            strategy=strat,
                            novelty_estimate=0.6,
                            density_estimate=density,
                            confidence=0.75
                        ))

                elif strat == ProbeStrategy.ORTHOGONAL:
                    ortho = self.generate_orthogonal_probes(known_points)
                    for probe in ortho[:n_results]:
                        density = self.estimate_density(probe, known_points)
                        results.append(ProbeResult(
                            target=probe,
                            strategy=strat,
                            novelty_estimate=0.65,
                            density_estimate=density,
                            confidence=0.5
                        ))

            # Sort by novelty * confidence (expected value)
            results.sort(key=lambda r: r.novelty_estimate * r.confidence, reverse=True)

            return results[:n_results]

    def directed_curiosity(
        self,
        known_points: List[torch.Tensor],
        repulsion_field: Optional[RepulsionField] = None,
        dopamine: float = 0.5,
        n_targets: int = 3
    ) -> List[ProbeResult]:
        """Generate curiosity-driven exploration targets.

        High dopamine -> more aggressive exploration (frontier, extrapolate)
        Low dopamine -> safer exploration (interpolate, orthogonal)

        Args:
            known_points: Known embeddings
            repulsion_field: Optional repulsion field to avoid
            dopamine: 0-1 normalized dopamine level
            n_targets: Number of targets to return

        Returns:
            Probe results optimized for current dopamine level
        """
        # Adjust strategy weights based on dopamine
        adjusted_weights = {}
        for strat, base_weight in self.strategy_weights.items():
            if strat in [ProbeStrategy.FRONTIER, ProbeStrategy.EXTRAPOLATE]:
                # Aggressive strategies boosted by dopamine
                modifier = 0.5 + dopamine
            elif strat == ProbeStrategy.VOID_SEEK:
                # Void seeking is neutral
                modifier = 1.0
            else:
                # Conservative strategies at low dopamine
                modifier = 1.5 - dopamine

            adjusted_weights[strat] = base_weight * modifier

        # Normalize
        total = sum(adjusted_weights.values())
        for k in adjusted_weights:
            adjusted_weights[k] /= total

        # Sample strategies
        all_results = []

        for strat, weight in adjusted_weights.items():
            n_for_strat = max(1, int(n_targets * weight * 2))
            strat_results = self.probe(known_points, strategy=strat, n_results=n_for_strat)
            all_results.extend(strat_results)

        # Filter by repulsion if provided
        if repulsion_field:
            filtered = []
            for result in all_results:
                repulsion_score, _ = repulsion_field.compute_repulsion(result.target)
                # Reduce novelty estimate by repulsion strength
                result.novelty_estimate *= (1.0 - repulsion_score)
                if result.novelty_estimate > 0.1:  # Still worth exploring
                    filtered.append(result)
            all_results = filtered

        # Sort and return top
        all_results.sort(key=lambda r: r.novelty_estimate * r.confidence, reverse=True)

        return all_results[:n_targets]

    def update_from_exploration(
        self,
        explored_point: torch.Tensor,
        was_productive: bool,
        novelty_found: float
    ) -> None:
        """Update internal state based on exploration results.

        This allows the prober to learn which strategies work best.
        """
        with self._lock:
            # For now, just cache explored points
            # Future: Update strategy weights based on productivity

            # Remove voids that were explored
            self._known_voids = [
                v for v in self._known_voids
                if (v.center - explored_point).norm() > v.radius
            ]

    def get_curiosity_statistics(self) -> Dict[str, Any]:
        """Get statistics about curiosity probing."""
        return {
            "strategy_weights": dict(self.strategy_weights),
            "known_voids": len(self._known_voids),
            "void_threshold": self.void_threshold,
            "frontier_margin": self.frontier_margin
        }


class CuriosityDrivenWill:
    """Integrates ActiveCuriosityProber with will generation.

    Generates will vectors that are:
    1. Directed toward curiosity targets
    2. Modulated by dopamine
    3. Constrained by repulsion field
    """

    def __init__(
        self,
        content_dim: int = 32,
        prober: Optional[ActiveCuriosityProber] = None,
        repulsion_field: Optional[RepulsionField] = None
    ):
        self.content_dim = content_dim
        self.prober = prober or ActiveCuriosityProber(content_dim)
        self.repulsion_field = repulsion_field

        # Will generation parameters
        self.curiosity_weight = 0.6
        self.random_weight = 0.2
        self.context_weight = 0.2

    def generate_will(
        self,
        known_points: List[torch.Tensor],
        dopamine: float = 0.5,
        context: Optional[torch.Tensor] = None,
        intensity: float = 1.0
    ) -> Tuple[torch.Tensor, ProbeResult]:
        """Generate a will vector driven by curiosity.

        Args:
            known_points: Known embeddings to explore from
            dopamine: Curiosity intensity
            context: Optional context to bias exploration
            intensity: Will intensity

        Returns:
            (will_vector, best_probe_result)
        """
        # Get curiosity targets
        probe_results = self.prober.directed_curiosity(
            known_points,
            repulsion_field=self.repulsion_field,
            dopamine=dopamine,
            n_targets=5
        )

        if not probe_results:
            # Fallback to random
            random_will = safe_normalize(torch.randn(self.content_dim))
            fake_probe = ProbeResult(
                target=random_will,
                strategy=ProbeStrategy.VOID_SEEK,
                novelty_estimate=0.5,
                density_estimate=0.5,
                confidence=0.3
            )
            return random_will * intensity, fake_probe

        # Select best probe
        best_probe = probe_results[0]

        # Construct will vector
        curiosity_direction = safe_normalize(best_probe.target)

        # Add random component
        random_component = safe_normalize(torch.randn(self.content_dim))

        # Add context component if available
        context_component = safe_normalize(context) if context is not None else torch.zeros(self.content_dim)

        # Combine
        will = (
            self.curiosity_weight * curiosity_direction
            + self.random_weight * random_component
            + self.context_weight * context_component
        )

        # Modulate by dopamine (high dopamine = more directed)
        dopamine_focus = 0.5 + 0.5 * dopamine
        will = dopamine_focus * curiosity_direction + (1 - dopamine_focus) * will

        will = safe_normalize(will) * intensity

        return will, best_probe

    def batch_wills(
        self,
        known_points: List[torch.Tensor],
        n_wills: int = 5,
        dopamine: float = 0.5
    ) -> List[Tuple[torch.Tensor, ProbeResult]]:
        """Generate multiple diverse will vectors.

        Args:
            known_points: Known embeddings
            n_wills: Number of wills to generate
            dopamine: Curiosity intensity

        Returns:
            List of (will_vector, probe_result) pairs
        """
        probe_results = self.prober.directed_curiosity(
            known_points,
            repulsion_field=self.repulsion_field,
            dopamine=dopamine,
            n_targets=n_wills * 2
        )

        wills = []
        used_strategies = set()

        for probe in probe_results:
            if len(wills) >= n_wills:
                break

            # Prefer diversity in strategies
            if len(used_strategies) < 3 or probe.strategy not in used_strategies:
                will = safe_normalize(probe.target)
                # Add small variation
                will = will + 0.1 * torch.randn_like(will)
                will = safe_normalize(will)

                wills.append((will, probe))
                used_strategies.add(probe.strategy)

        # Fill remaining with random if needed
        while len(wills) < n_wills:
            random_will = safe_normalize(torch.randn(self.content_dim))
            fake_probe = ProbeResult(
                target=random_will,
                strategy=ProbeStrategy.VOID_SEEK,
                novelty_estimate=0.4,
                density_estimate=0.5,
                confidence=0.3
            )
            wills.append((random_will, fake_probe))

        return wills
