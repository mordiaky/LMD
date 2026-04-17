"""Coupling Field - How memories interact with each other.

The coupling field determines how living memories influence each other:
- Content similarity: Similar experiences couple
- Valence compatibility: Similar emotional arcs resonate
- Phase alignment: Memories at similar story points connect

Gamma_ij = sim(c_i, c_j) * valence_compat(v_i, v_j) * phase_align(phi_i, phi_j)

Invented by Joshua R. Thomas, January 2026.
"""

import math
from typing import List, Optional, Tuple

import torch

from .config import LMDConfig
from .living_memory import LivingMemory


class CouplingField:
    """Computes and manages coupling between living memories.

    The coupling field is a dynamic matrix that determines how
    memories influence each other through:
    - Content resonance (semantic similarity)
    - Valence resonance (emotional compatibility)
    - Phase alignment (narrative position matching)
    """

    def __init__(self, config: LMDConfig):
        self.config = config
        self._coupling_matrix: Optional[torch.Tensor] = None
        self._memory_ids: List[int] = []

    def compute_coupling(
        self,
        memories: List[LivingMemory],
        include_phase: bool = True
    ) -> torch.Tensor:
        """Compute the full coupling matrix for a set of memories.

        Args:
            memories: List of living memories
            include_phase: Whether to include phase alignment in coupling

        Returns:
            [n, n] coupling matrix where Gamma[i,j] is coupling strength
        """
        n = len(memories)
        if n == 0:
            return torch.zeros((0, 0))

        # Stack content vectors
        contents = torch.stack([m.content for m in memories])  # [n, dim]

        # === Content Similarity ===
        # Cosine similarity matrix
        norms = torch.norm(contents, dim=1, keepdim=True)  # [n, 1]
        norms = torch.clamp(norms, min=1e-8)
        normalized = contents / norms
        content_sim = torch.mm(normalized, normalized.t())  # [n, n]

        # === Valence Compatibility ===
        valence_compat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    valence_compat[i, j] = memories[i].valence_compatibility(memories[j])

        # === Phase Alignment (optional) ===
        if include_phase:
            phase_align = torch.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        phase_align[i, j] = memories[i].phase_alignment(memories[j])
        else:
            phase_align = torch.ones((n, n))

        # === Combined Coupling ===
        # Gamma = w1 * content + w2 * valence + w3 * phase
        w1 = self.config.coupling_content_weight
        w2 = self.config.coupling_valence_weight
        w3 = self.config.coupling_phase_weight if include_phase else 0.0

        # Normalize weights
        total_w = w1 + w2 + w3
        if total_w > 0:
            w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w

        coupling = w1 * content_sim + w2 * valence_compat + w3 * phase_align

        # Zero out diagonal (no self-coupling)
        coupling.fill_diagonal_(0.0)

        # Apply threshold
        coupling = torch.where(
            coupling > self.config.coupling_threshold,
            coupling,
            torch.zeros_like(coupling)
        )

        # Store for later use
        self._coupling_matrix = coupling
        self._memory_ids = [m.id for m in memories]

        return coupling

    def get_coupling(self, memory_i: LivingMemory, memory_j: LivingMemory) -> float:
        """Get coupling strength between two specific memories."""
        # Direct computation
        content_sim = memory_i.similarity(memory_j)
        valence_compat = memory_i.valence_compatibility(memory_j)
        phase_align = memory_i.phase_alignment(memory_j)

        w1 = self.config.coupling_content_weight
        w2 = self.config.coupling_valence_weight
        w3 = self.config.coupling_phase_weight

        total_w = w1 + w2 + w3
        if total_w > 0:
            w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w

        coupling = w1 * content_sim + w2 * valence_compat + w3 * phase_align

        if coupling < self.config.coupling_threshold:
            return 0.0

        return coupling

    def get_neighbors(
        self,
        memory: LivingMemory,
        all_memories: List[LivingMemory],
        top_k: int = 5
    ) -> List[Tuple[LivingMemory, float]]:
        """Get the most strongly coupled neighbors of a memory.

        Args:
            memory: The memory to find neighbors for
            all_memories: All memories in the system
            top_k: Number of neighbors to return

        Returns:
            List of (neighbor_memory, coupling_strength) tuples
        """
        couplings = []
        for other in all_memories:
            if other.id != memory.id:
                strength = self.get_coupling(memory, other)
                if strength > 0:
                    couplings.append((other, strength))

        # Sort by coupling strength
        couplings.sort(key=lambda x: x[1], reverse=True)

        return couplings[:top_k]

    def compute_resonance_force(
        self,
        memory: LivingMemory,
        all_memories: List[LivingMemory]
    ) -> Tuple[float, float]:
        """Compute the resonance force on a memory from all others.

        Returns:
            (valence_force, phase_force) - forces pulling memory toward neighbors
        """
        valence_force = 0.0
        phase_force = 0.0
        total_weight = 0.0

        for other in all_memories:
            if other.id != memory.id and other.is_alive:
                coupling = self.get_coupling(memory, other)
                if coupling > 0:
                    # Valence pull: toward similar emotional states
                    valence_diff = other.current_valence - memory.current_valence
                    valence_force += coupling * valence_diff

                    # Phase pull: toward aligned narrative positions
                    phase_diff = other.phase - memory.phase
                    # Handle wrap-around
                    if phase_diff > math.pi:
                        phase_diff -= 2 * math.pi
                    elif phase_diff < -math.pi:
                        phase_diff += 2 * math.pi
                    phase_force += coupling * phase_diff

                    total_weight += coupling

        # Normalize
        if total_weight > 0:
            valence_force /= total_weight
            phase_force /= total_weight

        return valence_force, phase_force

    def cluster_by_coupling(
        self,
        memories: List[LivingMemory],
        threshold: float = 0.5
    ) -> List[List[LivingMemory]]:
        """Cluster memories by coupling strength.

        Simple connected-components clustering based on coupling.
        """
        if not memories:
            return []

        n = len(memories)
        coupling = self.compute_coupling(memories, include_phase=False)

        # Union-find for clustering
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Connect coupled memories
        for i in range(n):
            for j in range(i + 1, n):
                if coupling[i, j] > threshold:
                    union(i, j)

        # Group by cluster
        clusters = {}
        for i, m in enumerate(memories):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(m)

        return list(clusters.values())

    def coupling_statistics(self, memories: List[LivingMemory]) -> dict:
        """Compute statistics about the coupling field."""
        if len(memories) < 2:
            return {
                "mean_coupling": 0.0,
                "max_coupling": 0.0,
                "active_pairs": 0,
                "total_pairs": 0,
                "density": 0.0,
            }

        coupling = self.compute_coupling(memories)

        # Exclude diagonal
        mask = ~torch.eye(len(memories), dtype=torch.bool)
        values = coupling[mask]

        active = (values > 0).sum().item()
        total = len(values)

        return {
            "mean_coupling": values.mean().item() if total > 0 else 0.0,
            "max_coupling": values.max().item() if total > 0 else 0.0,
            "std_coupling": values.std().item() if total > 0 else 0.0,
            "active_pairs": int(active),
            "total_pairs": total,
            "density": active / total if total > 0 else 0.0,
        }
