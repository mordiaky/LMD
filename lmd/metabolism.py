"""Memory Metabolism - Energy dynamics for living memories.

Memories are ALIVE - they have metabolic energy that:
- Decays naturally over time (forgetting)
- Increases when activated (recall strengthens)
- Transfers between coupled memories (resonance)
- Determines memory state (vivid, dormant, fading, ghost)

dE/dt = activation - decay + transfer - replay_cost

Invented by Joshua R. Thomas, January 2026.
"""

from typing import Dict, List, Tuple

import torch

from .config import LMDConfig
from .coupling import CouplingField
from .living_memory import LivingMemory, MetabolicState


class MemoryMetabolism:
    """Manages the metabolic dynamics of living memories.

    Core equation:
    dE_i/dt = rho * A_i(t) - delta * E_i + mu * sum_j(Gamma_ij * E_j) - sigma * replay

    Where:
    - rho * A_i(t) = activation input (external triggers)
    - delta * E_i = natural decay (forgetting)
    - mu * sum_j(Gamma_ij * E_j) = energy transfer from coupled memories
    - sigma * replay = cost of being replayed (consolidation)
    """

    def __init__(self, config: LMDConfig, coupling_field: CouplingField):
        self.config = config
        self.coupling = coupling_field

        # Track statistics
        self.total_energy_history: List[float] = []
        self.deaths: int = 0
        self.births: int = 0

    def step(
        self,
        memories: List[LivingMemory],
        activations: Dict[int, float] = None,
        dt: float = 1.0
    ) -> Dict[str, float]:
        """Advance metabolism by one timestep.

        Args:
            memories: All living memories
            activations: Dict mapping memory_id -> activation strength
            dt: Timestep size

        Returns:
            Metrics about the metabolic step
        """
        if not memories:
            return {"total_energy": 0.0, "alive_count": 0}

        activations = activations or {}

        # Compute coupling matrix for energy transfer
        coupling_matrix = self.coupling.compute_coupling(memories, include_phase=True)
        n = len(memories)

        # Get current energies
        energies = torch.tensor([m.energy for m in memories])

        # === Natural Decay ===
        decay = self.config.energy_decay_rate ** dt
        new_energies = energies * decay

        # === Activation Input ===
        for i, m in enumerate(memories):
            if m.id in activations:
                boost = activations[m.id] * self.config.activation_boost
                new_energies[i] += boost

        # === Spontaneous Activation (NEW) ===
        # High-valence memories have a chance to spontaneously reactivate
        import random
        for i, m in enumerate(memories):
            if random.random() < self.config.spontaneous_activation_prob:
                # Base reactivation
                boost = 0.1
                # Bonus for emotionally significant memories
                valence_magnitude = abs(m.current_valence)
                boost += valence_magnitude * self.config.valence_activation_bonus
                new_energies[i] += boost

        # === Mutual Sustenance (NEW) ===
        # Coupled memories help keep each other alive
        if n > 1:
            sustenance_rate = self.config.mutual_sustenance_rate * dt
            for i in range(n):
                sustenance = 0.0
                for j in range(n):
                    if i != j and coupling_matrix[i, j] > 0.3:  # Strong coupling only
                        # Coupled memory provides sustenance proportional to its energy
                        sustenance += coupling_matrix[i, j] * energies[j] * sustenance_rate
                new_energies[i] += sustenance

        # === Energy Transfer (from coupled memories) ===
        if n > 1:
            # Energy flows from high-energy to low-energy through coupling
            transfer_rate = self.config.energy_transfer_rate * dt

            for i in range(n):
                for j in range(n):
                    if i != j and coupling_matrix[i, j] > 0:
                        # Energy flows proportional to coupling and energy difference
                        energy_diff = energies[j] - energies[i]
                        flow = transfer_rate * coupling_matrix[i, j] * energy_diff
                        new_energies[i] += flow

        # === Apply Bounds ===
        new_energies = torch.clamp(
            new_energies,
            min=self.config.min_energy,
            max=self.config.max_energy
        )

        # === Update Memories ===
        deaths_this_step = 0
        for i, m in enumerate(memories):
            old_state = m.metabolic_state
            m.energy = new_energies[i].item()
            new_state = m.metabolic_state

            # Track state transitions
            if old_state != MetabolicState.GHOST and new_state == MetabolicState.GHOST:
                deaths_this_step += 1

        self.deaths += deaths_this_step

        # === Track History ===
        total_energy = new_energies.sum().item()
        self.total_energy_history.append(total_energy)

        # === Return Metrics ===
        alive_count = sum(1 for m in memories if m.is_alive)

        return {
            "total_energy": total_energy,
            "mean_energy": total_energy / n if n > 0 else 0.0,
            "alive_count": alive_count,
            "ghost_count": n - alive_count,
            "deaths_this_step": deaths_this_step,
            "energy_variance": new_energies.var().item() if n > 1 else 0.0,
        }

    def activate_memory(
        self,
        memory: LivingMemory,
        strength: float = 1.0,
        timestep: int = 0
    ):
        """Activate a specific memory, boosting its energy."""
        boost = strength * self.config.activation_boost
        memory.energy = min(self.config.max_energy, memory.energy + boost)
        memory.last_activated = timestep
        memory.activation_count += 1

    def propagate_activation(
        self,
        seed_memory: LivingMemory,
        all_memories: List[LivingMemory],
        strength: float = 1.0,
        decay_per_hop: float = 0.5,
        max_hops: int = 3,
        timestep: int = 0
    ) -> List[LivingMemory]:
        """Propagate activation through coupling field.

        Starts at seed_memory and spreads to coupled neighbors,
        decaying with each hop.

        Returns:
            List of all activated memories
        """
        activated = [seed_memory]
        self.activate_memory(seed_memory, strength, timestep)

        current_strength = strength
        frontier = [seed_memory]
        visited = {seed_memory.id}

        for _hop in range(max_hops):
            current_strength *= decay_per_hop
            if current_strength < 0.1:
                break

            new_frontier = []
            for memory in frontier:
                neighbors = self.coupling.get_neighbors(memory, all_memories)
                for neighbor, coupling in neighbors:
                    if neighbor.id not in visited and neighbor.is_alive:
                        visited.add(neighbor.id)
                        effective_strength = current_strength * coupling
                        self.activate_memory(neighbor, effective_strength, timestep)
                        activated.append(neighbor)
                        new_frontier.append(neighbor)

            frontier = new_frontier

        return activated

    def apply_replay_cost(
        self,
        memory: LivingMemory,
        cost: float = 0.1
    ):
        """Apply metabolic cost for replaying a memory.

        Replay strengthens consolidation but costs energy.
        """
        memory.energy = max(self.config.min_energy, memory.energy - cost)

    def prune_dead_memories(
        self,
        memories: List[LivingMemory]
    ) -> Tuple[List[LivingMemory], List[LivingMemory]]:
        """Remove dead memories (below death threshold).

        Returns:
            (surviving_memories, dead_memories)
        """
        surviving = []
        dead = []

        for m in memories:
            if m.energy > self.config.death_threshold:
                surviving.append(m)
            else:
                dead.append(m)
                self.deaths += 1

        return surviving, dead

    def get_energy_distribution(
        self,
        memories: List[LivingMemory]
    ) -> Dict[MetabolicState, int]:
        """Get count of memories in each metabolic state."""
        distribution = dict.fromkeys(MetabolicState, 0)
        for m in memories:
            distribution[m.metabolic_state] += 1
        return distribution

    def compute_aliveness_score(
        self,
        memories: List[LivingMemory]
    ) -> float:
        """Compute overall 'aliveness' of the memory system.

        Score based on:
        - Energy variance (memories differ in aliveness)
        - Active ratio (how many are engaged)
        - Energy dynamics (not just flat decay)

        Returns:
            Aliveness score in [0, 1]
        """
        if not memories:
            return 0.0

        energies = torch.tensor([m.energy for m in memories])

        # Energy variance (normalized)
        mean_e = energies.mean()
        var_e = energies.var()
        variance_score = min(1.0, var_e / (mean_e + 1e-8))

        # Active ratio
        active_count = sum(1 for m in memories if m.is_active)
        active_ratio = active_count / len(memories)

        # Not-dead ratio
        alive_count = sum(1 for m in memories if m.is_alive)
        alive_ratio = alive_count / len(memories)

        # Combined score
        score = 0.3 * variance_score + 0.3 * active_ratio + 0.4 * alive_ratio

        return score

    def metabolism_statistics(self, memories: List[LivingMemory]) -> dict:
        """Get comprehensive metabolism statistics."""
        if not memories:
            return {
                "total_energy": 0.0,
                "mean_energy": 0.0,
                "energy_variance": 0.0,
                "aliveness_score": 0.0,
                "state_distribution": {},
                "total_deaths": self.deaths,
            }

        energies = torch.tensor([m.energy for m in memories])

        return {
            "total_energy": energies.sum().item(),
            "mean_energy": energies.mean().item(),
            "max_energy": energies.max().item(),
            "min_energy": energies.min().item(),
            "energy_variance": energies.var().item(),
            "aliveness_score": self.compute_aliveness_score(memories),
            "state_distribution": {
                k.value: v for k, v in self.get_energy_distribution(memories).items()
            },
            "total_deaths": self.deaths,
            "total_births": self.births,
        }
