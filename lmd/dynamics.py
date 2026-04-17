"""LMD Dynamics Engine - The core evolution equations.

The Joshua R. Thomas Memory Equation:
dM/dt = grad_phi(N) + sum_j(Gamma_ij * R(v_i, v_j)) + A(M, context) + kappa * eta(t)

Where:
- N(phi) = Narrative Potential (story attractor landscape)
- R(v_i, v_j) = Resonance Function (emotional coupling)
- A(M, context) = Activation Function (contextual triggering)
- eta(t) = Creative Noise (generative stochasticity)

Invented by Joshua R. Thomas, January 2026.
"""

import math
from typing import Dict, List, Optional

import torch

from .config import LMDConfig
from .coupling import CouplingField
from .living_memory import LivingMemory, NarrativePhase
from .metabolism import MemoryMetabolism


class LMDDynamics:
    """The core dynamics engine for Living Memory Dynamics.

    Integrates all components:
    - Narrative flow (phase advancement through story space)
    - Valence resonance (emotional coupling between memories)
    - Metabolism (energy dynamics, life and death)
    - Creative noise (stochastic exploration)
    """

    def __init__(self, config: LMDConfig):
        self.config = config
        self.coupling = CouplingField(config)
        self.metabolism = MemoryMetabolism(config, self.coupling)

        # Time tracking
        self.timestep = 0

        # History for analysis
        self.phase_history: List[List[float]] = []
        self.energy_history: List[List[float]] = []

    def narrative_potential(self, phi: float) -> float:
        """Compute narrative potential at phase phi.

        The potential creates attractors at stable narrative positions.
        V(phi) = -cos(phi) creates minima at 0 and 2*pi (setup/integration)
        and a maximum at pi (climax).

        This makes memories naturally flow through story structure.
        """
        # Basic cosine potential - stable at setup/integration
        V = -math.cos(phi)

        # Add strength scaling
        V *= self.config.narrative_potential_strength

        return V

    def narrative_force(self, phi: float) -> float:
        """Compute force from narrative potential gradient.

        F = -dV/dphi = -sin(phi) * strength

        This pulls memories through the narrative arc.
        """
        return -math.sin(phi) * self.config.narrative_potential_strength

    def step(
        self,
        memories: List[LivingMemory],
        context: Optional[torch.Tensor] = None,
        activations: Dict[int, float] = None,
        dt: float = 1.0
    ) -> Dict[str, float]:
        """Advance the entire LMD system by one timestep.

        Args:
            memories: All living memories in the system
            context: Optional context vector for activation
            activations: Optional dict of memory_id -> activation strength
            dt: Timestep size

        Returns:
            Metrics from this step
        """
        self.timestep += 1
        activations = activations or {}

        if not memories:
            return {
                "n_memories": 0,
                "timestep": self.timestep,
            }

        # === 1. Compute Coupling ===
        coupling_matrix = self.coupling.compute_coupling(memories)

        # === 2. Narrative Flow (Phase Advancement) ===
        for i, memory in enumerate(memories):
            if memory.is_alive:
                # Base velocity
                velocity = self.config.narrative_velocity

                # Modulate by energy (more energy = faster progression)
                velocity *= (0.5 + 0.5 * memory.energy)

                # Add narrative force
                force = self.narrative_force(memory.phase)
                velocity += force * 0.1

                # Phase synchronization with coupled memories
                _, phase_force = self.coupling.compute_resonance_force(memory, memories)
                velocity += phase_force * self.config.phase_sync_rate

                # Advance phase
                memory.advance_phase(velocity * dt)

        # === 3. Valence Resonance ===
        for i, memory in enumerate(memories):
            if memory.is_alive:
                valence_force, _ = self.coupling.compute_resonance_force(memory, memories)
                # Valence resonance doesn't change trajectory, but affects current emotional state
                # This is captured through the coupling dynamics

        # === 4. Metabolism (Energy Dynamics) ===
        metabolism_metrics = self.metabolism.step(memories, activations, dt)

        # === 5. Creative Noise ===
        if self.config.noise_scale > 0:
            for memory in memories:
                if memory.energy > self.config.noise_energy_threshold:
                    # Add small random perturbation to phase
                    noise = torch.randn(1).item() * self.config.noise_scale
                    memory.phase += noise

        # === 6. Context-Based Activation ===
        if context is not None:
            for memory in memories:
                # Compute similarity to context
                similarity = torch.nn.functional.cosine_similarity(
                    memory.content.unsqueeze(0),
                    context.unsqueeze(0)
                ).item()

                if similarity > 0.5:
                    # Activate similar memories
                    boost = (similarity - 0.5) * 2 * self.config.activation_boost
                    memory.activate(boost, self.timestep)

        # === 7. Track History ===
        self.phase_history.append([m.phase for m in memories])
        self.energy_history.append([m.energy for m in memories])

        # === 8. Compile Metrics ===
        phases = torch.tensor([m.phase for m in memories])
        coupling_stats = self.coupling.coupling_statistics(memories)

        metrics = {
            "timestep": self.timestep,
            "n_memories": len(memories),
            "n_alive": sum(1 for m in memories if m.is_alive),
            "mean_phase": phases.mean().item(),
            "phase_variance": phases.var().item() if len(memories) > 1 else 0.0,
            "phase_coherence": self._compute_phase_coherence(memories),
            **metabolism_metrics,
            **coupling_stats,
        }

        return metrics

    def _compute_phase_coherence(self, memories: List[LivingMemory]) -> float:
        """Compute how coherent/synchronized the phases are.

        Uses circular statistics (mean resultant length).
        High coherence = memories are at similar story positions.
        """
        if len(memories) < 2:
            return 1.0

        phases = torch.tensor([m.phase for m in memories if m.is_alive])
        if len(phases) < 2:
            return 1.0

        # Circular mean resultant length
        cos_sum = torch.cos(phases).sum()
        sin_sum = torch.sin(phases).sum()
        n = len(phases)

        R = math.sqrt(cos_sum ** 2 + sin_sum ** 2) / n

        return R

    def run_simulation(
        self,
        memories: List[LivingMemory],
        n_steps: int = 100,
        activations_schedule: Dict[int, Dict[int, float]] = None
    ) -> List[Dict[str, float]]:
        """Run a full simulation for multiple timesteps.

        Args:
            memories: Initial memories
            n_steps: Number of timesteps
            activations_schedule: Dict mapping timestep -> memory activations

        Returns:
            List of metrics for each timestep
        """
        activations_schedule = activations_schedule or {}
        history = []

        for step in range(n_steps):
            activations = activations_schedule.get(step, {})
            metrics = self.step(memories, activations=activations)
            history.append(metrics)

        return history

    def get_narrative_distribution(
        self,
        memories: List[LivingMemory]
    ) -> Dict[NarrativePhase, int]:
        """Get distribution of memories across narrative phases."""
        distribution = dict.fromkeys(NarrativePhase, 0)
        for m in memories:
            if m.is_alive:
                distribution[m.narrative_phase] += 1
        return distribution

    def get_dynamics_summary(self, memories: List[LivingMemory]) -> dict:
        """Get comprehensive summary of current dynamics state."""
        if not memories:
            return {"status": "empty"}

        alive_memories = [m for m in memories if m.is_alive]

        return {
            "timestep": self.timestep,
            "total_memories": len(memories),
            "alive_memories": len(alive_memories),
            "dead_memories": len(memories) - len(alive_memories),
            "phase_coherence": self._compute_phase_coherence(memories),
            "narrative_distribution": {
                k.name: v for k, v in self.get_narrative_distribution(memories).items()
            },
            "metabolism": self.metabolism.metabolism_statistics(memories),
            "coupling": self.coupling.coupling_statistics(memories),
        }
