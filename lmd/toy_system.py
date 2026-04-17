"""LMD Toy System - Test harness for validating Living Memory Dynamics.

This module provides:
1. Easy memory creation with various patterns
2. Simulation runner with metrics collection
3. Comparison to static baselines
4. Visualization-ready output

Invented by Joshua R. Thomas, January 2026.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .config import LMDConfig
from .dynamics import LMDDynamics
from .living_memory import LivingMemory, ValenceTrajectory
from .narrative import GeneratedNarrative, NarrativeSynthesizer


@dataclass
class LMDMetrics:
    """Comprehensive metrics for LMD validation."""

    # Aliveness metrics
    aliveness_score: float = 0.0
    energy_variance: float = 0.0
    alive_ratio: float = 0.0

    # Coupling metrics
    coupling_coherence: float = 0.0  # intra/inter cluster ratio
    mean_coupling: float = 0.0
    coupling_density: float = 0.0

    # Narrative metrics
    phase_progression_score: float = 0.0  # % frames with increasing phase
    valence_arc_score: float = 0.0  # recognizable arc quality
    narrative_coherence: float = 0.0

    # Comparison metrics
    lmd_vs_random_ratio: float = 0.0  # narrative coherence vs random

    def is_alive(self) -> bool:
        """Check if system shows signs of 'aliveness'."""
        return (
            self.aliveness_score > 0.3 and
            self.energy_variance > 0.05 and
            self.alive_ratio > 0.5
        )

    def has_coupling(self) -> bool:
        """Check if coupling is working."""
        return (
            self.coupling_coherence > 1.5 and
            self.mean_coupling > 0.2
        )

    def generates_narratives(self) -> bool:
        """Check if narrative generation is working."""
        return (
            self.phase_progression_score > 0.6 and
            self.narrative_coherence > 0.4 and
            self.lmd_vs_random_ratio > 1.5
        )

    def all_tests_pass(self) -> bool:
        """Check if all validation criteria pass."""
        return self.is_alive() and self.has_coupling() and self.generates_narratives()

    def summary(self) -> str:
        """Get summary string."""
        status = "PASS" if self.all_tests_pass() else "FAIL"
        return (
            f"LMD Validation: {status}\n"
            f"  Aliveness: {self.aliveness_score:.3f} (alive={self.is_alive()})\n"
            f"  Coupling: coherence={self.coupling_coherence:.2f} (working={self.has_coupling()})\n"
            f"  Narrative: coherence={self.narrative_coherence:.3f}, vs_random={self.lmd_vs_random_ratio:.2f}x "
            f"(working={self.generates_narratives()})"
        )


class LMDToySystem:
    """Toy-scale Living Memory Dynamics system for validation.

    Provides easy setup, simulation, and measurement of LMD.
    """

    def __init__(self, config: Optional[LMDConfig] = None):
        """Initialize the toy system.

        Args:
            config: LMD configuration (defaults to toy_scale)
        """
        self.config = config or LMDConfig.toy_scale()
        self.dynamics = LMDDynamics(self.config)
        self.synthesizer = NarrativeSynthesizer(
            self.config,
            self.dynamics.coupling,
            self.dynamics.metabolism
        )

        self.memories: List[LivingMemory] = []
        self.next_id = 0

        # Tracking
        self.simulation_history: List[Dict] = []

    def create_memory(
        self,
        content: Optional[torch.Tensor] = None,
        valence: Optional[ValenceTrajectory] = None,
        phase: float = 0.0,
        energy: float = 1.0,
        label: Optional[str] = None
    ) -> LivingMemory:
        """Create and add a new memory to the system."""
        if content is None:
            content = torch.randn(self.config.content_dim)

        if valence is None:
            valence = ValenceTrajectory.random(self.config.valence_points)

        memory = LivingMemory(
            id=self.next_id,
            content=content,
            valence=valence,
            phase=phase,
            energy=energy,
            created_at=self.dynamics.timestep,
            label=label
        )

        self.next_id += 1
        self.memories.append(memory)
        self.dynamics.metabolism.births += 1

        return memory

    def create_cluster(
        self,
        n_memories: int,
        base_content: torch.Tensor,
        valence_type: str = "random",
        content_noise: float = 0.3,
        label_prefix: str = "cluster"
    ) -> List[LivingMemory]:
        """Create a cluster of similar memories.

        Args:
            n_memories: Number of memories in cluster
            base_content: Base content vector
            valence_type: Type of valence ("positive", "negative", "redemption", "tragedy", "random")
            content_noise: Amount of noise to add to content
            label_prefix: Prefix for memory labels
        """
        memories = []

        for i in range(n_memories):
            # Add noise to content
            content = base_content + torch.randn_like(base_content) * content_noise

            # Create valence based on type
            if valence_type == "positive":
                valence = ValenceTrajectory.constant(random.uniform(0.3, 0.8))
            elif valence_type == "negative":
                valence = ValenceTrajectory.constant(random.uniform(-0.8, -0.3))
            elif valence_type == "redemption":
                valence = ValenceTrajectory.redemption()
            elif valence_type == "tragedy":
                valence = ValenceTrajectory.tragedy()
            else:
                valence = ValenceTrajectory.random()

            # Random initial phase
            phase = random.uniform(0, 2 * math.pi)

            memory = self.create_memory(
                content=content,
                valence=valence,
                phase=phase,
                label=f"{label_prefix}_{i}"
            )
            memories.append(memory)

        return memories

    def create_diverse_memories(self, n_memories: int) -> List[LivingMemory]:
        """Create a diverse set of memories for testing."""
        memories = []

        # Create clusters with different characteristics
        n_per_cluster = n_memories // 4

        # Positive cluster
        base_positive = torch.randn(self.config.content_dim)
        memories.extend(self.create_cluster(
            n_per_cluster, base_positive, "positive", label_prefix="happy"
        ))

        # Negative cluster
        base_negative = torch.randn(self.config.content_dim)
        memories.extend(self.create_cluster(
            n_per_cluster, base_negative, "negative", label_prefix="sad"
        ))

        # Redemption arcs
        base_redemption = torch.randn(self.config.content_dim)
        memories.extend(self.create_cluster(
            n_per_cluster, base_redemption, "redemption", label_prefix="growth"
        ))

        # Random/diverse
        remaining = n_memories - len(memories)
        for i in range(remaining):
            self.create_memory(label=f"random_{i}")

        return self.memories

    def run_simulation(
        self,
        n_steps: int = 100,
        activation_probability: float = 0.1
    ) -> List[Dict]:
        """Run simulation for multiple steps.

        Args:
            n_steps: Number of timesteps
            activation_probability: Probability of random activation per memory per step
        """
        history = []

        for _step in range(n_steps):
            # Random activations
            activations = {}
            for m in self.memories:
                if random.random() < activation_probability:
                    activations[m.id] = random.uniform(0.5, 1.0)

            # Step dynamics
            metrics = self.dynamics.step(self.memories, activations=activations)
            history.append(metrics)

            # Prune dead memories
            self.memories, dead = self.dynamics.metabolism.prune_dead_memories(self.memories)

        self.simulation_history.extend(history)
        return history

    def generate_narrative(
        self,
        seed_id: Optional[int] = None,
        target_length: int = 10
    ) -> GeneratedNarrative:
        """Generate a narrative from memories.

        Args:
            seed_id: ID of seed memory (random if None)
            target_length: Target narrative length
        """
        if not self.memories:
            raise ValueError("No memories in system")

        if seed_id is not None:
            seed = next((m for m in self.memories if m.id == seed_id), None)
            if seed is None:
                raise ValueError(f"Memory {seed_id} not found")
        else:
            # Pick highest energy memory
            seed = max(self.memories, key=lambda m: m.energy)

        return self.synthesizer.generate_narrative(
            seed, self.memories, target_length
        )

    def compute_metrics(self) -> LMDMetrics:
        """Compute comprehensive validation metrics."""
        metrics = LMDMetrics()

        if not self.memories:
            return metrics

        # === Aliveness Metrics ===
        metabolism_stats = self.dynamics.metabolism.metabolism_statistics(self.memories)
        metrics.aliveness_score = metabolism_stats["aliveness_score"]
        metrics.energy_variance = metabolism_stats["energy_variance"]
        metrics.alive_ratio = sum(1 for m in self.memories if m.is_alive) / len(self.memories)

        # === Coupling Metrics ===
        coupling_stats = self.dynamics.coupling.coupling_statistics(self.memories)
        metrics.mean_coupling = coupling_stats["mean_coupling"]
        metrics.coupling_density = coupling_stats["density"]

        # Compute intra vs inter cluster coupling
        clusters = self.dynamics.coupling.cluster_by_coupling(self.memories, threshold=0.3)
        if len(clusters) > 1:
            # Compute intra-cluster coupling
            intra_couplings = []
            for cluster in clusters:
                if len(cluster) > 1:
                    for i, m1 in enumerate(cluster):
                        for m2 in cluster[i + 1:]:
                            intra_couplings.append(self.dynamics.coupling.get_coupling(m1, m2))

            # Compute inter-cluster coupling
            inter_couplings = []
            for i, c1 in enumerate(clusters):
                for c2 in clusters[i + 1:]:
                    for m1 in c1:
                        for m2 in c2:
                            inter_couplings.append(self.dynamics.coupling.get_coupling(m1, m2))

            intra_mean = sum(intra_couplings) / len(intra_couplings) if intra_couplings else 0
            inter_mean = sum(inter_couplings) / len(inter_couplings) if inter_couplings else 0.001

            metrics.coupling_coherence = intra_mean / inter_mean if inter_mean > 0 else intra_mean * 10
        else:
            metrics.coupling_coherence = 1.0

        # === Narrative Metrics ===
        try:
            narrative = self.generate_narrative(target_length=min(10, len(self.memories)))
            metrics.narrative_coherence = narrative.coherence_score

            # Phase progression
            phases = narrative.phase_progression
            if len(phases) > 1:
                increases = sum(1 for i in range(1, len(phases)) if phases[i] >= phases[i - 1])
                metrics.phase_progression_score = increases / (len(phases) - 1)

            # Valence arc quality
            arc = narrative.valence_arc
            if arc:
                valence_range = max(arc) - min(arc)
                metrics.valence_arc_score = min(1.0, valence_range / 0.5)

            # Compare to random
            comparison = self.synthesizer.compare_to_random(narrative, self.memories, n_random=10)
            metrics.lmd_vs_random_ratio = comparison["improvement_ratio"]

        except Exception as e:
            print(f"Warning: Narrative generation failed: {e}")

        return metrics

    def run_validation_experiment(
        self,
        n_memories: int = 20,
        n_steps: int = 100
    ) -> Tuple[LMDMetrics, Dict]:
        """Run a complete validation experiment.

        Args:
            n_memories: Number of memories to create
            n_steps: Number of simulation steps

        Returns:
            (metrics, detailed_results)
        """
        # Reset
        self.memories = []
        self.next_id = 0
        self.simulation_history = []
        self.dynamics = LMDDynamics(self.config)
        self.synthesizer = NarrativeSynthesizer(
            self.config,
            self.dynamics.coupling,
            self.dynamics.metabolism
        )

        # Create memories
        self.create_diverse_memories(n_memories)

        # Run simulation
        history = self.run_simulation(n_steps)

        # Compute metrics
        metrics = self.compute_metrics()

        # Generate multiple narratives
        narratives = self.synthesizer.generate_multiple_narratives(
            self.memories, n_narratives=5
        )

        # Compile detailed results
        results = {
            "config": {
                "n_memories": n_memories,
                "n_steps": n_steps,
                "content_dim": self.config.content_dim,
            },
            "final_state": self.dynamics.get_dynamics_summary(self.memories),
            "simulation_history": history,
            "narratives": [
                {
                    "length": n.total_duration,
                    "coherence": n.coherence_score,
                    "arc_type": n.arc_type,
                    "valence_range": max(n.valence_arc) - min(n.valence_arc) if n.valence_arc else 0,
                }
                for n in narratives
            ],
            "metrics": {
                "aliveness_score": metrics.aliveness_score,
                "energy_variance": metrics.energy_variance,
                "alive_ratio": metrics.alive_ratio,
                "coupling_coherence": metrics.coupling_coherence,
                "mean_coupling": metrics.mean_coupling,
                "narrative_coherence": metrics.narrative_coherence,
                "lmd_vs_random_ratio": metrics.lmd_vs_random_ratio,
            },
            "validation": {
                "is_alive": metrics.is_alive(),
                "has_coupling": metrics.has_coupling(),
                "generates_narratives": metrics.generates_narratives(),
                "all_pass": metrics.all_tests_pass(),
            }
        }

        return metrics, results

    def print_status(self):
        """Print current system status."""
        print("\n=== LMD Toy System Status ===")
        print(f"Memories: {len(self.memories)}")
        print(f"Timestep: {self.dynamics.timestep}")

        if self.memories:
            summary = self.dynamics.get_dynamics_summary(self.memories)
            print(f"Alive: {summary['alive_memories']}/{summary['total_memories']}")
            print(f"Phase coherence: {summary['phase_coherence']:.3f}")

            print("\nNarrative distribution:")
            for phase, count in summary['narrative_distribution'].items():
                if count > 0:
                    print(f"  {phase}: {count}")

            print("\nMetabolism:")
            for state, count in summary['metabolism']['state_distribution'].items():
                if count > 0:
                    print(f"  {state}: {count}")
