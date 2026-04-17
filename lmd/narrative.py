"""Narrative Synthesis - Generating internal movies from living memories.

The key innovation: memories don't just store and retrieve, they GENERATE
coherent narratives by:
1. Seeding with an initial memory
2. Propagating activation through coupling field
3. Aligning narrative phases
4. Synthesizing temporal sequence

This creates an "internal movie" - a living story built from memory fragments.

Invented by Joshua R. Thomas, January 2026.
"""

from dataclasses import dataclass
from typing import Dict, List

from .config import LMDConfig
from .coupling import CouplingField
from .living_memory import LivingMemory
from .metabolism import MemoryMetabolism


@dataclass
class NarrativeFrame:
    """A single frame in a generated narrative."""
    memory: LivingMemory
    position: int              # Position in narrative sequence
    phase: float              # Narrative phase at this frame
    valence: float            # Emotional valence at this frame
    coupling_to_prev: float   # Coupling strength to previous frame
    contribution: float       # How much this memory contributes


@dataclass
class GeneratedNarrative:
    """A complete generated narrative (internal movie)."""
    frames: List[NarrativeFrame]
    seed_memory: LivingMemory
    total_duration: int
    coherence_score: float
    valence_arc: List[float]
    phase_progression: List[float]

    @property
    def memories(self) -> List[LivingMemory]:
        """Get the sequence of memories."""
        return [f.memory for f in self.frames]

    @property
    def arc_type(self) -> str:
        """Classify the narrative arc type."""
        if not self.valence_arc:
            return "empty"

        start = self.valence_arc[0]
        end = self.valence_arc[-1]
        peak = max(self.valence_arc)
        trough = min(self.valence_arc)

        delta = end - start

        if abs(delta) < 0.2:
            if peak - start > 0.3:
                return "climax"
            elif start - trough > 0.3:
                return "valley"
            else:
                return "flat"
        elif delta > 0.3:
            return "redemption"
        elif delta < -0.3:
            return "tragedy"
        else:
            return "drift"

    def summary(self) -> str:
        """Get a text summary of the narrative."""
        return (
            f"Narrative: {len(self.frames)} frames, "
            f"arc={self.arc_type}, "
            f"coherence={self.coherence_score:.2f}, "
            f"valence: {self.valence_arc[0]:.2f} -> {self.valence_arc[-1]:.2f}"
        )


class NarrativeSynthesizer:
    """Generates internal narratives (movies) from living memories.

    The synthesizer creates coherent stories by:
    1. Starting from a seed memory
    2. Following coupling connections
    3. Ordering by narrative phase
    4. Ensuring emotional coherence
    """

    def __init__(self, config: LMDConfig, coupling: CouplingField, metabolism: MemoryMetabolism):
        self.config = config
        self.coupling = coupling
        self.metabolism = metabolism

    def generate_narrative(
        self,
        seed_memory: LivingMemory,
        all_memories: List[LivingMemory],
        target_length: int = 10,
        min_coupling: float = 0.2
    ) -> GeneratedNarrative:
        """Generate a narrative starting from seed memory.

        IMPROVED: Uses greedy path selection that balances:
        - Coupling continuity (stay connected)
        - Phase progression (advance the story)
        - Valence continuity (smooth emotional transitions)

        Args:
            seed_memory: The memory to start the narrative from
            all_memories: All available memories
            target_length: Target number of frames
            min_coupling: Minimum coupling to include a memory

        Returns:
            Generated narrative with frames ordered by story structure
        """
        # === 1. Activate seed and propagate ===
        activated = self.metabolism.propagate_activation(
            seed_memory,
            all_memories,
            strength=1.0,
            decay_per_hop=0.6,
            max_hops=4,
            timestep=0
        )

        # Filter by energy threshold
        candidates = [
            m for m in activated
            if m.energy > self.config.movie_activation_threshold
        ]

        if not candidates:
            candidates = [seed_memory]

        # === 2. IMPROVED: Greedy path selection ===
        # Start with seed, greedily pick next by coupling + phase + valence
        selected = [seed_memory]
        used_ids = {seed_memory.id}
        current = seed_memory

        while len(selected) < target_length:
            best_next = None
            best_score = -float('inf')

            for candidate in candidates:
                if candidate.id in used_ids:
                    continue

                # Coupling to current (want high)
                coupling_score = self.coupling.get_coupling(current, candidate)

                # Phase progression (want forward movement)
                phase_diff = candidate.phase - current.phase
                # Normalize to [0, 1] - prefer small positive advancement
                if phase_diff < 0:
                    phase_diff += 2 * 3.14159  # Wrap around
                phase_score = 1.0 / (1.0 + abs(phase_diff - 0.5))  # Prefer ~0.5 radian steps

                # Valence continuity (want smooth transitions)
                valence_diff = abs(candidate.current_valence - current.current_valence)
                valence_score = 1.0 / (1.0 + valence_diff)

                # Energy (prefer alive memories)
                energy_score = candidate.energy

                # Combined score (weighted)
                combined = (
                    0.35 * coupling_score +  # Coupling most important
                    0.25 * phase_score +
                    0.25 * valence_score +
                    0.15 * energy_score
                )

                if combined > best_score:
                    best_score = combined
                    best_next = candidate

            if best_next is None:
                break  # No more valid candidates

            selected.append(best_next)
            used_ids.add(best_next.id)
            current = best_next

        # === 3. Final ordering: sort by phase for narrative flow ===
        # But only if it doesn't break coupling too much
        # Try phase-sorted order and compare coherence
        phase_sorted = sorted(selected, key=lambda m: m.phase)

        # Compute coupling continuity of greedy vs phase-sorted
        greedy_coupling = self._path_coupling(selected)
        phase_coupling = self._path_coupling(phase_sorted)

        # Use whichever has better coupling (greedy usually wins)
        if greedy_coupling >= phase_coupling * 0.8:  # Greedy within 80%
            selected = selected  # Keep greedy order
        else:
            selected = phase_sorted  # Phase order is significantly better

        # === 4. Build frames ===
        frames = []
        prev_memory = None

        for i, memory in enumerate(selected):
            coupling_to_prev = 0.0
            if prev_memory is not None:
                coupling_to_prev = self.coupling.get_coupling(prev_memory, memory)

            frame = NarrativeFrame(
                memory=memory,
                position=i,
                phase=memory.phase,
                valence=memory.current_valence,
                coupling_to_prev=coupling_to_prev,
                contribution=memory.energy / sum(m.energy for m in selected)
            )
            frames.append(frame)
            prev_memory = memory

        # === 5. Compute narrative metrics ===
        valence_arc = [f.valence for f in frames]
        phase_progression = [f.phase for f in frames]
        coherence = self._compute_coherence(frames)

        return GeneratedNarrative(
            frames=frames,
            seed_memory=seed_memory,
            total_duration=len(frames),
            coherence_score=coherence,
            valence_arc=valence_arc,
            phase_progression=phase_progression
        )

    def _compute_coherence(self, frames: List[NarrativeFrame]) -> float:
        """Compute narrative coherence score.

        Based on:
        - Sequential coupling (adjacent frames should be coupled)
        - Phase progression (phases should generally increase)
        - Valence smoothness (emotional transitions should be gradual)
        """
        if len(frames) < 2:
            return 1.0

        # Sequential coupling score
        coupling_scores = [f.coupling_to_prev for f in frames[1:]]
        coupling_mean = sum(coupling_scores) / len(coupling_scores) if coupling_scores else 0

        # Phase progression score
        phase_increases = 0
        for i in range(1, len(frames)):
            if frames[i].phase >= frames[i - 1].phase:
                phase_increases += 1
        phase_score = phase_increases / (len(frames) - 1)

        # Valence smoothness (inverse of variance of changes)
        valence_changes = []
        for i in range(1, len(frames)):
            change = abs(frames[i].valence - frames[i - 1].valence)
            valence_changes.append(change)

        if valence_changes:
            smoothness = 1.0 / (1.0 + sum(valence_changes) / len(valence_changes))
        else:
            smoothness = 1.0

        # Combined score
        coherence = 0.4 * coupling_mean + 0.3 * phase_score + 0.3 * smoothness

        return coherence

    def _path_coupling(self, memories: List[LivingMemory]) -> float:
        """Compute total coupling along a path of memories."""
        if len(memories) < 2:
            return 1.0

        total = 0.0
        for i in range(1, len(memories)):
            total += self.coupling.get_coupling(memories[i-1], memories[i])

        return total / (len(memories) - 1)

    def generate_multiple_narratives(
        self,
        all_memories: List[LivingMemory],
        n_narratives: int = 5,
        target_length: int = 10
    ) -> List[GeneratedNarrative]:
        """Generate multiple narratives from different seeds.

        Selects diverse seed memories to create varied narratives.
        """
        if not all_memories:
            return []

        narratives = []

        # Select diverse seeds (high energy, varied valence)
        candidates = sorted(all_memories, key=lambda m: m.energy, reverse=True)
        seeds_used = set()

        for candidate in candidates:
            if len(narratives) >= n_narratives:
                break

            # Check if sufficiently different from used seeds
            is_diverse = True
            for seed_id in seeds_used:
                seed = next((m for m in all_memories if m.id == seed_id), None)
                if seed and candidate.similarity(seed) > 0.7:
                    is_diverse = False
                    break

            if is_diverse and candidate.is_alive:
                narrative = self.generate_narrative(
                    candidate, all_memories, target_length
                )
                narratives.append(narrative)
                seeds_used.add(candidate.id)

        return narratives

    def narrative_statistics(
        self,
        narratives: List[GeneratedNarrative]
    ) -> Dict[str, float]:
        """Compute statistics over multiple narratives."""
        if not narratives:
            return {}

        return {
            "n_narratives": len(narratives),
            "mean_length": sum(n.total_duration for n in narratives) / len(narratives),
            "mean_coherence": sum(n.coherence_score for n in narratives) / len(narratives),
            "arc_distribution": {
                arc: sum(1 for n in narratives if n.arc_type == arc)
                for arc in ["redemption", "tragedy", "climax", "valley", "flat", "drift"]
            },
            "mean_valence_range": sum(
                max(n.valence_arc) - min(n.valence_arc) for n in narratives
            ) / len(narratives),
        }

    def compare_to_random(
        self,
        narrative: GeneratedNarrative,
        all_memories: List[LivingMemory],
        n_random: int = 10
    ) -> Dict[str, float]:
        """Compare narrative coherence to random sequences.

        This validates that LMD generates meaningful narratives,
        not just random memory sequences.
        """
        import random

        random_coherences = []
        for _ in range(n_random):
            # Generate random sequence of same length
            random_memories = random.sample(
                [m for m in all_memories if m.is_alive],
                min(narrative.total_duration, len(all_memories))
            )

            # Build frames
            frames = []
            prev = None
            for i, m in enumerate(random_memories):
                coupling = self.coupling.get_coupling(prev, m) if prev else 0
                frame = NarrativeFrame(
                    memory=m,
                    position=i,
                    phase=m.phase,
                    valence=m.current_valence,
                    coupling_to_prev=coupling,
                    contribution=1.0 / len(random_memories)
                )
                frames.append(frame)
                prev = m

            random_coherences.append(self._compute_coherence(frames))

        mean_random = sum(random_coherences) / len(random_coherences)
        std_random = (sum((c - mean_random) ** 2 for c in random_coherences) / len(random_coherences)) ** 0.5

        return {
            "lmd_coherence": narrative.coherence_score,
            "random_mean_coherence": mean_random,
            "random_std_coherence": std_random,
            "improvement_ratio": narrative.coherence_score / mean_random if mean_random > 0 else float('inf'),
            "z_score": (narrative.coherence_score - mean_random) / std_random if std_random > 0 else 0,
        }
