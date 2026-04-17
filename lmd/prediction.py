"""Prediction Task - Test LMD's ability to predict/generate narrative continuations.

The key test: Given a partial story encoded as memories, can LMD:
1. Generate a coherent "internal movie" continuation?
2. Predict the emotional arc of what comes next?
3. Maintain narrative coherence across generated frames?

This validates that LMD isn't just storing - it's GENERATING.

Invented by Joshua R. Thomas, January 2026.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import LMDConfig
from .dynamics import LMDDynamics
from .living_memory import LivingMemory
from .narrative import GeneratedNarrative, NarrativeSynthesizer
from .story_encoder import EncodedStory


@dataclass
class PredictionResult:
    """Result of a narrative prediction task."""

    # The generated continuation
    generated_narrative: GeneratedNarrative

    # Ground truth (if available)
    ground_truth_memories: List[LivingMemory]
    ground_truth_valences: List[float]

    # Prediction quality metrics
    valence_mae: float = 0.0  # Mean absolute error on valence prediction
    phase_progression_accuracy: float = 0.0  # Did we predict phase increases?
    content_similarity: float = 0.0  # Cosine sim to actual next memories
    arc_type_match: bool = False  # Did we predict correct arc type?

    # Coherence metrics
    internal_coherence: float = 0.0  # Coherence of generated sequence
    transition_smoothness: float = 0.0  # How smooth are the transitions

    @property
    def overall_score(self) -> float:
        """Combined prediction quality score."""
        arc_bonus = 0.2 if self.arc_type_match else 0.0
        # Normalize valence MAE (max possible is 2.0 since valence in [-1, 1])
        valence_score = max(0.0, 1.0 - self.valence_mae / 2.0)
        return (
            0.3 * valence_score +
            0.2 * self.phase_progression_accuracy +
            0.2 * self.content_similarity +
            0.2 * self.internal_coherence +
            0.1 * self.transition_smoothness +
            arc_bonus
        )

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Prediction Quality: {self.overall_score:.3f}\n"
            f"  Valence MAE: {self.valence_mae:.3f}\n"
            f"  Phase accuracy: {self.phase_progression_accuracy:.3f}\n"
            f"  Content similarity: {self.content_similarity:.3f}\n"
            f"  Arc match: {self.arc_type_match}\n"
            f"  Generated arc: {self.generated_narrative.arc_type}\n"
            f"  Internal coherence: {self.internal_coherence:.3f}"
        )


class NarrativePredictor:
    """Predicts narrative continuations using LMD.

    Given partial memories from a story, generates what comes next
    and evaluates against ground truth.
    """

    def __init__(
        self,
        config: LMDConfig,
        dynamics: LMDDynamics,
        synthesizer: NarrativeSynthesizer
    ):
        self.config = config
        self.dynamics = dynamics
        self.synthesizer = synthesizer

    def predict_continuation(
        self,
        context_memories: List[LivingMemory],
        n_frames: int = 5,
        warm_up_steps: int = 20
    ) -> GeneratedNarrative:
        """Generate a predicted continuation from context.

        IMPROVED: Better seed selection based on:
        - High phase (late in story)
        - High energy (most alive)
        - Valence trajectory direction (for arc continuity)

        Args:
            context_memories: The memories to condition on
            n_frames: How many frames to generate
            warm_up_steps: Steps to run dynamics before generating

        Returns:
            Generated narrative continuation
        """
        # Make a copy to avoid modifying originals
        memories = [
            LivingMemory(
                id=m.id,
                content=m.content.clone(),
                valence=m.valence,
                phase=m.phase,
                energy=m.energy,
                created_at=m.created_at,
                label=m.label
            )
            for m in context_memories
        ]

        # Warm up the dynamics
        for _ in range(warm_up_steps):
            self.dynamics.step(memories)

        # === IMPROVED: Smart seed selection ===
        # Score each memory as potential seed
        scored_seeds = []
        for m in memories:
            if not m.is_alive:
                continue

            # Phase score (prefer late in story for continuation)
            phase_score = m.phase / (2 * 3.14159)  # Normalize to [0, 1]

            # Energy score
            energy_score = m.energy

            # Valence momentum (prefer memories trending in a direction)
            valence_score = abs(m.current_valence)  # Strong emotion = good seed

            combined = 0.4 * phase_score + 0.3 * energy_score + 0.3 * valence_score
            scored_seeds.append((m, combined))

        if not scored_seeds:
            # Fallback to any memory
            seed = memories[0] if memories else context_memories[0]
        else:
            # Pick best seed
            scored_seeds.sort(key=lambda x: x[1], reverse=True)
            seed = scored_seeds[0][0]

        # Generate continuation
        narrative = self.synthesizer.generate_narrative(
            seed, memories, target_length=n_frames
        )

        return narrative

    def evaluate_prediction(
        self,
        encoded_story: EncodedStory,
        context_ratio: float = 0.6,
        prediction_length: int = 5
    ) -> PredictionResult:
        """Evaluate prediction quality on a real story.

        Args:
            encoded_story: The full encoded story
            context_ratio: Fraction of story to use as context
            prediction_length: How many frames to predict

        Returns:
            PredictionResult with quality metrics
        """
        n_memories = len(encoded_story.memories)
        split_point = int(n_memories * context_ratio)

        if split_point < 2 or n_memories - split_point < 2:
            raise ValueError("Story too short for prediction task")

        # Split into context and ground truth
        context_memories = encoded_story.memories[:split_point]
        ground_truth = encoded_story.memories[split_point:split_point + prediction_length]
        ground_truth_valences = encoded_story.valences[split_point:split_point + prediction_length]

        # Generate prediction
        generated = self.predict_continuation(
            context_memories,
            n_frames=min(prediction_length, len(ground_truth)),
            warm_up_steps=30
        )

        # Evaluate
        result = PredictionResult(
            generated_narrative=generated,
            ground_truth_memories=ground_truth,
            ground_truth_valences=ground_truth_valences
        )

        # Valence MAE
        gen_valences = generated.valence_arc
        if gen_valences and ground_truth_valences:
            n_compare = min(len(gen_valences), len(ground_truth_valences))
            mae = sum(
                abs(g - t)
                for g, t in zip(gen_valences[:n_compare], ground_truth_valences[:n_compare])
            ) / n_compare
            result.valence_mae = mae

        # Phase progression accuracy
        gen_phases = generated.phase_progression
        if len(gen_phases) > 1:
            increases = sum(1 for i in range(1, len(gen_phases)) if gen_phases[i] >= gen_phases[i-1])
            result.phase_progression_accuracy = increases / (len(gen_phases) - 1)

        # Content similarity (average cosine sim to ground truth)
        if ground_truth:
            similarities = []
            for gen_frame in generated.frames:
                max_sim = 0.0
                for gt_mem in ground_truth:
                    sim = torch.nn.functional.cosine_similarity(
                        gen_frame.memory.content.unsqueeze(0),
                        gt_mem.content.unsqueeze(0)
                    ).item()
                    max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            result.content_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Arc type match
        gt_arc = self._infer_arc_type(ground_truth_valences)
        result.arc_type_match = (generated.arc_type == gt_arc)

        # Internal coherence
        result.internal_coherence = generated.coherence_score

        # Transition smoothness
        if len(generated.frames) > 1:
            valence_changes = [
                abs(generated.frames[i].valence - generated.frames[i-1].valence)
                for i in range(1, len(generated.frames))
            ]
            result.transition_smoothness = 1.0 / (1.0 + sum(valence_changes))

        return result

    def _infer_arc_type(self, valences: List[float]) -> str:
        """Infer arc type from valence sequence."""
        if not valences or len(valences) < 2:
            return "flat"

        start = valences[0]
        end = valences[-1]
        peak = max(valences)
        trough = min(valences)
        delta = end - start

        if abs(delta) < 0.2:
            if peak - start > 0.3:
                return "climax"
            elif start - trough > 0.3:
                return "valley"
            return "flat"
        elif delta > 0.3:
            return "redemption"
        elif delta < -0.3:
            return "tragedy"
        return "drift"

    def compare_to_random(
        self,
        encoded_story: EncodedStory,
        n_trials: int = 10,
        context_ratio: float = 0.6
    ) -> Dict[str, float]:
        """Compare LMD prediction to random baseline.

        Args:
            encoded_story: The story to test on
            n_trials: Number of random trials
            context_ratio: Fraction for context

        Returns:
            Comparison metrics
        """
        import random

        n_memories = len(encoded_story.memories)
        split_point = int(n_memories * context_ratio)
        prediction_length = min(5, n_memories - split_point)

        # LMD prediction
        lmd_result = self.evaluate_prediction(
            encoded_story, context_ratio, prediction_length
        )

        # Random predictions
        random_scores = []
        context_memories = encoded_story.memories[:split_point]

        for _ in range(n_trials):
            # Random selection of memories
            random_selection = random.sample(context_memories, min(prediction_length, len(context_memories)))

            # Compute "score" based on valence variance (lower = more random)
            valences = [m.current_valence for m in random_selection]
            if len(valences) > 1:
                mean_v = sum(valences) / len(valences)
                var = sum((v - mean_v)**2 for v in valences) / len(valences)
                random_scores.append(1.0 / (1.0 + var))
            else:
                random_scores.append(0.5)

        random_mean = sum(random_scores) / len(random_scores) if random_scores else 0.5

        return {
            "lmd_score": lmd_result.overall_score,
            "random_mean_score": random_mean,
            "improvement_ratio": lmd_result.overall_score / random_mean if random_mean > 0 else float('inf'),
            "valence_mae": lmd_result.valence_mae,
            "arc_matched": lmd_result.arc_type_match,
            "lmd_arc": lmd_result.generated_narrative.arc_type,
        }


def run_prediction_benchmark(
    encoded_story: EncodedStory,
    config: Optional[LMDConfig] = None
) -> Dict[str, float]:
    """Run full prediction benchmark on a story."""
    config = config or LMDConfig.toy_scale()

    dynamics = LMDDynamics(config)
    synthesizer = NarrativeSynthesizer(config, dynamics.coupling, dynamics.metabolism)
    predictor = NarrativePredictor(config, dynamics, synthesizer)

    # Run prediction
    result = predictor.evaluate_prediction(encoded_story, context_ratio=0.6)

    # Compare to random
    comparison = predictor.compare_to_random(encoded_story, n_trials=20)

    return {
        "prediction_score": result.overall_score,
        "valence_mae": result.valence_mae,
        "phase_accuracy": result.phase_progression_accuracy,
        "content_similarity": result.content_similarity,
        "arc_match": result.arc_type_match,
        "generated_arc": result.generated_narrative.arc_type,
        "internal_coherence": result.internal_coherence,
        "lmd_vs_random": comparison["improvement_ratio"],
    }
