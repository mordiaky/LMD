"""Plausibility Field - Learned coherence constraints for imagination.

Imagination isn't random - it respects learned physical/semantic constraints:
- Dragons can have wings (transplant from birds/butterflies)
- But not square circles (violates geometry)
- Entities can sit on couches (learned affordance)
- But not swim through concrete (violates physics)

The Plausibility Field learns these constraints from memory patterns.

Invented by Joshua R. Thomas, January 2026.
"""

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch

from .imagination import SlotType, StructuredMemory, Transform, TransformType
from .living_memory import LivingMemory

# Epsilon for numerical stability
EPS = 1e-8


@dataclass
class PlausibilityScore:
    """Score indicating how plausible an imagined entity is."""

    # Component scores (0-1, higher = more plausible)
    structural_coherence: float = 1.0   # Parts fit together
    semantic_coherence: float = 1.0     # Meaning makes sense
    physical_coherence: float = 1.0     # Respects physics
    relational_coherence: float = 1.0   # Relations are valid

    # Novelty (0-1, higher = more novel)
    novelty: float = 0.0

    # Combined scores
    @property
    def total_plausibility(self) -> float:
        """Overall plausibility score."""
        return (
            0.3 * self.structural_coherence +
            0.3 * self.semantic_coherence +
            0.2 * self.physical_coherence +
            0.2 * self.relational_coherence
        )

    @property
    def creative_value(self) -> float:
        """Balance of novelty and plausibility.

        High value = novel AND plausible (the sweet spot)
        """
        # Geometric mean emphasizes balance
        return math.sqrt(self.novelty * self.total_plausibility)

    @property
    def is_acceptable(self) -> bool:
        """Minimum threshold for acceptability."""
        return self.total_plausibility > 0.3

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Plausibility: {self.total_plausibility:.3f}\n"
            f"  Structural: {self.structural_coherence:.3f}\n"
            f"  Semantic: {self.semantic_coherence:.3f}\n"
            f"  Physical: {self.physical_coherence:.3f}\n"
            f"  Relational: {self.relational_coherence:.3f}\n"
            f"  Novelty: {self.novelty:.3f}\n"
            f"  Creative Value: {self.creative_value:.3f}"
        )


class PlausibilityField:
    """Learned coherence constraints for imagination.

    Learns from memory patterns what combinations are plausible:
    - Which parts go together (structural)
    - Which concepts combine (semantic)
    - What physical laws apply (physical)
    - How entities can relate (relational)

    Thread-safe and bounded to prevent memory leaks.
    """

    def __init__(
        self,
        content_dim: int = 32,
        max_cooccurrence_pairs: int = 500,
        max_observed_combinations: int = 200
    ):
        self.content_dim = content_dim
        self.max_cooccurrence_pairs = max_cooccurrence_pairs
        self.max_observed_combinations = max_observed_combinations

        self._lock = threading.RLock()

        # Learned constraints (in practice, trained on memories)
        # For prototype, use heuristics

        # Co-occurrence matrix: which slots appear together
        self.slot_cooccurrence: Dict[Tuple[str, str], float] = {}

        # Compatibility matrix: which slot values are compatible
        self.value_compatibility: Dict[str, torch.Tensor] = {}

        # Physical constraints (simplified)
        self.physical_rules: List[callable] = []

        # Track what we've seen
        self.observed_combinations: Set[frozenset] = set()
        self.memory_centroid: Optional[torch.Tensor] = None
        self.memory_variance: float = 1.0

        # Statistics for incremental learning
        self._n_observations: int = 0
        self._centroid_sum: Optional[torch.Tensor] = None
        self._variance_sum: float = 0.0

    def learn_from_memories(self, memories: List[LivingMemory]) -> None:
        """Learn plausibility constraints from observed memories."""
        if not memories:
            return

        with self._lock:
            # Compute memory statistics
            contents = torch.stack([m.content for m in memories])
            self.memory_centroid = contents.mean(0)
            self.memory_variance = max(contents.var().item(), EPS)

            # Initialize incremental learning state
            self._n_observations = len(memories)
            self._centroid_sum = contents.sum(0)

    def learn_from_structured(self, structured_memories: List[StructuredMemory]) -> None:
        """Learn from structured memory representations.

        This method updates the plausibility field incrementally, closing
        the feedback loop from ideation back to plausibility scoring.
        """
        if not structured_memories:
            return

        with self._lock:
            embeddings = []

            for mem in structured_memories:
                # Track slot co-occurrence
                slot_names = list(mem.slots.keys())
                for i, s1 in enumerate(slot_names):
                    for s2 in slot_names[i+1:]:
                        key = (min(s1, s2), max(s1, s2))
                        self.slot_cooccurrence[key] = self.slot_cooccurrence.get(key, 0) + 1

                # Track combinations
                self.observed_combinations.add(frozenset(slot_names))

                # Collect embeddings for centroid update
                embedding = mem.to_embedding(self.content_dim)
                if not torch.isnan(embedding).any():
                    embeddings.append(embedding)

            # Update memory centroid incrementally
            if embeddings:
                new_embeddings = torch.stack(embeddings)
                new_sum = new_embeddings.sum(0)
                new_count = len(embeddings)

                if self._centroid_sum is None:
                    self._centroid_sum = new_sum
                    self._n_observations = new_count
                else:
                    self._centroid_sum = self._centroid_sum + new_sum
                    self._n_observations += new_count

                # Update centroid
                if self._n_observations > 0:
                    self.memory_centroid = self._centroid_sum / self._n_observations

                # Update variance (simplified: use new embeddings only)
                if self.memory_centroid is not None and len(embeddings) > 1:
                    diffs = new_embeddings - self.memory_centroid.unsqueeze(0)
                    new_var = (diffs ** 2).mean().item()
                    # Exponential moving average for variance
                    self.memory_variance = 0.9 * self.memory_variance + 0.1 * new_var + EPS

            # Prune if structures grow too large
            self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        """Prune data structures to prevent unbounded growth."""
        # Prune cooccurrence pairs (keep most frequent)
        if len(self.slot_cooccurrence) > self.max_cooccurrence_pairs:
            sorted_pairs = sorted(
                self.slot_cooccurrence.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.slot_cooccurrence = dict(sorted_pairs[:self.max_cooccurrence_pairs])

        # Prune observed combinations (keep most recent by converting to list)
        if len(self.observed_combinations) > self.max_observed_combinations:
            # Convert to list, keep last N (approximates recency)
            combo_list = list(self.observed_combinations)
            self.observed_combinations = set(combo_list[-self.max_observed_combinations:])

    def score(self, memory: StructuredMemory) -> PlausibilityScore:
        """Score the plausibility of a structured memory."""
        with self._lock:
            score = PlausibilityScore()

            # Structural coherence: do the parts fit together?
            score.structural_coherence = self._score_structural(memory)

            # Semantic coherence: does the meaning make sense?
            score.semantic_coherence = self._score_semantic(memory)

            # Physical coherence: respects physics?
            score.physical_coherence = self._score_physical(memory)

            # Relational coherence: valid relations?
            score.relational_coherence = self._score_relational(memory)

            # Novelty: how new is this?
            score.novelty = self._score_novelty(memory)

            return score

    def _score_structural(self, memory: StructuredMemory) -> float:
        """Score structural coherence - parts fitting together."""
        if not memory.slots:
            return 0.5

        # Check if slot combinations have been seen
        slot_names = frozenset(memory.slots.keys())

        if slot_names in self.observed_combinations:
            base_score = 0.9  # Seen this combination before
        else:
            # Partial overlap?
            max_overlap = 0.0
            for observed in self.observed_combinations:
                union_size = len(slot_names | observed)
                if union_size > 0:
                    overlap = len(slot_names & observed) / union_size
                    max_overlap = max(max_overlap, overlap)
            base_score = 0.5 + 0.4 * max_overlap

        # Penalize for low-confidence slots
        confidences = [max(s.confidence, EPS) for s in memory.slots.values()]
        confidence_factor = sum(confidences) / max(len(confidences), 1)

        return base_score * confidence_factor

    def _score_semantic(self, memory: StructuredMemory) -> float:
        """Score semantic coherence - meaning makes sense."""
        if not memory.slots:
            return 0.5

        # Check slot content similarity (slots should be somewhat related)
        contents = [s.content for s in memory.slots.values()]
        if len(contents) < 2:
            return 0.8

        # Compute average pairwise similarity
        total_sim = 0.0
        count = 0
        for i, c1 in enumerate(contents):
            for c2 in contents[i+1:]:
                sim = torch.nn.functional.cosine_similarity(
                    c1.unsqueeze(0), c2.unsqueeze(0)
                ).item()
                total_sim += sim
                count += 1

        avg_sim = total_sim / count if count > 0 else 0.5

        # Too similar = boring, too different = incoherent
        # Sweet spot around 0.3-0.7
        if avg_sim < 0:
            coherence = 0.3 + 0.3 * (avg_sim + 1)  # -1 to 0 -> 0.3 to 0.6
        elif avg_sim > 0.8:
            coherence = 0.8 - 0.2 * (avg_sim - 0.8) / 0.2  # 0.8 to 1 -> 0.8 to 0.6
        else:
            coherence = 0.6 + 0.4 * min(avg_sim / 0.8, 1.0)  # 0 to 0.8 -> 0.6 to 1.0

        return coherence

    def _score_physical(self, memory: StructuredMemory) -> float:
        """Score physical coherence - respects physics."""
        # Check explicit physical rules
        for rule in self.physical_rules:
            if not rule(memory):
                return 0.3

        # Default: assume physical if not obviously impossible
        # Check for "impossible" slot combinations (heuristic)

        # If we have location and action, check compatibility
        location = memory.get_slot("location")
        action = memory.get_slot("action")

        if location and action:
            # Check if action is possible in location (simplified)
            compatibility = torch.nn.functional.cosine_similarity(
                location.content.unsqueeze(0),
                action.content.unsqueeze(0)
            ).item()
            # Allow some incompatibility (imagination stretches physics)
            return 0.5 + 0.5 * max(0, compatibility + 0.5) / 1.5

        return 0.8  # Default: probably physical

    def _score_relational(self, memory: StructuredMemory) -> float:
        """Score relational coherence - valid relations."""
        relation_slots = memory.get_slots_by_type(SlotType.RELATION)

        if not relation_slots:
            return 0.8  # No relations to check

        # Check each relation
        scores = []
        for rel in relation_slots:
            # Relations should have moderate confidence
            scores.append(max(rel.confidence, EPS))

        return sum(scores) / max(len(scores), 1)

    def _score_novelty(self, memory: StructuredMemory) -> float:
        """Score novelty - how new/creative is this?"""
        if self.memory_centroid is None:
            return 0.5

        # Distance from centroid of observed memories
        embedding = memory.to_embedding(self.content_dim)

        # Handle NaN/inf in embedding
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            return 0.5

        distance = (embedding - self.memory_centroid).norm().item()

        # Normalize by variance (with protection against near-zero variance)
        safe_var = max(self.memory_variance, EPS)
        normalized_distance = distance / math.sqrt(safe_var)

        # Sigmoid to bound (with protection against overflow)
        normalized_distance = min(normalized_distance, 10.0)  # Cap to prevent exp overflow
        novelty = 2 / (1 + math.exp(-normalized_distance)) - 1

        # Also factor in source diversity
        n_sources = len(memory.source_memories) if memory.source_memories else 0
        source_diversity = min(n_sources / 5.0, 1.0)  # More sources = more novel, capped at 1

        return min(1.0, max(0.0, 0.7 * novelty + 0.3 * source_diversity))

    def add_physical_rule(self, rule: callable) -> None:
        """Add a physical constraint rule.

        Rule should take StructuredMemory and return bool (True = passes).
        """
        self.physical_rules.append(rule)


class IdeaEvaluator:
    """Evaluates ideas for the brainstorming loop.

    Score = Novelty * Coherence * Relevance * Valence
    """

    def __init__(
        self,
        plausibility_field: PlausibilityField,
        relevance_weight: float = 0.3,
        novelty_weight: float = 0.3,
        coherence_weight: float = 0.3,
        valence_weight: float = 0.1
    ):
        self.plausibility = plausibility_field
        self.relevance_weight = relevance_weight
        self.novelty_weight = novelty_weight
        self.coherence_weight = coherence_weight
        self.valence_weight = valence_weight

    def evaluate(
        self,
        idea: StructuredMemory,
        goal_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate an idea on multiple dimensions."""

        # Get plausibility scores
        plausibility_score = self.plausibility.score(idea)

        # Relevance to goal (if provided)
        if goal_embedding is not None:
            idea_embedding = idea.to_embedding(self.plausibility.content_dim)
            relevance = torch.nn.functional.cosine_similarity(
                idea_embedding.unsqueeze(0),
                goal_embedding.unsqueeze(0)
            ).item()
            relevance = (relevance + 1) / 2  # Normalize to 0-1
        else:
            relevance = 0.5  # Neutral if no goal

        # Extract components
        novelty = plausibility_score.novelty
        coherence = plausibility_score.total_plausibility
        valence = (idea.valence + 1) / 2  # Normalize to 0-1

        # Weighted combination
        total = (
            self.relevance_weight * relevance +
            self.novelty_weight * novelty +
            self.coherence_weight * coherence +
            self.valence_weight * valence
        )

        return {
            "total": total,
            "relevance": relevance,
            "novelty": novelty,
            "coherence": coherence,
            "valence": valence,
            "creative_value": plausibility_score.creative_value,
            "is_acceptable": plausibility_score.is_acceptable
        }

    def rank_ideas(
        self,
        ideas: List[StructuredMemory],
        goal_embedding: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> List[Tuple[StructuredMemory, Dict[str, float]]]:
        """Rank ideas by evaluation score."""
        evaluated = []
        for idea in ideas:
            scores = self.evaluate(idea, goal_embedding)
            evaluated.append((idea, scores))

        # Sort by total score
        evaluated.sort(key=lambda x: x[1]["total"], reverse=True)

        return evaluated[:top_k]


class CreativityOptimizer:
    """Optimizes the novelty-coherence tradeoff.

    Too much novelty -> incoherent nonsense
    Too little novelty -> boring rehash
    Sweet spot: novel AND coherent
    """

    def __init__(
        self,
        plausibility_field: PlausibilityField,
        target_novelty: float = 0.6,
        min_coherence: float = 0.4
    ):
        self.plausibility = plausibility_field
        self.target_novelty = target_novelty
        self.min_coherence = min_coherence

    def suggest_transform(
        self,
        memory: StructuredMemory,
        available_slots: List[str] = None
    ) -> Optional[Transform]:
        """Suggest a transform to improve creative value."""
        current_score = self.plausibility.score(memory)

        if current_score.novelty < self.target_novelty:
            # Need more novelty - suggest disrupting transform
            if available_slots:
                target = available_slots[0]
            else:
                slots = list(memory.slots.keys())
                target = slots[0] if slots else "default"

            return Transform(
                transform_type=TransformType.MORPH,
                target_slot=target,
                magnitude=0.3 + 0.3 * (self.target_novelty - current_score.novelty)
            )

        elif current_score.total_plausibility < self.min_coherence:
            # Need more coherence - suggest consolidating transform
            return Transform(
                transform_type=TransformType.SCALE,
                target_slot="all",
                magnitude=-0.1,  # Scale down = simplify
                parameters={"factor": 0.9}
            )

        return None  # Already in sweet spot

    def balance_portfolio(
        self,
        ideas: List[StructuredMemory],
        n_select: int = 3
    ) -> List[StructuredMemory]:
        """Select a balanced portfolio of ideas.

        Mix of:
        - High novelty (even if less coherent)
        - High coherence (even if less novel)
        - Balanced (the sweet spot)
        """
        if len(ideas) <= n_select:
            return ideas

        scored = [(idea, self.plausibility.score(idea)) for idea in ideas]

        selected = []

        # Select highest novelty
        scored.sort(key=lambda x: x[1].novelty, reverse=True)
        if scored:
            selected.append(scored[0][0])
            scored = scored[1:]

        # Select highest coherence
        scored.sort(key=lambda x: x[1].total_plausibility, reverse=True)
        if scored and len(selected) < n_select:
            selected.append(scored[0][0])
            scored = scored[1:]

        # Select highest creative value (balance)
        scored.sort(key=lambda x: x[1].creative_value, reverse=True)
        while len(selected) < n_select and scored:
            selected.append(scored[0][0])
            scored = scored[1:]

        return selected
