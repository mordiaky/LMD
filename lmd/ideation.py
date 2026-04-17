"""Ideation Engine - Autonomous brainstorming and idea generation.

The brain can come up with its own ideas through:
1. DIVERGE: Generate many candidate ideas
2. IMAGINE: Transform and combine on mental canvas
3. EVALUATE: Score for novelty, coherence, relevance
4. SELECT: Keep the best ideas
5. COMBINE: Cross-pollinate good ideas
6. EVOLVE: Iterate until satisfactory

"If we get this working the brain should be able to come up with
its own ideas, its own brainstorming" - Joshua

Invented by Joshua R. Thomas, January 2026.
"""

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import LMDConfig
from .dynamics import LMDDynamics
from .imagination import (
    MemoryDecomposer,
    MemorySlot,
    MentalCanvas,
    SlotType,
    StructuredMemory,
    Transform,
    TransformOps,
    TransformType,
    WillGenerator,
    WillVector,
)
from .living_memory import LivingMemory, ValenceTrajectory
from .plausibility import CreativityOptimizer, IdeaEvaluator, PlausibilityField
from .safeguards import (
    AutonomyController,
    AutonomyTrigger,
    RealityAnchor,
    RepulsionField,
    ResourceBudget,
    get_id_generator,
    safe_normalize,
)


class IdeationPhase(Enum):
    """Phases of the ideation process."""
    DIVERGE = auto()    # Generate many ideas
    IMAGINE = auto()    # Transform and elaborate
    EVALUATE = auto()   # Score and filter
    SELECT = auto()     # Choose best
    COMBINE = auto()    # Cross-pollinate
    EVOLVE = auto()     # Iterate improvements
    CONSOLIDATE = auto() # Finalize to memory


@dataclass
class IdeationConfig:
    """Configuration for ideation process."""

    # Divergence parameters
    n_initial_ideas: int = 10          # Ideas to generate in diverge phase
    divergence_noise: float = 0.3      # Noise for variation
    n_transforms_per_idea: int = 3     # Transforms to apply

    # Selection parameters
    survival_ratio: float = 0.4        # Fraction to keep each round
    min_coherence: float = 0.3         # Minimum acceptable coherence
    target_novelty: float = 0.5        # Target novelty level

    # Combination parameters
    n_combinations: int = 5            # Combinations to try
    combination_noise: float = 0.1     # Noise when combining

    # Evolution parameters
    n_iterations: int = 5              # Brainstorming iterations
    convergence_threshold: float = 0.05  # Stop if improvement < this

    # Output
    n_final_ideas: int = 3             # Number of final ideas to return

    @classmethod
    def quick(cls) -> "IdeationConfig":
        """Quick ideation for testing."""
        return cls(
            n_initial_ideas=5,
            n_transforms_per_idea=2,
            n_iterations=3,
            n_final_ideas=2
        )

    @classmethod
    def thorough(cls) -> "IdeationConfig":
        """Thorough ideation for quality results."""
        return cls(
            n_initial_ideas=20,
            n_transforms_per_idea=5,
            n_iterations=10,
            n_final_ideas=5
        )


@dataclass
class IdeationResult:
    """Result of an ideation session."""

    # Final ideas
    ideas: List[StructuredMemory]
    scores: List[Dict[str, float]]

    # Process metadata
    n_iterations: int = 0
    total_ideas_generated: int = 0
    total_ideas_discarded: int = 0
    convergence_achieved: bool = False

    # Best idea
    @property
    def best_idea(self) -> Optional[StructuredMemory]:
        if not self.ideas:
            return None
        return self.ideas[0]

    @property
    def best_score(self) -> float:
        if not self.scores:
            return 0.0
        return self.scores[0].get("total", 0.0)

    # Statistics
    @property
    def average_novelty(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.get("novelty", 0) for s in self.scores) / len(self.scores)

    @property
    def average_coherence(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.get("coherence", 0) for s in self.scores) / len(self.scores)

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Ideation Complete:\n"
            f"  Ideas generated: {self.total_ideas_generated}\n"
            f"  Ideas discarded: {self.total_ideas_discarded}\n"
            f"  Final ideas: {len(self.ideas)}\n"
            f"  Iterations: {self.n_iterations}\n"
            f"  Converged: {self.convergence_achieved}\n"
            f"  Best score: {self.best_score:.3f}\n"
            f"  Avg novelty: {self.average_novelty:.3f}\n"
            f"  Avg coherence: {self.average_coherence:.3f}"
        )


class IdeationEngine:
    """Autonomous ideation and brainstorming engine.

    Implements the full ideation loop:
    DIVERGE -> IMAGINE -> EVALUATE -> SELECT -> COMBINE -> EVOLVE

    Can work with:
    - External will (user provides goal)
    - Internal will (curiosity, problems)
    - Combined will (external + internal)
    """

    def __init__(
        self,
        config: LMDConfig,
        ideation_config: Optional[IdeationConfig] = None
    ):
        self.lmd_config = config
        self.ideation_config = ideation_config or IdeationConfig()

        # Core components
        self.content_dim = config.content_dim
        self.decomposer = MemoryDecomposer(config.content_dim)
        self.transform_ops = TransformOps(config.content_dim)
        self.canvas = MentalCanvas(config.content_dim)
        self.will_generator = WillGenerator(config.content_dim)
        self.plausibility = PlausibilityField(config.content_dim)
        self.evaluator = IdeaEvaluator(self.plausibility)
        self.optimizer = CreativityOptimizer(self.plausibility)

        # === SAFEGUARDS ===
        self.id_generator = get_id_generator()
        self.repulsion_field = RepulsionField(config.content_dim)
        self.reality_anchor = RealityAnchor(config.content_dim)

        # State
        self.current_phase = IdeationPhase.DIVERGE
        self.idea_pool: List[StructuredMemory] = []
        self.iteration_count = 0
        self.history: List[Dict[str, Any]] = []

    def ideate(
        self,
        memories: List[LivingMemory],
        will: Optional[WillVector] = None,
        goal_description: Optional[str] = None,
        budget: Optional[ResourceBudget] = None
    ) -> IdeationResult:
        """Run full ideation process with safeguards.

        Args:
            memories: Source memories to draw from
            will: External will vector (if user-directed)
            goal_description: Optional text description of goal
            budget: Optional resource budget (limits compute/time)

        Returns:
            IdeationResult with final ideas
        """
        # Initialize with resource budget
        self._initialize(memories)
        budget = budget or ResourceBudget(
            max_ideas=self.ideation_config.n_initial_ideas * 3,
            max_iterations=self.ideation_config.n_iterations,
            max_time_seconds=30.0
        )

        # Generate will if not provided
        if will is None:
            will = self.will_generator.generate_combined_will(memories)

        goal_embedding = will.direction

        # Track stats
        total_generated = 0
        total_discarded = 0
        prev_best_score = 0.0

        # Main loop with budget checking
        for iteration in range(self.ideation_config.n_iterations):
            # Check resource budget
            if budget.is_exhausted():
                break

            self.iteration_count = iteration
            budget.record_iteration()

            # 1. DIVERGE: Generate new ideas (with repulsion)
            new_ideas = self._diverge(memories, will)
            total_generated += len(new_ideas)
            self.idea_pool.extend(new_ideas)
            budget.record_ops(len(new_ideas) * 100)  # Estimate ops

            # 2. IMAGINE: Transform ideas on canvas
            self._imagine(will)
            budget.record_ops(len(self.idea_pool) * 50)

            # 3. EVALUATE: Score all ideas with repulsion penalty
            scored_ideas = self._evaluate_with_safeguards(goal_embedding)
            budget.record_ops(len(self.idea_pool) * 80)

            # 4. SELECT: Keep best
            n_keep = max(2, int(len(scored_ideas) * self.ideation_config.survival_ratio))
            survivors = scored_ideas[:n_keep]
            total_discarded += len(scored_ideas) - n_keep

            # Mark discarded ideas as explored (for repulsion) with low quality
            for idea, scores in scored_ideas[n_keep:]:
                embedding = idea.to_embedding(self.content_dim)
                self.repulsion_field.mark_explored(
                    embedding,
                    quality_score=scores.get("total", 0.3),
                    was_productive=False  # Discarded = not productive
                )
                # Record outcome for reality anchor calibration
                self.reality_anchor.record_outcome(
                    idea_id=idea.id,
                    predicted_valence=idea.valence,
                    actual_outcome=scores.get("total", 0.0) * 2 - 1  # Scale to -1..1
                )

            self.idea_pool = [idea for idea, _ in survivors]

            # 5. COMBINE: Cross-pollinate
            combinations = self._combine(self.idea_pool, will)
            total_generated += len(combinations)
            self.idea_pool.extend(combinations)
            budget.record_ops(len(combinations) * 60)

            # 6. CHECK CONVERGENCE
            if survivors:
                best_score = survivors[0][1]["total"]
                improvement = best_score - prev_best_score

                self.history.append({
                    "iteration": iteration,
                    "pool_size": len(self.idea_pool),
                    "best_score": best_score,
                    "avg_novelty": sum(s["novelty"] for _, s in survivors) / len(survivors),
                    "avg_coherence": sum(s["coherence"] for _, s in survivors) / len(survivors),
                    "repulsion_active": len(self.repulsion_field.explored_regions)
                })

                if improvement < self.ideation_config.convergence_threshold:
                    break

                prev_best_score = best_score

        # Final selection
        final_scored = self._evaluate_with_safeguards(goal_embedding)[:self.ideation_config.n_final_ideas]

        # Mark final ideas as explored with HIGH quality (productive)
        for idea, scores in final_scored:
            embedding = idea.to_embedding(self.content_dim)
            self.repulsion_field.mark_explored(
                embedding,
                quality_score=scores.get("total", 0.7),
                was_productive=True  # Final selections are productive!
            )
            # Record outcome as positive (selected ideas are "good")
            self.reality_anchor.record_outcome(
                idea_id=idea.id,
                predicted_valence=idea.valence,
                actual_outcome=scores.get("total", 0.5) * 2 - 1
            )
            # Also mark in will generator for curiosity repulsion
            self.will_generator.mark_explored(embedding)
            budget.record_idea()

        # Learn from final structured ideas for future plausibility scoring
        self.plausibility.learn_from_structured([idea for idea, _ in final_scored])

        return IdeationResult(
            ideas=[idea for idea, _ in final_scored],
            scores=[score for _, score in final_scored],
            n_iterations=self.iteration_count + 1,
            total_ideas_generated=total_generated,
            total_ideas_discarded=total_discarded,
            convergence_achieved=self.iteration_count < self.ideation_config.n_iterations - 1
        )

    def _evaluate_with_safeguards(
        self,
        goal_embedding: torch.Tensor
    ) -> List[Tuple[StructuredMemory, Dict[str, float]]]:
        """Evaluate ideas with repulsion penalty and reality grounding."""
        scored = []

        for idea in self.idea_pool:
            # Get base evaluation
            base_scores = self.evaluator.evaluate(idea, goal_embedding)

            # Apply repulsion penalty to novelty
            embedding = idea.to_embedding(self.content_dim)
            repulsion_penalty = self.repulsion_field.novelty_penalty(embedding)
            adjusted_novelty = base_scores["novelty"] * (1.0 - 0.5 * repulsion_penalty)

            # Apply reality grounding to valence
            grounded_valence = self.reality_anchor.ground_valence(
                idea.valence,
                embedding,
                idea.id
            )

            # Recompute total with adjustments
            adjusted_total = (
                0.3 * base_scores["relevance"] +
                0.3 * adjusted_novelty +
                0.3 * base_scores["coherence"] +
                0.1 * (grounded_valence + 1) / 2  # Normalize to 0-1
            )

            adjusted_scores = base_scores.copy()
            adjusted_scores["novelty"] = adjusted_novelty
            adjusted_scores["valence"] = grounded_valence
            adjusted_scores["total"] = adjusted_total
            adjusted_scores["repulsion_penalty"] = repulsion_penalty

            scored.append((idea, adjusted_scores))

        # Sort by adjusted total
        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        return scored

    def _initialize(self, memories: List[LivingMemory]) -> None:
        """Initialize ideation from memories."""
        self.idea_pool = []
        self.iteration_count = 0
        self.history = []
        self.canvas.clear()

        # Learn plausibility from memories
        self.plausibility.learn_from_memories(memories)

    def _diverge(
        self,
        memories: List[LivingMemory],
        will: WillVector
    ) -> List[StructuredMemory]:
        """DIVERGE: Generate many candidate ideas.

        Strategies:
        1. Decompose random memories
        2. Apply random transforms
        3. Combine parts from different memories
        4. Follow will direction with noise
        """
        ideas = []
        n_ideas = self.ideation_config.n_initial_ideas

        # Strategy 1: Direct decomposition (base ideas)
        n_decompose = n_ideas // 3
        sampled = random.sample(memories, min(n_decompose, len(memories)))
        for mem in sampled:
            structured = self.decomposer.decompose(mem)
            ideas.append(structured)

        # Strategy 2: Transformed decompositions
        n_transform = n_ideas // 3
        for _ in range(n_transform):
            if not memories:
                continue
            base_mem = random.choice(memories)
            structured = self.decomposer.decompose(base_mem)

            # Apply random transforms
            for _ in range(self.ideation_config.n_transforms_per_idea):
                transform = self._random_transform(structured)
                structured = self.transform_ops.apply(structured, transform)

            ideas.append(structured)

        # Strategy 3: Will-directed generation
        n_directed = n_ideas - len(ideas)
        for _ in range(n_directed):
            # Start from will direction
            content = will.direction + torch.randn(self.content_dim) * self.ideation_config.divergence_noise

            structured = StructuredMemory(id=len(ideas))
            structured.add_slot("core", MemorySlot(
                slot_type=SlotType.AGENT,
                name="core",
                content=content,
                confidence=0.7
            ))

            # Enrich with parts from memories
            if memories:
                source_mem = random.choice(memories)
                decomposed = self.decomposer.decompose(source_mem)
                for name, slot in decomposed.slots.items():
                    if random.random() < 0.5:  # Randomly include slots
                        slot_copy = slot.clone()
                        slot_copy.name = f"borrowed_{name}"
                        structured.add_slot(slot_copy.name, slot_copy)

            ideas.append(structured)

        return ideas

    def _random_transform(self, memory: StructuredMemory) -> Transform:
        """Generate a random transform for an idea."""
        transform_types = [
            TransformType.MORPH,
            TransformType.SCALE,
            TransformType.BULGE,
            TransformType.RECOLOR,
            TransformType.ADD,
        ]

        slots = list(memory.slots.keys())
        target_slot = random.choice(slots) if slots else "default"

        return Transform(
            transform_type=random.choice(transform_types),
            target_slot=target_slot,
            magnitude=random.uniform(0.2, 0.6)
        )

    def _imagine(self, will: WillVector) -> None:
        """IMAGINE: Elaborate ideas on mental canvas.

        Put ideas on canvas, apply transforms, let them interact.
        """
        self.canvas.clear()

        # Put top ideas on canvas
        for i, idea in enumerate(self.idea_pool[:5]):
            position = (float(i), 0.0, 0.0)
            self.canvas.add_entity(idea, position)

        # Apply will-directed transforms
        for entity_id in list(self.canvas.entities.keys()):
            entity = self.canvas.get_entity(entity_id)
            if entity is None:
                continue

            # Suggest transform based on creativity optimization
            transform = self.optimizer.suggest_transform(entity.memory)
            if transform:
                self.canvas.transform_entity(entity_id, transform)

            # Apply will bias
            if random.random() < will.strength * 0.3:
                for slot in entity.memory.slots.values():
                    slot.content = will.apply_to(slot.content, alpha=0.1)

        # Update pool from canvas
        for entity_id, entity in self.canvas.entities.items():
            if entity_id < len(self.idea_pool):
                self.idea_pool[entity_id] = entity.memory

    def _combine(
        self,
        ideas: List[StructuredMemory],
        will: WillVector
    ) -> List[StructuredMemory]:
        """COMBINE: Cross-pollinate good ideas.

        Take parts from different ideas and combine them.
        """
        if len(ideas) < 2:
            return []

        combinations = []
        n_combos = self.ideation_config.n_combinations

        for _ in range(n_combos):
            # Pick two parents
            parent1, parent2 = random.sample(ideas, 2)

            # Create child by combining slots
            child = StructuredMemory(id=len(ideas) + len(combinations))

            all_slots = list(parent1.slots.items()) + list(parent2.slots.items())
            random.shuffle(all_slots)

            for name, slot in all_slots:
                if name not in child.slots:
                    # Add with some noise
                    slot_copy = slot.clone()
                    slot_copy.content = slot.content + torch.randn_like(slot.content) * self.ideation_config.combination_noise
                    child.add_slot(name, slot_copy)

            # Inherit properties
            child.valence = (parent1.valence + parent2.valence) / 2
            child.novelty = max(parent1.novelty, parent2.novelty) + 0.1

            combinations.append(child)

        return combinations

    def brainstorm(
        self,
        memories: List[LivingMemory],
        problem: str,
        n_ideas: int = 5
    ) -> IdeationResult:
        """Brainstorm ideas for a specific problem.

        Convenience method that creates will from problem description.
        """
        # Create problem-based will
        # In production, would use text encoder
        problem_embedding = torch.randn(self.content_dim)  # Placeholder
        problem_embedding = safe_normalize(problem_embedding)

        will = self.will_generator.generate_problem_will(problem, problem_embedding)

        # Adjust config for problem-solving
        config = IdeationConfig(
            n_initial_ideas=n_ideas * 2,
            n_final_ideas=n_ideas,
            target_novelty=0.4,  # Problems need practical solutions
            min_coherence=0.5   # Solutions must be coherent
        )
        self.ideation_config = config

        return self.ideate(memories, will, problem)

    def explore_curiosity(
        self,
        memories: List[LivingMemory],
        n_ideas: int = 3
    ) -> IdeationResult:
        """Autonomous exploration driven by curiosity.

        No external goal - let internal curiosity guide ideation.
        """
        # Generate curiosity-based will
        will = self.will_generator.generate_curiosity_will(memories)

        # Adjust config for exploration
        config = IdeationConfig(
            n_initial_ideas=n_ideas * 3,
            n_final_ideas=n_ideas,
            target_novelty=0.7,  # Exploration prizes novelty
            min_coherence=0.3   # Allow some incoherence
        )
        self.ideation_config = config

        return self.ideate(memories, will)


class AutonomousIdeator:
    """Fully autonomous ideation with trigger-based activation.

    The brain generates its own questions and explores answers.
    Respects resource limits and trigger conditions.
    """

    def __init__(
        self,
        config: LMDConfig,
        dynamics: LMDDynamics
    ):
        self.config = config
        self.dynamics = dynamics
        self.engine = IdeationEngine(config)

        # Autonomy controller with resource limits
        self.autonomy = AutonomyController(
            max_ideas_per_hour=50,
            max_compute_per_session=100000,
            min_interval_seconds=30.0
        )

        # Track what we've explored
        self.explored_ideas: List[StructuredMemory] = []
        self.interesting_questions: List[str] = []

        # Metrics for trigger updates
        self.last_activity_time = 0.0
        self.recent_novelty_scores: List[float] = []

    def update_triggers(
        self,
        idle_seconds: float = 0.0,
        curiosity_score: float = 0.0,
        recent_novelty: float = 0.5
    ) -> None:
        """Update trigger conditions based on current state."""
        self.autonomy.update_trigger(AutonomyTrigger.IDLE, idle_seconds)
        self.autonomy.update_trigger(AutonomyTrigger.CURIOSITY, curiosity_score)

        # Boredom = low novelty for a while
        self.recent_novelty_scores.append(recent_novelty)
        if len(self.recent_novelty_scores) > 10:
            self.recent_novelty_scores = self.recent_novelty_scores[-10:]
        avg_novelty = sum(self.recent_novelty_scores) / len(self.recent_novelty_scores)
        boredom = 1.0 - avg_novelty
        self.autonomy.update_trigger(AutonomyTrigger.BOREDOM, boredom)

    def should_ideate(self) -> Optional[AutonomyTrigger]:
        """Check if autonomous ideation should trigger.

        Returns trigger type if should ideate, None otherwise.
        """
        return self.autonomy.check_triggers()

    def run_autonomous_session(
        self,
        memories: List[LivingMemory],
        n_rounds: int = 3,
        ideas_per_round: int = 2
    ) -> List[IdeationResult]:
        """Run autonomous ideation session with resource limits.

        The system:
        1. Checks if ideation is allowed (resource limits)
        2. Identifies gaps/curiosities in memory
        3. Generates questions to explore
        4. Ideates answers with budget constraints
        5. Consolidates interesting findings
        """
        results = []

        # Check if we can ideate
        if not self.autonomy.can_ideate():
            return results

        for round_idx in range(n_rounds):
            # Check per-round limits
            if not self.autonomy.can_ideate():
                break

            # Generate internal will from curiosity
            will = self.engine.will_generator.generate_curiosity_will(memories)

            # Create budget for this round
            budget = ResourceBudget(
                max_ideas=ideas_per_round * 2,
                max_iterations=5,
                max_time_seconds=10.0
            )

            # Run ideation with budget
            result = self.engine.ideate(
                memories,
                will,
                goal_description=f"Autonomous exploration round {round_idx + 1}",
                budget=budget
            )

            results.append(result)

            # Record resource usage
            self.autonomy.record_ideation(
                n_ideas=len(result.ideas),
                compute_ops=budget.ops_used
            )

            # Update explored ideas
            self.explored_ideas.extend(result.ideas)

            # Generate questions about findings (for next round)
            if result.ideas:
                best_idea = result.best_idea
                if best_idea:
                    # Mark explored regions using the proper method
                    embedding = best_idea.to_embedding(self.config.content_dim)
                    self.engine.will_generator.mark_explored(embedding)

        return results

    def consolidate_to_memories(
        self,
        ideas: List[StructuredMemory],
        dynamics: LMDDynamics
    ) -> List[LivingMemory]:
        """Convert ideated structures back to living memories.

        Good ideas become real memories that can participate in LMD.
        Uses proper ID generator to prevent collisions.
        """
        new_memories = []
        id_gen = get_id_generator()

        for idea in ideas:
            content = idea.to_embedding(self.config.content_dim)

            # Get unique ID from generator (consolidated namespace)
            memory_id = id_gen.next_id("consolidated")

            memory = LivingMemory(
                id=memory_id,
                content=content,
                valence=ValenceTrajectory.constant(idea.valence),
                energy=0.8,  # Start with good energy (it's a good idea!)
                created_at=dynamics.timestep,
                label=f"imagined_{memory_id}"
            )

            new_memories.append(memory)

        return new_memories


def run_ideation_demo(memories: List[LivingMemory], config: LMDConfig) -> IdeationResult:
    """Demo function to show ideation in action."""
    engine = IdeationEngine(config, IdeationConfig.quick())

    # External will: "I want something creative with wings"
    will = WillVector(
        direction=torch.randn(config.content_dim),
        strength=0.8,
        specificity=0.5,
        source="external",
        description="Something creative with wings"
    )

    result = engine.ideate(memories, will)
    return result
