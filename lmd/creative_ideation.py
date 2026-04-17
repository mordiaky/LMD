"""Creative Ideation Engine: Unified System for Human-Like Creative Leaps.

Integrates all advanced divergence operators into a single coherent system:
- CreativeLeapEngine: Analogical transfer, manifold walking, orthogonal composition
- HierarchicalIdea: Tree-structured ideas with graftable components
- ActiveCuriosityProber: Targeted frontier exploration
- ValenceModulation: Dopamine-driven operator selection

This enables human-like creative leaps:
- "dragon fire" + "underwater glass" -> "prismatic breath weapon"
- "modular robot" graft onto "armor scales" -> "puzzle-piece armor"

Invented by Joshua R. Thomas, January 2026.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import LMDConfig
from .creative_leaps import CreativeLeap, CreativeLeapConfig, CreativeLeapEngine
from .curiosity_prober import ActiveCuriosityProber, CuriosityDrivenWill
from .hierarchical_ideas import (
    ComponentType,
    HierarchicalIdea,
    HierarchicalIdeaFactory,
    IdeaComponent,
    IdeaGrafter,
)
from .living_memory import LivingMemory
from .plausibility import PlausibilityField
from .safeguards import RealityAnchor, RepulsionField, safe_normalize


class IdeaForm(Enum):
    """Form of generated ideas."""
    FLAT = auto()         # Single embedding vector
    HIERARCHICAL = auto()  # Tree-structured composite
    LEAP = auto()          # Result of creative leap


@dataclass
class CreativeIdea:
    """A creative idea that can be flat or hierarchical."""
    id: str
    form: IdeaForm

    # Content (mutually exclusive based on form)
    embedding: Optional[torch.Tensor] = None        # For FLAT
    hierarchical: Optional[HierarchicalIdea] = None  # For HIERARCHICAL
    leap: Optional[CreativeLeap] = None             # For LEAP

    # Scores and metadata
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    plausibility_score: float = 0.0
    total_score: float = 0.0
    source_strategy: str = ""
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_embedding(self, content_dim: int = 32) -> torch.Tensor:
        """Get embedding representation regardless of form."""
        if self.form == IdeaForm.FLAT and self.embedding is not None:
            return self.embedding
        elif self.form == IdeaForm.HIERARCHICAL and self.hierarchical is not None:
            return self.hierarchical.to_embedding(content_dim)
        elif self.form == IdeaForm.LEAP and self.leap is not None:
            return self.leap.embedding
        else:
            return safe_normalize(torch.randn(content_dim))


@dataclass
class CreativeIdeationResult:
    """Result of creative ideation session."""
    ideas: List[CreativeIdea]
    best_idea: Optional[CreativeIdea]
    best_score: float
    total_generated: int
    strategies_used: Dict[str, int]
    average_novelty: float
    average_coherence: float
    leap_types_used: Dict[str, int]
    session_time: float


@dataclass
class CreativeIdeationConfig:
    """Configuration for creative ideation."""
    content_dim: int = 32

    # Generation counts
    ideas_per_round: int = 10
    flat_ratio: float = 0.3       # Fraction of flat embeddings
    hierarchical_ratio: float = 0.3  # Fraction of hierarchical
    leap_ratio: float = 0.4       # Fraction of creative leaps

    # Quality thresholds
    min_novelty: float = 0.2
    min_coherence: float = 0.3
    min_plausibility: float = 0.3

    # Creativity parameters
    temperature: float = 1.0
    exploration_bonus: float = 0.2

    # Operator biases
    prefer_analogical: float = 1.0
    prefer_orthogonal: float = 1.0
    prefer_grafting: float = 1.0

    @classmethod
    def conservative(cls) -> "CreativeIdeationConfig":
        """Conservative: more coherent, less wild."""
        return cls(
            flat_ratio=0.5,
            hierarchical_ratio=0.3,
            leap_ratio=0.2,
            temperature=0.5,
            min_coherence=0.5
        )

    @classmethod
    def wild(cls) -> "CreativeIdeationConfig":
        """Wild: more creative leaps, higher temperature."""
        return cls(
            flat_ratio=0.1,
            hierarchical_ratio=0.3,
            leap_ratio=0.6,
            temperature=1.5,
            prefer_analogical=1.5,
            prefer_orthogonal=1.2
        )

    @classmethod
    def balanced(cls) -> "CreativeIdeationConfig":
        """Balanced default configuration."""
        return cls()


class CreativeIdeationEngine:
    """Unified engine for creative ideation with human-like leaps.

    Combines:
    - Flat embedding generation (basic)
    - Hierarchical idea construction (structured)
    - Creative leaps (analogical, orthogonal, diffusion)
    - Curiosity-driven exploration (targeted novelty)

    All modulated by dopamine for valence-driven creativity.
    """

    def __init__(
        self,
        config: Optional[CreativeIdeationConfig] = None,
        lmd_config: Optional[LMDConfig] = None
    ):
        self.config = config or CreativeIdeationConfig.balanced()
        self.lmd_config = lmd_config or LMDConfig.toy_scale()

        # Initialize sub-engines
        leap_config = CreativeLeapConfig(
            content_dim=self.config.content_dim,
            analogical_weight=0.3 * self.config.prefer_analogical,
            orthogonal_weight=0.25 * self.config.prefer_orthogonal,
            temperature=self.config.temperature
        )
        self.leap_engine = CreativeLeapEngine(leap_config)

        self.idea_factory = HierarchicalIdeaFactory(self.config.content_dim)
        self.grafter = IdeaGrafter(self.config.content_dim)

        self.repulsion_field = RepulsionField(content_dim=self.config.content_dim)
        self.curiosity_prober = ActiveCuriosityProber(
            content_dim=self.config.content_dim,
            repulsion_field=self.repulsion_field
        )
        self.curiosity_will = CuriosityDrivenWill(
            content_dim=self.config.content_dim,
            prober=self.curiosity_prober,
            repulsion_field=self.repulsion_field
        )

        self.plausibility = PlausibilityField(content_dim=self.config.content_dim)
        self.reality_anchor = RealityAnchor(content_dim=self.config.content_dim)

        # ID generation
        self._id_counter = 0
        self._lock = threading.RLock()

        # Statistics
        self.session_count = 0
        self.total_ideas_generated = 0
        self.strategy_history: Dict[str, int] = {}

    def _next_id(self) -> str:
        with self._lock:
            self._id_counter += 1
            return f"idea_{self._id_counter}"

    def _extract_embeddings(self, memories: List[LivingMemory]) -> List[torch.Tensor]:
        """Extract embeddings from living memories."""
        return [m.content for m in memories if m.content is not None]

    def _extract_components(self, memories: List[LivingMemory]) -> List[IdeaComponent]:
        """Extract components from living memories for grafting."""
        components = []
        for m in memories:
            comp = IdeaComponent(
                id=str(m.id),
                embedding=m.content,
                component_type=ComponentType.CORE,
                label=m.label or f"memory_{m.id}",
                weight=m.energy
            )
            components.append(comp)
        return components

    def generate_flat_ideas(
        self,
        memories: List[LivingMemory],
        n_ideas: int,
        dopamine: float = 0.5
    ) -> List[CreativeIdea]:
        """Generate flat embedding ideas via curiosity-driven exploration."""
        embeddings = self._extract_embeddings(memories)

        wills = self.curiosity_will.batch_wills(
            embeddings,
            n_wills=n_ideas,
            dopamine=dopamine
        )

        ideas = []
        for will, probe in wills:
            # Generate idea embedding influenced by will
            if embeddings:
                # Interpolate between random source and will direction
                idx = torch.randint(len(embeddings), (1,)).item()
                base = embeddings[idx]
                idea_emb = safe_normalize(0.5 * base + 0.5 * will)
            else:
                idea_emb = will

            # Add temperature-scaled noise
            noise = torch.randn_like(idea_emb) * 0.1 * self.config.temperature
            idea_emb = safe_normalize(idea_emb + noise)

            idea = CreativeIdea(
                id=self._next_id(),
                form=IdeaForm.FLAT,
                embedding=idea_emb,
                novelty_score=probe.novelty_estimate,
                source_strategy=f"flat_{probe.strategy.name}"
            )
            ideas.append(idea)

        return ideas

    def generate_hierarchical_ideas(
        self,
        memories: List[LivingMemory],
        n_ideas: int,
        dopamine: float = 0.5
    ) -> List[CreativeIdea]:
        """Generate hierarchical tree-structured ideas."""
        embeddings = self._extract_embeddings(memories)
        components = self._extract_components(memories)

        ideas = []

        for _ in range(n_ideas):
            # Strategy selection based on dopamine
            if torch.rand(1).item() < dopamine:
                # High dopamine: create from scratch or merge
                if len(embeddings) >= 2 and torch.rand(1).item() < 0.5:
                    # Merge two ideas
                    indices = torch.randperm(len(embeddings))[:2]
                    idea_a = self.idea_factory.from_embedding(embeddings[indices[0]], n_components=2)
                    idea_b = self.idea_factory.from_embedding(embeddings[indices[1]], n_components=2)

                    strategy = ["combine", "graft", "blend"][torch.randint(3, (1,)).item()]
                    merged = self.idea_factory.merge(idea_a, idea_b, merge_strategy=strategy)
                    source_strategy = f"hierarchical_merge_{strategy}"
                else:
                    # Create from single embedding with random components
                    if embeddings:
                        idx = torch.randint(len(embeddings), (1,)).item()
                        base = embeddings[idx]
                    else:
                        base = torch.randn(self.config.content_dim)

                    merged = self.idea_factory.from_embedding(base, n_components=3 + int(dopamine * 2))
                    source_strategy = "hierarchical_create"
            else:
                # Low dopamine: mutate existing
                if embeddings:
                    idx = torch.randint(len(embeddings), (1,)).item()
                    base_idea = self.idea_factory.from_embedding(embeddings[idx], n_components=3)

                    # Apply grafting mutations
                    if components:
                        graft_results = self.grafter.mutate(
                            base_idea,
                            donor_pool=components,
                            dopamine=dopamine,
                            n_mutations=1 + int(dopamine * 2)
                        )
                        if graft_results:
                            merged = graft_results[-1].idea
                            source_strategy = f"hierarchical_mutate_{graft_results[-1].operation.name}"
                        else:
                            merged = base_idea
                            source_strategy = "hierarchical_base"
                    else:
                        merged = base_idea
                        source_strategy = "hierarchical_base"
                else:
                    merged = self.idea_factory.random()
                    source_strategy = "hierarchical_random"

            idea = CreativeIdea(
                id=self._next_id(),
                form=IdeaForm.HIERARCHICAL,
                hierarchical=merged,
                novelty_score=0.5,  # Will be recomputed
                source_strategy=source_strategy
            )
            ideas.append(idea)

        return ideas

    def generate_leap_ideas(
        self,
        memories: List[LivingMemory],
        n_ideas: int,
        dopamine: float = 0.5
    ) -> List[CreativeIdea]:
        """Generate ideas via creative leaps (analogical, orthogonal, diffusion)."""
        embeddings = self._extract_embeddings(memories)

        ideas = []

        # Generate diverse leaps
        leaps = self.leap_engine.batch_leap(
            embeddings,
            n_leaps=n_ideas,
            dopamine=dopamine,
            diversity_bonus=0.3
        )

        for leap in leaps:
            idea = CreativeIdea(
                id=self._next_id(),
                form=IdeaForm.LEAP,
                leap=leap,
                novelty_score=leap.novelty_score,
                source_strategy=f"leap_{leap.leap_type.name}"
            )
            ideas.append(idea)

        return ideas

    def score_idea(
        self,
        idea: CreativeIdea,
        memories: List[LivingMemory]
    ) -> Dict[str, float]:
        """Score an idea on novelty, coherence, and plausibility."""
        embedding = idea.to_embedding(self.config.content_dim)
        memory_embeddings = self._extract_embeddings(memories)

        scores = {}

        # Novelty: distance from existing memories
        if memory_embeddings:
            stacked = torch.stack(memory_embeddings)
            dists = (stacked - embedding.unsqueeze(0)).norm(dim=1)
            min_dist = dists.min().item()
            scores["novelty"] = min(1.0, min_dist / 2.0)
        else:
            scores["novelty"] = idea.novelty_score or 0.5

        # Repulsion penalty
        repulsion_score, _ = self.repulsion_field.compute_repulsion(embedding)
        scores["novelty"] *= (1.0 - 0.5 * repulsion_score)

        # Coherence: how well embedding fits known patterns
        if memory_embeddings:
            stacked = torch.stack(memory_embeddings)
            sims = F.cosine_similarity(embedding.unsqueeze(0), stacked)
            scores["coherence"] = sims.max().item()
        else:
            scores["coherence"] = 0.5

        # Plausibility from plausibility field
        # For hierarchical ideas, convert to structured format
        if idea.form == IdeaForm.HIERARCHICAL and idea.hierarchical:
            # Create structured memory from hierarchical idea
            from .imagination import MemorySlot, SlotType, StructuredMemory
            slots = {}
            for comp_id, comp in idea.hierarchical.components.items():
                # Map ComponentType to SlotType
                if comp.component_type == ComponentType.CORE:
                    slot_type = SlotType.AGENT
                elif comp.component_type == ComponentType.ACTION:
                    slot_type = SlotType.ACTION
                elif comp.component_type == ComponentType.PART:
                    slot_type = SlotType.PART
                else:
                    slot_type = SlotType.ATTRIBUTE
                slot_name = comp.label or comp_id
                slots[slot_name] = MemorySlot(
                    slot_type=slot_type,
                    name=slot_name,
                    content=comp.embedding,
                    confidence=comp.weight
                )
            struct = StructuredMemory(id=idea.id, slots=slots, valence=0.5)
            plaus_score = self.plausibility.score(struct)
            scores["plausibility"] = plaus_score.total_plausibility
        else:
            # For flat/leap, use embedding distance to centroid
            if self.plausibility.memory_centroid is not None:
                dist = (embedding - self.plausibility.memory_centroid).norm()
                scores["plausibility"] = max(0, 1 - dist.item() / 3)
            else:
                scores["plausibility"] = 0.5

        # Total score (weighted combination)
        scores["total"] = (
            0.35 * scores["novelty"]
            + 0.35 * scores["coherence"]
            + 0.30 * scores["plausibility"]
        )

        return scores

    def ideate(
        self,
        memories: List[LivingMemory],
        dopamine: float = 0.5,
        n_ideas: Optional[int] = None
    ) -> CreativeIdeationResult:
        """Run a creative ideation session.

        Args:
            memories: Source memories to draw from
            dopamine: 0-1 creativity intensity
            n_ideas: Total ideas to generate (default from config)

        Returns:
            CreativeIdeationResult with scored ideas
        """
        start_time = time.time()

        n_ideas = n_ideas or self.config.ideas_per_round

        # Compute counts for each type
        n_flat = max(1, int(n_ideas * self.config.flat_ratio))
        n_hierarchical = max(1, int(n_ideas * self.config.hierarchical_ratio))
        n_leap = max(1, int(n_ideas * self.config.leap_ratio))

        # Adjust to ensure we hit target
        total = n_flat + n_hierarchical + n_leap
        if total < n_ideas:
            n_leap += (n_ideas - total)

        all_ideas = []
        strategies_used: Dict[str, int] = {}
        leap_types_used: Dict[str, int] = {}

        # Generate flat ideas
        flat_ideas = self.generate_flat_ideas(memories, n_flat, dopamine)
        all_ideas.extend(flat_ideas)
        for idea in flat_ideas:
            strategies_used[idea.source_strategy] = strategies_used.get(idea.source_strategy, 0) + 1

        # Generate hierarchical ideas
        hier_ideas = self.generate_hierarchical_ideas(memories, n_hierarchical, dopamine)
        all_ideas.extend(hier_ideas)
        for idea in hier_ideas:
            strategies_used[idea.source_strategy] = strategies_used.get(idea.source_strategy, 0) + 1

        # Generate leap ideas
        leap_ideas = self.generate_leap_ideas(memories, n_leap, dopamine)
        all_ideas.extend(leap_ideas)
        for idea in leap_ideas:
            strategies_used[idea.source_strategy] = strategies_used.get(idea.source_strategy, 0) + 1
            if idea.leap:
                leap_type = idea.leap.leap_type.name
                leap_types_used[leap_type] = leap_types_used.get(leap_type, 0) + 1

        # Score all ideas
        for idea in all_ideas:
            scores = self.score_idea(idea, memories)
            idea.novelty_score = scores["novelty"]
            idea.coherence_score = scores["coherence"]
            idea.plausibility_score = scores["plausibility"]
            idea.total_score = scores["total"]
            idea.generation_time = time.time() - start_time

        # Filter by thresholds
        filtered_ideas = [
            idea for idea in all_ideas
            if idea.novelty_score >= self.config.min_novelty
            and idea.coherence_score >= self.config.min_coherence
            and idea.plausibility_score >= self.config.min_plausibility
        ]

        # Sort by total score
        filtered_ideas.sort(key=lambda i: i.total_score, reverse=True)

        # Select best
        best_idea = filtered_ideas[0] if filtered_ideas else None
        best_score = best_idea.total_score if best_idea else 0.0

        # Mark explored regions
        for idea in all_ideas:
            emb = idea.to_embedding(self.config.content_dim)
            was_good = idea.total_score >= 0.5
            self.repulsion_field.mark_explored(
                emb,
                quality_score=idea.total_score,
                was_productive=was_good
            )

        # Learn from good ideas
        if filtered_ideas:
            from .imagination import MemorySlot, SlotType, StructuredMemory
            for idea in filtered_ideas[:5]:  # Top 5
                if idea.form == IdeaForm.HIERARCHICAL and idea.hierarchical:
                    slots = {}
                    for comp_id, comp in idea.hierarchical.components.items():
                        # Map ComponentType to SlotType
                        if comp.component_type == ComponentType.CORE:
                            slot_type = SlotType.AGENT
                        elif comp.component_type == ComponentType.ACTION:
                            slot_type = SlotType.ACTION
                        elif comp.component_type == ComponentType.PART:
                            slot_type = SlotType.PART
                        else:
                            slot_type = SlotType.ATTRIBUTE
                        slot_name = comp.label or comp_id
                        slots[slot_name] = MemorySlot(
                            slot_type=slot_type,
                            name=slot_name,
                            content=comp.embedding,
                            confidence=comp.weight
                        )
                    struct = StructuredMemory(id=idea.id, slots=slots, valence=0.5)
                    self.plausibility.learn_from_structured([struct])

        # Update statistics
        self.session_count += 1
        self.total_ideas_generated += len(all_ideas)
        for strategy, count in strategies_used.items():
            self.strategy_history[strategy] = self.strategy_history.get(strategy, 0) + count

        # Compute averages
        avg_novelty = sum(i.novelty_score for i in all_ideas) / len(all_ideas) if all_ideas else 0
        avg_coherence = sum(i.coherence_score for i in all_ideas) / len(all_ideas) if all_ideas else 0

        session_time = time.time() - start_time

        return CreativeIdeationResult(
            ideas=filtered_ideas,
            best_idea=best_idea,
            best_score=best_score,
            total_generated=len(all_ideas),
            strategies_used=strategies_used,
            average_novelty=avg_novelty,
            average_coherence=avg_coherence,
            leap_types_used=leap_types_used,
            session_time=session_time
        )

    def consolidate_to_memories(
        self,
        ideas: List[CreativeIdea],
        quality_threshold: float = 0.5
    ) -> List[LivingMemory]:
        """Convert high-quality ideas into living memories."""
        from .living_memory import NarrativePhase, ValenceTrajectory

        new_memories = []

        for idea in ideas:
            if idea.total_score < quality_threshold:
                continue

            embedding = idea.to_embedding(self.config.content_dim)

            # Derive valence from scores
            valence = 0.3 * idea.novelty_score + 0.4 * idea.coherence_score + 0.3 * idea.plausibility_score

            # Create valence trajectory from onset -> peak -> resolution
            valence_points = torch.tensor([
                max(0.0, valence - 0.1),  # onset
                valence,                   # peak
                min(1.0, valence + 0.1)   # resolution
            ])
            memory = LivingMemory(
                id=hash(idea.id) % (2**31),
                content=embedding,
                valence=ValenceTrajectory(points=valence_points),
                energy=0.8,  # New ideas are energetic
                created_at=time.time(),
                phase=NarrativePhase.SETUP,  # New ideas start at setup phase
                label=f"generated_{idea.source_strategy}"
            )
            new_memories.append(memory)

        return new_memories

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "session_count": self.session_count,
            "total_ideas_generated": self.total_ideas_generated,
            "strategy_history": self.strategy_history,
            "leap_engine_stats": self.leap_engine.get_statistics(),
            "curiosity_stats": self.curiosity_prober.get_curiosity_statistics(),
            "repulsion_field_size": len(self.repulsion_field.explored_regions)
        }


def run_creative_ideation_demo(n_rounds: int = 5, verbose: bool = True) -> CreativeIdeationResult:
    """Run a demonstration of creative ideation.

    Args:
        n_rounds: Number of ideation rounds
        verbose: Print progress

    Returns:
        Final ideation result
    """
    from .living_memory import ValenceTrajectory

    # Create engine
    config = CreativeIdeationConfig.balanced()
    engine = CreativeIdeationEngine(config)

    # Create initial memories
    memories = []
    for i in range(15):
        mem = LivingMemory(
            id=i,
            content=safe_normalize(torch.randn(config.content_dim)),
            valence=ValenceTrajectory.random(),
            energy=0.7 + 0.3 * torch.rand(1).item(),
            created_at=time.time() - i * 60,
            label=f"seed_memory_{i}"
        )
        memories.append(mem)

    if verbose:
        print(f"Starting creative ideation demo with {len(memories)} seed memories")
        print("-" * 60)

    all_results = []
    dopamine = 0.5

    for round_idx in range(n_rounds):
        # Vary dopamine to simulate mood
        dopamine = 0.3 + 0.4 * math.sin(round_idx * 0.5) + 0.3

        result = engine.ideate(memories, dopamine=dopamine)
        all_results.append(result)

        if verbose:
            print(f"\nRound {round_idx + 1} (dopamine={dopamine:.2f}):")
            print(f"  Generated: {result.total_generated} ideas")
            print(f"  Filtered: {len(result.ideas)} ideas")
            print(f"  Best score: {result.best_score:.3f}")
            print(f"  Avg novelty: {result.average_novelty:.3f}")
            print(f"  Avg coherence: {result.average_coherence:.3f}")
            print(f"  Strategies: {dict(list(result.strategies_used.items())[:3])}")
            print(f"  Leap types: {result.leap_types_used}")

            if result.best_idea:
                print(f"  Best idea: {result.best_idea.source_strategy}")

        # Consolidate good ideas as new memories
        new_memories = engine.consolidate_to_memories(result.ideas[:3])
        memories.extend(new_memories)

        if verbose and new_memories:
            print(f"  Consolidated {len(new_memories)} new memories")

    if verbose:
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        stats = engine.get_statistics()
        print(f"Total sessions: {stats['session_count']}")
        print(f"Total ideas: {stats['total_ideas_generated']}")
        print(f"Final memory count: {len(memories)}")
        print("\nStrategy usage:")
        for strat, count in sorted(stats['strategy_history'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {strat}: {count}")

    return all_results[-1] if all_results else None
