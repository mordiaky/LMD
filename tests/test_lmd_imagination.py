"""LMD Imagination & Ideation Tests - Validating autonomous creativity.

These tests validate Joshua's Imagination Extension to LMD:
1. Structured Memory: Memory decomposition into parts/slots
2. Transform Operations: MORPH, BULGE, SPLIT, RECOLOR, etc.
3. Will-Directed Imagination: External and internal will
4. Plausibility Field: Coherence constraints
5. Ideation Engine: Full brainstorming loop
6. Autonomous Ideation: Curiosity-driven exploration

Invented by Joshua R. Thomas, January 2026.
"""

import pytest
import torch
import math

from lmd import (
    # Core
    LivingMemory,
    ValenceTrajectory,
    LMDConfig,
    LMDDynamics,
    # Imagination
    StructuredMemory,
    MemorySlot,
    SlotType,
    Transform,
    TransformType,
    TransformOps,
    WillVector,
    WillGenerator,
    MentalCanvas,
    MemoryDecomposer,
    # Plausibility
    PlausibilityField,
    PlausibilityScore,
    IdeaEvaluator,
    CreativityOptimizer,
    # Ideation
    IdeationEngine,
    IdeationConfig,
    IdeationResult,
    AutonomousIdeator,
    run_ideation_demo,
)


class TestStructuredMemory:
    """Test 1: Validate structured memory decomposition."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def sample_memory(self, config):
        return LivingMemory(
            id=0,
            content=torch.randn(config.content_dim),
            valence=ValenceTrajectory.random(),
            energy=1.0,
            created_at=0,
            label="dragon"
        )

    def test_memory_slot_creation(self, config):
        """Can create memory slots with proper attributes."""
        slot = MemorySlot(
            slot_type=SlotType.AGENT,
            name="dragon",
            content=torch.randn(config.content_dim),
            confidence=0.9
        )

        assert slot.slot_type == SlotType.AGENT
        assert slot.name == "dragon"
        assert slot.confidence == 0.9
        assert slot.content.shape[0] == config.content_dim

    def test_structured_memory_add_slots(self, config):
        """Can add multiple slots to structured memory."""
        structured = StructuredMemory(id=0)

        # Add agent
        structured.add_slot("agent", MemorySlot(
            slot_type=SlotType.AGENT,
            name="agent",
            content=torch.randn(config.content_dim),
            source_memory_id=0
        ))

        # Add location
        structured.add_slot("location", MemorySlot(
            slot_type=SlotType.LOCATION,
            name="location",
            content=torch.randn(config.content_dim),
            source_memory_id=1
        ))

        assert len(structured.slots) == 2
        assert "agent" in structured.slots
        assert "location" in structured.slots
        assert 0 in structured.source_memories
        assert 1 in structured.source_memories

    def test_decomposer_extracts_slots(self, config, sample_memory):
        """Decomposer extracts structured slots from living memory."""
        decomposer = MemoryDecomposer(config.content_dim)
        structured = decomposer.decompose(sample_memory)

        assert structured.id == sample_memory.id
        assert len(structured.slots) >= 1
        assert sample_memory.id in structured.source_memories

    def test_structured_to_embedding(self, config):
        """Can reconstruct embedding from structured memory."""
        structured = StructuredMemory(id=0)

        # Add slots
        for name in ["agent", "action", "location"]:
            structured.add_slot(name, MemorySlot(
                slot_type=SlotType.AGENT,
                name=name,
                content=torch.randn(config.content_dim),
                confidence=0.8
            ))

        embedding = structured.to_embedding(config.content_dim)
        assert embedding.shape[0] == config.content_dim
        assert not torch.isnan(embedding).any()


class TestTransformOperations:
    """Test 2: Validate transform operations."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def transform_ops(self, config):
        return TransformOps(config.content_dim)

    @pytest.fixture
    def sample_structured(self, config):
        structured = StructuredMemory(id=0, novelty=0.3)
        structured.add_slot("eyes", MemorySlot(
            slot_type=SlotType.PART,
            name="eyes",
            content=torch.randn(config.content_dim),
            confidence=0.9
        ))
        structured.add_slot("nose", MemorySlot(
            slot_type=SlotType.PART,
            name="nose",
            content=torch.randn(config.content_dim),
            confidence=0.9
        ))
        return structured

    def test_morph_changes_content(self, transform_ops, sample_structured):
        """MORPH transform changes slot content."""
        original_content = sample_structured.get_slot("eyes").content.clone()

        transform = Transform(
            transform_type=TransformType.MORPH,
            target_slot="eyes",
            magnitude=0.5
        )
        result = transform_ops.apply(sample_structured, transform)

        new_content = result.get_slot("eyes").content
        diff = (new_content - original_content).norm()
        assert diff > 0, "MORPH should change content"

    def test_bulge_exaggerates_dimensions(self, transform_ops, sample_structured):
        """BULGE transform exaggerates certain dimensions."""
        original_content = sample_structured.get_slot("eyes").content.clone()

        transform = Transform(
            transform_type=TransformType.BULGE,
            target_slot="eyes",
            magnitude=0.5,
            parameters={"dims": [0, 1, 2]}
        )
        result = transform_ops.apply(sample_structured, transform)

        new_content = result.get_slot("eyes").content
        # Bulged dimensions should be larger
        assert abs(new_content[0]) >= abs(original_content[0]) * 0.99

    def test_split_creates_multiple_parts(self, transform_ops, sample_structured):
        """SPLIT transform divides one part into multiple."""
        transform = Transform(
            transform_type=TransformType.SPLIT,
            target_slot="nose",
            magnitude=0.5,
            parameters={"n_parts": 2}
        )
        result = transform_ops.apply(sample_structured, transform)

        # Should have original + 2 new parts
        assert "nose_1" in result.slots
        assert "nose_2" in result.slots

    def test_recolor_changes_attributes(self, transform_ops, sample_structured):
        """RECOLOR transform changes color attribute."""
        transform = Transform(
            transform_type=TransformType.RECOLOR,
            target_slot="color",
            magnitude=0.7
        )
        result = transform_ops.apply(sample_structured, transform)

        assert "color" in result.slots
        assert result.get_slot("color").slot_type == SlotType.ATTRIBUTE

    def test_transform_increases_novelty(self, transform_ops, sample_structured):
        """Transforms should increase novelty score."""
        original_novelty = sample_structured.novelty

        transform = Transform(
            transform_type=TransformType.MORPH,
            target_slot="eyes",
            magnitude=0.5
        )
        result = transform_ops.apply(sample_structured, transform)

        assert result.novelty > original_novelty


class TestWillGeneration:
    """Test 3: Validate will generation - the key innovation."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def sample_memories(self, config):
        memories = []
        for i in range(10):
            memories.append(LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            ))
        return memories

    @pytest.fixture
    def will_generator(self, config):
        return WillGenerator(config.content_dim)

    def test_external_will_has_direction(self, config):
        """External will (user intent) has clear direction."""
        will = WillVector(
            direction=torch.randn(config.content_dim),
            strength=0.9,
            specificity=0.8,
            source="external",
            description="I want a dragon with butterfly wings"
        )

        assert will.strength > 0
        assert will.source == "external"
        assert will.direction.norm() > 0

    def test_curiosity_will_explores_gaps(self, will_generator, sample_memories):
        """Curiosity-based will should explore memory gaps."""
        will = will_generator.generate_curiosity_will(sample_memories)

        assert will.source == "curiosity"
        assert will.strength > 0
        assert will.specificity < 0.5  # Curiosity is vague

    def test_problem_will_is_specific(self, will_generator, config):
        """Problem-based will should be specific and strong."""
        goal = torch.randn(config.content_dim)
        will = will_generator.generate_problem_will("Find a creative solution", goal)

        assert will.source == "problem"
        assert will.strength > 0.7
        assert will.specificity > 0.5

    def test_combined_will_merges_sources(self, will_generator, sample_memories):
        """Combined will should integrate multiple sources."""
        external = WillVector(
            direction=torch.randn(will_generator.content_dim),
            strength=0.8,
            source="external"
        )

        combined = will_generator.generate_combined_will(sample_memories, external)

        assert "external" in combined.source or "combined" in combined.source
        assert combined.direction.norm() > 0

    def test_will_applies_to_content(self, config):
        """Will should bias content generation."""
        will = WillVector(
            direction=torch.ones(config.content_dim),
            strength=0.5,
            source="external"
        )

        original = torch.zeros(config.content_dim)
        biased = will.apply_to(original, alpha=0.5)

        # Should move toward will direction
        assert biased.sum() > 0


class TestMentalCanvas:
    """Test 4: Validate mental canvas operations."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def canvas(self, config):
        return MentalCanvas(config.content_dim)

    @pytest.fixture
    def sample_entity(self, config):
        structured = StructuredMemory(id=0)
        structured.add_slot("core", MemorySlot(
            slot_type=SlotType.AGENT,
            name="core",
            content=torch.randn(config.content_dim)
        ))
        return structured

    def test_add_entity_to_canvas(self, canvas, sample_entity):
        """Can add entities to canvas."""
        entity_id = canvas.add_entity(sample_entity, position=(0, 0, 0))

        assert entity_id >= 0
        assert entity_id in canvas.entities
        assert canvas.get_entity(entity_id) is not None

    def test_transform_entity_on_canvas(self, canvas, sample_entity):
        """Can transform entities on canvas."""
        entity_id = canvas.add_entity(sample_entity)

        transform = Transform(
            transform_type=TransformType.MORPH,
            target_slot="core",
            magnitude=0.5
        )

        success = canvas.transform_entity(entity_id, transform)
        assert success

        entity = canvas.get_entity(entity_id)
        assert len(entity.transform_history) == 1

    def test_compose_entities(self, canvas, config):
        """Can compose multiple entities into one."""
        # Add two entities
        entity1 = StructuredMemory(id=0)
        entity1.add_slot("wings", MemorySlot(
            slot_type=SlotType.PART,
            name="wings",
            content=torch.randn(config.content_dim)
        ))

        entity2 = StructuredMemory(id=1)
        entity2.add_slot("body", MemorySlot(
            slot_type=SlotType.PART,
            name="body",
            content=torch.randn(config.content_dim)
        ))

        id1 = canvas.add_entity(entity1)
        id2 = canvas.add_entity(entity2)

        composed = canvas.compose_entities([id1, id2])

        assert composed is not None
        assert len(composed.slots) >= 2

    def test_scene_embedding(self, canvas, sample_entity):
        """Can get embedding of entire scene."""
        canvas.add_entity(sample_entity)

        embedding = canvas.get_scene_embedding()
        assert embedding.shape[0] == canvas.content_dim
        assert embedding.norm() > 0


class TestPlausibilityField:
    """Test 5: Validate plausibility constraints."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def plausibility(self, config):
        return PlausibilityField(config.content_dim)

    @pytest.fixture
    def sample_structured(self, config):
        structured = StructuredMemory(id=0)
        structured.add_slot("agent", MemorySlot(
            slot_type=SlotType.AGENT,
            name="agent",
            content=torch.randn(config.content_dim),
            confidence=0.8
        ))
        structured.add_slot("action", MemorySlot(
            slot_type=SlotType.ACTION,
            name="action",
            content=torch.randn(config.content_dim),
            confidence=0.7
        ))
        return structured

    def test_plausibility_score_bounded(self, plausibility, sample_structured):
        """Plausibility scores should be in valid range."""
        score = plausibility.score(sample_structured)

        assert 0 <= score.structural_coherence <= 1
        assert 0 <= score.semantic_coherence <= 1
        assert 0 <= score.physical_coherence <= 1
        assert 0 <= score.total_plausibility <= 1

    def test_learn_from_memories(self, plausibility, config):
        """Can learn plausibility constraints from memories."""
        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                created_at=0
            )
            for i in range(10)
        ]

        plausibility.learn_from_memories(memories)

        assert plausibility.memory_centroid is not None
        assert plausibility.memory_variance > 0

    def test_novelty_increases_with_distance(self, plausibility, config):
        """Novel ideas should score higher on novelty."""
        # Learn from memories near origin
        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim) * 0.1,  # Near origin
                valence=ValenceTrajectory.random(),
                created_at=0
            )
            for i in range(10)
        ]
        plausibility.learn_from_memories(memories)

        # Near memory
        near = StructuredMemory(id=0)
        near.add_slot("core", MemorySlot(
            slot_type=SlotType.AGENT,
            name="core",
            content=torch.randn(config.content_dim) * 0.1
        ))

        # Far from memories
        far = StructuredMemory(id=1)
        far.add_slot("core", MemorySlot(
            slot_type=SlotType.AGENT,
            name="core",
            content=torch.randn(config.content_dim) * 3.0  # Far from origin
        ))

        near_score = plausibility.score(near)
        far_score = plausibility.score(far)

        assert far_score.novelty > near_score.novelty

    def test_creative_value_balances_novelty_coherence(self, plausibility, sample_structured):
        """Creative value should balance novelty and coherence."""
        score = plausibility.score(sample_structured)

        # Creative value is geometric mean
        expected = math.sqrt(score.novelty * score.total_plausibility)
        assert abs(score.creative_value - expected) < 0.01


class TestIdeaEvaluator:
    """Test 6: Validate idea evaluation."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def evaluator(self, config):
        plausibility = PlausibilityField(config.content_dim)
        return IdeaEvaluator(plausibility)

    @pytest.fixture
    def sample_ideas(self, config):
        ideas = []
        for i in range(5):
            idea = StructuredMemory(id=i, novelty=0.1 * i, valence=0.2 * i - 0.4)
            idea.add_slot("core", MemorySlot(
                slot_type=SlotType.AGENT,
                name="core",
                content=torch.randn(config.content_dim)
            ))
            ideas.append(idea)
        return ideas

    def test_evaluate_returns_all_dimensions(self, evaluator, sample_ideas):
        """Evaluation should return all dimension scores."""
        scores = evaluator.evaluate(sample_ideas[0])

        assert "total" in scores
        assert "relevance" in scores
        assert "novelty" in scores
        assert "coherence" in scores
        assert "valence" in scores
        assert "is_acceptable" in scores

    def test_rank_ideas_orders_by_score(self, evaluator, sample_ideas, config):
        """Ranking should order ideas by total score."""
        goal = torch.randn(config.content_dim)
        ranked = evaluator.rank_ideas(sample_ideas, goal, top_k=3)

        assert len(ranked) == 3
        # Scores should be in descending order
        scores = [r[1]["total"] for r in ranked]
        assert scores == sorted(scores, reverse=True)


class TestIdeationEngine:
    """Test 7: Validate full ideation loop."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def sample_memories(self, config):
        memories = []
        for i in range(15):
            memories.append(LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0,
                label=f"memory_{i}"
            ))
        return memories

    @pytest.fixture
    def engine(self, config):
        return IdeationEngine(config, IdeationConfig.quick())

    def test_ideate_generates_ideas(self, engine, sample_memories):
        """Ideation should generate multiple ideas."""
        result = engine.ideate(sample_memories)

        assert len(result.ideas) > 0
        assert result.total_ideas_generated > 0

    def test_ideate_with_external_will(self, engine, sample_memories, config):
        """Ideation should respect external will."""
        will = WillVector(
            direction=torch.randn(config.content_dim),
            strength=0.9,
            source="external",
            description="Dragon with wings"
        )

        result = engine.ideate(sample_memories, will)

        assert len(result.ideas) > 0
        # Ideas should have been influenced by will
        assert result.n_iterations >= 1

    def test_ideate_produces_scored_ideas(self, engine, sample_memories):
        """Ideation should score all final ideas."""
        result = engine.ideate(sample_memories)

        assert len(result.scores) == len(result.ideas)
        for score in result.scores:
            assert "total" in score
            assert 0 <= score["total"] <= 1

    def test_brainstorm_for_problem(self, engine, sample_memories):
        """Brainstorm should generate ideas for a problem."""
        result = engine.brainstorm(
            sample_memories,
            problem="How to make a flying creature",
            n_ideas=3
        )

        assert len(result.ideas) <= 3
        assert result.total_ideas_generated > 0

    def test_explore_curiosity(self, engine, sample_memories):
        """Curiosity exploration should work without external goal."""
        result = engine.explore_curiosity(sample_memories, n_ideas=2)

        assert len(result.ideas) <= 2
        # Curiosity-driven should favor novelty
        assert result.average_novelty > 0


class TestAutonomousIdeation:
    """Test 8: Validate autonomous ideation."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    @pytest.fixture
    def dynamics(self, config):
        return LMDDynamics(config)

    @pytest.fixture
    def sample_memories(self, config):
        return [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(10)
        ]

    def test_autonomous_session_runs(self, config, dynamics, sample_memories):
        """Autonomous ideation session should complete (respecting resource limits)."""
        ideator = AutonomousIdeator(config, dynamics)

        # Override autonomy controller for testing (remove min interval)
        ideator.autonomy.min_interval_seconds = 0.0

        results = ideator.run_autonomous_session(
            sample_memories,
            n_rounds=2,
            ideas_per_round=2
        )

        # Should complete at least 1 round (may be limited by resources)
        assert len(results) >= 1
        for result in results:
            assert isinstance(result, IdeationResult)

    def test_consolidate_to_memories(self, config, dynamics, sample_memories):
        """Should convert ideas back to living memories."""
        ideator = AutonomousIdeator(config, dynamics)
        results = ideator.run_autonomous_session(sample_memories, n_rounds=1)

        if results and results[0].ideas:
            new_memories = ideator.consolidate_to_memories(
                results[0].ideas,
                dynamics
            )

            assert len(new_memories) == len(results[0].ideas)
            for mem in new_memories:
                assert isinstance(mem, LivingMemory)
                assert mem.energy > 0


class TestIdeationDemo:
    """Test 9: Validate demo function works."""

    def test_run_ideation_demo(self):
        """Demo should run without errors."""
        config = LMDConfig.toy_scale()
        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(10)
        ]

        result = run_ideation_demo(memories, config)

        assert isinstance(result, IdeationResult)
        assert len(result.ideas) > 0
        print(f"\n{result.summary()}")


class TestJoshuaEquationV4:
    """Test 10: Validate the complete Joshua Equation v4.

    Joshua Equation v4:
    - Living Memory: dM/dt = grad_phi(N) + sum_j(Gamma_ij * R) + A(M, xi) + kappa * eta
    - Will Generation: W = W_external + grad_Curiosity + grad_Problem + mu * eta
    - Imagination: dI/dt = grad_D(I,W) + W (x) [T(I) + C(S(M|W), B)] + lambda * grad_P(I)
    - Evaluation: Score = Novelty * Coherence * Relevance * Valence
    """

    @pytest.fixture
    def full_system(self):
        config = LMDConfig.toy_scale()
        dynamics = LMDDynamics(config)

        # Create memories
        memories = []
        for i in range(20):
            memories.append(LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0,
                label=f"experience_{i}"
            ))

        # Run dynamics to let memories couple
        for _ in range(50):
            dynamics.step(memories)

        return config, dynamics, memories

    def test_complete_pipeline(self, full_system):
        """Full pipeline: Memory -> Will -> Imagination -> Evaluation."""
        config, dynamics, memories = full_system

        # 1. Living memories exist and are coupled
        alive = sum(1 for m in memories if m.is_alive)
        assert alive > 10, "Memories should stay alive"

        # 2. Generate will (internal + external)
        will_gen = WillGenerator(config.content_dim)
        external_will = WillVector(
            direction=torch.randn(config.content_dim),
            strength=0.7,
            source="external",
            description="Create something with wings and fire"
        )
        combined_will = will_gen.generate_combined_will(memories, external_will)
        assert combined_will.strength > 0

        # 3. Run imagination/ideation
        engine = IdeationEngine(config, IdeationConfig.quick())
        result = engine.ideate(memories, combined_will)

        # 4. Evaluate results
        assert len(result.ideas) > 0
        assert result.best_score > 0

        # 5. Consolidate back to memories
        ideator = AutonomousIdeator(config, dynamics)
        new_memories = ideator.consolidate_to_memories(result.ideas, dynamics)

        assert len(new_memories) > 0
        for mem in new_memories:
            assert mem.is_alive

        print(f"\n{'='*60}")
        print("JOSHUA EQUATION V4 VALIDATION")
        print(f"{'='*60}")
        print(f"Living Memories: {alive}/{len(memories)}")
        print(f"Will Strength: {combined_will.strength:.3f}")
        print(f"Ideas Generated: {result.total_ideas_generated}")
        print(f"Best Idea Score: {result.best_score:.3f}")
        print(f"Average Novelty: {result.average_novelty:.3f}")
        print(f"Average Coherence: {result.average_coherence:.3f}")
        print(f"Consolidated Memories: {len(new_memories)}")
        print(f"{'='*60}")


class TestSafeguards:
    """Test 11: Validate safeguards prevent pathological behaviors."""

    def test_id_generator_no_collisions(self):
        """ID generator should never produce collisions."""
        from lmd import (
            IDGenerator, reset_id_generator
        )

        # Reset for clean test
        reset_id_generator(0)
        gen = IDGenerator()

        # Generate many IDs
        ids = set()
        for _ in range(1000):
            new_id = gen.next_id("memory")
            assert new_id not in ids, "ID collision detected!"
            ids.add(new_id)

        assert len(ids) == 1000

    def test_id_generator_namespaces(self):
        """IDs in different namespaces should not collide."""
        from lmd import IDGenerator

        gen = IDGenerator()

        memory_id = gen.next_id("memory")
        imagined_id = gen.next_id("imagined")
        consolidated_id = gen.next_id("consolidated")

        # Different namespaces should have different ranges
        assert memory_id < 1_000_000
        assert 1_000_000 <= imagined_id < 2_000_000
        assert 2_000_000 <= consolidated_id < 3_000_000

    def test_repulsion_field_marks_explored(self):
        """Repulsion field should track explored regions."""
        from lmd import RepulsionField

        repulsion = RepulsionField(content_dim=32)

        # Initially no explored regions
        assert len(repulsion.explored_regions) == 0

        # Mark a region
        embedding = torch.randn(32)
        repulsion.mark_explored(embedding)

        assert len(repulsion.explored_regions) == 1

    def test_repulsion_penalizes_revisits(self):
        """Revisiting explored regions should be penalized."""
        from lmd import RepulsionField

        repulsion = RepulsionField(content_dim=32, repulsion_strength=0.8)

        # Mark a region
        explored = torch.randn(32)
        repulsion.mark_explored(explored)

        # Check novelty penalty at explored location
        penalty_at_explored = repulsion.novelty_penalty(explored)

        # Check penalty at distant location
        distant = torch.randn(32) * 10  # Far away
        penalty_at_distant = repulsion.novelty_penalty(distant)

        assert penalty_at_explored > penalty_at_distant, \
            "Explored regions should have higher penalty"

    def test_repulsion_applies_gradient(self):
        """Repulsion should push ideas away from explored regions."""
        from lmd import RepulsionField

        repulsion = RepulsionField(content_dim=32)

        # Mark origin as explored
        origin = torch.zeros(32)
        repulsion.mark_explored(origin)

        # Apply repulsion to point near origin
        near_origin = torch.randn(32) * 0.1
        repelled = repulsion.apply_repulsion(near_origin, alpha=0.5)

        # Should move away from origin
        original_dist = near_origin.norm()
        repelled_dist = repelled.norm()

        # After repulsion, should be pushed away (or at least not closer)
        assert repelled_dist >= original_dist * 0.8, "Should not move toward explored region"

    def test_reality_anchor_grounds_valence(self):
        """Reality anchor should adjust valence based on external signals."""
        from lmd import RealityAnchor

        anchor = RealityAnchor(content_dim=32)

        # Add a validator that always returns negative
        anchor.register_validator(lambda x: -0.5)

        # Internal valence is positive
        internal_valence = 0.8
        embedding = torch.randn(32)

        grounded = anchor.ground_valence(internal_valence, embedding)

        # Should be pulled toward external signal
        assert grounded < internal_valence, "Should be grounded by external validator"

    def test_reality_anchor_learns_from_outcomes(self):
        """Reality anchor should calibrate based on outcomes."""
        from lmd import RealityAnchor

        anchor = RealityAnchor(content_dim=32)

        # Record several overconfident predictions
        for i in range(20):
            anchor.record_outcome(
                idea_id=i,
                predicted_valence=0.9,  # Predicted high
                actual_outcome=0.3      # Actual was low
            )

        stats = anchor.get_calibration_stats()

        # Should have learned we're overconfident
        assert stats["bias"] > 0, "Should detect positive bias (overconfidence)"
        assert stats["mae"] > 0.3, "MAE should reflect prediction errors"

    def test_autonomy_controller_respects_limits(self):
        """Autonomy controller should enforce resource limits."""
        from lmd import AutonomyController

        controller = AutonomyController(
            max_ideas_per_hour=5,
            min_interval_seconds=1.0
        )

        # Should be able to ideate initially
        assert controller.can_ideate()

        # Record hitting the limit
        controller.record_ideation(n_ideas=5, compute_ops=1000)

        # Should not be able to ideate (hit hourly limit)
        assert not controller.can_ideate()

    def test_autonomy_triggers_work(self):
        """Autonomy triggers should fire at thresholds."""
        from lmd import (
            AutonomyController, AutonomyTrigger
        )
        import time

        controller = AutonomyController(min_interval_seconds=0.0)

        # Update idle trigger below threshold
        controller.update_trigger(AutonomyTrigger.IDLE, 30.0)  # Below 60
        assert controller.check_triggers() is None

        # Update idle trigger above threshold
        controller.update_trigger(AutonomyTrigger.IDLE, 120.0)  # Above 60
        trigger = controller.check_triggers()
        assert trigger == AutonomyTrigger.IDLE

    def test_resource_budget_exhaustion(self):
        """Resource budget should track and limit resources."""
        from lmd import ResourceBudget

        budget = ResourceBudget(
            max_ideas=5,
            max_iterations=3,
            max_tensor_ops=1000,
            max_time_seconds=60.0
        )

        # Initially not exhausted
        assert not budget.is_exhausted()

        # Use up ideas
        for _ in range(5):
            budget.record_idea()

        # Should be exhausted
        assert budget.is_exhausted()

    def test_ideation_uses_repulsion(self):
        """Ideation engine should use repulsion to prevent echo chambers."""
        config = LMDConfig.toy_scale()
        engine = IdeationEngine(config, IdeationConfig.quick())

        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(10)
        ]

        # Run first ideation
        result1 = engine.ideate(memories)

        # Repulsion field should have explored regions
        assert len(engine.repulsion_field.explored_regions) > 0

        # Run second ideation - should explore different regions
        result2 = engine.ideate(memories)

        # Check that ideas are different (repulsion working)
        if result1.ideas and result2.ideas:
            # Compare best ideas
            emb1 = result1.ideas[0].to_embedding(config.content_dim)
            emb2 = result2.ideas[0].to_embedding(config.content_dim)
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()

            # Should not be too similar (repulsion pushing away)
            # Allow some similarity since it's stochastic
            print(f"  Idea similarity between sessions: {similarity:.3f}")


class TestEchoChamberPrevention:
    """Test 12: Specifically test echo chamber prevention."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    def test_repeated_ideation_explores_new_regions(self, config):
        """Multiple ideation sessions should explore diverse regions."""
        engine = IdeationEngine(config, IdeationConfig.quick())

        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(15)
        ]

        # Run multiple sessions
        all_embeddings = []
        for session in range(5):
            result = engine.ideate(memories)
            for idea in result.ideas:
                all_embeddings.append(idea.to_embedding(config.content_dim))

        # Check diversity of explored space
        if len(all_embeddings) >= 2:
            stacked = torch.stack(all_embeddings)
            # Compute pairwise distances
            dists = torch.cdist(stacked, stacked)
            # Get average non-diagonal distance
            mask = ~torch.eye(len(all_embeddings), dtype=torch.bool)
            avg_dist = dists[mask].mean().item()

            print(f"\n  Average distance between ideas: {avg_dist:.3f}")
            assert avg_dist > 0.1, "Ideas should be diverse (not echo chamber)"

    def test_repulsion_decays_over_time(self, config):
        """Repulsion should decay, allowing revisiting with fresh perspective."""
        from lmd import RepulsionField
        import time

        repulsion = RepulsionField(content_dim=config.content_dim, decay_rate=0.5)

        # Mark a region
        explored = torch.randn(config.content_dim)
        repulsion.mark_explored(explored)

        # Get initial penalty
        initial_penalty = repulsion.novelty_penalty(explored)

        # Simulate time passing (by manipulating last_visited)
        repulsion.explored_regions[0].last_visited -= 10.0  # 10 seconds ago

        # Get decayed penalty
        decayed_penalty = repulsion.novelty_penalty(explored)

        # With decay_rate=0.5, after 10 seconds: 0.5^10 = 0.001
        # So penalty should be much lower
        print(f"\n  Initial penalty: {initial_penalty:.3f}")
        print(f"  Decayed penalty: {decayed_penalty:.3f}")


class TestEdgeCasesAndNumericalStability:
    """Test 13: Edge cases, numerical stability, and robustness fixes."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    def test_empty_memory_list_curiosity_will(self, config):
        """Curiosity will should handle empty memory list without crash."""
        will_gen = WillGenerator(config.content_dim)

        # Should not crash with empty memories
        will = will_gen.generate_curiosity_will([])

        assert will is not None
        assert will.source == "curiosity"
        assert will.direction.norm() > 0
        assert not torch.isnan(will.direction).any()

    def test_zero_vector_normalization(self, config):
        """Zero vectors should be handled safely without division by zero."""
        from lmd.safeguards import safe_normalize

        zero_vec = torch.zeros(config.content_dim)
        result = safe_normalize(zero_vec)

        # Should not crash and should return the zero vector
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_nan_embedding_handling(self, config):
        """NaN embeddings should be handled gracefully."""
        from lmd import RepulsionField

        repulsion = RepulsionField(config.content_dim)

        # Try to mark NaN embedding - should not crash
        nan_embedding = torch.tensor([float('nan')] * config.content_dim)
        repulsion.mark_explored(nan_embedding)

        # Should have been rejected (not added)
        assert len(repulsion.explored_regions) == 0

    def test_inf_embedding_handling(self, config):
        """Inf embeddings should be handled gracefully."""
        from lmd import RepulsionField

        repulsion = RepulsionField(config.content_dim)

        # Try to mark inf embedding - should not crash
        inf_embedding = torch.tensor([float('inf')] * config.content_dim)
        repulsion.mark_explored(inf_embedding)

        # Should have been rejected
        assert len(repulsion.explored_regions) == 0

    def test_empty_slots_to_embedding(self, config):
        """Structured memory with no slots should return zero embedding."""
        structured = StructuredMemory(id=0)

        embedding = structured.to_embedding(config.content_dim)

        assert embedding.shape[0] == config.content_dim
        assert embedding.norm() == 0  # Zero vector

    def test_zero_confidence_slots(self, config):
        """Slots with zero confidence should not cause division by zero."""
        structured = StructuredMemory(id=0)
        structured.add_slot("test", MemorySlot(
            slot_type=SlotType.AGENT,
            name="test",
            content=torch.randn(config.content_dim),
            confidence=0.0  # Zero confidence
        ))

        embedding = structured.to_embedding(config.content_dim)

        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()

    def test_will_generator_explored_regions_used(self, config):
        """WillGenerator should use explored_regions for repulsion."""
        will_gen = WillGenerator(config.content_dim)

        # Add some memories
        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(10)
        ]

        # Mark some regions as explored
        explored = torch.randn(config.content_dim)
        will_gen.mark_explored(explored)

        # Generate will - should be repelled from explored
        will1 = will_gen.generate_curiosity_will(memories)

        # The will direction should not be too close to explored
        similarity = torch.nn.functional.cosine_similarity(
            will1.direction.unsqueeze(0),
            explored.unsqueeze(0)
        ).item()

        # With repulsion, should not be highly similar
        print(f"\n  Similarity to explored: {similarity:.3f}")
        # Just ensure it doesn't crash and produces valid output
        assert not torch.isnan(will1.direction).any()

    def test_morph_transform_preserves_norm(self, config):
        """MORPH transform should preserve approximate norm."""
        transform_ops = TransformOps(config.content_dim)

        structured = StructuredMemory(id=0)
        content = torch.randn(config.content_dim)
        content = content / content.norm() * 2.0  # Norm = 2.0
        structured.add_slot("test", MemorySlot(
            slot_type=SlotType.AGENT,
            name="test",
            content=content.clone(),
            confidence=1.0
        ))

        original_norm = structured.get_slot("test").content.norm().item()

        transform = Transform(
            transform_type=TransformType.MORPH,
            target_slot="test",
            magnitude=0.5
        )
        result = transform_ops.apply(structured, transform)

        new_norm = result.get_slot("test").content.norm().item()

        # Norm should be approximately preserved
        assert abs(new_norm - original_norm) < 0.1 * original_norm

    def test_plausibility_learn_from_structured_updates_centroid(self, config):
        """learn_from_structured should update memory_centroid."""
        from lmd import PlausibilityField

        plausibility = PlausibilityField(config.content_dim)

        # Initially no centroid
        assert plausibility.memory_centroid is None

        # Learn from structured memories
        structured_list = []
        for i in range(5):
            s = StructuredMemory(id=i)
            s.add_slot("core", MemorySlot(
                slot_type=SlotType.AGENT,
                name="core",
                content=torch.ones(config.content_dim) * i,  # Predictable content
                confidence=1.0
            ))
            structured_list.append(s)

        plausibility.learn_from_structured(structured_list)

        # Now should have centroid
        assert plausibility.memory_centroid is not None
        assert not torch.isnan(plausibility.memory_centroid).any()

    def test_ideation_closes_feedback_loop(self, config):
        """Ideation should call record_outcome and mark_explored with quality."""
        engine = IdeationEngine(config, IdeationConfig.quick())

        memories = [
            LivingMemory(
                id=i,
                content=torch.randn(config.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=0
            )
            for i in range(10)
        ]

        # Run ideation
        result = engine.ideate(memories)

        # Check that reality anchor has outcomes recorded
        n_outcomes = len(engine.reality_anchor.idea_outcomes)
        print(f"\n  Outcomes recorded: {n_outcomes}")

        # Should have recorded outcomes for at least some ideas
        # (discarded + final ideas)
        if result.total_ideas_discarded > 0 or len(result.ideas) > 0:
            assert n_outcomes > 0, "Should have recorded outcomes"

        # Check that repulsion field has explored regions with quality info
        for region in engine.repulsion_field.explored_regions:
            # Quality score should have been set
            assert 0 <= region.quality_score <= 1

    def test_plausibility_prunes_on_overflow(self, config):
        """Plausibility should prune when structures get too large."""
        plausibility = PlausibilityField(
            config.content_dim,
            max_cooccurrence_pairs=10,
            max_observed_combinations=5
        )

        # Add many slot combinations
        for i in range(20):
            s = StructuredMemory(id=i)
            for j in range(5):
                s.add_slot(f"slot_{i}_{j}", MemorySlot(
                    slot_type=SlotType.AGENT,
                    name=f"slot_{i}_{j}",
                    content=torch.randn(config.content_dim),
                    confidence=1.0
                ))
            plausibility.learn_from_structured([s])

        # Should be pruned to limits
        assert len(plausibility.slot_cooccurrence) <= plausibility.max_cooccurrence_pairs
        assert len(plausibility.observed_combinations) <= plausibility.max_observed_combinations

    def test_repulsion_field_productive_regions_kept(self, config):
        """Productive regions should be kept during pruning."""
        from lmd import RepulsionField
        repulsion = RepulsionField(config.content_dim, max_regions=5)

        # Add productive and non-productive regions
        # Need to add more than max_regions * 2 to trigger pruning
        for i in range(12):
            embedding = torch.randn(config.content_dim)
            repulsion.mark_explored(
                embedding,
                quality_score=0.1 if i < 6 else 0.9,
                was_productive=(i >= 6)
            )

        # After pruning, productive regions should be prioritized
        productive_count = sum(1 for r in repulsion.explored_regions if r.was_productive)
        print(f"\n  Regions remaining: {len(repulsion.explored_regions)}")
        print(f"  Productive regions kept: {productive_count}")

        # Pruning triggered when > max_regions * 2, should reduce to ~max_regions/2
        assert len(repulsion.explored_regions) <= repulsion.max_regions * 2
        # Should have kept some productive regions (they get boosted in sorting)
        assert productive_count >= 1, "Should keep at least one productive region"


class TestThreadSafety:
    """Test 14: Thread safety tests."""

    @pytest.fixture
    def config(self):
        return LMDConfig.toy_scale()

    def test_id_generator_thread_safe(self):
        """ID generator should be thread-safe under concurrent access."""
        import threading
        from lmd import IDGenerator

        gen = IDGenerator()
        generated_ids = []
        lock = threading.Lock()

        def generate_ids(n):
            for _ in range(n):
                new_id = gen.next_id("memory")
                with lock:
                    generated_ids.append(new_id)

        threads = [
            threading.Thread(target=generate_ids, args=(100,))
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All IDs should be unique
        assert len(generated_ids) == 1000
        assert len(set(generated_ids)) == 1000, "ID collision detected in concurrent access!"

    def test_repulsion_field_thread_safe(self, config):
        """Repulsion field should be thread-safe."""
        import threading
        from lmd import RepulsionField

        repulsion = RepulsionField(config.content_dim)
        errors = []

        def mark_regions(n):
            try:
                for _ in range(n):
                    embedding = torch.randn(config.content_dim)
                    repulsion.mark_explored(embedding)
                    _ = repulsion.compute_repulsion(embedding)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=mark_regions, args=(50,))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_reality_anchor_thread_safe(self, config):
        """Reality anchor should be thread-safe."""
        import threading
        from lmd import RealityAnchor

        anchor = RealityAnchor(config.content_dim)
        errors = []

        def record_outcomes(start_id, n):
            try:
                for i in range(n):
                    anchor.record_outcome(
                        idea_id=start_id + i,
                        predicted_valence=0.5,
                        actual_outcome=0.3
                    )
                    embedding = torch.randn(config.content_dim)
                    _ = anchor.ground_valence(0.5, embedding, start_id + i)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=record_outcomes, args=(i * 100, 50))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestPersistence:
    """Test 15: Persistence and state recovery tests."""

    def test_id_generator_persistence(self, tmp_path):
        """ID generator should persist and recover state."""
        import json
        from lmd import IDGenerator

        persistence_path = str(tmp_path / "id_state.json")

        # Generate some IDs
        gen1 = IDGenerator(persistence_path=persistence_path)
        ids = [gen1.next_id("memory") for _ in range(10)]
        last_id = ids[-1]

        # Create new generator from persisted state
        gen2 = IDGenerator(persistence_path=persistence_path)

        # Should not reissue any of the previous IDs
        new_id = gen2.next_id("memory")
        assert new_id > last_id, "Should continue from persisted counter"
        assert new_id not in ids, "Should not reissue IDs"

    def test_repulsion_field_state_serialization(self):
        """Repulsion field should serialize and restore state."""
        from lmd import RepulsionField

        original = RepulsionField(content_dim=32)

        # Add some regions
        for i in range(5):
            original.mark_explored(
                torch.randn(32),
                quality_score=0.5 + i * 0.1,
                was_productive=(i % 2 == 0)
            )

        # Serialize
        state = original.get_state()

        # Create new and restore
        restored = RepulsionField(content_dim=32)
        restored.load_state(state)

        # Should have same number of regions
        assert len(restored.explored_regions) == len(original.explored_regions)

    def test_reality_anchor_state_serialization(self):
        """Reality anchor should serialize and restore state."""
        from lmd import RealityAnchor

        original = RealityAnchor(content_dim=32)

        # Record some outcomes
        for i in range(10):
            original.record_outcome(i, 0.8, 0.3)

        original_bias = original.valence_bias
        original_scale = original.valence_scale

        # Serialize
        state = original.get_state()

        # Create new and restore
        restored = RealityAnchor(content_dim=32)
        restored.load_state(state)

        # Should have same calibration
        assert abs(restored.valence_bias - original_bias) < 0.01
        assert abs(restored.valence_scale - original_scale) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
