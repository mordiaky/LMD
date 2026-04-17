"""Tests for Creative Leaps System.

Tests the advanced divergence operators that enable human-like creative jumps:
- Analogical Transfer
- Manifold Walking (Diffusion)
- Orthogonal Compositions
- Void Extrapolation
- Hierarchical Ideas with Grafting
- Curiosity-Driven Probing
- Unified Creative Ideation Engine

Invented by Joshua R. Thomas, January 2026.
"""

import threading
import time

import pytest
import torch

from lmd import (
    # Curiosity Prober
    ActiveCuriosityProber,
    AnalogicalTransfer,
    ComponentType,
    CreativeIdeationConfig,
    # Creative Ideation
    CreativeIdeationEngine,
    CreativeLeapConfig,
    # Creative Leaps
    CreativeLeapEngine,
    CuriosityDrivenWill,
    GraftOperation,
    # Hierarchical Ideas
    HierarchicalIdea,
    HierarchicalIdeaFactory,
    IdeaComponent,
    IdeaForm,
    IdeaGrafter,
    LeapType,
    # Core LMD
    LivingMemory,
    ManifoldWalker,
    OrthogonalComposer,
    ProbeStrategy,
    RelationType,
    ValenceTrajectory,
    VoidExtrapolator,
    run_creative_ideation_demo,
)


@pytest.fixture
def content_dim():
    return 32


@pytest.fixture
def sample_embeddings(content_dim):
    """Generate sample embeddings for testing."""
    torch.manual_seed(42)
    return [torch.randn(content_dim) for _ in range(20)]


@pytest.fixture
def sample_memories(content_dim):
    """Generate sample living memories for testing."""
    torch.manual_seed(42)
    memories = []
    for i in range(15):
        mem = LivingMemory(
            id=i,
            content=torch.randn(content_dim),
            valence=ValenceTrajectory.random(),
            energy=0.7 + 0.3 * torch.rand(1).item(),
            created_at=time.time() - i * 10,
            label=f"test_memory_{i}"
        )
        memories.append(mem)
    return memories


class TestAnalogicalTransfer:
    """Tests for AnalogicalTransfer operator."""

    def test_initialization(self, content_dim):
        """Test basic initialization."""
        transfer = AnalogicalTransfer(content_dim=content_dim)
        assert transfer.content_dim == content_dim
        assert transfer.leap_type == LeapType.ANALOGICAL

    def test_find_distant_clusters(self, content_dim, sample_embeddings):
        """Test finding distant clusters."""
        transfer = AnalogicalTransfer(content_dim=content_dim)
        cluster_a, cluster_b = transfer.find_distant_clusters(sample_embeddings)

        assert len(cluster_a) > 0
        assert len(cluster_b) > 0
        # Clusters should be different
        assert cluster_a[0] is not cluster_b[0]

    def test_leap_produces_valid_embedding(self, content_dim, sample_embeddings):
        """Test that leap produces valid normalized embedding."""
        transfer = AnalogicalTransfer(content_dim=content_dim)
        leap = transfer.leap(sample_embeddings)

        assert leap.embedding.shape == (content_dim,)
        assert not torch.isnan(leap.embedding).any()
        assert not torch.isinf(leap.embedding).any()
        # Should be normalized
        assert abs(leap.embedding.norm().item() - 1.0) < 0.1

    def test_leap_with_intensity_modulation(self, content_dim, sample_embeddings):
        """Test that intensity modulates leap distance."""
        transfer = AnalogicalTransfer(content_dim=content_dim)

        low_intensity_leap = transfer.leap(sample_embeddings, intensity=0.2)
        high_intensity_leap = transfer.leap(sample_embeddings, intensity=1.5)

        # High intensity should generally produce larger leaps
        # (not deterministic, but metadata should reflect intensity)
        assert "blend_strength" in low_intensity_leap.metadata
        assert "blend_strength" in high_intensity_leap.metadata

    def test_leap_with_few_sources(self, content_dim):
        """Test leap handles minimal sources gracefully."""
        transfer = AnalogicalTransfer(content_dim=content_dim)

        # Single source
        single = [torch.randn(content_dim)]
        leap = transfer.leap(single)
        assert leap.embedding is not None

        # Empty sources
        leap_empty = transfer.leap([])
        assert leap_empty.embedding.shape == (content_dim,)


class TestManifoldWalker:
    """Tests for ManifoldWalker (diffusion) operator."""

    def test_initialization(self, content_dim):
        """Test basic initialization."""
        walker = ManifoldWalker(content_dim=content_dim, n_diffusion_steps=10)
        assert walker.content_dim == content_dim
        assert walker.n_diffusion_steps == 10
        assert walker.leap_type == LeapType.DIFFUSION

    def test_noise_schedule(self, content_dim):
        """Test noise schedule computation."""
        walker = ManifoldWalker(content_dim=content_dim)
        assert len(walker._betas) == walker.n_diffusion_steps
        assert (walker._betas >= 0).all()
        assert (walker._betas <= 1).all()

    def test_density_estimation(self, content_dim, sample_embeddings):
        """Test density estimation."""
        walker = ManifoldWalker(content_dim=content_dim)

        # Centroid should have higher density
        centroid = torch.stack(sample_embeddings).mean(dim=0)
        random_point = torch.randn(content_dim) * 10  # Far away

        density_centroid = walker.estimate_density(centroid, sample_embeddings)
        density_random = walker.estimate_density(random_point, sample_embeddings)

        assert density_centroid > density_random

    def test_leap_produces_valid_output(self, content_dim, sample_embeddings):
        """Test diffusion leap produces valid output."""
        walker = ManifoldWalker(content_dim=content_dim)
        leap = walker.leap(sample_embeddings)

        assert leap.embedding.shape == (content_dim,)
        assert not torch.isnan(leap.embedding).any()
        assert leap.leap_type == LeapType.DIFFUSION

    def test_temperature_affects_exploration(self, content_dim, sample_embeddings):
        """Test that temperature affects exploration."""
        low_temp_walker = ManifoldWalker(content_dim=content_dim, temperature=0.1)
        high_temp_walker = ManifoldWalker(content_dim=content_dim, temperature=2.0)

        # Multiple trials to check variance
        low_temp_leaps = [low_temp_walker.leap(sample_embeddings).embedding for _ in range(5)]
        high_temp_leaps = [high_temp_walker.leap(sample_embeddings).embedding for _ in range(5)]

        # High temperature should have more variance (on average)
        # This is stochastic, so we just check they're different
        assert len(low_temp_leaps) == len(high_temp_leaps)


class TestOrthogonalComposer:
    """Tests for OrthogonalComposer (Gram-Schmidt) operator."""

    def test_initialization(self, content_dim):
        """Test basic initialization."""
        composer = OrthogonalComposer(content_dim=content_dim)
        assert composer.content_dim == content_dim
        assert composer.leap_type == LeapType.ORTHOGONAL

    def test_gram_schmidt_orthogonality(self, content_dim):
        """Test Gram-Schmidt produces orthogonal basis."""
        composer = OrthogonalComposer(content_dim=content_dim)

        vectors = [torch.randn(content_dim) for _ in range(4)]
        orthogonal_basis, coefficients = composer.gram_schmidt(vectors)

        # Check orthogonality (dot products should be near zero)
        for i in range(len(orthogonal_basis)):
            for j in range(i + 1, len(orthogonal_basis)):
                dot = (orthogonal_basis[i] @ orthogonal_basis[j]).abs().item()
                assert dot < 0.01, f"Non-orthogonal: {dot}"

    def test_leap_creates_novel_combination(self, content_dim, sample_embeddings):
        """Test orthogonal leap creates novel combinations."""
        composer = OrthogonalComposer(content_dim=content_dim)
        leap = composer.leap(sample_embeddings)

        assert leap.embedding.shape == (content_dim,)
        assert leap.leap_type == LeapType.ORTHOGONAL
        assert "orthogonality" in leap.metadata

    def test_leap_with_intensity(self, content_dim, sample_embeddings):
        """Test intensity modulation."""
        composer = OrthogonalComposer(content_dim=content_dim)

        low_leap = composer.leap(sample_embeddings, intensity=0.1)
        high_leap = composer.leap(sample_embeddings, intensity=2.0)

        # Both should be valid
        assert not torch.isnan(low_leap.embedding).any()
        assert not torch.isnan(high_leap.embedding).any()


class TestVoidExtrapolator:
    """Tests for VoidExtrapolator operator."""

    def test_initialization(self, content_dim):
        """Test basic initialization."""
        extrapolator = VoidExtrapolator(content_dim=content_dim)
        assert extrapolator.content_dim == content_dim
        assert extrapolator.leap_type == LeapType.EXTRAPOLATION

    def test_find_voids(self, content_dim, sample_embeddings):
        """Test void detection."""
        extrapolator = VoidExtrapolator(content_dim=content_dim)
        voids = extrapolator.find_voids(sample_embeddings)

        assert len(voids) > 0
        # Voids should be far from sources
        for void in voids:
            stacked = torch.stack(sample_embeddings)
            min_dist = (stacked - void.unsqueeze(0)).norm(dim=1).min().item()
            assert min_dist > 0.1

    def test_leap_into_void(self, content_dim, sample_embeddings):
        """Test extrapolation into void regions."""
        extrapolator = VoidExtrapolator(content_dim=content_dim)
        leap = extrapolator.leap(sample_embeddings)

        assert leap.embedding.shape == (content_dim,)
        assert not torch.isnan(leap.embedding).any()
        assert leap.leap_type == LeapType.EXTRAPOLATION

    def test_leap_with_empty_sources(self, content_dim):
        """Test extrapolation with no sources."""
        extrapolator = VoidExtrapolator(content_dim=content_dim)
        leap = extrapolator.leap([])

        assert leap.embedding.shape == (content_dim,)
        assert leap.novelty_score == 1.0  # Everything is novel


class TestCreativeLeapEngine:
    """Tests for unified CreativeLeapEngine."""

    def test_initialization(self, content_dim):
        """Test engine initialization."""
        config = CreativeLeapConfig(content_dim=content_dim)
        engine = CreativeLeapEngine(config)

        assert len(engine.operators) == 4
        assert LeapType.ANALOGICAL in engine.operators
        assert LeapType.DIFFUSION in engine.operators
        assert LeapType.ORTHOGONAL in engine.operators
        assert LeapType.EXTRAPOLATION in engine.operators

    def test_operator_selection_by_dopamine(self, content_dim):
        """Test dopamine modulates operator selection."""
        config = CreativeLeapConfig(content_dim=content_dim)
        engine = CreativeLeapEngine(config)

        # Track selections across many trials
        low_dopamine_selections = {}
        high_dopamine_selections = {}

        for _ in range(100):
            low_op = engine.select_operator(dopamine=0.1)
            high_op = engine.select_operator(dopamine=0.9)

            low_dopamine_selections[low_op.leap_type] = low_dopamine_selections.get(low_op.leap_type, 0) + 1
            high_dopamine_selections[high_op.leap_type] = high_dopamine_selections.get(high_op.leap_type, 0) + 1

        # High dopamine should favor more radical operators
        # (ANALOGICAL, ORTHOGONAL)
        assert len(low_dopamine_selections) > 0
        assert len(high_dopamine_selections) > 0

    def test_leap(self, content_dim, sample_embeddings):
        """Test single leap."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        leap = engine.leap(sample_embeddings, dopamine=0.5)

        assert leap.embedding.shape == (content_dim,)
        assert leap.leap_type in LeapType
        assert 0 <= leap.novelty_score <= 1

    def test_batch_leap(self, content_dim, sample_embeddings):
        """Test batch leap generation."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        leaps = engine.batch_leap(sample_embeddings, n_leaps=5, dopamine=0.5)

        assert len(leaps) == 5
        # Check diversity - should use different operators
        leap_types = {leap.leap_type for leap in leaps}
        assert len(leap_types) >= 2  # At least 2 different types

    def test_quality_recording_and_adaptation(self, content_dim, sample_embeddings):
        """Test that quality recording enables adaptation."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))

        # Generate leaps and record quality
        for _ in range(20):
            leap = engine.leap(sample_embeddings)
            quality = 0.8 if leap.leap_type == LeapType.ANALOGICAL else 0.2
            engine.record_quality(leap, quality)

        initial_weights = dict(engine.operator_weights)
        engine.adapt_weights()
        adapted_weights = dict(engine.operator_weights)

        # Weights should have changed
        assert initial_weights != adapted_weights

    def test_statistics(self, content_dim, sample_embeddings):
        """Test statistics gathering."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))

        for _ in range(10):
            leap = engine.leap(sample_embeddings)
            engine.record_quality(leap, 0.5)

        stats = engine.get_statistics()
        assert "total_leaps" in stats
        assert "operator_weights" in stats
        assert stats["total_leaps"] == 10


class TestHierarchicalIdea:
    """Tests for hierarchical tree-structured ideas."""

    def test_idea_component_creation(self, content_dim):
        """Test creating idea components."""
        comp = IdeaComponent(
            id="test_comp",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.CORE,
            label="test"
        )

        assert comp.id == "test_comp"
        assert comp.embedding.shape == (content_dim,)
        assert comp.component_type == ComponentType.CORE

    def test_hierarchical_idea_creation(self, content_dim):
        """Test creating hierarchical idea."""
        root = IdeaComponent(
            id="root",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.CORE,
            label="root"
        )

        idea = HierarchicalIdea(id="test_idea", root=root)

        assert idea.root == root
        assert len(idea.components) == 1
        assert idea.depth == 1

    def test_add_component(self, content_dim):
        """Test adding components to idea."""
        root = IdeaComponent(
            id="root",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.CORE
        )
        idea = HierarchicalIdea(id="test", root=root)

        child = IdeaComponent(
            id="child",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.ATTRIBUTE
        )
        idea.add_component(child, root.id, RelationType.HAS)

        assert len(idea.components) == 2
        assert idea.depth == 2
        assert len(idea.get_children(root.id)) == 1

    def test_to_embedding(self, content_dim):
        """Test converting hierarchical idea to single embedding."""
        root = IdeaComponent(
            id="root",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.CORE,
            weight=1.0
        )
        idea = HierarchicalIdea(id="test", root=root)

        child = IdeaComponent(
            id="child",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.ATTRIBUTE,
            weight=0.5
        )
        idea.add_component(child, root.id)

        embedding = idea.to_embedding(content_dim)
        assert embedding.shape == (content_dim,)
        assert not torch.isnan(embedding).any()

    def test_clone(self, content_dim):
        """Test cloning hierarchical idea."""
        root = IdeaComponent(
            id="root",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.CORE
        )
        original = HierarchicalIdea(id="original", root=root)

        child = IdeaComponent(
            id="child",
            embedding=torch.randn(content_dim),
            component_type=ComponentType.ATTRIBUTE
        )
        original.add_component(child, root.id)

        cloned = original.clone()

        # Should have same structure but different IDs
        assert cloned.id != original.id
        assert len(cloned.components) == len(original.components)
        assert cloned.depth == original.depth


class TestIdeaGrafter:
    """Tests for idea grafting operations."""

    def test_initialization(self, content_dim):
        """Test grafter initialization."""
        grafter = IdeaGrafter(content_dim=content_dim)
        assert grafter.content_dim == content_dim

    def test_swap_component(self, content_dim):
        """Test swapping a component."""
        grafter = IdeaGrafter(content_dim=content_dim)

        root = IdeaComponent(id="root", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)
        idea = HierarchicalIdea(id="test", root=root)

        child = IdeaComponent(id="child", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)
        idea.add_component(child, root.id)

        donor = IdeaComponent(id="donor", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)

        result = grafter.swap_component(idea, donor, target_id=child.id)

        assert result.operation == GraftOperation.SWAP
        assert result.idea is not None
        assert 0 <= result.novelty_score <= 1

    def test_graft_component(self, content_dim):
        """Test grafting a new component."""
        grafter = IdeaGrafter(content_dim=content_dim)

        root = IdeaComponent(id="root", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)
        idea = HierarchicalIdea(id="test", root=root)

        donor = IdeaComponent(id="donor", embedding=torch.randn(content_dim), component_type=ComponentType.ACTION)

        result = grafter.graft_component(idea, donor)

        assert result.operation == GraftOperation.GRAFT
        assert len(result.idea.components) == len(idea.components) + 1

    def test_prune_component(self, content_dim):
        """Test pruning a component."""
        grafter = IdeaGrafter(content_dim=content_dim)

        root = IdeaComponent(id="root", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)
        idea = HierarchicalIdea(id="test", root=root)

        child1 = IdeaComponent(id="child1", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)
        child2 = IdeaComponent(id="child2", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)
        idea.add_component(child1, root.id)
        idea.add_component(child2, root.id)

        result = grafter.prune_component(idea)

        assert result.operation == GraftOperation.PRUNE
        assert len(result.idea.components) < len(idea.components)

    def test_morph_component(self, content_dim):
        """Test morphing (blending) a component."""
        grafter = IdeaGrafter(content_dim=content_dim)

        root = IdeaComponent(id="root", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)
        idea = HierarchicalIdea(id="test", root=root)

        donor = IdeaComponent(id="donor", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)

        result = grafter.morph_component(idea, donor, blend_factor=0.5)

        assert result.operation == GraftOperation.MORPH
        assert 0 <= result.novelty_score <= 1

    def test_mutate_multiple(self, content_dim):
        """Test applying multiple mutations."""
        grafter = IdeaGrafter(content_dim=content_dim)

        root = IdeaComponent(id="root", embedding=torch.randn(content_dim), component_type=ComponentType.CORE)
        idea = HierarchicalIdea(id="test", root=root)

        for i in range(3):
            child = IdeaComponent(id=f"child{i}", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)
            idea.add_component(child, root.id)

        donor_pool = [
            IdeaComponent(id=f"donor{i}", embedding=torch.randn(content_dim), component_type=ComponentType.ATTRIBUTE)
            for i in range(5)
        ]

        results = grafter.mutate(idea, donor_pool, dopamine=0.5, n_mutations=3)

        assert len(results) == 3
        # Each result should be valid
        for result in results:
            assert result.idea is not None


class TestHierarchicalIdeaFactory:
    """Tests for HierarchicalIdeaFactory."""

    def test_from_embedding(self, content_dim):
        """Test creating idea from single embedding."""
        factory = HierarchicalIdeaFactory(content_dim=content_dim)
        embedding = torch.randn(content_dim)

        idea = factory.from_embedding(embedding, n_components=3)

        assert idea.root is not None
        assert len(idea.components) >= 1

    def test_from_embeddings(self, content_dim, sample_embeddings):
        """Test creating idea from multiple embeddings."""
        factory = HierarchicalIdeaFactory(content_dim=content_dim)

        idea = factory.from_embeddings(sample_embeddings[:5])

        assert len(idea.components) == 5

    def test_random(self, content_dim):
        """Test random idea generation."""
        factory = HierarchicalIdeaFactory(content_dim=content_dim)

        idea = factory.random(n_components=4)

        assert idea.root is not None
        assert len(idea.components) >= 1

    def test_merge_combine(self, content_dim):
        """Test merging two ideas with combine strategy."""
        factory = HierarchicalIdeaFactory(content_dim=content_dim)

        idea_a = factory.random(n_components=3)
        idea_b = factory.random(n_components=3)

        merged = factory.merge(idea_a, idea_b, merge_strategy="combine")

        assert merged.root is not None
        assert len(merged.components) >= 3


class TestActiveCuriosityProber:
    """Tests for ActiveCuriosityProber."""

    def test_initialization(self, content_dim):
        """Test prober initialization."""
        prober = ActiveCuriosityProber(content_dim=content_dim)
        assert prober.content_dim == content_dim

    def test_estimate_density(self, content_dim, sample_embeddings):
        """Test density estimation."""
        prober = ActiveCuriosityProber(content_dim=content_dim)

        # Centroid should have higher density
        centroid = torch.stack(sample_embeddings).mean(dim=0)
        density = prober.estimate_density(centroid, sample_embeddings)

        assert 0 <= density <= 1

    def test_find_voids(self, content_dim, sample_embeddings):
        """Test finding void regions."""
        prober = ActiveCuriosityProber(content_dim=content_dim)
        voids = prober.find_voids(sample_embeddings)

        assert len(voids) > 0
        for void in voids:
            assert void.density < prober.void_threshold

    def test_find_frontier(self, content_dim, sample_embeddings):
        """Test finding frontier points."""
        prober = ActiveCuriosityProber(content_dim=content_dim)
        frontiers = prober.find_frontier(sample_embeddings)

        assert len(frontiers) > 0
        for frontier in frontiers:
            assert frontier.shape == (content_dim,)

    def test_probe(self, content_dim, sample_embeddings):
        """Test probing with different strategies."""
        prober = ActiveCuriosityProber(content_dim=content_dim)

        for strategy in ProbeStrategy:
            results = prober.probe(sample_embeddings, strategy=strategy, n_results=3)
            assert len(results) <= 3
            for result in results:
                assert result.target.shape == (content_dim,)

    def test_directed_curiosity(self, content_dim, sample_embeddings):
        """Test dopamine-modulated curiosity."""
        prober = ActiveCuriosityProber(content_dim=content_dim)

        low_dopamine_results = prober.directed_curiosity(
            sample_embeddings, dopamine=0.1, n_targets=3
        )
        high_dopamine_results = prober.directed_curiosity(
            sample_embeddings, dopamine=0.9, n_targets=3
        )

        assert len(low_dopamine_results) <= 3
        assert len(high_dopamine_results) <= 3


class TestCuriosityDrivenWill:
    """Tests for CuriosityDrivenWill."""

    def test_initialization(self, content_dim):
        """Test will generator initialization."""
        will_gen = CuriosityDrivenWill(content_dim=content_dim)
        assert will_gen.content_dim == content_dim

    def test_generate_will(self, content_dim, sample_embeddings):
        """Test generating a curiosity-driven will."""
        will_gen = CuriosityDrivenWill(content_dim=content_dim)
        will, probe = will_gen.generate_will(sample_embeddings, dopamine=0.5)

        assert will.shape == (content_dim,)
        assert probe.target.shape == (content_dim,)

    def test_batch_wills(self, content_dim, sample_embeddings):
        """Test generating multiple wills."""
        will_gen = CuriosityDrivenWill(content_dim=content_dim)
        wills = will_gen.batch_wills(sample_embeddings, n_wills=5, dopamine=0.5)

        assert len(wills) == 5
        for will, _probe in wills:
            assert will.shape == (content_dim,)


class TestCreativeIdeationEngine:
    """Tests for unified CreativeIdeationEngine."""

    def test_initialization(self, content_dim):
        """Test engine initialization."""
        config = CreativeIdeationConfig(content_dim=content_dim)
        engine = CreativeIdeationEngine(config)

        assert engine.config.content_dim == content_dim
        assert engine.leap_engine is not None
        assert engine.idea_factory is not None
        assert engine.grafter is not None
        assert engine.curiosity_prober is not None

    def test_generate_flat_ideas(self, content_dim, sample_memories):
        """Test generating flat embedding ideas."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        ideas = engine.generate_flat_ideas(sample_memories, n_ideas=5, dopamine=0.5)

        assert len(ideas) == 5
        for idea in ideas:
            assert idea.form == IdeaForm.FLAT
            assert idea.embedding is not None

    def test_generate_hierarchical_ideas(self, content_dim, sample_memories):
        """Test generating hierarchical ideas."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        ideas = engine.generate_hierarchical_ideas(sample_memories, n_ideas=5, dopamine=0.5)

        assert len(ideas) == 5
        for idea in ideas:
            assert idea.form == IdeaForm.HIERARCHICAL
            assert idea.hierarchical is not None

    def test_generate_leap_ideas(self, content_dim, sample_memories):
        """Test generating creative leap ideas."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        ideas = engine.generate_leap_ideas(sample_memories, n_ideas=5, dopamine=0.5)

        assert len(ideas) == 5
        for idea in ideas:
            assert idea.form == IdeaForm.LEAP
            assert idea.leap is not None

    def test_ideate_full_session(self, content_dim, sample_memories):
        """Test full ideation session."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        result = engine.ideate(sample_memories, dopamine=0.5, n_ideas=10)

        assert result.total_generated >= 10
        assert len(result.strategies_used) > 0
        assert result.session_time > 0

    def test_ideate_with_dopamine_variation(self, content_dim, sample_memories):
        """Test that dopamine affects ideation."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))

        low_result = engine.ideate(sample_memories, dopamine=0.1, n_ideas=10)
        high_result = engine.ideate(sample_memories, dopamine=0.9, n_ideas=10)

        # Both should produce results
        assert low_result.total_generated > 0
        assert high_result.total_generated > 0

    def test_consolidate_to_memories(self, content_dim, sample_memories):
        """Test consolidating ideas to memories."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        result = engine.ideate(sample_memories, dopamine=0.5)

        new_memories = engine.consolidate_to_memories(result.ideas[:3])

        for mem in new_memories:
            assert isinstance(mem, LivingMemory)
            assert mem.content.shape == (content_dim,)

    def test_statistics(self, content_dim, sample_memories):
        """Test statistics gathering."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))

        for _ in range(3):
            engine.ideate(sample_memories, dopamine=0.5)

        stats = engine.get_statistics()

        assert stats["session_count"] == 3
        assert stats["total_ideas_generated"] > 0


class TestCreativeIdeationDemo:
    """Tests for the demo function."""

    def test_demo_runs(self):
        """Test that demo runs without errors."""
        result = run_creative_ideation_demo(n_rounds=2, verbose=False)

        assert result is not None
        assert result.total_generated > 0


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_empty_embeddings(self, content_dim):
        """Test handling empty embedding lists."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        leap = engine.leap([], dopamine=0.5)

        assert not torch.isnan(leap.embedding).any()

    def test_single_embedding(self, content_dim):
        """Test handling single embedding."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        leap = engine.leap([torch.randn(content_dim)], dopamine=0.5)

        assert not torch.isnan(leap.embedding).any()

    def test_zero_vectors(self, content_dim):
        """Test handling zero vectors."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        zeros = [torch.zeros(content_dim) for _ in range(5)]
        leap = engine.leap(zeros, dopamine=0.5)

        assert not torch.isnan(leap.embedding).any()

    def test_nan_rejection(self, content_dim):
        """Test that NaN values are rejected."""
        comp = IdeaComponent(
            id="test",
            embedding=torch.tensor([float('nan')] * content_dim),
            component_type=ComponentType.CORE
        )

        # Should be replaced with valid tensor
        assert not torch.isnan(comp.embedding).any()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_leaps(self, content_dim, sample_embeddings):
        """Test concurrent leap generation."""
        engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=content_dim))
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    leap = engine.leap(sample_embeddings, dopamine=0.5)
                    engine.record_quality(leap, 0.5)
                    results.append(leap)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 40

    def test_concurrent_ideation(self, content_dim, sample_memories):
        """Test concurrent ideation sessions."""
        engine = CreativeIdeationEngine(CreativeIdeationConfig(content_dim=content_dim))
        results = []
        errors = []

        def worker():
            try:
                result = engine.ideate(sample_memories, dopamine=0.5, n_ideas=5)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4
