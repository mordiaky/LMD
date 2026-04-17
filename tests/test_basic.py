"""Basic tests for LMD package."""

import pytest
import torch


class TestImports:
    """Test that all core imports work."""

    def test_core_imports(self):
        """Test core module imports."""

    def test_creative_leaps_imports(self):
        """Test creative leaps imports."""

    def test_hierarchical_imports(self):
        """Test hierarchical ideas imports."""

    def test_curiosity_imports(self):
        """Test curiosity prober imports."""


class TestLivingMemory:
    """Test LivingMemory creation and basic operations."""

    def test_create_memory(self):
        """Test memory creation."""
        from lmd import LivingMemory, NarrativePhase, ValenceTrajectory

        embedding = torch.randn(256)
        memory = LivingMemory(
            id="test_001",
            content=embedding,
            energy=1.0,
            valence=ValenceTrajectory(points=torch.tensor([0.3, 0.8, 0.5])),
            phase=NarrativePhase.SETUP,
        )

        assert memory.id == "test_001"
        assert memory.energy == 1.0
        assert memory.content.shape == (256,)

    def test_metabolic_state(self):
        """Test metabolic state from energy."""
        from lmd import MetabolicState

        assert MetabolicState.from_energy(1.5) == MetabolicState.VIVID
        assert MetabolicState.from_energy(0.7) == MetabolicState.ACTIVE
        assert MetabolicState.from_energy(0.4) == MetabolicState.DORMANT
        assert MetabolicState.from_energy(0.2) == MetabolicState.FADING
        assert MetabolicState.from_energy(0.05) == MetabolicState.GHOST


class TestCreativeLeaps:
    """Test creative leap operators."""

    def test_analogical_transfer(self):
        """Test analogical transfer operator."""
        from lmd import AnalogicalTransfer

        dim = 128
        operator = AnalogicalTransfer(dim)

        # Create source embeddings
        sources = [torch.randn(dim) for _ in range(10)]
        sources = [s / s.norm() for s in sources]

        leap = operator.leap(sources, intensity=0.8)

        assert leap.embedding.shape == (dim,)
        assert 0 <= leap.novelty <= 1
        assert 0 <= leap.coherence <= 1

    def test_orthogonal_composer(self):
        """Test orthogonal composer."""
        from lmd import OrthogonalComposer

        dim = 128
        composer = OrthogonalComposer(dim)

        sources = [torch.randn(dim) for _ in range(5)]
        sources = [s / s.norm() for s in sources]

        leap = composer.leap(sources, intensity=0.9)

        assert leap.embedding.shape == (dim,)

    def test_void_extrapolator(self):
        """Test void extrapolator."""
        from lmd import VoidExtrapolator

        dim = 128
        extrapolator = VoidExtrapolator(dim, n_probes=50)

        sources = [torch.randn(dim) for _ in range(10)]
        sources = [s / s.norm() for s in sources]

        leap = extrapolator.leap(sources, intensity=1.0)

        assert leap.embedding.shape == (dim,)
        # Void extrapolation should find novel regions
        assert leap.novelty > 0


class TestHierarchicalIdeas:
    """Test hierarchical idea operations."""

    def test_create_from_embedding(self):
        """Test creating hierarchical idea from embedding."""
        from lmd import HierarchicalIdeaFactory

        dim = 256
        factory = HierarchicalIdeaFactory(dim)

        embedding = torch.randn(dim)
        embedding = embedding / embedding.norm()

        idea = factory.from_embedding(embedding, depth=3, label="test")

        assert idea.root is not None
        assert len(idea.components) > 0

    def test_graft_swap(self):
        """Test swap graft operation."""
        from lmd import HierarchicalIdeaFactory, IdeaGrafter

        dim = 256
        factory = HierarchicalIdeaFactory(dim)
        grafter = IdeaGrafter(dim)

        # Create two ideas
        idea_a = factory.from_embedding(torch.randn(dim), depth=2, label="a")
        idea_b = factory.from_embedding(torch.randn(dim), depth=2, label="b")

        # Get donor and target
        donor = list(idea_b.components.values())[1]
        target_id = list(idea_a.components.keys())[1]

        result = grafter.swap_component(idea_a, donor, target_id)

        assert result.operation.name == "SWAP"
        assert 0 <= result.novelty <= 1
        assert 0 <= result.coherence <= 1


class TestCudaFallback:
    """Test CUDA fallback operations work on CPU."""

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity."""
        from lmd.cuda.fallback import batch_cosine_similarity

        A = torch.randn(10, 128)
        B = torch.randn(5, 128)

        sim = batch_cosine_similarity(A, B)

        assert sim.shape == (10, 5)
        assert sim.min() >= -1.0
        assert sim.max() <= 1.0

    def test_density_estimation(self):
        """Test density estimation."""
        from lmd.cuda.fallback import density_estimation

        queries = torch.randn(10, 128)
        points = torch.randn(50, 128)

        density = density_estimation(queries, points, bandwidth=0.5)

        assert density.shape == (10,)
        assert density.min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
