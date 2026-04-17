#!/usr/bin/env python3
"""Hierarchical Ideas example for Living Memory Dynamics (LMD).

This example demonstrates:
1. Creating tree-structured ideas with components
2. Grafting operations (swap, graft, prune, morph, transplant)
3. Merging ideas from different domains
"""

import torch
from lmd import (
    HierarchicalIdea,
    HierarchicalIdeaFactory,
    IdeaGrafter,
    IdeaComponent,
    ComponentType,
    RelationType,
    GraftOperation,
)


def create_concept_embedding(concept_name: str, content_dim: int) -> torch.Tensor:
    """Create a pseudo-embedding for a concept (for demo purposes).

    In practice, you'd use a real encoder (CLIP, sentence transformers, etc.)
    """
    # Use hash of name to create deterministic embedding
    torch.manual_seed(hash(concept_name) % (2**32))
    embedding = torch.randn(content_dim)
    return embedding / embedding.norm()


def demo_create_hierarchical_idea(factory: HierarchicalIdeaFactory, content_dim: int):
    """Demonstrate creating a hierarchical idea from scratch."""
    print("\n" + "=" * 60)
    print("1. Creating Hierarchical Ideas")
    print("=" * 60)

    # Create "dragon" concept with components
    dragon_emb = create_concept_embedding("dragon", content_dim)

    print("\nCreating 'dragon' idea with components...")
    dragon = factory.from_embedding(dragon_emb, depth=3, label="dragon")

    print(f"   Root: {dragon.root.label} ({dragon.root.component_type.name})")
    print(f"   Components: {len(dragon.components)}")
    print(f"   Relations: {len(dragon.relations)}")

    # Show component tree
    print("\n   Component tree:")
    for comp_id, comp in dragon.components.items():
        depth = dragon._get_component_depth(comp_id)
        indent = "   " + "  " * depth
        print(f"{indent}- {comp.label} ({comp.component_type.name}, weight={comp.weight:.2f})")

    return dragon


def demo_graft_operations(grafter: IdeaGrafter, dragon: HierarchicalIdea,
                          factory: HierarchicalIdeaFactory, content_dim: int):
    """Demonstrate the 5 graft operations."""
    print("\n" + "=" * 60)
    print("2. Graft Operations")
    print("=" * 60)

    # Create a "crystal" concept to graft from
    crystal_emb = create_concept_embedding("crystal", content_dim)
    crystal = factory.from_embedding(crystal_emb, depth=2, label="crystal")

    # Get a component from crystal to use as donor
    crystal_comp = list(crystal.components.values())[1]  # Get first non-root

    # Find target in dragon
    dragon_target = list(dragon.components.keys())[1]  # First non-root

    # 1. SWAP - Replace component
    print("\n2.1 SWAP Operation")
    print("    Replaces a component with one from another domain")
    result = grafter.swap_component(dragon, crystal_comp, dragon_target)
    print(f"    Novelty: {result.novelty:.3f}")
    print(f"    Coherence: {result.coherence:.3f}")
    print(f"    Operation: {result.operation.name}")

    # 2. GRAFT - Add component
    print("\n2.2 GRAFT Operation")
    print("    Adds a new component as child of existing one")
    dragon_parent = dragon.root.id
    result = grafter.graft_component(dragon, crystal_comp, dragon_parent)
    print(f"    Novelty: {result.novelty:.3f}")
    print(f"    Coherence: {result.coherence:.3f}")
    print(f"    New component added as child of root")

    # 3. PRUNE - Remove component
    print("\n2.3 PRUNE Operation")
    print("    Removes a component (and its subtree)")
    prune_target = list(dragon.components.keys())[2]
    result = grafter.prune_component(dragon, prune_target)
    print(f"    Novelty: {result.novelty:.3f}")
    print(f"    Coherence: {result.coherence:.3f}")
    print(f"    Component removed")

    # 4. MORPH - Blend embeddings
    print("\n2.4 MORPH Operation")
    print("    Blends component embedding with donor (interpolation)")
    morph_target = list(dragon.components.keys())[1]
    result = grafter.morph_component(dragon, crystal_comp, morph_target, blend_factor=0.5)
    print(f"    Novelty: {result.novelty:.3f}")
    print(f"    Coherence: {result.coherence:.3f}")
    print(f"    50% blend between dragon and crystal components")

    # 5. TRANSPLANT - Move subtree
    print("\n2.5 TRANSPLANT Operation")
    print("    Moves entire subtree to new parent")
    if len(dragon.components) > 3:
        subtree_root = list(dragon.components.keys())[2]
        new_parent = list(dragon.components.keys())[1]
        result = grafter.transplant_subtree(dragon, subtree_root, new_parent)
        print(f"    Novelty: {result.novelty:.3f}")
        print(f"    Coherence: {result.coherence:.3f}")
        print(f"    Subtree moved to new parent")
    else:
        print("    (Skipped - not enough components)")


def demo_merge_ideas(factory: HierarchicalIdeaFactory, content_dim: int):
    """Demonstrate merging ideas from different domains."""
    print("\n" + "=" * 60)
    print("3. Merging Ideas")
    print("=" * 60)

    # Create two concepts
    fire_emb = create_concept_embedding("fire", content_dim)
    water_emb = create_concept_embedding("water", content_dim)

    fire = factory.from_embedding(fire_emb, depth=2, label="fire")
    water = factory.from_embedding(water_emb, depth=2, label="water")

    print("\nMerging 'fire' and 'water' ideas...")

    # Different merge strategies
    for strategy in ["combine", "graft", "blend"]:
        print(f"\n   Strategy: {strategy}")
        merged = factory.merge(fire, water, strategy=strategy)
        print(f"   Result components: {len(merged.components)}")
        print(f"   Root: {merged.root.label}")


def demo_creative_mutations(grafter: IdeaGrafter, factory: HierarchicalIdeaFactory,
                             content_dim: int):
    """Demonstrate dopamine-modulated mutations."""
    print("\n" + "=" * 60)
    print("4. Dopamine-Modulated Mutations")
    print("=" * 60)

    # Create base idea
    robot_emb = create_concept_embedding("robot", content_dim)
    robot = factory.from_embedding(robot_emb, depth=3, label="robot")

    # Create donor pool
    donors = []
    for concept in ["bird", "octopus", "plant", "lightning"]:
        emb = create_concept_embedding(concept, content_dim)
        idea = factory.from_embedding(emb, depth=2, label=concept)
        donors.extend(list(idea.components.values()))

    print("\nMutating 'robot' with different dopamine levels...")

    for dopamine in [0.2, 0.5, 0.8]:
        print(f"\n   Dopamine: {dopamine}")
        results = grafter.mutate(robot, donors, dopamine=dopamine, n_mutations=3)

        for i, result in enumerate(results):
            print(f"      Mutation {i+1}: {result.operation.name} "
                  f"(novelty={result.novelty:.2f}, coherence={result.coherence:.2f})")


def main():
    print("Living Memory Dynamics - Hierarchical Ideas Example")
    print("=" * 60)

    # Configuration
    content_dim = 256

    # Create tools
    factory = HierarchicalIdeaFactory(content_dim)
    grafter = IdeaGrafter(content_dim)

    # Demos
    dragon = demo_create_hierarchical_idea(factory, content_dim)
    demo_graft_operations(grafter, dragon, factory, content_dim)
    demo_merge_ideas(factory, content_dim)
    demo_creative_mutations(grafter, factory, content_dim)

    print("\n" + "=" * 60)
    print("Hierarchical Ideas Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
