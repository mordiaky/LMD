"""Hierarchical Ideas: Tree-Structured Composites for Creative Mutation.

Ideas evolve as composites, not flat vectors:
- Each Idea = (core_embedding, list of component_embeddings + relations)
- Diverge by grafting/substituting components (tree-like mutation)

Example:
    Start with "dragon" (scales + wings + fire)
    Mutate by swapping "fire" component with distant "glass refraction"
    Result: "prismatic breath weapon"

Store as small trees (depth 3-5), embed leaves, score whole.

Invented by Joshua R. Thomas, January 2026.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import copy
import uuid

from .safeguards import safe_normalize, safe_divide, EPS


class ComponentType(Enum):
    """Types of components in a hierarchical idea."""
    CORE = auto()         # Central concept
    ATTRIBUTE = auto()    # Property of a concept
    ACTION = auto()       # Behavior or capability
    RELATION = auto()     # Connection between concepts
    MODIFIER = auto()     # Modifies another component
    PART = auto()         # Physical or abstract part


class RelationType(Enum):
    """Types of relations between components."""
    HAS = auto()          # Ownership/composition (dragon HAS wings)
    IS = auto()           # Identity/classification (dragon IS creature)
    CAN = auto()          # Capability (dragon CAN breathe fire)
    LIKE = auto()         # Similarity (scales LIKE armor)
    CAUSES = auto()       # Causation (fire CAUSES burning)
    MODIFIES = auto()     # Modification (prismatic MODIFIES light)


@dataclass
class IdeaComponent:
    """A single component in a hierarchical idea tree."""
    id: str
    embedding: torch.Tensor
    component_type: ComponentType
    label: str = ""  # Optional human-readable label
    weight: float = 1.0  # Importance weight
    mutable: bool = True  # Can this be grafted/swapped?

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        # Validate embedding
        if torch.isnan(self.embedding).any() or torch.isinf(self.embedding).any():
            self.embedding = safe_normalize(torch.randn_like(self.embedding))

    def clone(self) -> "IdeaComponent":
        """Create a deep copy."""
        return IdeaComponent(
            id=str(uuid.uuid4())[:8],  # New ID
            embedding=self.embedding.clone(),
            component_type=self.component_type,
            label=self.label,
            weight=self.weight,
            mutable=self.mutable
        )

    def similarity(self, other: "IdeaComponent") -> float:
        """Compute cosine similarity with another component."""
        return F.cosine_similarity(
            self.embedding.unsqueeze(0),
            other.embedding.unsqueeze(0)
        ).item()


@dataclass
class ComponentRelation:
    """A relation between two components."""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0  # Relation strength


@dataclass
class HierarchicalIdea:
    """A tree-structured idea with components and relations.

    Structure:
        root (CORE)
        ├── component1 (ATTRIBUTE) -- HAS relation
        │   └── subcomponent (MODIFIER)
        ├── component2 (ACTION) -- CAN relation
        └── component3 (PART) -- HAS relation
    """
    id: str
    root: IdeaComponent
    components: Dict[str, IdeaComponent] = field(default_factory=dict)
    relations: List[ComponentRelation] = field(default_factory=list)
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        # Ensure root is in components
        self.components[self.root.id] = self.root
        # Compute depth
        self._update_depth()

    def _update_depth(self):
        """Compute tree depth."""
        if not self.relations:
            self.depth = 1
            return

        # Build adjacency list
        children: Dict[str, List[str]] = {c: [] for c in self.components}
        for rel in self.relations:
            if rel.source_id in children:
                children[rel.source_id].append(rel.target_id)

        # BFS to find max depth
        visited = set()
        queue = [(self.root.id, 1)]
        max_depth = 1

        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            max_depth = max(max_depth, depth)

            for child_id in children.get(node_id, []):
                queue.append((child_id, depth + 1))

        self.depth = max_depth

    def add_component(
        self,
        component: IdeaComponent,
        parent_id: str,
        relation_type: RelationType = RelationType.HAS
    ) -> None:
        """Add a component as child of parent."""
        self.components[component.id] = component
        self.relations.append(ComponentRelation(
            source_id=parent_id,
            target_id=component.id,
            relation_type=relation_type
        ))
        self._update_depth()

    def get_children(self, parent_id: str) -> List[IdeaComponent]:
        """Get all children of a component."""
        child_ids = [r.target_id for r in self.relations if r.source_id == parent_id]
        return [self.components[cid] for cid in child_ids if cid in self.components]

    def get_parent(self, child_id: str) -> Optional[IdeaComponent]:
        """Get parent of a component."""
        for rel in self.relations:
            if rel.target_id == child_id:
                return self.components.get(rel.source_id)
        return None

    def get_mutable_components(self) -> List[IdeaComponent]:
        """Get all components that can be mutated/grafted."""
        return [c for c in self.components.values() if c.mutable]

    def get_leaf_components(self) -> List[IdeaComponent]:
        """Get components with no children."""
        parents = {r.source_id for r in self.relations}
        return [c for c in self.components.values() if c.id not in parents]

    def to_embedding(self, content_dim: int = 32) -> torch.Tensor:
        """Convert hierarchical idea to single embedding.

        Combines components weighted by their importance and depth.
        """
        if not self.components:
            return safe_normalize(torch.randn(content_dim))

        # Weight by: component weight * depth_decay
        embeddings = []
        weights = []

        for comp_id, comp in self.components.items():
            # Find depth of this component
            depth = self._get_component_depth(comp_id)
            depth_decay = 1.0 / (1.0 + 0.2 * depth)  # Deeper = less weight

            embeddings.append(comp.embedding)
            weights.append(comp.weight * depth_decay)

        stacked = torch.stack(embeddings)
        weight_tensor = torch.tensor(weights).unsqueeze(1)
        weight_tensor = weight_tensor / (weight_tensor.sum() + EPS)

        combined = (stacked * weight_tensor).sum(dim=0)
        return safe_normalize(combined)

    def _get_component_depth(self, comp_id: str) -> int:
        """Get depth of a component in the tree."""
        if comp_id == self.root.id:
            return 0

        parent = self.get_parent(comp_id)
        if parent is None:
            return 1

        return 1 + self._get_component_depth(parent.id)

    def clone(self) -> "HierarchicalIdea":
        """Create a deep copy."""
        new_root = self.root.clone()
        new_idea = HierarchicalIdea(
            id=str(uuid.uuid4())[:8],
            root=new_root,
            metadata=copy.deepcopy(self.metadata)
        )

        # Map old IDs to new IDs
        id_map = {self.root.id: new_root.id}

        # Clone components
        for comp_id, comp in self.components.items():
            if comp_id != self.root.id:
                new_comp = comp.clone()
                id_map[comp_id] = new_comp.id
                new_idea.components[new_comp.id] = new_comp

        # Clone relations with new IDs
        for rel in self.relations:
            new_source = id_map.get(rel.source_id, rel.source_id)
            new_target = id_map.get(rel.target_id, rel.target_id)
            new_idea.relations.append(ComponentRelation(
                source_id=new_source,
                target_id=new_target,
                relation_type=rel.relation_type,
                strength=rel.strength
            ))

        new_idea._update_depth()
        return new_idea

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [f"HierarchicalIdea {self.id} (depth={self.depth})"]
        lines.append(f"  Root: {self.root.label or self.root.id} ({self.root.component_type.name})")
        lines.append(f"  Components: {len(self.components)}")
        lines.append(f"  Relations: {len(self.relations)}")
        return "\n".join(lines)


class GraftOperation(Enum):
    """Types of graft operations."""
    SWAP = auto()        # Replace one component with another
    GRAFT = auto()       # Add component from another idea
    PRUNE = auto()       # Remove a component
    MORPH = auto()       # Blend component embeddings
    TRANSPLANT = auto()  # Move subtree to different location


@dataclass
class GraftResult:
    """Result of a graft operation."""
    idea: HierarchicalIdea
    operation: GraftOperation
    source_component: Optional[str]
    target_component: Optional[str]
    novelty_score: float
    coherence_score: float

    @property
    def novelty(self) -> float:
        """Alias for novelty_score — matches the public API shown in README examples."""
        return self.novelty_score

    @property
    def coherence(self) -> float:
        """Alias for coherence_score — matches the public API shown in README examples."""
        return self.coherence_score


class IdeaGrafter:
    """Performs graft operations on hierarchical ideas.

    Mutations:
    - SWAP: Replace component with one from distant domain
    - GRAFT: Add component from another idea
    - PRUNE: Remove component (simplification)
    - MORPH: Blend two component embeddings
    - TRANSPLANT: Move subtree to different parent
    """

    def __init__(
        self,
        content_dim: int = 32,
        swap_probability: float = 0.3,
        graft_probability: float = 0.25,
        prune_probability: float = 0.15,
        morph_probability: float = 0.2,
        transplant_probability: float = 0.1
    ):
        self.content_dim = content_dim
        self.operation_probs = {
            GraftOperation.SWAP: swap_probability,
            GraftOperation.GRAFT: graft_probability,
            GraftOperation.PRUNE: prune_probability,
            GraftOperation.MORPH: morph_probability,
            GraftOperation.TRANSPLANT: transplant_probability
        }
        self._normalize_probs()
        self._lock = threading.RLock()

    def _normalize_probs(self):
        total = sum(self.operation_probs.values())
        if total > EPS:
            for k in self.operation_probs:
                self.operation_probs[k] /= total

    def select_operation(self, dopamine: float = 0.5) -> GraftOperation:
        """Select operation with dopamine modulation.

        High dopamine -> more radical (SWAP, TRANSPLANT)
        Low dopamine -> conservative (MORPH, PRUNE)
        """
        modulated = {}
        for op, prob in self.operation_probs.items():
            if op in [GraftOperation.SWAP, GraftOperation.TRANSPLANT]:
                modulator = 0.5 + dopamine
            elif op == GraftOperation.GRAFT:
                modulator = 0.7 + 0.6 * dopamine
            else:  # PRUNE, MORPH
                modulator = 1.5 - dopamine
            modulated[op] = prob * modulator

        total = sum(modulated.values())
        r = torch.rand(1).item() * total
        cumsum = 0.0
        for op, prob in modulated.items():
            cumsum += prob
            if r <= cumsum:
                return op

        return GraftOperation.MORPH  # Fallback

    def swap_component(
        self,
        idea: HierarchicalIdea,
        donor_component: IdeaComponent,
        target_id: Optional[str] = None
    ) -> GraftResult:
        """Swap a component with one from a different domain.

        Args:
            idea: Idea to mutate
            donor_component: Component to swap in
            target_id: ID of component to replace (None = random mutable)

        Returns:
            GraftResult with mutated idea
        """
        mutated = idea.clone()

        # Find target
        mutable = mutated.get_mutable_components()
        if not mutable:
            return GraftResult(
                idea=mutated,
                operation=GraftOperation.SWAP,
                source_component=None,
                target_component=None,
                novelty_score=0.0,
                coherence_score=1.0
            )

        if target_id and target_id in mutated.components:
            target = mutated.components[target_id]
        else:
            idx = torch.randint(len(mutable), (1,)).item()
            target = mutable[idx]

        # Compute novelty before swap
        original_sim = target.similarity(donor_component)
        novelty = max(0.0, min(1.0, 1.0 - original_sim))  # Clamp to [0, 1]

        # Perform swap
        new_comp = donor_component.clone()
        new_comp.component_type = target.component_type  # Keep type
        new_comp.weight = target.weight

        # Update ID mapping in relations
        for rel in mutated.relations:
            if rel.source_id == target.id:
                rel.source_id = new_comp.id
            if rel.target_id == target.id:
                rel.target_id = new_comp.id

        # Replace in components dict
        del mutated.components[target.id]
        mutated.components[new_comp.id] = new_comp

        # If swapped root, update root reference
        if target.id == idea.root.id:
            mutated.root = new_comp

        # Coherence: how well does new component fit with neighbors?
        coherence = self._compute_local_coherence(mutated, new_comp.id)

        return GraftResult(
            idea=mutated,
            operation=GraftOperation.SWAP,
            source_component=donor_component.id,
            target_component=target.id,
            novelty_score=novelty,
            coherence_score=coherence
        )

    def graft_component(
        self,
        idea: HierarchicalIdea,
        donor_component: IdeaComponent,
        parent_id: Optional[str] = None,
        relation_type: RelationType = RelationType.HAS
    ) -> GraftResult:
        """Add a new component from another domain.

        Args:
            idea: Idea to extend
            donor_component: Component to graft
            parent_id: Where to attach (None = random)
            relation_type: Type of relation to parent

        Returns:
            GraftResult with extended idea
        """
        mutated = idea.clone()

        # Select parent
        if parent_id and parent_id in mutated.components:
            parent = mutated.components[parent_id]
        else:
            # Prefer nodes with fewer children
            child_counts = {}
            for rel in mutated.relations:
                child_counts[rel.source_id] = child_counts.get(rel.source_id, 0) + 1

            candidates = list(mutated.components.keys())
            weights = [1.0 / (child_counts.get(c, 0) + 1) for c in candidates]
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            r = torch.rand(1).item()
            cumsum = 0.0
            parent = mutated.components[candidates[0]]
            for c, w in zip(candidates, weights):
                cumsum += w
                if r <= cumsum:
                    parent = mutated.components[c]
                    break

        # Clone and add donor
        new_comp = donor_component.clone()
        mutated.add_component(new_comp, parent.id, relation_type)

        # Novelty: how different from existing components
        similarities = [new_comp.similarity(c) for c in mutated.components.values() if c.id != new_comp.id]
        novelty = 1.0 - (max(similarities) if similarities else 0.0)

        coherence = self._compute_local_coherence(mutated, new_comp.id)

        return GraftResult(
            idea=mutated,
            operation=GraftOperation.GRAFT,
            source_component=donor_component.id,
            target_component=parent.id,
            novelty_score=novelty,
            coherence_score=coherence
        )

    def prune_component(
        self,
        idea: HierarchicalIdea,
        target_id: Optional[str] = None
    ) -> GraftResult:
        """Remove a component (and its subtree).

        Args:
            idea: Idea to simplify
            target_id: Component to remove (None = random leaf)

        Returns:
            GraftResult with simplified idea
        """
        mutated = idea.clone()

        # Can't prune root
        mutable = [c for c in mutated.get_mutable_components() if c.id != mutated.root.id]
        if not mutable:
            return GraftResult(
                idea=mutated,
                operation=GraftOperation.PRUNE,
                source_component=None,
                target_component=None,
                novelty_score=0.0,
                coherence_score=1.0
            )

        # Prefer leaves for pruning
        leaves = mutated.get_leaf_components()
        mutable_leaves = [c for c in leaves if c.mutable and c.id != mutated.root.id]

        if target_id and target_id in mutated.components:
            target = mutated.components[target_id]
        elif mutable_leaves:
            idx = torch.randint(len(mutable_leaves), (1,)).item()
            target = mutable_leaves[idx]
        else:
            idx = torch.randint(len(mutable), (1,)).item()
            target = mutable[idx]

        # Find all nodes in subtree
        subtree = self._get_subtree(mutated, target.id)

        # Remove subtree components and relations
        for node_id in subtree:
            if node_id in mutated.components:
                del mutated.components[node_id]

        mutated.relations = [
            r for r in mutated.relations
            if r.source_id not in subtree and r.target_id not in subtree
        ]

        mutated._update_depth()

        return GraftResult(
            idea=mutated,
            operation=GraftOperation.PRUNE,
            source_component=None,
            target_component=target.id,
            novelty_score=0.2,  # Pruning is mild novelty
            coherence_score=0.9  # Usually maintains coherence
        )

    def morph_component(
        self,
        idea: HierarchicalIdea,
        donor_component: IdeaComponent,
        target_id: Optional[str] = None,
        blend_factor: float = 0.5
    ) -> GraftResult:
        """Blend a component's embedding with another.

        Args:
            idea: Idea to mutate
            donor_component: Component to blend with
            target_id: Component to morph (None = random)
            blend_factor: How much of donor to use (0-1)

        Returns:
            GraftResult with morphed idea
        """
        mutated = idea.clone()

        mutable = mutated.get_mutable_components()
        if not mutable:
            return GraftResult(
                idea=mutated,
                operation=GraftOperation.MORPH,
                source_component=None,
                target_component=None,
                novelty_score=0.0,
                coherence_score=1.0
            )

        if target_id and target_id in mutated.components:
            target = mutated.components[target_id]
        else:
            idx = torch.randint(len(mutable), (1,)).item()
            target = mutable[idx]

        # Blend embeddings
        original = target.embedding.clone()
        blended = (1 - blend_factor) * target.embedding + blend_factor * donor_component.embedding
        target.embedding = safe_normalize(blended)

        # Novelty is proportional to blend factor and donor distance
        original_sim = F.cosine_similarity(original.unsqueeze(0), donor_component.embedding.unsqueeze(0)).item()
        novelty = blend_factor * (1.0 - original_sim)

        coherence = self._compute_local_coherence(mutated, target.id)

        return GraftResult(
            idea=mutated,
            operation=GraftOperation.MORPH,
            source_component=donor_component.id,
            target_component=target.id,
            novelty_score=novelty,
            coherence_score=coherence
        )

    def transplant_subtree(
        self,
        idea: HierarchicalIdea,
        source_id: Optional[str] = None,
        new_parent_id: Optional[str] = None
    ) -> GraftResult:
        """Move a subtree to a different parent.

        Args:
            idea: Idea to restructure
            source_id: Root of subtree to move
            new_parent_id: New parent for subtree

        Returns:
            GraftResult with restructured idea
        """
        mutated = idea.clone()

        # Get movable components (not root)
        movable = [c for c in mutated.components.values() if c.id != mutated.root.id]
        if len(movable) < 2:  # Need at least 2 components to transplant
            return GraftResult(
                idea=mutated,
                operation=GraftOperation.TRANSPLANT,
                source_component=None,
                target_component=None,
                novelty_score=0.0,
                coherence_score=1.0
            )

        # Select source
        if source_id and source_id in mutated.components:
            source = mutated.components[source_id]
        else:
            idx = torch.randint(len(movable), (1,)).item()
            source = movable[idx]

        # Select new parent (not in source's subtree)
        subtree = self._get_subtree(mutated, source.id)
        valid_parents = [c for c in mutated.components.values() if c.id not in subtree]

        if not valid_parents:
            return GraftResult(
                idea=mutated,
                operation=GraftOperation.TRANSPLANT,
                source_component=source.id,
                target_component=None,
                novelty_score=0.0,
                coherence_score=1.0
            )

        if new_parent_id and new_parent_id in mutated.components and new_parent_id not in subtree:
            new_parent = mutated.components[new_parent_id]
        else:
            idx = torch.randint(len(valid_parents), (1,)).item()
            new_parent = valid_parents[idx]

        # Remove old parent relation
        old_relation = None
        for i, rel in enumerate(mutated.relations):
            if rel.target_id == source.id:
                old_relation = rel
                break

        if old_relation:
            mutated.relations.remove(old_relation)

        # Add new parent relation
        mutated.relations.append(ComponentRelation(
            source_id=new_parent.id,
            target_id=source.id,
            relation_type=old_relation.relation_type if old_relation else RelationType.HAS,
            strength=old_relation.strength if old_relation else 1.0
        ))

        mutated._update_depth()

        # Novelty: moderate for restructuring
        novelty = 0.4

        # Coherence: check fit with new context
        coherence = self._compute_local_coherence(mutated, source.id)

        return GraftResult(
            idea=mutated,
            operation=GraftOperation.TRANSPLANT,
            source_component=source.id,
            target_component=new_parent.id,
            novelty_score=novelty,
            coherence_score=coherence
        )

    def _get_subtree(self, idea: HierarchicalIdea, root_id: str) -> Set[str]:
        """Get all node IDs in subtree rooted at root_id."""
        subtree = {root_id}
        queue = [root_id]

        while queue:
            node = queue.pop(0)
            children = [r.target_id for r in idea.relations if r.source_id == node]
            for child in children:
                if child not in subtree:
                    subtree.add(child)
                    queue.append(child)

        return subtree

    def _compute_local_coherence(
        self,
        idea: HierarchicalIdea,
        component_id: str
    ) -> float:
        """Compute how well a component fits with its neighbors."""
        if component_id not in idea.components:
            return 0.0

        comp = idea.components[component_id]

        # Find neighbors (parent + children + siblings)
        neighbors = []

        parent = idea.get_parent(component_id)
        if parent:
            neighbors.append(parent)
            # Siblings
            siblings = idea.get_children(parent.id)
            neighbors.extend([s for s in siblings if s.id != component_id])

        children = idea.get_children(component_id)
        neighbors.extend(children)

        if not neighbors:
            return 1.0  # No neighbors to compare

        # Average similarity to neighbors
        similarities = [comp.similarity(n) for n in neighbors]
        avg_similarity = sum(similarities) / len(similarities)

        # High similarity = high coherence. Cosine similarity lives in [-1, 1];
        # negative values (opposite vectors) are maximally incoherent, so
        # clamp to [0, 1] to keep the score meaningful as a "coherence".
        return max(0.0, min(1.0, avg_similarity))

    def mutate(
        self,
        idea: HierarchicalIdea,
        donor_pool: List[IdeaComponent],
        dopamine: float = 0.5,
        n_mutations: int = 1
    ) -> List[GraftResult]:
        """Apply random mutations to an idea.

        Args:
            idea: Idea to mutate
            donor_pool: Pool of components to draw from
            dopamine: Controls mutation intensity
            n_mutations: Number of mutations to apply

        Returns:
            List of graft results (cumulative mutations)
        """
        results = []
        current = idea

        for _ in range(n_mutations):
            operation = self.select_operation(dopamine)

            # Select donor if needed
            donor = None
            if donor_pool and operation in [GraftOperation.SWAP, GraftOperation.GRAFT, GraftOperation.MORPH]:
                idx = torch.randint(len(donor_pool), (1,)).item()
                donor = donor_pool[idx]

            if operation == GraftOperation.SWAP and donor:
                result = self.swap_component(current, donor)
            elif operation == GraftOperation.GRAFT and donor:
                result = self.graft_component(current, donor)
            elif operation == GraftOperation.PRUNE:
                result = self.prune_component(current)
            elif operation == GraftOperation.MORPH and donor:
                blend = 0.3 + 0.4 * dopamine  # Higher dopamine = stronger blend
                result = self.morph_component(current, donor, blend_factor=blend)
            elif operation == GraftOperation.TRANSPLANT:
                result = self.transplant_subtree(current)
            else:
                # Fallback: small morph with noise
                noise_comp = IdeaComponent(
                    id="noise",
                    embedding=safe_normalize(torch.randn(self.content_dim)),
                    component_type=ComponentType.MODIFIER
                )
                result = self.morph_component(current, noise_comp, blend_factor=0.1)

            results.append(result)
            current = result.idea

        return results


class HierarchicalIdeaFactory:
    """Factory for creating hierarchical ideas from various sources."""

    def __init__(self, content_dim: int = 32, max_depth: int = 4):
        self.content_dim = content_dim
        self.max_depth = max_depth

    def from_embedding(
        self,
        embedding: torch.Tensor,
        n_components: int = 3,
        label: str = "",
        depth: Optional[int] = None,  # alias for n_components — matches README + tests
    ) -> HierarchicalIdea:
        """Create hierarchical idea from a single embedding.

        Decomposes embedding into components via projection onto random bases.

        Args:
            embedding: Source embedding to decompose.
            n_components: Number of child components to spawn from the root (default 3).
            label: Human-readable label propagated to sub-components.
            depth: Alias for n_components, kept for documented API compatibility.
        """
        if depth is not None:
            n_components = depth
        # Root is the main embedding
        root = IdeaComponent(
            id=str(uuid.uuid4())[:8],
            embedding=safe_normalize(embedding),
            component_type=ComponentType.CORE,
            label=label
        )

        idea = HierarchicalIdea(id=str(uuid.uuid4())[:8], root=root)

        # Decompose into components
        # Use random orthogonal projections
        if n_components > 1:
            # Generate random orthogonal basis
            random_dirs = torch.randn(n_components - 1, self.content_dim)
            random_dirs = F.normalize(random_dirs, dim=1)

            component_types = [ComponentType.ATTRIBUTE, ComponentType.ACTION, ComponentType.PART]

            for i in range(n_components - 1):
                # Project embedding onto direction
                projection = (embedding @ random_dirs[i]) * random_dirs[i]
                residual = embedding - projection

                # Create component from projection + noise
                comp_embedding = safe_normalize(projection + 0.3 * torch.randn(self.content_dim))

                comp = IdeaComponent(
                    id=str(uuid.uuid4())[:8],
                    embedding=comp_embedding,
                    component_type=component_types[i % len(component_types)],
                    label=f"{label}_component_{i}" if label else "",
                    weight=0.5 + 0.5 * torch.rand(1).item()
                )

                idea.add_component(comp, root.id, RelationType.HAS)

        return idea

    def from_embeddings(
        self,
        embeddings: List[torch.Tensor],
        labels: Optional[List[str]] = None
    ) -> HierarchicalIdea:
        """Create hierarchical idea from multiple embeddings.

        First embedding becomes root, others become components.
        """
        if not embeddings:
            return self.random()

        labels = labels or [""] * len(embeddings)

        root = IdeaComponent(
            id=str(uuid.uuid4())[:8],
            embedding=safe_normalize(embeddings[0]),
            component_type=ComponentType.CORE,
            label=labels[0]
        )

        idea = HierarchicalIdea(id=str(uuid.uuid4())[:8], root=root)

        component_types = [ComponentType.ATTRIBUTE, ComponentType.ACTION, ComponentType.PART, ComponentType.MODIFIER]

        for i, emb in enumerate(embeddings[1:], 1):
            comp = IdeaComponent(
                id=str(uuid.uuid4())[:8],
                embedding=safe_normalize(emb),
                component_type=component_types[i % len(component_types)],
                label=labels[i] if i < len(labels) else ""
            )
            idea.add_component(comp, root.id, RelationType.HAS)

        return idea

    def random(self, n_components: int = 4) -> HierarchicalIdea:
        """Create random hierarchical idea."""
        return self.from_embedding(
            torch.randn(self.content_dim),
            n_components=n_components
        )

    def merge(
        self,
        idea_a: HierarchicalIdea,
        idea_b: HierarchicalIdea,
        merge_strategy: str = "combine"  # "combine", "graft", "blend"
    ) -> HierarchicalIdea:
        """Merge two hierarchical ideas.

        Strategies:
        - combine: Create new root with both ideas as children
        - graft: Attach idea_b as subtree of idea_a
        - blend: Blend root embeddings, merge components
        """
        if merge_strategy == "combine":
            # New root from blended embeddings
            new_root_emb = safe_normalize(
                0.5 * idea_a.root.embedding + 0.5 * idea_b.root.embedding
            )
            new_root = IdeaComponent(
                id=str(uuid.uuid4())[:8],
                embedding=new_root_emb,
                component_type=ComponentType.CORE,
                label="merged"
            )

            merged = HierarchicalIdea(id=str(uuid.uuid4())[:8], root=new_root)

            # Add roots of both ideas as children
            comp_a = idea_a.root.clone()
            comp_a.component_type = ComponentType.PART
            merged.add_component(comp_a, new_root.id, RelationType.HAS)

            comp_b = idea_b.root.clone()
            comp_b.component_type = ComponentType.PART
            merged.add_component(comp_b, new_root.id, RelationType.HAS)

            return merged

        elif merge_strategy == "graft":
            # Clone idea_a and graft idea_b onto it
            merged = idea_a.clone()
            graft_root = idea_b.root.clone()

            # Find best attachment point (most similar component)
            best_sim = -1
            best_parent = merged.root

            for comp in merged.components.values():
                sim = comp.similarity(graft_root)
                if sim > best_sim:
                    best_sim = sim
                    best_parent = comp

            merged.add_component(graft_root, best_parent.id, RelationType.LIKE)
            return merged

        else:  # blend
            # Create new idea with blended root and sampled components
            blended_root_emb = safe_normalize(
                0.5 * idea_a.root.embedding + 0.5 * idea_b.root.embedding
            )
            blended_root = IdeaComponent(
                id=str(uuid.uuid4())[:8],
                embedding=blended_root_emb,
                component_type=ComponentType.CORE
            )

            merged = HierarchicalIdea(id=str(uuid.uuid4())[:8], root=blended_root)

            # Sample components from both ideas
            all_components = (
                list(idea_a.components.values()) +
                list(idea_b.components.values())
            )
            # Exclude roots
            components = [c for c in all_components if c.component_type != ComponentType.CORE]

            # Take up to 4 random components
            n_sample = min(4, len(components))
            if components:
                indices = torch.randperm(len(components))[:n_sample]
                for idx in indices:
                    comp = components[idx].clone()
                    merged.add_component(comp, blended_root.id, RelationType.HAS)

            return merged
