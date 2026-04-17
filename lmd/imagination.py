"""Imagination Engine - Will-directed mental simulation and transformation.

The brain can imagine things that don't exist by:
1. Combining parts from different memories (dragon + butterfly wings)
2. Transforming properties (morph, bulge, split, recolor)
3. Placing imagined entities in contexts (on couch, smoking pipe)

Key insight: "The I WANT part is the Key thing" - Joshua
- No random generation without volitional control
- Will (W) directs imagination (I)
- External will (user intent) or internal will (curiosity, problems)

Invented by Joshua R. Thomas, January 2026.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .living_memory import LivingMemory
from .safeguards import EPS, safe_divide, safe_normalize


class SlotType(Enum):
    """Types of slots in structured memory."""
    AGENT = auto()      # Who/what (dragon, person, object)
    ACTION = auto()     # What doing (flying, sitting, smoking)
    LOCATION = auto()   # Where (couch, sky, forest)
    ATTRIBUTE = auto()  # Properties (color, size, shape)
    RELATION = auto()   # How things connect (on, under, with)
    PART = auto()       # Component parts (wings, eyes, nose)


@dataclass
class MemorySlot:
    """A slot in structured memory representation."""
    slot_type: SlotType
    name: str                      # e.g., "wings", "color", "location"
    content: torch.Tensor          # Embedding of this slot's value
    source_memory_id: Optional[int] = None  # Where this came from
    confidence: float = 1.0        # How certain we are about this slot

    def clone(self) -> "MemorySlot":
        """Create a deep copy."""
        return MemorySlot(
            slot_type=self.slot_type,
            name=self.name,
            content=self.content.clone(),
            source_memory_id=self.source_memory_id,
            confidence=self.confidence
        )


@dataclass
class StructuredMemory:
    """Memory decomposed into parts/slots for manipulation.

    Instead of a single embedding, memories are structured:
    - Agent: dragon
    - Parts: wings (butterfly-style), eyes (bulging), nose (splitting)
    - Attributes: color (green), size (large)
    - Location: couch
    - Action: smoking pipe
    """
    id: int
    slots: Dict[str, MemorySlot] = field(default_factory=dict)
    valence: float = 0.0           # Emotional value
    novelty: float = 0.0           # How novel/creative
    coherence: float = 1.0         # Internal consistency
    source_memories: Set[int] = field(default_factory=set)  # Contributing memories

    def add_slot(self, name: str, slot: MemorySlot) -> None:
        """Add or replace a slot."""
        self.slots[name] = slot
        if slot.source_memory_id is not None:
            self.source_memories.add(slot.source_memory_id)

    def get_slot(self, name: str) -> Optional[MemorySlot]:
        """Get a slot by name."""
        return self.slots.get(name)

    def get_slots_by_type(self, slot_type: SlotType) -> List[MemorySlot]:
        """Get all slots of a given type."""
        return [s for s in self.slots.values() if s.slot_type == slot_type]

    def to_embedding(self, dim: int = 32) -> torch.Tensor:
        """Combine slots back into single embedding."""
        if not self.slots:
            return torch.zeros(dim)

        # Weighted average of slot contents
        embeddings = []
        weights = []
        for slot in self.slots.values():
            embeddings.append(slot.content)
            weights.append(max(slot.confidence, EPS))  # Ensure non-zero weight

        # Stack and weight
        stacked = torch.stack(embeddings)
        weight_tensor = torch.tensor(weights).unsqueeze(1)
        total_weight = sum(weights)
        if total_weight < EPS:
            total_weight = 1.0  # Fallback to equal weighting
        weighted = (stacked * weight_tensor).sum(0) / total_weight

        # Ensure correct dimension
        if weighted.shape[0] != dim:
            # Simple projection
            if weighted.shape[0] > dim:
                weighted = weighted[:dim]
            else:
                padding = torch.zeros(dim - weighted.shape[0])
                weighted = torch.cat([weighted, padding])

        return weighted

    def clone(self) -> "StructuredMemory":
        """Create a deep copy."""
        new_mem = StructuredMemory(
            id=self.id,
            valence=self.valence,
            novelty=self.novelty,
            coherence=self.coherence,
            source_memories=self.source_memories.copy()
        )
        for name, slot in self.slots.items():
            new_mem.slots[name] = slot.clone()
        return new_mem


class TransformType(Enum):
    """Types of imagination transforms."""
    MORPH = auto()      # Gradual shape change
    SCALE = auto()      # Make bigger/smaller
    BULGE = auto()      # Local inflation
    SPLIT = auto()      # Divide into parts
    MERGE = auto()      # Combine parts
    RECOLOR = auto()    # Change color/texture
    TRANSPLANT = auto() # Take part from one, add to another
    REMOVE = auto()     # Delete a part
    ADD = auto()        # Add new part
    ROTATE = auto()     # Change orientation
    ANIMATE = auto()    # Add motion/action


@dataclass
class Transform:
    """A transformation to apply to imagination."""
    transform_type: TransformType
    target_slot: str              # Which slot to transform
    parameters: Dict[str, Any] = field(default_factory=dict)
    magnitude: float = 0.5        # How much to transform (0-1)

    def describe(self) -> str:
        """Human-readable description."""
        return f"{self.transform_type.name}({self.target_slot}, mag={self.magnitude:.2f})"


class TransformOps:
    """Transform operations for imagination.

    Implements: MORPH, BULGE, SPLIT, RECOLOR, TRANSPLANT, etc.
    """

    def __init__(self, content_dim: int = 32):
        self.content_dim = content_dim

    def apply(
        self,
        memory: StructuredMemory,
        transform: Transform
    ) -> StructuredMemory:
        """Apply a transform to a structured memory."""
        result = memory.clone()

        method = getattr(self, f"_apply_{transform.transform_type.name.lower()}", None)
        if method:
            result = method(result, transform)

        # Update novelty (transforms increase novelty)
        result.novelty = min(1.0, result.novelty + 0.1 * transform.magnitude)

        return result

    def _apply_morph(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Gradual shape change - interpolate toward target."""
        slot = memory.get_slot(transform.target_slot)
        if slot is None:
            return memory

        # Morph by adding noise in a direction
        target = transform.parameters.get("target_embedding")
        if target is not None:
            # Interpolate toward target
            new_content = (1 - transform.magnitude) * slot.content + transform.magnitude * target
        else:
            # Random morph direction
            noise = torch.randn_like(slot.content) * transform.magnitude * 0.3
            new_content = slot.content + noise

        # Preserve original norm while normalizing safely
        orig_norm = slot.content.norm().item()
        slot.content = safe_normalize(new_content) * max(orig_norm, EPS)
        slot.confidence *= (1 - 0.1 * transform.magnitude)  # Morphing reduces confidence
        return memory

    def _apply_scale(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Scale a part bigger or smaller."""
        slot = memory.get_slot(transform.target_slot)
        if slot is None:
            return memory

        # Scale factor: 1.0 = no change, >1 = bigger, <1 = smaller
        scale_factor = transform.parameters.get("factor", 1.0 + transform.magnitude)
        slot.content = slot.content * scale_factor
        return memory

    def _apply_bulge(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Local inflation - exaggerate certain dimensions."""
        slot = memory.get_slot(transform.target_slot)
        if slot is None:
            return memory

        # Bulge specific dimensions
        bulge_dims = transform.parameters.get("dims", list(range(min(8, self.content_dim))))
        bulge_factor = 1.0 + transform.magnitude

        for dim in bulge_dims:
            if dim < slot.content.shape[0]:
                slot.content[dim] *= bulge_factor

        return memory

    def _apply_split(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Split one part into multiple."""
        slot = memory.get_slot(transform.target_slot)
        if slot is None:
            return memory

        n_parts = transform.parameters.get("n_parts", 2)
        base_name = transform.target_slot

        # Create split parts with slight variations
        for i in range(n_parts):
            new_slot = slot.clone()
            new_slot.name = f"{base_name}_{i+1}"
            # Add variation
            noise = torch.randn_like(slot.content) * 0.1 * transform.magnitude
            new_slot.content = slot.content + noise
            new_slot.confidence *= 0.8  # Split reduces confidence
            memory.add_slot(new_slot.name, new_slot)

        # Keep original as weakened
        slot.confidence *= 0.5
        return memory

    def _apply_merge(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Merge multiple parts into one."""
        slots_to_merge = transform.parameters.get("slots", [])
        if len(slots_to_merge) < 2:
            return memory

        # Average the contents
        contents = []
        for slot_name in slots_to_merge:
            slot = memory.get_slot(slot_name)
            if slot:
                contents.append(slot.content)

        if contents:
            merged_content = torch.stack(contents).mean(0)
            new_slot = MemorySlot(
                slot_type=SlotType.PART,
                name=transform.target_slot,
                content=merged_content,
                confidence=0.7
            )
            memory.add_slot(transform.target_slot, new_slot)

        return memory

    def _apply_recolor(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Change color/texture attributes."""
        # Find or create color attribute slot
        color_slot = memory.get_slot("color")
        if color_slot is None:
            color_slot = MemorySlot(
                slot_type=SlotType.ATTRIBUTE,
                name="color",
                content=torch.randn(self.content_dim),
                confidence=0.9
            )
            memory.add_slot("color", color_slot)

        # Apply color change
        target_color = transform.parameters.get("target_color")
        if target_color is not None:
            color_slot.content = (1 - transform.magnitude) * color_slot.content + transform.magnitude * target_color
        else:
            # Random color shift
            color_slot.content = color_slot.content + torch.randn_like(color_slot.content) * transform.magnitude * 0.5

        return memory

    def _apply_transplant(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Take a part from source memory and add to target."""
        source_slot = transform.parameters.get("source_slot")
        if source_slot is None:
            return memory

        # Add the transplanted slot
        new_slot = source_slot.clone()
        new_slot.name = transform.target_slot
        new_slot.confidence *= 0.8  # Transplant reduces confidence
        memory.add_slot(transform.target_slot, new_slot)
        memory.source_memories.add(source_slot.source_memory_id)

        return memory

    def _apply_remove(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Remove a part."""
        if transform.target_slot in memory.slots:
            del memory.slots[transform.target_slot]
        return memory

    def _apply_add(self, memory: StructuredMemory, transform: Transform) -> StructuredMemory:
        """Add a new part."""
        content = transform.parameters.get("content", torch.randn(self.content_dim))
        slot_type = transform.parameters.get("slot_type", SlotType.PART)

        new_slot = MemorySlot(
            slot_type=slot_type,
            name=transform.target_slot,
            content=content,
            confidence=0.7 * (1 - transform.magnitude * 0.3)  # Novel parts have lower confidence
        )
        memory.add_slot(transform.target_slot, new_slot)
        return memory


@dataclass
class WillVector:
    """Represents a will/intent for imagination.

    'The I WANT part is the Key thing' - Joshua

    Will directs imagination - no random generation without volitional control.
    """
    direction: torch.Tensor        # What we want (embedding space direction)
    strength: float = 1.0          # How strongly we want it
    specificity: float = 0.5       # How specific (0=vague, 1=precise)
    source: str = "external"       # "external" (user) or "internal" (curiosity)
    description: Optional[str] = None  # Human-readable description

    def apply_to(self, content: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        """Apply will as a bias to content generation."""
        # Will biases the direction of imagination
        bias = self.direction * self.strength * alpha
        return content + bias


class WillGenerator:
    """Generates internal will from curiosity, problems, and exploration.

    W_internal = Curiosity(M) + Problem(G) + Random(eta)

    Where:
    - Curiosity: Gaps in memory coverage (what don't we know?)
    - Problem: Active goals needing solutions
    - Random: Exploration noise for creativity
    """

    def __init__(self, content_dim: int = 32, curiosity_weight: float = 0.4):
        self.content_dim = content_dim
        self.curiosity_weight = curiosity_weight
        self.problem_weight = 0.4
        self.exploration_weight = 0.2

        # Track what we've explored
        self.explored_regions: List[torch.Tensor] = []
        self.active_problems: List[WillVector] = []

    def generate_curiosity_will(
        self,
        memories: List[LivingMemory],
        n_samples: int = 100
    ) -> WillVector:
        """Generate will from curiosity - explore gaps in memory.

        Curiosity = gradient toward unexplored regions of content space.
        """
        if not memories:
            # No memories - explore randomly
            direction = torch.randn(self.content_dim)
            direction = safe_normalize(direction)
            return WillVector(
                direction=direction,
                strength=0.8,
                specificity=0.2,
                source="curiosity",
                description="Exploring unknown territory"
            )

        # Sample points and find least covered region
        memory_contents = torch.stack([m.content for m in memories])

        # Generate random probe points
        probes = torch.randn(n_samples, self.content_dim)

        # Find probe with maximum distance from all memories
        distances = torch.cdist(probes, memory_contents)
        min_distances = distances.min(dim=1).values  # Distance to nearest memory

        # Also penalize probes near explored regions (repulsion)
        if self.explored_regions:
            explored_stack = torch.stack(self.explored_regions)
            explored_dists = torch.cdist(probes, explored_stack)
            min_explored_dist = explored_dists.min(dim=1).values
            # Reduce score for probes near explored regions
            adjusted_distances = min_distances * (0.5 + 0.5 * torch.sigmoid(min_explored_dist - 0.5))
        else:
            adjusted_distances = min_distances

        # Pick probe that's farthest from memories AND explored regions
        best_probe_idx = adjusted_distances.argmax()
        curiosity_direction = probes[best_probe_idx]
        curiosity_direction = safe_normalize(curiosity_direction)

        # Strength based on how far the gap is
        gap_distance = min_distances[best_probe_idx].item()
        strength = min(1.0, gap_distance / 2.0)

        return WillVector(
            direction=curiosity_direction,
            strength=strength,
            specificity=0.3,  # Curiosity is somewhat vague
            source="curiosity",
            description=f"Curious about unexplored region (gap={gap_distance:.2f})"
        )

    def generate_problem_will(
        self,
        goal_description: str,
        goal_embedding: torch.Tensor
    ) -> WillVector:
        """Generate will from an active problem/goal."""
        # Normalize goal direction safely
        direction = safe_normalize(goal_embedding)

        will = WillVector(
            direction=direction,
            strength=0.9,  # Problems are high priority
            specificity=0.7,  # Problems are usually specific
            source="problem",
            description=f"Solving: {goal_description}"
        )

        self.active_problems.append(will)
        return will

    def generate_exploration_will(self) -> WillVector:
        """Generate random exploration will for creativity."""
        direction = torch.randn(self.content_dim)
        direction = safe_normalize(direction)

        return WillVector(
            direction=direction,
            strength=0.5,
            specificity=0.1,  # Very vague - just exploring
            source="exploration",
            description="Random creative exploration"
        )

    def mark_explored(self, embedding: torch.Tensor, max_regions: int = 100) -> None:
        """Mark a region as explored to avoid revisiting.

        This enables repulsion from already-explored idea space regions.
        """
        if embedding is None or embedding.numel() == 0:
            return
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            return

        self.explored_regions.append(embedding.clone().detach())

        # Prune if too many (keep most recent)
        if len(self.explored_regions) > max_regions:
            self.explored_regions = self.explored_regions[-max_regions:]

    def clear_explored(self) -> None:
        """Clear explored regions (e.g., after sleep/consolidation)."""
        self.explored_regions.clear()

    def generate_combined_will(
        self,
        memories: List[LivingMemory],
        external_will: Optional[WillVector] = None
    ) -> WillVector:
        """Generate combined will from all sources.

        W = W_external + alpha*Curiosity + beta*Problem + gamma*Random
        """
        components = []
        weights = []

        # External will (if provided) - highest priority
        if external_will is not None:
            components.append(external_will.direction * external_will.strength)
            weights.append(0.5)

        # Curiosity will
        curiosity = self.generate_curiosity_will(memories)
        components.append(curiosity.direction * curiosity.strength * self.curiosity_weight)
        weights.append(self.curiosity_weight)

        # Problem will (use most recent if any)
        if self.active_problems:
            problem = self.active_problems[-1]
            components.append(problem.direction * problem.strength * self.problem_weight)
            weights.append(self.problem_weight)

        # Exploration will
        exploration = self.generate_exploration_will()
        components.append(exploration.direction * exploration.strength * self.exploration_weight)
        weights.append(self.exploration_weight)

        # Combine safely
        total_weight = sum(weights)
        combined = sum(c * w for c, w in zip(components, weights))
        combined = safe_divide(1.0, total_weight, default=1.0) * combined
        combined = safe_normalize(combined)

        # Compute overall strength and specificity
        strength = safe_divide(
            sum(w * (0.8 if i == 0 and external_will else 0.5) for i, w in enumerate(weights)),
            total_weight,
            default=0.5
        )
        specificity = 0.7 if external_will else 0.3

        source = "combined"
        if external_will:
            source = "external+internal"

        return WillVector(
            direction=combined,
            strength=strength,
            specificity=specificity,
            source=source,
            description="Combined will from multiple sources"
        )


@dataclass
class CanvasEntity:
    """An entity on the mental canvas."""
    id: int
    memory: StructuredMemory
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Spatial position
    scale: float = 1.0
    active: bool = True
    transform_history: List[Transform] = field(default_factory=list)


class MentalCanvas:
    """The imagination workspace - where imagined entities exist.

    A mental canvas for composing and transforming imagined scenes.
    """

    def __init__(self, content_dim: int = 32, max_entities: int = 10):
        self.content_dim = content_dim
        self.max_entities = max_entities
        self.entities: Dict[int, CanvasEntity] = {}
        self.next_id = 0
        self.transform_ops = TransformOps(content_dim)

    def add_entity(
        self,
        memory: StructuredMemory,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> int:
        """Add an entity to the canvas."""
        entity_id = self.next_id
        self.next_id += 1

        self.entities[entity_id] = CanvasEntity(
            id=entity_id,
            memory=memory.clone(),
            position=position
        )

        # Enforce max entities
        if len(self.entities) > self.max_entities:
            # Remove oldest inactive
            for eid in list(self.entities.keys()):
                if not self.entities[eid].active:
                    del self.entities[eid]
                    break

        return entity_id

    def transform_entity(
        self,
        entity_id: int,
        transform: Transform
    ) -> bool:
        """Apply a transform to an entity."""
        if entity_id not in self.entities:
            return False

        entity = self.entities[entity_id]
        entity.memory = self.transform_ops.apply(entity.memory, transform)
        entity.transform_history.append(transform)
        return True

    def compose_entities(
        self,
        entity_ids: List[int],
        composition_type: str = "scene"
    ) -> Optional[StructuredMemory]:
        """Compose multiple entities into a new structured memory."""
        if not entity_ids:
            return None

        # Gather all entities
        entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]
        if not entities:
            return None

        # Create composed memory
        composed = StructuredMemory(id=-1)

        for i, entity in enumerate(entities):
            for slot_name, slot in entity.memory.slots.items():
                # Prefix with entity index to avoid collisions
                new_name = f"entity_{i}_{slot_name}"
                composed.add_slot(new_name, slot.clone())

        # Compute composed properties
        composed.novelty = sum(e.memory.novelty for e in entities) / len(entities)
        composed.valence = sum(e.memory.valence for e in entities) / len(entities)

        # Coherence decreases with more entities
        composed.coherence = 1.0 / (1.0 + 0.1 * len(entities))

        return composed

    def get_entity(self, entity_id: int) -> Optional[CanvasEntity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def clear(self) -> None:
        """Clear the canvas."""
        self.entities.clear()

    def get_scene_embedding(self) -> torch.Tensor:
        """Get embedding of entire scene."""
        if not self.entities:
            return torch.zeros(self.content_dim)

        embeddings = []
        for entity in self.entities.values():
            if entity.active:
                embeddings.append(entity.memory.to_embedding(self.content_dim))

        if not embeddings:
            return torch.zeros(self.content_dim)

        return torch.stack(embeddings).mean(0)


class MemoryDecomposer:
    """Decomposes LivingMemory into StructuredMemory.

    Takes a holistic memory embedding and extracts parts/slots.
    """

    def __init__(self, content_dim: int = 32):
        self.content_dim = content_dim

        # Learned slot extractors (in practice, these would be trained)
        # For prototype, use simple projections
        self.slot_projectors = {
            SlotType.AGENT: self._random_projection(),
            SlotType.ACTION: self._random_projection(),
            SlotType.LOCATION: self._random_projection(),
            SlotType.ATTRIBUTE: self._random_projection(),
        }

    def _random_projection(self) -> torch.Tensor:
        """Create a random projection matrix."""
        proj = torch.randn(self.content_dim, self.content_dim)
        return proj / proj.norm()

    def decompose(
        self,
        memory: LivingMemory,
        slot_names: Optional[Dict[SlotType, str]] = None
    ) -> StructuredMemory:
        """Decompose a LivingMemory into structured slots."""
        structured = StructuredMemory(
            id=memory.id,
            valence=memory.current_valence,
            source_memories={memory.id}
        )

        # Default slot names
        if slot_names is None:
            slot_names = {
                SlotType.AGENT: "agent",
                SlotType.ACTION: "action",
                SlotType.LOCATION: "location",
                SlotType.ATTRIBUTE: "attribute"
            }

        # Extract each slot type
        for slot_type, name in slot_names.items():
            if slot_type in self.slot_projectors:
                proj = self.slot_projectors[slot_type]
                slot_content = memory.content @ proj

                slot = MemorySlot(
                    slot_type=slot_type,
                    name=name,
                    content=slot_content,
                    source_memory_id=memory.id,
                    confidence=0.8
                )
                structured.add_slot(name, slot)

        return structured

    def decompose_with_parts(
        self,
        memory: LivingMemory,
        part_names: List[str]
    ) -> StructuredMemory:
        """Decompose with specific part names."""
        structured = self.decompose(memory)

        # Add parts
        for i, part_name in enumerate(part_names):
            # Project different subspaces for different parts
            start_idx = (i * 8) % self.content_dim
            end_idx = min(start_idx + 8, self.content_dim)

            part_content = torch.zeros(self.content_dim)
            part_content[start_idx:end_idx] = memory.content[start_idx:end_idx]

            slot = MemorySlot(
                slot_type=SlotType.PART,
                name=part_name,
                content=part_content,
                source_memory_id=memory.id,
                confidence=0.7
            )
            structured.add_slot(part_name, slot)

        return structured
