"""Living Memory - Core datastructure for LMD.

A LivingMemory is not a static pattern - it is a dynamic entity that:
- Has metabolic energy (can be vivid, dormant, fading, or ghost)
- Follows a narrative phase (setup -> conflict -> climax -> resolution)
- Has an emotional trajectory (valence arc over narrative time)
- Couples with other memories through resonance

Invented by Joshua R. Thomas, January 2026.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class NarrativePhase(Enum):
    """Phases in the narrative arc of a memory.

    Memories exist on a circular manifold representing story structure.
    Phase advances as the memory is processed/replayed.
    """
    SETUP = 0           # phi = 0: Establishing context
    RISING = 1          # phi = pi/2: Tension building
    CLIMAX = 2          # phi = pi: Peak emotional intensity
    RESOLUTION = 3      # phi = 3pi/2: Emotional processing
    INTEGRATION = 4     # phi -> 2pi: Becomes part of self-narrative

    @classmethod
    def from_phase(cls, phi: float) -> NarrativePhase:
        """Convert continuous phase to discrete narrative phase."""
        # Normalize to [0, 2pi)
        phi = phi % (2 * math.pi)

        if phi < math.pi / 4:
            return cls.SETUP
        elif phi < 3 * math.pi / 4:
            return cls.RISING
        elif phi < 5 * math.pi / 4:
            return cls.CLIMAX
        elif phi < 7 * math.pi / 4:
            return cls.RESOLUTION
        else:
            return cls.INTEGRATION


class MetabolicState(Enum):
    """Metabolic state of a living memory based on energy level."""
    VIVID = "vivid"       # E > 1.0: Actively generating, high detail
    ACTIVE = "active"     # 0.5 < E <= 1.0: Engaged, responsive
    DORMANT = "dormant"   # 0.3 < E <= 0.5: Can be triggered, waiting
    FADING = "fading"     # 0.1 < E <= 0.3: Losing detail, may be pruned
    GHOST = "ghost"       # E <= 0.1: Only emotional residue remains

    @classmethod
    def from_energy(cls, energy: float) -> MetabolicState:
        """Determine metabolic state from energy level."""
        if energy > 1.0:
            return cls.VIVID
        elif energy > 0.5:
            return cls.ACTIVE
        elif energy > 0.3:
            return cls.DORMANT
        elif energy > 0.1:
            return cls.FADING
        else:
            return cls.GHOST


@dataclass
class ValenceTrajectory:
    """Emotional arc of a memory over narrative time.

    Unlike static valence tags, this captures the JOURNEY:
    - Redemption arcs: Bad -> Good (learning from failure)
    - Tragedy arcs: Good -> Bad (loss, grief)
    - Oscillating: Ambivalent, complex emotions

    tau ranges from 0.0 (start of memory) to 1.0 (end/resolution)
    """
    points: torch.Tensor  # [n_points] valence values at each tau

    def __post_init__(self):
        """Ensure points is a tensor."""
        if not isinstance(self.points, torch.Tensor):
            self.points = torch.tensor(self.points, dtype=torch.float32)

    @classmethod
    def constant(cls, valence: float, n_points: int = 5) -> ValenceTrajectory:
        """Create a flat valence trajectory (static emotion)."""
        return cls(points=torch.full((n_points,), valence))

    @classmethod
    def redemption(cls, start: float = -0.8, end: float = 0.6, n_points: int = 5) -> ValenceTrajectory:
        """Create a redemption arc: negative -> positive."""
        return cls(points=torch.linspace(start, end, n_points))

    @classmethod
    def tragedy(cls, start: float = 0.7, end: float = -0.5, n_points: int = 5) -> ValenceTrajectory:
        """Create a tragedy arc: positive -> negative."""
        return cls(points=torch.linspace(start, end, n_points))

    @classmethod
    def climax(cls, low: float = 0.2, peak: float = 0.9, n_points: int = 5) -> ValenceTrajectory:
        """Create a climax arc: builds to peak then resolves."""
        half = n_points // 2
        rising = torch.linspace(low, peak, half + 1)
        falling = torch.linspace(peak, low + 0.3, n_points - half)
        return cls(points=torch.cat([rising[:-1], falling]))

    @classmethod
    def random(cls, n_points: int = 5, mean: float = 0.0, std: float = 0.5) -> ValenceTrajectory:
        """Create a random valence trajectory."""
        points = torch.randn(n_points) * std + mean
        return cls(points=torch.clamp(points, -1.0, 1.0))

    def at_tau(self, tau: float) -> float:
        """Get valence at specific narrative time (interpolated)."""
        tau = max(0.0, min(1.0, tau))
        n = len(self.points)

        if n == 1:
            return self.points[0].item()

        # Linear interpolation
        idx_float = tau * (n - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, n - 1)
        frac = idx_float - idx_low

        return (self.points[idx_low] * (1 - frac) + self.points[idx_high] * frac).item()

    def current_valence(self, phase: float) -> float:
        """Get valence at current narrative phase."""
        # Map phase [0, 2pi] to tau [0, 1]
        tau = (phase % (2 * math.pi)) / (2 * math.pi)
        return self.at_tau(tau)

    def arc_type(self) -> str:
        """Classify the emotional arc type."""
        start = self.points[0].item()
        end = self.points[-1].item()
        peak = self.points.max().item()
        trough = self.points.min().item()

        delta = end - start

        if abs(delta) < 0.2:
            if peak - start > 0.3:
                return "climax"
            elif start - trough > 0.3:
                return "valley"
            else:
                return "flat"
        elif delta > 0.3:
            return "redemption"
        elif delta < -0.3:
            return "tragedy"
        else:
            return "drift"

    def gradient(self) -> float:
        """Overall emotional gradient (slope)."""
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1] - self.points[0]).item()

    def intensity(self) -> float:
        """Emotional intensity (variance of trajectory)."""
        return self.points.var().item()


@dataclass
class LivingMemory:
    """A single living memory entity.

    This is the core unit of LMD - a memory that is ALIVE:
    - Has content (what happened)
    - Has emotional trajectory (how it felt over time)
    - Has narrative phase (where in the story)
    - Has metabolic energy (how alive it is)
    - Can couple with other memories

    Invented by Joshua R. Thomas, January 2026.
    """
    id: int
    content: torch.Tensor           # [content_dim] semantic embedding
    valence: ValenceTrajectory      # Emotional arc
    phase: float = 0.0              # Narrative phase [0, 2pi)
    energy: float = 1.0             # Metabolic energy

    # Metadata
    created_at: int = 0             # Timestep when created
    last_activated: int = 0         # Last activation timestep
    activation_count: int = 0       # Total activations

    # Optional context
    label: Optional[str] = None     # Human-readable label
    source: Optional[str] = None    # Where memory came from

    def __post_init__(self):
        """Ensure content is a tensor."""
        if not isinstance(self.content, torch.Tensor):
            self.content = torch.tensor(self.content, dtype=torch.float32)

    @property
    def narrative_phase(self) -> NarrativePhase:
        """Get discrete narrative phase."""
        return NarrativePhase.from_phase(self.phase)

    @property
    def metabolic_state(self) -> MetabolicState:
        """Get metabolic state based on energy."""
        return MetabolicState.from_energy(self.energy)

    @property
    def current_valence(self) -> float:
        """Get valence at current narrative phase."""
        return self.valence.current_valence(self.phase)

    @property
    def is_alive(self) -> bool:
        """Check if memory is still alive (not ghost)."""
        return self.energy > 0.1

    @property
    def is_active(self) -> bool:
        """Check if memory is actively engaged."""
        return self.energy > 0.5

    def activate(self, energy_boost: float = 0.2, timestep: int = 0):
        """Activate this memory, boosting its energy."""
        self.energy = min(2.0, self.energy + energy_boost)
        self.last_activated = timestep
        self.activation_count += 1

    def advance_phase(self, delta_phi: float):
        """Advance narrative phase."""
        self.phase = (self.phase + delta_phi) % (2 * math.pi)

    def decay(self, decay_rate: float = 0.99):
        """Apply metabolic decay."""
        self.energy *= decay_rate

    def similarity(self, other: LivingMemory) -> float:
        """Compute content similarity with another memory."""
        # Cosine similarity
        dot = torch.dot(self.content, other.content)
        norm = torch.norm(self.content) * torch.norm(other.content)
        if norm < 1e-8:
            return 0.0
        return (dot / norm).item()

    def valence_compatibility(self, other: LivingMemory) -> float:
        """Compute emotional compatibility with another memory."""
        # Compare valence trajectories
        v1 = self.valence.points
        v2 = other.valence.points

        # Interpolate if different lengths
        if len(v1) != len(v2):
            n = max(len(v1), len(v2))
            v1 = torch.nn.functional.interpolate(
                v1.unsqueeze(0).unsqueeze(0), size=n, mode='linear'
            ).squeeze()
            v2 = torch.nn.functional.interpolate(
                v2.unsqueeze(0).unsqueeze(0), size=n, mode='linear'
            ).squeeze()

        # Correlation of trajectories
        v1_centered = v1 - v1.mean()
        v2_centered = v2 - v2.mean()

        numer = torch.dot(v1_centered, v2_centered)
        denom = torch.norm(v1_centered) * torch.norm(v2_centered)

        if denom < 1e-8:
            return 0.0

        return (numer / denom).item()

    def phase_alignment(self, other: LivingMemory) -> float:
        """Compute narrative phase alignment (0 = opposite, 1 = aligned)."""
        phase_diff = abs(self.phase - other.phase)
        # Wrap around
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        # Convert to [0, 1] similarity
        return 1.0 - (phase_diff / math.pi)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content.tolist(),
            "valence": self.valence.points.tolist(),
            "phase": self.phase,
            "energy": self.energy,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
            "label": self.label,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LivingMemory:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=torch.tensor(data["content"]),
            valence=ValenceTrajectory(points=torch.tensor(data["valence"])),
            phase=data["phase"],
            energy=data["energy"],
            created_at=data.get("created_at", 0),
            last_activated=data.get("last_activated", 0),
            activation_count=data.get("activation_count", 0),
            label=data.get("label"),
            source=data.get("source"),
        )

    def __repr__(self) -> str:
        return (
            f"LivingMemory(id={self.id}, "
            f"phase={self.narrative_phase.name}, "
            f"energy={self.energy:.2f}/{self.metabolic_state.value}, "
            f"valence={self.current_valence:.2f}, "
            f"arc={self.valence.arc_type()})"
        )
