"""Safeguards for LMD Imagination - Preventing pathological behaviors.

Addresses critical issues:
1. ID Collision: Proper unique ID generation with persistence
2. Echo Chambers: Repulsion from explored ideas (novelty decay)
3. Valence Drift: External reality grounding with outcome learning
4. Runaway Ideation: Resource limits and termination conditions
5. Thread Safety: Proper locking for concurrent access
6. Persistence: Serialization support for restart recovery

"The brain needs guardrails, not just gas pedals" - Joshua, January 2026.
"""

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

# Epsilon for numerical stability
EPS = 1e-8


def safe_normalize(tensor: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Safely normalize a tensor, handling zero vectors."""
    norm = tensor.norm()
    if norm < eps:
        return tensor  # Return as-is if near-zero
    return tensor / norm


def safe_divide(a: float, b: float, default: float = 0.0, eps: float = EPS) -> float:
    """Safely divide, returning default if denominator near zero."""
    if abs(b) < eps:
        return default
    return a / b


class IDGenerator:
    """Thread-safe unique ID generator with persistence support.

    Prevents ID collisions by:
    - Tracking all issued IDs
    - Using atomic counter with locking
    - Supporting ID namespaces (memories vs imagined)
    - Persistence to disk for restart recovery
    """

    def __init__(self, start_id: int = 0, persistence_path: Optional[str] = None):
        self._counter = start_id
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._issued_ids: Set[int] = set()
        self._namespace_offsets = {
            "memory": 0,
            "imagined": 1_000_000,
            "consolidated": 2_000_000,
            "temporary": 3_000_000,
        }
        self._persistence_path = persistence_path

        # Load persisted state if available
        if persistence_path and os.path.exists(persistence_path):
            self._load_state()

    def next_id(self, namespace: str = "memory") -> int:
        """Get next unique ID in namespace."""
        with self._lock:
            offset = self._namespace_offsets.get(namespace, 0)
            # Find next available ID in namespace
            while True:
                new_id = offset + self._counter
                self._counter += 1
                if new_id not in self._issued_ids:
                    break
            self._issued_ids.add(new_id)
            self._maybe_persist()
            return new_id

    def reserve_id(self, id_value: int) -> bool:
        """Reserve a specific ID (returns False if already taken)."""
        with self._lock:
            if id_value in self._issued_ids:
                return False
            self._issued_ids.add(id_value)
            # Update counter if needed
            namespace_offset = 0
            for offset in self._namespace_offsets.values():
                if id_value >= offset:
                    namespace_offset = max(namespace_offset, offset)
            local_id = id_value - namespace_offset
            if local_id >= self._counter:
                self._counter = local_id + 1
            self._maybe_persist()
            return True

    def reserve_ids(self, id_values: List[int]) -> int:
        """Reserve multiple IDs, returns count of successfully reserved."""
        with self._lock:
            reserved = 0
            for id_val in id_values:
                if id_val not in self._issued_ids:
                    self._issued_ids.add(id_val)
                    reserved += 1
            self._maybe_persist()
            return reserved

    def is_issued(self, id_value: int) -> bool:
        """Check if an ID has been issued."""
        with self._lock:
            return id_value in self._issued_ids

    def release_id(self, id_value: int) -> None:
        """Release an ID back to the pool (use carefully)."""
        with self._lock:
            self._issued_ids.discard(id_value)
            self._maybe_persist()

    def get_namespace(self, id_value: int) -> str:
        """Determine which namespace an ID belongs to."""
        for name, offset in sorted(self._namespace_offsets.items(),
                                    key=lambda x: x[1], reverse=True):
            if id_value >= offset:
                return name
        return "unknown"

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for persistence."""
        with self._lock:
            return {
                "counter": self._counter,
                "issued_ids": list(self._issued_ids),
                "version": 1
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialized form."""
        with self._lock:
            self._counter = state.get("counter", 0)
            self._issued_ids = set(state.get("issued_ids", []))

    def _maybe_persist(self) -> None:
        """Persist state if path configured."""
        if self._persistence_path:
            self._save_state()

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self._persistence_path:
            return
        try:
            state = self.get_state()
            with open(self._persistence_path, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass  # Fail silently on persistence errors

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self._persistence_path:
            return
        try:
            with open(self._persistence_path) as f:
                state = json.load(f)
            self.load_state(state)
        except Exception:
            pass  # Fail silently on load errors


@dataclass
class ExploredRegion:
    """Tracks an explored region in idea space with quality metrics."""
    centroid: torch.Tensor
    radius: float
    visit_count: int = 1
    last_visited: float = 0.0  # timestamp
    decay_rate: float = 0.99   # How fast repulsion decays
    quality_score: float = 0.5  # How good were ideas from this region
    was_productive: bool = False  # Did this region produce kept ideas?

    def repulsion_strength(self, current_time: float) -> float:
        """Compute current repulsion strength (decays over time).

        Productive regions have LOWER repulsion (worth revisiting).
        """
        time_elapsed = max(0, current_time - self.last_visited)
        # Cap time elapsed to prevent overflow
        time_elapsed = min(time_elapsed, 1000)
        decay = self.decay_rate ** time_elapsed

        base_repulsion = min(1.0, self.visit_count * 0.1)

        # Reduce repulsion for productive regions
        if self.was_productive:
            base_repulsion *= 0.5

        return base_repulsion * decay

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "centroid": self.centroid.tolist(),
            "radius": self.radius,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited,
            "decay_rate": self.decay_rate,
            "quality_score": self.quality_score,
            "was_productive": self.was_productive
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], content_dim: int) -> "ExploredRegion":
        """Deserialize from persistence."""
        return cls(
            centroid=torch.tensor(data["centroid"]),
            radius=data["radius"],
            visit_count=data["visit_count"],
            last_visited=data["last_visited"],
            decay_rate=data["decay_rate"],
            quality_score=data.get("quality_score", 0.5),
            was_productive=data.get("was_productive", False)
        )


class RepulsionField:
    """Thread-safe repulsion field preventing echo chambers.

    Implements negative novelty for already-explored regions:
    - Tracks explored idea centroids with quality metrics
    - Computes repulsion gradient
    - Decays repulsion over time (allows revisiting with fresh perspective)
    - Differentiates productive vs unproductive regions
    """

    def __init__(
        self,
        content_dim: int,
        repulsion_strength: float = 0.5,
        decay_rate: float = 0.995,
        merge_threshold: float = 0.3,
        max_regions: int = 100
    ):
        self.content_dim = content_dim
        self.repulsion_strength = repulsion_strength
        self.decay_rate = decay_rate
        self.merge_threshold = merge_threshold
        self.max_regions = max_regions

        self._lock = threading.RLock()
        self.explored_regions: List[ExploredRegion] = []

    def mark_explored(
        self,
        embedding: torch.Tensor,
        radius: float = 0.5,
        quality_score: float = 0.5,
        was_productive: bool = False
    ) -> None:
        """Mark a region as explored with quality info."""
        with self._lock:
            current_time = time.time()

            # Validate embedding
            if embedding is None or embedding.numel() == 0:
                return
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                return

            # Check if near existing region
            for region in self.explored_regions:
                diff = embedding - region.centroid
                dist = diff.norm().item()
                if dist < self.merge_threshold:
                    # Merge: update centroid and increment count
                    alpha = safe_divide(1.0, region.visit_count + 1, default=0.5)
                    region.centroid = (1 - alpha) * region.centroid + alpha * embedding
                    region.visit_count += 1
                    region.last_visited = current_time
                    # Update quality (moving average)
                    region.quality_score = 0.8 * region.quality_score + 0.2 * quality_score
                    region.was_productive = region.was_productive or was_productive
                    return

            # New region
            new_region = ExploredRegion(
                centroid=embedding.clone().detach(),
                radius=radius,
                visit_count=1,
                last_visited=current_time,
                decay_rate=self.decay_rate,
                quality_score=quality_score,
                was_productive=was_productive
            )
            self.explored_regions.append(new_region)

            # Prune if too many
            if len(self.explored_regions) > self.max_regions:
                self._prune_weakest()

    def compute_repulsion(self, embedding: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Compute repulsion score and gradient for a point.

        Returns:
            (repulsion_score, repulsion_gradient)
            - score: 0 = no repulsion, 1 = maximum repulsion
            - gradient: direction to move away from explored regions
        """
        with self._lock:
            if not self.explored_regions:
                return 0.0, torch.zeros(self.content_dim)

            # Validate embedding
            if embedding is None or embedding.numel() == 0:
                return 0.0, torch.zeros(self.content_dim)
            if torch.isnan(embedding).any():
                return 0.0, torch.zeros(self.content_dim)

            current_time = time.time()
            total_repulsion = 0.0
            gradient = torch.zeros(self.content_dim)

            for region in self.explored_regions:
                diff = embedding - region.centroid
                dist = diff.norm().item() + EPS

                # Repulsion inversely proportional to distance
                if dist < region.radius * 2:
                    strength = region.repulsion_strength(current_time)
                    local_repulsion = strength * (1.0 - dist / (region.radius * 2))
                    total_repulsion += local_repulsion

                    # Gradient points away from region
                    gradient += (diff / dist) * local_repulsion * self.repulsion_strength

            # Normalize
            total_repulsion = min(1.0, total_repulsion)
            grad_norm = gradient.norm().item()
            if grad_norm > EPS:
                gradient = gradient / grad_norm

            return total_repulsion, gradient

    def apply_repulsion(
        self,
        embedding: torch.Tensor,
        alpha: float = 0.3
    ) -> torch.Tensor:
        """Apply repulsion to move embedding away from explored regions."""
        if embedding is None or embedding.numel() == 0:
            return embedding

        repulsion_score, gradient = self.compute_repulsion(embedding)

        if repulsion_score > 0.1:
            # Move away from explored regions
            orig_norm = embedding.norm().item()
            repelled = embedding + alpha * repulsion_score * gradient
            # Preserve original norm
            if orig_norm > EPS:
                repelled = safe_normalize(repelled) * orig_norm
            return repelled

        return embedding

    def novelty_penalty(self, embedding: torch.Tensor) -> float:
        """Compute novelty penalty (reduces novelty score for explored regions)."""
        repulsion_score, _ = self.compute_repulsion(embedding)
        return repulsion_score  # 0 = novel, 1 = very explored

    def _prune_weakest(self) -> None:
        """Remove weakest explored regions, keeping productive ones."""
        with self._lock:
            current_time = time.time()

            # Sort by repulsion strength (ascending), but keep productive regions
            def sort_key(r):
                strength = r.repulsion_strength(current_time)
                # Boost productive regions so they're kept
                if r.was_productive:
                    strength += 10.0
                return strength

            self.explored_regions.sort(key=sort_key)

            # Keep strongest half
            keep_count = max(self.max_regions // 2, 10)
            self.explored_regions = self.explored_regions[-keep_count:]

    def decay_all(self) -> None:
        """Apply time decay and remove negligible regions."""
        with self._lock:
            current_time = time.time()

            # Remove regions with negligible repulsion
            self.explored_regions = [
                r for r in self.explored_regions
                if r.repulsion_strength(current_time) > 0.01
            ]

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        with self._lock:
            return {
                "content_dim": self.content_dim,
                "repulsion_strength": self.repulsion_strength,
                "decay_rate": self.decay_rate,
                "merge_threshold": self.merge_threshold,
                "max_regions": self.max_regions,
                "regions": [r.to_dict() for r in self.explored_regions],
                "version": 1
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load from serialized state, adjusting timestamps to current time."""
        with self._lock:
            self.explored_regions = []
            current_time = time.time()

            for r_data in state.get("regions", []):
                region = ExploredRegion.from_dict(r_data, self.content_dim)
                # Adjust timestamp relative to current time
                # (preserve relative decay, not absolute time)
                region.last_visited = current_time - 10.0  # Assume 10 seconds since last visit
                self.explored_regions.append(region)


class RealityAnchor:
    """Grounds valence to external reality signals with active learning.

    Prevents "feels good" loops by:
    - Requiring external validation for high-valence ideas
    - Tracking prediction accuracy (did the idea work?)
    - Penalizing ideas that only feel good but don't work
    - Learning from outcomes to calibrate predictions
    """

    def __init__(self, content_dim: int, max_history: int = 100):
        self.content_dim = content_dim
        self._lock = threading.RLock()

        # Track idea outcomes
        self.idea_outcomes: Dict[int, float] = {}  # idea_id -> actual outcome
        self.prediction_errors: deque = deque(maxlen=max_history)

        # Calibration: learned mapping from internal valence to reality
        self.valence_bias = 0.0  # Systematic over/under estimation
        self.valence_scale = 1.0  # Scaling factor

        # External validators
        self.validators: List[Callable[[torch.Tensor], float]] = []

        # Default validator: novelty-based (novel ideas are uncertain)
        self._add_default_validators()

    def _add_default_validators(self) -> None:
        """Add sensible default validators."""
        # Validator 1: Extreme embeddings are suspicious
        def extreme_detector(embedding: torch.Tensor) -> float:
            if embedding is None or embedding.numel() == 0:
                return 0.0
            norm = embedding.norm().item()
            # Very large or very small norms are suspicious
            if norm > 10.0:
                return -0.3  # Penalize extreme
            if norm < 0.1:
                return -0.2  # Penalize near-zero
            return 0.0  # Neutral

        self.validators.append(extreme_detector)

    def register_validator(self, validator: Callable[[torch.Tensor], float]) -> None:
        """Register an external validator function.

        Validator takes embedding, returns reality score (-1 to 1).
        """
        with self._lock:
            self.validators.append(validator)

    def ground_valence(
        self,
        internal_valence: float,
        embedding: torch.Tensor,
        idea_id: Optional[int] = None
    ) -> float:
        """Ground internal valence with external reality checks."""
        with self._lock:
            # Handle invalid inputs
            if math.isnan(internal_valence) or math.isinf(internal_valence):
                internal_valence = 0.0

            # Start with calibrated internal valence
            calibrated = (internal_valence - self.valence_bias) * self.valence_scale

            # Apply external validators
            external_scores = []
            for validator in self.validators:
                try:
                    score = validator(embedding)
                    if not math.isnan(score) and not math.isinf(score):
                        external_scores.append(score)
                except Exception:
                    pass  # Validator failed, skip

            if external_scores:
                # Average external validation
                external_avg = sum(external_scores) / len(external_scores)
                # Blend internal and external (trust external more if we've been wrong)
                error_rate = self._get_error_rate()
                trust_external = min(0.8, 0.3 + error_rate * 0.5)
                grounded = (1 - trust_external) * calibrated + trust_external * external_avg
            else:
                grounded = calibrated

            # Apply historical correction if we have outcome data
            if idea_id is not None and idea_id in self.idea_outcomes:
                actual = self.idea_outcomes[idea_id]
                # Penalize if we were overconfident
                if internal_valence > actual + 0.2:
                    grounded *= 0.8  # Reduce confidence

            return max(-1.0, min(1.0, grounded))

    def record_outcome(
        self,
        idea_id: int,
        predicted_valence: float,
        actual_outcome: float
    ) -> None:
        """Record the actual outcome of an idea for calibration.

        This MUST be called to close the learning loop!
        """
        with self._lock:
            self.idea_outcomes[idea_id] = actual_outcome

            # Track prediction error
            error = predicted_valence - actual_outcome
            if not math.isnan(error) and not math.isinf(error):
                self.prediction_errors.append(error)

            # Update calibration periodically
            if len(self.prediction_errors) >= 10:
                self._update_calibration()

    def _update_calibration(self) -> None:
        """Update calibration based on prediction errors."""
        if len(self.prediction_errors) < 5:
            return

        errors = list(self.prediction_errors)
        mean_error = sum(errors) / len(errors)

        # Update bias (systematic error)
        self.valence_bias = 0.9 * self.valence_bias + 0.1 * mean_error

        # Update scale if we're consistently wrong by a lot
        abs_errors = [abs(e) for e in errors]
        mae = sum(abs_errors) / len(abs_errors)
        if mae > 0.3:
            # We're overconfident, reduce scale
            self.valence_scale = max(0.5, self.valence_scale * 0.95)
        elif mae < 0.1:
            # We're underconfident, increase scale slightly
            self.valence_scale = min(1.5, self.valence_scale * 1.02)

    def _get_error_rate(self) -> float:
        """Get current error rate for trust calculation."""
        if len(self.prediction_errors) < 3:
            return 0.5  # Default uncertainty
        errors = list(self.prediction_errors)
        abs_errors = [abs(e) for e in errors[-10:]]  # Recent errors
        return min(1.0, sum(abs_errors) / len(abs_errors))

    def get_calibration_stats(self) -> Dict[str, float]:
        """Get current calibration statistics."""
        with self._lock:
            if not self.prediction_errors:
                return {"bias": 0.0, "scale": 1.0, "mae": 0.0, "n_samples": 0}

            errors = list(self.prediction_errors)
            return {
                "bias": self.valence_bias,
                "scale": self.valence_scale,
                "mae": sum(abs(e) for e in errors) / len(errors),
                "n_samples": len(errors),
                "error_rate": self._get_error_rate()
            }

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        with self._lock:
            return {
                "valence_bias": self.valence_bias,
                "valence_scale": self.valence_scale,
                "prediction_errors": list(self.prediction_errors),
                "idea_outcomes": {str(k): v for k, v in self.idea_outcomes.items()},
                "version": 1
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load from serialized state."""
        with self._lock:
            self.valence_bias = state.get("valence_bias", 0.0)
            self.valence_scale = state.get("valence_scale", 1.0)
            self.prediction_errors = deque(
                state.get("prediction_errors", []),
                maxlen=100
            )
            self.idea_outcomes = {
                int(k): v for k, v in state.get("idea_outcomes", {}).items()
            }


class AutonomyTrigger(Enum):
    """Conditions that can trigger autonomous ideation."""
    IDLE = auto()           # System has been idle
    CURIOSITY = auto()      # Curiosity threshold exceeded
    PROBLEM = auto()        # Active problem needs solving
    SCHEDULED = auto()      # Scheduled ideation time
    BOREDOM = auto()        # Low novelty in recent experience
    CONSOLIDATION = auto()  # Post-sleep memory consolidation


@dataclass
class TriggerCondition:
    """A condition that can trigger autonomous ideation."""
    trigger_type: AutonomyTrigger
    threshold: float
    current_value: float = 0.0
    cooldown_seconds: float = 60.0
    last_triggered: float = 0.0

    def should_trigger(self, current_time: float) -> bool:
        """Check if this condition should trigger ideation."""
        # Check cooldown
        if current_time - self.last_triggered < self.cooldown_seconds:
            return False
        return self.current_value >= self.threshold

    def mark_triggered(self, current_time: float) -> None:
        """Mark this trigger as having fired."""
        self.last_triggered = current_time
        self.current_value = 0.0  # Reset


class AutonomyController:
    """Thread-safe controller for autonomous ideation.

    Monitors various conditions and triggers ideation when appropriate.
    Prevents runaway ideation with resource limits.
    """

    def __init__(
        self,
        max_ideas_per_hour: int = 50,
        max_compute_per_session: int = 100000,  # tensor ops
        min_interval_seconds: float = 30.0
    ):
        self.max_ideas_per_hour = max_ideas_per_hour
        self.max_compute_per_session = max_compute_per_session
        self.min_interval_seconds = min_interval_seconds

        self._lock = threading.RLock()

        # Resource tracking
        self.ideas_this_hour: int = 0
        self.hour_start: float = time.time()
        self.last_ideation: float = 0.0
        self.compute_used: int = 0

        # Trigger conditions
        self.triggers: Dict[AutonomyTrigger, TriggerCondition] = {
            AutonomyTrigger.IDLE: TriggerCondition(
                trigger_type=AutonomyTrigger.IDLE,
                threshold=60.0,  # 60 seconds idle
                cooldown_seconds=120.0
            ),
            AutonomyTrigger.CURIOSITY: TriggerCondition(
                trigger_type=AutonomyTrigger.CURIOSITY,
                threshold=0.7,  # Curiosity score
                cooldown_seconds=180.0
            ),
            AutonomyTrigger.BOREDOM: TriggerCondition(
                trigger_type=AutonomyTrigger.BOREDOM,
                threshold=0.8,  # Low novelty for a while
                cooldown_seconds=300.0
            ),
        }

        # Callbacks
        self.on_trigger: Optional[Callable[[AutonomyTrigger], None]] = None

    def update_trigger(self, trigger_type: AutonomyTrigger, value: float) -> None:
        """Update a trigger's current value."""
        with self._lock:
            if trigger_type in self.triggers:
                self.triggers[trigger_type].current_value = value

    def check_triggers(self) -> Optional[AutonomyTrigger]:
        """Check if any trigger should fire."""
        with self._lock:
            current_time = time.time()

            # Check resource limits first
            if not self._check_resource_limits(current_time):
                return None

            # Check each trigger
            for trigger_type, condition in self.triggers.items():
                if condition.should_trigger(current_time):
                    condition.mark_triggered(current_time)
                    return trigger_type

            return None

    def _check_resource_limits(self, current_time: float) -> bool:
        """Check if we're within resource limits."""
        # Reset hourly counter if needed
        if current_time - self.hour_start > 3600:
            self.ideas_this_hour = 0
            self.hour_start = current_time

        # Check limits
        if self.ideas_this_hour >= self.max_ideas_per_hour:
            return False

        if current_time - self.last_ideation < self.min_interval_seconds:
            return False

        return True

    def record_ideation(self, n_ideas: int, compute_ops: int) -> None:
        """Record that ideation occurred."""
        with self._lock:
            self.ideas_this_hour += n_ideas
            self.last_ideation = time.time()
            self.compute_used += compute_ops

    def can_ideate(self) -> bool:
        """Check if ideation is currently allowed."""
        with self._lock:
            return self._check_resource_limits(time.time())

    def get_status(self) -> Dict[str, Any]:
        """Get current autonomy status."""
        with self._lock:
            current_time = time.time()
            return {
                "ideas_this_hour": self.ideas_this_hour,
                "max_ideas_per_hour": self.max_ideas_per_hour,
                "seconds_since_last": current_time - self.last_ideation,
                "can_ideate": self.can_ideate(),
                "triggers": {
                    t.name: {
                        "value": c.current_value,
                        "threshold": c.threshold,
                        "would_trigger": c.should_trigger(current_time)
                    }
                    for t, c in self.triggers.items()
                }
            }


@dataclass
class ResourceBudget:
    """Budget for a single ideation session with safety limits."""
    max_ideas: int = 20
    max_iterations: int = 10
    max_tensor_ops: int = 50000
    max_time_seconds: float = 30.0

    # Tracking
    ideas_generated: int = 0
    iterations_used: int = 0
    ops_used: int = 0
    start_time: float = field(default_factory=time.time)

    # Safety: hard limits that can't be exceeded
    HARD_MAX_IDEAS: int = 100
    HARD_MAX_ITERATIONS: int = 50
    HARD_MAX_TIME: float = 120.0

    def __post_init__(self):
        # Enforce hard limits
        self.max_ideas = min(self.max_ideas, self.HARD_MAX_IDEAS)
        self.max_iterations = min(self.max_iterations, self.HARD_MAX_ITERATIONS)
        self.max_time_seconds = min(self.max_time_seconds, self.HARD_MAX_TIME)

    def is_exhausted(self) -> bool:
        """Check if any budget is exhausted."""
        elapsed = time.time() - self.start_time
        return (
            self.ideas_generated >= self.max_ideas or
            self.iterations_used >= self.max_iterations or
            self.ops_used >= self.max_tensor_ops or
            elapsed >= self.max_time_seconds
        )

    def record_idea(self) -> None:
        self.ideas_generated += 1

    def record_iteration(self) -> None:
        self.iterations_used += 1

    def record_ops(self, n_ops: int) -> None:
        self.ops_used += n_ops

    def remaining(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "ideas": self.max_ideas - self.ideas_generated,
            "iterations": self.max_iterations - self.iterations_used,
            "ops": self.max_tensor_ops - self.ops_used,
            "time": max(0, self.max_time_seconds - elapsed)
        }


# Global ID generator instance with thread-safe access
_global_id_generator: Optional[IDGenerator] = None
_global_id_lock = threading.Lock()


def get_id_generator(persistence_path: Optional[str] = None) -> IDGenerator:
    """Get or create the global ID generator."""
    global _global_id_generator
    with _global_id_lock:
        if _global_id_generator is None:
            _global_id_generator = IDGenerator(persistence_path=persistence_path)
        return _global_id_generator


def reset_id_generator(start_id: int = 0, persistence_path: Optional[str] = None) -> None:
    """Reset the global ID generator (use carefully, mainly for testing)."""
    global _global_id_generator
    with _global_id_lock:
        _global_id_generator = IDGenerator(start_id, persistence_path=persistence_path)
