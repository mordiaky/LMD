"""Heartbeat-Triggered Autonomous Ideation.

Integrates LMD's AutonomyController with the brain's heartbeat clock
to enable true autonomous ideation without external API calls.

The heartbeat provides the "tick" that drives spontaneous thinking.

Invented by Joshua R. Thomas, January 2026.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .config import LMDConfig
from .dynamics import LMDDynamics
from .ideation import AutonomousIdeator, IdeationResult
from .living_memory import LivingMemory, ValenceTrajectory
from .safeguards import AutonomyTrigger


@dataclass
class HeartbeatIdeationMetrics:
    """Metrics from heartbeat-driven ideation."""
    total_heartbeats: int = 0
    total_ideation_sessions: int = 0
    total_ideas_generated: int = 0
    total_ideas_consolidated: int = 0
    average_session_quality: float = 0.0
    last_ideation_time: float = 0.0
    idle_seconds: float = 0.0

    # Per-trigger counts
    trigger_counts: Dict[str, int] = field(default_factory=dict)

    def record_session(self, result: IdeationResult, trigger: str) -> None:
        """Record an ideation session."""
        self.total_ideation_sessions += 1
        self.total_ideas_generated += result.total_ideas_generated
        self.last_ideation_time = time.time()

        # Update average quality
        if result.best_score > 0:
            n = self.total_ideation_sessions
            self.average_session_quality = (
                (self.average_session_quality * (n - 1) + result.best_score) / n
            )

        # Track trigger
        self.trigger_counts[trigger] = self.trigger_counts.get(trigger, 0) + 1

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Heartbeat Ideation Metrics:\n"
            f"  Total heartbeats: {self.total_heartbeats}\n"
            f"  Ideation sessions: {self.total_ideation_sessions}\n"
            f"  Ideas generated: {self.total_ideas_generated}\n"
            f"  Ideas consolidated: {self.total_ideas_consolidated}\n"
            f"  Average quality: {self.average_session_quality:.3f}\n"
            f"  Idle time: {self.idle_seconds:.1f}s\n"
            f"  Triggers: {self.trigger_counts}"
        )


class HeartbeatIdeator:
    """Autonomous ideation triggered by brain heartbeat.

    This bridges the gap between LMD (memory/ideation) and the brain's
    heartbeat clock. Each heartbeat pulse can trigger ideation checks.

    Usage:
        >>> ideator = HeartbeatIdeator(config, dynamics)
        >>> # In your brain's forward loop:
        >>> if metrics['heartbeat_signal'] == 1:
        >>>     ideator.on_heartbeat(memories, dopamine=metrics.get('dopamine', 16384))
    """

    def __init__(
        self,
        config: LMDConfig,
        dynamics: LMDDynamics,
        heartbeats_per_check: int = 100,  # Check triggers every N heartbeats (~1.5s at 66Hz)
        min_memories_for_ideation: int = 5,
        ideas_per_session: int = 3,
        consolidate_threshold: float = 0.6,  # Quality threshold to consolidate
        on_idea_generated: Optional[Callable[[IdeationResult], None]] = None
    ):
        self.config = config
        self.dynamics = dynamics
        self.heartbeats_per_check = heartbeats_per_check
        self.min_memories_for_ideation = min_memories_for_ideation
        self.ideas_per_session = ideas_per_session
        self.consolidate_threshold = consolidate_threshold
        self.on_idea_generated = on_idea_generated

        # Create ideation engine
        self.ideator = AutonomousIdeator(config, dynamics)

        # Override autonomy controller settings for heartbeat-driven mode
        self.ideator.autonomy.min_interval_seconds = 0.0  # Heartbeat provides timing

        # State
        self.heartbeat_count = 0
        self.last_input_time = time.time()
        self.last_dopamine = 16384  # Neutral
        self.metrics = HeartbeatIdeationMetrics()

        # Consolidated ideas waiting to become memories
        self.pending_consolidation: List[IdeationResult] = []

        # Thread safety
        self._lock = threading.RLock()
        self._running = False

    def on_heartbeat(
        self,
        memories: List[LivingMemory],
        dopamine: int = 16384,
        external_input: bool = False
    ) -> Optional[IdeationResult]:
        """Called on each heartbeat pulse.

        Args:
            memories: Current living memories
            dopamine: Current dopamine level (0-32767)
            external_input: Whether external input just arrived (resets idle)

        Returns:
            IdeationResult if ideation was triggered, None otherwise
        """
        with self._lock:
            self.heartbeat_count += 1
            self.metrics.total_heartbeats += 1
            self.last_dopamine = dopamine

            # Reset idle time on external input
            if external_input:
                self.last_input_time = time.time()

            # Calculate idle time
            idle_time = time.time() - self.last_input_time
            self.metrics.idle_seconds = idle_time

            # Only check triggers every N heartbeats (efficiency)
            if self.heartbeat_count % self.heartbeats_per_check != 0:
                return None

            # Update autonomy triggers based on current state
            self._update_triggers(memories, dopamine, idle_time)

            # Check if we should ideate
            if not self.ideator.autonomy.can_ideate():
                return None

            trigger = self.ideator.autonomy.check_triggers()
            if trigger is None:
                return None

            # Not enough memories to work with
            if len(memories) < self.min_memories_for_ideation:
                return None

            # Run ideation!
            return self._run_ideation(memories, trigger)

    def _update_triggers(
        self,
        memories: List[LivingMemory],
        dopamine: int,
        idle_time: float
    ) -> None:
        """Update autonomy triggers based on current brain state."""
        controller = self.ideator.autonomy

        # Idle trigger: fire when brain is idle
        controller.update_trigger(AutonomyTrigger.IDLE, idle_time)

        # Curiosity trigger: based on memory diversity
        if memories:
            contents = torch.stack([m.content for m in memories])
            variance = contents.var().item()
            # Low variance = low diversity = bored = curious
            curiosity = 1.0 / (variance + 0.1)
            controller.update_trigger(AutonomyTrigger.CURIOSITY, curiosity)

        # Boredom trigger: based on recent memory novelty
        recent_novelty = sum(
            m.energy for m in memories
            if time.time() - m.created_at < 60
        ) / max(len(memories), 1)
        # Low recent novelty = bored = need new ideas
        controller.update_trigger(AutonomyTrigger.BOREDOM, 1.0 - recent_novelty)

        # Dopamine modulation: high dopamine = more likely to ideate
        dopamine_norm = dopamine / 32767.0
        if dopamine_norm > 0.7:
            # Excited state - lower thresholds temporarily
            controller.triggers[AutonomyTrigger.IDLE].threshold = 30.0  # Faster ideation
        else:
            controller.triggers[AutonomyTrigger.IDLE].threshold = 60.0  # Normal

    def _run_ideation(
        self,
        memories: List[LivingMemory],
        trigger: AutonomyTrigger
    ) -> IdeationResult:
        """Run an ideation session."""
        trigger_name = trigger.name

        # Run single round of ideation
        results = self.ideator.run_autonomous_session(
            memories,
            n_rounds=1,
            ideas_per_round=self.ideas_per_session
        )

        if not results:
            return None

        result = results[0]

        # Record metrics
        self.metrics.record_session(result, trigger_name)

        # Record in autonomy controller
        self.ideator.autonomy.record_ideation(
            n_ideas=len(result.ideas),
            compute_ops=result.total_ideas_generated * 100  # Estimate
        )

        # Check for consolidation candidates
        if result.best_score >= self.consolidate_threshold:
            self.pending_consolidation.append(result)

        # Callback
        if self.on_idea_generated:
            self.on_idea_generated(result)

        return result

    def consolidate_pending(self) -> List[LivingMemory]:
        """Consolidate pending high-quality ideas into living memories.

        Call this periodically (e.g., during sleep/idle) to turn
        good ideas into actual memories.
        """
        with self._lock:
            if not self.pending_consolidation:
                return []

            all_new_memories = []

            for result in self.pending_consolidation:
                # Only consolidate ideas above threshold
                good_ideas = [
                    idea for idea, score in zip(result.ideas, result.scores)
                    if score.get("total", 0) >= self.consolidate_threshold
                ]

                if good_ideas:
                    new_memories = self.ideator.consolidate_to_memories(
                        good_ideas,
                        self.dynamics
                    )
                    all_new_memories.extend(new_memories)

            self.metrics.total_ideas_consolidated += len(all_new_memories)
            self.pending_consolidation.clear()

            return all_new_memories

    def get_metrics(self) -> HeartbeatIdeationMetrics:
        """Get current metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics (e.g., at start of new session)."""
        with self._lock:
            self.metrics = HeartbeatIdeationMetrics()
            self.heartbeat_count = 0


def run_long_running_demo(
    duration_seconds: float = 30.0,
    heartbeat_hz: float = 66.0,
    verbose: bool = True
) -> HeartbeatIdeationMetrics:
    """Run a long-running demo of heartbeat-triggered ideation.

    This simulates a brain running with heartbeat-driven autonomous thinking.

    Args:
        duration_seconds: How long to run
        heartbeat_hz: Simulated heartbeat frequency
        verbose: Print progress

    Returns:
        Final metrics
    """
    import time

    # Setup
    config = LMDConfig.toy_scale()
    dynamics = LMDDynamics(config)

    # Create initial memories
    memories = []
    for i in range(20):
        memories.append(LivingMemory(
            id=i,
            content=torch.randn(config.content_dim),
            valence=ValenceTrajectory.random(),
            energy=0.8 + 0.2 * torch.rand(1).item(),
            created_at=time.time() - i * 10,  # Spread out creation times
            label=f"experience_{i}"
        ))

    # Track generated ideas
    all_results = []

    def on_idea(result: IdeationResult):
        all_results.append(result)
        if verbose:
            print(f"\n[IDEA] Score: {result.best_score:.3f}, "
                  f"Novelty: {result.average_novelty:.3f}")

    # Create heartbeat ideator
    ideator = HeartbeatIdeator(
        config,
        dynamics,
        heartbeats_per_check=50,  # Check every ~0.75s at 66Hz
        ideas_per_session=3,
        consolidate_threshold=0.5,
        on_idea_generated=on_idea
    )

    # Make triggers more aggressive for demo
    for trigger in ideator.ideator.autonomy.triggers.values():
        trigger.cooldown_seconds = 5.0  # 5 second cooldown instead of 120+
        trigger.threshold *= 0.5  # Lower threshold to trigger more often

    # Simulate heartbeat loop
    heartbeat_interval = 1.0 / heartbeat_hz
    start_time = time.time()
    last_print = start_time

    # Simulate dopamine fluctuations
    dopamine = 16384  # Start neutral
    dopamine_direction = 1

    if verbose:
        print(f"Starting {duration_seconds}s simulation at {heartbeat_hz}Hz...")
        print(f"Initial memories: {len(memories)}")
        print("-" * 60)

    while time.time() - start_time < duration_seconds:
        # Simulate heartbeat pulse
        result = ideator.on_heartbeat(
            memories,
            dopamine=dopamine,
            external_input=False
        )

        # Simulate dopamine drift
        dopamine += dopamine_direction * 100
        if dopamine > 28000:
            dopamine_direction = -1
        elif dopamine < 8000:
            dopamine_direction = 1

        # Consolidate new memories periodically
        if ideator.heartbeat_count % 500 == 0:
            new_mems = ideator.consolidate_pending()
            if new_mems:
                memories.extend(new_mems)
                if verbose:
                    print(f"  [CONSOLIDATE] {len(new_mems)} new memories! "
                          f"Total: {len(memories)}")

        # Run memory dynamics occasionally
        if ideator.heartbeat_count % 100 == 0:
            dynamics.step(memories)

        # Progress update
        if verbose and time.time() - last_print > 5.0:
            elapsed = time.time() - start_time
            metrics = ideator.get_metrics()
            print(f"\n[{elapsed:.1f}s] Heartbeats: {metrics.total_heartbeats}, "
                  f"Sessions: {metrics.total_ideation_sessions}, "
                  f"Ideas: {metrics.total_ideas_generated}, "
                  f"Dopamine: {dopamine}")
            last_print = time.time()

        # Sleep to simulate real-time
        time.sleep(heartbeat_interval)

    # Final consolidation
    final_new = ideator.consolidate_pending()
    if final_new:
        memories.extend(final_new)

    # Final metrics
    metrics = ideator.get_metrics()

    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(metrics.summary())
        print(f"\nFinal memory count: {len(memories)}")
        print(f"Total ideas across sessions: {sum(r.total_ideas_generated for r in all_results)}")

        if all_results:
            avg_quality = sum(r.best_score for r in all_results) / len(all_results)
            avg_novelty = sum(r.average_novelty for r in all_results) / len(all_results)
            print(f"Average best score: {avg_quality:.3f}")
            print(f"Average novelty: {avg_novelty:.3f}")

    return metrics


if __name__ == "__main__":
    run_long_running_demo(duration_seconds=30.0, verbose=True)
