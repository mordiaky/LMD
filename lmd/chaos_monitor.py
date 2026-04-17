"""Chaos Monitor - Track stability and edge-of-chaos dynamics in LMD.

Monitors:
1. Lyapunov Exponents - measure sensitivity to initial conditions
2. Phase Space Stability - track divergence of trajectories
3. Energy Stability - detect runaway growth or collapse
4. Creative Noise Effects - measure impact of stochastic term

Edge of chaos target: lambda ~ 0 (not too stable, not too chaotic)

Invented by Joshua R. Thomas, January 2026.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import LMDConfig
from .dynamics import LMDDynamics
from .living_memory import LivingMemory


@dataclass
class ChaosMetrics:
    """Metrics for chaos/stability analysis."""

    # Lyapunov exponent (λ)
    # λ < 0: stable/convergent
    # λ ~ 0: edge of chaos (ideal)
    # λ > 0: chaotic/divergent
    lyapunov_exponent: float = 0.0

    # Phase stability
    phase_variance_rate: float = 0.0  # How fast phase variance grows
    phase_coherence_trend: float = 0.0  # Trend in coherence over time

    # Energy stability
    energy_growth_rate: float = 0.0  # Positive = growing, negative = decaying
    energy_variance_stability: float = 0.0  # How stable is variance

    # Trajectory divergence
    trajectory_divergence: float = 0.0  # How much do similar ICs diverge

    # Noise sensitivity
    noise_amplification: float = 0.0  # How much noise gets amplified

    @property
    def stability_score(self) -> float:
        """Overall stability score (0-1, higher = more stable but not dead)."""
        # Ideal: small positive or near-zero Lyapunov
        lyap_score = 1.0 / (1.0 + abs(self.lyapunov_exponent))

        # Ideal: low phase variance rate
        phase_score = 1.0 / (1.0 + abs(self.phase_variance_rate))

        # Ideal: near-zero energy growth
        energy_score = 1.0 / (1.0 + abs(self.energy_growth_rate))

        return (lyap_score + phase_score + energy_score) / 3

    @property
    def edge_of_chaos_score(self) -> float:
        """How close to edge of chaos (λ ~ 0). Higher = better."""
        # We want λ near 0 but slightly positive
        ideal_lambda = 0.01
        distance = abs(self.lyapunov_exponent - ideal_lambda)
        return 1.0 / (1.0 + 10 * distance)

    @property
    def is_stable(self) -> bool:
        """Check if system is stable (not exploding).

        Note: Negative energy_growth_rate is expected (decay).
        Positive Lyapunov is okay if small (edge of chaos).
        """
        return (
            self.lyapunov_exponent < 0.2 and  # Edge of chaos acceptable
            abs(self.energy_growth_rate) < 0.25 and  # Bounded energy change
            self.trajectory_divergence < 1500.0  # Allow more divergence
        )

    @property
    def is_alive(self) -> bool:
        """Check if system is 'alive' (not dead/frozen)."""
        return (
            self.lyapunov_exponent > -0.5 and
            self.phase_variance_rate > 0.001 and
            self.noise_amplification > 0.1
        )

    def summary(self) -> str:
        """Get summary string."""
        status = "EDGE OF CHAOS" if (self.is_stable and self.is_alive) else (
            "CHAOTIC" if not self.is_stable else "FROZEN"
        )
        return (
            f"Chaos Analysis: {status}\n"
            f"  Lyapunov: {self.lyapunov_exponent:.4f}\n"
            f"  Phase variance rate: {self.phase_variance_rate:.4f}\n"
            f"  Energy growth rate: {self.energy_growth_rate:.4f}\n"
            f"  Trajectory divergence: {self.trajectory_divergence:.4f}\n"
            f"  Edge-of-chaos score: {self.edge_of_chaos_score:.3f}\n"
            f"  Stability score: {self.stability_score:.3f}"
        )


class ChaosMonitor:
    """Monitors chaos and stability in LMD systems.

    Implements:
    - Lyapunov exponent estimation via trajectory separation
    - Phase space stability analysis
    - Long-run stability testing
    """

    def __init__(self, config: LMDConfig):
        self.config = config

        # History tracking
        self.phase_history: List[List[float]] = []
        self.energy_history: List[List[float]] = []
        self.coherence_history: List[float] = []

    def estimate_lyapunov(
        self,
        dynamics: LMDDynamics,
        memories: List[LivingMemory],
        n_steps: int = 100,
        perturbation: float = 1e-4
    ) -> float:
        """Estimate largest Lyapunov exponent.

        Method: Create perturbed copy, evolve both, measure separation growth.

        λ = lim (1/t) * ln(d(t)/d(0))

        Args:
            dynamics: The LMD dynamics engine
            memories: Initial memories
            n_steps: Number of steps to evolve
            perturbation: Initial perturbation size

        Returns:
            Estimated Lyapunov exponent
        """
        if len(memories) < 2:
            return 0.0

        # Create deep copy with small perturbation
        memories_pert = self._deep_copy_with_perturbation(memories, perturbation)

        # Create separate dynamics for perturbed system
        dynamics_pert = LMDDynamics(self.config)

        # Track separation over time
        separations = []
        d0 = self._compute_separation(memories, memories_pert)
        separations.append(d0)

        for _step in range(n_steps):
            # Evolve both systems
            dynamics.step(memories)
            dynamics_pert.step(memories_pert)

            # Measure separation
            d_t = self._compute_separation(memories, memories_pert)
            separations.append(d_t)

            # Renormalize to prevent overflow
            if d_t > 1.0:
                scale = perturbation / d_t
                for m, m_p in zip(memories, memories_pert):
                    m_p.phase = m.phase + (m_p.phase - m.phase) * scale
                    m_p.energy = m.energy + (m_p.energy - m.energy) * scale

        # Compute Lyapunov from average logarithmic growth
        if d0 > 0 and len(separations) > 1:
            # Sum of log ratios
            log_growth = 0.0
            count = 0
            for i in range(1, len(separations)):
                if separations[i] > 1e-10 and separations[i-1] > 1e-10:
                    log_growth += math.log(separations[i] / separations[i-1])
                    count += 1

            if count > 0:
                lyapunov = log_growth / count
                return lyapunov

        return 0.0

    def _deep_copy_with_perturbation(
        self,
        memories: List[LivingMemory],
        perturbation: float
    ) -> List[LivingMemory]:
        """Create perturbed copy of memories."""
        perturbed = []
        for m in memories:
            m_copy = LivingMemory(
                id=m.id,
                content=m.content.clone(),
                valence=copy.deepcopy(m.valence),
                phase=m.phase + torch.randn(1).item() * perturbation,
                energy=m.energy + torch.randn(1).item() * perturbation * 0.1,
                created_at=m.created_at,
                label=m.label
            )
            perturbed.append(m_copy)
        return perturbed

    def _compute_separation(
        self,
        memories1: List[LivingMemory],
        memories2: List[LivingMemory]
    ) -> float:
        """Compute phase space separation between two memory sets."""
        if len(memories1) != len(memories2):
            return float('inf')

        total_sep = 0.0
        for m1, m2 in zip(memories1, memories2):
            # Phase separation (circular)
            phase_diff = abs(m1.phase - m2.phase)
            if phase_diff > math.pi:
                phase_diff = 2 * math.pi - phase_diff

            # Energy separation
            energy_diff = abs(m1.energy - m2.energy)

            total_sep += phase_diff ** 2 + energy_diff ** 2

        return math.sqrt(total_sep / len(memories1))

    def analyze_long_run(
        self,
        dynamics: LMDDynamics,
        memories: List[LivingMemory],
        n_steps: int = 1000,
        sample_interval: int = 10
    ) -> ChaosMetrics:
        """Run long simulation and analyze chaos metrics.

        Args:
            dynamics: LMD dynamics engine
            memories: Initial memories
            n_steps: Number of steps to run
            sample_interval: How often to sample state

        Returns:
            ChaosMetrics with full analysis
        """
        self.phase_history = []
        self.energy_history = []
        self.coherence_history = []

        # Run simulation with sampling
        for step in range(n_steps):
            metrics = dynamics.step(memories)

            if step % sample_interval == 0:
                self.phase_history.append([m.phase for m in memories])
                self.energy_history.append([m.energy for m in memories])
                self.coherence_history.append(metrics.get('phase_coherence', 0))

        # Compute chaos metrics
        return self._compute_chaos_metrics(dynamics, memories)

    def _compute_chaos_metrics(
        self,
        dynamics: LMDDynamics,
        memories: List[LivingMemory]
    ) -> ChaosMetrics:
        """Compute all chaos metrics from history."""
        metrics = ChaosMetrics()

        if len(self.phase_history) < 3:
            return metrics

        # Lyapunov exponent (estimated from dynamics)
        metrics.lyapunov_exponent = self.estimate_lyapunov(
            dynamics, memories, n_steps=50, perturbation=1e-4
        )

        # Phase variance rate
        phase_variances = []
        for phases in self.phase_history:
            if phases:
                mean_phase = sum(phases) / len(phases)
                var = sum((p - mean_phase) ** 2 for p in phases) / len(phases)
                phase_variances.append(var)

        if len(phase_variances) > 1:
            # Linear fit for growth rate
            n = len(phase_variances)
            x_mean = (n - 1) / 2
            y_mean = sum(phase_variances) / n

            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(phase_variances))
            den = sum((i - x_mean) ** 2 for i in range(n))

            metrics.phase_variance_rate = num / den if den > 0 else 0

        # Phase coherence trend
        if len(self.coherence_history) > 1:
            n = len(self.coherence_history)
            x_mean = (n - 1) / 2
            y_mean = sum(self.coherence_history) / n

            num = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(self.coherence_history))
            den = sum((i - x_mean) ** 2 for i in range(n))

            metrics.phase_coherence_trend = num / den if den > 0 else 0

        # Energy growth rate
        total_energies = [sum(e) for e in self.energy_history]
        if len(total_energies) > 1:
            n = len(total_energies)
            x_mean = (n - 1) / 2
            y_mean = sum(total_energies) / n

            num = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(total_energies))
            den = sum((i - x_mean) ** 2 for i in range(n))

            metrics.energy_growth_rate = num / den if den > 0 else 0

        # Energy variance stability
        energy_vars = []
        for energies in self.energy_history:
            if energies:
                mean_e = sum(energies) / len(energies)
                var = sum((e - mean_e) ** 2 for e in energies) / len(energies)
                energy_vars.append(var)

        if energy_vars:
            mean_var = sum(energy_vars) / len(energy_vars)
            var_of_var = sum((v - mean_var) ** 2 for v in energy_vars) / len(energy_vars)
            metrics.energy_variance_stability = 1.0 / (1.0 + var_of_var)

        # Trajectory divergence (use Lyapunov-based estimate)
        metrics.trajectory_divergence = math.exp(metrics.lyapunov_exponent * len(self.phase_history))
        metrics.trajectory_divergence = min(metrics.trajectory_divergence, 1000)  # Cap

        # Noise amplification
        if len(self.phase_history) > 10:
            # Compare early vs late variance
            early_var = sum(
                sum((p - sum(phases)/len(phases))**2 for p in phases)/len(phases)
                for phases in self.phase_history[:5]
            ) / 5
            late_var = sum(
                sum((p - sum(phases)/len(phases))**2 for p in phases)/len(phases)
                for phases in self.phase_history[-5:]
            ) / 5

            if early_var > 0:
                metrics.noise_amplification = late_var / early_var
            else:
                metrics.noise_amplification = late_var

        return metrics

    def test_noise_sensitivity(
        self,
        dynamics: LMDDynamics,
        memories: List[LivingMemory],
        noise_levels: List[float] = None,
        n_steps: int = 100
    ) -> Dict[float, float]:
        """Test sensitivity to different noise levels.

        Args:
            dynamics: LMD dynamics engine
            memories: Initial memories
            noise_levels: List of noise scales to test
            n_steps: Steps per test

        Returns:
            Dict mapping noise_level -> final_divergence
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]

        results = {}

        # Save original noise setting
        original_noise = self.config.noise_scale

        for noise in noise_levels:
            # Reset memories
            memories_test = self._deep_copy_with_perturbation(memories, 0)

            # Set noise level
            self.config.noise_scale = noise

            # Create fresh dynamics
            test_dynamics = LMDDynamics(self.config)

            # Run simulation
            initial_phases = [m.phase for m in memories_test]
            for _ in range(n_steps):
                test_dynamics.step(memories_test)

            # Measure divergence from initial
            final_phases = [m.phase for m in memories_test]
            divergence = sum(
                abs(f - i) for f, i in zip(final_phases, initial_phases)
            ) / len(memories_test)

            results[noise] = divergence

        # Restore original
        self.config.noise_scale = original_noise

        return results


def run_chaos_analysis(
    dynamics: LMDDynamics,
    memories: List[LivingMemory],
    n_steps: int = 1000
) -> ChaosMetrics:
    """Convenience function to run full chaos analysis."""
    monitor = ChaosMonitor(dynamics.config)
    return monitor.analyze_long_run(dynamics, memories, n_steps)
