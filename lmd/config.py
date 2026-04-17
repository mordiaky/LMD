"""Configuration for Living Memory Dynamics.

Invented by Joshua R. Thomas, January 2026.
"""

from dataclasses import dataclass


@dataclass
class LMDConfig:
    """Configuration for the LMD system.

    Parameters control the dynamics of living memories:
    - Narrative flow speed
    - Coupling strength
    - Metabolic rates
    - Creative noise level
    """

    # === Memory Dimensions ===
    content_dim: int = 32           # Dimension of content embeddings
    valence_points: int = 5         # Points in valence trajectory
    max_memories: int = 100         # Maximum memories in system

    # === Narrative Dynamics ===
    narrative_velocity: float = 0.1     # Base phase advancement per step
    narrative_potential_strength: float = 0.5  # Strength of story attractors

    # === Coupling Parameters ===
    coupling_content_weight: float = 0.4    # Weight for content similarity
    coupling_valence_weight: float = 0.4    # Weight for valence compatibility
    coupling_phase_weight: float = 0.2      # Weight for phase alignment
    coupling_threshold: float = 0.3         # Minimum coupling to consider

    # === Metabolism Parameters ===
    energy_decay_rate: float = 0.995        # Natural energy decay per step
    activation_boost: float = 0.3           # Energy boost on activation
    energy_transfer_rate: float = 0.1       # Rate of energy flow between coupled memories
    min_energy: float = 0.01                # Minimum energy (below = dead)
    max_energy: float = 2.0                 # Maximum energy cap

    # === Resonance Parameters ===
    resonance_strength: float = 0.2         # Strength of valence resonance
    phase_sync_rate: float = 0.05           # Rate of phase synchronization

    # === Creative Noise ===
    noise_scale: float = 0.01               # Scale of creative noise
    noise_energy_threshold: float = 0.5     # Only add noise if energy > threshold

    # === Narrative Generation ===
    movie_activation_threshold: float = 0.3  # Min energy to include in movie
    movie_phase_tolerance: float = 0.5       # Phase difference tolerance for sequencing

    # === Death/Birth ===
    death_threshold: float = 0.05           # Energy below this = memory dies
    ghost_threshold: float = 0.1            # Energy below this = ghost state

    # === Sustenance Parameters (NEW) ===
    mutual_sustenance_rate: float = 0.05   # Coupled memories share energy
    spontaneous_activation_prob: float = 0.02  # Random reactivation chance
    valence_activation_bonus: float = 0.1  # High-valence memories reactivate more

    # === Toy Scale Overrides ===
    @classmethod
    def toy_scale(cls) -> "LMDConfig":
        """Configuration for toy-scale experiments."""
        return cls(
            content_dim=32,
            valence_points=5,
            max_memories=50,
            energy_decay_rate=0.997,  # Balanced decay
            narrative_velocity=0.15,  # Faster progression
            noise_scale=0.02,  # More noise for variety
            mutual_sustenance_rate=0.02,  # TUNED: lower sustenance
            spontaneous_activation_prob=0.01,  # TUNED: less reactivation
        )

    @classmethod
    def production_scale(cls) -> "LMDConfig":
        """Configuration for production use."""
        return cls(
            content_dim=256,
            valence_points=20,
            max_memories=10000,
            energy_decay_rate=0.999,  # Slower decay
            narrative_velocity=0.05,   # Slower progression
            noise_scale=0.005,  # Less noise
        )
