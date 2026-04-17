#!/usr/bin/env python3
"""Basic usage example for Living Memory Dynamics (LMD).

This example demonstrates:
1. Creating living memories
2. Setting up the dynamics engine
3. Evolving memories over time
4. Observing emergent behavior
"""

import torch
from lmd import (
    LivingMemory,
    ValenceTrajectory,
    NarrativePhase,
    LMDDynamics,
    LMDConfig,
)


def main():
    # Configuration
    content_dim = 256
    n_memories = 20
    n_steps = 100

    print("Living Memory Dynamics - Basic Usage Example")
    print("=" * 50)

    # 1. Create living memories
    print("\n1. Creating living memories...")
    memories = []
    for i in range(n_memories):
        # Random embedding
        embedding = torch.randn(content_dim)
        embedding = embedding / embedding.norm()  # Normalize

        # Create valence trajectory (emotional arc)
        onset = torch.rand(1).item() * 0.5  # Start emotion
        peak = 0.5 + torch.rand(1).item() * 0.5  # Peak emotion
        resolution = onset + torch.rand(1).item() * (peak - onset)  # End emotion

        memory = LivingMemory(
            id=i,
            content=embedding,
            energy=0.5 + torch.rand(1).item(),  # Random energy 0.5-1.5
            valence=ValenceTrajectory(
                points=torch.tensor([onset, peak, resolution])
            ),
            phase=0.0,  # Starts at phase 0 (SETUP). Phase is a float radian value.
            label=f"memory_{i:03d}",
        )
        memories.append(memory)

    print(f"   Created {len(memories)} living memories")

    # 2. Setup dynamics engine
    print("\n2. Setting up dynamics engine...")
    config = LMDConfig(
        content_dim=content_dim,
        coupling_content_weight=0.5,
        narrative_potential_strength=0.4,
        noise_scale=0.01,
    )
    dynamics = LMDDynamics(config)
    print(f"   Config: coupling_content_weight={config.coupling_content_weight}, noise={config.noise_scale}")

    # 3. Evolve memories
    print("\n3. Evolving memories...")
    dt = 0.01

    for step in range(n_steps):
        # Step the dynamics
        dynamics.step(memories, dt=dt)

        # Log every 25 steps
        if (step + 1) % 25 == 0:
            # Calculate statistics
            energies = [m.energy for m in memories]
            avg_energy = sum(energies) / len(energies)

            phases = {}
            for m in memories:
                phase_name = m.narrative_phase.name
                phases[phase_name] = phases.get(phase_name, 0) + 1

            metabolic_states = {}
            for m in memories:
                state = m.metabolic_state.value
                metabolic_states[state] = metabolic_states.get(state, 0) + 1

            print(f"   Step {step + 1}/{n_steps}:")
            print(f"      Avg Energy: {avg_energy:.3f}")
            print(f"      Phases: {phases}")
            print(f"      Metabolic: {metabolic_states}")

    # 4. Final state
    print("\n4. Final memory states:")
    for m in memories[:5]:  # Show first 5
        print(f"   {m.id}: energy={m.energy:.2f}, phase={m.narrative_phase.name}, "
              f"state={m.metabolic_state.value}")
    print(f"   ... and {len(memories) - 5} more")

    # 5. Find coupled memories
    print("\n5. Finding strongly coupled memories...")
    coupling_field = dynamics.coupling
    for i, m1 in enumerate(memories[:5]):
        for j, m2 in enumerate(memories[i+1:i+6], start=i+1):
            coupling = coupling_field.get_coupling(m1, m2)
            if abs(coupling) > 0.1:
                print(f"   {m1.id} <-> {m2.id}: coupling={coupling:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
