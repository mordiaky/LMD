# Living Memory Dynamics (LMD)

> **A Novel Framework for Narrative-Generating Episodic Memory with Creative Leaps and Language Grounding**

[![PyPI](https://img.shields.io/badge/PyPI-living--memory--dynamics-blue.svg)](https://pypi.org/project/living-memory-dynamics/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Custom-orange.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.3.0-green.svg)](https://pypi.org/project/living-memory-dynamics/)
[![CI](https://github.com/mordiaky/LMD/actions/workflows/ci.yml/badge.svg)](https://github.com/mordiaky/LMD/actions/workflows/ci.yml)

## What Makes LMD Different?

Traditional memory systems store static embeddings. **LMD treats memories as living entities** that:

- **Breathe**: Memories have metabolic energy and pass through states (vivid, active, dormant, fading, ghost)
- **Feel**: Emotional trajectories, not single valence tags
- **Tell Stories**: Narrative phases (`SETUP → RISING → CLIMAX → RESOLUTION → INTEGRATION`)
- **Resonate**: Memories couple and influence each other through resonance fields
- **Create**: Generate novel ideas through internal creative leaps

## The Joshua R. Thomas Memory Equation

```
dM/dt = ∇φ(N) + Σⱼ Γᵢⱼ R(vᵢ, vⱼ) + A(M, ξ) + κη(t)
```

Where:
- `∇φ(N)` = Narrative potential (story attractor landscape)
- `R(vᵢ, vⱼ)` = Resonance function (emotional coupling between memories)
- `A(M, ξ)` = Activation function (contextual triggering)
- `κη(t)` = Creative noise (generative stochasticity)

## Creative Leaps (v1.1.0)

LMD doesn't just store — it **invents**. Four internal operators enable human-like creative jumps:

| Operator | What It Does | Example |
|----------|-------------|---------|
| **Analogical Transfer** | Transplants patterns between distant domains | "dragon fire" + "glass refraction" → "prismatic breath weapon" |
| **Manifold Walker** | Diffuses through embedding space | Gradual concept morphing |
| **Orthogonal Composer** | Gram-Schmidt perpendicular merges | Combines concepts along independent axes |
| **Void Extrapolator** | Ray-traces into unexplored territory | Discovers genuinely novel concepts |

```python
import torch
from lmd import CreativeLeapEngine, CreativeLeapConfig

# Build engine. batch_leap works on embedding tensors directly —
# extract them from your memories if that's where they live.
engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=256))

embeddings = [torch.randn(256) for _ in range(10)]
embeddings = [e / e.norm() for e in embeddings]

leaps = engine.batch_leap(embeddings, n_leaps=5, dopamine=0.8)
for leap in leaps:
    print(f"{leap.leap_type.name}: novelty={leap.novelty:.2f}")
```

## Language Grounding (v1.3.0)

**The Problem**: LMD creates ideas as vectors — you can't read what they mean.

**The Solution**: Language Grounding bridges vectors and human-readable text.

```
Text Input --> Embedding --> Creative Leap --> New Embedding --> Text Output
"dragon"   --> [0.2, 0.8..] --> ORTHOGONAL --> [0.5, 0.3..] --> "prismatic creature"
```

### Does LMD work without Language Grounding?

**Yes!** Language Grounding is **100% optional**. Core LMD works fine without it:
- Core LMD: `pip install living-memory-dynamics` (vectors only)
- With Language: `pip install living-memory-dynamics[language]` (text in/out)

### Quick Example

```python
from lmd import create_grounding, CreativeLeapEngine, CreativeLeapConfig

# Create grounding (downloads MiniLM model ~80MB first time)
grounding = create_grounding(encoder="minilm")

# Encode text to embeddings
dragon = grounding.encode("fire-breathing dragon")
glass = grounding.encode("crystalline glass structure")

# Build a corpus for decoding
grounding.add_to_corpus("fire-breathing dragon")
grounding.add_to_corpus("crystalline glass structure")
grounding.add_to_corpus("stained glass window")
grounding.add_to_corpus("dragon scales armor")

# Blend two concepts
blended = (dragon + glass) / 2
blended = blended / blended.norm()

# Decode back to text
result = grounding.decode(blended, top_k=3)
print(result.interpolated_description)
# e.g. "blend of 'crystalline glass structure' (50%) and 'fire-breathing dragon' (50%)"
```

### Generate New Ideas with Text Output

```python
import torch

engine = CreativeLeapEngine(CreativeLeapConfig(content_dim=grounding.embedding_dim))

sources = [
    grounding.encode("volcanic eruption").cpu(),
    grounding.encode("frozen ice sculpture").cpu(),
    grounding.encode("rainbow spectrum").cpu(),
]

leap = engine.leap(sources, dopamine=0.7)

description = grounding.describe_leap(
    leap_type=leap.leap_type.name,
    sources=sources,
    result=leap.embedding,
)
print(description.synthesized_description)
# e.g. "Extrapolated beyond 'volcanic eruption' + 'frozen ice sculpture' + ..."
print(f"Novelty: {description.novelty_score}")
```

### Ground Living Memories to Readable Text

```python
import torch
from lmd import LivingMemory, ValenceTrajectory

embedding = grounding.encode("ancient dragon guarding treasure")
memory = LivingMemory(
    id=0,
    content=embedding,
    valence=ValenceTrajectory.climax(low=0.5, peak=0.7),
    energy=0.8,
    phase=0.0,  # SETUP phase (0 radians)
    label="dragon_memory",
)

grounded = grounding.ground_memory(memory)
print(grounded.text)
# e.g. "blend of 'ancient dragon guarding treasure' (64%) and ..."
```

### Run the Full Demo

```bash
# Install with language support
pip install living-memory-dynamics[language]

# Run the demo
python examples/language_grounding.py
```

## Installation

**LMD is available on PyPI!**

```bash
# Core only (vectors, no text)
pip install living-memory-dynamics

# With language grounding (text in/out)
pip install living-memory-dynamics[language]

# With GPU acceleration (Triton CUDA kernels)
pip install living-memory-dynamics[cuda]

# Everything (language + cuda + dev tools)
pip install living-memory-dynamics[all]
```

**PyPI Package:** https://pypi.org/project/living-memory-dynamics/

**Or install from source:**

```bash
git clone https://github.com/mordiaky/LMD.git
cd LMD
pip install -e ".[language]"  # or [all] for everything
```

## Quick Start

### 1. Create Living Memories

```python
import torch
from lmd import LivingMemory, ValenceTrajectory

memory = LivingMemory(
    id=1,                                # Integer identifier
    content=torch.randn(256),            # Embedding vector
    valence=ValenceTrajectory.climax(low=0.3, peak=0.9),
    energy=1.0,                          # Metabolic energy (0 - 2.0)
    phase=0.0,                           # Narrative phase in radians [0, 2π)
    label="my_first_memory",
)
```

**Valence trajectory factories:**

| Factory | Shape | Use |
|---|---|---|
| `ValenceTrajectory.constant(v)` | flat | static emotion |
| `ValenceTrajectory.redemption(start, end)` | negative → positive | learning from failure |
| `ValenceTrajectory.tragedy(start, end)` | positive → negative | loss, grief |
| `ValenceTrajectory.climax(low, peak)` | builds then resolves | story arc |
| `ValenceTrajectory.random()` | noise | test fixtures |

### 2. Let Memories Evolve

```python
from lmd import LMDDynamics, LMDConfig

config = LMDConfig(content_dim=256)
dynamics = LMDDynamics(config)

for _ in range(100):
    dynamics.step(memories, dt=0.01)
    # Memories naturally evolve, couple, and generate narratives
```

### 3. Generate Creative Ideas

```python
from lmd import CreativeIdeationEngine, CreativeIdeationConfig

config = CreativeIdeationConfig(content_dim=256)
engine = CreativeIdeationEngine(config)

# Ideate with dopamine modulation
result = engine.ideate(memories, dopamine=0.7, n_ideas=10)

for idea in result.ideas[:5]:
    print(f"Form: {idea.form}, Novelty: {idea.novelty:.2f}, Score: {idea.total_score:.2f}")
```

### 4. Hierarchical Ideas with Grafting

```python
import torch
from lmd import HierarchicalIdeaFactory, IdeaGrafter

factory = HierarchicalIdeaFactory(content_dim=256)
grafter = IdeaGrafter(content_dim=256)

# Create tree-structured ideas (depth controls number of child components)
dragon = factory.from_embedding(torch.randn(256), depth=3, label="dragon")
crystal = factory.from_embedding(torch.randn(256), depth=3, label="crystal")

# Swap a component from crystal into dragon. Pick any mutable component
# from crystal as the donor; target_id is optional (random if omitted).
donor = list(crystal.components.values())[1]
target_id = list(dragon.components.keys())[1]

result = grafter.swap_component(dragon, donor, target_id=target_id)
print(f"novelty={result.novelty:.2f}, coherence={result.coherence:.2f}")
# Result: dragon with one component replaced by a crystal component
```

## Architecture

```
lmd/
├── living_memory.py           # LivingMemory datastructure, ValenceTrajectory, NarrativePhase, MetabolicState
├── config.py                  # LMDConfig (toy_scale / production_scale presets)
├── dynamics.py                # LMDDynamics engine (ties everything together)
├── coupling.py                # Memory resonance fields
├── metabolism.py              # Energy dynamics (decay, activation, sustenance)
├── narrative.py               # Story generation
├── imagination.py             # Mental canvas & transform operations
├── plausibility.py            # Reality grounding / coherence constraints
├── creative_leaps.py          # 4 creative operators (analogical, manifold, orthogonal, void)
├── hierarchical_ideas.py      # Tree-structured ideas + IdeaGrafter
├── curiosity_prober.py        # Void exploration (active curiosity)
├── ideation.py                # High-level IdeationEngine + AutonomousIdeator
├── creative_ideation.py       # Unified CreativeIdeationEngine (flat + hierarchical + leap)
├── language_grounding.py      # Text ↔ embedding bridge (v1.3.0)
├── story_encoder.py           # Real-story encoding for benchmarks
├── prediction.py              # NarrativePredictor benchmark
├── chaos_monitor.py           # Chaos/edge-of-stability metrics
├── heartbeat_integration.py   # Long-running heartbeat ideator
├── toy_system.py              # LMDToySystem harness for experiments
├── safeguards.py              # Repulsion, anchoring, budgets, ID generation
└── cuda/                      # Optional Triton CUDA kernels
    ├── kernels.py
    ├── batch_ops.py
    └── fallback.py
```

## Benchmarks

| Operation | Throughput | Memory |
|-----------|-----------|--------|
| Memory Evolution | ~10,000 steps/s | O(n) |
| Analogical Transfer | ~500 leaps/s | O(n²) |
| Orthogonal Composition | ~800 leaps/s | O(n) |
| Void Extrapolation | ~600 leaps/s | O(n) |
| Full Ideation Cycle | ~50 ideas/s | O(n²) |

*Benchmarked on NVIDIA GeForce RTX 5080 (16 GB VRAM), CUDA 13.1, 256-dim embeddings, 100 memories. Reproduce locally with:*

```bash
python benchmarks/benchmarks.py
```

## Key Features

- **No LLM Required**: All operations are internal to embedding space
- **Emergent Narratives**: Stories arise from memory dynamics
- **Creative Recombination**: Generates novel ideas by operating on stored memories (not from nothing)
- **Biologically Inspired**: Metabolic states, resonance, narrative arcs
- **GPU Accelerated**: Full CUDA support via Triton kernels (optional)
- **Thread Safe**: Concurrent access supported
- **Optional Language Grounding**: Read and write text, not just vectors

## Research Paper

See [RESEARCH_PAPER_LMD.md](./docs/RESEARCH_PAPER_LMD.md) for the full technical paper including:
- Mathematical foundations
- Algorithm pseudocode
- Comprehensive benchmarks
- Comparison with existing systems

## Examples

All examples live in [`examples/`](./examples/) and are tested to run against the published API.

```bash
python examples/<example_name>.py
```

| Example | Description | Requires |
|---------|-------------|----------|
| [basic_usage.py](examples/basic_usage.py) | Create and evolve memories, observe coupling | Core |
| [creative_leaps.py](examples/creative_leaps.py) | Run each of the 4 creative operators | Core |
| [hierarchical_ideas.py](examples/hierarchical_ideas.py) | Tree-structured concepts + graft operations | Core |
| [language_grounding.py](examples/language_grounding.py) | Text in/out with real output | `[language]` |

## Testing

```bash
pip install -e ".[dev,language]"
pytest -q
```

The suite is ~179 tests across metabolism, coupling, narrative, creative leaps, hierarchical ideas, imagination, plausibility, and end-to-end story generation. CI runs the same suite on Python 3.10 / 3.11 / 3.12 via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

This project uses a [custom license](LICENSE) that allows free use for research and personal projects while reserving commercial rights.

For commercial licensing inquiries, please contact the author.

## Citation

If you use LMD in your research, please cite:

```bibtex
@software{lmd2026,
  author = {Thomas, Joshua R.},
  title = {Living Memory Dynamics: A Novel Framework for Narrative-Generating Episodic Memory},
  year = {2026},
  version = {1.3.0},
  url = {https://github.com/mordiaky/LMD}
}
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting PRs.

---

*Invented by Joshua R. Thomas, January 2026.*

*Contact: mordiaky@gmail.com*
