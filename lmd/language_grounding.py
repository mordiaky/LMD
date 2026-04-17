"""Language Grounding Module for Living Memory Dynamics.

Bridges the gap between LMD's embedding space and human-readable text.

Features:
1. Text → Embedding encoding via sentence-transformers
2. Corpus-based retrieval for embedding → text decoding
3. Creative leap description via nearest neighbor interpolation
4. Optional LLM integration for rich descriptions

Invented by Joshua R. Thomas, January 2026.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    # Forward-ref annotations only; the runtime import is deferred to inside
    # ground_memory() / ground_memories() to avoid a circular import with
    # lmd.living_memory.
    from .living_memory import LivingMemory

# Optional imports - graceful degradation
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class EncoderType(Enum):
    """Supported encoder types."""
    MINILM = "all-MiniLM-L6-v2"           # 384 dim, fast
    MPNET = "all-mpnet-base-v2"            # 768 dim, better quality
    INSTRUCTOR = "hkunlp/instructor-large" # 768 dim, instruction-tuned
    CUSTOM = "custom"                       # User-provided


@dataclass
class GroundingConfig:
    """Configuration for language grounding."""
    encoder_type: EncoderType = EncoderType.MINILM
    custom_model_name: Optional[str] = None
    embedding_dim: int = 384  # Will be auto-detected from model
    corpus_path: Optional[str] = None
    max_corpus_size: int = 100000
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.3
    use_gpu: bool = True
    cache_embeddings: bool = True
    normalize_embeddings: bool = True


@dataclass
class GroundedText:
    """A text with its embedding."""
    text: str
    embedding: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.text)


@dataclass
class RetrievalResult:
    """Result of retrieving text from embedding."""
    query_embedding: torch.Tensor
    matches: List[Tuple[str, float]]  # (text, similarity)
    interpolated_description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CreativeLeapDescription:
    """Human-readable description of a creative leap."""
    leap_type: str
    source_texts: List[str]
    result_embedding: torch.Tensor
    nearest_texts: List[Tuple[str, float]]
    synthesized_description: str
    novelty_score: float
    grounding_confidence: float


class TextCorpus:
    """Manages a corpus of texts with their embeddings for retrieval."""

    def __init__(self, config: GroundingConfig):
        self.config = config
        self.texts: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._text_to_idx: Dict[str, int] = {}

    def add(self, text: str, embedding: torch.Tensor, metadata: Optional[Dict] = None) -> int:
        """Add a text to the corpus."""
        with self._lock:
            if text in self._text_to_idx:
                return self._text_to_idx[text]

            if len(self.texts) >= self.config.max_corpus_size:
                # Remove oldest entry
                old_text = self.texts.pop(0)
                del self._text_to_idx[old_text]
                self.metadata.pop(0)
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[1:]
                # Update indices
                self._text_to_idx = {t: i for i, t in enumerate(self.texts)}

            idx = len(self.texts)
            self.texts.append(text)
            self._text_to_idx[text] = idx
            self.metadata.append(metadata or {})

            # Add embedding
            emb = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
            if self.embeddings is None:
                self.embeddings = emb
            else:
                self.embeddings = torch.cat([self.embeddings, emb], dim=0)

            return idx

    def add_batch(self, texts: List[str], embeddings: torch.Tensor,
                  metadata: Optional[List[Dict]] = None) -> List[int]:
        """Add multiple texts at once."""
        indices = []
        metadata = metadata or [{}] * len(texts)
        for text, emb, meta in zip(texts, embeddings, metadata):
            idx = self.add(text, emb, meta)
            indices.append(idx)
        return indices

    def search(self, query_embedding: torch.Tensor, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search for nearest texts to query embedding."""
        with self._lock:
            if self.embeddings is None or len(self.texts) == 0:
                return []

            top_k = top_k or self.config.retrieval_top_k

            # Ensure query is on same device as corpus
            query = query_embedding.to(self.embeddings.device)

            # Normalize for cosine similarity
            query_norm = F.normalize(query.unsqueeze(0), p=2, dim=1)
            corpus_norm = F.normalize(self.embeddings, p=2, dim=1)

            # Compute similarities
            similarities = torch.mm(query_norm, corpus_norm.t()).squeeze(0)

            # Get top-k
            k = min(top_k, len(self.texts))
            top_sims, top_indices = similarities.topk(k)

            results = []
            for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
                if sim >= self.config.similarity_threshold:
                    results.append((self.texts[idx], sim))

            return results

    def save(self, path: str):
        """Save corpus to disk."""
        with self._lock:
            data = {
                "texts": self.texts,
                "metadata": self.metadata,
                "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
            }
            with open(path, 'w') as f:
                json.dump(data, f)

    def load(self, path: str):
        """Load corpus from disk."""
        with self._lock:
            with open(path) as f:
                data = json.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            if data["embeddings"]:
                self.embeddings = torch.tensor(data["embeddings"])
            self._text_to_idx = {t: i for i, t in enumerate(self.texts)}

    def __len__(self) -> int:
        return len(self.texts)


class LanguageGrounding:
    """Main class for grounding LMD embeddings in language.

    Example usage:
        ```python
        from lmd import LanguageGrounding, GroundingConfig

        # Initialize
        config = GroundingConfig(encoder_type=EncoderType.MINILM)
        grounding = LanguageGrounding(config)

        # Encode text to embedding
        embedding = grounding.encode("a fire-breathing dragon")

        # Add to corpus for later retrieval
        grounding.add_to_corpus("a fire-breathing dragon")
        grounding.add_to_corpus("crystalline glass structure")
        grounding.add_to_corpus("prismatic light refraction")

        # Decode embedding back to text
        result = grounding.decode(some_embedding)
        print(result.matches)  # [("fire-breathing dragon", 0.85), ...]

        # Describe a creative leap
        description = grounding.describe_leap(
            leap_type="ANALOGICAL",
            sources=[dragon_emb, glass_emb],
            result=prismatic_breath_emb
        )
        print(description.synthesized_description)
        ```
    """

    def __init__(self, config: Optional[GroundingConfig] = None):
        self.config = config or GroundingConfig()
        self._lock = threading.RLock()
        self._encoder = None
        self._corpus = TextCorpus(self.config)
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._device = None

        # Initialize encoder
        self._init_encoder()

        # Load corpus if path provided
        if self.config.corpus_path and Path(self.config.corpus_path).exists():
            self._corpus.load(self.config.corpus_path)

    def _init_encoder(self):
        """Initialize the sentence transformer encoder."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not installed. "
                  "Install with: pip install sentence-transformers")
            return

        model_name = (self.config.custom_model_name
                      if self.config.encoder_type == EncoderType.CUSTOM
                      else self.config.encoder_type.value)

        self._encoder = SentenceTransformer(model_name)

        # Set device
        if self.config.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._encoder = self._encoder.to(self._device)
        else:
            self._device = torch.device("cpu")

        # Auto-detect embedding dimension
        test_emb = self._encoder.encode(["test"], convert_to_tensor=True)
        self.config.embedding_dim = test_emb.shape[1]

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.config.embedding_dim

    @property
    def is_available(self) -> bool:
        """Check if encoder is available."""
        return self._encoder is not None

    def encode(self, text: Union[str, List[str]],
               use_cache: bool = True) -> torch.Tensor:
        """Encode text(s) to embedding(s).

        Args:
            text: Single text or list of texts
            use_cache: Whether to use embedding cache

        Returns:
            Embedding tensor (dim,) for single text, (n, dim) for list
        """
        if not self.is_available:
            raise RuntimeError("Encoder not available. Install sentence-transformers.")

        single_input = isinstance(text, str)
        texts = [text] if single_input else text

        # Check cache
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, t in enumerate(texts):
            cache_key = self._cache_key(t)
            if use_cache and self.config.cache_embeddings and cache_key in self._embedding_cache:
                results.append((i, self._embedding_cache[cache_key]))
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_texts:
            with self._lock:
                embeddings = self._encoder.encode(
                    uncached_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )

                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)

                for idx, t, emb in zip(uncached_indices, uncached_texts, embeddings):
                    if self.config.cache_embeddings:
                        self._embedding_cache[self._cache_key(t)] = emb
                    results.append((idx, emb))

        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        embeddings = torch.stack([r[1] for r in results])

        if single_input:
            return embeddings.squeeze(0)
        return embeddings

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def add_to_corpus(self, text: Union[str, List[str]],
                      metadata: Optional[Union[Dict, List[Dict]]] = None) -> Union[int, List[int]]:
        """Add text(s) to the retrieval corpus.

        Args:
            text: Text or list of texts to add
            metadata: Optional metadata for the text(s)

        Returns:
            Index or list of indices in corpus
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text

        # Encode
        embeddings = self.encode(texts)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Add to corpus
        if single_input:
            meta = metadata if isinstance(metadata, dict) else None
            return self._corpus.add(texts[0], embeddings[0], meta)
        else:
            metas = metadata if isinstance(metadata, list) else [None] * len(texts)
            return self._corpus.add_batch(texts, embeddings, metas)

    def decode(self, embedding: torch.Tensor,
               top_k: Optional[int] = None) -> RetrievalResult:
        """Decode embedding to nearest texts in corpus.

        Args:
            embedding: The embedding to decode
            top_k: Number of nearest neighbors to return

        Returns:
            RetrievalResult with matched texts and similarities
        """
        matches = self._corpus.search(embedding, top_k)

        # Calculate confidence based on top match similarity
        confidence = matches[0][1] if matches else 0.0

        # Generate interpolated description if multiple matches
        interpolated = None
        if len(matches) >= 2:
            interpolated = self._interpolate_description(matches)
        elif len(matches) == 1:
            interpolated = matches[0][0]

        return RetrievalResult(
            query_embedding=embedding,
            matches=matches,
            interpolated_description=interpolated,
            confidence=confidence
        )

    def _interpolate_description(self, matches: List[Tuple[str, float]]) -> str:
        """Create interpolated description from multiple matches."""
        # Weight by similarity
        total_sim = sum(sim for _, sim in matches)

        if total_sim == 0:
            return matches[0][0] if matches else ""

        # Simple approach: combine top matches with weights
        # Format: "blend of X (70%) and Y (30%)"
        if len(matches) >= 2:
            text1, sim1 = matches[0]
            text2, sim2 = matches[1]
            pct1 = int(100 * sim1 / (sim1 + sim2))
            pct2 = 100 - pct1
            return f"blend of '{text1}' ({pct1}%) and '{text2}' ({pct2}%)"

        return matches[0][0]

    def describe_leap(self, leap_type: str,
                      sources: List[torch.Tensor],
                      result: torch.Tensor,
                      source_texts: Optional[List[str]] = None) -> CreativeLeapDescription:
        """Generate human-readable description of a creative leap.

        Args:
            leap_type: Type of leap (ANALOGICAL, DIFFUSION, ORTHOGONAL, EXTRAPOLATION)
            sources: Source embeddings used in the leap
            result: Resulting embedding from the leap
            source_texts: Optional known texts for sources

        Returns:
            CreativeLeapDescription with synthesized description
        """
        # Decode sources if texts not provided
        if source_texts is None:
            source_texts = []
            for src in sources[:3]:  # Limit to top 3 sources
                decoded = self.decode(src, top_k=1)
                if decoded.matches:
                    source_texts.append(decoded.matches[0][0])
                else:
                    source_texts.append("[unknown concept]")

        # Decode result
        result_decoded = self.decode(result, top_k=3)

        # Calculate novelty (how different is result from sources)
        novelty = self._calculate_novelty(sources, result)

        # Synthesize description based on leap type
        description = self._synthesize_description(
            leap_type, source_texts, result_decoded
        )

        return CreativeLeapDescription(
            leap_type=leap_type,
            source_texts=source_texts,
            result_embedding=result,
            nearest_texts=result_decoded.matches,
            synthesized_description=description,
            novelty_score=novelty,
            grounding_confidence=result_decoded.confidence
        )

    def _calculate_novelty(self, sources: List[torch.Tensor],
                           result: torch.Tensor) -> float:
        """Calculate how novel the result is compared to sources."""
        if not sources:
            return 1.0

        # Stack sources
        source_stack = torch.stack(sources)

        # Normalize
        result_norm = F.normalize(result.unsqueeze(0), p=2, dim=1)
        sources_norm = F.normalize(source_stack, p=2, dim=1)

        # Max similarity to any source
        similarities = torch.mm(result_norm, sources_norm.t())
        max_sim = similarities.max().item()

        # Novelty is inverse of similarity
        return max(0.0, min(1.0, 1.0 - max_sim))

    def _synthesize_description(self, leap_type: str,
                                source_texts: List[str],
                                result_decoded: RetrievalResult) -> str:
        """Synthesize a description based on leap type."""
        sources_str = " + ".join(f"'{t}'" for t in source_texts[:3])

        if not result_decoded.matches:
            result_str = "[novel concept beyond corpus]"
        elif result_decoded.confidence > 0.8:
            result_str = f"'{result_decoded.matches[0][0]}'"
        else:
            result_str = result_decoded.interpolated_description or "[unclear]"

        templates = {
            "ANALOGICAL": f"Transferred patterns from {sources_str} -> {result_str}",
            "DIFFUSION": f"Morphed through space from {sources_str} -> {result_str}",
            "ORTHOGONAL": f"Perpendicular composition of {sources_str} -> {result_str}",
            "EXTRAPOLATION": f"Extrapolated beyond {sources_str} into {result_str}",
        }

        return templates.get(leap_type, f"{leap_type}: {sources_str} -> {result_str}")

    def ground_memory(self, memory: "LivingMemory") -> GroundedText:
        """Ground a LivingMemory in language.

        Args:
            memory: A LivingMemory instance

        Returns:
            GroundedText with decoded text
        """

        decoded = self.decode(memory.content)

        text = decoded.interpolated_description or "[ungrounded]"

        return GroundedText(
            text=text,
            embedding=memory.content,
            metadata={
                "memory_id": memory.id,
                "energy": memory.energy,
                "phase": memory.phase.name if hasattr(memory.phase, 'name') else str(memory.phase),
                "confidence": decoded.confidence
            }
        )

    def ground_memories(self, memories: List["LivingMemory"]) -> List[GroundedText]:
        """Ground multiple memories in language."""
        return [self.ground_memory(m) for m in memories]

    def save_corpus(self, path: Optional[str] = None):
        """Save the corpus to disk."""
        path = path or self.config.corpus_path
        if path:
            self._corpus.save(path)

    def load_corpus(self, path: Optional[str] = None):
        """Load corpus from disk."""
        path = path or self.config.corpus_path
        if path and Path(path).exists():
            self._corpus.load(path)

    @property
    def corpus_size(self) -> int:
        """Get number of texts in corpus."""
        return len(self._corpus)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()


class LLMDescriber:
    """Optional LLM integration for richer descriptions.

    Supports multiple backends:
    - OpenAI API
    - Local Ollama
    - HuggingFace transformers
    """

    def __init__(self, backend: str = "ollama", model: str = "llama2"):
        self.backend = backend
        self.model = model
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize LLM client based on backend."""
        if self.backend == "ollama":
            try:
                import ollama
                self._client = ollama
            except ImportError:
                print("Ollama not installed. Install with: pip install ollama")
        elif self.backend == "openai":
            try:
                import openai
                self._client = openai.OpenAI()
            except ImportError:
                print("OpenAI not installed. Install with: pip install openai")

    def describe(self, grounding: LanguageGrounding,
                 embedding: torch.Tensor,
                 context: Optional[str] = None) -> str:
        """Use LLM to describe an embedding with richer language.

        Args:
            grounding: LanguageGrounding instance for retrieval
            embedding: The embedding to describe
            context: Optional context for the description

        Returns:
            Rich text description from LLM
        """
        if self._client is None:
            return "[LLM not available]"

        # Get nearest texts
        decoded = grounding.decode(embedding, top_k=5)

        # Build prompt
        matches_str = "\n".join(f"- {text} (similarity: {sim:.2f})"
                                for text, sim in decoded.matches)

        prompt = f"""Based on these related concepts:
{matches_str}

{f'Context: {context}' if context else ''}

Describe what new concept or idea this might represent.
Be creative but grounded in the related concepts.
Keep it to 1-2 sentences."""

        # Call LLM
        if self.backend == "ollama":
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        elif self.backend == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content

        return "[Unknown backend]"

    def describe_leap(self, grounding: LanguageGrounding,
                      leap_description: CreativeLeapDescription) -> str:
        """Use LLM to create rich description of creative leap."""
        context = f"Creative leap type: {leap_description.leap_type}\n"
        context += f"Source concepts: {', '.join(leap_description.source_texts)}\n"
        context += f"Novelty: {leap_description.novelty_score:.1%}"

        return self.describe(grounding, leap_description.result_embedding, context)


# Convenience function
def create_grounding(encoder: str = "minilm",
                     use_gpu: bool = True) -> LanguageGrounding:
    """Create a LanguageGrounding instance with sensible defaults.

    Args:
        encoder: One of "minilm" (fast), "mpnet" (better), "instructor" (best)
        use_gpu: Whether to use GPU acceleration

    Returns:
        Configured LanguageGrounding instance
    """
    encoder_map = {
        "minilm": EncoderType.MINILM,
        "mpnet": EncoderType.MPNET,
        "instructor": EncoderType.INSTRUCTOR,
    }

    config = GroundingConfig(
        encoder_type=encoder_map.get(encoder, EncoderType.MINILM),
        use_gpu=use_gpu
    )

    return LanguageGrounding(config)
