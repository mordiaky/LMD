"""Story Encoder - Convert real text narratives into Living Memories.

Takes actual stories (text) and creates LivingMemory objects with:
- Content: Sentence embeddings (using simple word averaging or external encoder)
- Valence: Extracted emotional valence from text
- Phase: Narrative position in the story

This allows testing LMD on REAL data, not synthetic.

Invented by Joshua R. Thomas, January 2026.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import LMDConfig
from .living_memory import LivingMemory, ValenceTrajectory

# Simple sentiment lexicon for valence extraction
# In production, use a proper sentiment model
POSITIVE_WORDS = {
    'happy', 'joy', 'love', 'wonderful', 'beautiful', 'good', 'great', 'best',
    'amazing', 'excited', 'hope', 'dream', 'success', 'win', 'victory', 'peace',
    'kind', 'gentle', 'warm', 'bright', 'light', 'smile', 'laugh', 'celebrate',
    'treasure', 'gift', 'magic', 'blessed', 'grateful', 'free', 'alive', 'safe',
    'hero', 'brave', 'strong', 'wise', 'clever', 'saved', 'rescue', 'triumph'
}

NEGATIVE_WORDS = {
    'sad', 'angry', 'hate', 'terrible', 'ugly', 'bad', 'worst', 'awful',
    'scared', 'fear', 'despair', 'fail', 'lose', 'defeat', 'war', 'cruel',
    'harsh', 'cold', 'dark', 'frown', 'cry', 'mourn', 'curse', 'poison',
    'danger', 'threat', 'evil', 'wicked', 'villain', 'monster', 'death', 'die',
    'pain', 'hurt', 'suffer', 'trap', 'lost', 'alone', 'betray', 'abandon'
}

# Story structure markers
CLIMAX_MARKERS = {
    'suddenly', 'finally', 'at last', 'then', 'but', 'however', 'until',
    'confronted', 'faced', 'fought', 'discovered', 'realized', 'revealed'
}

RESOLUTION_MARKERS = {
    'after', 'eventually', 'in the end', 'thereafter', 'since then',
    'happily', 'peacefully', 'forever', 'always', 'returned', 'home'
}


@dataclass
class EncodedStory:
    """A story encoded as living memories."""
    memories: List[LivingMemory]
    sentences: List[str]
    valences: List[float]
    phases: List[float]
    title: str

    def get_memory_by_sentence(self, sentence_idx: int) -> Optional[LivingMemory]:
        """Get memory for a specific sentence."""
        if 0 <= sentence_idx < len(self.memories):
            return self.memories[sentence_idx]
        return None

    def get_context_window(self, center_idx: int, window: int = 2) -> List[LivingMemory]:
        """Get memories in a window around an index."""
        start = max(0, center_idx - window)
        end = min(len(self.memories), center_idx + window + 1)
        return self.memories[start:end]


class StoryEncoder:
    """Encodes text stories into Living Memories for LMD testing.

    Supports:
    - Simple bag-of-words encoding (no external dependencies)
    - Valence extraction from text
    - Automatic narrative phase assignment
    - Optional external sentence encoders (sentence-transformers)
    """

    def __init__(
        self,
        config: LMDConfig,
        vocab_size: int = 5000,
        use_external_encoder: bool = False
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.use_external_encoder = use_external_encoder

        # Build vocabulary from common words
        self.word_to_idx: Dict[str, int] = {}
        self.embedding_dim = config.content_dim

        # Random word embeddings (in production, use pretrained)
        self.word_embeddings = torch.randn(vocab_size, self.embedding_dim) * 0.1

        # External encoder (lazy load)
        self._external_encoder = None

    def encode_story(
        self,
        text: str,
        title: str = "Untitled"
    ) -> EncodedStory:
        """Encode a full story into living memories.

        Args:
            text: The story text
            title: Story title for reference

        Returns:
            EncodedStory with memories, sentences, and metadata
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return EncodedStory([], [], [], [], title)

        # Encode each sentence
        memories = []
        valences = []
        phases = []

        n_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            # Compute content embedding
            content = self._encode_sentence(sentence)

            # Extract valence
            valence = self._extract_valence(sentence)
            valences.append(valence)

            # Assign narrative phase based on position and content
            phase = self._assign_phase(sentence, i, n_sentences)
            phases.append(phase)

            # Create valence trajectory
            # Look ahead to determine arc
            if i < n_sentences - 1:
                next_valence = self._extract_valence(sentences[i + 1])
                trajectory = self._create_trajectory(valence, next_valence)
            else:
                trajectory = ValenceTrajectory.constant(valence)

            # Create living memory
            memory = LivingMemory(
                id=i,
                content=content,
                valence=trajectory,
                phase=phase,
                energy=1.0,
                created_at=i,
                label=f"s{i}: {sentence[:30]}..."
            )
            memories.append(memory)

        return EncodedStory(
            memories=memories,
            sentences=sentences,
            valences=valences,
            phases=phases,
            title=title
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _encode_sentence(self, sentence: str) -> torch.Tensor:
        """Encode a sentence into a content vector."""
        if self.use_external_encoder:
            return self._encode_external(sentence)

        # Simple bag-of-words encoding
        words = self._tokenize(sentence)

        if not words:
            return torch.randn(self.embedding_dim) * 0.01

        # Average word embeddings
        embeddings = []
        for word in words:
            idx = self._get_word_idx(word)
            embeddings.append(self.word_embeddings[idx])

        content = torch.stack(embeddings).mean(dim=0)

        # Normalize
        norm = content.norm()
        if norm > 0:
            content = content / norm

        return content

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return [w for w in words if len(w) > 1]

    def _get_word_idx(self, word: str) -> int:
        """Get index for a word (hash-based for unknown words)."""
        if word not in self.word_to_idx:
            # Hash to vocab
            self.word_to_idx[word] = hash(word) % self.vocab_size
        return self.word_to_idx[word]

    def _extract_valence(self, sentence: str) -> float:
        """Extract emotional valence from sentence."""
        words = set(self._tokenize(sentence))

        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)

        total = pos_count + neg_count
        if total == 0:
            return 0.0  # Neutral

        # Valence in [-1, 1]
        valence = (pos_count - neg_count) / total

        # Scale and clamp
        valence = max(-1.0, min(1.0, valence))

        return valence

    def _assign_phase(self, sentence: str, idx: int, total: int) -> float:
        """Assign narrative phase based on position and content."""
        # Base phase from position (0 to 2*pi)
        position_phase = (idx / total) * 2 * math.pi

        # Adjust based on content markers
        words = set(self._tokenize(sentence))

        # Check for climax markers
        if words & CLIMAX_MARKERS:
            # Pull toward climax (pi)
            position_phase = 0.7 * position_phase + 0.3 * math.pi

        # Check for resolution markers
        if words & RESOLUTION_MARKERS:
            # Pull toward resolution (1.5*pi)
            position_phase = 0.7 * position_phase + 0.3 * 1.5 * math.pi

        return position_phase % (2 * math.pi)

    def _create_trajectory(self, current: float, next_val: float) -> ValenceTrajectory:
        """Create valence trajectory between current and next."""
        delta = next_val - current

        if delta > 0.3:
            return ValenceTrajectory.redemption()
        elif delta < -0.3:
            return ValenceTrajectory.tragedy()
        elif current > 0.3:
            return ValenceTrajectory.constant(current)
        elif current < -0.3:
            return ValenceTrajectory.constant(current)
        else:
            return ValenceTrajectory.random(n_points=5)

    def _encode_external(self, sentence: str) -> torch.Tensor:
        """Encode using external sentence encoder (if available)."""
        try:
            if self._external_encoder is None:
                from sentence_transformers import SentenceTransformer
                self._external_encoder = SentenceTransformer('all-MiniLM-L6-v2')

            embedding = self._external_encoder.encode(sentence)
            tensor = torch.tensor(embedding, dtype=torch.float32)

            # Project to config dimension if needed
            if len(tensor) != self.embedding_dim:
                # Simple projection
                proj = torch.randn(len(tensor), self.embedding_dim) * 0.1
                tensor = tensor @ proj

            return tensor

        except ImportError:
            # Fall back to simple encoding
            return self._encode_sentence(sentence)


# Sample stories for testing
SAMPLE_STORIES = {
    "the_hero": """
    Once upon a time, there was a young hero named Alex who lived in a peaceful village.
    Alex was kind and brave, always helping others in need.
    One dark day, an evil dragon attacked the village, spreading fear and destruction.
    The villagers were terrified and hid in their homes.
    Alex knew something had to be done to save everyone.
    With courage in heart, Alex journeyed to the dragon's mountain lair.
    The path was dangerous, filled with traps and monsters.
    Alex faced many challenges but never gave up hope.
    Finally, Alex confronted the mighty dragon in its cave.
    The battle was fierce and Alex was badly hurt.
    But with one final blow, Alex defeated the evil beast.
    The dragon fell, and peace returned to the land.
    Alex returned home as a celebrated hero.
    The village rejoiced and celebrated for many days.
    And they all lived happily ever after.
    """,

    "the_loss": """
    Sarah had always been happy, living with her beloved grandmother.
    They would spend wonderful days gardening and baking together.
    Grandmother told the most magical stories by the fireplace.
    But one cold winter, grandmother became very ill.
    Sarah was scared and prayed for her recovery.
    The doctors tried everything but could not help.
    Grandmother grew weaker with each passing day.
    Sarah held her hand and cried bitter tears.
    On a quiet night, grandmother passed away peacefully.
    Sarah was devastated and felt completely alone.
    The house felt empty and dark without her.
    For months, Sarah grieved in deep sadness.
    But slowly, she began to remember the happy times.
    She found grandmother's old recipe book and smiled.
    Sarah decided to carry on grandmother's legacy of kindness.
    Though the pain remained, hope began to return.
    """,

    "the_discovery": """
    Maya was a curious scientist working alone in her lab.
    She had been studying a mysterious signal from space.
    Everyone thought she was wasting her time on nothing.
    But Maya believed there was something important out there.
    Night after night, she analyzed the strange patterns.
    Then one evening, the signal suddenly became clear.
    Maya's heart raced as she decoded the message.
    It was proof of intelligent life beyond Earth.
    She was terrified and excited at the same time.
    The discovery would change everything humanity knew.
    Maya faced a difficult choice about revealing the truth.
    She decided the world deserved to know.
    The announcement shocked and amazed everyone.
    Some were afraid, but many felt wonder and hope.
    Maya had opened a door to the universe.
    A new chapter in human history had begun.
    """
}


def get_sample_story(name: str = "the_hero") -> str:
    """Get a sample story by name."""
    return SAMPLE_STORIES.get(name, SAMPLE_STORIES["the_hero"])
