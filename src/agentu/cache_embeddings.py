"""Configurable embedding providers for semantic cache matching."""

import hashlib
import logging
import math
from typing import List, Optional, Protocol, runtime_checkable

import aiohttp

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]: ...
    def dimension(self) -> int: ...


class FakeEmbedding:
    """Deterministic embeddings for testing. NOT for production."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    async def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        values = []
        for i in range(0, min(len(h), self._dimension * 2), 2):
            values.append(int(h[i:i+2], 16) / 255.0)
        while len(values) < self._dimension:
            values.append(0.0)
        # Normalize
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        return values[:self._dimension]

    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding:
    """Local embeddings via sentence-transformers. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # default for MiniLM

    def available(self) -> bool:
        try:
            import sentence_transformers
            return True
        except (ImportError, TypeError):
            return False

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> List[float]:
        if not self.available():
            raise RuntimeError("sentence-transformers not installed")
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def dimension(self) -> int:
        return self._dimension


class APIEmbedding:
    """Embeddings via OpenAI-compatible API (Ollama, OpenAI, etc.)."""

    def __init__(self, api_base: str = "http://localhost:11434/v1",
                 model: str = "nomic-embed-text", api_key: Optional[str] = None):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._dimension = 768  # common default

    async def embed(self, text: str) -> List[float]:
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {"input": text, "model": self.model}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/embeddings", json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Embedding API returned status {resp.status}")
                    data = await resp.json()
                    embedding = data["data"][0]["embedding"]
                    self._dimension = len(embedding)
                    return embedding
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Embedding API call failed: %s", e)
            raise RuntimeError(f"Embedding API call failed: {e}") from e

    def dimension(self) -> int:
        return self._dimension


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
