"""Semantic index for cache similarity matching."""

import sqlite3
import json
import hashlib
import logging
from typing import Optional, Dict, Any

from .cache_embeddings import EmbeddingProvider, cosine_similarity

logger = logging.getLogger(__name__)


class SemanticIndex:
    """Stores embeddings alongside cache keys for semantic lookup."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_path: Optional[str] = None,
        threshold: float = 0.95,
    ):
        self.provider = embedding_provider
        self.threshold = threshold

        if db_path is None:
            from pathlib import Path
            cache_dir = Path.home() / ".agentu"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "semantic_index.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    async def add(self, text: str, cache_key: str) -> None:
        embedding = await self.provider.embed(text)
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, text_hash, embedding) VALUES (?, ?, ?)",
                (cache_key, text_hash, json.dumps(embedding))
            )
            conn.commit()
        finally:
            conn.close()

    async def search(self, text: str) -> Optional[str]:
        query_embedding = await self.provider.embed(text)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT cache_key, embedding FROM embeddings")
            rows = cursor.fetchall()
        finally:
            conn.close()

        best_key = None
        best_score = -1.0

        for cache_key, emb_json in rows:
            stored_embedding = json.loads(emb_json)
            score = cosine_similarity(query_embedding, stored_embedding)
            if score >= self.threshold and score > best_score:
                best_score = score
                best_key = cache_key

        return best_key

    async def remove(self, cache_key: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM embeddings WHERE cache_key = ?", (cache_key,))
            conn.commit()
        finally:
            conn.close()

    async def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
        finally:
            conn.close()

    async def stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
        finally:
            conn.close()
        return {
            "embedding_count": count,
            "threshold": self.threshold,
            "provider": type(self.provider).__name__,
        }
