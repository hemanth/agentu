"""Background cache sync daemon."""

import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .storage import SQLiteBackend

logger = logging.getLogger(__name__)


class CacheSync:
    """Periodically syncs cache to a snapshot location."""

    def __init__(
        self,
        source_backend: SQLiteBackend,
        sync_path: str,
        sync_interval: int = 300,
    ):
        self.source = source_backend
        self.sync_path = Path(sync_path)
        self.sync_path.mkdir(parents=True, exist_ok=True)
        self.sync_interval = sync_interval
        self._dirty = False
        self._task: Optional[asyncio.Task] = None
        self.running = False

    def mark_dirty(self):
        self._dirty = True

    @property
    def snapshot_path(self) -> Path:
        return self.sync_path / "cache_snapshot.db"

    async def sync_once(self) -> bool:
        if not self._dirty:
            return False
        try:
            shutil.copy2(self.source.db_path, str(self.snapshot_path))
            self._dirty = False
            logger.info("Cache synced to %s", self.snapshot_path)
            return True
        except Exception:
            logger.error("Cache sync failed", exc_info=True)
            return False

    async def restore(self, target_backend: SQLiteBackend) -> bool:
        if not self.snapshot_path.exists():
            return False
        try:
            src_conn = sqlite3.connect(str(self.snapshot_path))
            cursor = src_conn.execute("SELECT key, value, created_at, expires_at FROM cache_store")
            rows = cursor.fetchall()
            src_conn.close()

            dst_conn = sqlite3.connect(target_backend.db_path)
            for key, value, created_at, expires_at in rows:
                dst_conn.execute(
                    "INSERT OR REPLACE INTO cache_store (key, value, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (key, value, created_at, expires_at)
                )
            dst_conn.commit()
            dst_conn.close()
            logger.info("Restored %d entries from snapshot", len(rows))
            return True
        except Exception:
            logger.error("Cache restore failed", exc_info=True)
            return False

    def start(self) -> asyncio.Task:
        self.running = True
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Final sync
        await self.sync_once()

    async def _loop(self):
        try:
            while self.running:
                await asyncio.sleep(self.sync_interval)
                await self.sync_once()
        except asyncio.CancelledError:
            pass
