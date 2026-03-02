"""Tests for cache background sync."""

import pytest
import pytest_asyncio
import asyncio

from agentu.cache_sync import CacheSync
from agentu.cache_storage_backends import SQLiteBackend


class TestCacheSync:
    @pytest_asyncio.fixture
    async def setup(self, tmp_path):
        source = SQLiteBackend(db_path=str(tmp_path / "source.db"))
        sync_path = str(tmp_path / "sync")
        sync = CacheSync(source_backend=source, sync_path=sync_path, sync_interval=1)
        return source, sync, sync_path

    @pytest.mark.asyncio
    async def test_snapshot_created(self, setup):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        sync.mark_dirty()
        await sync.sync_once()
        from pathlib import Path
        snapshot = Path(sync_path) / "cache_snapshot.db"
        assert snapshot.exists()

    @pytest.mark.asyncio
    async def test_restore_from_snapshot(self, setup, tmp_path):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        sync.mark_dirty()
        await sync.sync_once()

        # New backend, restore from snapshot
        new_backend = SQLiteBackend(db_path=str(tmp_path / "restored.db"))
        restored = await sync.restore(new_backend)
        assert restored is True
        result = await new_backend.get("k1")
        assert result == {"r": "1"}

    @pytest.mark.asyncio
    async def test_no_sync_when_clean(self, setup):
        source, sync, sync_path = setup
        await source.set("k1", {"r": "1"})
        # Don't mark dirty
        await sync.sync_once()
        from pathlib import Path
        snapshot = Path(sync_path) / "cache_snapshot.db"
        assert not snapshot.exists()

    @pytest.mark.asyncio
    async def test_start_stop(self, setup):
        source, sync, sync_path = setup
        task = sync.start()
        assert sync.running is True
        await sync.stop()
        assert sync.running is False
