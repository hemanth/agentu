"""Tests for memory storage backends."""

import os
import tempfile
import pytest
from agentu.memory import MemoryEntry
from agentu.memory_storage import JSONStorage, SQLiteStorage, create_storage


class TestJSONStorage:
    """Test JSON storage backend."""

    def test_save_and_load(self):
        """Test saving and loading with JSON storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            storage = JSONStorage(temp_path)

            # Create test entries
            entries = [
                MemoryEntry(content="Test 1", timestamp=123.0, metadata={}, memory_type="fact"),
                MemoryEntry(content="Test 2", timestamp=124.0, metadata={"key": "value"}, memory_type="task")
            ]

            # Save
            storage.save(entries)
            assert os.path.exists(temp_path)

            # Load
            loaded = storage.load()
            assert len(loaded) == 2
            assert loaded[0]['content'] == "Test 1"
            assert loaded[1]['metadata'] == {"key": "value"}

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        storage = JSONStorage("/tmp/nonexistent_memory.json")
        loaded = storage.load()
        assert loaded == []
        storage.close()


class TestSQLiteStorage:
    """Test SQLite storage backend."""

    def test_save_and_load(self):
        """Test saving and loading with SQLite storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = SQLiteStorage(temp_path)

            # Create test entries
            entries = [
                MemoryEntry(content="Python programming", timestamp=123.0, metadata={}, memory_type="fact", importance=0.8),
                MemoryEntry(content="JavaScript coding", timestamp=124.0, metadata={"lang": "js"}, memory_type="fact", importance=0.6)
            ]

            # Save
            storage.save(entries)

            # Load
            loaded = storage.load()
            assert len(loaded) == 2
            assert loaded[0]['content'] == "JavaScript coding"  # Sorted by timestamp DESC
            assert loaded[1]['content'] == "Python programming"

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_full_text_search(self):
        """Test full-text search functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = SQLiteStorage(temp_path)

            # Create test entries
            entries = [
                MemoryEntry(content="Python is great for data science", timestamp=123.0, metadata={}, memory_type="fact"),
                MemoryEntry(content="JavaScript is used for web development", timestamp=124.0, metadata={}, memory_type="fact"),
                MemoryEntry(content="Python programming basics", timestamp=125.0, metadata={}, memory_type="fact")
            ]

            storage.save(entries)

            # Search for "Python"
            results = storage.search("Python", limit=5)
            assert len(results) == 2
            assert all("Python" in r['content'] for r in results)

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_by_type(self):
        """Test getting entries by type."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = SQLiteStorage(temp_path)

            # Create test entries
            entries = [
                MemoryEntry(content="Task 1", timestamp=123.0, metadata={}, memory_type="task", importance=0.7),
                MemoryEntry(content="Fact 1", timestamp=124.0, metadata={}, memory_type="fact", importance=0.8),
                MemoryEntry(content="Task 2", timestamp=125.0, metadata={}, memory_type="task", importance=0.9)
            ]

            storage.save(entries)

            # Get by type
            tasks = storage.get_by_type("task")
            assert len(tasks) == 2
            assert all(t['memory_type'] == "task" for t in tasks)
            # Should be sorted by importance DESC
            assert tasks[0]['importance'] == 0.9

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_by_type_with_limit(self):
        """Test getting entries by type with limit."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = SQLiteStorage(temp_path)

            # Create test entries
            entries = [
                MemoryEntry(content=f"Task {i}", timestamp=float(i), metadata={}, memory_type="task")
                for i in range(10)
            ]

            storage.save(entries)

            # Get with limit
            tasks = storage.get_by_type("task", limit=3)
            assert len(tasks) == 3

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_database_schema(self):
        """Test that database schema is created correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = SQLiteStorage(temp_path)

            # Check tables exist
            cursor = storage.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert 'memory_entries' in tables
            assert 'memory_fts' in tables

            # Check indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]

            assert any('idx_memory_type' in idx for idx in indexes)
            assert any('idx_timestamp' in idx for idx in indexes)
            assert any('idx_importance' in idx for idx in indexes)

        finally:
            storage.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_persistence_across_instances(self):
        """Test that data persists across storage instances."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            # First instance - save data
            storage1 = SQLiteStorage(temp_path)
            entries = [
                MemoryEntry(content="Persistent data", timestamp=123.0, metadata={}, memory_type="fact")
            ]
            storage1.save(entries)
            storage1.close()

            # Second instance - load data
            storage2 = SQLiteStorage(temp_path)
            loaded = storage2.load()
            assert len(loaded) == 1
            assert loaded[0]['content'] == "Persistent data"
            storage2.close()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestStorageFactory:
    """Test storage factory function."""

    def test_create_sqlite_storage(self):
        """Test creating SQLite storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
            temp_path = f.name

        try:
            storage = create_storage(temp_path, use_sqlite=True)
            assert isinstance(storage, SQLiteStorage)
            storage.close()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_create_json_storage(self):
        """Test creating JSON storage."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            storage = create_storage(temp_path, use_sqlite=False)
            assert isinstance(storage, JSONStorage)
            storage.close()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_auto_extension_conversion(self):
        """Test automatic file extension conversion."""
        # JSON to DB
        storage_db = create_storage("memory.json", use_sqlite=True)
        assert isinstance(storage_db, SQLiteStorage)
        assert storage_db.db_path.endswith('.db')
        storage_db.close()

        # DB to JSON
        storage_json = create_storage("memory.db", use_sqlite=False)
        assert isinstance(storage_json, JSONStorage)
        assert storage_json.file_path.endswith('.json')
        storage_json.close()
