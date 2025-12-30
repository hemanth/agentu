"""Storage backends for agentu memory system."""

import json
import sqlite3
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    def save(self, entries: List[Any]) -> None:
        """Save memory entries to storage."""
        pass

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load memory entries from storage."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""
        pass


class JSONStorage(MemoryStorage):
    """JSON file-based storage backend."""

    def __init__(self, file_path: str):
        """Initialize JSON storage.

        Args:
            file_path: Path to JSON file
        """
        self.file_path = file_path

    def save(self, entries: List[Any]) -> None:
        """Save entries to JSON file."""
        try:
            from datetime import datetime
            data = {
                'entries': [entry.to_dict() for entry in entries],
                'saved_at': datetime.now().isoformat()
            }

            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(entries)} memories to {self.file_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            raise

    def load(self) -> List[Dict[str, Any]]:
        """Load entries from JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            return data.get('entries', [])
        except FileNotFoundError:
            logger.debug(f"No existing JSON file found at {self.file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading from JSON: {str(e)}")
            raise

    def close(self) -> None:
        """No-op for JSON storage."""
        pass


class SQLiteStorage(MemoryStorage):
    """SQLite database storage backend with indexing and efficient querying."""

    def __init__(self, db_path: str):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database schema with indexes."""
        cursor = self.conn.cursor()

        # Create memory entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memory_entries(memory_type)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memory_entries(timestamp DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_importance
            ON memory_entries(importance DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_last_accessed
            ON memory_entries(last_accessed DESC)
        ''')

        # Full-text search index for content
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(content, content_rowid UNINDEXED)
        ''')

        self.conn.commit()

    def save(self, entries: List[Any]) -> None:
        """Save entries to SQLite database.

        This replaces all existing entries with the new list.
        """
        cursor = self.conn.cursor()

        try:
            # Clear existing entries
            cursor.execute('DELETE FROM memory_entries')
            cursor.execute('DELETE FROM memory_fts')

            # Insert new entries
            for entry in entries:
                metadata_json = json.dumps(entry.metadata)

                cursor.execute('''
                    INSERT INTO memory_entries
                    (content, timestamp, metadata, memory_type, importance, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.content,
                    entry.timestamp,
                    metadata_json,
                    entry.memory_type,
                    entry.importance,
                    entry.access_count,
                    entry.last_accessed
                ))

                # Add to FTS index
                rowid = cursor.lastrowid
                cursor.execute('''
                    INSERT INTO memory_fts (rowid, content)
                    VALUES (?, ?)
                ''', (rowid, entry.content))

            self.conn.commit()
            logger.debug(f"Saved {len(entries)} memories to SQLite database")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving to SQLite: {str(e)}")
            raise

    def load(self) -> List[Dict[str, Any]]:
        """Load all entries from SQLite database."""
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                SELECT content, timestamp, metadata, memory_type,
                       importance, access_count, last_accessed
                FROM memory_entries
                ORDER BY timestamp DESC
            ''')

            entries = []
            for row in cursor.fetchall():
                entry_dict = {
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'metadata': json.loads(row['metadata']),
                    'memory_type': row['memory_type'],
                    'importance': row['importance'],
                    'access_count': row['access_count'],
                    'last_accessed': row['last_accessed']
                }
                entries.append(entry_dict)

            return entries

        except Exception as e:
            logger.error(f"Error loading from SQLite: {str(e)}")
            raise

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search entries using full-text search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entry dictionaries
        """
        cursor = self.conn.cursor()

        try:
            # Use FTS for efficient text search
            cursor.execute('''
                SELECT m.content, m.timestamp, m.metadata, m.memory_type,
                       m.importance, m.access_count, m.last_accessed
                FROM memory_fts f
                JOIN memory_entries m ON f.rowid = m.id
                WHERE memory_fts MATCH ?
                ORDER BY m.importance DESC, m.access_count DESC
                LIMIT ?
            ''', (query, limit))

            entries = []
            for row in cursor.fetchall():
                entry_dict = {
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'metadata': json.loads(row['metadata']),
                    'memory_type': row['memory_type'],
                    'importance': row['importance'],
                    'access_count': row['access_count'],
                    'last_accessed': row['last_accessed']
                }
                entries.append(entry_dict)

            return entries

        except Exception as e:
            logger.error(f"Error searching SQLite: {str(e)}")
            # Fallback to LIKE search if FTS fails
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback search using LIKE operator."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT content, timestamp, metadata, memory_type,
                   importance, access_count, last_accessed
            FROM memory_entries
            WHERE content LIKE ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        entries = []
        for row in cursor.fetchall():
            entry_dict = {
                'content': row['content'],
                'timestamp': row['timestamp'],
                'metadata': json.loads(row['metadata']),
                'memory_type': row['memory_type'],
                'importance': row['importance'],
                'access_count': row['access_count'],
                'last_accessed': row['last_accessed']
            }
            entries.append(entry_dict)

        return entries

    def get_by_type(self, memory_type: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get entries by type using indexed query.

        Args:
            memory_type: Type to filter by
            limit: Maximum results

        Returns:
            List of entry dictionaries
        """
        cursor = self.conn.cursor()

        try:
            if limit:
                cursor.execute('''
                    SELECT content, timestamp, metadata, memory_type,
                           importance, access_count, last_accessed
                    FROM memory_entries
                    WHERE memory_type = ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                ''', (memory_type, limit))
            else:
                cursor.execute('''
                    SELECT content, timestamp, metadata, memory_type,
                           importance, access_count, last_accessed
                    FROM memory_entries
                    WHERE memory_type = ?
                    ORDER BY importance DESC, timestamp DESC
                ''', (memory_type,))

            entries = []
            for row in cursor.fetchall():
                entry_dict = {
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'metadata': json.loads(row['metadata']),
                    'memory_type': row['memory_type'],
                    'importance': row['importance'],
                    'access_count': row['access_count'],
                    'last_accessed': row['last_accessed']
                }
                entries.append(entry_dict)

            return entries

        except Exception as e:
            logger.error(f"Error querying by type from SQLite: {str(e)}")
            raise

    def update_access_stats(self, content: str, timestamp: float) -> None:
        """Update access count and last_accessed for an entry.

        Args:
            content: Content of the entry
            timestamp: Timestamp of the entry
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                UPDATE memory_entries
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE content = ? AND timestamp = ?
            ''', (timestamp, content, timestamp))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error updating access stats: {str(e)}")
            self.conn.rollback()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug(f"Closed SQLite connection to {self.db_path}")


def create_storage(storage_path: str, use_sqlite: bool = True) -> MemoryStorage:
    """Factory function to create appropriate storage backend.

    Args:
        storage_path: Path to storage file
        use_sqlite: If True, use SQLite; otherwise use JSON

    Returns:
        MemoryStorage instance
    """
    if use_sqlite:
        if not storage_path.endswith('.db'):
            storage_path = storage_path.replace('.json', '.db')
        return SQLiteStorage(storage_path)
    else:
        if not storage_path.endswith('.json'):
            storage_path = storage_path.replace('.db', '.json')
        return JSONStorage(storage_path)
