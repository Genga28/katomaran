import sqlite3
import pickle
import os
import datetime

class DB:
    def __init__(self, db_path):
        self.db_path = db_path
        # Create DB folder if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # faces table
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                uuid TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        # events table
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT,
                event_type TEXT,
                image_path TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def register_face(self, uuid, embedding):
        self.cur.execute(
            "INSERT INTO faces (uuid, embedding) VALUES (?, ?)",
            (uuid, pickle.dumps(embedding))
        )
        self.conn.commit()

    def get_all_embeddings(self):
        self.cur.execute("SELECT uuid, embedding FROM faces")
        rows = self.cur.fetchall()
        return [(uuid, pickle.loads(emb)) for uuid, emb in rows]

    def add_event(self, uuid, event_type, image_path):
        ts = datetime.datetime.utcnow().isoformat()
        self.cur.execute(
            "INSERT INTO events (uuid, event_type, image_path, timestamp) VALUES (?, ?, ?, ?)",
            (uuid, event_type, image_path, ts)
        )
        self.conn.commit()

    def unique_count(self):
        self.cur.execute("SELECT COUNT(*) FROM faces")
        return self.cur.fetchone()[0]
