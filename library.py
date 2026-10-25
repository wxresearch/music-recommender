"""Local SQLite metadata and audio library. Never commit library_data/."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "library_data"
AUDIO = ROOT / "audio"
DB = ROOT / "library.sqlite3"


def connect():
    AUDIO.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS songs (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, extension TEXT NOT NULL,
        features TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        song_id TEXT PRIMARY KEY, preference INTEGER NOT NULL DEFAULT 0
    )""")
    con.execute('CREATE TABLE IF NOT EXISTS similarity_feedback (song_a TEXT, song_b TEXT, rating INTEGER, PRIMARY KEY(song_a, song_b))')
    con.execute("CREATE TABLE IF NOT EXISTS discoveries (artist TEXT NOT NULL, track TEXT NOT NULL, status TEXT NOT NULL, url TEXT, PRIMARY KEY(artist, track))")
    con.commit()
    return con


def add_song(song_id, title, extension, content, features):
    if extension not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise ValueError("Unsupported file extension")
    with connect() as con:
        if con.execute("SELECT 1 FROM songs WHERE id=?", (song_id,)).fetchone():
            return False
        # Files are local; do not store authentication data or upload audio to GitHub.
        (AUDIO / (song_id + extension)).write_bytes(content)
        con.execute("INSERT INTO songs VALUES (?,?,?,?)",
                    (song_id, title, extension, json.dumps(features)))
    return True


def songs():
    with connect() as con:
        return [{"id": r["id"], "title": r["title"],
                 "path": AUDIO / (r["id"] + r["extension"]),
                 "features": json.loads(r["features"])}
                for r in con.execute("SELECT * FROM songs ORDER BY title")]


def get_feedback():
    with connect() as con:
        return {r["song_id"]: r["preference"] for r in con.execute("SELECT * FROM feedback")}


def set_feedback(song_id, preference):
    if preference not in (-1, 0, 1):
        raise ValueError("Preference must be -1, 0 or 1")
    with connect() as con:
        con.execute("INSERT INTO feedback (song_id,preference) VALUES (?,?) "
                    "ON CONFLICT(song_id) DO UPDATE SET preference=excluded.preference",
                    (song_id, preference))


def delete_song(song_id):
    with connect() as con:
        row = con.execute("SELECT extension FROM songs WHERE id=?", (song_id,)).fetchone()
        if not row:
            return
        con.execute("DELETE FROM songs WHERE id=?", (song_id,))
        con.execute("DELETE FROM feedback WHERE song_id=?", (song_id,))
        con.execute("DELETE FROM similarity_feedback WHERE song_a=? OR song_b=?", (song_id, song_id))
        (AUDIO / (song_id + row["extension"])).unlink(missing_ok=True)


def get_similarity_feedback():
    with connect() as con:
        return {(r['song_a'], r['song_b']): r['rating'] for r in con.execute('SELECT * FROM similarity_feedback')}


def rate_similarity(a, b, rating):
    if a == b or rating not in (1, 2, 3, 4, 5):
        raise ValueError('Invalid similarity rating')
    a, b = sorted((a, b))
    with connect() as con:
        con.execute('INSERT INTO similarity_feedback VALUES (?,?,?) ON CONFLICT(song_a,song_b) DO UPDATE SET rating=excluded.rating', (a,b,rating))

def saved_discoveries():
    with connect() as con:
        return {(row["artist"].casefold(), row["track"].casefold()): dict(row)
                for row in con.execute("SELECT * FROM discoveries")}


def set_discovery_status(artist, track, status, url=""):
    if status not in ("Saved", "Not interested", "Already heard"):
        raise ValueError("Invalid status")
    with connect() as con:
        con.execute("INSERT INTO discoveries VALUES (?,?,?,?) ON CONFLICT(artist,track) DO UPDATE SET status=excluded.status, url=excluded.url",
                    (artist, track, status, url))
