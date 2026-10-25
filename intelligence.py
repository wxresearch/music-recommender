"""Opt-in CLAP music tags, taste learning, playlist generation and portable backups."""
import io
import json
import hashlib
import re
import zipfile
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["ambient", "classical", "electronic", "jazz", "rock", "pop", "hip hop",
          "folk", "soundtrack", "lo-fi", "piano instrumental", "orchestral"]
MOODS = ["calm", "dreamy", "melancholic", "upbeat", "energetic", "dark",
         "nostalgic", "peaceful", "dramatic", "hopeful"]


def tag_songs(library):
    """Zero-shot CLAP similarity labels. Scores are relative, not probabilities."""
    from ai_embeddings import _model, embedding_for
    processor, model, torch = _model()
    phrases = [f"A {g} music recording" for g in GENRES] + [
        f"A {m} sounding music recording" for m in MOODS]
    with torch.no_grad():
        outputs = model.get_text_features(**processor(
            text=phrases, padding=True, return_tensors="pt"))
    if not isinstance(outputs, torch.Tensor):
        outputs = getattr(outputs, "pooler_output", outputs[0])
    vecs = outputs.detach().cpu().numpy()
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    result = {}
    for song in library:
        audio = embedding_for(song)
        scores = vecs @ audio
        result[song["id"]] = {
            "genre": GENRES[int(np.argmax(scores[:len(GENRES)]))],
            "mood": MOODS[int(np.argmax(scores[len(GENRES):]))],
        }
    return result


def _feature_matrix(library):
    from recommender import _embedding
    return _embedding(pd.DataFrame([s["features"] for s in library]))


def learned_taste(library, feedback):
    """Regularized logistic classifier for sufficiently diverse explicit ratings."""
    from sklearn.linear_model import LogisticRegression
    rated = [(i, feedback.get(s["id"])) for i, s in enumerate(library)
             if feedback.get(s["id"]) in (-1, 1)]
    if len(rated) < 4 or len({label for _, label in rated}) < 2:
        return None
    x = _feature_matrix(library)
    idx, labels = zip(*rated)
    model = LogisticRegression(C=0.3, max_iter=1000, random_state=17)
    model.fit(x[list(idx)], np.array(labels) == 1)
    result = model.predict_proba(x)[:, list(model.classes_).index(True)]
    return {song["id"]: float(prob) for song, prob in zip(library, result)}


def blended_taste_results(library, selected_id, feedback, limit=10):
    """Rank unrated tracks by a combination of similarity and learned rating patterns."""
    from recommender import recommendations
    data = pd.DataFrame([{"id": s["id"], "title": s["title"], **s["features"]} for s in library])
    base = recommendations(data, selected_id, limit=len(library), feedback=feedback,
                           preference_weight=.55, mode="personalized")
    learned = learned_taste(library, feedback)
    if learned is None:
        return base.head(limit), False
    if base.empty:
        return base, True
    base = base.copy()
    lo, hi = base["rank_score"].min(), base["rank_score"].max()
    relative = (base["rank_score"] - lo) / max(hi - lo, 1e-9)
    base["learned_score"] = [.65 * learned[sid] + .35 * value
                             for sid, value in zip(base["id"], relative)]
    return base.sort_values("learned_score", ascending=False).head(limit), True


def generate_playlist(library, prompt, count, feedback=None, tags=None):
    """CLAP free-text search, taste score and varied selection (maximal marginal relevance)."""
    from smart_features import text_recommendations
    feedback = feedback or {}
    candidates = text_recommendations(library, prompt, limit=len(library))
    ranked = [(s, score) for s, score in candidates if feedback.get(s["id"]) != -1]
    if not ranked:
        return []
    from ai_embeddings import embedding_for
    vectors = {s["id"]: embedding_for(s) for s, _ in ranked}
    taste = learned_taste(library, feedback) or {}
    chosen, remaining = [], ranked[:]
    while remaining and len(chosen) < count:
        def quality(entry):
            song, score = entry
            diversity = (max(float(vectors[song["id"]] @ vectors[x["id"]])
                             for x in chosen) if chosen else 0)
            return .72 * score + .16 * taste.get(song["id"], .5) - .12 * diversity
        next_song, _ = max(remaining, key=quality)
        chosen.append(next_song)
        remaining = [entry for entry in remaining if entry[0]["id"] != next_song["id"]]
    return chosen


def cluster_descriptions(library, ids, tags):
    """Describe a group by dominant predicted mood/genre, not an inferred ground truth."""
    from collections import Counter
    members = [tags[sid] for sid in ids if sid in tags]
    if not members:
        return "Unlabeled sound group"
    mood = Counter(x["mood"] for x in members).most_common(1)[0][0]
    genre = Counter(x["genre"] for x in members).most_common(1)[0][0]
    return f"{mood.title()} · {genre.title()}"


def export_backup():
    """In-memory ZIP with JSON metadata and actual audio; excludes secrets and model caches."""
    from library import connect, AUDIO
    records = {}
    with connect() as con:
        for table in ["songs", "feedback", "similarity_feedback", "discoveries",
                      "playlists", "playlist_songs", "listening_events", "ai_tags"]:
            records[table] = [dict(row) for row in con.execute(f"SELECT * FROM {table}")]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps({"format": "music-recommender-backup-v1",
                                                "tables": records}, ensure_ascii=False))
        for row in records["songs"]:
            filename = row["id"] + row["extension"]
            audio = AUDIO / filename
            if audio.exists():
                z.write(audio, "audio/" + filename)
    return payload.getvalue()


def restore_backup(content):
    """Merge a validated backup without deleting or overwriting existing entries."""
    from library import connect, AUDIO
    if len(content) > 1500 * 1024 * 1024:
        raise ValueError("Backup exceeds the 1.5 GB upload limit.")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if len(z.infolist()) > 2000 or sum(i.file_size for i in z.infolist()) > 2 * 1024**3:
            raise ValueError("Backup too large.")
        metadata = json.loads(z.read("manifest.json"))
        if metadata.get("format") != "music-recommender-backup-v1":
            raise ValueError("Unknown backup format.")
        tables = metadata["tables"]
        required = {"songs", "feedback", "similarity_feedback", "discoveries",
                    "playlists", "playlist_songs", "listening_events"}
        if not required.issubset(tables):
            raise ValueError("Backup is incomplete.")
        incoming = []
        for row in tables["songs"]:
            sid, extension = row["id"], row["extension"]
            if not re.fullmatch(r"[0-9a-f]{64}", sid) or extension not in (".mp3", ".wav", ".flac", ".ogg"):
                raise ValueError("Unsafe filename in backup.")
            data = z.read("audio/" + sid + extension)
            if hashlib.sha256(data).hexdigest() != sid:
                raise ValueError("Audio checksum mismatch.")
            incoming.append((row, data))
        from library import songs
        existing = {song["id"] for song in songs()}
        with connect() as con:
            for row, audio in incoming:
                if row["id"] in existing:
                    continue
                path = AUDIO / (row["id"] + row["extension"])
                path.write_bytes(audio)
                con.execute("INSERT OR IGNORE INTO songs(id,title,artist,extension,features) VALUES (?,?,?,?,?)",
                            (row["id"], row["title"], row.get("artist", ""), row["extension"], row["features"]))
            for row in tables["feedback"]:
                con.execute("INSERT OR IGNORE INTO feedback(song_id,preference) VALUES (?,?)",
                            (row["song_id"], row["preference"]))
            for row in tables["similarity_feedback"]:
                con.execute("INSERT OR IGNORE INTO similarity_feedback(song_a,song_b,rating) VALUES (?,?,?)",
                            (row["song_a"], row["song_b"], row["rating"]))
            for row in tables["discoveries"]:
                con.execute("INSERT OR IGNORE INTO discoveries(artist,track,status,url) VALUES (?,?,?,?)",
                            (row["artist"], row["track"], row["status"], row.get("url", "")))
            mapping = {}
            for row in tables["playlists"]:
                name = row["name"]
                con.execute("INSERT OR IGNORE INTO playlists(name) VALUES (?)", (name,))
                new_id = con.execute("SELECT id FROM playlists WHERE name=?", (name,)).fetchone()["id"]
                mapping[row["id"]] = new_id
            valid_ids = {row["id"] for row in con.execute("SELECT id FROM songs")}
            for row in tables["playlist_songs"]:
                if row["song_id"] in valid_ids and row["playlist_id"] in mapping:
                    con.execute("INSERT OR IGNORE INTO playlist_songs(playlist_id,song_id) VALUES (?,?)",
                                (mapping[row["playlist_id"]], row["song_id"]))
            for row in tables.get("ai_tags", []):
                if row["song_id"] in valid_ids:
                    con.execute("INSERT OR IGNORE INTO ai_tags(song_id,genre,mood) VALUES (?,?,?)",
                                (row["song_id"], row["genre"], row["mood"]))
            # Listening events are intentionally omitted to avoid doubling counts.
    return sum(row["id"] not in existing for row, _ in incoming)
