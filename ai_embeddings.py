"""Optional pretrained CLAP audio embeddings and cosine similarity.

Install separately: python -m pip install -r requirements-ai.txt
The model downloads on first use. Computation runs on CPU by default.
"""
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

from library import ROOT

MODEL_ID = "laion/clap-htsat-unfused"
EMBED_DIR = ROOT / "embeddings"


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model, torch


def embedding_for(song):
    """Average beginning, middle and end CLAP embeddings; cache v2."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    target = EMBED_DIR / (song["id"] + "_v2.npy")
    if target.exists():
        return np.load(target, allow_pickle=False)
    processor, model, torch = _model()
    audio, sr = librosa.load(str(song["path"]), sr=48000, mono=True, duration=180)
    if len(audio) < sr:
        raise ValueError("Audio too short")
    duration = len(audio) / sr
    starts = sorted(set(round(x, 2) for x in
                        (0, max(0, (duration - 10) / 2), max(0, duration - 10))))
    vectors = []
    for start in starts:
        clip = audio[int(start * sr):int((start + 10) * sr)]
        inputs = processor(audios=[clip], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            vec = model.get_audio_features(**inputs)
        if not isinstance(vec, torch.Tensor):
            vec = getattr(vec, "pooler_output", vec[0])
        arr = np.asarray(vec.detach().cpu().numpy()[0], dtype="float32")
        arr /= max(float(np.linalg.norm(arr)), 1e-12)
        vectors.append(arr)
    mean = np.mean(vectors, axis=0).astype("float32")
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    np.save(target, mean)
    return mean


def segment_embeddings_for(song):
    """Retrieve/cache a 3 x D matrix of beginning/middle/end embeddings."""
    target = EMBED_DIR / (song["id"] + "_sections_v1.npy")
    if target.exists():
        return np.load(target, allow_pickle=False)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    processor, model, torch = _model()
    audio, sr = librosa.load(str(song["path"]), sr=48000, mono=True, duration=180)
    if len(audio) < sr:
        raise ValueError("Audio too short")
    duration = len(audio) / sr
    starts = sorted(set(round(x, 2) for x in
                        (0, max(0, (duration - 10) / 2), max(0, duration - 10))))
    vectors = []
    for start in starts:
        clip = audio[int(start * sr):int((start + 10) * sr)]
        inputs = processor(audios=[clip], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            vec = model.get_audio_features(**inputs)
        if not isinstance(vec, torch.Tensor):
            vec = getattr(vec, "pooler_output", vec[0])
        arr = np.asarray(vec.detach().cpu().numpy()[0], dtype="float32")
        arr /= max(float(np.linalg.norm(arr)), 1e-12)
        vectors.append(arr)
    result = np.stack(vectors)
    np.save(target, result)
    return result


def ai_recommendations(library, selected_id, feedback=None, mode="similar",
                       weight=0.7, limit=10, acoustic_weight=0.25, pair_ratings=None):
    """Rank with CLAP embedding cosine similarity, optionally user ratings."""
    feedback = feedback or {}
    vecs = np.stack([embedding_for(song) for song in library])
    ids = [song["id"] for song in library]
    idx = ids.index(selected_id)
    similarity = vecs @ vecs[idx]
    from recommender import _embedding
    import pandas as pd
    features = pd.DataFrame([s["features"] for s in library])
    z = _embedding(features)
    dist = np.linalg.norm(z - z[idx], axis=1)
    nonzero = dist[dist > 1e-8]
    scale = max(float(np.median(nonzero)), 1.0) if len(nonzero) else 1.0
    acoustic = np.exp(-dist / scale)
    blend = float(np.clip(acoustic_weight, 0, 1))
    hybrid = (1 - blend) * ((similarity + 1) / 2) + blend * acoustic
    pair_ratings = pair_ratings or {}
    bonus = np.array([(pair_ratings.get(tuple(sorted((selected_id, sid))), 0) - 3) / 2
                       if pair_ratings.get(tuple(sorted((selected_id, sid))), 0) else 0
                       for sid in ids])
    hybrid += 0.12 * bonus
    if mode == "personalized":
        likes = [i for i, sid in enumerate(ids) if feedback.get(sid) == 1]
        dislikes = [i for i, sid in enumerate(ids) if feedback.get(sid) == -1]
        if likes or dislikes:
            taste = np.zeros(len(ids), dtype=float)
            if likes:
                taste += np.mean(vecs @ vecs[likes].T, axis=1)
            if dislikes:
                taste -= np.mean(vecs @ vecs[dislikes].T, axis=1)
            scores = (1 - weight) * hybrid + weight * ((taste + 1) / 2)
            allowed = [i for i, sid in enumerate(ids)
                       if i != idx and feedback.get(sid, 0) == 0]
        else:
            scores = hybrid.copy()
            allowed = [i for i in range(len(ids)) if i != idx]
    else:
        scores = hybrid.copy()
        allowed = [i for i in range(len(ids)) if i != idx]
    allowed.sort(key=lambda i: float(scores[i]), reverse=True)
    return [{"id": ids[i], "title": library[i]["title"],
             "cosine_similarity": float(similarity[i]),
             "hybrid_score": float(hybrid[i]),
             "rank_score": float(scores[i])} for i in allowed[:limit]]
