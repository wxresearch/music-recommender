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
    """Compute once per local song ID and reuse from disk."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    target = EMBED_DIR / (song["id"] + ".npy")
    if target.exists():
        return np.load(target)
    processor, model, torch = _model()
    # CLAP accepts 48 kHz audio; keep a 10-second segment for quick local use.
    audio, _ = librosa.load(str(song["path"]), sr=48000, mono=True, duration=10)
    if len(audio) < 48000:
        raise ValueError("Audio too short for AI embedding")
    inputs = processor(audios=[audio], sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        vec = model.get_audio_features(**inputs)
    # Transformers versions may expose a tensor or a model output.
    if not isinstance(vec, torch.Tensor):
        vec = getattr(vec, "pooler_output", vec[0])
    arr = np.asarray(vec.detach().cpu().numpy()[0], dtype="float32")
    arr /= max(float(np.linalg.norm(arr)), 1e-12)
    np.save(target, arr)
    return arr


def ai_recommendations(library, selected_id, feedback=None, mode="similar",
                       weight=0.7, limit=10):
    """Rank with CLAP embedding cosine similarity, optionally user ratings."""
    feedback = feedback or {}
    vecs = np.stack([embedding_for(song) for song in library])
    ids = [song["id"] for song in library]
    idx = ids.index(selected_id)
    similarity = vecs @ vecs[idx]
    if mode == "personalized":
        likes = [i for i, sid in enumerate(ids) if feedback.get(sid) == 1]
        dislikes = [i for i, sid in enumerate(ids) if feedback.get(sid) == -1]
        if likes or dislikes:
            taste = np.zeros(len(ids), dtype=float)
            if likes:
                taste += np.mean(vecs @ vecs[likes].T, axis=1)
            if dislikes:
                taste -= np.mean(vecs @ vecs[dislikes].T, axis=1)
            scores = (1 - weight) * similarity + weight * taste
            allowed = [i for i, sid in enumerate(ids)
                       if i != idx and feedback.get(sid, 0) == 0]
        else:
            scores = similarity.copy()
            allowed = [i for i in range(len(ids)) if i != idx]
    else:
        scores = similarity.copy()
        allowed = [i for i in range(len(ids)) if i != idx]
    allowed.sort(key=lambda i: float(scores[i]), reverse=True)
    return [{"id": ids[i], "title": library[i]["title"],
             "cosine_similarity": float(similarity[i]),
             "rank_score": float(scores[i])} for i in allowed[:limit]]
