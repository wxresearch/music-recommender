"""Acoustic nearest-neighbor ranking and preference-based music discovery."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ["tempo", "rms", "brightness", "bandwidth",
                 "rolloff", "zcr", "onset_strength"]


def _embedding(data):
    cols = BASE_FEATURES + [f"mfcc_{i}" for i in range(1, 14)] + [
        f"chroma_{i}" for i in range(12)]
    X = data[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)
    z = StandardScaler().fit_transform(X)
    weights = {
        "tempo": 1.4, "rms": 1.0, "brightness": 1.0,
        "bandwidth": 0.7, "rolloff": 0.7,
        "zcr": 0.8, "onset_strength": 1.1,
    }
    for i, col in enumerate(cols):
        if col.startswith("mfcc_"):
            z[:, i] *= 1.8 / np.sqrt(13)
        elif col.startswith("chroma_"):
            z[:, i] *= 1.2 / np.sqrt(12)
        else:
            z[:, i] *= weights[col]
    return z


def recommendations(data: pd.DataFrame, selected_id: str, limit=10,
                    feedback=None, preference_weight=0.35, mode="similar"):
    """Rank by acoustic distance OR by a global user taste profile.

    Similar mode: ascending acoustic distance from selected song.
    For-you mode: higher score is better; positive labels act as
    explicit preference signals and unrated candidates are preferred.
    """
    if len(data) < 2:
        return pd.DataFrame()
    ids = data["id"].tolist()
    if selected_id not in ids:
        raise ValueError("Selected song is not in the library")
    z = _embedding(data)
    idx = ids.index(selected_id)
    acoustic = np.linalg.norm(z - z[idx], axis=1)
    feedback = feedback or {}
    out = data[["id", "title", "tempo", "rms", "brightness"]].copy()
    out["acoustic_distance"] = acoustic

    if mode == "similar":
        out["rank_score"] = acoustic
        return (out[out["id"] != selected_id]
                .sort_values("acoustic_distance")
                .head(limit).reset_index(drop=True))

    if mode != "personalized":
        raise ValueError("Unknown recommendation mode")

    liked = [i for i, sid in enumerate(ids) if feedback.get(sid) == 1]
    disliked = [i for i, sid in enumerate(ids) if feedback.get(sid) == -1]
    if not liked and not disliked:
        # No learned preference yet: reasonable, deterministic fallback.
        return recommendations(data, selected_id, limit, mode="similar")

    # A global taste direction computed from previously rated songs.
    # Positive ratings raise scores; negative ratings lower scores.
    # Use the same feature space as acoustic matching.
    scale = max(float(np.median(np.linalg.norm(z, axis=1))), 1.0)
    similarity_to_likes = (np.mean(np.exp(
        -np.linalg.norm(z[:, None, :] - z[None, liked, :], axis=2) / scale),
        axis=1) if liked else np.zeros(len(ids)))
    similarity_to_dislikes = (np.mean(np.exp(
        -np.linalg.norm(z[:, None, :] - z[None, disliked, :], axis=2) / scale),
        axis=1) if disliked else np.zeros(len(ids)))
    # 0 -> proximity to the selected song; 1 -> rating-derived preference.
    acoustic_affinity = np.exp(-acoustic / scale)
    taste = similarity_to_likes - similarity_to_dislikes
    weight = float(np.clip(preference_weight, 0.0, 1.0))
    score = (1 - weight) * acoustic_affinity + weight * taste
    out["rank_score"] = score
    # Recommend unexplored/unrated songs before resurfacing favorites.
    candidate_mask = out["id"].map(lambda sid: feedback.get(sid, 0) == 0)
    candidate_mask &= out["id"] != selected_id
    candidates = out[candidate_mask]
    if candidates.empty:
        # Explicit empty result avoids recommending already-liked songs
        # as though they were new discoveries.
        return candidates.reset_index(drop=True)
    return candidates.sort_values("rank_score", ascending=False).head(limit).reset_index(drop=True)
