"""Explainable acoustic similarity with optional like/dislike personalization."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ["tempo", "rms", "brightness", "bandwidth", "rolloff",
                 "zcr", "onset_strength"]


def recommendations(data: pd.DataFrame, selected_id: str, limit=10,
                    feedback=None, preference_weight=0.35):
    if len(data) < 2:
        return pd.DataFrame()
    cols = BASE_FEATURES + [f"mfcc_{i}" for i in range(1, 14)] + [
        f"chroma_{i}" for i in range(12)]
    X = data[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    z = StandardScaler().fit_transform(X)
    # Normalize groups to prevent 13 MFCC dimensions drowning out all else.
    group_weights = {"tempo": 1.4, "rms": 1.0, "brightness": 1.0,
                     "bandwidth": 0.7, "rolloff": 0.7, "zcr": 0.8,
                     "onset_strength": 1.1}
    for j, name in enumerate(cols):
        if name.startswith("mfcc_"):
            z[:, j] *= 1.8 / np.sqrt(13)
        elif name.startswith("chroma_"):
            z[:, j] *= 1.2 / np.sqrt(12)
        else:
            z[:, j] *= group_weights[name]
    ids = data["id"].tolist()
    selected_idx = ids.index(selected_id)
    acoustic = np.linalg.norm(z - z[selected_idx], axis=1)
    scores = acoustic.copy()
    feedback = feedback or {}
    liked = [i for i, sid in enumerate(ids) if feedback.get(sid) == 1]
    disliked = [i for i, sid in enumerate(ids) if feedback.get(sid) == -1]
    # Favor recordings close to liked examples, penalize disliked examples.
    # Small weight and transparent scores: heuristic, not trained prediction.
    if liked:
        scores += preference_weight * np.min(
            np.linalg.norm(z[:, None, :] - z[None, liked, :], axis=2), axis=1)
    if disliked:
        scores -= preference_weight * np.min(
            np.linalg.norm(z[:, None, :] - z[None, disliked, :], axis=2), axis=1)
    candidates = [i for i in np.argsort(scores) if i != selected_idx][:limit]
    out = data.iloc[candidates][["id", "title", "tempo", "rms", "brightness"]].copy()
    out["acoustic_distance"] = acoustic[candidates]
    out["rank_score"] = scores[candidates]
    return out.reset_index(drop=True)
