"""Evaluate audio similarity without using ratings in prediction scores."""
import numpy as np
import pandas as pd
from recommender import _embedding


def rank_correlation(scores, ratings):
    """Spearman rank correlation with average ranks for ties."""
    if len(scores) < 3:
        return None
    x = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(ratings).rank(method="average").to_numpy(dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(library, pair_ratings, embeddings, hybrid_weight=0.25):
    """Return per-pair scores and model agreement with human similarity ratings.

    All predictions are independent of the labels being evaluated:
    no pair-rating bonus and no personalized feedback in this benchmark.
    """
    ids = [s["id"] for s in library]
    index = {sid: i for i, sid in enumerate(ids)}
    if len(library) < 2:
        return pd.DataFrame(), pd.DataFrame()
    acoustic_z = _embedding(pd.DataFrame([s["features"] for s in library]))
    emb = np.stack([embeddings[sid] for sid in ids])
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    # Use one constant acoustic distance scale for all evaluated pairs.
    diff = acoustic_z[:, None, :] - acoustic_z[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nonzero = dists[np.triu_indices(len(ids), k=1)]
    nonzero = nonzero[nonzero > 1e-10]
    scale = max(float(np.median(nonzero)), 1.0) if len(nonzero) else 1.0
    alpha = float(np.clip(hybrid_weight, 0, 1))
    rows = []
    for (a, b), rating in pair_ratings.items():
        if a not in index or b not in index or a == b:
            continue
        i, j = index[a], index[b]
        cosine = float(np.clip(emb[i] @ emb[j], -1, 1))
        clap = (cosine + 1) / 2
        acoustic = float(np.exp(-dists[i, j] / scale))
        hybrid = (1 - alpha) * clap + alpha * acoustic
        rows.append({
            "Song A": library[i]["title"], "Song B": library[j]["title"],
            "Human rating": int(rating), "CLAP only": clap,
            "Acoustic only": acoustic, "Hybrid": hybrid
        })
    data = pd.DataFrame(rows)
    summary = []
    if not data.empty:
        for method in ("CLAP only", "Acoustic only", "Hybrid"):
            corr = rank_correlation(data[method], data["Human rating"])
            summary.append({"Method": method,
                            "Spearman correlation": corr,
                            "Pairs rated": len(data)})
    return data, pd.DataFrame(summary)
