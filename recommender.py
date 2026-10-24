import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def recommendations(data: pd.DataFrame, selected_id: str, limit=5):
    """Return nearest songs; distances are not preference probabilities."""
    if len(data) < 2:
        return pd.DataFrame()
    feature_cols = [c for c in data.columns if c.startswith(("mfcc_", "chroma_"))]
    feature_cols += ["tempo", "rms", "brightness", "bandwidth", "rolloff", "zcr", "onset_strength"]
    features = data[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    values = StandardScaler().fit_transform(features)
    for prefix, weight in [("mfcc_", 1 / np.sqrt(13)), ("chroma_", 1 / np.sqrt(12))]:
        idxs = [i for i, c in enumerate(feature_cols) if c.startswith(prefix)]
        values[:, idxs] *= weight
    idx = data.index[data["id"] == selected_id][0]
    distances = np.linalg.norm(values - values[idx], axis=1)
    order = np.argsort(distances)
    order = [i for i in order if i != idx][:limit]
    result = data.iloc[order][["id", "title", "duration", "tempo", "brightness"]].copy()
    result["distance"] = distances[order]
    return result.reset_index(drop=True)
