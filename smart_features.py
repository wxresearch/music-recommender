"""Local recommendation explanations, diversity, duplicate candidates and text-to-CLAP search."""
import numpy as np
import pandas as pd


def explain_match(seed, candidate):
    """Honest feature comparisons, not causal attribution of an embedding model."""
    keys = [("tempo", "tempo"), ("rms", "energy"), ("brightness", "brightness"),
            ("onset_strength", "rhythmic activity")]
    notes = []
    for key, label in keys:
        a, b = float(seed["features"].get(key, 0)), float(candidate["features"].get(key, 0))
        if key == "tempo":
            notes.append(f"Tempo: {a:.0f} vs {b:.0f} BPM")
        elif max(abs(a), abs(b)) > 0 and abs(a-b)/max(abs(a), abs(b)) < 0.22:
            notes.append(f"Similar {label}")
    return notes[:3] or ["This track has a nearby overall acoustic-feature profile."]


def surprise_me(library, seed_id, feedback, novelty=0.5, count=8, random_seed=None):
    """Randomized diversity among unrated candidates, balanced against similarity."""
    from recommender import _embedding
    if len(library) < 2:
        return []
    rng = np.random.default_rng(random_seed)
    ids = [s["id"] for s in library]
    x = _embedding(pd.DataFrame([s["features"] for s in library]))
    ref = x[ids.index(seed_id)]
    d = np.linalg.norm(x-ref, axis=1)
    scale = max(float(np.median(d[d > 0])) if np.any(d > 0) else 1, 1)
    affinity = np.exp(-d/scale)
    pool = [i for i, sid in enumerate(ids) if sid != seed_id and feedback.get(sid, 0) == 0]
    if not pool:
        pool = [i for i, sid in enumerate(ids) if sid != seed_id]
    novelty = float(np.clip(novelty, 0, 1))
    # At high novelty, lift less similar candidates while preserving randomness.
    scores = (1-novelty)*affinity + novelty*(1-affinity) + rng.uniform(0, .20, len(ids))
    ordered = sorted(pool, key=lambda i: -scores[i])
    return [library[i] for i in ordered[:count]]


def duplicate_candidates(library, threshold=0.94):
    """Screen candidates using cached acoustic descriptors; review before deletion."""
    from recommender import _embedding
    if len(library) < 2:
        return []
    z = _embedding(pd.DataFrame([s["features"] for s in library]))
    distances = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    nonzero = distances[np.triu_indices(len(library), 1)]
    if len(nonzero) == 0:
        return []
    # Threshold controls closeness in the normalized acoustic space.
    cutoff = max(.30, float(np.median(nonzero)) * (1 - threshold) * 3)
    found = []
    for i in range(len(library)):
        for j in range(i+1, len(library)):
            a, b = library[i], library[j]
            duration_a, duration_b = a["features"].get("duration", 0), b["features"].get("duration", 0)
            if abs(duration_a-duration_b) > 2:
                continue
            if distances[i, j] <= cutoff:
                found.append((a, b, round(float(distances[i,j]), 3)))
    return sorted(found, key=lambda triple: triple[2])


def text_recommendations(library, description, limit=10):
    """Score local audio with the CLAP text encoder; never scores online tracks."""
    from ai_embeddings import _model, embedding_for
    processor, model, torch = _model()
    text_inputs = processor(text=[description], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**text_inputs)
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "pooler_output", out[0])
    text_vector = np.asarray(out.detach().cpu().numpy()[0], dtype=float)
    text_vector /= max(np.linalg.norm(text_vector), 1e-12)
    results = []
    for song in library:
        audio_vector = embedding_for(song)
        results.append((song, float(np.dot(text_vector, audio_vector))))
    return sorted(results, key=lambda pair: pair[1], reverse=True)[:limit]
