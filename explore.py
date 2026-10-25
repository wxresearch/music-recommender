"""Additional music discovery, visualisation and Spotify helpers."""
from pathlib import Path
import base64
import io
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_title(filename):
    stem = Path(filename).stem
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", stem


def cover_for(song):
    """Read embedded cover art from local files without uploading audio."""
    try:
        from mutagen import File
        tags = File(str(song["path"]))
        if tags is None:
            return None
        artwork = None
        if hasattr(tags, "pictures") and tags.pictures:
            artwork = tags.pictures[0].data
        elif tags.tags:
            for key in tags.tags.keys():
                value = tags.tags[key]
                if str(key).startswith("APIC"):
                    artwork = value.data
                    break
                if str(key).lower() == "covr":
                    artwork = bytes(value[0])
                    break
        return artwork
    except Exception:
        return None


def taste_summary(library, feedback):
    metrics = ["tempo", "rms", "brightness", "onset_strength"]
    rows = []
    for song in library:
        label = feedback.get(song["id"], 0)
        if label:
            rows.append({"Rating": "Liked" if label == 1 else "Disliked",
                         **{key: float(song["features"].get(key, 0)) for key in metrics}})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("Rating")[metrics].mean().T


def music_map(library, vectors=None):
    """Project CLAP vectors (or acoustic features) into a 2D PCA map."""
    if vectors:
        ids = [song["id"] for song in library]
        X = np.stack([vectors[sid] for sid in ids])
    else:
        from recommender import _embedding
        X = _embedding(pd.DataFrame([song["features"] for song in library]))
    coords = PCA(n_components=2).fit_transform(X)
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                         "title": [s["title"] for s in library],
                         "id": [s["id"] for s in library]})


def mood_search(mood, api_key, limit=20):
    """Last.fm tag search; not an audio-embedding similarity score."""
    from online_discovery import _request, _items, spotify_search_url
    tags = {"Dreamy piano": "piano", "Melancholic": "melancholic",
            "Energetic electronic": "electronic", "Peaceful ambient": "ambient",
            "Lo-fi": "lo-fi", "Instrumental": "instrumental"}
    tag = tags.get(mood, mood)
    data = _request("tag.gettoptracks", api_key, tag=tag, limit=limit)
    found = []
    for rec in _items(data, "tracks", "track"):
        artist = rec.get("artist", {})
        artist = artist.get("name", "") if isinstance(artist, dict) else str(artist)
        title = rec.get("name", "")
        if artist and title:
            found.append({"artist": artist, "track": title,
                          "spotify_url": spotify_search_url(artist, title),
                          "source": "Mood tag: " + tag, "lastfm_match": None,
                          "lastfm_url": rec.get("url", "")})
    return found


def closest_section(reference, candidate):
    """Find the best pair of 10-second CLAP segments among up to three each."""
    from ai_embeddings import segment_embeddings_for
    va = segment_embeddings_for(reference)
    vb = segment_embeddings_for(candidate)
    scores = va @ vb.T
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    return int(i), int(j), float(scores[i, j])


def spotify_export(track_pairs, client_id, client_secret, redirect_uri,
                   playlist_name="Music Recommender Discoveries"):
    """Explicit user-initiated Spotify OAuth + playlist creation via spotipy."""
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope="playlist-modify-private",
                        open_browser=True, cache_path=str(Path.home() / ".music_recommender_spotify_token"))
    client = spotipy.Spotify(auth_manager=oauth)
    me = client.current_user()
    uris, missing = [], []
    for artist, title in track_pairs:
        result = client.search(q=f'track:"{title}" artist:"{artist}"',
                               type="track", limit=1)
        items = result.get("tracks", {}).get("items", [])
        if items:
            uris.append(items[0]["uri"])
        else:
            missing.append(f"{artist} — {title}")
    if not uris:
        return None, missing
    playlist = client.user_playlist_create(me["id"], playlist_name, public=False)
    for i in range(0, len(uris), 100):
        client.playlist_add_items(playlist["id"], uris[i:i+100])
    return playlist["external_urls"]["spotify"], missing
