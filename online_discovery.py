"""Discover external tracks via Last.fm; link to Spotify searches.

No streamed audio is downloaded or analyzed.
"""
import json
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_URL = "https://ws.audioscrobbler.com/2.0/"


def spotify_search_url(artist, track):
    query = f"{artist} {track}".strip()
    return "https://open.spotify.com/search/" + quote(query, safe="")


def find_similar_tracks(artist, track, api_key, limit=15):
    """Return ranked Last.fm matches with links; the score is not CLAP."""
    artist, track = artist.strip(), track.strip()
    if not (artist and track and api_key.strip()):
        raise ValueError("Enter an artist, a track title and a Last.fm API key.")
    params = urlencode({
        "method": "track.getsimilar", "artist": artist, "track": track,
        "api_key": api_key.strip(), "format": "json",
        "autocorrect": 1, "limit": max(1, min(int(limit), 50)),
    })
    request = Request(API_URL + "?" + params,
                      headers={"User-Agent": "MusicRecommender/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Could not reach Last.fm; check your connection and API key.") from exc
    if "error" in payload:
        raise RuntimeError("Last.fm: " + str(payload.get("message", "Request failed")))
    candidates = payload.get("similartracks", {}).get("track", [])
    if isinstance(candidates, dict):
        candidates = [candidates]
    found = []
    seen = set()
    for candidate in candidates:
        artist_value = candidate.get("artist", {})
        other_artist = (artist_value.get("name") if isinstance(artist_value, dict)
                        else artist_value) or ""
        other_track = candidate.get("name", "")
        identity = (other_artist.casefold(), other_track.casefold())
        if not all(identity) or identity in seen or identity == (artist.casefold(), track.casefold()):
            continue
        seen.add(identity)
        found.append({
            "artist": other_artist,
            "track": other_track,
            "lastfm_match": float(candidate.get("match") or 0),
            "lastfm_url": candidate.get("url", ""),
            "spotify_url": spotify_search_url(other_artist, other_track),
        })
    return sorted(found, key=lambda x: x["lastfm_match"], reverse=True)
