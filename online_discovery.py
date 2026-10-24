"""Discover music through Last.fm, with Spotify search links."""
import json
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_URL = "https://ws.audioscrobbler.com/2.0/"


def spotify_search_url(artist, track):
    return "https://open.spotify.com/search/" + quote(f"{artist} {track}".strip(), safe="")


def _request(method, api_key, **params):
    query = urlencode({"method": method, "api_key": api_key, "format": "json", **params})
    req = Request(API_URL + "?" + query, headers={"User-Agent": "MusicRecommender/1.1"})
    try:
        with urlopen(req, timeout=12) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Unable to contact Last.fm.") from exc
    if "error" in result:
        raise RuntimeError("Last.fm: " + str(result.get("message", "Request failed")))
    return result


def _items(payload, *keys):
    obj = payload
    for key in keys:
        obj = obj.get(key, {}) if isinstance(obj, dict) else {}
    if isinstance(obj, list):
        return obj
    return [obj] if isinstance(obj, dict) and obj else []


def find_similar_tracks(artist, track, api_key, limit=15):
    """Discover from similar tracks, falling back to tags and related artists."""
    artist, track, api_key = artist.strip(), track.strip(), api_key.strip()
    if not (artist and track and api_key):
        raise ValueError("Enter artist, track and Last.fm API key.")
    limit = max(1, min(int(limit), 50))
    found, seen = [], set()

    def add(candidates, source, score_key="match"):
        for candidate in candidates:
            info = candidate.get("artist", {})
            name = (info.get("name") if isinstance(info, dict) else info) or ""
            title = candidate.get("name", "")
            identity = (name.strip().casefold(), title.strip().casefold())
            if not all(identity) or identity in seen or identity == (artist.casefold(), track.casefold()):
                continue
            seen.add(identity)
            found.append({
                "artist": name, "track": title,
                "lastfm_match": float(candidate.get(score_key) or 0) if score_key else None,
                "source": source,
                "lastfm_url": candidate.get("url", ""),
                "spotify_url": spotify_search_url(name, title),
            })
            if len(found) >= limit:
                break

    direct = _request("track.getsimilar", api_key, artist=artist, track=track,
                      autocorrect=1, limit=limit)
    add(_items(direct, "similartracks", "track"), "Similar track")
    if len(found) >= limit:
        return found

    # Tag-based fallback works when an obscure song has no similar-track graph.
    try:
        tags = _request("track.gettoptags", api_key, artist=artist, track=track,
                        autocorrect=1)
        for tag in _items(tags, "toptags", "tag")[:3]:
            label = tag.get("name", "")
            if not label or label.lower() in ("seen live", "favorites", "favourite", "my favorite"):
                continue
            tracks = _request("tag.gettoptracks", api_key, tag=label, limit=limit)
            add(_items(tracks, "tracks", "track"), f"Tag: {label}", None)
            if len(found) >= limit:
                return found
    except RuntimeError:
        pass  # A missing tag should not disable other discovery methods.

    # Also look for tracks from related artists.
    try:
        related = _request("artist.getsimilar", api_key, artist=artist,
                           autocorrect=1, limit=5)
        for other in _items(related, "similarartists", "artist")[:4]:
            name = other.get("name", "")
            if not name:
                continue
            tracks = _request("artist.gettoptracks", api_key, artist=name, limit=4)
            add(_items(tracks, "toptracks", "track"), "Related artist", None)
            if len(found) >= limit:
                return found
    except RuntimeError:
        pass
    return found
