"""Offline tests for external discovery links and API response parsing."""
import io
import json
import unittest
from unittest.mock import patch
from online_discovery import find_similar_tracks, spotify_search_url


class DiscoveryTests(unittest.TestCase):
    def test_spotify_search_unicode(self):
        url = spotify_search_url("艺术家", "Song & Dance")
        self.assertTrue(url.startswith("https://open.spotify.com/search/"))
        self.assertIn("%26", url)

    def test_results_sorted_and_deduplicated(self):
        payload = {"similartracks": {"track": [
            {"artist": {"name": "Artist B"}, "name": "Track B", "match": "0.6"},
            {"artist": {"name": "Artist A"}, "name": "Track A", "match": "0.9"},
            {"artist": {"name": "Artist B"}, "name": "Track B", "match": "0.6"},
        ]}}
        class Response:
            def __enter__(self):
                return io.BytesIO(json.dumps(payload).encode())
            def __exit__(self, *args):
                return False
        with patch("online_discovery.urlopen", return_value=Response()):
            results = find_similar_tracks("Seed", "Song", "test-key")
        self.assertEqual([r["track"] for r in results], ["Track A", "Track B"])

    def test_missing_key(self):
        with self.assertRaises(ValueError):
            find_similar_tracks("A", "B", "")


if __name__ == "__main__":
    unittest.main()
