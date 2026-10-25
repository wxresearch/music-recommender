import unittest
from unittest.mock import patch
import numpy as np
from explore import parse_title, taste_summary, music_map, mood_search


class ExplorationTests(unittest.TestCase):
    def test_filename_title_artist(self):
        self.assertEqual(parse_title("2月25日 - Rzira.mp3"), ("Rzira", "2月25日"))

    def test_taste_summary(self):
        tracks = [
            {"id": "a", "features": {"tempo": 100, "rms": .2, "brightness": 50, "onset_strength": 1}},
            {"id": "b", "features": {"tempo": 130, "rms": .6, "brightness": 100, "onset_strength": 2}},
        ]
        result = taste_summary(tracks, {"a": 1, "b": -1})
        self.assertEqual(result.loc["tempo", "Liked"], 100)
        self.assertEqual(result.loc["tempo", "Disliked"], 130)

    def test_map(self):
        features = []
        for i in range(3):
            d = {"id": str(i), "title": str(i)}
            from recommender import BASE_FEATURES
            d["features"] = {name: i + j for j, name in enumerate(BASE_FEATURES)}
            d["features"].update({f"mfcc_{j}": i + j for j in range(1, 14)})
            d["features"].update({f"chroma_{j}": i + j for j in range(12)})
            features.append(d)
        points = music_map(features)
        self.assertEqual(points.shape, (3, 4))
        self.assertTrue(np.isfinite(points[["x", "y"]].to_numpy()).all())

    def test_mood_source(self):
        reply = {"tracks": {"track": [{"name": "A", "artist": {"name": "B"}}]}}
        with patch("online_discovery._request", return_value=reply):
            result = mood_search("Peaceful ambient", "dummy")
        self.assertEqual(result[0]["artist"], "B")
        self.assertIn("spotify.com", result[0]["spotify_url"])


if __name__ == "__main__":
    unittest.main()
