import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from smart_features import explain_match, surprise_me, duplicate_candidates


def song(sid, tempo, rms):
    features = {"tempo": tempo, "rms": rms, "brightness": 1200 + tempo,
                "bandwidth": 1000, "rolloff": 3000, "zcr": .05,
                "onset_strength": .4}
    features.update({f"mfcc_{i}": float(i + tempo/100) for i in range(1,14)})
    features.update({f"chroma_{i}": .1 + i/100 for i in range(12)})
    features["duration"] = 100
    return {"id": sid, "title": sid, "artist": "", "features": features}


class FeaturesTests(unittest.TestCase):
    def test_explanations(self):
        details = explain_match(song("a", 100, .2), song("b", 102, .21))
        self.assertTrue(any("Tempo" in x for x in details))

    def test_surprise_excludes_seed(self):
        tracks = [song("a", 100, .2), song("b", 140, .4), song("c", 90, .15)]
        found = surprise_me(tracks, "a", {}, random_seed=7)
        self.assertNotIn("a", [x["id"] for x in found])

    def test_duplicate_candidates_does_not_delete(self):
        tracks = [song("a", 100, .2), song("b", 100, .2), song("c", 140, .4)]
        found = duplicate_candidates(tracks)
        self.assertTrue(any(a["id"] == "a" and b["id"] == "b" for a,b,_ in found))


if __name__ == "__main__":
    unittest.main()
