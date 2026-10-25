"""Offline checks for ranking and cluster descriptions."""
import unittest
import numpy as np
from intelligence import learned_taste, cluster_descriptions


def track(n):
    features = {"tempo": 80 + n*13, "rms": .1 + n*.04, "brightness": 800+n*120,
                "bandwidth": 1000 + n*50, "rolloff": 1800 + n*100, "zcr": .03+n*.01,
                "onset_strength": .4+n*.04}
    features.update({f"mfcc_{i}": float(n+i)/10 for i in range(1,14)})
    features.update({f"chroma_{i}": float(n+i)/100 for i in range(12)})
    return {"id": str(n), "title": str(n), "artist": "", "features": features}


class IntelligenceTests(unittest.TestCase):
    def test_taste_needs_both_classes(self):
        library = [track(i) for i in range(5)]
        self.assertIsNone(learned_taste(library, {"0":1,"1":1,"2":1,"3":1}))

    def test_taste_predicts(self):
        library = [track(i) for i in range(6)]
        pred = learned_taste(library, {"0":-1,"1":-1,"3":1,"4":1})
        self.assertEqual(set(pred), {s["id"] for s in library})
        self.assertTrue(all(0 <= v <= 1 for v in pred.values()))

    def test_clusters(self):
        tags = {"one":{"genre":"ambient","mood":"dreamy"},
                "two":{"genre":"ambient","mood":"dreamy"}}
        self.assertEqual(cluster_descriptions([], ["one", "two"], tags), "Dreamy · Ambient")


if __name__ == "__main__":
    unittest.main()
