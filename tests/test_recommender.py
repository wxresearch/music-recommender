"""Regression tests for ranking behavior. Run: python -m unittest discover -s tests"""
import unittest
import numpy as np
import pandas as pd
from recommender import recommendations, BASE_FEATURES


def fixtures():
    data = []
    for idx in range(4):
        row = {"id": str(idx), "title": f"Song {idx}"}
        for j, col in enumerate(BASE_FEATURES):
            row[col] = float(idx * 2 + j / 7)
        for j in range(1, 14):
            row[f"mfcc_{j}"] = float(idx + j / 10)
        for j in range(12):
            row[f"chroma_{j}"] = float(idx * 0.25 + j / 10)
        data.append(row)
    return pd.DataFrame(data)


class RecommendationTests(unittest.TestCase):
    def test_similarity_excludes_selected(self):
        results = recommendations(fixtures(), "0", mode="similar")
        self.assertEqual(len(results), 3)
        self.assertNotIn("0", results["id"].tolist())
        self.assertTrue(np.all(np.diff(results["acoustic_distance"]) >= 0))

    def test_personalized_excludes_rated(self):
        results = recommendations(fixtures(), "0", feedback={"1": 1, "2": -1},
                                  mode="personalized")
        self.assertEqual(results["id"].tolist(), ["3"])

    def test_personalized_with_all_rated_returns_empty(self):
        results = recommendations(fixtures(), "0",
                                  feedback={"1": 1, "2": 1, "3": -1},
                                  mode="personalized")
        self.assertTrue(results.empty)

    def test_no_feedback_fallback(self):
        a = recommendations(fixtures(), "0", mode="personalized")
        b = recommendations(fixtures(), "0", mode="similar")
        self.assertEqual(a["id"].tolist(), b["id"].tolist())


if __name__ == "__main__":
    unittest.main()
