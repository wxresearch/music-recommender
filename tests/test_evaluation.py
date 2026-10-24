import unittest
import numpy as np
import pandas as pd
from evaluation import rank_correlation, evaluate
from recommender import BASE_FEATURES


class EvaluationTests(unittest.TestCase):
    def test_correlation_ordering(self):
        self.assertAlmostEqual(rank_correlation([0.1, 0.5, 0.9], [1, 3, 5]), 1.0)
        self.assertAlmostEqual(rank_correlation([0.9, 0.5, 0.1], [1, 3, 5]), -1.0)

    def test_constant_labels_undefined(self):
        self.assertIsNone(rank_correlation([0.1, 0.5, 0.9], [3, 3, 3]))

    def test_scores_ignore_human_labels(self):
        library = []
        for i in range(4):
            features = {k: float(i + j * .01) for j, k in enumerate(BASE_FEATURES)}
            features.update({f"mfcc_{j}": float(i + j / 10) for j in range(1,14)})
            features.update({f"chroma_{j}": float(i + j / 10) for j in range(12)})
            library.append({"id": str(i), "title": str(i), "features": features})
        emb = {str(i): np.array([1.0, float(i)]) for i in range(4)}
        pairs = {("0", "1"): 1, ("0", "2"): 3, ("0", "3"): 5}
        a, _ = evaluate(library, pairs, emb)
        b, _ = evaluate(library, {k: 6-v for k, v in pairs.items()}, emb)
        for method in ["CLAP only", "Acoustic only", "Hybrid"]:
            np.testing.assert_allclose(a[method], b[method])


if __name__ == "__main__":
    unittest.main()
