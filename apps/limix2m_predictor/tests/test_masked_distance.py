"""Check retrieval distances against the direct scientific definition."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limix_core import LimiX2MEngine


class MaskedDistanceTest(unittest.TestCase):
    def test_matches_direct_rms_for_individual_missing_patterns(self):
        rng = np.random.default_rng(8)
        library = rng.normal(size=(9, 36))
        raw = 10 ** rng.normal(size=(3, 36))
        raw[1, 18:] = np.nan
        raw[2, 10:28] = np.nan
        engine = LimiX2MEngine.__new__(LimiX2MEngine)
        engine.train_log36 = library
        actual = engine._distance_matrix(raw)
        expected = np.empty((3, 9))
        for i, row in enumerate(raw):
            mask = np.isfinite(row)
            expected[i] = np.sqrt(np.mean((np.log10(row[mask]) - library[:, mask]) ** 2, axis=1))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_unmeasured_library_columns_cannot_change_distance(self):
        engine = LimiX2MEngine.__new__(LimiX2MEngine)
        engine.train_log36 = np.zeros((2, 36))
        raw = np.ones((1, 36))
        raw[:, 18:] = np.nan
        before = engine._distance_matrix(raw)
        engine.train_log36[:, 18:] = 50.0
        np.testing.assert_allclose(engine._distance_matrix(raw), before, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
