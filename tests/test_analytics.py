from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from live_ingest.analytics import (
    compute_log_returns_vectorized,
    residual_zscore_latest,
    rolling_zscore_latest,
)


class AnalyticsTests(unittest.TestCase):
    def _sample_matrix(self, rows: int = 520, cols: int = 4) -> pd.DataFrame:
        base = datetime(2026, 4, 5, 9, 30, tzinfo=timezone.utc)
        idx = [base + timedelta(minutes=i) for i in range(rows)]
        arr = np.linspace(100.0, 130.0, rows * cols, dtype=float).reshape(rows, cols)
        if cols > 1:
            arr[:, 1] *= 1.001
        if cols > 2:
            arr[:, 2] *= 0.999
        if cols > 3:
            arr[:, 3] *= 1.002
        cols_names = [f"T{i}" for i in range(cols)]
        return pd.DataFrame(arr, index=idx, columns=cols_names)

    def test_log_returns(self) -> None:
        mat = self._sample_matrix(rows=10, cols=3)
        out = compute_log_returns_vectorized(mat)
        self.assertEqual(out.shape[0], 9)
        self.assertEqual(out.shape[1], 3)

    def test_rolling_zscore(self) -> None:
        mat = self._sample_matrix(rows=80, cols=3)
        ret = compute_log_returns_vectorized(mat)
        zdf, latest = rolling_zscore_latest(ret, window=20)
        self.assertFalse(zdf.empty)
        self.assertEqual(len(latest.index), 3)

    def test_residual_zscore_latest(self) -> None:
        mat = self._sample_matrix(rows=520, cols=4)
        ret = compute_log_returns_vectorized(mat)
        rz = residual_zscore_latest(ret, window=500, factors=1)
        self.assertEqual(len(rz.index), 4)


if __name__ == "__main__":
    unittest.main()
