from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from live_ingest.type2_fallback import Type2FallbackVerifier


class Type2FallbackTests(unittest.TestCase):
    def test_backfill_reads_parquet_window(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "outputs" / "live" / "parquet" / "date=2026-04-05" / "hour=10"
            p.mkdir(parents=True, exist_ok=True)
            file_path = p / "bars.parquet"

            base = datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc)
            rows = []
            for i in range(3):
                end = base + timedelta(minutes=i + 1)
                start = end - timedelta(minutes=1)
                rows.append(
                    {
                        "symbol": "AAPL",
                        "source": "type2",
                        "bar_start": start.isoformat(),
                        "bar_end": end.isoformat(),
                        "open_fixed": 100_000_000_000 + i,
                        "high_fixed": 101_000_000_000 + i,
                        "low_fixed": 99_000_000_000 + i,
                        "close_fixed": 100_500_000_000 + i,
                        "volume": 10 + i,
                    }
                )
            pd.DataFrame(rows).to_parquet(file_path, index=False)

            verifier = Type2FallbackVerifier(root_dir=root, parquet_root="outputs/live/parquet")
            bars = verifier.backfill_bars(
                symbols=("AAPL",),
                after_bar_end=(base + timedelta(minutes=1)).isoformat(),
                until=base + timedelta(minutes=3, seconds=30),
                price_scale=1_000_000_000,
            )
            self.assertEqual(len(bars), 2)
            self.assertTrue(all(b.symbol == "AAPL" for b in bars))


if __name__ == "__main__":
    unittest.main()
