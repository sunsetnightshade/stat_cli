from __future__ import annotations

import unittest
from datetime import datetime, timezone

from live_ingest.aggregator import MinuteBarAggregator
from live_ingest.models import TickEvent


class AggregatorTests(unittest.TestCase):
    def test_minute_bar_emit_on_rollover(self) -> None:
        agg = MinuteBarAggregator(price_scale=1_000_000_000, source="test")

        t0 = datetime(2026, 4, 5, 10, 0, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 4, 5, 10, 0, 30, tzinfo=timezone.utc)
        t2 = datetime(2026, 4, 5, 10, 1, 0, tzinfo=timezone.utc)

        self.assertEqual(
            agg.ingest_tick(TickEvent("AAPL", 100.0, 10.0, t0, "test")),
            [],
        )
        self.assertEqual(
            agg.ingest_tick(TickEvent("AAPL", 101.0, 5.0, t1, "test")),
            [],
        )
        bars = agg.ingest_tick(TickEvent("AAPL", 99.0, 2.0, t2, "test"))
        self.assertEqual(len(bars), 1)
        bar = bars[0]
        self.assertEqual(bar.symbol, "AAPL")
        self.assertTrue(bar.high_fixed >= bar.low_fixed)
        self.assertEqual(bar.volume, 15)


if __name__ == "__main__":
    unittest.main()
