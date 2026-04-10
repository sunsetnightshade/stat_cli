from __future__ import annotations

import unittest

from live_ingest.precision import from_fixed_price, to_fixed_price


class PrecisionTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        scale = 1_000_000_000
        value = 150.25
        fixed = to_fixed_price(value, scale=scale)
        restored = from_fixed_price(fixed, scale=scale)
        self.assertAlmostEqual(value, restored, places=9)


if __name__ == "__main__":
    unittest.main()
