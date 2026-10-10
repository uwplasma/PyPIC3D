"""Focused cross-device regression tests."""

from tests.support.static_metric_esirkepov_fixtures import (
    GRESirkepovContinuityFixtures,
    _continuity_residual,
    unittest,
)


class TestGRESirkepovContinuity(GRESirkepovContinuityFixtures, unittest.TestCase):
    def test_continuity_is_exact_across_a_tile_boundary(self):
        # the domain spans [-4, 4) with two tiles in x, so the particle crosses
        # the interior tile seam at x = 0
        residual, scale = _continuity_residual(
            "flat_cartesian",
            (-0.25, -0.70, 1.10),
            (0.20, -0.41, 1.33),
            tile_shape=(4, 8, 8),
        )
        self.assertGreater(scale, 0.0)
        self.assertLess(residual / scale, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
