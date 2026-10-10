"""Focused cross-device regression tests."""

from tests.support.rho_fixtures import (
    TiledRhoFixtures,
    unittest,
)


class TestTiledRho(TiledRhoFixtures, unittest.TestCase):
    def test_tiled_rho_matches_compute_rho_for_shape_factor_1(self):
        self._compare_tiled_to_standard(shape_factor=1, alpha=1.0)


    def test_tiled_rho_matches_compute_rho_for_shape_factor_2(self):
        self._compare_tiled_to_standard(shape_factor=2, alpha=1.0)


    def test_tiled_rho_matches_compute_rho_after_digital_filter(self):
        self._compare_tiled_to_standard(shape_factor=2, alpha=0.55, current_filter="digital")


if __name__ == "__main__":
    unittest.main()
