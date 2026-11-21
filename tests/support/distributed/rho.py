from tests.support.rho_fixtures import (
    TiledRhoFixtures as SharedFixtures,
    jnp,
)


class TiledRhoFixtures(SharedFixtures):
    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

    def _compare_tiled_to_standard(self, shape_factor, alpha, current_filter="none"):
        parameter_set = self._build_parameter_values(shape_factor)
        parameter_set["current_filter"] = current_filter
        dynamic_values = {"alpha": alpha}
        particles = self._particles(parameter_set)
        _, rho_from_tiles = self._deposit_and_assemble(particles, parameter_set, self._simulation_parameters(), dynamic_values)
        _, rho_reference = self._deposit_and_assemble(particles, parameter_set, self._one_tile_parameters(parameter_set), dynamic_values)

        self.assertTrue(
            jnp.allclose(
                rho_from_tiles[1:-1, 1:-1, 1:-1],
                rho_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
