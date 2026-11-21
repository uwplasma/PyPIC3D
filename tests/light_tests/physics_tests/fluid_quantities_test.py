"""One-tile numerical and physical checks."""

from tests.support.fluid_quantities_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    TiledFluidQuantitiesFixtures,
    build_tiled_particles,
    fluid_velocity,
    jnp,
    particle_species,
    unittest,
)


class TestTiledFluidQuantities(TiledFluidQuantitiesFixtures, unittest.TestCase):
    def test_fluid_velocity_computes_weighted_local_average(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
