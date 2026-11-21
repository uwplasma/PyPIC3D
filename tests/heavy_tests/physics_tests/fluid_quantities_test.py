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

    def test_reflecting_wall_fluid_velocity_remains_a_bounded_particle_average(self):
        periodic_static, dynamic_parameters = self._build_parameters(shape_factor=2)
        reflecting_static = periodic_static._replace(
            particle_boundary_conditions=(
                BC_PERIODIC,
                BC_PERIODIC,
                BC_CONDUCTING,
            ),
        )
        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-0.2, 0.2, -0.2, 0.2]),
                x2=jnp.zeros(4),
                x3=jnp.array([-0.99, -0.76, 0.76, 0.99]),
                v1=jnp.array([-0.4, 0.2, 0.1, 0.3]),
                v3=jnp.array([0.25, -0.15, 0.05, -0.2]),
            )
        ]
        tiled_particles, species_config = build_tiled_particles(
            particles,
            reflecting_static,
            dynamic_parameters,
        )
        g = int(reflecting_static.guard_cells)

        ux = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(reflecting_static, dynamic_parameters),
            0,
            reflecting_static,
            dynamic_parameters,
        )
        uz = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(reflecting_static, dynamic_parameters),
            2,
            reflecting_static,
            dynamic_parameters,
        )

        self.assertLessEqual(float(jnp.max(jnp.abs(ux[:, :, :, g:-g, g:-g, g:-g]))), 0.4 + 1.0e-12)
        self.assertLessEqual(float(jnp.max(jnp.abs(uz[:, :, :, g:-g, g:-g, g:-g]))), 0.25 + 1.0e-12)
        self.assertFalse(bool(jnp.any(jnp.isnan(ux))))
        self.assertFalse(bool(jnp.any(jnp.isnan(uz))))
        # nodal images about the lower wall node g: ghost g-k mirrors g+k
        for k in range(1, g + 1):
            self.assertTrue(bool(jnp.allclose(ux[0, 0, 0, g:-g, g:-g, g - k], ux[0, 0, 0, g:-g, g:-g, g + k])))
            self.assertTrue(bool(jnp.allclose(uz[0, 0, 0, g:-g, g:-g, g - k], -uz[0, 0, 0, g:-g, g:-g, g + k])))


if __name__ == "__main__":
    unittest.main()
