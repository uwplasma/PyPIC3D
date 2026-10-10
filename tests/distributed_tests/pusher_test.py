"""Focused cross-device regression tests."""

from tests.support.pusher_fixtures import (
    TiledParticlePusherFixtures,
    jax,
    jnp,
    np,
    particle_species,
    unittest,
)


class TestTiledParticlePusher(TiledParticlePusherFixtures, unittest.TestCase):
    def test_particle_push_matches_one_tile_boris(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)

        species = self._species(parameter_set)
        reference = self._species(parameter_set)
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=False,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=False,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)
        tiled_u = jax.device_get(tiled_u)
        reference_u = jax.device_get(reference_u)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))


    def test_particle_push_matches_one_tile_higuera_cary(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (4, 3, 2)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=0.25)
        B = self._deterministic_vector_field(parameter_set, scale=0.05)

        def make_species():
            return particle_species(
                name="higuera cary particles",
                charge=-1.0,
                mass=2.0,
                weight=0.5,
                x1=jnp.array([-1.95, -1.01, 0.99, 1.95]),
                x2=jnp.array([-1.35, -0.01, 0.49, 1.35]),
                x3=jnp.array([-0.95, -0.01, 0.49, 0.95]),
                v1=jnp.array([0.02, -0.01, 0.03, -0.015]),
                v2=jnp.array([0.01, 0.02, -0.015, 0.005]),
                v3=jnp.array([-0.005, 0.015, 0.01, -0.02]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            particle_pusher="higuera_cary",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            particle_pusher="higuera_cary",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)
        tiled_u = jax.device_get(tiled_u)
        reference_u = jax.device_get(reference_u)

        self.assertTrue(jnp.allclose(tiled_u, reference_u, rtol=1.0e-12, atol=1.0e-12))


    def test_particle_push_matches_relativistic_one_tile_boris_on_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 1, 1)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=0.1)
        B = self._deterministic_vector_field(parameter_set, scale=0.02)
        def make_species():
            return particle_species(
                name="one dimensional",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-1.25, 0.15, 1.25]),
                x2=jnp.zeros(3),
                x3=jnp.zeros(3),
                v1=jnp.array([0.02, -0.01, 0.03]),
                v2=jnp.array([0.0, 0.01, -0.02]),
                v3=jnp.array([-0.005, 0.025, 0.01]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))


    def test_particle_push_matches_one_tile_boris_on_two_guard_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, shape_factor=2)
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 1, 1)
        g = 2
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape, g=g)
        E = self._deterministic_vector_field(parameter_set, scale=0.1)
        B = self._deterministic_vector_field(parameter_set, scale=0.02)

        def make_species():
            return particle_species(
                name="two guard one dimensional",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-1.25, 0.15, 1.25]),
                x2=jnp.zeros(3),
                x3=jnp.zeros(3),
                v1=jnp.array([0.02, -0.01, 0.03]),
                v2=jnp.array([0.0, 0.01, -0.02]),
                v3=jnp.array([-0.005, 0.025, 0.01]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, g)
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
