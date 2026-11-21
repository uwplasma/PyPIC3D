"""One-tile numerical and physical checks."""

from tests.support.esirkepov_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    Esirkepov_current,
    TiledEsirkepovCurrentFixtures,
    compute_rho,
    jnp,
    kernel_parameters_from_values,
    refresh_tiled_particle_tiles,
    unittest,
    update_tiled_particle_positions,
)


class TestTiledEsirkepovCurrent(TiledEsirkepovCurrentFixtures, unittest.TestCase):
    def test_esirkepov_continuity_for_dimensions_and_shapes(self):
        for shape_factor in (1, 2):
            two_dimensional_case = (8, 8, 1) if shape_factor == 1 else (1, 8, 8)
            cases = (
                (8, 1, 1),
                (1, 8, 1),
                (1, 1, 8),
                two_dimensional_case,
                (4, 4, 4),
            )
            for Nx, Ny, Nz in cases:
                with self.subTest(shape_factor=shape_factor, shape=(Nx, Ny, Nz)):
                    parameter_set = self._build_parameter_values(Nx=Nx, Ny=Ny, Nz=Nz, dt=0.05, shape_factor=shape_factor)
                    active_axes = sum(int(width > 1) for width in (Nx, Ny, Nz))
                    # The production tiled Yee startup promotes Esirkepov current storage to two guards.
                    if shape_factor == 2 or active_axes == 3:
                        parameter_set["guard_cells"] = 2
                    x_old, u = self._basic_positions_and_velocities(parameter_set)
                    self._assert_single_tile_continuity(parameter_set, x_old, u)

    def test_tiled_esirkepov_satisfies_discrete_continuity_at_conducting_particle_walls(self):
        walls = {"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC}
        parameter_set = self._build_parameter_values(
            Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1, particle_boundary_conditions=walls)
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dx, dt = parameter_set["dx"], parameter_set["dt"]
        x_min = -0.5 * parameter_set["x_wind"]
        x_max = 0.5 * parameter_set["x_wind"]
        # two particles cross a wall during the step and reflect; one stays inside
        x_old = jnp.array([[x_min + 0.3 * dx, 0.0, 0.0],
                           [x_max - 0.2 * dx, 0.0, 0.0],
                           [0.1 * dx, 0.0, 0.0]])
        u = jnp.array([[-0.7 * dx / dt, 0.0, 0.0],
                       [0.6 * dx / dt, 0.0, 0.0],
                       [0.4 * dx / dt, 0.0, 0.0]])
        tiled_particles, species_config = self._particles_from_arrays(parameter_set, tile_shape, x_old, u)
        g = int(parameter_set["guard_cells"])
        n = tile_shape[0]
        rho_tiles = self._build_tiled_array(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)

        rho_old = compute_rho(tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        J_tiles = Esirkepov_current(
            tiled_particles,
            update_tiled_particle_positions(tiled_particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=tiled_particles.u,
        )
        new_particles = update_tiled_particle_positions(tiled_particles, species_config, dt)
        new_particles, overflow = refresh_tiled_particle_tiles(new_particles, static_parameters, dynamic_parameters)
        rho_new = compute_rho(new_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        # every owned C node, including both wall nodes g and g+n
        owned = (slice(None),) * 3 + (slice(g, g + n + 1), slice(g, -g), slice(g, -g))
        before = (slice(None),) * 3 + (slice(g - 1, g + n), slice(g, -g), slice(g, -g))
        drhodt = (rho_new[owned] - rho_old[owned]) / dt
        dJxdx = (J_tiles[0][owned] - J_tiles[0][before]) / dx
        continuity = drhodt + dJxdx
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(drhodt)) + jnp.max(jnp.abs(dJxdx)))
        self.assertGreater(float(jnp.max(jnp.abs(rho_new[owned][0, 0, 0, 0]))), 0.0)
        self.assertLessEqual(float(jnp.max(jnp.abs(continuity))), float(1.0e-12 * scale))


if __name__ == "__main__":
    unittest.main()
