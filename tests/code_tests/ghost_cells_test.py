"""Single-device numerical tests."""

from tests.support.ghost_cells_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    GhostCellsFixtures,
    SimpleNamespace,
    _assert_allclose,
    _particle_wall_parameters,
    ghost_cells,
    jax,
    jnp,
    unittest,
)


class TestGhostCells(GhostCellsFixtures, unittest.TestCase):
    def test_update_tiled_ghost_cells_requires_static_field_mesh(self):
        # tiled halo exchange should use the startup-owned field mesh, not infer
        # a device mesh from the current array shape.

        field_tiles = jnp.zeros((1, 1, 1, 4, 4, 4))

        incomplete_parameters = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

        with self.assertRaises(AttributeError):
            ghost_cells.update_tiled_ghost_cells(field_tiles, incomplete_parameters, self.g)


    def test_apply_tiled_zero_boundary_zeros_global_tangential_faces(self):
        # this tests the application of axis-wise conducting boundary conditions to a tiled electric field

        parameter_set = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        )
        # create a parameter_set with conducting boundary conditions in all directions

        E = tuple(jnp.ones((1, 1, 1, 4, 4, 4)) for _ in range(3))
        # create a tuple of three electric field components (Ex, Ey, Ez) with shape (1, 1, 1, 4, 4, 4) and all values set to 1.0

        Ex, Ey, Ez = E
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=0, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=0, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=1, num_guard_cells=self.g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, parameter_set, axis=1, num_guard_cells=self.g)
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, parameter_set, axis=2, num_guard_cells=self.g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, parameter_set, axis=2, num_guard_cells=self.g)
        # call the shared scalar zero boundary method for each tangential electric component

        self.assertTrue(jnp.all(Ey[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, 1, :, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ez[0, 0, 0, :, 1, :] == 0.0))
        self.assertTrue(jnp.all(Ex[0, 0, 0, :, :, 1] == 0.0))
        self.assertTrue(jnp.all(Ey[0, 0, 0, :, :, 1] == 0.0))


    def test_apply_tiled_zero_boundary_periodic_axis_keeps_refreshed_field(self):
        parameter_set = self._parameters_with_field_mesh((1, 1, 1))
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        # create a scalar field with shape (1, 1, 1, 4, 4, 4) and values from 0 to 63
        refreshed = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)
        result = ghost_cells.apply_tiled_zero_boundary(refreshed, parameter_set, axis=0, num_guard_cells=self.g)
        # periodic field boundaries do not zero the physical plane

        self.assertTrue(jnp.allclose(result, refreshed))


    def test_apply_tiled_constant_boundary_copies_adjacent_interior_to_global_ghosts(self):
        parameter_set = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))

        result = ghost_cells.apply_tiled_constant_boundary(field, parameter_set, axis=0, num_guard_cells=self.g)

        self.assertTrue(jnp.allclose(result[0, 0, 0, 0, :, :], result[0, 0, 0, 1, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, -1, :, :], result[0, 0, 0, -2, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 1:-1, 1:-1, 1:-1], field[0, 0, 0, 1:-1, 1:-1, 1:-1]))


    def test_apply_tiled_constant_boundary_periodic_axis_keeps_refreshed_field(self):
        parameter_set = self._parameters_with_field_mesh((1, 1, 1))
        field = jnp.arange(4 * 4 * 4, dtype=float).reshape((1, 1, 1, 4, 4, 4))
        refreshed = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)

        result = ghost_cells.apply_tiled_constant_boundary(field, parameter_set, axis=0, num_guard_cells=self.g)

        self.assertTrue(jnp.allclose(result, refreshed))


    def test_update_tiled_ghost_cells_constant_copies_adjacent_global_interiors(self):
        parameter_set = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=self.tile_shape,
            guard_cells=self.g,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONSTANT, BC_PERIODIC, BC_PERIODIC),
        )
        field = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        field = field.at[0, 0, 0, 1, :, :].set(2.0)
        field = field.at[0, 0, 0, 2, :, :].set(7.0)

        result = ghost_cells.update_tiled_ghost_cells(field, parameter_set, self.g)

        self.assertTrue(jnp.allclose(result[0, 0, 0, 0, :, :], result[0, 0, 0, 1, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, -1, :, :], result[0, 0, 0, -2, :, :]))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 1, :, :], 2.0))
        self.assertTrue(jnp.allclose(result[0, 0, 0, 2, :, :], 7.0))


    def test_particle_scalar_fold_and_refresh_mirror_both_reflecting_walls(self):
        g = 2
        parameters = _particle_wall_parameters((2, 2, 4), g, (BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING))
        deposits = jnp.zeros((1, 1, 1, 6, 6, 8), dtype=float)
        line = (0, 0, 0, g, g)
        # owned C nodes are 2..6 (walls at 2 and 6); ghosts are 0, 1 and 7
        deposits = deposits.at[line].set(jnp.array([1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 3.0, 4.0]))

        even = ghost_cells.fold_tiled_ghost_cells(
            deposits, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
        odd = ghost_cells.fold_tiled_ghost_cells(
            deposits, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE, reflecting_parity=(1, 1, -1))

        _assert_allclose(self, even[line], [0.0, 0.0, 20.0, 22.0, 31.0, 44.0, 6.0, 0.0])
        _assert_allclose(self, odd[line], [0.0, 0.0, 0.0, 18.0, 29.0, 36.0, 0.0, 0.0])

        refreshed_even = ghost_cells.update_tiled_ghost_cells(
            even, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
        refreshed_odd = ghost_cells.update_tiled_ghost_cells(
            odd, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE, reflecting_parity=(1, 1, -1))
        _assert_allclose(self, refreshed_even[line], [31.0, 22.0, 20.0, 22.0, 31.0, 44.0, 6.0, 44.0])
        _assert_allclose(self, refreshed_odd[line], [-29.0, -18.0, 0.0, 18.0, 29.0, 36.0, 0.0, -36.0])


    def test_particle_vector_has_tangential_even_and_normal_odd_wall_parity(self):
        g = 2
        parameters = _particle_wall_parameters((2, 2, 4), g, (BC_PERIODIC, BC_PERIODIC, BC_CONDUCTING))
        scalar = jnp.zeros((1, 1, 1, 6, 6, 8), dtype=float)
        line = (0, 0, 0, g, g)
        scalar = scalar.at[line].set(jnp.array([1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 3.0, 4.0]))
        vector = (scalar, scalar, scalar)

        folded = ghost_cells.fold_tiled_vector_ghost_cells(
            vector, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
        refreshed = ghost_cells.update_tiled_vector_ghost_cells(
            folded, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)

        # Jx and Jy are tangential to the z-wall and collocated in z: even C images
        for component in (0, 1):
            _assert_allclose(self, folded[component][line], [0.0, 0.0, 20.0, 22.0, 31.0, 44.0, 6.0, 0.0])
            _assert_allclose(self, refreshed[component][line], [31.0, 22.0, 20.0, 22.0, 31.0, 44.0, 6.0, 44.0])
        # Jz is normal and staggered in z: odd V images, owned nodes 2..5
        _assert_allclose(self, folded[2][line], [0.0, 0.0, 8.0, 19.0, 26.0, 37.0, 0.0, 0.0])
        _assert_allclose(self, refreshed[2][line], [-19.0, -8.0, 8.0, 19.0, 26.0, 37.0, -37.0, -26.0])


    def test_particle_vector_reflection_parity_applies_on_every_axis(self):
        g = 1
        tile_shape = (2, 2, 2)
        scalar = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        scalar = scalar.at[0, 0, 0, g:-g, g:-g, g:-g].set(7.0)
        vector = (scalar, scalar, scalar)

        for wall_axis in range(3):
            particle_boundaries = [BC_PERIODIC, BC_PERIODIC, BC_PERIODIC]
            particle_boundaries[wall_axis] = BC_CONDUCTING
            parameters = _particle_wall_parameters(tile_shape, g, tuple(particle_boundaries))
            refreshed = ghost_cells.update_tiled_vector_ghost_cells(
                vector, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
            lower_ghost = [0, 0, 0, g, g, g]
            lower_ghost[3 + wall_axis] = 0
            upper_end = [0, 0, 0, g, g, g]
            upper_end[3 + wall_axis] = -1
            for component in range(3):
                with self.subTest(wall_axis=wall_axis, component=component):
                    if component == wall_axis:
                        # normal component, staggered: odd images on both sides
                        self.assertEqual(float(refreshed[component][tuple(lower_ghost)]), -7.0)
                        self.assertEqual(float(refreshed[component][tuple(upper_end)]), -7.0)
                    else:
                        # tangential, collocated: even image of C1; the upper
                        # wall node is owned and keeps its (empty) deposit
                        self.assertEqual(float(refreshed[component][tuple(lower_ghost)]), 7.0)
                        self.assertEqual(float(refreshed[component][tuple(upper_end)]), 0.0)


    def test_reduced_reflecting_axis_uses_scalar_parity(self):
        g = 1
        parameters = _particle_wall_parameters((1, 2, 2), g, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))
        deposits = jnp.zeros((1, 1, 1, 3, 4, 4), dtype=float)
        line = (0, 0, 0, slice(None), g, g)
        # a one-cell axis has two owned wall nodes, 1 and 2
        deposits = deposits.at[line].set(jnp.array([2.0, 10.0, 3.0]))

        even = ghost_cells.fold_tiled_ghost_cells(
            deposits, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
        odd = ghost_cells.fold_tiled_ghost_cells(
            deposits, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE, reflecting_parity=(-1, 1, 1))
        # the ghost folds onto C1, then both wall nodes are doubled;
        # trapezoid weights conserve the deposit: (20 + 10) / 2 = 15
        _assert_allclose(self, even[line], [0.0, 20.0, 10.0])
        _assert_allclose(self, odd[line], [0.0, 0.0, 0.0])

        refreshed_even = ghost_cells.update_tiled_ghost_cells(
            even, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE)
        _assert_allclose(self, refreshed_even[line], [10.0, 20.0, 10.0])


    def test_reflecting_corner_composes_axis_parity(self):
        g = 1
        parameters = _particle_wall_parameters((2, 2, 2), g, (BC_CONDUCTING, BC_PERIODIC, BC_CONDUCTING))
        parity = (-1, 1, -1)
        # C1 in x and z, the first node inside both walls
        field = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        field = field.at[0, 0, 0, 2, g, 2].set(7.0)
        refreshed = ghost_cells.update_tiled_ghost_cells(
            field, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE, reflecting_parity=parity)
        self.assertEqual(float(refreshed[0, 0, 0, 0, g, 2]), -7.0)
        self.assertEqual(float(refreshed[0, 0, 0, 2, g, 0]), -7.0)
        self.assertEqual(float(refreshed[0, 0, 0, 0, g, 0]), 7.0)

        deposits = jnp.zeros_like(field).at[0, 0, 0, 0, g, 0].set(3.0)
        folded = ghost_cells.fold_tiled_ghost_cells(
            deposits, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE, reflecting_parity=parity)
        self.assertEqual(float(folded[0, 0, 0, 2, g, 2]), 3.0)


    def test_flat_and_static_metric_fold_particle_walls_identically(self):
        g = 2
        rng = jax.random.PRNGKey(0)
        scalar_key, vector_key = jax.random.split(rng)
        scalar = jax.random.normal(scalar_key, (1, 1, 1, 6, 6, 8))
        vector = tuple(jax.random.normal(key, (1, 1, 1, 6, 6, 8)) for key in jax.random.split(vector_key, 3))
        walls = (BC_CONDUCTING, BC_PERIODIC, BC_CONDUCTING)
        results = []
        for solver in ("electrodynamic_yee", "static_metric"):
            parameters = _particle_wall_parameters((2, 2, 4), g, walls, solver=solver)
            results.append((
                ghost_cells.fold_tiled_ghost_cells(scalar, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE),
                ghost_cells.fold_tiled_vector_ghost_cells(vector, parameters, g, bc_type=ghost_cells.BC_TYPE_PARTICLE),
            ))
        (flat_rho, flat_J), (gr_rho, gr_J) = results
        _assert_allclose(self, flat_rho, gr_rho, rtol=0.0, atol=0.0)
        for flat, gr in zip(flat_J, gr_J):
            _assert_allclose(self, flat, gr, rtol=0.0, atol=0.0)


    def test_reflecting_parity_is_rejected_for_field_boundaries(self):
        parameters = SimpleNamespace(
            solver="electrodynamic_yee",
            tile_shape=(2, 2, 2),
            guard_cells=1,
            field_mesh=ghost_cells.make_field_mesh((1, 1, 1)),
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
        )
        scalar = jnp.zeros((1, 1, 1, 4, 4, 4), dtype=float)
        with self.assertRaisesRegex(ValueError, "only valid for particle"):
            ghost_cells.update_tiled_ghost_cells(
                scalar,
                parameters,
                reflecting_parity=(1, 1, 1),
            )



if __name__ == "__main__":
    unittest.main()
