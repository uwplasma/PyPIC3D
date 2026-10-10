"""Focused cross-device regression tests."""

from tests.support.distributed_ghost_cells_fixtures import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
    DistributedGhostCellsFixtures,
    _coordinate_tiles,
    _mesh,
    _reference_fold,
    _reference_update,
    _static_parameters,
    ghost_cells,
    jax,
    jnp,
    unittest,
)


class TestDistributedGhostCells(DistributedGhostCellsFixtures, unittest.TestCase):
    def test_scalar_halo_refresh_matches_reference_on_2x2x2_periodic_mesh(self):
        mesh_shape = (2, 2, 2)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, 0, 0], tiles[1, 1, 1, -2, -2, -2])


    def test_scalar_halo_refresh_matches_reference_on_mixed_2x2x1_mesh(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, :, 0, :], jnp.zeros_like(actual[0, 0, 0, :, 0, :]))


    def test_reduced_physical_axis_uses_single_interior_cell_not_neighbor_exchange(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 1)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[:, :, :, :, :, 0], actual[:, :, :, :, :, 1])
        self.assert_allclose(actual[:, :, :, :, :, -1], actual[:, :, :, :, :, 1])


    def test_vector_refresh_absorbing_particle_axis_exchanges_internal_halos_and_zeros_walls(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (BC_ABSORBING, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        stacked = jnp.stack((tiles, tiles + 1000.0, tiles + 2000.0), axis=0)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        refreshed = ghost_cells.update_tiled_vector_ghost_cells(
            stacked,
            static_parameters,
            1,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )

        for component in range(3):
            self.assert_allclose(
                refreshed[component],
                _reference_update(stacked[component], particle_bcs, tile_shape),
            )
            self.assert_allclose(refreshed[component, 0, 0, 0, 0, :, :], 0.0)
            self.assert_allclose(refreshed[component, -1, 0, 0, -1, :, :], 0.0)
            self.assert_allclose(
                refreshed[component, 0, 0, 0, -1, 1:-1, 1:-1],
                stacked[component, 1, 0, 0, 1, 1:-1, 1:-1],
            )
            self.assert_allclose(
                refreshed[component, 1, 0, 0, 0, 1:-1, 1:-1],
                stacked[component, 0, 0, 0, -2, 1:-1, 1:-1],
            )


    def test_reflecting_particle_axis_preserves_internal_exchange_and_mirrors_global_walls(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        g = 1
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            g=g,
            particle_boundary_conditions=particle_bcs,
        )
        tiles = jnp.zeros(mesh_shape + (4, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, g:-g, g:-g, g:-g].set(1.0)
        tiles = tiles.at[1, 0, 0, g:-g, g:-g, g:-g].set(2.0)

        refreshed = ghost_cells.update_tiled_ghost_cells(
            tiles,
            static_parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        # Collocated scalars mirror about the wall nodes; the last tile owns
        # the upper wall node, which keeps its (empty) value.
        self.assert_allclose(refreshed[0, 0, 0, 0, g:-g, g:-g], 1.0)
        self.assert_allclose(refreshed[0, 0, 0, -1, g:-g, g:-g], 2.0)
        self.assert_allclose(refreshed[1, 0, 0, 0, g:-g, g:-g], 1.0)
        self.assert_allclose(refreshed[1, 0, 0, -1, g:-g, g:-g], 0.0)

        stacked = jnp.stack((tiles, tiles, tiles), axis=0)
        vector = ghost_cells.update_tiled_vector_ghost_cells(
            stacked,
            static_parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        # Jx is normal and staggered in x (odd images about the wall faces);
        # Jy and Jz are tangential and collocated in x (even nodal images).
        self.assert_allclose(vector[0, 0, 0, 0, 0, g:-g, g:-g], -1.0)
        self.assert_allclose(vector[1:, 0, 0, 0, 0, g:-g, g:-g], 1.0)
        self.assert_allclose(vector[:, 0, 0, 0, -1, g:-g, g:-g], 2.0)
        self.assert_allclose(vector[:, 1, 0, 0, 0, g:-g, g:-g], 1.0)
        self.assert_allclose(vector[0, 1, 0, 0, -1, g:-g, g:-g], -2.0)
        self.assert_allclose(vector[1:, 1, 0, 0, -1, g:-g, g:-g], 0.0)

        deposits = jnp.zeros_like(tiles)
        deposits = deposits.at[0, 0, 0, 0, g, g].set(3.0)
        deposits = deposits.at[0, 0, 0, -1, g, g].set(5.0)
        deposits = deposits.at[1, 0, 0, 0, g, g].set(7.0)
        deposits = deposits.at[1, 0, 0, -1, g, g].set(11.0)
        folded = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            static_parameters,
            g,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        # The lower ghost C-1 folds onto C1 beside the internal deposit from
        # tile 1; the empty lower wall node stays zero after doubling; the
        # internal halo moves to tile 1; the last tile's upper wall node is
        # owned and receives its coincident image (doubled).
        self.assertEqual(float(folded[0, 0, 0, g, g, g]), 0.0)
        self.assertEqual(float(folded[0, 0, 0, g + 1, g, g]), 10.0)
        self.assertEqual(float(folded[1, 0, 0, g, g, g]), 5.0)
        self.assertEqual(float(folded[1, 0, 0, g + 2, g, g]), 22.0)
        self.assert_allclose(folded[:, :, :, 0, :, :], 0.0)
        self.assert_allclose(folded[0, :, :, -1, :, :], 0.0)


    def test_stacked_and_tuple_vector_halo_refresh_preserve_layouts(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_CONDUCTING)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        stacked = jnp.stack((tiles, tiles + 1000.0, tiles + 2000.0), axis=0)
        updater = ghost_cells.make_distributed_vector_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)

        stacked_actual = jax.jit(updater)(stacked)
        tuple_actual = updater((stacked[0], stacked[1], stacked[2]))

        self.assertEqual(stacked_actual.shape, stacked.shape)
        self.assertIsInstance(tuple_actual, tuple)
        for i in range(3):
            self.assert_allclose(stacked_actual[i], _reference_update(stacked[i], bcs, tile_shape))
            self.assert_allclose(tuple_actual[i], stacked_actual[i])


    def test_axis_zero_boundary_builds_conducting_electric_bc_on_global_walls(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_CONDUCTING, BC_CONDUCTING, BC_PERIODIC)
        E = tuple(jnp.ones((mesh_shape + (4, 4, 4)), dtype=jnp.float64) * (i + 1.0) for i in range(3))
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        def apply_electric_bc(Ex, Ey, Ez):
            Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=0, num_guard_cells=1)
            Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=0, num_guard_cells=1)
            Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=1, num_guard_cells=1)
            Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=1, num_guard_cells=1)
            return Ex, Ey, Ez

        Ex, Ey, Ez = jax.jit(apply_electric_bc)(*E)

        self.assert_allclose(Ey[0, :, :, 1, :, :], 0.0)
        self.assert_allclose(Ez[-1, :, :, -1, :, :], 0.0)
        self.assert_allclose(Ex[:, 0, :, :, 1, :], 0.0)
        self.assert_allclose(Ez[:, -1, :, :, -1, :], 0.0)
        self.assert_allclose(Ey[0, :, :, 2, 1:-1, 1:-1], 2.0)
        self.assert_allclose(Ex[:, 0, :, 1:-1, 2, 1:-1], 1.0)


    def test_axis_constant_boundary_preserves_internal_exchange_and_copies_global_ghosts(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        field = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        actual = jax.jit(lambda tiles: ghost_cells.apply_tiled_constant_boundary(tiles, static_parameters, 0, 1))(field)

        self.assert_allclose(actual[0, 0, 0, 0, :, :], actual[0, 0, 0, 1, :, :])
        self.assert_allclose(actual[-1, 0, 0, -1, :, :], actual[-1, 0, 0, -2, :, :])
        self.assert_allclose(actual[0, 0, 0, -1, 1:-1, 1:-1], field[1, 0, 0, 1, 1:-1, 1:-1])
        self.assert_allclose(actual[1, 0, 0, 0, 1:-1, 1:-1], field[0, 0, 0, -2, 1:-1, 1:-1])


    def test_scalar_and_vector_folding_match_reference_and_clear_ghosts(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (4, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[:, :, :, 0, :, :].set(1.0)
        tiles = tiles.at[:, :, :, -1, :, :].set(2.0)
        tiles = tiles.at[:, :, :, :, 0, :].set(3.0)
        tiles = tiles.at[:, :, :, :, -1, :].set(4.0)
        tiles = tiles.at[:, :, :, :, :, 0].set(5.0)
        tiles = tiles.at[:, :, :, :, :, -1].set(6.0)
        folder = ghost_cells.make_distributed_ghost_folder(_mesh(mesh_shape), tile_shape, bcs, 1)
        vector_folder = ghost_cells.make_distributed_vector_ghost_folder(_mesh(mesh_shape), tile_shape, bcs, 1)

        actual = jax.jit(folder)(tiles)
        vector_actual = jax.jit(vector_folder)((tiles, tiles + 1.0, tiles + 2.0))

        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        for component, base in zip(vector_actual, (tiles, tiles + 1.0, tiles + 2.0)):
            self.assert_allclose(component, _reference_fold(base, bcs, tile_shape))
            self.assert_allclose(component[:, :, :, 0, :, :], 0.0)
            self.assert_allclose(component[:, :, :, -1, :, :], 0.0)


    def test_standard_refresh_uses_initialization_owned_nonwrapping_axis(self):
        mesh_shape = (2, 1, 1)
        tile_shape = (2, 2, 2)
        static_parameters = _static_parameters((BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC), tile_shape, mesh_shape)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1)
        expected = _reference_update(tiles, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC), tile_shape)

        self.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
