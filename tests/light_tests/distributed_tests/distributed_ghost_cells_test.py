"""Focused cross-device regression tests."""

from tests.support.distributed.distributed_ghost_cells import (
    NamedSharding,
)
from tests.support.distributed.distributed_ghost_cells import (
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

        updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, 0, 0], tiles[1, 1, 1, -2, -2, -2])


    def test_scalar_halo_refresh_matches_reference_on_mixed_2x2x1_mesh(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_CONDUCTING, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1)
        actual = jax.jit(updater)(tiles)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, :, 0, :], jnp.zeros_like(actual[0, 0, 0, :, 0, :]))


    def test_reduced_physical_axis_uses_single_interior_cell_not_neighbor_exchange(self):
        mesh_shape = (2, 2, 1)
        tile_shape = (2, 2, 1)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1)
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

        refreshed = ghost_cells.update_tiled_ghost_cells(
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
        vector = ghost_cells.update_tiled_ghost_cells(
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
        updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1, vector=True)

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
        folder = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1, fold=True)
        vector_folder = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1, fold=True, vector=True)

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


    def test_public_scalar_update_on_one_device_uses_mapped_periodic_self_exchange(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_tiles = jax.device_put(tiles, sharding)
        actual = jax.jit(lambda field: ghost_cells.update_tiled_ghost_cells(field, static_parameters, 1))(sharded_tiles)

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, 1:-1, 1:-1], tiles[0, 0, 0, -2, 1:-1, 1:-1])
        self.assert_allclose(actual[0, 0, 0, -1, 1:-1, 1:-1], tiles[0, 0, 0, 1, 1:-1, 1:-1])

    def test_public_vector_update_on_one_device_preserves_stacked_and_tuple_layouts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        stacked = jnp.stack((tiles, tiles + 1000.0, tiles + 2000.0), axis=0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.VECTOR_TILE_SPEC)

        sharded_stacked = jax.device_put(stacked, sharding)
        stacked_actual = jax.jit(
            lambda field: ghost_cells.update_tiled_ghost_cells(field, static_parameters, 1)
        )(sharded_stacked)
        tuple_actual = ghost_cells.update_tiled_ghost_cells((stacked[0], stacked[1], stacked[2]), static_parameters, 1)

        self.assertEqual(stacked_actual.sharding, sharding)
        self.assertEqual(stacked_actual.shape, stacked.shape)
        self.assertIsInstance(tuple_actual, tuple)
        for i in range(3):
            self.assert_allclose(stacked_actual[i], _reference_update(stacked[i], bcs, tile_shape))
            self.assert_allclose(tuple_actual[i], stacked_actual[i])

    def test_one_device_periodic_fold_uses_self_exchange_and_clears_ghosts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_tiles = jax.device_put(tiles, sharding)
        actual = jax.jit(lambda field: ghost_cells.fold_tiled_ghost_cells(field, static_parameters, 1))(sharded_tiles)

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, -2, 1:-1, 1:-1], 3.0)
        self.assert_allclose(actual[0, 0, 0, 1, 1:-1, 1:-1], 5.0)
        self.assert_allclose(actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -1, :, :], 0.0)

    def test_one_device_conducting_fold_keeps_physical_reflection_sign(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        actual = jax.jit(lambda field: ghost_cells.fold_tiled_ghost_cells(field, static_parameters, 1))(tiles)

        self.assert_allclose(actual, _reference_fold(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 1, 1:-1, 1:-1], -3.0)
        self.assert_allclose(actual[0, 0, 0, -2, 1:-1, 1:-1], -5.0)
        self.assert_allclose(actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -1, :, :], 0.0)

    def test_reduced_axis_on_one_device_does_not_use_periodic_self_exchange(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (1, 3, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        tiles = tiles.at[0, 0, 0, 0, :, :].set(-100.0)
        tiles = tiles.at[0, 0, 0, -1, :, :].set(100.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)

        actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1)

        self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))
        self.assert_allclose(actual[0, 0, 0, 0, :, :], actual[0, 0, 0, 1, :, :])
        self.assert_allclose(actual[0, 0, 0, -1, :, :], actual[0, 0, 0, 1, :, :])

    def test_public_axis_zero_boundary_on_one_device_runs_inside_mapped_path(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        field = jnp.ones(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_field = jax.device_put(field, sharding)
        actual = jax.jit(lambda tiles: ghost_cells.apply_tiled_zero_boundary(tiles, static_parameters, 0, 1))(
            sharded_field
        )

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual[0, 0, 0, 1, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, -1, :, :], 0.0)
        self.assert_allclose(actual[0, 0, 0, 2, :, :], 1.0)

    def test_public_axis_constant_boundary_on_one_device_runs_inside_mapped_path(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        bcs = (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC)
        field = _coordinate_tiles(mesh_shape, tile_shape)
        field = field.at[0, 0, 0, 0, :, :].set(-100.0)
        field = field.at[0, 0, 0, -1, :, :].set(100.0)
        static_parameters = _static_parameters(bcs, tile_shape, mesh_shape)
        sharding = NamedSharding(static_parameters.field_mesh, ghost_cells.SCALAR_TILE_SPEC)

        sharded_field = jax.device_put(field, sharding)
        actual = jax.jit(lambda tiles: ghost_cells.apply_tiled_constant_boundary(tiles, static_parameters, 0, 1))(
            sharded_field
        )

        self.assertEqual(actual.sharding, sharding)
        self.assert_allclose(actual[0, 0, 0, 0, :, :], actual[0, 0, 0, 1, :, :])
        self.assert_allclose(actual[0, 0, 0, -1, :, :], actual[0, 0, 0, -2, :, :])
        self.assert_allclose(actual[0, 0, 0, 2, 1:-1, 1:-1], field[0, 0, 0, 2, 1:-1, 1:-1])

    def test_one_device_uses_same_mapped_periodic_and_conducting_paths(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)

        for bcs in (
            (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
            (BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
        ):
            updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1)
            actual = jax.jit(updater)(tiles)
            self.assert_allclose(actual, _reference_update(tiles, bcs, tile_shape))

    def test_public_update_bc_type_selects_field_or_particle_boundaries(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (BC_ABSORBING, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        field_bc_actual = ghost_cells.update_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=0)
        particle_bc_actual = ghost_cells.update_tiled_ghost_cells(
            tiles,
            static_parameters,
            1,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )

        self.assert_allclose(field_bc_actual, _reference_update(tiles, field_bcs, tile_shape))
        self.assert_allclose(particle_bc_actual, _reference_update(tiles, particle_bcs, tile_shape))
        self.assert_allclose(field_bc_actual[0, 0, 0, 0, 1:-1, 1:-1], tiles[0, 0, 0, -2, 1:-1, 1:-1])
        self.assert_allclose(particle_bc_actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, -1, :, :], 0.0)

    def test_public_fold_bc_type_selects_absorbing_particle_boundaries(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (3, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (BC_ABSORBING, BC_PERIODIC, BC_PERIODIC)
        tiles = jnp.zeros(mesh_shape + (5, 4, 4), dtype=jnp.float64)
        tiles = tiles.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        tiles = tiles.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        field_bc_actual = ghost_cells.fold_tiled_ghost_cells(tiles, static_parameters, 1, bc_type=0)
        particle_bc_actual = ghost_cells.fold_tiled_ghost_cells(
            tiles,
            static_parameters,
            1,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )

        self.assert_allclose(field_bc_actual, _reference_fold(tiles, field_bcs, tile_shape))
        self.assert_allclose(particle_bc_actual, _reference_fold(tiles, particle_bcs, tile_shape))
        self.assert_allclose(field_bc_actual[0, 0, 0, -2, 1:-1, 1:-1], 3.0)
        self.assert_allclose(field_bc_actual[0, 0, 0, 1, 1:-1, 1:-1], 5.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, 1:-1, 1:-1, 1:-1], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(particle_bc_actual[0, 0, 0, -1, :, :], 0.0)

    def test_reduced_absorbing_particle_axis_discards_ghosts(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (1, 2, 2)
        field_bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        particle_bcs = (BC_ABSORBING, BC_PERIODIC, BC_PERIODIC)
        tiles = _coordinate_tiles(mesh_shape, tile_shape)
        static_parameters = _static_parameters(
            field_bcs,
            tile_shape,
            mesh_shape,
            particle_boundary_conditions=particle_bcs,
        )

        refreshed = ghost_cells.update_tiled_ghost_cells(
            tiles,
            static_parameters,
            1,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )
        self.assert_allclose(refreshed, _reference_update(tiles, particle_bcs, tile_shape))
        self.assert_allclose(refreshed[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(refreshed[0, 0, 0, -1, :, :], 0.0)

        deposits = jnp.zeros(mesh_shape + (3, 4, 4), dtype=jnp.float64)
        deposits = deposits.at[0, 0, 0, 0, 1:-1, 1:-1].set(3.0)
        deposits = deposits.at[0, 0, 0, -1, 1:-1, 1:-1].set(5.0)
        folded = ghost_cells.fold_tiled_ghost_cells(
            deposits,
            static_parameters,
            1,
            bc_type=ghost_cells.BC_TYPE_PARTICLE,
        )

        self.assert_allclose(folded, _reference_fold(deposits, particle_bcs, tile_shape))
        self.assert_allclose(folded[0, 0, 0, 1, :, :], 0.0)
        self.assert_allclose(folded[0, 0, 0, 0, :, :], 0.0)
        self.assert_allclose(folded[0, 0, 0, -1, :, :], 0.0)

    def test_validation_rejects_multiple_logical_tiles_per_device(self):
        mesh_shape = (1, 1, 1)
        tile_shape = (2, 2, 2)
        bcs = (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        updater = ghost_cells.make_tiled_halo_op(_mesh(mesh_shape), tile_shape, bcs, 1)
        tiles = _coordinate_tiles((2, 1, 1), tile_shape)

        with self.assertRaisesRegex(ValueError, "one logical tile per device"):
            updater(tiles)


if __name__ == "__main__":
    unittest.main()
