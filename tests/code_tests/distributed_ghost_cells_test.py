"""Single-device numerical tests."""

from tests.support.distributed_ghost_cells_fixtures import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
    DistributedGhostCellsFixtures,
    NamedSharding,
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
            lambda field: ghost_cells.update_tiled_vector_ghost_cells(field, static_parameters, 1)
        )(sharded_stacked)
        tuple_actual = ghost_cells.update_tiled_vector_ghost_cells((stacked[0], stacked[1], stacked[2]), static_parameters, 1)

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
            updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
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
        updater = ghost_cells.make_distributed_ghost_updater(_mesh(mesh_shape), tile_shape, bcs, 1)
        tiles = _coordinate_tiles((2, 1, 1), tile_shape)

        with self.assertRaisesRegex(ValueError, "one logical tile per device"):
            updater(tiles)



if __name__ == "__main__":
    unittest.main()
