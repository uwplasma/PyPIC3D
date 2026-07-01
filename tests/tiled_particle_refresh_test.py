import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particle_refresh import (
    _adjacent_tile_offset,
    _refresh_tiled_particle_tiles_compacting,
    _refresh_tiled_particle_tiles_sparse,
    _tiles_need_repack,
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)


jax.config.update("jax_enable_x64", True)


class TestTiledParticleRefresh(unittest.TestCase):
    def _build_world(self):
        return {
            "Nx": 4,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 1.0,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }

    def _species(self, world, x1, v1, active_mask=None, update_x=True, x_bc="periodic"):
        if active_mask is None:
            active_mask = jnp.ones_like(jnp.asarray(x1), dtype=bool)
        n_particles = len(x1)
        return particle_species(
            name="moving",
            N_particles=n_particles,
            charge=2.0,
            mass=3.0,
            weight=4.0,
            T=1.0,
            x1=jnp.asarray(x1, dtype=float),
            x2=jnp.zeros(n_particles),
            x3=jnp.zeros(n_particles),
            v1=jnp.asarray(v1, dtype=float),
            v2=jnp.zeros(n_particles),
            v3=jnp.zeros(n_particles),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
            active_mask=active_mask,
            update_x=update_x,
            x_bc=x_bc,
        )

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

    def _active_rows(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.argsort(x[:, 0])
        return x[order], u[order]

    def test_update_tiled_particle_positions_respects_active_and_update_flags(self):
        world = self._build_world()
        species = self._species(
            world,
            x1=[-1.5, -0.5, 0.5],
            v1=[0.25, 0.5, 0.75],
            active_mask=jnp.array([True, False, True]),
        )
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())

        moved = update_tiled_particle_positions(tiled_particles, species_config, world["dt"])

        x, _ = self._active_rows(moved)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.25, 1.25])))
        self.assertTrue(jnp.allclose(moved.x[~tiled_particles.active], tiled_particles.x[~tiled_particles.active]))

        fixed_species = self._species(world, x1=[-1.5], v1=[0.25], update_x=False)
        fixed, fixed_species_config = to_tiled_particles([fixed_species], world, self._simulation_parameters())
        fixed_moved = update_tiled_particle_positions(fixed, fixed_species_config, world["dt"])
        self.assertTrue(jnp.allclose(fixed_moved.x, fixed.x))

    def test_adjacent_tile_offset_handles_periodic_edges(self):
        source = jnp.array([0, 0, 1, 1])
        dest = jnp.array([0, 1, 0, 1])

        offset = _adjacent_tile_offset(dest, source, tile_count=2)

        self.assertTrue(jnp.array_equal(offset, jnp.array([0, 1, -1, 0])))

    def test_tiles_need_repack_is_false_when_active_particles_stay_in_tile(self):
        active = jnp.array([[[[[True, False]]], [[[True, False]]]]])
        zero_offsets = jnp.zeros_like(active, dtype=int)

        repack = _tiles_need_repack(zero_offsets, zero_offsets, zero_offsets, active)

        self.assertTrue(jnp.array_equal(repack, jnp.zeros(active.shape[:-1], dtype=bool)))

    def test_refresh_moves_particles_to_neighbor_tiles_with_static_shape(self):
        world = self._build_world()
        species = self._species(world, x1=[-1.5, -0.25, 0.25], v1=[0.0, 0.0, 0.0])
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved_x = tiled_particles.x.at[0, 0, 0, 0, 1, 0].set(0.25)
        moved = tiled_particles._replace(x=moved_x)

        refreshed, overflow = refresh_tiled_particle_tiles(moved, world, tile_shape=(2, 1, 1))

        self.assertEqual(refreshed.x.shape, tiled_particles.x.shape)
        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active[0, 0, 0, 0])), 1)
        self.assertEqual(int(jnp.sum(refreshed.active[1, 0, 0, 0])), 2)
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.5, 0.25, 0.25])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.0, 0.0, 0.0])))

    def test_refresh_wraps_periodic_particle_to_opposite_tile(self):
        world = self._build_world()
        species = self._species(world, x1=[1.75], v1=[0.0])
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(x=tiled_particles.x.at[1, 0, 0, 0, 0, 0].set(2.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, world, tile_shape=(2, 1, 1))

        self.assertFalse(bool(overflow))
        self.assertTrue(bool(refreshed.active[0, 0, 0, 0, 0]))
        self.assertTrue(jnp.allclose(refreshed.x[0, 0, 0, 0, 0, 0], -1.75))

    def test_refresh_reflects_particle_from_global_boundary_condition(self):
        world = self._build_world()
        world["particle_boundary_conditions"] = {"x": 1, "y": 0, "z": 0}
        species = self._species(world, x1=[1.75, -1.75], v1=[0.5, -0.25], x_bc="periodic")
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(
            x=tiled_particles.x
            .at[1, 0, 0, 0, 0, 0].set(2.25)
            .at[0, 0, 0, 0, 0, 0].set(-2.10)
        )

        refreshed, overflow = refresh_tiled_particle_tiles(moved, world, tile_shape=(2, 1, 1))

        self.assertFalse(bool(overflow))
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-1.90, 1.75])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.25, -0.5])))

    def test_refresh_absorbs_particle_from_global_boundary_condition(self):
        world = self._build_world()
        world["particle_boundary_conditions"] = {"x": 2, "y": 0, "z": 0}
        species = self._species(world, x1=[1.75, -0.25], v1=[0.5, 0.0], x_bc="periodic")
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(x=tiled_particles.x.at[1, 0, 0, 0, 0, 0].set(2.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, world, tile_shape=(2, 1, 1))

        self.assertFalse(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 1)
        x, u = self._active_rows(refreshed)
        self.assertTrue(jnp.allclose(x[:, 0], jnp.array([-0.25])))
        self.assertTrue(jnp.allclose(u[:, 0], jnp.array([0.0])))

    def test_refresh_reports_overflow_without_changing_shape(self):
        world = self._build_world()
        species = self._species(world, x1=[-1.5, 0.5, 1.5], v1=[0.0, 0.0, 0.0])
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(x=tiled_particles.x.at[0, 0, 0, 0, 0, 0].set(0.25))

        refreshed, overflow = refresh_tiled_particle_tiles(moved, world, tile_shape=(2, 1, 1))

        self.assertEqual(refreshed.x.shape, tiled_particles.x.shape)
        self.assertTrue(bool(overflow))
        self.assertEqual(int(jnp.sum(refreshed.active)), 2)

    def test_sparse_refresh_matches_compacting_refresh_for_neighbor_exchange(self):
        world = self._build_world()
        species = self._species(
            world,
            x1=[-1.5, -0.25, 0.25, 1.25],
            v1=[0.0, 0.0, 0.0, 0.0],
        )
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(
            x=tiled_particles.x
            .at[0, 0, 0, 0, 1, 0].set(0.25)
            .at[1, 0, 0, 0, 1, 0].set(-0.25)
        )

        compacted, compacted_overflow = _refresh_tiled_particle_tiles_compacting(moved, world, tile_shape=(2, 1, 1))
        sparse, sparse_overflow = _refresh_tiled_particle_tiles_sparse(moved, world, tile_shape=(2, 1, 1))

        self.assertEqual(sparse.x.shape, moved.x.shape)
        self.assertEqual(bool(sparse_overflow), bool(compacted_overflow))
        self.assertEqual(int(jnp.sum(sparse.active)), int(jnp.sum(compacted.active)))
        compacted_x, compacted_u = self._active_rows(compacted)
        sparse_x, sparse_u = self._active_rows(sparse)
        self.assertTrue(jnp.allclose(sparse_x, compacted_x))
        self.assertTrue(jnp.allclose(sparse_u, compacted_u))

    def test_sparse_refresh_reports_same_overflow_as_compacting_refresh(self):
        world = self._build_world()
        species = self._species(world, x1=[-1.5, 0.5, 1.5], v1=[0.0, 0.0, 0.0])
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        moved = tiled_particles._replace(x=tiled_particles.x.at[0, 0, 0, 0, 0, 0].set(0.25))

        compacted, compacted_overflow = _refresh_tiled_particle_tiles_compacting(moved, world, tile_shape=(2, 1, 1))
        sparse, sparse_overflow = _refresh_tiled_particle_tiles_sparse(moved, world, tile_shape=(2, 1, 1))

        self.assertEqual(bool(sparse_overflow), bool(compacted_overflow))
        self.assertEqual(int(jnp.sum(sparse.active)), int(jnp.sum(compacted.active)))


if __name__ == "__main__":
    unittest.main()
