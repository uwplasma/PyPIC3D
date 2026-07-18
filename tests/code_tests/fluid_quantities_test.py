import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.diagnostics.fluid_quantities import compute_velocity_field, fluid_velocity
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_scalar_field
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.initial_particles import build_tiled_particles, tiled_species


jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    interior_tiles = field[1:-1, 1:-1, 1:-1]
    interior_tiles = interior_tiles.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
    interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)

    field_tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=field.dtype,
    )
    field_tiles = field_tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)

    world = dict(world)
    world["tile_shape"] = tuple(int(width) for width in tile_shape)
    world["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
    return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g)


class TestTiledFluidQuantities(unittest.TestCase):
    def _build_world(self, shape_factor=2, tile_shape=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        world = {
            "dx": x_wind / 8,
            "dy": y_wind / 6,
            "dz": z_wind / 4,
            "Nx": 8,
            "Ny": 6,
            "Nz": 4,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": 0.08,
            "shape_factor": shape_factor,
            "guard_cells": 2,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return self._world_with_tiled_grids(world, tile_shape=tile_shape)

    def _world_with_tiled_grids(self, world, tile_shape=None):
        if tile_shape is None:
            tile_shape = (world["Nx"], world["Ny"], world["Nz"])

        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((
            int(world["Nx"]) // int(tile_shape[0]),
            int(world["Ny"]) // int(tile_shape[1]),
            int(world["Nz"]) // int(tile_shape[2]),
        ))
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, g)
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
        world["grids"] = grids
        return world

    def _empty_scalar(self, world):
        return jnp.zeros((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

    def _scalar_tiles(self, world):
        return tile_scalar_field(
            self._empty_scalar(world),
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

    def _weighted_average_particles(self, world):
        electrons = tiled_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=1.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([2.0]),
            v2=jnp.array([-4.0]),
            v3=jnp.array([0.5]),
        )
        ions = tiled_species(
            name="ions",
            charge=1.0,
            mass=4.0,
            weight=3.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([10.0]),
            v2=jnp.array([4.0]),
            v3=jnp.array([-1.5]),
        )
        return [electrons, ions]

    def _spread_particles(self, world):
        electrons = tiled_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
        )
        ions = tiled_species(
            name="ions",
            charge=2.0,
            mass=5.0,
            weight=0.25,
            x1=jnp.array([-1.25, -0.20, 0.75]),
            x2=jnp.array([1.15, -0.75, 0.45]),
            x3=jnp.array([0.35, -0.45, 0.85]),
            v1=jnp.array([-0.1, 0.2, -0.25]),
            v2=jnp.array([0.3, -0.05, 0.15]),
            v3=jnp.array([0.1, 0.05, -0.2]),
            active_mask=jnp.array([True, False, True]),
        )
        return [electrons, ions]

    def test_fluid_velocity_computes_weighted_local_average(self):
        world = self._build_world(shape_factor=1)
        particles = self._weighted_average_particles(world)
        tiled_particles, species_config = build_tiled_particles(particles, world, tile_shape=world["tile_shape"])

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            0,
            world,
            world,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))

    def test_compute_velocity_field_uses_selected_direction(self):
        world = self._build_world(shape_factor=1)
        particles = self._weighted_average_particles(world)
        tiled_particles, species_config = build_tiled_particles(particles, world, tile_shape=world["tile_shape"])

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            1,
            world,
            world,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 2.0, rtol=1.0e-12, atol=1.0e-12))

    def test_inactive_slots_do_not_contribute_to_fluid_velocity(self):
        world = self._build_world(shape_factor=1)
        particles = self._weighted_average_particles(world)
        tiled_particles, species_config = build_tiled_particles(
            particles,
            world,
            tile_shape=world["tile_shape"],
            capacity_factor=2.0,
        )

        inactive = ~tiled_particles.active
        x = tiled_particles.x.at[inactive, 0].set(0.0)
        x = x.at[inactive, 1].set(0.0)
        x = x.at[inactive, 2].set(0.0)
        u = tiled_particles.u.at[inactive, 0].set(1000.0)
        noisy_tiled_particles = tiled_particles._replace(x=x, u=u)

        velocity_tiles = fluid_velocity(
            noisy_tiled_particles,
            species_config,
            self._scalar_tiles(world),
            0,
            world,
            world,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))

    def test_empty_cells_are_zero(self):
        world = self._build_world(shape_factor=1)
        particles = self._weighted_average_particles(world)
        tiled_particles, species_config = build_tiled_particles(particles, world, tile_shape=world["tile_shape"])

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            0,
            world,
            world,
        )

        self.assertTrue(jnp.any(velocity_tiles == 0.0))
        self.assertFalse(jnp.any(jnp.isnan(velocity_tiles)))

    def test_tile_major_velocity_matches_single_tile_reference_when_devices_are_available(self):
        tile_shape = (4, 3, 2)
        n_tiles = (8 // tile_shape[0]) * (6 // tile_shape[1]) * (4 // tile_shape[2])
        if len(jax.devices()) < n_tiles:
            self.skipTest("multi-tile field mesh needs one logical device per tile")

        world = self._build_world(shape_factor=2, tile_shape=tile_shape)
        reference_world = self._build_world(shape_factor=2)
        particles = self._spread_particles(world)
        reference_particles = self._spread_particles(reference_world)
        tiled_particles, species_config = build_tiled_particles(particles, world, tile_shape=world["tile_shape"])
        reference_tiled_particles, reference_species_config = build_tiled_particles(
            reference_particles,
            reference_world,
            tile_shape=reference_world["tile_shape"],
        )

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            0,
            world,
            world,
        )

        reference_tiles = fluid_velocity(
            reference_tiled_particles,
            reference_species_config,
            self._scalar_tiles(reference_world),
            0,
            reference_world,
            reference_world,
        )


        velocity_from_tiles = assemble_tiled_scalar_field(
            velocity_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )
        velocity_reference = assemble_tiled_scalar_field(
            reference_tiles,
            reference_world,
            reference_world["tile_shape"],
            num_guard_cells=int(reference_world["guard_cells"]),
        )

        self.assertTrue(
            jnp.allclose(
                velocity_from_tiles[1:-1, 1:-1, 1:-1],
                velocity_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
