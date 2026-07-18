import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.Esirkepov import (
    Esirkepov_current,
    get_3D_esirkepov_weights,
)
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field, fields_for_output
from PyPIC3D.initialization import build_tiled_array, initialize_fields, initialize_simulation
from PyPIC3D.parameters import build_static_parameters
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.first_order_yee import (
    update_E,
)
from PyPIC3D.utilities.grids import build_tiled_yee_grids
from tests.parameter_helpers import field_initialization_parameters, split_test_parameters

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    static_parameters, _ = field_initialization_parameters(world)
    return ghost_cells.update_tiled_ghost_cells(field_tiles, static_parameters, g)


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


def _update_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field,
    )
    return field


class TestTiledEsirkepovCurrent(unittest.TestCase):

    def _build_world(
        self,
        Nx=8,
        Ny=1,
        Nz=1,
        dt=0.05,
        shape_factor=1,
        boundary_conditions=None,
        particle_boundary_conditions=None,
    ):
        x_wind = 4.0 if Nx > 1 else 1.0
        y_wind = 4.0 if Ny > 1 else 1.0
        z_wind = 4.0 if Nz > 1 else 1.0
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
        if particle_boundary_conditions is None:
            particle_boundary_conditions = {"x": 0, "y": 0, "z": 0}
        world = {
            "dx": x_wind / Nx,
            "dy": y_wind / Ny,
            "dz": z_wind / Nz,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": dt,
            "shape_factor": shape_factor,
            "boundary_conditions": boundary_conditions,
            "particle_boundary_conditions": particle_boundary_conditions,
            "current_deposition": "esirkepov",
            "current_filter": "none",
            "guard_cells": 1,
        }
        center_grid = (
            jnp.linspace(-x_wind / 2 - world["dx"] / 2, x_wind / 2 + world["dx"] / 2, Nx + 2),
            jnp.linspace(-y_wind / 2 - world["dy"] / 2, y_wind / 2 + world["dy"] / 2, Ny + 2),
            jnp.linspace(-z_wind / 2 - world["dz"] / 2, z_wind / 2 + world["dz"] / 2, Nz + 2),
        )
        world["grids"] = {"center": center_grid, "vertex": center_grid}
        return world

    def _empty_J(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _species_config(self, charge=-1.0, mass=1.0, weight=0.5):
        return SpeciesConfig(
            charge=jnp.asarray([charge], dtype=float),
            mass=jnp.asarray([mass], dtype=float),
            weight=jnp.asarray([weight], dtype=float),
            update_x=jnp.ones((1, 3), dtype=bool),
            update_u=jnp.ones((1, 3), dtype=bool),
        )

    def _tile_index_for_position(self, position, world, tile_shape):
        x, y, z = [float(component) for component in position]
        tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]

        ix = int(jnp.floor((x + 0.5 * world["x_wind"]) / world["dx"]))
        iy = int(jnp.floor((y + 0.5 * world["y_wind"]) / world["dy"]))
        iz = int(jnp.floor((z + 0.5 * world["z_wind"]) / world["dz"]))

        ix = min(max(ix, 0), int(world["Nx"]) - 1)
        iy = min(max(iy, 0), int(world["Ny"]) - 1)
        iz = min(max(iz, 0), int(world["Nz"]) - 1)

        return ix // tile_nx, iy // tile_ny, iz // tile_nz

    def _empty_tiled_particles(self, world, tile_shape, n_slots):
        tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
        ntx = _tile_axis_count(world["Nx"], tile_nx)
        nty = _tile_axis_count(world["Ny"], tile_ny)
        ntz = _tile_axis_count(world["Nz"], tile_nz)
        shape = (ntx, nty, ntz, 1, n_slots, 3)

        return TiledParticles(
            x=jnp.zeros(shape),
            u=jnp.zeros(shape),
            active=jnp.zeros(shape[:-1], dtype=bool),
        )

    def _set_tiled_particle(self, particles, tile, slot, x, u, active=True):
        tx, ty, tz = tile
        return particles._replace(
            x=particles.x.at[tx, ty, tz, 0, slot].set(jnp.asarray(x, dtype=float)),
            u=particles.u.at[tx, ty, tz, 0, slot].set(jnp.asarray(u, dtype=float)),
            active=particles.active.at[tx, ty, tz, 0, slot].set(active),
        )

    def _particles_from_arrays(self, world, tile_shape, x, u, active_mask=None):
        x = jnp.asarray(x, dtype=float)
        u = jnp.asarray(u, dtype=float)
        active_mask = jnp.ones(x.shape[0], dtype=bool) if active_mask is None else jnp.asarray(active_mask, dtype=bool)
        particles = self._empty_tiled_particles(world, tile_shape, max(1, int(x.shape[0])))
        write_counts = {}

        for particle_index in range(int(x.shape[0])):
            tile = self._tile_index_for_position(x[particle_index], world, tile_shape)
            slot = write_counts.get(tile, 0)
            write_counts[tile] = slot + 1
            particles = self._set_tiled_particle(
                particles,
                tile,
                slot,
                x[particle_index],
                u[particle_index],
                bool(active_mask[particle_index]),
            )

        return particles, self._species_config()

    def _one_tile_particles_from_tiled(self, particles):
        n_species = particles.active.shape[3]
        n_slots = (
            particles.active.shape[0]
            * particles.active.shape[1]
            * particles.active.shape[2]
            * particles.active.shape[4]
        )
        return TiledParticles(
            x=particles.x.transpose(3, 0, 1, 2, 4, 5).reshape(1, 1, 1, n_species, n_slots, 3),
            u=particles.u.transpose(3, 0, 1, 2, 4, 5).reshape(1, 1, 1, n_species, n_slots, 3),
            active=particles.active.transpose(3, 0, 1, 2, 4).reshape(1, 1, 1, n_species, n_slots),
        )

    def _one_dimensional_particles(self, world, x1, tile_shape):
        x = jnp.stack((x1, jnp.zeros_like(x1), jnp.zeros_like(x1)), axis=-1)
        u = jnp.stack(
            (
                jnp.array([0.08, -0.06, 0.05], dtype=float),
                jnp.zeros_like(x1),
                jnp.zeros_like(x1),
            ),
            axis=-1,
        )
        return self._particles_from_arrays(world, tile_shape, x, u)

    def _tile_shape_for_world(self, world):
        return (
            2 if world["Nx"] > 1 else 1,
            2 if world["Ny"] > 1 else 1,
            2 if world["Nz"] > 1 else 1,
        )

    def _one_tile_shape_for_world(self, world):
        return (int(world["Nx"]), int(world["Ny"]), int(world["Nz"]))

    def _assert_tile_scalar_field_rebuilds_halos(self, world, tile_shape, num_guard_cells):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        field = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        field = field.at[0, :, :].set(-1001.0)
        field = field.at[-1, :, :].set(-1002.0)
        field = field.at[:, 0, :].set(-1003.0)
        field = field.at[:, -1, :].set(-1004.0)
        field = field.at[:, :, 0].set(-1005.0)
        field = field.at[:, :, -1].set(-1006.0)

        tiles = tile_scalar_field(field, world, tile_shape, num_guard_cells=num_guard_cells)
        assembled = assemble_tiled_vector_field(
            (tiles, tiles, tiles),
            world,
            tile_shape,
            num_guard_cells=num_guard_cells,
        )[0]
        expected = _update_ghost_cells(
            field,
            world["boundary_conditions"]["x"],
            world["boundary_conditions"]["y"],
            world["boundary_conditions"]["z"],
        )

        self.assertTrue(jnp.allclose(assembled, expected, rtol=1.0e-15, atol=1.0e-15))

    def test_tile_scalar_field_one_guard_rebuilds_periodic_halos_from_interiors(self):
        world = self._build_world(Nx=8, Ny=6, Nz=4)
        self._assert_tile_scalar_field_rebuilds_halos(world, (2, 3, 2), num_guard_cells=1)

    def test_tile_scalar_field_two_guards_rebuilds_periodic_halos_from_interiors(self):
        world = self._build_world(Nx=8, Ny=6, Nz=4)
        self._assert_tile_scalar_field_rebuilds_halos(world, (2, 3, 2), num_guard_cells=2)

    def test_tile_scalar_field_rebuilds_conducting_halos_from_interiors(self):
        world = self._build_world(
            Nx=8,
            Ny=6,
            Nz=4,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC},
        )
        self._assert_tile_scalar_field_rebuilds_halos(world, (2, 3, 2), num_guard_cells=1)

    def test_source_has_no_legacy_particle_fixture_imports(self):
        with open(__file__, "r") as source_file:
            source = source_file.read()

        banned_tokens = [
            "tiled_" + "particle_" + "fixtures",
            "particle_" + "species",
            "to_" + "tiled_" + "particles",
        ]
        for token in banned_tokens:
            self.assertNotIn(token, source)

    def test_esirkepov_uses_shared_ghost_cell_folding(self):
        source = (REPO_ROOT / "PyPIC3D" / "deposition" / "Esirkepov.py").read_text()

        self.assertNotIn("fold_tiled_esirkepov_ghost_cells", source)
        self.assertNotIn("fold_tiled_esirkepov_vector_ghost_cells", source)
        self.assertIn("fold_tiled_vector_ghost_cells", source)
        self.assertIn("bc_type=bc_type", source)

    def _world_with_tiled_grids(self, world, tile_shape):
        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((
            int(world["Nx"]) // int(tile_shape[0]),
            int(world["Ny"]) // int(tile_shape[1]),
            int(world["Nz"]) // int(tile_shape[2]),
        ))
        grid_static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g)
        grid_dynamic_parameters = SimpleNamespace(
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            grids=SimpleNamespace(vertex=grids["vertex"], center=grids["center"]),
        )
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(
            grid_static_parameters,
            grid_dynamic_parameters,
        )
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
        world["grids"] = grids
        return world

    def _initialize_fields(self, world, constants=None):
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
        return initialize_fields(static_parameters, dynamic_parameters)

    def _build_tiled_array(self, world, constants=None):
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
        return build_tiled_array(static_parameters, dynamic_parameters)

    def _assembled_esirkepov_current(self, world, tiled_particles, species_config, constants, tile_shape):
        world = self._world_with_tiled_grids(world, tile_shape)
        g = int(world["guard_cells"])
        _, _, J_template, _, _ = self._initialize_fields(world, constants)
        static_parameters, dynamic_parameters = split_test_parameters(world, constants)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=g)

        return J_tiles, J_from_tiles

    def _assert_tiled_current_matches_reference(self, world, x_old, u, tile_shape=None):
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        if tile_shape is None:
            tile_shape = self._tile_shape_for_world(world)

        tiled_particles, species_config = self._particles_from_arrays(world, tile_shape, x_old, u)
        J_tiles, J_from_tiles = self._assembled_esirkepov_current(
            world,
            tiled_particles,
            species_config,
            constants,
            tile_shape,
        )
        _, J_reference = self._assembled_esirkepov_current(
            world,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            constants,
            self._one_tile_shape_for_world(world),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )

    def _basic_positions_and_velocities(self, world):
        dx, dy, dz = world["dx"], world["dy"], world["dz"]
        x = jnp.array(
            [
                [-0.30 * world["x_wind"] if world["Nx"] > 1 else 0.0,
                 -0.20 * world["y_wind"] if world["Ny"] > 1 else 0.0,
                 -0.10 * world["z_wind"] if world["Nz"] > 1 else 0.0],
                [0.05 * world["x_wind"] if world["Nx"] > 1 else 0.0,
                 0.15 * world["y_wind"] if world["Ny"] > 1 else 0.0,
                 0.20 * world["z_wind"] if world["Nz"] > 1 else 0.0],
                [0.25 * world["x_wind"] if world["Nx"] > 1 else 0.0,
                 -0.30 * world["y_wind"] if world["Ny"] > 1 else 0.0,
                 0.30 * world["z_wind"] if world["Nz"] > 1 else 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.35 * dx / world["dt"] if world["Nx"] > 1 else 0.0,
                 -0.20 * dy / world["dt"] if world["Ny"] > 1 else 0.0,
                 0.25 * dz / world["dt"] if world["Nz"] > 1 else 0.0],
                [-0.15 * dx / world["dt"] if world["Nx"] > 1 else 0.0,
                 0.30 * dy / world["dt"] if world["Ny"] > 1 else 0.0,
                 -0.10 * dz / world["dt"] if world["Nz"] > 1 else 0.0],
                [0.10 * dx / world["dt"] if world["Nx"] > 1 else 0.0,
                 0.15 * dy / world["dt"] if world["Ny"] > 1 else 0.0,
                 0.20 * dz / world["dt"] if world["Nz"] > 1 else 0.0],
            ]
        )
        return x, u

    def test_initialize_fields_builds_tiled_current_with_startup_guard_cells(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        world["guard_cells"] = 2
        tile_shape = (2, 1, 1)
        world["tile_shape"] = tile_shape

        _, _, J_tiles, _, _ = self._initialize_fields(world)

        self.assertEqual(J_tiles[0].shape, (4, 1, 1, 6, 5, 5))
        self.assertTrue(jnp.allclose(J_tiles[0], 0.0))

    def test_tiled_esirkepov_honors_one_guard_cell_startup_depth(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        world["guard_cells"] = 1
        x_old, u = self._basic_positions_and_velocities(world)

        self._assert_tiled_current_matches_reference(world, x_old, u, tile_shape=(2, 1, 1))

    def test_shared_guard_current_tiles_assemble_to_one_guard_global_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        world["guard_cells"] = 2
        tile_shape = (2, 1, 1)
        world["tile_shape"] = tile_shape
        g = int(world["guard_cells"])
        _, _, J_tiles, _, _ = self._initialize_fields(world)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[1, 0, 0, 2, 2, 2].set(3.0)

        assembled = assemble_tiled_vector_field((Jx, Jy, Jz), world, tile_shape, num_guard_cells=g)

        self.assertEqual(assembled[0].shape, (10, 3, 3))
        self.assertEqual(float(assembled[0][3, 1, 1]), 3.0)

    def test_output_adapter_uses_world_guard_cells(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        world["guard_cells"] = 2
        tile_shape = (2, 1, 1)
        world["tile_shape"] = tile_shape
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        g = int(world["guard_cells"])
        E_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        _, _, J_tiles, _, _ = self._initialize_fields(world)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[1, 0, 0, 2, 2, 2].set(7.0)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        fields = (E_tiles, B_tiles, (Jx, Jy, Jz), rho, phi, (E_tiles, B_tiles), None)

        static_parameters = build_static_parameters(world)
        output_fields = fields_for_output(fields, static_parameters)

        self.assertEqual(output_fields[2][0].shape, shape)
        self.assertEqual(float(output_fields[2][0][3, 1, 1]), 7.0)
        self.assertEqual(fields[2][0].shape[-3:], (6, 5, 5))

    def test_update_E_reads_two_guard_current_interior(self):
        world = self._build_world(Nx=4, Ny=1, Nz=1, dt=0.25)
        constants = {"C": 1.0, "eps": 2.0}
        tile_shape = (2, 1, 1)
        world["guard_cells"] = 2
        g = int(world["guard_cells"])
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        E_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        world = self._world_with_tiled_grids(world, tile_shape)
        _, _, J_tiles, _, _ = self._initialize_fields(world, constants)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 2:-2, 2:-2, 2:-2].set(4.0)

        static_parameters, dynamic_parameters = split_test_parameters(world, constants)
        E_after, pml_state = update_E(E_tiles, B_tiles, (Jx, Jy, Jz), static_parameters, dynamic_parameters)

        self.assertIsNone(pml_state)
        self.assertTrue(jnp.allclose(E_after[0][:, :, :, g:-g, g:-g, g:-g], -0.5))

    def test_tiled_esirkepov_matches_global_1d_periodic_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(world, x_old, tile_shape)

        g = int(world["guard_cells"])
        _, _, J_template, _, _ = self._initialize_fields(world, constants)
        static_parameters, dynamic_parameters = split_test_parameters(world, constants)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=g)
        _, J_reference = self._assembled_esirkepov_current(
            world,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            constants,
            self._one_tile_shape_for_world(world),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )

    def test_public_Esirkepov_current_dispatches_tiled_particles_to_tile_local_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        world["guard_cells"] = 2
        world = self._world_with_tiled_grids(world, (2, 1, 1))
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        x_old = jnp.array([-1.10, -0.10, 1.05])
        tiled_particles, species_config = self._one_dimensional_particles(world, x_old, world["tile_shape"])

        _, _, J_template, _, _ = self._initialize_fields(world, constants)
        static_parameters, dynamic_parameters = split_test_parameters(world, constants)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )
        _, J_reference = self._assembled_esirkepov_current(
            world,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            constants,
            self._one_tile_shape_for_world(world),
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )

    def test_tiled_esirkepov_matches_global_current_for_dimensions_and_shapes(self):
        cases = [
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (8, 8, 1),
            (8, 1, 8),
            (1, 8, 8),
            (6, 6, 6),
        ]
        for shape_factor in (1, 2):
            for Nx, Ny, Nz in cases:
                with self.subTest(shape_factor=shape_factor, shape=(Nx, Ny, Nz)):
                    world = self._build_world(Nx=Nx, Ny=Ny, Nz=Nz, dt=0.05, shape_factor=shape_factor)
                    if shape_factor == 2:
                        world["guard_cells"] = 2
                    x_old, u = self._basic_positions_and_velocities(world)
                    self._assert_tiled_current_matches_reference(world, x_old, u)

    def test_tiled_esirkepov_folds_internal_tile_and_periodic_boundary_crossings(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05)
        dx = world["dx"]
        dt = world["dt"]
        x_old = jnp.array(
            [
                [-1.0 * dx, 0.0, 0.0],
                [0.5 * world["x_wind"] - 0.2 * dx, 0.0, 0.0],
                [-0.5 * world["x_wind"] + 0.2 * dx, 0.0, 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.7 * dx / dt, 0.0, 0.0],
                [0.6 * dx / dt, 0.0, 0.0],
                [-0.6 * dx / dt, 0.0, 0.0],
            ]
        )

        self._assert_tiled_current_matches_reference(world, x_old, u, tile_shape=(2, 1, 1))

    def test_tiled_esirkepov_bc_type_selects_field_or_particle_boundaries(self):
        tile_shape = (2, 1, 1)
        periodic_particle_world = self._build_world(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={"x": 0, "y": 0, "z": 0},
        )
        absorbing_particle_world = self._build_world(
            Nx=4,
            Ny=1,
            Nz=1,
            dt=0.05,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            particle_boundary_conditions={"x": 2, "y": 0, "z": 0},
        )
        periodic_particle_world["guard_cells"] = 2
        absorbing_particle_world["guard_cells"] = 2
        periodic_particle_world = self._world_with_tiled_grids(periodic_particle_world, tile_shape)
        absorbing_particle_world = self._world_with_tiled_grids(absorbing_particle_world, tile_shape)

        dx = periodic_particle_world["dx"]
        dt = periodic_particle_world["dt"]
        x_old = jnp.array([[1.75, 0.0, 0.0], [-1.75, 0.0, 0.0]])
        u = jnp.array([[0.6 * dx / dt, 0.0, 0.0], [-0.6 * dx / dt, 0.0, 0.0]])
        particles, species_config = self._particles_from_arrays(periodic_particle_world, tile_shape, x_old, u)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}

        _, _, J_template, _, _ = self._initialize_fields(periodic_particle_world, constants)
        static_periodic, dynamic_periodic = split_test_parameters(periodic_particle_world, constants)
        static_absorbing, dynamic_absorbing = split_test_parameters(absorbing_particle_world, constants)

        field_bc_current = Esirkepov_current(
            particles,
            species_config,
            J_template,
            static_absorbing,
            dynamic_absorbing,
        )
        same_field_bc_current = Esirkepov_current(
            particles,
            species_config,
            J_template,
            static_periodic,
            dynamic_periodic,
        )
        particle_bc_current = Esirkepov_current(
            particles,
            species_config,
            J_template,
            static_absorbing,
            dynamic_absorbing,
            bc_type=1,
        )

        for actual_component, expected_component in zip(field_bc_current, same_field_bc_current):
            self.assertTrue(jnp.allclose(actual_component, expected_component, rtol=1.0e-12, atol=1.0e-12))

        max_difference = max(
            float(jnp.max(jnp.abs(field_component - particle_component)))
            for field_component, particle_component in zip(field_bc_current, particle_bc_current)
        )
        self.assertGreater(max_difference, 1.0e-12)

    def test_tiled_esirkepov_satisfies_tile_local_discrete_continuity(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        dx = world["dx"]
        x_old = jnp.array(
            [
                [-1.0 * dx, 0.0, 0.0],
                [0.5 * world["x_wind"] - 0.2 * dx, 0.0, 0.0],
                [0.25 * world["x_wind"], 0.0, 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.7 * dx / world["dt"], 0.0, 0.0],
                [0.6 * dx / world["dt"], 0.0, 0.0],
                [-0.4 * dx / world["dt"], 0.0, 0.0],
            ]
        )
        tiled_particles, species_config = self._particles_from_arrays(world, tile_shape, x_old, u)
        g = int(world["guard_cells"])
        rho_tiles = self._build_tiled_array(world, constants)
        static_parameters, dynamic_parameters = split_test_parameters(world, constants)

        rho_old = compute_rho(tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        _, _, J_template, _, _ = self._initialize_fields(world, constants)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
        )
        new_particles = update_tiled_particle_positions(tiled_particles, species_config, world["dt"])
        new_particles, overflow = refresh_tiled_particle_tiles(new_particles, static_parameters, dynamic_parameters)
        rho_new = compute_rho(new_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)

        self.assertFalse(bool(overflow))
        drhodt = (rho_new[:, :, :, g:-g, g:-g, g:-g] - rho_old[:, :, :, g:-g, g:-g, g:-g]) / world["dt"]
        dJxdx = (J_tiles[0][:, :, :, g:-g, g:-g, g:-g] - J_tiles[0][:, :, :, g - 1:-g - 1, g:-g, g:-g]) / world["dx"]
        continuity = drhodt + dJxdx
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(drhodt)) + jnp.max(jnp.abs(dJxdx)))

        self.assertLessEqual(float(jnp.max(jnp.abs(continuity))), float(1.0e-12 * scale))

    def test_initialize_tiled_yee_esirkepov_uses_two_guard_current_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "guard_cells": 2,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            _loop, particles, fields, static_parameters, *_rest = initialize_simulation(toml.load(config_path))
            E_tiles, _B_tiles, J_tiles, *_ = fields

            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(E_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(J_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(int(static_parameters.guard_cells), 2)
            self.assertFalse(hasattr(static_parameters, "current_guard_cells"))
            self.assertEqual(static_parameters.current_deposition, "esirkepov")
            self.assertEqual(static_parameters.current_filter, "none")

    def test_initialize_rejects_filtered_esirkepov_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "reject filtered esirkepov",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "digital",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            with self.assertRaisesRegex(ValueError, "Esirkepov current filtering is not supported"):
                initialize_simulation(config)

    def test_tiled_yee_esirkepov_loop_advances_particles_after_deposition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_initial = np.array([-1.5, -0.5, 0.5, 1.5])
            vx_initial = np.array([0.10, -0.05, 0.07, -0.02])
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, x_initial)
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, vx_initial)

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov step smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }
            config_path = os.path.join(tmpdir, "tiled_yee_esirkepov_step.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            (
                loop,
                particles,
                fields,
                static_parameters,
                dynamic_parameters,
                _plotting_parameters,
                _plasma_parameters,
                species_config,
            ) = initialize_simulation(toml.load(config_path))

            particles, fields = loop(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

            active_x = np.asarray(particles.x[..., 0][particles.active])
            expected_x = np.sort(x_initial + vx_initial * float(dynamic_parameters.dt))
            self.assertTrue(np.allclose(np.sort(active_x), expected_x, rtol=1.0e-12, atol=1.0e-12))
            self.assertEqual(int(static_parameters.guard_cells), 2)
            self.assertEqual(fields[2][0].shape[-3:], (6, 5, 5))
            self.assertFalse(bool(fields[-1]))

    def test_tiled_yee_esirkepov_staging_uses_world_contract_not_function_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_initial = np.array([-1.5, -0.5, 0.5, 1.5])
            vx_initial = np.array([0.10, -0.05, 0.07, -0.02])
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, x_initial)
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, vx_initial)

            config = {
                "simulation_parameters": {
                    "name": "tiled yee esirkepov alias staging",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            (
                loop,
                particles,
                fields,
                static_parameters,
                dynamic_parameters,
                _plotting_parameters,
                _plasma_parameters,
                species_config,
            ) = initialize_simulation(config)
            self.assertEqual(static_parameters.current_deposition, "esirkepov")
            particles, fields = loop(
                particles,
                species_config,
                fields,
                static_parameters,
                dynamic_parameters,
            )

            world = {**static_parameters._asdict(), **dynamic_parameters._asdict()}
            world["grids"] = world["grids"]._asdict()
            constants = {"C": float(dynamic_parameters.C), "eps": float(dynamic_parameters.eps), "alpha": float(dynamic_parameters.alpha)}

            reference_particles, reference_species_config = self._particles_from_arrays(
                world,
                self._one_tile_shape_for_world(world),
                jnp.column_stack((jnp.asarray(x_initial), jnp.zeros(4), jnp.zeros(4))),
                jnp.column_stack((jnp.asarray(vx_initial), jnp.zeros(4), jnp.zeros(4))),
            )
            _, reference_J = self._assembled_esirkepov_current(
                world,
                reference_particles,
                reference_species_config,
                constants,
                self._one_tile_shape_for_world(world),
            )
            tiled_J = assemble_tiled_vector_field(
                fields[2],
                world,
                tuple(int(width) for width in world["tile_shape"]),
                num_guard_cells=int(world["guard_cells"]),
            )

            for reference_component, tiled_component in zip(reference_J, tiled_J):
                self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
