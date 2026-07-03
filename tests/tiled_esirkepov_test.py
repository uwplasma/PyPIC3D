import os
import tempfile
import unittest
import functools

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.Esirkepov import (
    _active_stencil_indices,
    _compact_1d_esirkepov_weights,
    _compact_2d_esirkepov_weights,
    Esirkepov_current,
    get_1D_esirkepov_weights,
    get_2D_esirkepov_weights,
)
from PyPIC3D.deposition.current_methods import CURRENT_ESIRKEPOV
from PyPIC3D.deposition.rho_tiled import compute_tiled_rho_from_tiled_particles
from PyPIC3D.diagnostics.output_adapters import fields_for_output
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.tiled_particle_refresh import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_vector_field,
    empty_tiled_scalar_field,
    empty_tiled_vector_field,
    tile_grid_axes,
    tile_vector_field,
    update_tiled_E,
)

jax.config.update("jax_enable_x64", True)


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
            "current_calculation": CURRENT_ESIRKEPOV,
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

    def _species(self, world, x1):
        return particle_species(
            name="electrons",
            N_particles=3,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=x1,
            x2=jnp.zeros_like(x1),
            x3=jnp.zeros_like(x1),
            v1=jnp.array([0.08, -0.06, 0.05]),
            v2=jnp.zeros_like(x1),
            v3=jnp.zeros_like(x1),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def _species_from_arrays(self, world, x, u, weight=0.5, active_mask=None):
        return particle_species(
            name="electrons",
            N_particles=x.shape[0],
            charge=-1.0,
            mass=1.0,
            weight=weight,
            T=1.0,
            x1=x[:, 0],
            x2=x[:, 1],
            x3=x[:, 2],
            v1=u[:, 0],
            v2=u[:, 1],
            v3=u[:, 2],
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
            active_mask=active_mask,
        )

    def _tile_shape_for_world(self, world):
        return (
            2 if world["Nx"] > 1 else 1,
            2 if world["Ny"] > 1 else 1,
            2 if world["Nz"] > 1 else 1,
        )

    def _one_tile_shape_for_world(self, world):
        return (int(world["Nx"]), int(world["Ny"]), int(world["Nz"]))

    def _world_with_tiled_grids(self, world, tile_shape):
        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        grids["tiled_center_grid"] = tile_grid_axes(
            grids["center"],
            world,
            tile_shape,
            num_guard_cells=g,
        )
        grids["tiled_vertex_grid"] = tile_grid_axes(
            grids["vertex"],
            world,
            tile_shape,
            num_guard_cells=g,
        )
        world["grids"] = grids
        return world

    def _assembled_esirkepov_current(self, world, old_species, constants, tile_shape):
        world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        g = int(world["guard_cells"])
        tiled_particles, species_config = to_tiled_particles([old_species], world, simulation_parameters)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            empty_tiled_vector_field(world, tile_shape, num_guard_cells=g),
            constants,
            world,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=g)

        return J_tiles, J_from_tiles

    def _assert_tiled_current_matches_reference(self, world, x_old, u, tile_shape=None):
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        if tile_shape is None:
            tile_shape = self._tile_shape_for_world(world)

        old_species = self._species_from_arrays(world, x_old, u)
        J_tiles, J_from_tiles = self._assembled_esirkepov_current(world, old_species, constants, tile_shape)
        _, J_reference = self._assembled_esirkepov_current(
            world,
            old_species,
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

    def test_empty_tiled_vector_field_uses_startup_guard_cells(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        world["guard_cells"] = 2
        tile_shape = (2, 1, 1)

        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=int(world["guard_cells"]))

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
        g = int(world["guard_cells"])
        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=g)
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
        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=g)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[1, 0, 0, 2, 2, 2].set(7.0)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        fields = (E_tiles, B_tiles, (Jx, Jy, Jz), rho, phi, (E_tiles, B_tiles), None)

        output_fields = fields_for_output(fields, world)

        self.assertEqual(output_fields[2][0].shape, shape)
        self.assertEqual(float(output_fields[2][0][3, 1, 1]), 7.0)
        self.assertEqual(fields[2][0].shape[-3:], (6, 5, 5))

    def test_update_tiled_E_reads_two_guard_current_interior(self):
        world = self._build_world(Nx=4, Ny=1, Nz=1, dt=0.25)
        constants = {"C": 1.0, "eps": 2.0}
        tile_shape = (2, 1, 1)
        world["guard_cells"] = 2
        g = int(world["guard_cells"])
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zeros = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        E_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        B_tiles = tile_vector_field(zeros, world, tile_shape, num_guard_cells=g)
        J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=g)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 2:-2, 2:-2, 2:-2].set(4.0)

        E_after = update_tiled_E(E_tiles, B_tiles, (Jx, Jy, Jz), world, constants, None, tile_shape, g)

        self.assertTrue(jnp.allclose(E_after[0][:, :, :, g:-g, g:-g, g:-g], -0.5))

    def test_reduced_axis_esirkepov_scatter_uses_only_collapsed_stencil_index(self):
        self.assertEqual(_active_stencil_indices(True), (0, 1, 2, 3, 4))
        self.assertEqual(_active_stencil_indices(False), (2,))

    def test_compact_1d_esirkepov_weights_match_full_center_line(self):
        xw = [jnp.array([0.0, 0.0]), jnp.array([0.2, 0.3]), jnp.array([0.6, 0.5]), jnp.array([0.2, 0.2]), jnp.array([0.0, 0.0])]
        oxw = [jnp.array([0.0, 0.0]), jnp.array([0.1, 0.2]), jnp.array([0.7, 0.6]), jnp.array([0.2, 0.2]), jnp.array([0.0, 0.0])]
        yw = [jnp.zeros(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2), jnp.zeros(2)]
        zw = [jnp.zeros(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2), jnp.zeros(2)]
        oyw = yw
        ozw = zw

        Wx_line, Wy_line, Wz_line = _compact_1d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, dim=0)
        Wx_full, Wy_full, Wz_full = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles=2, dim=0)

        self.assertTrue(jnp.allclose(Wx_line, Wx_full[:, 2, 2, :]))
        self.assertTrue(jnp.allclose(Wy_line, Wy_full[:, 2, 2, :]))
        self.assertTrue(jnp.allclose(Wz_line, Wz_full[:, 2, 2, :]))

    def test_compact_2d_esirkepov_weights_match_full_active_plane(self):
        xw = [jnp.array([0.0, 0.0]), jnp.array([0.2, 0.3]), jnp.array([0.6, 0.5]), jnp.array([0.2, 0.2]), jnp.array([0.0, 0.0])]
        oxw = [jnp.array([0.0, 0.0]), jnp.array([0.1, 0.2]), jnp.array([0.7, 0.6]), jnp.array([0.2, 0.2]), jnp.array([0.0, 0.0])]
        yw = [jnp.array([0.0, 0.0]), jnp.array([0.3, 0.2]), jnp.array([0.4, 0.5]), jnp.array([0.3, 0.3]), jnp.array([0.0, 0.0])]
        oyw = [jnp.array([0.0, 0.0]), jnp.array([0.2, 0.1]), jnp.array([0.5, 0.6]), jnp.array([0.3, 0.3]), jnp.array([0.0, 0.0])]
        zw = [jnp.zeros(2), jnp.zeros(2), jnp.ones(2), jnp.zeros(2), jnp.zeros(2)]
        ozw = zw

        Wx_plane, Wy_plane, Wz_plane = _compact_2d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, null_dim=2)
        Wx_full, Wy_full, Wz_full = get_2D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles=2, null_dim=2)

        self.assertTrue(jnp.allclose(Wx_plane, Wx_full[:, :, 2, :]))
        self.assertTrue(jnp.allclose(Wy_plane, Wy_full[:, :, 2, :]))
        self.assertTrue(jnp.allclose(Wz_plane, Wz_full[:, :, 2, :]))

    def test_tiled_esirkepov_matches_global_1d_periodic_current(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        x_old = jnp.array([-1.10, -0.10, 1.05])
        old_species = self._species(world, x_old)

        tiled_particles, species_config = to_tiled_particles([old_species], world, simulation_parameters)
        g = int(world["guard_cells"])
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            empty_tiled_vector_field(world, tile_shape, num_guard_cells=g),
            constants,
            world,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=g)
        _, J_reference = self._assembled_esirkepov_current(
            world,
            old_species,
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
        simulation_parameters = {
            "particle_tile_nx": world["tile_shape"][0],
            "particle_tile_ny": world["tile_shape"][1],
            "particle_tile_nz": world["tile_shape"][2],
        }
        x_old = jnp.array([-1.10, -0.10, 1.05])
        old_species = self._species(world, x_old)

        tiled_particles, species_config = to_tiled_particles([old_species], world, simulation_parameters)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            empty_tiled_vector_field(world, world["tile_shape"], num_guard_cells=int(world["guard_cells"])),
            constants,
            world,
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )
        _, J_reference = self._assembled_esirkepov_current(
            world,
            old_species,
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

    def test_public_Esirkepov_current_rejects_tiled_filter_modes(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        world["guard_cells"] = 2
        tile_shape = (2, 1, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        old_species = self._species(world, jnp.array([-1.10, -0.10, 1.05]))
        tiled_particles, species_config = to_tiled_particles([old_species], world, simulation_parameters)

        with self.assertRaisesRegex(ValueError, "Esirkepov current filtering is not supported"):
            Esirkepov_current(
                tiled_particles,
                species_config,
                empty_tiled_vector_field(world, tile_shape, num_guard_cells=int(world["guard_cells"])),
                constants,
                world,
                filter="digital",
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

    def test_tiled_esirkepov_matches_global_current_with_particle_boundaries(self):
        cases = [
            (
                {"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC},
                {"x": 1, "y": 0, "z": 0},
                jnp.array([[1.75, 0.0, 0.0], [-1.75, 0.0, 0.0]]),
                jnp.array([[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]]),
                (8, 1, 1),
            ),
            (
                {"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_CONDUCTING},
                {"x": 1, "y": 0, "z": 2},
                jnp.array([[1.75, -0.50, 1.75], [-1.75, 0.50, -1.75]]),
                jnp.array([[0.6, 0.4, 0.6], [-0.6, -0.4, -0.6]]),
                (8, 4, 4),
            ),
        ]
        for field_bc, particle_bc, x_old, u_cells, shape in cases:
            with self.subTest(field_bc=field_bc, particle_bc=particle_bc, shape=shape):
                world = self._build_world(
                    Nx=shape[0],
                    Ny=shape[1],
                    Nz=shape[2],
                    dt=0.05,
                    boundary_conditions=field_bc,
                    particle_boundary_conditions=particle_bc,
                )
                world["guard_cells"] = 2
                u = u_cells.at[:, 0].set(u_cells[:, 0] * world["dx"] / world["dt"])
                u = u.at[:, 1].set(u_cells[:, 1] * world["dy"] / world["dt"])
                u = u.at[:, 2].set(u_cells[:, 2] * world["dz"] / world["dt"])
                self._assert_tiled_current_matches_reference(world, x_old, u)

    def test_tiled_esirkepov_satisfies_tile_local_discrete_continuity(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1, dt=0.05, shape_factor=1)
        constants = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
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
        old_species = self._species_from_arrays(world, x_old, u)
        tiled_particles, species_config = to_tiled_particles([old_species], world, simulation_parameters)
        g = int(world["guard_cells"])
        rho_tiles = empty_tiled_scalar_field(world, tile_shape, num_guard_cells=g)

        rho_old = compute_tiled_rho_from_tiled_particles(tiled_particles, species_config, rho_tiles, world, constants, tile_shape=tile_shape, g=g)
        J_tiles = Esirkepov_current(
            tiled_particles,
            species_config,
            empty_tiled_vector_field(world, tile_shape, num_guard_cells=g),
            constants,
            world,
        )
        new_particles = update_tiled_particle_positions(tiled_particles, species_config, world["dt"])
        new_particles, overflow = refresh_tiled_particle_tiles(new_particles, world, tile_shape)
        rho_new = compute_tiled_rho_from_tiled_particles(new_particles, species_config, rho_tiles, world, constants, tile_shape=tile_shape, g=g)

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

            _loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))
            E_tiles, _B_tiles, J_tiles, *_ = fields

            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(E_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(J_tiles[0].shape[-3:], (6, 5, 5))
            self.assertEqual(int(world["guard_cells"]), 2)
            self.assertNotIn("current_guard_cells", world)
            self.assertEqual(int(world["current_calculation"]), CURRENT_ESIRKEPOV)

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

            loop, particles, fields, world, _simulation_parameters, constants, _plotting_parameters, _plasma_parameters, \
                solver, _electrostatic, _verbose, _GPUs, _Nt, curl_func, J_func, relativistic, particle_pusher, species_config = initialize_simulation(toml.load(config_path))

            particles, fields = loop(
                particles,
                species_config,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )

            active_x = np.asarray(particles.x[..., 0][particles.active])
            expected_x = np.sort(x_initial + vx_initial * float(world["dt"]))
            self.assertTrue(np.allclose(np.sort(active_x), expected_x, rtol=1.0e-12, atol=1.0e-12))
            self.assertEqual(int(world["guard_cells"]), 2)
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

            loop, particles, fields, world, _simulation_parameters, constants, _plotting_parameters, _plasma_parameters, \
                solver, _electrostatic, _verbose, _GPUs, _Nt, curl_func, _J_func, relativistic, particle_pusher, species_config = initialize_simulation(config)
            alias_J_func = functools.partial(Esirkepov_current)
            particles, fields = loop(
                particles,
                species_config,
                fields,
                world,
                constants,
                curl_func,
                alias_J_func,
                solver,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )

            old_species = self._species_from_arrays(
                world,
                jnp.column_stack((jnp.asarray(x_initial), jnp.zeros(4), jnp.zeros(4))),
                jnp.column_stack((jnp.asarray(vx_initial), jnp.zeros(4), jnp.zeros(4))),
            )
            _, reference_J = self._assembled_esirkepov_current(
                world,
                old_species,
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
