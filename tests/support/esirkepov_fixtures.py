import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
)
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field, vector_field_for_output
from PyPIC3D.initialization import build_tiled_array, initialize_fields, initialize_simulation
from PyPIC3D.utilities.parameters import build_static_parameters
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.yee.first_order_yee import (
    update_E,
)
from PyPIC3D.utilities.grids import build_tiled_yee_grids
from tests.kernel_fixtures import _tile_axis_count, kernel_parameters_from_values, tile_vector_field

REPO_ROOT = Path(__file__).resolve().parents[2]


class TiledEsirkepovCurrentFixtures:

    def _build_parameter_values(
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
        parameter_set = {
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
            jnp.linspace(-x_wind / 2 - parameter_set["dx"] / 2, x_wind / 2 + parameter_set["dx"] / 2, Nx + 2),
            jnp.linspace(-y_wind / 2 - parameter_set["dy"] / 2, y_wind / 2 + parameter_set["dy"] / 2, Ny + 2),
            jnp.linspace(-z_wind / 2 - parameter_set["dz"] / 2, z_wind / 2 + parameter_set["dz"] / 2, Nz + 2),
        )
        parameter_set["grids"] = {"center": center_grid, "vertex": center_grid}
        return parameter_set

    def _species_config(self, charge=-1.0, mass=1.0, weight=0.5):
        return SpeciesConfig(
            charge=jnp.asarray([charge], dtype=float),
            mass=jnp.asarray([mass], dtype=float),
            weight=jnp.asarray([weight], dtype=float),
            update_x=jnp.ones((1, 3), dtype=bool),
        )

    def _tile_index_for_position(self, position, parameter_set, tile_shape):
        x, y, z = [float(component) for component in position]
        tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]

        ix = int(jnp.floor((x + 0.5 * parameter_set["x_wind"]) / parameter_set["dx"]))
        iy = int(jnp.floor((y + 0.5 * parameter_set["y_wind"]) / parameter_set["dy"]))
        iz = int(jnp.floor((z + 0.5 * parameter_set["z_wind"]) / parameter_set["dz"]))

        ix = min(max(ix, 0), int(parameter_set["Nx"]) - 1)
        iy = min(max(iy, 0), int(parameter_set["Ny"]) - 1)
        iz = min(max(iz, 0), int(parameter_set["Nz"]) - 1)

        return ix // tile_nx, iy // tile_ny, iz // tile_nz

    def _empty_tiled_particles(self, parameter_set, tile_shape, n_slots):
        tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
        ntx = _tile_axis_count(parameter_set["Nx"], tile_nx)
        nty = _tile_axis_count(parameter_set["Ny"], tile_ny)
        ntz = _tile_axis_count(parameter_set["Nz"], tile_nz)
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

    def _particles_from_arrays(self, parameter_set, tile_shape, x, u, active_mask=None):
        x = jnp.asarray(x, dtype=float)
        u = jnp.asarray(u, dtype=float)
        active_mask = jnp.ones(x.shape[0], dtype=bool) if active_mask is None else jnp.asarray(active_mask, dtype=bool)
        particles = self._empty_tiled_particles(parameter_set, tile_shape, max(1, int(x.shape[0])))
        write_counts = {}

        for particle_index in range(int(x.shape[0])):
            tile = self._tile_index_for_position(x[particle_index], parameter_set, tile_shape)
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

        x_by_species = []
        u_by_species = []
        max_slots = 1
        for species_index in range(n_species):
            active = jnp.asarray(jax.device_get(particles.active[:, :, :, species_index, :])).reshape(-1)
            x = jnp.asarray(jax.device_get(particles.x[:, :, :, species_index, :, :])).reshape(-1, 3)[active]
            u = jnp.asarray(jax.device_get(particles.u[:, :, :, species_index, :, :])).reshape(-1, 3)[active]
            x_by_species.append(x)
            u_by_species.append(u)
            max_slots = max(max_slots, int(x.shape[0]))

        x_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots, 3), dtype=particles.x.dtype)
        u_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots, 3), dtype=particles.u.dtype)
        active_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots), dtype=bool)

        for species_index in range(n_species):
            n_active = int(x_by_species[species_index].shape[0])
            if n_active == 0:
                continue
            x_one_tile = x_one_tile.at[0, 0, 0, species_index, :n_active].set(jnp.asarray(x_by_species[species_index]))
            u_one_tile = u_one_tile.at[0, 0, 0, species_index, :n_active].set(jnp.asarray(u_by_species[species_index]))
            active_one_tile = active_one_tile.at[0, 0, 0, species_index, :n_active].set(True)

        return TiledParticles(
            x=x_one_tile,
            u=u_one_tile,
            active=active_one_tile,
        )

    def _one_dimensional_particles(self, parameter_set, x1, tile_shape):
        x = jnp.stack((x1, jnp.zeros_like(x1), jnp.zeros_like(x1)), axis=-1)
        u = jnp.stack(
            (
                jnp.array([0.08, -0.06, 0.05], dtype=float),
                jnp.zeros_like(x1),
                jnp.zeros_like(x1),
            ),
            axis=-1,
        )
        return self._particles_from_arrays(parameter_set, tile_shape, x, u)

    def _tile_shape_for_parameters(self, parameter_set):
        return (
            2 if parameter_set["Nx"] > 1 else 1,
            2 if parameter_set["Ny"] > 1 else 1,
            2 if parameter_set["Nz"] > 1 else 1,
        )

    def _one_tile_shape_for_parameters(self, parameter_set):
        return (int(parameter_set["Nx"]), int(parameter_set["Ny"]), int(parameter_set["Nz"]))

    def _parameters_with_tiled_grids(self, parameter_set, tile_shape):
        g = int(parameter_set["guard_cells"])
        parameter_set = dict(parameter_set)
        grids = dict(parameter_set["grids"])
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        grid_static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g)
        grid_dynamic_parameters = SimpleNamespace(
            dx=parameter_set["dx"],
            dy=parameter_set["dy"],
            dz=parameter_set["dz"],
            grids=SimpleNamespace(vertex=grids["vertex"], center=grids["center"]),
        )
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(
            grid_static_parameters,
            grid_dynamic_parameters,
        )
        grids["tiled_center_grid"] = tiled_center_grid
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        parameter_set["grids"] = grids
        return parameter_set

    def _initialize_fields(self, parameter_set, dynamic_values=None):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        return initialize_fields(static_parameters, dynamic_parameters)

    def _build_tiled_array(self, parameter_set, dynamic_values=None):
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        return build_tiled_array(static_parameters, dynamic_parameters)

    def _assembled_esirkepov_current(self, parameter_set, tiled_particles, species_config, dynamic_values, tile_shape):
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        g = int(parameter_set["guard_cells"])
        _, _, J_template, _, _ = self._initialize_fields(parameter_set, dynamic_values)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        J_tiles = Esirkepov_current(
            tiled_particles,
            update_tiled_particle_positions(tiled_particles, species_config, dynamic_parameters.dt),
            species_config,
            J_template,
            static_parameters,
            dynamic_parameters,
            coordinate_velocity=tiled_particles.u,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=g)

        return J_tiles, J_from_tiles

    def _assert_tiled_current_matches_reference(self, parameter_set, x_old, u, tile_shape=None):
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        if tile_shape is None:
            tile_shape = self._tile_shape_for_parameters(parameter_set)

        tiled_particles, species_config = self._particles_from_arrays(parameter_set, tile_shape, x_old, u)
        J_tiles, J_from_tiles = self._assembled_esirkepov_current(
            parameter_set,
            tiled_particles,
            species_config,
            dynamic_values,
            tile_shape,
        )
        _, J_reference = self._assembled_esirkepov_current(
            parameter_set,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            dynamic_values,
            self._one_tile_shape_for_parameters(parameter_set),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )

    def _basic_positions_and_velocities(self, parameter_set):
        dx, dy, dz = parameter_set["dx"], parameter_set["dy"], parameter_set["dz"]
        x = jnp.array(
            [
                [-0.30 * parameter_set["x_wind"] if parameter_set["Nx"] > 1 else 0.0,
                 -0.20 * parameter_set["y_wind"] if parameter_set["Ny"] > 1 else 0.0,
                 -0.10 * parameter_set["z_wind"] if parameter_set["Nz"] > 1 else 0.0],
                [0.05 * parameter_set["x_wind"] if parameter_set["Nx"] > 1 else 0.0,
                 0.15 * parameter_set["y_wind"] if parameter_set["Ny"] > 1 else 0.0,
                 0.20 * parameter_set["z_wind"] if parameter_set["Nz"] > 1 else 0.0],
                [0.25 * parameter_set["x_wind"] if parameter_set["Nx"] > 1 else 0.0,
                 -0.30 * parameter_set["y_wind"] if parameter_set["Ny"] > 1 else 0.0,
                 0.30 * parameter_set["z_wind"] if parameter_set["Nz"] > 1 else 0.0],
            ]
        )
        u = jnp.array(
            [
                [0.35 * dx / parameter_set["dt"] if parameter_set["Nx"] > 1 else 0.0,
                 -0.20 * dy / parameter_set["dt"] if parameter_set["Ny"] > 1 else 0.0,
                 0.25 * dz / parameter_set["dt"] if parameter_set["Nz"] > 1 else 0.0],
                [-0.15 * dx / parameter_set["dt"] if parameter_set["Nx"] > 1 else 0.0,
                 0.30 * dy / parameter_set["dt"] if parameter_set["Ny"] > 1 else 0.0,
                 -0.10 * dz / parameter_set["dt"] if parameter_set["Nz"] > 1 else 0.0],
                [0.10 * dx / parameter_set["dt"] if parameter_set["Nx"] > 1 else 0.0,
                 0.15 * dy / parameter_set["dt"] if parameter_set["Ny"] > 1 else 0.0,
                 0.20 * dz / parameter_set["dt"] if parameter_set["Nz"] > 1 else 0.0],
            ]
        )
        return x, u


    def _assert_single_tile_continuity(self, parameter_set, x, velocity, current=None):
        shape = self._one_tile_shape_for_parameters(parameter_set)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, shape)
        particles, species = self._particles_from_arrays(parameter_set, shape, x, velocity)
        static, dynamic = kernel_parameters_from_values(parameter_set, {"C": 1.0, "eps": 1.0})
        rho_template = self._build_tiled_array(parameter_set)
        _, _, current_template, _, _ = self._initialize_fields(parameter_set)
        advanced = update_tiled_particle_positions(particles, species, dynamic.dt)
        if current is None:
            current = Esirkepov_current(particles, advanced, species, current_template, static, dynamic,
                                        coordinate_velocity=particles.u)
        advanced, overflow = refresh_tiled_particle_tiles(advanced, static, dynamic)
        self.assertFalse(bool(overflow))
        old = compute_rho(particles, species, rho_template, static, dynamic)
        new = compute_rho(advanced, species, rho_template, static, dynamic)
        g = static.guard_cells
        interior = (slice(None),) * 3 + (slice(g, -g),) * 3
        drho = (new[interior] - old[interior]) / dynamic.dt
        divergence = jnp.zeros_like(drho)
        for axis, (width, spacing) in enumerate(zip(shape, (dynamic.dx, dynamic.dy, dynamic.dz))):
            if width == 1:
                continue
            previous = list(interior)
            previous[axis + 3] = slice(g - 1, -g - 1)
            divergence += (current[axis][interior] - current[axis][tuple(previous)]) / spacing
        scale = jnp.maximum(1., jnp.max(jnp.abs(drho)) + jnp.max(jnp.abs(divergence)))
        self.assertLessEqual(float(jnp.max(jnp.abs(drho + divergence))), float(1.e-12 * scale))
