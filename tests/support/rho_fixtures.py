import unittest
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_ABSORBING, BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_scalar_field
from PyPIC3D.diagnostics.static_metric import node_weights
from tests.kernel_fixtures import build_tiled_particles, particle_parameters_from_tile_values, particle_species
from tests.kernel_fixtures import kernel_parameters_from_values, tile_scalar_field
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.filters import digital_filter

REPO_ROOT = Path(__file__).resolve().parents[2]


class TiledRhoFixtures:
    def _build_parameter_values(self, shape_factor, dt=0.08, particle_boundary_conditions=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if particle_boundary_conditions is None:
            particle_boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
        parameter_set = {
            "dx": x_wind / 8,
            "dy": y_wind / 6,
            "dz": z_wind / 4,
            "Nx": 8,
            "Ny": 6,
            "Nz": 4,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": dt,
            "shape_factor": shape_factor,
            "guard_cells": 2,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": particle_boundary_conditions,
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        return parameter_set

    def _parameters_with_tiled_grids(self, parameter_set, simulation_parameters):
        tile_shape = (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
        parameter_set = dict(parameter_set)
        grids = dict(parameter_set["grids"])
        parameter_set["tile_shape"] = tile_shape
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
        grids["tiled_center_grid"] = tiled_center_grid
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        parameter_set["grids"] = grids
        return parameter_set

    def _empty_scalar(self, parameter_set):
        return jnp.zeros((parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2))

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 4,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

    def _one_tile_parameters(self, parameter_set):
        return {
            "particle_tile_nx": parameter_set["Nx"],
            "particle_tile_ny": parameter_set["Ny"],
            "particle_tile_nz": parameter_set["Nz"],
        }

    def _particles(self, parameter_set):
        electrons = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75, parameter_set["x_wind"] / 2 - 0.03]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05, 0.0]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80, 0.0]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3, 1.4]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1, 0.0]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15, 0.0]),
        )
        ions = particle_species(
            name="ions",
            charge=2.0,
            mass=4.0,
            weight=0.25,
            x1=jnp.array([-1.25, -0.20, 0.75, 1.35]),
            x2=jnp.array([1.15, -0.75, 0.45, 0.05]),
            x3=jnp.array([0.35, -0.45, 0.85, -0.15]),
            v1=jnp.array([-0.1, 0.2, -0.25, 0.05]),
            v2=jnp.array([0.3, -0.05, 0.15, -0.2]),
            v3=jnp.array([0.1, 0.05, -0.2, 0.3]),
            active_mask=jnp.array([True, False, True, True]),
        )
        return [electrons, ions]

    def _tiled_with_noisy_inactive_slots(self, particles, parameter_set, simulation_parameters=None):
        if simulation_parameters is None:
            simulation_parameters = {
                "particle_tile_nx": parameter_set["tile_shape"][0],
                "particle_tile_ny": parameter_set["tile_shape"][1],
                "particle_tile_nz": parameter_set["tile_shape"][2],
            }
        particle_static, particle_dynamic = particle_parameters_from_tile_values(parameter_set, simulation_parameters)
        tiled_particles, species_config = build_tiled_particles(particles, particle_static, particle_dynamic)

        inactive = ~tiled_particles.active
        x = tiled_particles.x.at[inactive, 0].set(0.33)
        x = x.at[inactive, 1].set(-0.27)
        x = x.at[inactive, 2].set(0.18)
        u = tiled_particles.u.at[inactive, 0].set(4.0)

        return tiled_particles._replace(x=x, u=u), species_config

    def _zero_species_velocities(self, particles):
        for species in particles:
            species["u"] = jnp.zeros_like(species["u"])
        return particles

    def _zero_tiled_velocities(self, tiled_particles):
        return tiled_particles._replace(u=jnp.zeros_like(tiled_particles.u))

    def _deposit_and_assemble(self, particles, parameter_set, simulation_parameters, dynamic_values):
        parameter_set = self._parameters_with_tiled_grids(parameter_set, simulation_parameters)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, parameter_set, simulation_parameters)
        rho_tiles = tile_scalar_field(
            self._empty_scalar(parameter_set),
            parameter_set,
            parameter_set["tile_shape"],
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        rho_tiles = compute_rho(tiled_particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
        rho = assemble_tiled_scalar_field(
            rho_tiles,
            parameter_set,
            parameter_set["tile_shape"],
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        return rho_tiles, rho

    def _compare_tiled_to_standard(self, shape_factor, alpha, current_filter="none"):
        parameter_set = self._build_parameter_values(shape_factor)
        parameter_set["current_filter"] = current_filter
        dynamic_values = {"alpha": alpha}
        particles = self._particles(parameter_set)
        _, rho_from_tiles = self._deposit_and_assemble(particles, parameter_set, self._simulation_parameters(), dynamic_values)
        _, rho_reference = self._deposit_and_assemble(particles, parameter_set, self._one_tile_parameters(parameter_set), dynamic_values)

        self.assertTrue(
            jnp.allclose(
                rho_from_tiles[1:-1, 1:-1, 1:-1],
                rho_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
