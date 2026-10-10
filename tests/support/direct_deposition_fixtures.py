import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.shapes import get_first_order_weights
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import _tile_axis_count, kernel_parameters_from_values


class DirectDepositionFixtures:


    def _build_parameter_values(self, Nx=8, Ny=6, Nz=4, dt=0.05, boundary_conditions=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
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
            "shape_factor": 1,
            "guard_cells": 1,
            "boundary_conditions": boundary_conditions,
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        return parameter_set

    def _empty_J_tiles(self, parameter_set):
        tile_shape = tuple(int(width) for width in parameter_set["tile_shape"])
        tile_nx, tile_ny, tile_nz = tile_shape
        g = int(parameter_set["guard_cells"])
        shape = (
            parameter_set["Nx"] // tile_nx,
            parameter_set["Ny"] // tile_ny,
            parameter_set["Nz"] // tile_nz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        )
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    # create an empty tiled current density field with the appropriate shape based on the parameter_set and tile shape

    def _tile_shape(self, simulation_parameters):
        return (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
    # get the shape of the particle tiles from the simulation parameters

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
    # create a new parameter_set dictionary that includes the tiled grids based on the given tile shape

    def _one_tile_parameters(self, parameter_set):
        return {
            "particle_tile_nx": parameter_set["Nx"],
            "particle_tile_ny": parameter_set["Ny"],
            "particle_tile_nz": parameter_set["Nz"],
        }
    # create simulation parameters for a single tile that covers the entire parameter_set grid

    def _species_config(self, charges, masses, weights, update_x=None):
        n_species = len(charges)
        if update_x is None:
            update_x = [(True, True, True)] * n_species

        return SpeciesConfig(
            charge=jnp.asarray(charges, dtype=float),
            mass=jnp.asarray(masses, dtype=float),
            weight=jnp.asarray(weights, dtype=float),
            update_x=jnp.asarray(update_x, dtype=bool),
        )
    # create a SpeciesConfig object with the given charges, masses, weights, and directional update flags

    def _empty_tiled_particles(self, parameter_set, simulation_parameters, n_species, n_slots):
        tile_nx, tile_ny, tile_nz = self._tile_shape(simulation_parameters)
        ntx = _tile_axis_count(parameter_set["Nx"], tile_nx)
        nty = _tile_axis_count(parameter_set["Ny"], tile_ny)
        ntz = _tile_axis_count(parameter_set["Nz"], tile_nz)
        shape = (ntx, nty, ntz, n_species, n_slots, 3)

        return TiledParticles(
            x=jnp.zeros(shape),
            u=jnp.zeros(shape),
            active=jnp.zeros(shape[:-1], dtype=bool),
        )
    # create an empty TiledParticles object with the appropriate shape based on the parameter_set, simulation parameters, number of species, and number of slots

    def _set_tiled_particle(self, particles, tile, species, slot, x, u, active=True):
        tx, ty, tz = tile
        particles = particles._replace(
            x=particles.x.at[tx, ty, tz, species, slot].set(jnp.asarray(x, dtype=float)),
            u=particles.u.at[tx, ty, tz, species, slot].set(jnp.asarray(u, dtype=float)),
            active=particles.active.at[tx, ty, tz, species, slot].set(active),
        )
        return particles
    # set the position, velocity, and active status of a specific particle in the TiledParticles object based on the given tile, species, slot, position, velocity, and active flag

    def _particles_from_slots(self, parameter_set, simulation_parameters, n_species, n_slots, slots):
        # Positions, species and activity define the physical fixture. Compute
        # ownership instead of baking a particular test topology into slot IDs.
        tile_shape = self._tile_shape(simulation_parameters)
        counts = {}
        owned = []
        for _old_tile, species, _old_slot, position, velocity, active in slots:
            tile = tuple(min(parameter_set[axis] - 1, max(0, int((value + parameter_set[wind] / 2) / parameter_set[spacing]))) // width
                         for value, axis, wind, spacing, width in zip(position, ("Nx", "Ny", "Nz"),
                             ("x_wind", "y_wind", "z_wind"), ("dx", "dy", "dz"), tile_shape))
            slot = counts.get((tile, species), 0)
            counts[tile, species] = slot + 1
            owned.append((tile, species, slot, position, velocity, active))
        particles = self._empty_tiled_particles(parameter_set, simulation_parameters, n_species,
                                               max(8, n_slots, len(slots)))
        for values in owned:
            particles = self._set_tiled_particle(particles, *values)
        return particles
    # create a TiledParticles object from a list of slots, where each slot specifies the tile, species, slot index, position, velocity, and active status of a particle

    def _one_tile_particles_from_tiled(self, particles):
        n_species = particles.active.shape[3]
        active = np.asarray(particles.active).transpose(3, 0, 1, 2, 4).reshape(n_species, -1)
        x = np.asarray(particles.x).transpose(3, 0, 1, 2, 4, 5).reshape(n_species, -1, 3)
        u = np.asarray(particles.u).transpose(3, 0, 1, 2, 4, 5).reshape(n_species, -1, 3)
        # Reuse a small capacity across these fixtures instead of compiling a
        # different reference kernel for each tile topology's inactive padding.
        capacity = max(8, int(active.sum(axis=1).max()))
        flat_x = np.zeros((1, 1, 1, n_species, capacity, 3), dtype=x.dtype)
        flat_u = np.zeros_like(flat_x)
        flat_active = np.zeros(flat_x.shape[:-1], dtype=bool)
        for species in range(n_species):
            count = int(active[species].sum())
            flat_x[0, 0, 0, species, :count] = x[species, active[species]]
            flat_u[0, 0, 0, species, :count] = u[species, active[species]]
            flat_active[0, 0, 0, species, :count] = True
        return TiledParticles(jnp.asarray(flat_x), jnp.asarray(flat_u), jnp.asarray(flat_active))

    def _centered_tiled_particles(self, particles, parameter_set, simulation_parameters):
        """
        Build the tiled particle view expected by direct tiled deposition.

        ``J_from_rhov`` expects particles at the centered direct-current
        deposition position.  These fixtures start from the forward position,
        so the test view applies the half-step before deposition and refreshes
        tile ownership at the centered position.
        """

        particles = particles._replace(x=particles.x - 0.5 * particles.u * parameter_set["dt"])
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)

        centered_particles, overflow = refresh_tiled_particle_tiles(
            particles,
            static_parameters,
            dynamic_parameters,
        )
        self.assertFalse(bool(overflow))
        # ensure that no particles have overflowed their tiles after centering

        return centered_particles

    def _assembled_tiled_current(self, particles, species_config, parameter_set, simulation_parameters, dynamic_values, filter="none"):
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(current_filter=filter)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        g = int(parameter_set["guard_cells"])
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=g)
        # assemble the tiled current density field into a global field for comparison

        return J_tiles, J_from_tiles

    def _compare_tiled_to_one_tile(self, particles, species_config, parameter_set, simulation_parameters, filter="none", alpha=1.0):
        dynamic_values = {"C": 3.0e8, "alpha": alpha}
        J_tiles, J_from_tiles = self._assembled_tiled_current(
            particles, species_config, parameter_set, simulation_parameters, dynamic_values, filter=filter
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter=filter,
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
            # ensure that the tiled current density components have 6 dimensions (tile_x, tile_y, tile_z, tile_nx, tile_ny, tile_nz)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=5.0e-15, atol=5.0e-15))
