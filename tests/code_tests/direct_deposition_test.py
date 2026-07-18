import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.utilities.filters import (
    digital_filter,
    digital_filter_vector,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import kernel_parameters_from_values


jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)
# compute the number of tiles along each axis, ensuring that the number of cells is divisible by the number of cells per tile.


def tile_scalar_field(field, parameter_set, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)
    # get the number of tiles along each axis

    interior_tiles = field[1:-1, 1:-1, 1:-1]
    interior_tiles = interior_tiles.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
    interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)
    # reshape the interior of the field into tiles, and then transpose to get the correct order of axes

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
    # populate the field tiles with the interior tiles, leaving the guard cells as zeros

    parameter_set = dict(parameter_set)
    parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
    parameter_set["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
    static_parameters, _ = kernel_parameters_from_values(parameter_set)
    return ghost_cells.update_tiled_ghost_cells(field_tiles, static_parameters, g)
    # update the guard cells of the tiled field using the ghost_cells function


def tile_vector_field(field, parameter_set, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, parameter_set, tile_shape, num_guard_cells) for component in field)


def _field_static_parameters(parameter_set):
    static_parameters, _ = kernel_parameters_from_values(parameter_set)
    return static_parameters
    # call tile_scalar_field for each component of the vector field and return a tuple of tiled components


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


def _fold_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[1, :, :].add(f[-1, :, :]).at[-2, :, :].add(f[0, :, :]),
        lambda f: f.at[1, :, :].add(-f[0, :, :]).at[-2, :, :].add(-f[-1, :, :]),
        operand=field,
    )
    field = field.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 1, :].add(f[:, -1, :]).at[:, -2, :].add(f[:, 0, :]),
        lambda f: f.at[:, 1, :].add(-f[:, 0, :]).at[:, -2, :].add(-f[:, -1, :]),
        operand=field,
    )
    field = field.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 1].add(f[:, :, -1]).at[:, :, -2].add(f[:, :, 0]),
        lambda f: f.at[:, :, 1].add(-f[:, :, 0]).at[:, :, -2].add(-f[:, :, -1]),
        operand=field,
    )
    field = field.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    return field


class TestDirectDeposition(unittest.TestCase):
    def test_J_from_rhov_hard_codes_particle_bc_for_shared_ghost_cells(self):
        source = (REPO_ROOT / "PyPIC3D" / "deposition" / "J_from_rhov.py").read_text()

        self.assertNotIn('static_argnames=("static_parameters", "bc_type")', source)
        self.assertNotIn("bc_type=bc_type", source)
        self.assertNotIn("bc_type=0", source)
        self.assertIn("fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)", source)
        self.assertIn("update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)", source)
        self.assertIn("update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)", source)

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
        vertex_grid, center_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return parameter_set

    def _empty_J(self, parameter_set):
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

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
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(
            grid_static_parameters,
            grid_dynamic_parameters,
        )
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
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

    def _species_config(self, charges, masses, weights, update_x=None, update_u=None):
        n_species = len(charges)
        if update_x is None:
            update_x = [(True, True, True)] * n_species
        if update_u is None:
            update_u = [(True, True, True)] * n_species

        return SpeciesConfig(
            charge=jnp.asarray(charges, dtype=float),
            mass=jnp.asarray(masses, dtype=float),
            weight=jnp.asarray(weights, dtype=float),
            update_x=jnp.asarray(update_x, dtype=bool),
            update_u=jnp.asarray(update_u, dtype=bool),
        )
    # create a SpeciesConfig object with the given charges, masses, weights, and optional update flags for position and velocity

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
        particles = self._empty_tiled_particles(parameter_set, simulation_parameters, n_species, n_slots)
        for tile, species, slot, position, velocity, active in slots:
            particles = self._set_tiled_particle(
                particles,
                tile,
                species,
                slot,
                position,
                velocity,
                active,
            )
        return particles
    # create a TiledParticles object from a list of slots, where each slot specifies the tile, species, slot index, position, velocity, and active status of a particle

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
    # convert a TiledParticles object into a single-tile representation by transposing and reshaping the arrays to have a single tile dimension, while preserving the species and slot dimensions

    def _centered_tiled_particles(self, particles, parameter_set, simulation_parameters):
        """
        Build the tiled particle view expected by direct tiled deposition.

        ``J_from_rhov`` stores particles at the forward position and steps them
        back by ``0.5*u*dt`` internally.  The tiled deposition kernel is already
        centered, so the test view applies that half-step before deposition and
        refreshes tile ownership at the centered position.
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
            error = jnp.max(jnp.abs(tiled_component - reference_component))
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=5.0e-15, atol=5.0e-15))
            # compare the assembled tiled current density components to the reference components from the single-tile deposition, ensuring they are close within a specified tolerance


    def test_tiled_direct_deposition_matches_quadratic_with_two_guard_cells(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=2,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((1, 0, 0), 0, 0, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((1, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((2, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((2, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # ensure the direct deposition from tiled particles matches the deposition from a single tile representation, using quadratic shape factors and two guard cells

    def test_tiled_direct_deposition_matches_quadratic_saved_style_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test the direct deposition from tiled particles matches single tiled with reduced dimensions

    def test_tiled_direct_deposition_bilinear_matches_quadratic_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=20, Ny=1, Nz=1, dt=0.05)
        parameter_set["shape_factor"] = 2
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=3,
            slots=[
                ((0, 0, 0), 0, 0, (-1.95, 0.0, 0.0), (0.18, 0.03, -0.06), True),
                ((0, 0, 0), 0, 1, (-1.51, 0.0, 0.0), (-0.11, 0.17, 0.24), True),
                ((0, 0, 0), 0, 2, (-1.02, 0.0, 0.0), (0.07, -0.22, 0.11), True),
                ((1, 0, 0), 0, 0, (-0.48, 0.0, 0.0), (-0.04, 0.19, -0.14), True),
                ((2, 0, 0), 0, 0, (0.02, 0.0, 0.0), (0.21, -0.08, 0.05), True),
                ((2, 0, 0), 0, 1, (0.47, 0.0, 0.0), (-0.16, 0.12, -0.19), True),
                ((3, 0, 0), 0, 0, (1.04, 0.0, 0.0), (0.09, -0.15, 0.16), True),
                ((3, 0, 0), 0, 1, (1.88, 0.0, 0.0), (-0.13, 0.05, -0.07), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")
        # test the bilinear filtered direct deposition from tiled particles matches single tiled with reduced dimensions

    def _assert_tile_scalar_field_rebuilds_halos(self, parameter_set, tile_shape, num_guard_cells):
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        field = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        field = field.at[0, :, :].set(-1001.0)
        field = field.at[-1, :, :].set(-1002.0)
        field = field.at[:, 0, :].set(-1003.0)
        field = field.at[:, -1, :].set(-1004.0)
        field = field.at[:, :, 0].set(-1005.0)
        field = field.at[:, :, -1].set(-1006.0)

        tiles = tile_scalar_field(field, parameter_set, tile_shape, num_guard_cells=num_guard_cells)
        assembled = assemble_tiled_vector_field(
            (tiles, tiles, tiles),
            parameter_set,
            tile_shape,
            num_guard_cells=num_guard_cells,
        )[0]
        expected = _update_ghost_cells(
            field,
            parameter_set["boundary_conditions"]["x"],
            parameter_set["boundary_conditions"]["y"],
            parameter_set["boundary_conditions"]["z"],
        )

        self.assertTrue(jnp.allclose(assembled, expected, rtol=1.0e-15, atol=1.0e-15))
        # test that the tiled scalar field correctly rebuilds the halo regions from the interior values, comparing the assembled tiled field to the expected field with updated ghost cells

    def test_tile_scalar_field_one_guard_rebuilds_periodic_halos_from_interiors(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        self._assert_tile_scalar_field_rebuilds_halos(parameter_set, (2, 3, 2), num_guard_cells=1)
        # test that a tiled scalar field with one guard cell correctly rebuilds periodic halo regions from the interior values

    def test_tile_scalar_field_two_guards_rebuilds_periodic_halos_from_interiors(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        self._assert_tile_scalar_field_rebuilds_halos(parameter_set, (2, 3, 2), num_guard_cells=2)
        # test that a tiled scalar field with two guard cells correctly rebuilds periodic halo regions from the interior values

    def test_tile_scalar_field_rebuilds_conducting_halos_from_interiors(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC},
        )
        self._assert_tile_scalar_field_rebuilds_halos(parameter_set, (2, 3, 2), num_guard_cells=1)
        # test that a tiled scalar field with conducting boundary conditions correctly rebuilds halo regions from the interior values

    def test_tiled_digital_filter_matches_global_digital_filter(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        tile_shape = (2, 3, 2)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        alpha = 0.6
        shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
        bc_x = parameter_set["boundary_conditions"]["x"]
        bc_y = parameter_set["boundary_conditions"]["y"]
        bc_z = parameter_set["boundary_conditions"]["z"]

        base = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        base = _update_ghost_cells(base, bc_x, bc_y, bc_z)
        J = (
            base / 17.0,
            -0.5 * base + 0.25,
            jnp.sin(base / 11.0),
        )
        J = tuple(_update_ghost_cells(component, bc_x, bc_y, bc_z) for component in J)

        J_tiles = tile_vector_field(J, parameter_set, tile_shape)
        filtered_tiles = digital_filter_vector(J_tiles, alpha, num_guard_cells=1)
        filtered_tiles = ghost_cells.update_tiled_vector_ghost_cells(
            filtered_tiles,
            _field_static_parameters(parameter_set),
            num_guard_cells=1,
        )
        filtered_from_tiles = assemble_tiled_vector_field(filtered_tiles, parameter_set, tile_shape)
        filtered_reference = tuple(
            _update_ghost_cells(digital_filter(component, alpha), bc_x, bc_y, bc_z)
            for component in J
        )

        for reference_component, tiled_component in zip(filtered_reference, filtered_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))
        # test that the digital filter applied to the tiled vector field matches the result of applying the digital filter to the global vector field, ensuring consistency between tiled and global filtering

    def test_fold_tiled_ghost_cells_periodic_adds_current_deposits_to_neighbors(self):
        parameter_set = self._build_parameter_values(Nx=4, Ny=1, Nz=1)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, (2, 1, 1))
        tiles = jnp.zeros((2, 1, 1, 4, 3, 3))
        tiles = tiles.at[0, 0, 0, -1, 1, 1].set(2.0)
        tiles = tiles.at[1, 0, 0, 0, 1, 1].set(3.0)

        folded = ghost_cells.fold_tiled_ghost_cells(tiles, _field_static_parameters(parameter_set), num_guard_cells=1)

        self.assertEqual(float(folded[1, 0, 0, 1, 1, 1]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, -2, 1, 1]), 3.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, 0, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -1, :, :], 0.0))
        # test that folding tiled ghost cells with periodic boundary conditions correctly adds current deposits to neighboring tiles, and that the guard cells are zeroed out after folding

    def test_fold_tiled_ghost_cells_two_guard_layers_adds_deposits_to_neighbors(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=4, Nz=4)
        num_guard_cells = 2
        parameter_set["guard_cells"] = num_guard_cells
        parameter_set = self._parameters_with_tiled_grids(parameter_set, (4, 4, 4))
        tiles = jnp.zeros((2, 1, 1, 8, 8, 8))

        tiles = tiles.at[1, 0, 0, 0, 2, 2].set(2.0)
        tiles = tiles.at[1, 0, 0, 1, 2, 2].set(3.0)
        tiles = tiles.at[0, 0, 0, -2, 2, 2].set(5.0)
        tiles = tiles.at[0, 0, 0, -1, 2, 2].set(7.0)

        folded = ghost_cells.fold_tiled_ghost_cells(tiles, _field_static_parameters(parameter_set), num_guard_cells)

        self.assertEqual(float(folded[0, 0, 0, 4, 2, 2]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, 5, 2, 2]), 3.0)
        self.assertEqual(float(folded[1, 0, 0, 2, 2, 2]), 5.0)
        self.assertEqual(float(folded[1, 0, 0, 3, 2, 2]), 7.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, :num_guard_cells, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -num_guard_cells:, :, :], 0.0))
        # test that folding tiled ghost cells with two guard layers correctly adds deposits to neighboring tiles, and that the guard cells are zeroed out after folding

    def test_fold_tiled_ghost_cells_two_guard_reduced_axis_folds_to_single_active_cell(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        num_guard_cells = 2
        tile_shape = (4, 1, 1)
        parameter_set["guard_cells"] = num_guard_cells
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiles = jnp.zeros((2, 1, 1, 8, 5, 5))

        tiles = tiles.at[0, 0, 0, 2, 0, 2].set(1.0)
        tiles = tiles.at[0, 0, 0, 2, 1, 2].set(2.0)
        tiles = tiles.at[0, 0, 0, 2, 3, 2].set(3.0)
        tiles = tiles.at[0, 0, 0, 2, 4, 2].set(4.0)

        folded = ghost_cells.fold_tiled_ghost_cells(tiles, _field_static_parameters(parameter_set), num_guard_cells)

        self.assertEqual(float(folded[0, 0, 0, 2, 2, 2]), 10.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, :, :num_guard_cells, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, :, -num_guard_cells:, :], 0.0))
        # test that folding tiled ghost cells with two guard layers in a reduced axis configuration correctly folds all deposits into a single active cell, and that the guard cells are zeroed out after folding

    def test_fold_tiled_ghost_cells_conducting_reflects_exterior_deposits(self):
        parameter_set = self._build_parameter_values(
            Nx=4,
            Ny=1,
            Nz=1,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC},
        )
        parameter_set = self._parameters_with_tiled_grids(parameter_set, (2, 1, 1))
        tiles = jnp.zeros((2, 1, 1, 4, 3, 3))
        tiles = tiles.at[0, 0, 0, 0, 1, 1].set(4.0)
        tiles = tiles.at[-1, 0, 0, -1, 1, 1].set(7.0)
        tiles = tiles.at[0, 0, 0, -1, 1, 1].set(2.0)
        tiles = tiles.at[1, 0, 0, 0, 1, 1].set(3.0)

        folded = ghost_cells.fold_tiled_ghost_cells(tiles, _field_static_parameters(parameter_set), num_guard_cells=1)

        self.assertEqual(float(folded[0, 0, 0, 1, 1, 1]), -4.0)
        self.assertEqual(float(folded[-1, 0, 0, -2, 1, 1]), -7.0)
        self.assertEqual(float(folded[1, 0, 0, 1, 1, 1]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, -2, 1, 1]), 3.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, 0, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -1, :, :], 0.0))
        # test that folding tiled ghost cells with conducting boundary conditions correctly reflects exterior deposits back into the interior, and that the guard cells are zeroed out after folding

    def test_fold_tiled_ghost_cells_matches_global_fold_for_mixed_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=4,
            Ny=4,
            Nz=2,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_PERIODIC},
        )
        tile_shape = (2, 2, 1)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        field = jnp.zeros((parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2))
        tiles = jnp.zeros((2, 2, 2, 4, 4, 3))

        field = field.at[0, 2, 1].set(1.5)
        tiles = tiles.at[0, 0, 0, 0, 2, 1].set(1.5)

        field = field.at[-1, 3, 2].set(-0.5)
        tiles = tiles.at[-1, 1, 1, -1, 1, 1].set(-0.5)

        field = field.at[2, 0, 1].set(3.0)
        tiles = tiles.at[0, 0, 0, 2, 0, 1].set(3.0)

        field = field.at[3, -1, 2].set(-4.0)
        tiles = tiles.at[1, -1, 1, 1, -1, 1].set(-4.0)

        field = field.at[2, 2, 0].set(2.0)
        tiles = tiles.at[0, 0, 0, 2, 2, 0].set(2.0)

        field = field.at[3, 3, -1].set(5.0)
        tiles = tiles.at[1, 1, -1, 1, 1, -1].set(5.0)

        static_parameters = _field_static_parameters(parameter_set)
        folded_tiles = ghost_cells.fold_tiled_ghost_cells(tiles, static_parameters, num_guard_cells=1)
        folded_tiles = ghost_cells.update_tiled_ghost_cells(folded_tiles, static_parameters, num_guard_cells=1)
        folded_from_tiles = assemble_tiled_vector_field((folded_tiles, folded_tiles, folded_tiles), parameter_set, tile_shape, num_guard_cells=1)[0]
        folded_reference = _update_ghost_cells(
            _fold_ghost_cells(
                field,
                parameter_set["boundary_conditions"]["x"],
                parameter_set["boundary_conditions"]["y"],
                parameter_set["boundary_conditions"]["z"],
            ),
            parameter_set["boundary_conditions"]["x"],
            parameter_set["boundary_conditions"]["y"],
            parameter_set["boundary_conditions"]["z"],
        )

        self.assertTrue(jnp.allclose(folded_from_tiles, folded_reference, rtol=1.0e-15, atol=1.0e-15))
        # test that folding tiled ghost cells with mixed boundary conditions matches the result of folding the global field, ensuring consistency between tiled and global folding operations

    def test_tiled_direct_deposition_returns_only_local_current_tiles(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 1.0}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, parameter_set, tile_shape, num_guard_cells=int(parameter_set["guard_cells"]))

        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="none",
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))
        # test that the direct deposition from tiled particles returns only the local current tiles, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_matches_J_from_rhov_for_dummy_species(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }

        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=2,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((0, 2, 1), 1, 0, (-1.65, 1.15, 0.35), (-0.1, 0.3, 0.1), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 0, 0), 1, 0, (0.15, -0.75, -0.45), (0.2, -0.05, 0.05), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 1, 1), 1, 0, (1.75, 0.45, 0.85), (-0.25, 0.15, -0.2), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(
            charges=[-1.0, 2.0],
            masses=[1.0, 4.0],
            weights=[0.5, 0.25],
        )

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test that the direct deposition from tiled particles with multiple species (including a dummy species) matches the deposition from a single-tile representation, ensuring consistency across species

    def test_public_J_from_rhov_dispatches_tiled_particles_to_tile_local_current(self):
        parameter_set = self._build_parameter_values()
        parameter_set["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        tile_shape = self._tile_shape(simulation_parameters)
        parameter_set = self._parameters_with_tiled_grids(parameter_set, tile_shape)
        dynamic_values = {"C": 3.0e8, "alpha": 0.6}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])
        tiled_particles = self._centered_tiled_particles(particles, parameter_set, simulation_parameters)
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(current_filter="digital")

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(parameter_set),
            static_parameters,
            dynamic_parameters,
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            parameter_set,
            tile_shape,
            num_guard_cells=int(parameter_set["guard_cells"]),
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter="digital",
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-15, atol=1.0e-15))
        # test that the public J_from_rhov function correctly dispatches tiled particles to the tile-local current deposition, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_digital_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 0.6}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), True),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="digital", alpha=0.6)
        # test that the direct deposition from tiled particles with a digital filter matches the deposition from a single-tile representation, ensuring consistency between tiled and global deposition with filtering

    def test_tiled_direct_deposition_bilinear_filter_matches_J_from_rhov(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=6, Nz=4)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        dynamic_values = {"C": 3.0e8, "alpha": 1.0}
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=2,
            slots=[
                ((0, 0, 0), 0, 0, (-1.55, -1.10, -0.70), (0.18, 0.03, -0.06), True),
                ((1, 0, 0), 0, 0, (-0.52, -0.55, -0.04), (-0.11, 0.17, 0.24), True),
                ((1, 0, 1), 0, 0, (-0.03, -0.03, 0.03), (0.07, -0.22, 0.11), True),
                ((2, 1, 1), 0, 0, (0.49, 0.02, 0.31), (-0.04, 0.19, -0.14), True),
                ((2, 1, 1), 0, 1, (0.55, 0.52, 0.49), (0.21, -0.08, 0.05), True),
                ((3, 1, 1), 0, 0, (1.45, 1.05, 0.72), (-0.16, 0.12, -0.19), True),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters, filter="bilinear")
        # test that the direct deposition from tiled particles with a bilinear filter matches the deposition from a single-tile representation, ensuring consistency between tiled and global deposition with bilinear filtering

    def test_tiled_direct_deposition_respects_active_mask(self):
        parameter_set = self._build_parameter_values()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, -1.0, -0.65), (0.2, 0.0, -0.05), True),
                ((1, 1, 0), 0, 0, (-0.25, -0.25, -0.15), (-0.1, 0.15, 0.25), False),
                ((2, 1, 1), 0, 0, (0.65, 0.35, 0.25), (0.05, -0.2, 0.1), True),
                ((3, 2, 1), 0, 0, (1.45, 1.05, 0.75), (0.3, 0.1, -0.15), False),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)
        # test that the direct deposition from tiled particles respects the active mask, ensuring that only active particles contribute to the current deposition, and that the assembled tiled current matches the reference current from a single-tile representation

    def test_tiled_direct_deposition_periodic_boundary_crossing(self):
        parameter_set = self._build_parameter_values(Nx=10, Ny=6, Nz=4, dt=0.0)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 1, 1), 0, 0, (-parameter_set["x_wind"] / 2 - 0.2 * parameter_set["dx"], 0.0, 0.0), (-0.25, -0.2, 0.15), True),
                ((4, 1, 1), 0, 0, (parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], 0.0, 0.0), (0.5, 0.1, 0.0), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_conducting_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING},
        )
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 + 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.5, 0.1, -0.15),
                    True,
                ),
                (
                    (3, 1, 0),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], 0.0, -parameter_set["z_wind"] / 2 + 0.1 * parameter_set["dz"]),
                    (-0.25, -0.2, 0.35),
                    True,
                ),
                (
                    (2, 2, 1),
                    0,
                    0,
                    (0.0, parameter_set["y_wind"] / 2 - 0.1 * parameter_set["dy"], parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]),
                    (0.15, 0.3, -0.1),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_mixed_boundaries(self):
        parameter_set = self._build_parameter_values(
            Nx=8,
            Ny=6,
            Nz=4,
            dt=0.0,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_PERIODIC},
        )
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                (
                    (0, 0, 1),
                    0,
                    0,
                    (-parameter_set["x_wind"] / 2 - 0.1 * parameter_set["dx"], -parameter_set["y_wind"] / 2 + 0.1 * parameter_set["dy"], 0.0),
                    (0.2, 0.0, -0.05),
                    True,
                ),
                ((1, 1, 0), 0, 0, (-0.5, -0.25, -parameter_set["z_wind"] / 2 - 0.1 * parameter_set["dz"]), (0.05, -0.2, 0.1), True),
                ((2, 1, 1), 0, 0, (0.5, 0.25, parameter_set["z_wind"] / 2 + 0.2 * parameter_set["dz"]), (0.3, 0.1, -0.15), True),
                (
                    (3, 2, 1),
                    0,
                    0,
                    (parameter_set["x_wind"] / 2 + 0.2 * parameter_set["dx"], parameter_set["y_wind"] / 2 - 0.2 * parameter_set["dy"], 0.25),
                    (-0.1, 0.15, 0.25),
                    True,
                ),
            ],
        )
        species_config = self._species_config(charges=[-1.0], masses=[1.0], weights=[0.5])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)

    def test_tiled_direct_deposition_reduced_dimensions(self):
        parameter_set = self._build_parameter_values(Nx=16, Ny=1, Nz=1, dt=0.02)
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        particles = self._particles_from_slots(
            parameter_set,
            simulation_parameters,
            n_species=1,
            n_slots=1,
            slots=[
                ((0, 0, 0), 0, 0, (-1.25, 0.0, 0.0), (0.2, 0.3, -0.05), True),
                ((2, 0, 0), 0, 0, (0.15, 0.0, 0.0), (-0.1, 0.15, 0.25), True),
                ((3, 0, 0), 0, 0, (1.25, 0.0, 0.0), (0.05, -0.2, 0.1), True),
            ],
        )
        species_config = self._species_config(charges=[1.0], masses=[1.0], weights=[1.0])

        self._compare_tiled_to_one_tile(particles, species_config, parameter_set, simulation_parameters)


if __name__ == "__main__":
    unittest.main()
