import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.utilities.filters import (
    digital_filter,
    digital_filter_vector,
)
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particle_refresh import refresh_tiled_particle_tiles
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_vector_field,
    fold_tiled_ghost_cells,
    tile_scalar_field,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


jax.config.update("jax_enable_x64", True)


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


class TestDirectDepositionTiled(unittest.TestCase):
    def _build_world(self, Nx=8, Ny=6, Nz=4, dt=0.05, boundary_conditions=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
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
            "shape_factor": 1,
            "guard_cells": 1,
            "boundary_conditions": boundary_conditions,
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _empty_J(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _empty_J_tiles(self, world):
        tile_shape = tuple(int(width) for width in world["tile_shape"])
        tile_nx, tile_ny, tile_nz = tile_shape
        g = int(world["guard_cells"])
        shape = (
            world["Nx"] // tile_nx,
            world["Ny"] // tile_ny,
            world["Nz"] // tile_nz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        )
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _tile_shape(self, simulation_parameters):
        return (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )

    def _world_with_tiled_grids(self, world, tile_shape):
        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, g)
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
        world["grids"] = grids
        return world

    def _one_tile_parameters(self, world):
        return {
            "particle_tile_nx": world["Nx"],
            "particle_tile_ny": world["Ny"],
            "particle_tile_nz": world["Nz"],
        }

    def _centered_tiled_particles(self, particles, world, simulation_parameters):
        """
        Build the tiled particle view expected by direct tiled deposition.

        ``J_from_rhov`` stores particles at the forward position and steps them
        back by ``0.5*u*dt`` internally.  The tiled deposition kernel is already
        centered, so the test view applies that half-step before deposition and
        refreshes tile ownership at the centered position.
        """

        tiled_particles, species_config = to_tiled_particles(particles, world, simulation_parameters)
        tiled_particles = tiled_particles._replace(
            x=tiled_particles.x - 0.5 * tiled_particles.u * world["dt"]
        )

        centered_particles, overflow = refresh_tiled_particle_tiles(
            tiled_particles,
            world,
            self._tile_shape(simulation_parameters),
        )
        self.assertFalse(bool(overflow))

        return centered_particles, species_config

    def _assembled_tiled_current(self, particles, world, simulation_parameters, constants, filter="none"):
        tile_shape = self._tile_shape(simulation_parameters)
        world = self._world_with_tiled_grids(world, tile_shape)
        tiled_particles, species_config = self._centered_tiled_particles(particles, world, simulation_parameters)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(world),
            constants,
            world,
            filter=filter,
        )
        g = int(world["guard_cells"])
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=g)

        return J_tiles, J_from_tiles

    def _compare_tiled_to_one_tile(self, particles, world, simulation_parameters, filter="none", alpha=1.0):
        constants = {"C": 3.0e8, "alpha": alpha}
        J_tiles, J_from_tiles = self._assembled_tiled_current(
            particles, world, simulation_parameters, constants, filter=filter
        )
        _, J_reference = self._assembled_tiled_current(
            particles, world, self._one_tile_parameters(world), constants, filter=filter
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))

    def test_public_J_from_rhov_rejects_flat_particles(self):
        world = self._build_world()
        constants = {"C": 3.0e8, "alpha": 1.0}
        simulation_parameters = self._one_tile_parameters(world)
        world = self._world_with_tiled_grids(world, self._tile_shape(simulation_parameters))
        species = particle_species(
            name="flat direct current",
            N_particles=1,
            charge=-1.0,
            mass=1.0,
            T=1.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([0.0]),
            v2=jnp.array([0.0]),
            v3=jnp.array([0.0]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        _, species_config = to_tiled_particles([species], world, simulation_parameters)

        with self.assertRaises(AttributeError):
            J_from_rhov([species], species_config, self._empty_J_tiles(world), constants, world, filter="none")

    def test_tiled_direct_deposition_matches_quadratic_with_two_guard_cells(self):
        world = self._build_world(Nx=8, Ny=6, Nz=4)
        world["shape_factor"] = 2
        world["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }

        species = particle_species(
            name="quadratic current",
            N_particles=6,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.55, -0.52, -0.03, 0.49, 0.55, 1.45]),
            x2=jnp.array([-1.10, -0.55, -0.03, 0.02, 0.52, 1.05]),
            x3=jnp.array([-0.70, -0.04, 0.03, 0.31, 0.49, 0.72]),
            v1=jnp.array([0.18, -0.11, 0.07, -0.04, 0.21, -0.16]),
            v2=jnp.array([0.03, 0.17, -0.22, 0.19, -0.08, 0.12]),
            v3=jnp.array([-0.06, 0.24, 0.11, -0.14, 0.05, -0.19]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_matches_quadratic_saved_style_reduced_axes(self):
        world = self._build_world(Nx=20, Ny=1, Nz=1, dt=0.05)
        world["shape_factor"] = 2
        world["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

        species = particle_species(
            name="quadratic reduced current",
            N_particles=8,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.95, -1.51, -1.02, -0.48, 0.02, 0.47, 1.04, 1.88]),
            x2=jnp.zeros(8),
            x3=jnp.zeros(8),
            v1=jnp.array([0.18, -0.11, 0.07, -0.04, 0.21, -0.16, 0.09, -0.13]),
            v2=jnp.array([0.03, 0.17, -0.22, 0.19, -0.08, 0.12, -0.15, 0.05]),
            v3=jnp.array([-0.06, 0.24, 0.11, -0.14, 0.05, -0.19, 0.16, -0.07]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_bilinear_matches_quadratic_reduced_axes(self):
        world = self._build_world(Nx=20, Ny=1, Nz=1, dt=0.05)
        world["shape_factor"] = 2
        world["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 5,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        species = particle_species(
            name="bilinear quadratic reduced current",
            N_particles=8,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.95, -1.51, -1.02, -0.48, 0.02, 0.47, 1.04, 1.88]),
            x2=jnp.zeros(8),
            x3=jnp.zeros(8),
            v1=jnp.array([0.18, -0.11, 0.07, -0.04, 0.21, -0.16, 0.09, -0.13]),
            v2=jnp.array([0.03, 0.17, -0.22, 0.19, -0.08, 0.12, -0.15, 0.05]),
            v3=jnp.array([-0.06, 0.24, 0.11, -0.14, 0.05, -0.19, 0.16, -0.07]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        self._compare_tiled_to_one_tile([species], world, simulation_parameters, filter="bilinear")

    def test_tiled_digital_filter_matches_global_digital_filter(self):
        world = self._build_world(Nx=8, Ny=6, Nz=4)
        tile_shape = (2, 3, 2)
        alpha = 0.6
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        bc_x = world["boundary_conditions"]["x"]
        bc_y = world["boundary_conditions"]["y"]
        bc_z = world["boundary_conditions"]["z"]

        base = jnp.arange(jnp.prod(jnp.asarray(shape)), dtype=jnp.float64).reshape(shape)
        base = update_ghost_cells(base, bc_x, bc_y, bc_z)
        J = (
            base / 17.0,
            -0.5 * base + 0.25,
            jnp.sin(base / 11.0),
        )
        J = tuple(update_ghost_cells(component, bc_x, bc_y, bc_z) for component in J)

        J_tiles = tile_vector_field(J, world, tile_shape)
        filtered_tiles = digital_filter_vector(J_tiles, alpha, num_guard_cells=1)
        filtered_tiles = update_tiled_vector_ghost_cells(filtered_tiles, world, num_guard_cells=1, tile_shape=tile_shape)
        filtered_from_tiles = assemble_tiled_vector_field(filtered_tiles, world, tile_shape)
        filtered_reference = tuple(
            update_ghost_cells(digital_filter(component, alpha), bc_x, bc_y, bc_z)
            for component in J
        )

        for reference_component, tiled_component in zip(filtered_reference, filtered_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))

    def test_fold_tiled_ghost_cells_periodic_adds_current_deposits_to_neighbors(self):
        world = self._build_world(Nx=4, Ny=1, Nz=1)
        world = self._world_with_tiled_grids(world, (2, 1, 1))
        tiles = jnp.zeros((2, 1, 1, 4, 3, 3))
        tiles = tiles.at[0, 0, 0, -1, 1, 1].set(2.0)
        tiles = tiles.at[1, 0, 0, 0, 1, 1].set(3.0)

        folded = fold_tiled_ghost_cells(tiles, world, num_guard_cells=1, tile_shape=world["tile_shape"])

        self.assertEqual(float(folded[1, 0, 0, 1, 1, 1]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, -2, 1, 1]), 3.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, 0, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -1, :, :], 0.0))

    def test_fold_tiled_ghost_cells_two_guard_layers_adds_deposits_to_neighbors(self):
        world = self._build_world(Nx=8, Ny=4, Nz=4)
        num_guard_cells = 2
        world["guard_cells"] = num_guard_cells
        world = self._world_with_tiled_grids(world, (4, 4, 4))
        tiles = jnp.zeros((2, 1, 1, 8, 8, 8))

        tiles = tiles.at[1, 0, 0, 0, 2, 2].set(2.0)
        tiles = tiles.at[1, 0, 0, 1, 2, 2].set(3.0)
        tiles = tiles.at[0, 0, 0, -2, 2, 2].set(5.0)
        tiles = tiles.at[0, 0, 0, -1, 2, 2].set(7.0)

        folded = fold_tiled_ghost_cells(tiles, world, num_guard_cells, tile_shape=world["tile_shape"])

        self.assertEqual(float(folded[0, 0, 0, 4, 2, 2]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, 5, 2, 2]), 3.0)
        self.assertEqual(float(folded[1, 0, 0, 2, 2, 2]), 5.0)
        self.assertEqual(float(folded[1, 0, 0, 3, 2, 2]), 7.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, :num_guard_cells, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -num_guard_cells:, :, :], 0.0))

    def test_fold_tiled_ghost_cells_two_guard_reduced_axis_folds_to_single_active_cell(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        num_guard_cells = 2
        tile_shape = (4, 1, 1)
        world["guard_cells"] = num_guard_cells
        world = self._world_with_tiled_grids(world, tile_shape)
        tiles = jnp.zeros((2, 1, 1, 8, 5, 5))

        tiles = tiles.at[0, 0, 0, 2, 0, 2].set(1.0)
        tiles = tiles.at[0, 0, 0, 2, 1, 2].set(2.0)
        tiles = tiles.at[0, 0, 0, 2, 3, 2].set(3.0)
        tiles = tiles.at[0, 0, 0, 2, 4, 2].set(4.0)

        folded = fold_tiled_ghost_cells(tiles, world, num_guard_cells, tile_shape=tile_shape)

        self.assertEqual(float(folded[0, 0, 0, 2, 2, 2]), 10.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, :, :num_guard_cells, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, :, -num_guard_cells:, :], 0.0))

    def test_fold_tiled_ghost_cells_conducting_reflects_exterior_deposits(self):
        world = self._build_world(
            Nx=4,
            Ny=1,
            Nz=1,
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_PERIODIC, "z": BC_PERIODIC},
        )
        world = self._world_with_tiled_grids(world, (2, 1, 1))
        tiles = jnp.zeros((2, 1, 1, 4, 3, 3))
        tiles = tiles.at[0, 0, 0, 0, 1, 1].set(4.0)
        tiles = tiles.at[-1, 0, 0, -1, 1, 1].set(7.0)
        tiles = tiles.at[0, 0, 0, -1, 1, 1].set(2.0)
        tiles = tiles.at[1, 0, 0, 0, 1, 1].set(3.0)

        folded = fold_tiled_ghost_cells(tiles, world, num_guard_cells=1, tile_shape=world["tile_shape"])

        self.assertEqual(float(folded[0, 0, 0, 1, 1, 1]), -4.0)
        self.assertEqual(float(folded[-1, 0, 0, -2, 1, 1]), -7.0)
        self.assertEqual(float(folded[1, 0, 0, 1, 1, 1]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, -2, 1, 1]), 3.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, 0, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -1, :, :], 0.0))

    def test_fold_tiled_ghost_cells_matches_global_fold_for_mixed_boundaries(self):
        world = self._build_world(
            Nx=4,
            Ny=4,
            Nz=2,
            boundary_conditions={"x": BC_PERIODIC, "y": BC_CONDUCTING, "z": BC_PERIODIC},
        )
        tile_shape = (2, 2, 1)
        world = self._world_with_tiled_grids(world, tile_shape)
        field = jnp.zeros((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))
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

        folded_tiles = fold_tiled_ghost_cells(tiles, world, num_guard_cells=1, tile_shape=tile_shape)
        folded_from_tiles = assemble_tiled_vector_field((folded_tiles, folded_tiles, folded_tiles), world, tile_shape, num_guard_cells=1)[0]
        folded_reference = update_ghost_cells(
            fold_ghost_cells(
                field,
                world["boundary_conditions"]["x"],
                world["boundary_conditions"]["y"],
                world["boundary_conditions"]["z"],
            ),
            world["boundary_conditions"]["x"],
            world["boundary_conditions"]["y"],
            world["boundary_conditions"]["z"],
        )

        self.assertTrue(jnp.allclose(folded_from_tiles, folded_reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_direct_deposition_returns_only_local_current_tiles(self):
        world = self._build_world()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        constants = {"C": 3.0e8, "alpha": 1.0}
        species = particle_species(
            name="local current tiles",
            N_particles=4,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        tiled_particles, species_config = self._centered_tiled_particles([species], world, simulation_parameters)
        tile_shape = self._tile_shape(simulation_parameters)
        world = self._world_with_tiled_grids(world, tile_shape)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(world),
            constants,
            world,
            filter="none",
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=int(world["guard_cells"]))

        _, J_reference = self._assembled_tiled_current(
            [species], world, self._one_tile_parameters(world), constants, filter="none"
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_direct_deposition_matches_J_from_rhov_for_dummy_species(self):
        world = self._build_world()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }

        electrons = particle_species(
            name="electrons",
            N_particles=4,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        ions = particle_species(
            name="ions",
            N_particles=3,
            charge=2.0,
            mass=4.0,
            weight=0.25,
            T=1.0,
            x1=jnp.array([-1.65, 0.15, 1.75]),
            x2=jnp.array([1.15, -0.75, 0.45]),
            x3=jnp.array([0.35, -0.45, 0.85]),
            v1=jnp.array([-0.1, 0.2, -0.25]),
            v2=jnp.array([0.3, -0.05, 0.15]),
            v3=jnp.array([0.1, 0.05, -0.2]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([electrons, ions], world, simulation_parameters)

    def test_public_J_from_rhov_dispatches_tiled_particles_to_tile_local_current(self):
        world = self._build_world()
        world["guard_cells"] = 2
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        tile_shape = self._tile_shape(simulation_parameters)
        world = self._world_with_tiled_grids(world, tile_shape)
        constants = {"C": 3.0e8, "alpha": 0.6}
        species = particle_species(
            name="public tiled direct current",
            N_particles=4,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        tiled_particles, species_config = self._centered_tiled_particles([species], world, simulation_parameters)

        J_tiles = J_from_rhov(
            tiled_particles,
            species_config,
            self._empty_J_tiles(world),
            constants,
            world,
            filter="digital",
        )
        J_from_tiles = assemble_tiled_vector_field(
            J_tiles,
            world,
            tile_shape,
            num_guard_cells=int(world["guard_cells"]),
        )
        _, J_reference = self._assembled_tiled_current(
            [species], world, self._one_tile_parameters(world), constants, filter="digital"
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_direct_deposition_digital_filter_matches_J_from_rhov(self):
        world = self._build_world()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        constants = {"C": 3.0e8, "alpha": 0.6}
        species = particle_species(
            name="digital filtered current",
            N_particles=4,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        self._compare_tiled_to_one_tile([species], world, simulation_parameters, filter="digital", alpha=0.6)

    def test_tiled_direct_deposition_bilinear_filter_matches_J_from_rhov(self):
        world = self._build_world(Nx=8, Ny=6, Nz=4)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        constants = {"C": 3.0e8, "alpha": 1.0}
        species = particle_species(
            name="bilinear filtered current",
            N_particles=6,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.55, -0.52, -0.03, 0.49, 0.55, 1.45]),
            x2=jnp.array([-1.10, -0.55, -0.03, 0.02, 0.52, 1.05]),
            x3=jnp.array([-0.70, -0.04, 0.03, 0.31, 0.49, 0.72]),
            v1=jnp.array([0.18, -0.11, 0.07, -0.04, 0.21, -0.16]),
            v2=jnp.array([0.03, 0.17, -0.22, 0.19, -0.08, 0.12]),
            v3=jnp.array([-0.06, 0.24, 0.11, -0.14, 0.05, -0.19]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        self._compare_tiled_to_one_tile([species], world, simulation_parameters, filter="bilinear")

    def test_tiled_direct_deposition_respects_active_mask(self):
        world = self._build_world()
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 3,
            "particle_tile_nz": 2,
        }
        species = particle_species(
            name="partly active",
            N_particles=4,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
            active_mask=jnp.array([True, False, True, False]),
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_periodic_boundary_crossing(self):
        world = self._build_world(Nx=10, Ny=6, Nz=4, dt=0.0)
        simulation_parameters = {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }
        species = particle_species(
            name="crossing",
            N_particles=2,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([world["x_wind"] / 2 + 0.1 * world["dx"], -world["x_wind"] / 2 - 0.2 * world["dx"]]),
            x2=jnp.array([0.0, 0.0]),
            x3=jnp.array([0.0, 0.0]),
            v1=jnp.array([0.5, -0.25]),
            v2=jnp.array([0.1, -0.2]),
            v3=jnp.array([0.0, 0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_conducting_boundaries(self):
        world = self._build_world(
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
        species = particle_species(
            name="conducting wall deposits",
            N_particles=3,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([
                -world["x_wind"] / 2 + 0.1 * world["dx"],
                world["x_wind"] / 2 - 0.1 * world["dx"],
                0.0,
            ]),
            x2=jnp.array([
                -world["y_wind"] / 2 + 0.1 * world["dy"],
                0.0,
                world["y_wind"] / 2 - 0.1 * world["dy"],
            ]),
            x3=jnp.array([
                0.0,
                -world["z_wind"] / 2 + 0.1 * world["dz"],
                world["z_wind"] / 2 - 0.1 * world["dz"],
            ]),
            v1=jnp.array([0.5, -0.25, 0.15]),
            v2=jnp.array([0.1, -0.2, 0.3]),
            v3=jnp.array([-0.15, 0.35, -0.1]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_matches_J_from_rhov_for_mixed_boundaries(self):
        world = self._build_world(
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
        species = particle_species(
            name="mixed wall deposits",
            N_particles=4,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([
                -world["x_wind"] / 2 - 0.1 * world["dx"],
                world["x_wind"] / 2 + 0.2 * world["dx"],
                -0.5,
                0.5,
            ]),
            x2=jnp.array([
                -world["y_wind"] / 2 + 0.1 * world["dy"],
                world["y_wind"] / 2 - 0.2 * world["dy"],
                -0.25,
                0.25,
            ]),
            x3=jnp.array([
                0.0,
                0.25,
                -world["z_wind"] / 2 - 0.1 * world["dz"],
                world["z_wind"] / 2 + 0.2 * world["dz"],
            ]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)

    def test_tiled_direct_deposition_reduced_dimensions(self):
        world = self._build_world(Nx=16, Ny=1, Nz=1, dt=0.02)
        simulation_parameters = {
            "particle_tile_nx": 4,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }
        species = particle_species(
            name="one dimensional",
            N_particles=3,
            charge=1.0,
            mass=1.0,
            weight=1.0,
            T=1.0,
            x1=jnp.array([-1.25, 0.15, 1.25]),
            x2=jnp.array([0.0, 0.0, 0.0]),
            x3=jnp.array([0.0, 0.0, 0.0]),
            v1=jnp.array([0.2, -0.1, 0.05]),
            v2=jnp.array([0.3, 0.15, -0.2]),
            v3=jnp.array([-0.05, 0.25, 0.1]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

        self._compare_tiled_to_one_tile([species], world, simulation_parameters)


if __name__ == "__main__":
    unittest.main()
