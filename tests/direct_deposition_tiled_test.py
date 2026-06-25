import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.solvers.yee_tiled import assemble_tiled_vector_field, fold_tiled_ghost_cells_periodic
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


class TestDirectDepositionTiled(unittest.TestCase):
    def _build_world(self, Nx=8, Ny=6, Nz=4, dt=0.05):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
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
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _empty_J(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _empty_J_tiles(self, world, simulation_parameters):
        tile_shape = (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
        tile_nx, tile_ny, tile_nz = tile_shape
        shape = (
            world["Nx"] // tile_nx,
            world["Ny"] // tile_ny,
            world["Nz"] // tile_nz,
            tile_nx + 2,
            tile_ny + 2,
            tile_nz + 2,
        )
        return (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))

    def _compare_tiled_to_standard(self, particles, world, simulation_parameters):
        constants = {"C": 3.0e8, "alpha": 1.0}
        tiled_particles = to_tiled_particles(particles, world, simulation_parameters)
        tile_shape = (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )

        J_reference = J_from_rhov(particles, self._empty_J(world), constants, world, filter="none")
        J_tiles = direct_J_from_tiled_particles(
            tiled_particles,
            self._empty_J_tiles(world, simulation_parameters),
            constants,
            world,
            filter="none",
        )
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape)

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12))

    def test_fold_tiled_ghost_cells_periodic_adds_current_deposits_to_neighbors(self):
        tiles = jnp.zeros((2, 1, 1, 4, 3, 3))
        tiles = tiles.at[0, 0, 0, -1, 1, 1].set(2.0)
        tiles = tiles.at[1, 0, 0, 0, 1, 1].set(3.0)

        folded = fold_tiled_ghost_cells_periodic(tiles)

        self.assertEqual(float(folded[1, 0, 0, 1, 1, 1]), 2.0)
        self.assertEqual(float(folded[0, 0, 0, -2, 1, 1]), 3.0)
        self.assertTrue(jnp.allclose(folded[:, :, :, 0, :, :], 0.0))
        self.assertTrue(jnp.allclose(folded[:, :, :, -1, :, :], 0.0))

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
        tiled_particles = to_tiled_particles([species], world, simulation_parameters)

        J_tiles = direct_J_from_tiled_particles(
            tiled_particles,
            self._empty_J_tiles(world, simulation_parameters),
            constants,
            world,
            filter="none",
        )
        tile_shape = (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
        J_reference = J_from_rhov([species], self._empty_J(world), constants, world, filter="none")
        J_from_tiles = assemble_tiled_vector_field(J_tiles, world, tile_shape)

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

        self._compare_tiled_to_standard([electrons, ions], world, simulation_parameters)

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

        self._compare_tiled_to_standard([species], world, simulation_parameters)

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

        self._compare_tiled_to_standard([species], world, simulation_parameters)

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

        self._compare_tiled_to_standard([species], world, simulation_parameters)


if __name__ == "__main__":
    unittest.main()
