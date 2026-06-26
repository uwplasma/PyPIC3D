import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.rho_tiled import compute_tiled_rho_from_tiled_particles
from PyPIC3D.electrostatic_tiled import time_loop_electrostatic_tiled
from PyPIC3D.evolve import time_loop_electrostatic
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_scalar_field,
    assemble_tiled_vector_field,
    tile_scalar_field,
    tile_vector_field,
)
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


def unused_curl(Ex, Ey, Ez):
    return None


class TestTiledElectrostatic(unittest.TestCase):
    def _build_world(self, shape_factor=1):
        world = {
            "Nx": 8,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.5,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.01,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": shape_factor,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        world["tile_shape"] = (2, 1, 1)
        return world

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

    def _empty_fields(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zero = jnp.zeros(shape)
        E = (zero, zero, zero)
        B = (zero, zero, zero)
        J = (zero, zero, zero)
        external_fields = (E, B)
        return E, B, J, zero, zero, external_fields

    def _species(self, world):
        return particle_species(
            name="electrons",
            N_particles=4,
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            x2=jnp.zeros(4),
            x3=jnp.zeros(4),
            v1=jnp.array([0.10, -0.05, 0.07, -0.02]),
            v2=jnp.zeros(4),
            v3=jnp.zeros(4),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def _neutral_species(self, world):
        electrons = self._species(world)
        ions = particle_species(
            name="ions",
            N_particles=4,
            charge=1.0,
            mass=4.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.75, 1.75]),
            x2=jnp.zeros(4),
            x3=jnp.zeros(4),
            v1=jnp.array([-0.03, 0.04, -0.02, 0.01]),
            v2=jnp.zeros(4),
            v3=jnp.zeros(4),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        return [electrons, ions]

    def test_scalar_tiles_round_trip_global_rho_and_phi(self):
        world = self._build_world()
        tile_shape = world["tile_shape"]
        scalar = jnp.arange((world["Nx"] + 2) * (world["Ny"] + 2) * (world["Nz"] + 2), dtype=float)
        scalar = scalar.reshape((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

        scalar_tiles = tile_scalar_field(scalar, world, tile_shape)
        scalar_from_tiles = assemble_tiled_scalar_field(scalar_tiles, world, tile_shape)

        self.assertTrue(jnp.allclose(scalar_from_tiles[1:-1, 1:-1, 1:-1], scalar[1:-1, 1:-1, 1:-1]))

    def test_tiled_rho_assembles_to_global_rho(self):
        world = self._build_world(shape_factor=1)
        constants = {"alpha": 1.0}
        species = self._species(world)
        tiled_particles = to_tiled_particles([species], world, self._simulation_parameters())
        _, _, _, rho, _, _ = self._empty_fields(world)
        rho_tiles = tile_scalar_field(rho, world, world["tile_shape"])

        reference_rho = compute_rho([self._species(world)], rho, world, constants)
        tiled_rho = compute_tiled_rho_from_tiled_particles(
            tiled_particles,
            rho_tiles,
            world,
            constants,
        )
        assembled_rho = assemble_tiled_scalar_field(tiled_rho, world, world["tile_shape"])

        self.assertTrue(
            jnp.allclose(
                assembled_rho[1:-1, 1:-1, 1:-1],
                reference_rho[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tiled_electrostatic_step_matches_global_step_after_assembly(self):
        world = self._build_world(shape_factor=1)
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        E, B, J, rho, phi, external_fields = self._empty_fields(world)

        reference_particles, reference_fields = time_loop_electrostatic(
            self._neutral_species(world),
            (E, B, J, rho, phi, external_fields),
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles(self._neutral_species(world), world, self._simulation_parameters())
        tiled_fields = (
            tile_vector_field(E, world, world["tile_shape"]),
            tile_vector_field(B, world, world["tile_shape"]),
            tile_vector_field(J, world, world["tile_shape"]),
            tile_scalar_field(rho, world, world["tile_shape"]),
            tile_scalar_field(phi, world, world["tile_shape"]),
            (
                tile_vector_field(external_fields[0], world, world["tile_shape"]),
                tile_vector_field(external_fields[1], world, world["tile_shape"]),
            ),
            None,
        )

        tiled_particles, tiled_fields = time_loop_electrostatic_tiled(
            tiled_particles,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="tiled_yee",
            relativistic=False,
            particle_pusher="boris",
        )

        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, *_ = tiled_fields
        reference_E, reference_B, reference_J, reference_rho, reference_phi, *_ = reference_fields

        for reference, tiled in zip(reference_E, assemble_tiled_vector_field(E_tiles, world, world["tile_shape"])):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-8, atol=1.0e-8))
        for reference, tiled in zip(reference_B, assemble_tiled_vector_field(B_tiles, world, world["tile_shape"])):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(reference_J, assemble_tiled_vector_field(J_tiles, world, world["tile_shape"])):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

        rho_from_tiles = assemble_tiled_scalar_field(rho_tiles, world, world["tile_shape"])
        phi_from_tiles = assemble_tiled_scalar_field(phi_tiles, world, world["tile_shape"])
        self.assertTrue(jnp.allclose(rho_from_tiles[1:-1, 1:-1, 1:-1], reference_rho[1:-1, 1:-1, 1:-1], rtol=1.0e-12, atol=1.0e-12))
        tiled_phi_interior = phi_from_tiles[1:-1, 1:-1, 1:-1]
        reference_phi_interior = reference_phi[1:-1, 1:-1, 1:-1]
        tiled_phi_interior = tiled_phi_interior - jnp.mean(tiled_phi_interior)
        reference_phi_interior = reference_phi_interior - jnp.mean(reference_phi_interior)
        self.assertTrue(jnp.allclose(tiled_phi_interior, reference_phi_interior, rtol=1.0e-8, atol=1.0e-8))

    def test_initialize_simulation_accepts_tiled_electrostatic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled electrostatic init smoke",
                    "output_dir": tmpdir,
                    "solver": "tiled_yee",
                    "electrostatic": True,
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
                    "current_calculation": "j_from_rhov",
                    "filter_j": "none",
                    "fast_backend": "default",
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

            loop, particles, fields, world, simulation_parameters, *_ = initialize_simulation(toml.loads(toml.dumps(config)))

            self.assertIs(loop, time_loop_electrostatic_tiled)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(tuple(world["tile_shape"]), (2, 1, 1))
            self.assertEqual(tuple(simulation_parameters["tile_shape"]), (2, 1, 1))
            self.assertEqual(fields[0][0].shape[:3], (4, 1, 1))
            self.assertEqual(fields[3].shape[:3], (4, 1, 1))
            self.assertEqual(fields[4].shape[:3], (4, 1, 1))


if __name__ == "__main__":
    unittest.main()
