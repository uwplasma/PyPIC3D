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
from PyPIC3D.utils import build_collocated_grid, build_yee_grid


jax.config.update("jax_enable_x64", True)

ROUND_OFF_RTOL = 1.0e-11
ROUND_OFF_ATOL = 1.0e-11


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

    def _pad_tiled_particle_capacity(self, tiled_particles, min_slots):
        current_slots = tiled_particles.active.shape[-1]
        if current_slots >= min_slots:
            return tiled_particles

        pad_width = [(0, 0)] * tiled_particles.x.ndim
        pad_width[-2] = (0, min_slots - current_slots)
        x = jnp.pad(tiled_particles.x, pad_width)
        u = jnp.pad(tiled_particles.u, pad_width)

        scalar_pad_width = [(0, 0)] * tiled_particles.active.ndim
        scalar_pad_width[-1] = (0, min_slots - current_slots)

        return TiledParticles(
            x=x,
            u=u,
            active=jnp.pad(tiled_particles.active, scalar_pad_width),
        )

    def _long_two_stream_species(self, world):
        x0 = jnp.linspace(
            -0.5 * world["x_wind"],
            0.5 * world["x_wind"],
            world["Nx"],
            endpoint=False,
        ) + 0.5 * world["dx"]
        x0 = x0 + 0.02 * world["dx"] * jnp.sin(2.0 * jnp.pi * (x0 / world["x_wind"] + 0.5))
        zeros = jnp.zeros_like(x0)
        beam_speed = 0.08

        species_kwargs = {
            "N_particles": world["Nx"],
            "mass": 1.0,
            "weight": 1.0,
            "T": 0.0,
            "x1": x0,
            "x2": zeros,
            "x3": zeros,
            "v2": zeros,
            "v3": zeros,
            "xwind": world["x_wind"],
            "ywind": world["y_wind"],
            "zwind": world["z_wind"],
            "dx": world["dx"],
            "dy": world["dy"],
            "dz": world["dz"],
            "dt": world["dt"],
        }
        return [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]

    def _long_two_stream_state(self):
        world = {
            "Nx": 16,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.25,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.002,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_collocated_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}

        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 1, 1)
        world["tile_shape"] = tile_shape
        Nt = 1500

        E, B, J, rho, phi, external_fields = self._empty_fields(world)
        reference_fields = (E, B, J, rho, phi, external_fields)
        reference_particles = self._long_two_stream_species(world)

        tiled_particles, species_config = to_tiled_particles(
            self._long_two_stream_species(world),
            world,
            {
                "particle_tile_nx": tile_shape[0],
                "particle_tile_ny": tile_shape[1],
                "particle_tile_nz": tile_shape[2],
            },
        )
        tiled_particles = self._pad_tiled_particle_capacity(tiled_particles, min_slots=12)
        tiled_fields = (
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            tile_scalar_field(rho, world, tile_shape),
            tile_scalar_field(phi, world, tile_shape),
            (
                tile_vector_field(external_fields[0], world, tile_shape),
                tile_vector_field(external_fields[1], world, tile_shape),
            ),
            None,
        )

        self._assert_long_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            tile_shape,
            step=0,
        )

        return reference_particles, reference_fields, tiled_particles, tiled_fields, world, constants, tile_shape, Nt

    def _standard_species_rows(self, particles, species_index):
        species = particles[species_index]
        active = species.get_active_mask()
        x = jnp.stack(species.get_forward_position(), axis=1)[active]
        u = jnp.stack(species.get_velocity(), axis=1)[active]
        order = jnp.lexsort((u[:, 0], x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _tiled_species_rows(self, tiled_particles, species_index):
        active = tiled_particles.active[:, :, :, species_index, :]
        x = tiled_particles.x[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        u = tiled_particles.u[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        order = jnp.lexsort((u[:, 0], x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _assert_vector_fields_close(self, reference_field, tiled_field, step, name):
        for component_name, reference_component, tiled_component in zip(("x", "y", "z"), reference_field, tiled_field):
            diff = jnp.max(jnp.abs(reference_component - tiled_component))
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: {name}{component_name} max abs diff {diff}",
            )

    def _assert_long_tiled_state_matches_standard(
        self,
        reference_particles,
        reference_fields,
        tiled_particles,
        tiled_fields,
        world,
        tile_shape,
        step,
    ):
        if len(tiled_fields) > 7:
            self.assertFalse(bool(tiled_fields[-1]), f"step {step}: tiled particle capacity overflowed")

        for species_index in range(len(reference_particles)):
            reference_x, reference_u = self._standard_species_rows(reference_particles, species_index)
            tiled_x, tiled_u = self._tiled_species_rows(tiled_particles, species_index)
            self.assertTrue(
                jnp.allclose(tiled_x, reference_x, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} position mismatch",
            )
            self.assertTrue(
                jnp.allclose(tiled_u, reference_u, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} velocity mismatch",
            )

        reference_E, reference_B, reference_J, reference_rho, reference_phi, *_ = reference_fields
        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, *_ = tiled_fields
        self._assert_vector_fields_close(
            reference_E,
            assemble_tiled_vector_field(E_tiles, world, tile_shape),
            step,
            "E",
        )
        self._assert_vector_fields_close(
            reference_B,
            assemble_tiled_vector_field(B_tiles, world, tile_shape),
            step,
            "B",
        )
        self._assert_vector_fields_close(
            reference_J,
            assemble_tiled_vector_field(J_tiles, world, tile_shape),
            step,
            "J",
        )

        rho_from_tiles = assemble_tiled_scalar_field(rho_tiles, world, tile_shape)
        phi_from_tiles = assemble_tiled_scalar_field(phi_tiles, world, tile_shape)
        self.assertTrue(
            jnp.allclose(
                rho_from_tiles[1:-1, 1:-1, 1:-1],
                reference_rho[1:-1, 1:-1, 1:-1],
                rtol=ROUND_OFF_RTOL,
                atol=ROUND_OFF_ATOL,
            ),
            f"step {step}: rho max abs diff {jnp.max(jnp.abs(rho_from_tiles - reference_rho))}",
        )
        tiled_phi_interior = phi_from_tiles[1:-1, 1:-1, 1:-1]
        reference_phi_interior = reference_phi[1:-1, 1:-1, 1:-1]
        tiled_phi_interior = tiled_phi_interior - jnp.mean(tiled_phi_interior)
        reference_phi_interior = reference_phi_interior - jnp.mean(reference_phi_interior)
        self.assertTrue(
            jnp.allclose(tiled_phi_interior, reference_phi_interior, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
            f"step {step}: phi max abs diff {jnp.max(jnp.abs(tiled_phi_interior - reference_phi_interior))}",
        )

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
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        _, _, _, rho, _, _ = self._empty_fields(world)
        rho_tiles = tile_scalar_field(rho, world, world["tile_shape"])

        reference_rho = compute_rho([self._species(world)], rho, world, constants)
        tiled_rho = compute_tiled_rho_from_tiled_particles(
            tiled_particles,
            species_config,
            rho_tiles,
            world,
            constants,
            tile_shape=world["tile_shape"],
            g=1,
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

        tiled_particles, species_config = to_tiled_particles(self._neutral_species(world), world, self._simulation_parameters())
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
            species_config,
            tiled_fields,
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="tiled_yee",
            tile_shape=world["tile_shape"],
            g=1,
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

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_ELECTROSTATIC") == "1",
        "Set RUN_SLOW_TILED_ELECTROSTATIC=1 to run the 1500-step tiled electrostatic comparison.",
    )
    def test_long_two_stream_tiled_electrostatic_matches_global_step_every_step(self):
        (
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            constants,
            tile_shape,
            Nt,
        ) = self._long_two_stream_state()

        tiled_loop = jax.jit(
            time_loop_electrostatic_tiled,
            static_argnames=("curl_func", "J_func", "solver", "tile_shape", "g", "relativistic", "particle_pusher"),
        )

        for step in range(1, Nt + 1):
            reference_particles, reference_fields = time_loop_electrostatic(
                reference_particles,
                reference_fields,
                world,
                constants,
                unused_curl,
                J_func=None,
                solver="fdtd",
                relativistic=False,
                particle_pusher="boris",
            )
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                tiled_fields,
                world,
                constants,
                unused_curl,
                J_func=None,
                solver="tiled_yee",
                tile_shape=tile_shape,
                g=1,
                relativistic=False,
                particle_pusher="boris",
            )

            self._assert_long_tiled_state_matches_standard(
                reference_particles,
                reference_fields,
                tiled_particles,
                tiled_fields,
                world,
                tile_shape,
                step,
            )

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
            for vertex_axis, center_axis in zip(world["grids"]["vertex"], world["grids"]["center"]):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))
            self.assertEqual(fields[0][0].shape[:3], (4, 1, 1))
            self.assertEqual(fields[3].shape[:3], (4, 1, 1))
            self.assertEqual(fields[4].shape[:3], (4, 1, 1))


if __name__ == "__main__":
    unittest.main()
