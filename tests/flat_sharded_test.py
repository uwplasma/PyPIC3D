import unittest
from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boris import particle_push, particle_push_sharded
from PyPIC3D.deposition.J_from_rhov import J_from_rhov, J_from_rhov_sharded
from PyPIC3D.diagnostics.fluid_quantities import compute_velocity_field
from PyPIC3D.evolve import time_loop_electrodynamic, time_loop_electrodynamic_flat_sharded
from PyPIC3D.initialization import default_parameters, prepare_particle_backend
from PyPIC3D.particles.flat_particles import (
    to_flat_particles,
    to_flat_sharded_particles,
    unpad_sharded_array,
)
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


def build_world(nx, ny, nz, dt=0.01):
    world = {
        "dx": 1.0 / nx,
        "dy": 1.0 / ny,
        "dz": 1.0 / nz,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "dt": dt,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }
    vertex_grid, center_grid = build_yee_grid(world)
    world["grids"] = {"vertex": vertex_grid, "center": center_grid}
    return world


def build_constants():
    return {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}


def build_fields(world):
    shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
    ex = jnp.full(shape, 0.1)
    ey = jnp.full(shape, -0.05)
    ez = jnp.full(shape, 0.02)
    bx = jnp.full(shape, 0.03)
    by = jnp.full(shape, -0.04)
    bz = jnp.full(shape, 0.06)
    zeros = jnp.zeros(shape)
    return (ex, ey, ez), (bx, by, bz), (zeros, zeros, zeros), zeros, zeros


def build_species(world):
    x = jnp.array([-0.35, -0.10, 0.12, 0.31, 0.44], dtype=jnp.float64)
    y = jnp.array([0.0, 0.15, -0.2, 0.1, -0.05], dtype=jnp.float64)
    z = jnp.array([0.0, -0.1, 0.2, -0.15, 0.05], dtype=jnp.float64)
    if world["Ny"] == 1:
        y = jnp.zeros_like(y)
    if world["Nz"] == 1:
        z = jnp.zeros_like(z)

    return particle_species(
        name="electrons",
        N_particles=x.shape[0],
        charge=1.0,
        mass=2.0,
        weight=1.0,
        T=0.0,
        v1=jnp.array([0.12, -0.08, 0.05, -0.03, 0.18], dtype=jnp.float64),
        v2=jnp.array([0.04, -0.02, 0.01, 0.03, -0.05], dtype=jnp.float64),
        v3=jnp.array([0.0, 0.02, -0.03, 0.01, 0.04], dtype=jnp.float64),
        x1=x,
        x2=y,
        x3=z,
        xwind=world["x_wind"],
        ywind=world["y_wind"],
        zwind=world["z_wind"],
        dx=world["dx"],
        dy=world["dy"],
        dz=world["dz"],
        dt=world["dt"],
    )


def clone_fields(fields):
    return tuple(
        tuple(jnp.array(component) for component in field) if isinstance(field, tuple) else jnp.array(field)
        for field in fields
    )


class TestFlatShardedBackend(unittest.TestCase):
    def assert_species_match(self, flat_species, sharded_species):
        count = sharded_species.unpadded_particle_count
        self.assertTrue(jnp.allclose(flat_species.x1, unpad_sharded_array(sharded_species.x1, count)))
        self.assertTrue(jnp.allclose(flat_species.x2, unpad_sharded_array(sharded_species.x2, count)))
        self.assertTrue(jnp.allclose(flat_species.x3, unpad_sharded_array(sharded_species.x3, count)))
        self.assertTrue(jnp.allclose(flat_species.v1, unpad_sharded_array(sharded_species.v1, count)))
        self.assertTrue(jnp.allclose(flat_species.v2, unpad_sharded_array(sharded_species.v2, count)))
        self.assertTrue(jnp.allclose(flat_species.v3, unpad_sharded_array(sharded_species.v3, count)))

    def test_to_flat_sharded_particles_pads_and_tracks_metadata(self):
        world = build_world(8, 1, 1)
        species = build_species(world)

        sharded = to_flat_sharded_particles([species], n_devices=2, place_on_devices=False)[0]

        self.assertEqual(sharded.x1.shape, (2, 3))
        self.assertEqual(sharded.unpadded_particle_count, 5)
        self.assertEqual(sharded.padded_particle_count, 6)
        self.assertEqual(sharded.particles_per_shard, 3)
        self.assertFalse(bool(sharded.active_mask[-1, -1]))
        self.assertEqual(float(sharded.weight[-1, -1]), 0.0)
        self.assertEqual(float(sharded.charge[-1, -1]), 0.0)
        self.assertEqual(float(sharded.mass[-1, -1]), 1.0)
        self.assertTrue(
            jnp.allclose(
                unpad_sharded_array(sharded.x1, sharded.unpadded_particle_count),
                species.x1,
            )
        )

    def test_particle_push_matches_flat_backend(self):
        world = build_world(8, 1, 1)
        constants = build_constants()
        species = build_species(world)
        flat = to_flat_particles([species])[0]
        sharded = to_flat_sharded_particles([species], n_devices=2, place_on_devices=False)[0]
        fields = build_fields(world)
        E, B, _, _, _ = fields

        flat = particle_push(
            flat,
            E,
            B,
            world["grids"]["center"],
            world["grids"]["vertex"],
            world["dt"],
            constants,
            relativistic=False,
        )
        sharded = particle_push_sharded(
            sharded,
            E,
            B,
            world["grids"]["center"],
            world["grids"]["vertex"],
            world["dt"],
            constants,
            relativistic=False,
        )

        self.assert_species_match(flat, sharded)

    def test_j_from_rhov_matches_flat_backend(self):
        world = build_world(6, 4, 1)
        constants = build_constants()
        species = build_species(world)
        flat = to_flat_particles([species])
        sharded = to_flat_sharded_particles([species], n_devices=2, place_on_devices=False)
        _, _, J0, _, _ = build_fields(world)

        J_flat = J_from_rhov(flat, J0, constants, world, filter="none")
        J_sharded = J_from_rhov_sharded(sharded, J0, constants, world, filter="none")

        self.assertTrue(jnp.allclose(J_flat[0], J_sharded[0]))
        self.assertTrue(jnp.allclose(J_flat[1], J_sharded[1]))
        self.assertTrue(jnp.allclose(J_flat[2], J_sharded[2]))

    def test_compute_velocity_field_matches_flat_backend_in_2d(self):
        world = build_world(6, 4, 1)
        species = build_species(world)
        flat = to_flat_particles([species])
        sharded = to_flat_sharded_particles([species], n_devices=2, place_on_devices=False)
        rho0 = jnp.zeros((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

        ux_flat = compute_velocity_field(flat, rho0, 0, world)
        ux_sharded = compute_velocity_field(sharded, rho0, 0, world)

        self.assertTrue(jnp.allclose(ux_flat, ux_sharded))
        self.assertAlmostEqual(
            float(jnp.sum(ux_flat[1:-1, 1:-1, 1:-1])),
            float(jnp.mean(species.v1)),
        )

    def test_one_timestep_matches_flat_backend_in_1d_2d_3d(self):
        dimensions = [(8, 1, 1), (6, 4, 1), (4, 3, 2)]
        constants = build_constants()

        for dims in dimensions:
            with self.subTest(dims=dims):
                world = build_world(*dims)
                species = build_species(world)
                flat_particles = to_flat_particles([species])
                sharded_particles = to_flat_sharded_particles(
                    [species],
                    n_devices=2,
                    place_on_devices=False,
                )
                fields = build_fields(world)

                flat_out, flat_fields = time_loop_electrodynamic(
                    flat_particles,
                    clone_fields(fields),
                    world,
                    constants,
                    curl_func=None,
                    J_func=partial(J_from_rhov, filter="none"),
                    solver=None,
                    relativistic=False,
                )
                sharded_out, sharded_fields = time_loop_electrodynamic_flat_sharded(
                    sharded_particles,
                    clone_fields(fields),
                    world,
                    constants,
                    curl_func=None,
                    J_func=partial(J_from_rhov_sharded, filter="none"),
                    solver=None,
                    relativistic=False,
                )

                self.assert_species_match(flat_out[0], sharded_out[0])
                for flat_field, sharded_field in zip(flat_fields[:3], sharded_fields[:3]):
                    for flat_component, sharded_component in zip(flat_field, sharded_field):
                        self.assertTrue(jnp.allclose(flat_component, sharded_component))

    def test_prepare_particle_backend_falls_back_when_gpu_runtime_is_unavailable(self):
        world = build_world(8, 1, 1)
        species = build_species(world)
        _, simulation_parameters, _ = default_parameters()
        simulation_parameters["fast_backend"] = "flat_sharded"
        simulation_parameters["current_calculation"] = "j_from_rhov"
        simulation_parameters["GPUs"] = True

        particles, selected_backend, _ = prepare_particle_backend(
            [species],
            simulation_parameters,
            electrostatic=False,
            solver="fdtd",
            gpu_report={"device_count": 0, "devices": [], "error": "CUDA runtime unavailable"},
        )

        self.assertEqual(selected_backend, "flat")
        self.assertEqual(particles[0].backend, "flat")


if __name__ == "__main__":
    unittest.main()
