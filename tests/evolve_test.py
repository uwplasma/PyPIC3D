import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.evolve import time_loop_electrodynamic
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.utils import build_yee_grid

jax.config.update("jax_enable_x64", True)


def zero_current(particles, J, constants, world):
    return tuple(jnp.zeros_like(comp) for comp in J)


def unused_curl(Ex, Ey, Ez):
    return None


class TestEvolveExternalFields(unittest.TestCase):
    def test_external_electric_field_pushes_particles_without_evolving_maxwell_fields(self):
        world = {
            "Nx": 3,
            "Ny": 3,
            "Nz": 3,
            "dx": 1.0 / 3.0,
            "dy": 1.0 / 3.0,
            "dz": 1.0 / 3.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        external_E = (jnp.ones_like(external_E[0]), external_E[1], external_E[2])
        external_fields = (external_E, external_B)

        particles = [
            particle_species(
                name="test",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=0.0,
                v1=jnp.array([0.0]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([0.0]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            )
        ]
        fields = (E, B, J, rho, phi, external_fields, None)

        particles, fields = time_loop_electrodynamic(
            particles,
            fields,
            world,
            constants,
            unused_curl,
            zero_current,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        E_after, B_after, J_after, rho_after, phi_after, external_after, pml_state = fields
        vx, vy, vz = particles[0].get_velocity()

        self.assertIsNone(pml_state)
        self.assertGreater(float(vx[0]), 0.0)
        self.assertTrue(jnp.allclose(vy, 0.0))
        self.assertTrue(jnp.allclose(vz, 0.0))
        self.assertTrue(jnp.allclose(E_after[0], 0.0))
        self.assertTrue(jnp.allclose(B_after[0], 0.0))
        self.assertTrue(jnp.allclose(external_after[0][0], external_fields[0][0]))

    def test_absorbing_particle_mask_survives_jitted_electrodynamic_step(self):
        world = {
            "Nx": 3,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0 / 3.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        particles = [
            particle_species(
                name="absorbing",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=0.0,
                weight=1.0,
                v1=jnp.array([0.2]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([0.49]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                x_bc="absorbing",
                dt=world["dt"],
            )
        ]
        fields = (E, B, J, rho, phi, external_fields, None)

        particles, _ = time_loop_electrodynamic(
            particles,
            fields,
            world,
            constants,
            unused_curl,
            zero_current,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        x, _, _ = particles[0].get_forward_position()

        self.assertEqual(x.shape[0], 1)
        self.assertTrue(jnp.array_equal(particles[0].get_active_mask(), jnp.array([False])))


if __name__ == "__main__":
    unittest.main()
