import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.rho_tiled import compute_rho_from_tiled_particles, compute_tiled_rho_from_tiled_particles
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.solvers.yee_tiled import assemble_tiled_scalar_field, tile_scalar_field
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


class TestTiledRho(unittest.TestCase):
    def _build_world(self, shape_factor, dt=0.08):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        world = {
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
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _empty_scalar(self, world):
        return jnp.zeros((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }

    def _particles(self, world):
        electrons = particle_species(
            name="electrons",
            N_particles=5,
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75, world["x_wind"] / 2 - 0.03]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05, 0.0]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80, 0.0]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3, 1.4]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1, 0.0]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15, 0.0]),
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
            N_particles=4,
            charge=2.0,
            mass=4.0,
            weight=0.25,
            T=1.0,
            x1=jnp.array([-1.25, -0.20, 0.75, 1.35]),
            x2=jnp.array([1.15, -0.75, 0.45, 0.05]),
            x3=jnp.array([0.35, -0.45, 0.85, -0.15]),
            v1=jnp.array([-0.1, 0.2, -0.25, 0.05]),
            v2=jnp.array([0.3, -0.05, 0.15, -0.2]),
            v3=jnp.array([0.1, 0.05, -0.2, 0.3]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
            active_mask=jnp.array([True, False, True, True]),
        )
        return [electrons, ions]

    def _tiled_with_noisy_inactive_slots(self, particles, world):
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())

        inactive = ~tiled_particles.active
        x = tiled_particles.x.at[inactive, 0].set(0.33)
        x = x.at[inactive, 1].set(-0.27)
        x = x.at[inactive, 2].set(0.18)
        u = tiled_particles.u.at[inactive, 0].set(4.0)

        return tiled_particles._replace(x=x, u=u), species_config

    def _zero_species_velocities(self, particles):
        for species in particles:
            species.v1 = jnp.zeros_like(species.v1)
            species.v2 = jnp.zeros_like(species.v2)
            species.v3 = jnp.zeros_like(species.v3)
        return particles

    def _zero_tiled_velocities(self, tiled_particles):
        return tiled_particles._replace(u=jnp.zeros_like(tiled_particles.u))

    def _compare_tiled_to_standard(self, shape_factor, alpha):
        world = self._build_world(shape_factor)
        constants = {"alpha": alpha}
        particles = self._particles(world)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, world)

        rho_reference = compute_rho(particles, self._empty_scalar(world), world, constants)
        rho_tiled = compute_rho_from_tiled_particles(
            tiled_particles,
            species_config,
            self._empty_scalar(world),
            world,
            constants,
        )

        self.assertTrue(
            jnp.allclose(
                rho_tiled[1:-1, 1:-1, 1:-1],
                rho_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tiled_rho_matches_compute_rho_for_shape_factor_1(self):
        self._compare_tiled_to_standard(shape_factor=1, alpha=1.0)

    def test_tiled_rho_matches_compute_rho_for_shape_factor_2(self):
        self._compare_tiled_to_standard(shape_factor=2, alpha=1.0)

    def test_tiled_rho_matches_compute_rho_after_digital_filter(self):
        self._compare_tiled_to_standard(shape_factor=2, alpha=0.55)

    def test_compute_rho_uses_current_positions_not_half_step_back_positions(self):
        world = self._build_world(shape_factor=2)
        constants = {"alpha": 1.0}
        particles = self._particles(world)
        zero_velocity_particles = self._zero_species_velocities(self._particles(world))

        rho_with_velocity = compute_rho(particles, self._empty_scalar(world), world, constants)
        rho_with_zero_velocity = compute_rho(zero_velocity_particles, self._empty_scalar(world), world, constants)

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tiled_global_rho_uses_current_positions_not_half_step_back_positions(self):
        world = self._build_world(shape_factor=2)
        constants = {"alpha": 1.0}
        particles = self._particles(world)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, world)
        zero_velocity_tiled_particles = self._zero_tiled_velocities(tiled_particles)

        rho_with_velocity = compute_rho_from_tiled_particles(
            tiled_particles,
            species_config,
            self._empty_scalar(world),
            world,
            constants,
        )
        rho_with_zero_velocity = compute_rho_from_tiled_particles(
            zero_velocity_tiled_particles,
            species_config,
            self._empty_scalar(world),
            world,
            constants,
        )

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tile_major_rho_uses_current_positions_not_half_step_back_positions(self):
        world = self._build_world(shape_factor=2)
        world["tile_shape"] = (
            self._simulation_parameters()["particle_tile_nx"],
            self._simulation_parameters()["particle_tile_ny"],
            self._simulation_parameters()["particle_tile_nz"],
        )
        constants = {"alpha": 1.0}
        particles = self._particles(world)
        tiled_particles, species_config = self._tiled_with_noisy_inactive_slots(particles, world)
        zero_velocity_tiled_particles = self._zero_tiled_velocities(tiled_particles)
        rho_tiles = tile_scalar_field(self._empty_scalar(world), world, world["tile_shape"])

        rho_tiles_with_velocity = compute_tiled_rho_from_tiled_particles(
            tiled_particles,
            species_config,
            rho_tiles,
            world,
            constants,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        rho_tiles_with_zero_velocity = compute_tiled_rho_from_tiled_particles(
            zero_velocity_tiled_particles,
            species_config,
            rho_tiles,
            world,
            constants,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        rho_with_velocity = assemble_tiled_scalar_field(rho_tiles_with_velocity, world, world["tile_shape"], num_guard_cells=int(world["guard_cells"]))
        rho_with_zero_velocity = assemble_tiled_scalar_field(rho_tiles_with_zero_velocity, world, world["tile_shape"], num_guard_cells=int(world["guard_cells"]))

        self.assertTrue(
            jnp.allclose(
                rho_with_velocity[1:-1, 1:-1, 1:-1],
                rho_with_zero_velocity[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
