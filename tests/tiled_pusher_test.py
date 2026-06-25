import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.pusher.tiled_pusher import tiled_particle_push
from PyPIC3D.solvers.yee_tiled import tile_vector_field
from PyPIC3D.utils import build_yee_grid


jax.config.update("jax_enable_x64", True)


class TestTiledParticlePusher(unittest.TestCase):
    def _build_world(self, Nx=8, Ny=6, Nz=4, shape_factor=1):
        world = {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "dx": 4.0 / Nx,
            "dy": 3.0 / Ny,
            "dz": 2.0 / Nz,
            "dt": 0.05,
            "x_wind": 4.0,
            "y_wind": 3.0,
            "z_wind": 2.0,
            "shape_factor": shape_factor,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _deterministic_vector_field(self, world, scale):
        Nx, Ny, Nz = world["Nx"], world["Ny"], world["Nz"]
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx, dtype=float),
            jnp.arange(Ny, dtype=float),
            jnp.arange(Nz, dtype=float),
            indexing="ij",
        )

        shape = (Nx + 2, Ny + 2, Nz + 2)
        Fx = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.2 + 0.03 * ii - 0.02 * jj + 0.04 * kk))
        Fy = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (-0.1 + 0.05 * ii + 0.01 * jj - 0.03 * kk))
        Fz = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.3 - 0.04 * ii + 0.02 * jj + 0.01 * kk))
        return Fx, Fy, Fz

    def _species(self, world, active_mask=None, update_vx=True, update_vy=True, update_vz=True):
        if active_mask is None:
            active_mask = jnp.array([True, True, True, True])

        return particle_species(
            name="test particles",
            N_particles=4,
            charge=-1.0,
            mass=2.0,
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
            active_mask=active_mask,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
        )

    def _flatten_active_by_position(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def test_tiled_particle_push_matches_flat_boris(self):
        world = self._build_world()
        constants = {"C": 10.0}
        tile_shape = (2, 3, 2)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)

        species = self._species(world)
        reference = self._species(world)
        reference = particle_push(
            reference,
            E,
            B,
            world["grids"]["center"],
            world["grids"]["vertex"],
            world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles([species], world, simulation_parameters)
        pushed = tiled_particle_push(
            tiled_particles,
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            world,
            constants,
            tile_shape,
            relativistic=False,
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        flat_u = jnp.stack(reference.get_velocity(), axis=1)
        flat_x = jnp.stack(reference.get_forward_position(), axis=1)
        order = jnp.lexsort((flat_x[:, 2], flat_x[:, 1], flat_x[:, 0]))

        self.assertTrue(jnp.allclose(tiled_u, flat_u[order], rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_particle_push_respects_active_and_update_flags(self):
        world = self._build_world()
        constants = {"C": 10.0}
        tile_shape = (2, 3, 2)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        E = self._deterministic_vector_field(world, scale=1.0)
        B = self._deterministic_vector_field(world, scale=0.2)
        species = self._species(
            world,
            active_mask=jnp.array([True, False, True, True]),
            update_vx=False,
            update_vy=True,
            update_vz=False,
        )
        tiled_particles = to_tiled_particles([species], world, simulation_parameters)

        pushed = tiled_particle_push(
            tiled_particles,
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            world,
            constants,
            tile_shape,
            relativistic=False,
        )

        self.assertTrue(jnp.allclose(pushed.u[..., 0], tiled_particles.u[..., 0]))
        self.assertTrue(jnp.allclose(pushed.u[..., 2], tiled_particles.u[..., 2]))
        self.assertTrue(jnp.allclose(pushed.u[..., 1][~tiled_particles.active], tiled_particles.u[..., 1][~tiled_particles.active]))

    def test_tiled_particle_push_matches_relativistic_flat_boris_on_reduced_axes(self):
        world = self._build_world(Nx=8, Ny=1, Nz=1)
        constants = {"C": 10.0}
        tile_shape = (2, 1, 1)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        E = self._deterministic_vector_field(world, scale=0.1)
        B = self._deterministic_vector_field(world, scale=0.02)
        def make_species():
            return particle_species(
                name="one dimensional",
                N_particles=3,
                charge=1.0,
                mass=1.0,
                weight=1.0,
                T=1.0,
                x1=jnp.array([-1.25, 0.15, 1.25]),
                x2=jnp.zeros(3),
                x3=jnp.zeros(3),
                v1=jnp.array([0.02, -0.01, 0.03]),
                v2=jnp.array([0.0, 0.01, -0.02]),
                v3=jnp.array([-0.005, 0.025, 0.01]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            )

        species = make_species()
        reference = make_species()
        reference = particle_push(
            reference,
            E,
            B,
            world["grids"]["center"],
            world["grids"]["vertex"],
            world,
            constants,
            relativistic=True,
            particle_pusher="boris",
        )

        tiled_particles = to_tiled_particles([species], world, simulation_parameters)
        pushed = tiled_particle_push(
            tiled_particles,
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            world,
            constants,
            tile_shape,
            relativistic=True,
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        flat_u = jnp.stack(reference.get_velocity(), axis=1)
        flat_x = jnp.stack(reference.get_forward_position(), axis=1)
        order = jnp.lexsort((flat_x[:, 2], flat_x[:, 1], flat_x[:, 0]))

        self.assertTrue(jnp.allclose(tiled_u, flat_u[order], rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
