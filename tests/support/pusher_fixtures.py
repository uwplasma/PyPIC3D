import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from tests.kernel_fixtures import build_tiled_particles, particle_parameters_from_tile_values, particle_species
from tests.kernel_fixtures import kernel_parameters_from_values, tile_vector_field
from PyPIC3D.particles.particle_tile_communication import update_tiled_particle_positions
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.particle_push import particle_push, seed_leapfrog_velocity
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


class TiledParticlePusherFixtures:
    def _build_parameter_values(self, Nx=8, Ny=6, Nz=4, shape_factor=1):
        parameter_set = {
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
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        return parameter_set

    def _with_tiled_grids(self, parameter_set, tile_shape, g=1):
        parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
        parameter_set["guard_cells"] = int(g)
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
        parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
        parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        return parameter_set

    def _copy_parameters_for_tile_shape(self, parameter_set, tile_shape, g):
        tiled_parameters = dict(parameter_set)
        tiled_parameters["grids"] = dict(parameter_set["grids"])
        return self._with_tiled_grids(tiled_parameters, tile_shape, g=g)

    def _simulation_parameters_for_tile_shape(self, tile_shape):
        return {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

    def _push_tiled_species(self, species, parameter_set, tile_shape, E, B, dynamic_values, relativistic=True, particle_pusher="boris"):
        particle_static, particle_dynamic = particle_parameters_from_tile_values(
            parameter_set,
            self._simulation_parameters_for_tile_shape(tile_shape),
            dynamic_values=dynamic_values,
        )
        tiled_particles, species_config = build_tiled_particles(
            [species],
            particle_static,
            particle_dynamic,
        )
        g = int(parameter_set["guard_cells"])
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(
            relativistic=bool(relativistic),
            particle_pusher=particle_pusher,
        )
        return particle_push(
            tiled_particles,
            species_config,
            tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=g),
            tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=g),
            static_parameters,
            dynamic_parameters,
        )

    def _deterministic_vector_field(self, parameter_set, scale):
        Nx, Ny, Nz = parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"]
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

    def _species(self, parameter_set, active_mask=None, update_x=True, update_y=True, update_z=True):
        if active_mask is None:
            active_mask = jnp.array([True, True, True, True])

        return particle_species(
            name="test particles",
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            active_mask=active_mask,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
        )

    def _flatten_active_by_position(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]


class SeedLeapfrogVelocityFixtures:
    """
    ``seed_leapfrog_velocity`` puts the configured ``u(0)`` onto the half step
    the leapfrog actually starts from.  Without it the initial state carries an
    O(dt) error and the whole simulation drops to first order.
    """

    DT = 0.02

    def _one_particle(self, u):
        return TiledParticles(
            x=jnp.zeros((1, 1, 1, 1, 1, 3)),
            u=jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
            active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
        )

    def _species(self, charge=1.0, mass=2.0, update_x=(True, True, True)):
        return SpeciesConfig(
            charge=jnp.asarray([charge]),
            mass=jnp.asarray([mass]),
            weight=jnp.asarray([1.0]),
            update_x=jnp.asarray([list(update_x)]),
        )

    def _uniform(self, static_parameters, dynamic_parameters, values):
        empty = empty_tiled_vector(static_parameters, dynamic_parameters)
        return tuple(empty[i].at[...].set(values[i]) for i in range(3))

    def _flat_case(self, solver="electrodynamic_yee"):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8, Ny=8, Nz=8,
            x_wind=8.0, y_wind=8.0, z_wind=8.0,
            dt=self.DT, tile_shape=(8, 8, 8), solver=solver,
        )
        E = self._uniform(static_parameters, dynamic_parameters, (0.30, 0.0, 0.0))
        B = self._uniform(static_parameters, dynamic_parameters, (0.0, 0.0, 0.20))
        return static_parameters, dynamic_parameters, E, B

    def _gr_case(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8, Ny=8, Nz=8,
            x_wind=8.0, y_wind=8.0, z_wind=8.0,
            dt=self.DT, tile_shape=(8, 8, 8),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = self._uniform(static_parameters, dynamic_parameters, (0.30, 0.0, 0.0))
        B = self._uniform(static_parameters, dynamic_parameters, (0.0, 0.0, 0.20))
        return static_parameters, dynamic_parameters, D, B, metric
