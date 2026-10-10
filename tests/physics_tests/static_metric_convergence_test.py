import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    geodesic_acceleration,
    hybrid_boris_geodesic_push,
)
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.metrics.kerr_schild import initialize_kerr_schild_spherical_metric
from PyPIC3D.solvers.GR_yee.static_metric import update_B
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


def _single_particle_species(charge):
    return SpeciesConfig(
        charge=jnp.asarray([charge]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )


def _single_particle(x, u):
    return TiledParticles(
        x=x.reshape((1, 1, 1, 1, 1, 3)),
        u=u.reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
    )


class TestStaticMetricConvergence(unittest.TestCase):
    def test_spherical_kerr_metric_preserves_off_grid_azimuthal_symmetry(self):
        mass = 1.0
        spin = 0.995
        nr = 64
        ntheta = 64
        r_min = 0.99 * (mass + math.sqrt(mass**2 - spin**2))
        theta_min = 0.25 * 2.0 * math.pi / ntheta

        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=nr,
            Ny=ntheta,
            Nz=1,
            x_wind=30.0 - r_min,
            y_wind=2.0 * math.pi,
            z_wind=2.0 * math.pi,
            x_min=r_min,
            y_min=theta_min,
            z_min=0.0,
            dt=0.01,
            tile_shape=(nr, ntheta, 1),
            solver="static_metric",
            current_deposition="GR_direct",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_kerr_schild_spherical_metric(
            static_parameters,
            dynamic_parameters,
            mass=mass,
            spin=spin,
        )

        position = jnp.asarray((10.6498, 0.5 * math.pi, 0.0))
        u_cov = jnp.asarray((0.1891441241525076, 0.0, 2.0))
        active_axes = (True, True, False)
        inactive_axis_indices = (static_parameters.guard_cells,) * 3
        center_grid = tuple(
            axis[0, 0, 0]
            for axis in dynamic_parameters.grids.tiled_center_grid
        )
        sampled_metric = interpolate_metric(
            jax.tree.map(lambda array: array[0, 0, 0], metric.center),
            position,
            center_grid,
            static_parameters.metric,
            active_axes,
            inactive_axis_indices,
        )
        # The particle inverse is the exact inverse of the interpolated metric.
        self.assertTrue(
            bool(
                jnp.allclose(
                    sampled_metric.gamma @ sampled_metric.gamma_inv,
                    jnp.eye(3),
                    rtol=1.0e-13,
                    atol=1.0e-13,
                )
            )
        )
        # Axisymmetry: no phi derivative, so no azimuthal geodesic force.
        self.assertTrue(bool(jnp.all(sampled_metric.grad_gamma_inv[2] == 0.0)))
        du_dt = geodesic_acceleration(u_cov, sampled_metric)

        self.assertEqual(float(du_dt[2]), 0.0)

        particles = _single_particle(position, u_cov)
        species = _single_particle_species(charge=0.0)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)

        for _ in range(128):
            particles, _ = hybrid_boris_geodesic_push(
                particles,
                species,
                D,
                B,
                metric,
                static_parameters,
                dynamic_parameters,
            )

        self.assertEqual(
            float(particles.u[0, 0, 0, 0, 0, 2]),
            2.0,
        )

    def test_flat_vacuum_update_B_converges_for_smooth_mode(self):
        errors = []
        for Nx in (32, 64, 128):
            static_parameters, dynamic_parameters = kernel_parameters(
                Nx=Nx,
                Ny=1,
                Nz=1,
                x_wind=2.0 * math.pi,
                y_wind=1.0,
                z_wind=1.0,
                dx=2.0 * math.pi / Nx,
                dy=1.0,
                dz=1.0,
                dt=1.0e-3,
                tile_shape=(Nx, 1, 1),
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            D = empty_tiled_vector(static_parameters, dynamic_parameters)
            B = empty_tiled_vector(static_parameters, dynamic_parameters)
            Dx, Dy, Dz = D

            g = int(static_parameters.guard_cells)
            active = slice(g, -g)
            x_center = dynamic_parameters.grids.tiled_center_grid[0][:, :, :, active]
            Dz_values = jnp.sin(x_center)[:, :, :, :, jnp.newaxis, jnp.newaxis]
            Dz = Dz.at[:, :, :, active, active, active].set(Dz_values)

            # flat Cartesian: E_i = D^i and the densities equal the physical fields
            _Bx, By, _Bz = update_B((Dx, Dy, Dz), B, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt)

            x_vertex = dynamic_parameters.grids.tiled_vertex_grid[0][:, :, :, active]
            exact_By = dynamic_parameters.dt * jnp.cos(x_vertex)
            diff = By[:, :, :, active, active, active] - exact_By[:, :, :, :, jnp.newaxis, jnp.newaxis]
            errors.append(float(jnp.sqrt(jnp.mean(diff**2))))

        first_order = math.log(errors[0] / errors[1], 2.0)
        second_order = math.log(errors[1] / errors[2], 2.0)

        self.assertGreater(first_order, 1.8)
        self.assertGreater(second_order, 1.8)



if __name__ == "__main__":
    unittest.main()
