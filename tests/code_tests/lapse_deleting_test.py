"""Native-lapse particle deletion and its public configuration."""

import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import toml

from PyPIC3D.boundary_conditions.lapse_deleting import lapse_deleting
from PyPIC3D.initialization import default_parameters, initialize_simulation
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.filtering import particle_metric
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.utilities.parameters import build_static_parameters, static_parameters_for_output
from PyPIC3D.utilities.toml_helpers import update_parameters_from_toml
from tests.kernel_fixtures import kernel_parameters

jax.config.update("jax_enable_x64", True)


def manufactured_lapse():
    static, dynamic = kernel_parameters(
        Nx=8, Ny=8, Nz=8, x_wind=4., y_wind=4., z_wind=4.,
        x_min=0., y_min=0., z_min=0., solver="static_metric",
        particle_pusher="hybrid_boris_geodesic", particle_batch_size=4,
    )
    static = static._replace(lapse_deleting=True, lapse_threshold=.125)
    metric = initialize_flat_cartesian_metric(static, dynamic)
    x, y, z = (axis[0, 0, 0] for axis in dynamic.grids.tiled_center_grid)
    alpha = .125 + .0625 * (x[:, None, None] + 2 * y[None, :, None] - 3 * z[None, None, :])
    metric = metric._replace(center=metric.center._replace(lapse=alpha[None, None, None]))
    points = jnp.array([[.75, .625, .75], [1.25, .875, 1.], [1.5, .875, 1.],
                        [jnp.nan, jnp.nan, jnp.nan]])
    x = jnp.broadcast_to(points, (1, 1, 1, 2, 4, 3))
    u = jnp.broadcast_to(jnp.array([.2, -.1, .3]), x.shape)
    active = jnp.broadcast_to(jnp.array([True, True, True, False]), x.shape[:-1])
    return static, dynamic, metric, TiledParticles(x, u, active)


class LapseDeletingTests(unittest.TestCase):
    def test_three_dimensional_cut_preserves_equal_lapse_and_particle_arrays(self):
        static, dynamic, metric, particles = manufactured_lapse()
        # The affine lapse is reproduced exactly: these points have alpha
        # 7/64, 8/64 and 9/64, independently of the interpolation implementation.
        result = jax.jit(lambda p: lapse_deleting(p, metric, static, dynamic))(particles)
        expected = jnp.broadcast_to(jnp.array([False, True, True, False]), particles.active.shape)
        self.assertTrue(bool(jnp.array_equal(result.active, expected)))
        self.assertTrue(bool(jnp.array_equal(result.x, particles.x, equal_nan=True)))
        self.assertTrue(bool(jnp.array_equal(result.u, particles.u)))
        self.assertEqual(result.x.shape, particles.x.shape)
        self.assertEqual(result.active.shape, particles.active.shape)

    def test_configurable_threshold_and_repeated_calls_cannot_reactivate_slots(self):
        static, dynamic, metric, particles = manufactured_lapse()
        strict = static._replace(lapse_threshold=.13)
        result = lapse_deleting(particles, metric, strict, dynamic)
        expected = jnp.broadcast_to(jnp.array([False, False, True, False]), particles.active.shape)
        self.assertTrue(bool(jnp.array_equal(result.active, expected)))
        result = lapse_deleting(result, metric, static._replace(lapse_threshold=.01), dynamic)
        self.assertTrue(bool(jnp.array_equal(result.active, expected)))

    def test_disabled_and_empty_storage_do_not_need_a_metric(self):
        static, dynamic, _, particles = manufactured_lapse()
        result = lapse_deleting(particles, None, static._replace(lapse_deleting=False), dynamic)
        for actual, expected in zip(result, particles):
            self.assertTrue(bool(jnp.array_equal(actual, expected, equal_nan=True)))
        empty_x = jnp.zeros((1, 1, 1, 0, 0, 3))
        empty = TiledParticles(empty_x, empty_x, jnp.zeros(empty_x.shape[:-1], bool))
        result = lapse_deleting(empty, None, static, dynamic)
        self.assertEqual(result.x.shape, empty_x.shape)
        self.assertEqual(result.active.size, 0)

    def test_all_inactive_slots_remain_inactive_with_invalid_stored_positions(self):
        static, dynamic, metric, particles = manufactured_lapse()
        particles = particles._replace(x=jnp.full_like(particles.x, jnp.nan),
                                       active=jnp.zeros_like(particles.active))
        result = lapse_deleting(particles, metric, static, dynamic)
        self.assertFalse(bool(jnp.any(result.active)))
        self.assertTrue(bool(jnp.array_equal(result.u, particles.u)))

    def test_native_lapse_controls_deletion_when_coupling_filter_smooths_it(self):
        static, dynamic, metric, _ = manufactured_lapse()
        static = static._replace(current_filter="bilinear", lapse_threshold=.1)
        g = static.guard_cells
        index = (0, 0, 0, g + 2, g + 2, g + 2)
        alpha = jnp.ones_like(metric.center.lapse).at[index].set(.05)
        metric = metric._replace(center=metric.center._replace(lapse=alpha))
        position = jnp.stack(tuple(axis[0, 0, 0, g + 2] for axis in dynamic.grids.tiled_center_grid))
        x = position.reshape(1, 1, 1, 1, 1, 3)
        particles = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], bool))
        filtered = particle_metric(metric, static, dynamic)
        self.assertGreater(float(filtered.lapse[index]), static.lapse_threshold)
        self.assertFalse(bool(jnp.any(lapse_deleting(particles, metric, static, dynamic).active)))

    def test_boundary_uses_only_lapse_and_does_not_delete_nan_samples(self):
        static, dynamic, metric, particles = manufactured_lapse()
        center = metric.center._replace(gamma=jnp.full_like(metric.center.gamma, jnp.nan),
                                         gamma_inv=jnp.full_like(metric.center.gamma_inv, jnp.nan))
        metric = metric._replace(center=center)
        result = lapse_deleting(particles, metric, static, dynamic)
        self.assertEqual(int(jnp.sum(result.active)), 4)
        metric = metric._replace(center=center._replace(lapse=jnp.full_like(center.lapse, jnp.nan)))
        result = lapse_deleting(particles, metric, static, dynamic)
        self.assertTrue(bool(jnp.array_equal(result.active, particles.active)))


class LapseDeletingConfigurationTests(unittest.TestCase):
    def test_startup_outputs_and_leapfrog_seed_receive_only_surviving_particles(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "simulation_parameters": {
                    "solver": "static_metric", "metric": "flat_cartesian",
                    "particle_pusher": "hybrid_boris_geodesic", "filter_j": "none",
                    "current_calculation": "esirkepov",
                    "output_dir": output_dir, "Nx": 8, "Ny": 1, "Nz": 1,
                    "x_wind": 4., "y_wind": 1., "z_wind": 1., "dt": .001, "Nt": 1,
                    "particle_tile_nx": 8, "particle_tile_ny": 1, "particle_tile_nz": 1,
                    "C": 1., "eps": 1., "mu": 1., "kb": 1.,
                    "lapse_deleting": True, "lapse_threshold": 2.,
                },
                "particle1": {"name": "charges", "N_particles": 4, "charge": 1.,
                              "mass": 1., "temperature": 0., "weight": 1.},
                "plotting": {"dump_particles": True},
            }
            # Flat alpha=1 lies below this deliberately high test threshold.
            # Observe production startup at its output and seed boundaries.
            with patch("PyPIC3D.initialization.plot_initial_tiled_histograms") as histograms, \
                    patch("PyPIC3D.initialization.write_openpmd_initial_particles") as output, \
                    patch("PyPIC3D.initialization.seed_leapfrog_velocity", wraps=seed_leapfrog_velocity) as seed, \
                    contextlib.redirect_stdout(io.StringIO()):
                _, particles, *_ = initialize_simulation(config)
            for observed in (histograms.call_args.args[0], output.call_args.args[0],
                             seed.call_args.args[0], particles):
                self.assertFalse(bool(jnp.any(observed.active)))
                self.assertTrue(bool(jnp.all(observed.u == 0.)))

    def test_defaults_leave_every_solver_unchanged(self):
        _, defaults, _ = default_parameters()
        self.assertFalse(defaults["lapse_deleting"])
        self.assertEqual(defaults["lapse_threshold"], .1)
        template, _ = kernel_parameters(guard_cells=3)
        for solver in ("static_metric", "dynamic_metric", "electrodynamic_yee", "electrostatic"):
            with self.subTest(solver=solver):
                values = dict(template._asdict(), solver=solver)
                values.pop("lapse_deleting")
                values.pop("lapse_threshold")
                parameters = build_static_parameters(values)
                self.assertFalse(parameters.lapse_deleting)
                self.assertEqual(parameters.lapse_threshold, .1)

    def test_toml_configuration_and_saved_parameters_retain_the_boundary(self):
        for solver in ("static_metric", "dynamic_metric"):
            with self.subTest(solver=solver):
                template, _ = kernel_parameters(solver=solver, guard_cells=3)
                plotting, static, dynamic = default_parameters()
                static.update(template._asdict())
                config = toml.loads('[simulation_parameters]\nlapse_deleting = true\nlapse_threshold = 0.05\n')
                static, dynamic, plotting = update_parameters_from_toml(config, static, dynamic, plotting)
                parameters = build_static_parameters(static)
                self.assertTrue(parameters.lapse_deleting)
                self.assertEqual(parameters.lapse_threshold, .05)
                saved = toml.loads(toml.dumps({"static_parameters": static_parameters_for_output(parameters)}))
                self.assertTrue(saved["static_parameters"]["lapse_deleting"])
                self.assertEqual(saved["static_parameters"]["lapse_threshold"], .05)
                self.assertEqual(parameters.particle_boundary_conditions, template.particle_boundary_conditions)

    def test_enabled_boundary_rejects_non_gr_solvers(self):
        template, _ = kernel_parameters()
        for solver in ("electrodynamic_yee", "electrostatic", "dark_matter_yee"):
            with self.subTest(solver=solver), self.assertRaisesRegex(ValueError, "lapse_deleting requires"):
                build_static_parameters(dict(template._asdict(), solver=solver, lapse_deleting=True))

    def test_threshold_must_be_finite_and_positive(self):
        template, _ = kernel_parameters(solver="static_metric", guard_cells=3)
        for threshold in (0., -.1, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "lapse_threshold"):
                build_static_parameters(dict(template._asdict(), lapse_threshold=threshold))


if __name__ == "__main__":
    unittest.main()
