"""Periodic GR particle--grid filtering and its discrete conservation contract."""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence, electromagnetic_constraints
from PyPIC3D.initialization import _validate_current_filter_contract, initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.core import build_yee_metric
from PyPIC3D.relativity.filtering import filter_gr_grid, particle_metric, validate_gr_filter
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import empty_tiled_scalar, empty_tiled_vector, kernel_parameters


def parameters(choice="bilinear", **kwargs):
    settings = dict(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., dt=.05, guard_cells=3,
        solver="static_metric", metric="numerical", current_deposition="esirkepov",
        particle_pusher="hybrid_boris_geodesic", current_filter=choice,
        alpha=.6, particle_batch_size=4,
    )
    settings.update(kwargs)
    return kernel_parameters(**settings)


def varying_metric(position):
    phase = 2 * jnp.pi * position[0] / 8.
    lapse = 1. + .08 * jnp.cos(phase)
    shift = jnp.array([.03 * jnp.sin(phase), .02 * jnp.cos(phase), 0.])
    gamma = jnp.array([
        [1.4 + .2 * jnp.cos(phase), .04 * jnp.sin(phase), 0.],
        [.04 * jnp.sin(phase), 1.1 + .1 * jnp.sin(phase), 0.],
        [0., 0., 1.2 + .15 * jnp.cos(phase)],
    ])
    return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))


def particles_at(position, tile_counts=(1, 1, 1)):
    x = jnp.broadcast_to(jnp.asarray(position, dtype=jnp.float64), tile_counts + (1, 1, 3))
    active = jnp.zeros(tile_counts + (1, 1), bool).at[0, 0, 0].set(True)
    return TiledParticles(x, jnp.zeros_like(x), active)


def species():
    return SpeciesConfig(jnp.array([1.]), jnp.array([1.]), jnp.array([.7]),
                         jnp.ones((1, 3), bool))


class TestGRMatchedFilter(unittest.TestCase):
    def assertClose(self, actual, expected, atol=2.e-12, rtol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"max difference = {float(jnp.max(jnp.abs(actual - expected)))}")

    def test_fourier_response_and_constant_mode(self):
        for shape in ((16, 1, 1), (16, 8, 1), (16, 8, 4)):
            indices = jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij")
            wave = tuple(2 * jnp.pi / n if n > 1 else 0. for n in shape)
            mode = jnp.cos(sum(k * x for k, x in zip(wave, indices)))
            for choice in ("bilinear", "digital"):
                with self.subTest(shape=shape, filter=choice):
                    static, dynamic = parameters(choice, Nx=shape[0], Ny=shape[1], Nz=shape[2])
                    response = (jnp.prod(jnp.cos(jnp.array(wave) / 2) ** 2)
                                if choice == "bilinear" else
                                dynamic.alpha + (1 - dynamic.alpha) * sum(jnp.cos(k) for k in wave) / 3)
                    self.assertClose(filter_gr_grid(mode, static, dynamic), response * mode)
                    self.assertClose(filter_gr_grid(jnp.ones(shape), static, dynamic), jnp.ones(shape))
        static, dynamic = parameters()
        checkerboard = (-1.) ** jnp.arange(8)[:, None, None]
        self.assertClose(filter_gr_grid(checkerboard, static, dynamic), jnp.zeros_like(checkerboard))

    def test_scalar_vector_tensor_sum_and_adjoint(self):
        keys = jax.random.split(jax.random.PRNGKey(73), 6)
        for leading, key_a, key_b in zip(((), (3,), (3, 3)), keys[:3], keys[3:]):
            a = jax.random.normal(key_a, leading + (8, 4, 1), dtype=jnp.float64)
            b = jax.random.normal(key_b, a.shape, dtype=jnp.float64)
            for choice in ("bilinear", "digital"):
                static, dynamic = parameters(choice, Ny=4)
                fa = filter_gr_grid(a, static, dynamic)
                fb = filter_gr_grid(b, static, dynamic)
                self.assertClose(jnp.sum(fa, axis=(-3, -2, -1)), jnp.sum(a, axis=(-3, -2, -1)))
                self.assertClose(jnp.sum(a * fb), jnp.sum(fa * b))

    def test_none_preserves_arrays_and_metric_exactly(self):
        static, dynamic = parameters("none")
        metric = build_yee_metric(dynamic, varying_metric)
        raw = jax.random.normal(jax.random.PRNGKey(4), (3, 3, 8, 1, 1), dtype=jnp.float64)
        self.assertTrue(bool(jnp.array_equal(filter_gr_grid(raw, static, dynamic), raw)))
        for actual, expected in zip(particle_metric(metric, static, dynamic), metric.center):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_filtered_metric_and_hermite_derivatives_are_consistent(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice)
            raw = build_yee_metric(dynamic, varying_metric)
            metric = particle_metric(raw, static, dynamic)
            self.assertClose(jnp.einsum("...ij,...jk->...ik", metric.gamma, metric.gamma_inv), jnp.eye(3))
            self.assertClose(metric.sqrt_gamma ** 2, jnp.linalg.det(metric.gamma))
            self.assertGreater(float(jnp.min(jnp.linalg.eigvalsh(metric.gamma))), 0.)
            self.assertGreater(float(jnp.max(jnp.abs(metric.lapse - raw.center.lapse))), 1.e-4)

            local = jax.tree.map(lambda a: a[0, 0, 0], metric)
            grid = tuple(a[0, 0, 0] for a in dynamic.grids.tiled_center_grid)
            position = jnp.array([2.37, 0., 0.])
            def sample(x, derivatives=False):
                return interpolate_metric(local, x, grid, "numerical", (True, False, False),
                                          (3, 3, 3), derivatives=derivatives)
            value = sample(position, True)
            h = 1.e-5
            plus, minus = sample(position + jnp.array([h, 0., 0.])), sample(position - jnp.array([h, 0., 0.]))
            self.assertClose(value.grad_lapse[0], (plus.lapse - minus.lapse) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_shift[:, 0], (plus.shift - minus.shift) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_gamma_inv[0], (plus.gamma_inv - minus.gamma_inv) / (2 * h), atol=2.e-10)
            self.assertClose(value.grad_lapse[1:], jnp.zeros(2))
            self.assertClose(value.grad_gamma_inv[1:], jnp.zeros((2, 3, 3)))

    def test_push_filters_physical_fields_after_volume_conversion(self):
        static, dynamic = parameters()
        metric = build_yee_metric(dynamic, varying_metric)
        physical = jnp.array([.3, -.2, .1])
        density = tuple(value * sample.sqrt_gamma for value, sample in zip(physical, metric.D))
        zero = empty_tiled_vector(static, dynamic)
        old = particles_at([2.37, 0., 0.])
        old = old._replace(u=jnp.broadcast_to(jnp.array([.1, -.07, .02]), old.u.shape))
        actual, _ = hybrid_boris_geodesic_push(old, species(), density, zero, metric,
                                               static, dynamic, advance_positions=False)
        # A constant physical field remains constant. Only the particle metric
        # changes; Maxwell's node volumes still convert the supplied density.
        reference_metric = metric._replace(center=particle_metric(metric, static, dynamic))
        expected, _ = hybrid_boris_geodesic_push(old, species(), density, zero, reference_metric,
                                                static._replace(current_filter="none"), dynamic,
                                                advance_positions=False)
        self.assertClose(actual.u, expected.u)
        wrong_order = filter_gr_grid(density[0], static, dynamic, tiled=True) / metric.D[0].sqrt_gamma
        self.assertGreater(float(jnp.max(jnp.abs(wrong_order - physical[0]))), 1.e-4)

    def continuity_case(self, shape_factor, choice, *, tile_shape=None, crossing=7.95):
        static, dynamic = parameters(choice, shape_factor=shape_factor, tile_shape=tile_shape)
        counts = (8 // tile_shape[0], 1, 1) if tile_shape else (1, 1, 1)
        old = particles_at([crossing, 0., 0.], counts)
        new = old._replace(x=old.x + jnp.array([.12, .03, -.02]))
        template = empty_tiled_vector(static, dynamic)
        raw = Esirkepov_current(old, new, species(), template, static, dynamic)
        current = tuple(filter_gr_grid(c, static, dynamic, tiled=True) for c in raw)
        rho0 = compute_rho(old, species(), template[0], static, dynamic)
        rho1 = compute_rho(new, species(), template[0], static, dynamic)
        active = (Ellipsis,) + (slice(3, -3),) * 3
        residual = ((rho1 - rho0) / dynamic.dt + divergence(current, dynamic))[active]
        scale = jnp.max(jnp.abs(divergence(current, dynamic)[active]))
        self.assertGreater(float(scale), 0.)
        self.assertLess(float(jnp.max(jnp.abs(residual)) / scale), 2.e-12)
        self.assertClose(jnp.sum(rho0[active]) * dynamic.dx * dynamic.dy * dynamic.dz, .7)

    def test_cic_tsc_filtered_continuity_at_periodic_seam(self):
        for shape in (1, 2):
            for choice in ("bilinear", "digital"):
                with self.subTest(shape=shape, filter=choice):
                    self.continuity_case(shape, choice)

    @unittest.skipUnless(jax.local_device_count() >= 2, "requires two JAX devices")
    def test_cic_tsc_filtered_continuity_at_tile_seam(self):
        for shape in (1, 2):
            for choice in ("bilinear", "digital"):
                with self.subTest(shape=shape, filter=choice):
                    self.continuity_case(shape, choice, tile_shape=(4, 1, 1), crossing=3.95)

    @unittest.skipUnless(jax.local_device_count() >= 2, "requires two JAX devices")
    def test_global_tiled_vector_tensor_parity_and_halo_refresh(self):
        static, dynamic = parameters(Ny=4, y_wind=4., tile_shape=(4, 4, 1))
        g = static.guard_cells
        ix = (4 * jnp.arange(2)[:, None] + jnp.arange(-g, 4 + g)[None, :]) % 8
        iy = jnp.arange(-g, 4 + g) % 4
        iz = jnp.zeros(1 + 2 * g, dtype=jnp.int32)

        def periodic_tiles(global_values):
            return global_values[..., ix[:, None, None, :, None, None],
                                 iy[None, None, None, None, :, None],
                                 iz[None, None, None, None, None, :]]

        keys = jax.random.split(jax.random.PRNGKey(91), 2)
        for leading, key in zip(((3,), (3, 3)), keys):
            values = jax.random.normal(key, leading + (8, 4, 1), dtype=jnp.float64)
            tiled = periodic_tiles(values)
            # Invalid guards ensure that matching the global result requires
            # both the pre-filter and post-filter cross-device refreshes.
            owned = (Ellipsis,) + (slice(g, -g),) * 3
            tiled = jnp.full_like(tiled, jnp.nan).at[owned].set(tiled[owned])
            layout = P(*(None,) * len(leading), "tile_x", "tile_y", "tile_z", None, None, None)
            tiled = jax.device_put(tiled, NamedSharding(static.field_mesh, layout))
            self.assertEqual(len(tiled.addressable_shards), 2)
            for choice in ("bilinear", "digital"):
                with self.subTest(leading=leading, filter=choice):
                    settings = static._replace(current_filter=choice)
                    expected = periodic_tiles(filter_gr_grid(values, settings, dynamic))
                    actual = filter_gr_grid(tiled, settings, dynamic, tiled=True)
                    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
                    self.assertClose(actual, expected)

    def test_native_step_preserves_filtered_gauss_residual(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice, shape_factor=2)
            metric = initialize_flat_cartesian_metric(static, dynamic)
            old = particles_at([7.999, 0., 0.])
            old = old._replace(u=jnp.broadcast_to(jnp.array([.2, .03, -.01]), old.u.shape))
            zero = empty_tiled_vector(static, dynamic)
            rho = empty_tiled_scalar(static, dynamic)
            fields = (zero, zero, zero, rho, rho, (zero, zero), metric, (zero, zero), False)
            # The initial nonzero residual is intentional: exact continuity
            # preserves it even when this particle crosses the periodic seam.
            before, _ = electromagnetic_constraints(old, species(), fields, static, dynamic)
            new, after = jax.jit(lambda p, f: time_loop_static_metric(p, species(), f, static, dynamic))(old, fields)
            final, magnetic = electromagnetic_constraints(new, species(), after, static, dynamic)
            active = (Ellipsis,) + (slice(3, -3),) * 3
            self.assertClose(final[active], before[active])
            self.assertClose(magnetic[active], jnp.zeros_like(magnetic[active]))
            self.assertFalse(bool(after[-1]))
            self.assertEqual(int(jnp.sum(old.active)), int(jnp.sum(new.active)))

    def test_config_acceptance_and_rejection(self):
        for solver in ("static_metric", "dynamic_metric"):
            for choice in ("bilinear", "digital"):
                config = dict(solver=solver, metric="numerical", filter_j=choice,
                              current_calculation="esirkepov")
                _validate_current_filter_contract(config)
                for bad in (dict(metric="flat_spherical"), dict(x_bc="conducting"),
                            dict(particle_y_bc="absorbing")):
                    with self.assertRaisesRegex(ValueError, "Cartesian|periodic"):
                        _validate_current_filter_contract(dict(config, **bad))
        for solver in ("electrodynamic_yee", "dark_matter_yee"):
            with self.assertRaisesRegex(ValueError, "use filter_j='none'"):
                _validate_current_filter_contract(dict(solver=solver, filter_j="bilinear",
                                                       current_calculation="esirkepov"))
        static, _ = parameters()
        for bad in (dict(metric="flat_spherical"), dict(boundary_conditions=(0, 1, 0)),
                    dict(particle_boundary_conditions=(2, 0, 0))):
            with self.assertRaisesRegex(ValueError, "Cartesian|periodic"):
                validate_gr_filter(static._replace(**bad))
        validate_gr_filter(static._replace(current_filter="none", metric="flat_spherical"))
        # Global periodic defaults cannot override contradictory per-species
        # declarations silently when selecting the matched periodic method.
        with self.assertRaisesRegex(ValueError, "periodic particle species"):
            initialize_simulation({
                "simulation_parameters": dict(solver="static_metric", filter_j="bilinear"),
                "particle1": dict(x_bc="absorbing"),
            })

    def test_current_transform_cannot_double_filter_the_current(self):
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice)
            metric = initialize_flat_cartesian_metric(static, dynamic)
            zero = empty_tiled_vector(static, dynamic)
            rho = empty_tiled_scalar(static, dynamic)
            fields = (zero, zero, zero, rho, rho, (zero, zero), metric, (zero, zero), False)
            with self.assertRaisesRegex(ValueError, "filter_j or current_transform, not both"):
                time_loop_static_metric(particles_at([2.3, 0., 0.]), species(), fields,
                                        static, dynamic, current_transform=lambda current: current)

    def test_filter_requires_all_six_periodic_bssn_faces(self):
        try:
            from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
        except ModuleNotFoundError as error:
            if error.name == "JAX_BSSN":
                self.skipTest("Optional JAX_BSSN dependency is not installed")
            raise

        faces = ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")
        periodic = dict.fromkeys(faces, 0)
        for choice in ("bilinear", "digital"):
            static, dynamic = parameters(choice, solver="dynamic_metric")
            build_bssn_parameters(static, dynamic, evolution_parameters=periodic)
            for face in faces:
                with self.subTest(filter=choice, face=face):
                    controls = dict(periodic, **{face: 1})
                    with self.assertRaisesRegex(ValueError, "all six BSSN boundaries"):
                        build_bssn_parameters(static, dynamic, evolution_parameters=controls)
                    # The existing unfiltered solver still permits independent
                    # gravitational boundaries, including Sommerfeld faces.
                    unfiltered = build_bssn_parameters(static._replace(current_filter="none"),
                                                       dynamic, evolution_parameters=controls)
                    self.assertEqual(int(getattr(unfiltered.evolution_parameters, face)), 1)


if __name__ == "__main__":
    unittest.main()
