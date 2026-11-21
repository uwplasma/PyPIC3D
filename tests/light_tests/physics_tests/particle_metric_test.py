"""One-tile numerical and physical checks."""

from tests.support.particle_metric_fixtures import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
    interpolate_hermite,
    interpolate_metric,
    jax,
    jnp,
    kernel_parameters,
    manufactured,
    np,
    partial,
    sampled_rotation_checks,
    unittest,
)


class TestHermite(unittest.TestCase):
    def test_tensor_quadratic_reproduction(self):
        grid = tuple(jnp.arange(-3, 14, dtype=float) * h for h in (0.1, 0.13, 0.17))

        def f(p):
            x, y, z = jnp.moveaxis(p, -1, 0)
            return (
                1
                + 0.3 * x
                - 0.2 * y
                + 0.1 * z
                + x * x
                + 2 * y * y
                + 0.4 * z * z
                + 0.7 * x * y * z
                + 0.8 * x * x * y * y
            )

        values = f(jnp.stack(jnp.meshgrid(*grid, indexing="ij"), axis=-1))
        q = jnp.array([[0.247, 0.381, 0.418], [0.41, 0.53, 0.67]])
        get = lambda q: interpolate_hermite(values, q, grid, (True, True, True), (3, 3, 3))
        np.testing.assert_allclose(get(q), f(q), atol=3e-14, rtol=3e-14)
        for k in range(3):
            direction = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
            np.testing.assert_allclose(
                jax.jvp(get, (q,), (direction,))[1],
                jax.jvp(f, (q,), (direction,))[1],
                atol=1e-12,
            )

    def test_nodal_values_and_shared_slopes(self):
        x = jnp.arange(-3, 13, dtype=float) * 0.125
        grid = (x, jnp.arange(7, dtype=float), jnp.arange(7, dtype=float))
        f = jnp.sin(x) + 0.3 * jnp.cos(3 * x)
        data = jnp.broadcast_to(f[:, None, None], (len(x), 7, 7))
        get = lambda t: interpolate_hermite(
            data, jnp.array([t, 3.0, 3.0]), grid, (True, False, False), (3, 3, 3)
        )
        derivative = jax.grad(get)
        for i in range(2, len(x) - 3):
            self.assertAlmostEqual(float(get(x[i])), float(f[i]), places=13)
            expected = (f[i + 1] - f[i - 1]) / (2 * 0.125)
            self.assertLess(abs(float(derivative(x[i]) - expected)), 1e-12)
            self.assertLess(
                abs(float(derivative(x[i] - 1e-10) - derivative(x[i] + 1e-10))), 1e-8
            )


class TestParticleMetric(unittest.TestCase):
    def test_shared_reconstruction_for_every_supported_chart(self):
        providers = (
            ("flat_cartesian", initialize_flat_cartesian_metric),
            ("flat_cylindrical", initialize_flat_cylindrical_metric),
            ("flat_spherical", initialize_flat_spherical_metric),
            ("kerr_schild_cartesian", partial(initialize_kerr_schild_cartesian_metric, spin=.4)),
            ("kerr_schild_spherical", partial(initialize_kerr_schild_spherical_metric, spin=.4)),
        )
        for name, initialize in providers:
            with self.subTest(metric=name):
                s, d = kernel_parameters(
                    Nx=8, Ny=8, Nz=1, x_min=2., y_min=.6, z_min=0.,
                    x_wind=1., y_wind=.8, z_wind=1., solver="static_metric",
                    particle_pusher="hybrid_boris_geodesic", metric=name, metric_spin=.4,
                )
                m = initialize(s, d)
                tile = jax.tree.map(lambda a: a[0, 0, 0], m.center)
                grid = tuple(a[0, 0, 0] for a in d.grids.tiled_center_grid)
                q = jnp.array([[2.31, .83, .2], [2.67, 1.17, .8]])
                get = lambda metric, derivatives=True: interpolate_metric(
                    metric, q, grid, name, (True, True, False), (3, 3, 3),
                    derivatives=derivatives)
                original = get(tile)
                poisoned = tile._replace(gamma_inv=jnp.full_like(tile.gamma_inv, jnp.nan),
                                         sqrt_gamma=jnp.full_like(tile.sqrt_gamma, jnp.nan))
                for a, b in zip(original, get(poisoned)):
                    np.testing.assert_array_equal(a, b)
                values = get(tile, False)
                for a, b in zip(original[:5], values[:5]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(values[5:], (None, None, None))
                np.testing.assert_allclose(original.gamma @ original.gamma_inv,
                                           jnp.broadcast_to(jnp.eye(3), original.gamma.shape), atol=1e-12)
                np.testing.assert_allclose(original.sqrt_gamma**2, jnp.linalg.det(original.gamma), rtol=1e-12)
                np.testing.assert_array_equal(original.grad_lapse[..., 2], 0.)
                np.testing.assert_array_equal(original.grad_shift[..., 2], 0.)
                np.testing.assert_array_equal(original.grad_gamma_inv[..., 2, :, :], 0.)
                # A chart name must never override supplied grid primitives.
                changed = get(tile._replace(lapse=tile.lapse*.7, shift=tile.shift+.1,
                                            gamma=tile.gamma*2.))
                np.testing.assert_allclose(changed.lapse, original.lapse*.7, atol=1e-12)
                np.testing.assert_allclose(changed.shift, original.shift+.1, atol=1e-12)
                np.testing.assert_allclose(changed.gamma_inv, original.gamma_inv/2., atol=1e-12)

    def test_inverse_and_derivatives(self):
        grid, m = manufactured()
        q = jnp.array([[0.247, 0.381, 3.0], [0.415, 0.274, 3.0]])
        evaluate = lambda q: interpolate_metric(m, q, grid, "numerical", (True, True, False), (3, 3, 3))
        a = evaluate(q)
        dh = a.grad_gamma_inv
        np.testing.assert_allclose(
            a.gamma @ a.gamma_inv, jnp.broadcast_to(jnp.eye(3), a.gamma.shape), atol=1e-12
        )
        np.testing.assert_allclose(a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12)
        np.testing.assert_allclose(
            jnp.linalg.det(a.gamma) * jnp.linalg.det(a.gamma_inv), 1.0, atol=1e-12
        )
        for k in range(3):
            tangent = jnp.broadcast_to(jnp.eye(3)[k], q.shape)
            actual = jax.jvp(lambda x: evaluate(x).gamma_inv, (q,), (tangent,))[1]
            np.testing.assert_allclose(dh[:, k], actual, atol=1e-12, rtol=1e-12)
            eps = 1e-5
            plus = evaluate(q + eps * tangent)
            minus = evaluate(q - eps * tangent)
            np.testing.assert_allclose(
                dh[:, k], (plus.gamma_inv - minus.gamma_inv) / (2 * eps), atol=1e-10, rtol=1e-8
            )
            np.testing.assert_allclose(
                a.grad_lapse[:, k], (plus.lapse - minus.lapse) / (2 * eps), atol=1e-10
            )
            np.testing.assert_allclose(
                a.grad_shift[:, :, k], (plus.shift - minus.shift) / (2 * eps), atol=1e-10
            )
        np.testing.assert_array_equal(dh[:, 2], 0.0)

    def test_sampled_rotation_invariants(self):
        for row in sampled_rotation_checks(32):
            for key in ("inverse_defect", "norm_error", "parallel_error", "reversal_error"):
                self.assertLess(row[key], 1e-12, (key, row))

    def test_signed_orientation(self):
        grid, m = manufactured()
        for chart, q in [
            ("flat_cylindrical", jnp.array([[-0.15, 0.25, 3.0]])),
            ("flat_spherical", jnp.array([[0.25, -0.15, 3.0]])),
        ]:
            a = interpolate_metric(m, q, grid, chart, (True, True, False), (3, 3, 3))
            self.assertLess(float(a.sqrt_gamma[0]), 0.0)
            np.testing.assert_allclose(a.sqrt_gamma**2, jnp.linalg.det(a.gamma), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
