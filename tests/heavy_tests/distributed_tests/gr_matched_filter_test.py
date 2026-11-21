from tests.support.gr_matched_filter_runtime_fixtures import (
    Esirkepov_current,
    NamedSharding,
    P,
    compute_rho,
    divergence,
    empty_tiled_vector,
    filter_gr_grid,
    jax,
    jnp,
    parameters,
    particles_at,
    species,
    unittest,
)


class TestGRMatchedFilter(unittest.TestCase):
    def assertClose(self, actual, expected, atol=2.e-12, rtol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=atol, rtol=rtol)),
                        f"max difference = {float(jnp.max(jnp.abs(actual - expected)))}")

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


if __name__ == "__main__":
    unittest.main()
