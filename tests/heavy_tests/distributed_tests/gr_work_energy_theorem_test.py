from tests.support.gr_work_energy_theorem_runtime_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    WorkEnergyAssertions,
    build_bssn_parameters,
    jax,
    jnp,
    parameters,
    tiles_to_bssn,
    unittest,
    volume_integral,
)


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):
    def test_multitile_ownership_excludes_poisoned_ghosts_and_counts_pec_endpoints_once(self):
        origins, lengths = (.2, -.3, .4), (4., 3., 1.4)
        counts, widths = (2, 2, 2), (4, 3, 2)
        for boundaries in ((0, 0, 0), (1, 1, 1), (BC_CONSTANT, 1, 0), (1, 0, BC_CONSTANT)):
            with self.subTest(boundaries=boundaries):
                static, dynamic = parameters(
                    Nx=8, Ny=6, Nz=4, x_wind=lengths[0], y_wind=lengths[1], z_wind=lengths[2],
                    x_min=origins[0], y_min=origins[1], z_min=origins[2],
                    tile_shape=(8, 6, 4), boundary_conditions=boundaries)
                # Synthetic tiles test ownership independently of mesh/device
                # allocation. Internal duplicate endpoints remain poisoned.
                static = static._replace(tile_shape=widths, field_mesh=None)
                g = static.guard_cells
                spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
                slopes = tuple(s if bc else 0. for s, bc in zip((.3, -.2, .5), boundaries))
                tiled = jnp.full((2,) + counts + tuple(width + 2*g for width in widths), jnp.nan)
                for tx in range(2):
                    for ty in range(2):
                        for tz in range(2):
                            tile = (tx, ty, tz)
                            owned = tuple(width + (bc == BC_CONDUCTING and t == count-1)
                                          for t, count, width, bc in zip(tile, counts, widths, boundaries))
                            x, y, z = tuple(origin + dx*(t*width + jnp.arange(n))
                                            for origin, dx, t, width, n in zip(origins, spacing, tile, widths, owned))
                            linear = (4. + slopes[0]*x[:, None, None] + slopes[1]*y[None, :, None]
                                      + slopes[2]*z[None, None, :])
                            data = jnp.stack((jnp.ones_like(linear), linear))
                            interior = (slice(None),) + tile + tuple(slice(g, g+n) for n in owned)
                            tiled = tiled.at[interior].set(data)

                gathered = jax.jit(lambda values: tiles_to_bssn(values, static))(tiled)
                shape = tuple(count*width + (bc == BC_CONDUCTING)
                              for count, width, bc in zip(counts, widths, boundaries))
                x, y, z = tuple(origin + dx*jnp.arange(n)
                                for origin, dx, n in zip(origins, spacing, shape))
                expected = (4. + slopes[0]*x[:, None, None] + slopes[1]*y[None, :, None]
                            + slopes[2]*z[None, None, :])
                self.assertEqual(gathered.shape, (2,) + shape)
                self.assertClose(gathered[0], jnp.ones(shape))
                self.assertClose(gathered[1], expected)
                volume = lengths[0]*lengths[1]*lengths[2]
                self.assertClose(volume_integral(gathered[0], static, dynamic), volume)
                self.assertClose(volume_integral(gathered[1], static, dynamic),
                                 volume*(4. + sum(s*(origin + length/2)
                                                 for s, origin, length in zip(slopes, origins, lengths))))


if __name__ == "__main__":
    unittest.main()
