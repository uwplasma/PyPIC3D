"""One-tile numerical and physical checks."""

from tests.support.static_metric_esirkepov_fixtures import (
    Esirkepov_current,
    GRESirkepovContinuityFixtures,
    GRESirkepovConventionsFixtures,
    GaussConstraintPreservationFixtures,
    TiledParticles,
    _backward_divergence,
    _continuity_residual,
    _interior,
    _unit_species,
    empty_tiled_vector,
    initialize_flat_spherical_metric,
    jax,
    jnp,
    kernel_parameters,
    np,
    unittest,
    update_D,
    update_tiled_ghost_cells,
)


class TestGRESirkepovContinuity(GRESirkepovContinuityFixtures, unittest.TestCase):
    def test_continuity_is_exact_in_every_chart(self):
        for metric_name, x_old, x_new, wind, mins in self.CASES:
            for shape_factor in (1, 2):
                with self.subTest(metric=metric_name, shape_factor=shape_factor):
                    residual, scale = _continuity_residual(
                        metric_name,
                        x_old,
                        x_new,
                        shape_factor=shape_factor,
                        wind=wind,
                        mins=mins,
                    )
                    self.assertGreater(scale, 0.0, "the probe deposited no current")
                    self.assertLess(residual / scale, 1.0e-12)


class TestGRESirkepovConventions(GRESirkepovConventionsFixtures, unittest.TestCase):
    def test_endpoint_current_has_correct_integral_and_ignores_momentum(self):
        static, dynamic = self._flat_setup()
        position = jnp.asarray([0.30, -0.70, 1.10]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.asarray([1.7, 0.9, -1.3])
        old = TiledParticles(position, jnp.zeros_like(position), jnp.ones(position.shape[:-1], bool))
        new = old._replace(x=position + velocity * dynamic.dt)
        species = _unit_species()
        template = empty_tiled_vector(static, dynamic)
        current = Esirkepov_current(old, new, species, template, static, dynamic)
        changed_momentum = Esirkepov_current(
            old._replace(u=jnp.full_like(position, 37.0)),
            new._replace(u=jnp.full_like(position, -19.0)),
            species, template, static, dynamic,
        )
        g = static.guard_cells
        volume = dynamic.dx * dynamic.dy * dynamic.dz
        for axis, (actual, changed) in enumerate(zip(current, changed_momentum)):
            np.testing.assert_array_equal(actual, changed)
            # The integral of coordinate current is charge times mean velocity.
            np.testing.assert_allclose(
                jnp.sum(actual[..., g:-g, g:-g, g:-g]) * volume,
                velocity[axis], rtol=1e-12, atol=1e-12,
            )

    def test_unresolved_current_uses_coordinate_velocity_or_displacement(self):
        for ny in (1, 8):
            for shape in (1, 2):
                with self.subTest(ny=ny, shape=shape):
                    static, dynamic = kernel_parameters(
                        Nx=8, Ny=ny, Nz=1, x_wind=8., y_wind=float(ny), z_wind=1.,
                        tile_shape=(8, ny, 1), dt=.1, shape_factor=shape,
                        current_deposition="esirkepov", current_filter="none",
                    )
                    x = jnp.zeros((1, 1, 1, 1, 1, 3))
                    old = TiledParticles(x, jnp.full_like(x, 99.), jnp.ones(x.shape[:-1], bool))
                    speed = jnp.asarray([.2, -.3, .4]).reshape(x.shape)
                    new = old._replace(x=x + speed * dynamic.dt)
                    template = empty_tiled_vector(static, dynamic)
                    species = _unit_species()
                    derived = Esirkepov_current(old, new, species, template, static, dynamic)
                    supplied = Esirkepov_current(
                        old, new, species, template, static, dynamic, coordinate_velocity=2 * speed,
                    )
                    stopped = Esirkepov_current(old, old, species, template, static, dynamic)
                    inactive = Esirkepov_current(
                        old._replace(active=jnp.zeros_like(old.active)),
                        new._replace(active=jnp.zeros_like(new.active)),
                        species, template, static, dynamic,
                    )
                    g = static.guard_cells
                    volume = dynamic.dx * dynamic.dy * dynamic.dz
                    for axis in range(3):
                        np.testing.assert_array_equal(stopped[axis], 0.)
                        np.testing.assert_array_equal(inactive[axis], 0.)
                        factor = 2 if (axis == 2 or (axis == 1 and ny == 1)) else 1
                        np.testing.assert_allclose(supplied[axis], factor * derived[axis], atol=1e-12)
                        np.testing.assert_allclose(
                            jnp.sum(derived[axis][..., g:-g, g:-g, g:-g]) * volume,
                            speed.reshape(3)[axis], rtol=1e-12, atol=1e-12,
                        )


class TestGaussConstraintPreservation(GaussConstraintPreservationFixtures, unittest.TestCase):
    def test_gr_esirkepov_conserves_the_gauss_constraint(self):
        """
        d-(sqrt(g) D) - 4 pi sqrt(g) rho is a discrete invariant of the coupled
        update: the curl contributes nothing to it, and the deposit cancels the
        charge change exactly.  It is not zero here (D starts at zero while rho
        does not) -- it is *constant*, which is what a charge-conserving deposit
        buys and what direct deposition destroys.
        """

        drift, scale = self._constraint_drift("esirkepov")
        self.assertGreater(scale, 0.0)
        self.assertLess(drift / scale, 1.0e-12)

    def test_direct_deposition_does_not_conserve_the_gauss_constraint(self):
        """The contrast case: this is audit finding F3."""

        drift, scale = self._constraint_drift("GR_direct")
        self.assertGreater(drift / scale, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
