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

    def test_continuity_is_exact_on_a_reduced_axis(self):
        # Ny = 1 collapses the polar axis, exercising the out-of-plane
        # displacement branch and collapse_redundant_axis
        residual, scale = _continuity_residual(
            "flat_spherical",
            (5.30, 0.70, 4.50),
            (5.62, 0.93, 4.63),
            N=(8, 1, 8),
            wind=(8.0, 1.0, 8.0),
            mins=(2.0, 0.4, 0.3),
            tile_shape=(8, 1, 8),
        )
        self.assertGreater(scale, 0.0)
        self.assertLess(residual / scale, 1.0e-12)

    def test_stationary_particle_deposits_no_current(self):
        residual, scale = _continuity_residual(
            "flat_spherical",
            (5.30, 1.10, 1.00),
            (5.30, 1.10, 1.00),
            wind=(8.0, 1.6, 1.6),
            mins=(2.0, 0.4, 0.3),
        )
        self.assertEqual(scale, 0.0)
        self.assertEqual(residual, 0.0)


class TestGRESirkepovConventions(GRESirkepovConventionsFixtures, unittest.TestCase):


    def test_returns_densitized_current_in_a_spherical_chart(self):
        """
        The returned current is the density sqrt(gamma) J^i, so its flux
        sqrt(gamma) J^i d^3x is radius-independent while the physical current
        J^i is not.  Mirrors the equivalent GR_direct_deposition test.
        """

        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=1,
            Nz=1,
            x_wind=8.0,
            y_wind=1.0,
            z_wind=1.0,
            x_min=2.0,
            y_min=0.4,
            z_min=0.2,
            dt=0.1,
            tile_shape=(8, 1, 1),
            shape_factor=1,
            solver="static_metric",
            metric="flat_spherical",
            current_deposition="esirkepov",
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
        template = empty_tiled_vector(static_parameters, dynamic_parameters)
        species = _unit_species()
        interior = _interior(static_parameters)
        window = (slice(None), slice(None), slice(None), interior, interior, interior)
        radial_step = 0.05

        def deposit_radial_particle(radius):
            active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
            velocity = jnp.zeros((1, 1, 1, 1, 1, 3))
            old = jnp.asarray((radius, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3))
            new = jnp.asarray((radius + radial_step, 0.4, 0.2)).reshape((1, 1, 1, 1, 1, 3))
            J = Esirkepov_current(
                TiledParticles(x=old, u=velocity, active=active),
                TiledParticles(x=new, u=velocity, active=active),
                species, template, static_parameters, dynamic_parameters,
            )
            physical = jnp.sum(J[0][window] / metric.D[0].sqrt_gamma[window])
            conformal_flux = jnp.sum(
                J[0][window]
                * dynamic_parameters.dx
                * dynamic_parameters.dy
                * dynamic_parameters.dz
            )
            return physical, conformal_flux

        inner_current, inner_flux = deposit_radial_particle(2.5)
        outer_current, outer_flux = deposit_radial_particle(6.5)

        # the swept conformal charge flux is q * dr / dt regardless of radius
        expected_flux = radial_step / dynamic_parameters.dt
        self.assertTrue(bool(jnp.allclose(inner_flux, expected_flux, rtol=1.0e-10)))
        self.assertTrue(bool(jnp.allclose(outer_flux, expected_flux, rtol=1.0e-10)))
        self.assertGreater(float(inner_current), float(outer_current))


class TestGaussConstraintPreservation(GaussConstraintPreservationFixtures, unittest.TestCase):


    def test_backward_divergence_annihilates_the_ampere_curl(self):
        """
        With no current, update_D cannot change the coordinate
        divergence of D at all.  This is why a charge-conserving deposit is
        sufficient on its own, and why the averaged-current auxiliary chain in
        the time loop cannot leak into the constraint: it only ever reaches the
        stored field through a curl.
        """

        static_parameters, dynamic_parameters, metric, _, _ = self._loop_setup("esirkepov")
        key = jax.random.PRNGKey(0)
        shape = empty_tiled_vector(static_parameters, dynamic_parameters)[0].shape
        keys = jax.random.split(key, 6)
        D = update_tiled_ghost_cells(
            tuple(jax.random.normal(keys[i], shape) for i in range(3)),
            static_parameters,
            int(static_parameters.guard_cells),
        )
        H = update_tiled_ghost_cells(
            tuple(jax.random.normal(keys[3 + i], shape) for i in range(3)),
            static_parameters,
            int(static_parameters.guard_cells),
        )
        zero_current = empty_tiled_vector(static_parameters, dynamic_parameters)

        # D is the density sqrt(gamma) D^i, the variable update_D advances
        before = _backward_divergence(D, static_parameters, dynamic_parameters)
        D_next = update_D(
            D, H, zero_current, metric, static_parameters, dynamic_parameters,
            dynamic_parameters.dt,
        )
        after = _backward_divergence(D_next, static_parameters, dynamic_parameters)

        curl_scale = float(jnp.max(jnp.abs(after)))
        self.assertGreater(curl_scale, 0.0)
        self.assertLess(
            float(jnp.max(jnp.abs(after - before))) / curl_scale, 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
