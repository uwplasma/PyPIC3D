"""Single-device numerical tests."""

from tests.support.static_metric_esirkepov_fixtures import (
    DepositionDispatchFixtures,
    Esirkepov_current,
    GRESirkepovConfigurationFixtures,
    GRESirkepovContinuityFixtures,
    GRESirkepovConventionsFixtures,
    GaussConstraintPreservationFixtures,
    TiledParticles,
    _backward_divergence,
    _continuity_residual,
    _encode_current_calculation,
    _interior,
    _unit_species,
    _validate_current_filter_contract,
    _validate_tiled_yee_configuration,
    empty_tiled_vector,
    initialize_flat_spherical_metric,
    jax,
    jnp,
    kernel_parameters,
    np,
    unittest,
    update_D,
    update_tiled_vector_ghost_cells,
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


    def test_frozen_axes_deposit_no_current(self):
        static_parameters, dynamic_parameters = self._flat_setup()
        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        displaced = position + jnp.asarray([(0.31, 0.22, -0.18)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.zeros_like(position)
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        particles_old = TiledParticles(x=position, u=velocity, active=active)
        particles_new = TiledParticles(x=displaced, u=velocity, active=active)
        template = empty_tiled_vector(static_parameters, dynamic_parameters)

        def deposit(update_x):
            return Esirkepov_current(
                particles_old, particles_new,
                _unit_species()._replace(update_x=jnp.asarray([update_x])),
                template, static_parameters, dynamic_parameters,
            )

        full = deposit([True, True, True])
        masked = deposit([False, True, False])
        disabled = deposit([False, False, False])

        self.assertGreater(float(jnp.max(jnp.abs(full[1]))), 0.0)
        self.assertTrue(bool(jnp.allclose(masked[0], 0.0)))
        self.assertTrue(bool(jnp.allclose(masked[2], 0.0)))
        for component in disabled:
            self.assertTrue(bool(jnp.allclose(component, 0.0)))


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
        D = update_tiled_vector_ghost_cells(
            tuple(jax.random.normal(keys[i], shape) for i in range(3)),
            static_parameters,
            int(static_parameters.guard_cells),
        )
        H = update_tiled_vector_ghost_cells(
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



class TestDepositionDispatch(DepositionDispatchFixtures, unittest.TestCase):
    def test_both_schemes_jit_and_stay_finite(self):
        for current_deposition in ("esirkepov", "GR_direct"):
            with self.subTest(current_deposition=current_deposition):
                fields = self._one_jitted_step(current_deposition)
                for component in fields[0]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
                for component in fields[2]:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(component))))


    def test_the_selector_actually_switches_kernels(self):
        """
        Guards against the branch being silently ignored: the two schemes
        deposit genuinely different currents for the same particle state.
        """

        esirkepov = self._one_jitted_step("esirkepov")[2]
        direct = self._one_jitted_step("GR_direct")[2]
        self.assertGreater(float(jnp.max(jnp.abs(esirkepov[0]))), 0.0)
        self.assertGreater(float(jnp.max(jnp.abs(direct[0]))), 0.0)
        differs = any(
            not bool(jnp.allclose(esirkepov[i], direct[i])) for i in range(3)
        )
        self.assertTrue(differs)



class TestGRESirkepovConfiguration(GRESirkepovConfigurationFixtures, unittest.TestCase):
    def test_encoder_accepts_shared_esirkepov(self):
        self.assertEqual(_encode_current_calculation("esirkepov"), "esirkepov")


    def test_encoder_still_maps_the_existing_schemes(self):
        self.assertEqual(_encode_current_calculation("GR_direct_deposition"), "GR_direct")
        self.assertEqual(_encode_current_calculation("esirkepov"), "esirkepov")
        self.assertEqual(_encode_current_calculation("j_from_rhov"), "direct")


    def test_encoder_rejects_an_unknown_scheme(self):
        with self.assertRaises(ValueError):
            _encode_current_calculation("not_a_scheme")


    def test_static_metric_accepts_the_new_scheme(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(self._static_config(), dynamic_config)


    def test_static_metric_still_accepts_direct_deposition(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        _validate_tiled_yee_configuration(
            self._static_config(current_calculation="GR_direct_deposition"),
            dynamic_config,
        )


    def test_all_electromagnetic_solvers_accept_shared_esirkepov(self):
        dynamic_config = {"Nx": 8, "Ny": 8, "Nz": 8}
        for solver in ("electrodynamic_yee", "dark_matter_yee", "static_metric"):
            pusher = "hybrid_boris_geodesic" if solver == "static_metric" else "boris"
            _validate_tiled_yee_configuration(
                self._static_config(solver=solver, particle_pusher=pusher), dynamic_config,
            )


    def test_removed_configuration_name_points_to_shared_scheme(self):
        with self.assertRaisesRegex(ValueError, "Unsupported current_calculation.*'esirkepov'"):
            _encode_current_calculation("GR_esirkepov")


    def test_current_filtering_is_rejected(self):
        """Filtering destroys the exact continuity the scheme exists to give."""

        for solver in ("electrodynamic_yee", "dark_matter_yee", "static_metric"):
            for filter_name in ("digital", "bilinear"):
                with self.subTest(solver=solver, filter=filter_name):
                    with self.assertRaisesRegex(ValueError, "use filter_j='none'"):
                        _validate_current_filter_contract(
                            self._static_config(solver=solver, filter_j=filter_name)
                        )


    def test_unfiltered_configuration_is_accepted(self):
        _validate_current_filter_contract(self._static_config())


    def test_direct_deposition_may_still_be_filtered(self):
        _validate_current_filter_contract(
            self._static_config(current_calculation="GR_direct_deposition", filter_j="digital")
        )



if __name__ == "__main__":
    unittest.main()
