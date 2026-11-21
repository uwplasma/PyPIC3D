from tests.support.physics_static_metric_validation_fixtures import (
    FIELD_CASES,
    Metric,
    TestCurrentDepositionConvergenceFixtures,
    TestFieldSolverConvergenceFixtures,
    _flat_cylindrical_metric_at_position,
    _flat_spherical_metric_at_position,
    _kerr_schild_cartesian_metric_at_position,
    build_center_metric,
    contravariant_three_velocity,
    convergence_order,
    covariant_lorentz_factor,
    deposition_velocity_field,
    exact_source_velocity,
    gaussian_normal_metric_at_position,
    generic_lapse_shift_metric_at_position,
    jax,
    jnp,
    kerr_cartesian_metric_at_position,
    kerr_spherical_metric_at_position,
    magnetic_boris_rotation,
    parameters,
    sample_center_metric,
    schwarzschild_cartesian_metric_at_position,
    tearDownModule,
    unittest,
)


class TestAnalyticMetricsAgainstClosedForm(unittest.TestCase):
    def test_kerr_schild_cartesian_reproduces_the_closed_form_four_metric(self):
        """``(alpha, beta^i, gamma_ij)`` must rebuild ``eta_mu_nu + 2 H l_mu l_nu``."""

        def closed_form(position, mass, spin):
            x, y, z = position
            rho_squared = x * x + y * y + z * z
            r_squared = 0.5 * (
                rho_squared
                - spin**2
                + jnp.sqrt((rho_squared - spin**2) ** 2 + 4.0 * spin**2 * z * z)
            )
            r = jnp.sqrt(r_squared)
            denominator = r_squared + spin**2
            ell = jnp.array(
                [
                    1.0,
                    (r * x + spin * y) / denominator,
                    (r * y - spin * x) / denominator,
                    z / r,
                ]
            )
            H = mass * r**3 / (r**4 + spin**2 * z * z)
            return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])) + 2.0 * H * jnp.outer(ell, ell)

        for spin in (0.0, 0.5, 0.9, 0.998):
            for position in ((5.0, 2.0, 1.0), (3.0, -1.5, 0.8), (-4.0, 1.0, -2.0)):
                point = jnp.asarray(position, dtype=float)
                lapse, shift, gamma, _gamma_inv, _sqrt_gamma = (
                    _kerr_schild_cartesian_metric_at_position(point, mass=1.0, spin=spin)
                )
                shift_lower = gamma @ shift
                rebuilt = jnp.zeros((4, 4))
                rebuilt = rebuilt.at[0, 0].set(-(lapse**2) + jnp.dot(shift_lower, shift))
                rebuilt = rebuilt.at[0, 1:].set(shift_lower)
                rebuilt = rebuilt.at[1:, 0].set(shift_lower)
                rebuilt = rebuilt.at[1:, 1:].set(gamma)
                error = float(jnp.max(jnp.abs(rebuilt - closed_form(point, 1.0, spin))))
                self.assertLess(error, 1.0e-12, f"spin={spin} position={position} error={error}")

    def test_every_metric_is_algebraically_self_consistent(self):
        """``gamma . gamma_inv = I`` and ``sqrt_gamma**2 = det(gamma)``."""
        cases = (
            (_flat_cylindrical_metric_at_position, (2.7, 0.4, 0.1)),
            (_flat_spherical_metric_at_position, (3.1, 1.2, 0.3)),
            (schwarzschild_cartesian_metric_at_position, (5.0, 2.0, 1.0)),
            (kerr_cartesian_metric_at_position(0.9), (5.0, 2.0, 1.0)),
            (kerr_cartesian_metric_at_position(0.998), (5.0, 2.0, 1.0)),
            (kerr_spherical_metric_at_position(0.9), (7.0, 1.3, 0.4)),
            (kerr_spherical_metric_at_position(0.998), (2.2, 0.7, 0.4)),
            (gaussian_normal_metric_at_position, (0.4, -0.3, 0.9)),
            (generic_lapse_shift_metric_at_position, (0.4, -0.3, 0.9)),
        )
        for metric_at_position, position in cases:
            point = jnp.asarray(position, dtype=float)
            _lapse, _shift, gamma, gamma_inv, sqrt_gamma = metric_at_position(point)
            identity_error = float(jnp.max(jnp.abs(gamma @ gamma_inv - jnp.eye(3))))
            determinant = jnp.linalg.det(gamma)
            determinant_error = float(jnp.abs(sqrt_gamma**2 - determinant) / jnp.abs(determinant))
            self.assertLess(identity_error, 1.0e-12)
            self.assertLess(determinant_error, 1.0e-12)
            self.assertGreater(float(determinant), 0.0)


class TestFieldSolverConvergence(TestFieldSolverConvergenceFixtures, unittest.TestCase):


    def test_constitutive_relations_are_exact_when_no_interpolation_is_needed(self):
        """
        For a diagonal three-metric with zero shift, ``E_i`` and ``H_i`` need no
        off-location interpolation and must be reproduced to round-off.
        """
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES[:2]:
            error_E, error_H, _dB, _dD = self._field_errors(
                metric_at_position, wind, mins, metric_name, 16
            )
            self.assertLess(error_E, 1.0e-13, f"{name}: {error_E}")
            self.assertLess(error_H, 1.0e-13, f"{name}: {error_H}")


class TestCurrentDepositionConvergence(TestCurrentDepositionConvergenceFixtures, unittest.TestCase):


    def test_metric_sampled_source_velocity_is_third_order(self):
        """
        Hermite primitive values and their consistent inverse give a generally
        third-order source velocity ``alpha v^i - beta^i``. Sampling many
        off-node positions avoids special sub-cell superconvergence.
        """
        key = jax.random.PRNGKey(11)
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES[2:]:
            errors = []
            for N in (12, 24, 48):
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, 0.01, metric_name=metric_name
                )
                metric = build_center_metric(dynamic_parameters, metric_at_position)
                fractions = jax.random.uniform(key, (400, 3), minval=0.2, maxval=0.8)
                positions = jnp.asarray(mins) + fractions * jnp.asarray(wind)
                sampled = sample_center_metric(
                    metric, positions, static_parameters, dynamic_parameters
                )
                momenta = jax.vmap(deposition_velocity_field)(positions)
                velocity = contravariant_three_velocity(momenta, sampled.gamma_inv)
                computed = sampled.lapse[:, None] * velocity - sampled.shift
                exact = jax.vmap(
                    lambda point: exact_source_velocity(metric_at_position, point)
                )(positions)
                errors.append(
                    float(jnp.sqrt(jnp.mean(jnp.sum((computed - exact) ** 2, axis=1))))
                )
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 2.65, "\n".join(report))
                self.assertLess(order, 3.35, "\n".join(report))


class TestInvariantsAndConstraints(unittest.TestCase):
    def test_generalized_boris_rotation_preserves_the_lorentz_factor(self):
        """
        The magnetic half of the push is a rotation in the metric ``gamma_ij``
        (Aperture Eqs. 17-19, FPIC Eq. 16), so
        ``Gamma = sqrt(1 + gamma^ij u_i u_j)`` must be invariant to round-off for
        any time step, not just small ones.
        """
        charts = (
            ("flat cylindrical", _flat_cylindrical_metric_at_position, (2.7, 0.4, 0.1)),
            ("flat spherical", _flat_spherical_metric_at_position, (3.1, 1.2, 0.3)),
            ("Kerr a=0.9 KS-cart", kerr_cartesian_metric_at_position(0.9), (5.0, 2.0, 1.0)),
            ("Kerr a=0.9 KS-sph", kerr_spherical_metric_at_position(0.9), (7.0, 1.3, 0.4)),
            ("Gaussian-normal", gaussian_normal_metric_at_position, (0.4, -0.3, 0.9)),
            ("generic lapse+shift", generic_lapse_shift_metric_at_position, (0.4, -0.3, 0.9)),
        )
        key = jax.random.PRNGKey(3)
        for name, metric_at_position, position in charts:
            lapse, shift, gamma, gamma_inv, sqrt_gamma = metric_at_position(
                jnp.asarray(position, dtype=float)
            )
            sampled = Metric(
                lapse=jnp.asarray([lapse]),
                shift=shift[None, :],
                gamma=gamma[None, :, :],
                gamma_inv=gamma_inv[None, :, :],
                sqrt_gamma=jnp.asarray([sqrt_gamma]),
            )
            worst = 0.0
            for trial in range(25):
                key_u, key_B = jax.random.split(jax.random.fold_in(key, trial))
                u_cov = 1.5 * jax.random.normal(key_u, (1, 3))
                B_con = 2.0 * jax.random.normal(key_B, (1, 3))
                for dt in (0.01, 0.5, 5.0, 50.0):
                    rotated = magnetic_boris_rotation(
                        u_cov, B_con, sampled, jnp.asarray([1.3]), dt
                    )
                    before = float(covariant_lorentz_factor(u_cov, sampled.gamma_inv)[0])
                    after = float(covariant_lorentz_factor(rotated, sampled.gamma_inv)[0])
                    worst = max(worst, abs(after - before) / before)
            self.assertLess(worst, 1.0e-12, f"{name}: {worst}")


if __name__ == "__main__":
    unittest.main()
