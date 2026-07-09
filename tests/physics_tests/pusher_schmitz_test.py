import math
import unittest

import jax

from PyPIC3D.pusher.boris import relativistic_boris_single_particle
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle


jax.config.update("jax_enable_x64", True)


class TestSchmitzParticlePusherBenchmarks(unittest.TestCase):
    """
    Focused tests from Schmitz 2026, Section 3, for pushers supported here.

    The paper compares several relativistic pushers.  PyPIC3D currently exposes
    relativistic Boris and Higuera-Cary, so the checks below use those two
    update maps and preserve the paper's normalised parameters where possible.
    """

    constants = {"C": 1.0}
    q = 1.0
    m = 1.0

    def _gamma(self, v):
        v2 = sum(component * component for component in v)
        return 1.0 / math.sqrt(1.0 - v2)

    def _velocity_from_u(self, u):
        gamma = math.sqrt(1.0 + sum(component * component for component in u))
        return tuple(component / gamma for component in u)

    def _step(self, pusher, v, E, B, dt):
        vx, vy, vz = pusher(
            v[0],
            v[1],
            v[2],
            E[0],
            E[1],
            E[2],
            B[0],
            B[1],
            B[2],
            self.q,
            self.m,
            dt,
            self.constants,
        )
        return float(vx), float(vy), float(vz)

    def _advance(self, pusher, v, E, B, dt, steps):
        for _ in range(steps):
            v = self._step(pusher, v, E, B, dt)
        return v

    def test_constant_magnetic_field_preserves_energy_for_boris_and_higuera_cary(self):
        # Section 3.1: E = 0, B = B_z zhat, gamma0 = 1.001 and 10,
        # with ten steps of dt omega_c / (2 pi) = 0.1.
        B = (0.0, 0.0, 1.0)
        E = (0.0, 0.0, 0.0)

        for gamma0 in (1.001, 10.0):
            u0 = (math.sqrt(gamma0**2 - 1.0), 0.0, 0.0)
            v0 = self._velocity_from_u(u0)
            dt = 0.2 * math.pi * gamma0

            for pusher in (relativistic_boris_single_particle, higuera_cary_single_particle):
                v = self._advance(pusher, v0, E, B, dt, 10)

                self.assertAlmostEqual(self._gamma(v), gamma0, delta=2.0e-12)

    def test_force_free_crossed_fields_expose_boris_relativistic_weakness(self):
        # Section 3.2: E = -v0 x B, gamma0 = 10.  Higuera-Cary is designed
        # to keep the force-free trajectory exact; Boris is retained as a
        # bounded known-weakness comparison rather than forced to pass HC's
        # near-machine-precision criterion.
        gamma0 = 10.0
        u0 = math.sqrt(gamma0**2 - 1.0)
        v0 = u0 / gamma0
        E = (0.0, v0, 0.0)
        B = (0.0, 0.0, 1.0)
        dt = 0.2 * math.pi * gamma0

        higuera_cary_v = (v0, 0.0, 0.0)
        boris_v = (v0, 0.0, 0.0)
        higuera_cary_angle = 0.0
        higuera_cary_energy_error = 0.0
        boris_angle = 0.0
        boris_energy_error = 0.0

        for _ in range(20):
            higuera_cary_v = self._step(higuera_cary_single_particle, higuera_cary_v, E, B, dt)
            boris_v = self._step(relativistic_boris_single_particle, boris_v, E, B, dt)

            higuera_cary_angle = max(higuera_cary_angle, abs(math.atan2(higuera_cary_v[1], higuera_cary_v[0])))
            higuera_cary_energy_error = max(
                higuera_cary_energy_error,
                abs((self._gamma(higuera_cary_v) - gamma0) / (gamma0 - 1.0)),
            )
            boris_angle = max(boris_angle, abs(math.atan2(boris_v[1], boris_v[0])))
            boris_energy_error = max(
                boris_energy_error,
                abs((self._gamma(boris_v) - gamma0) / (gamma0 - 1.0)),
            )

        self.assertLess(higuera_cary_angle, 1.0e-13)
        self.assertLess(higuera_cary_energy_error, 1.0e-12)
        self.assertGreater(boris_angle, 1.0e-3)
        self.assertLess(boris_angle, 2.0e-1)
        self.assertGreater(boris_energy_error, 1.0e-2)

    def _boosted_gyration_exact_velocity(self, lab_time, gamma_particle=10.0, gamma_frame=5.0):
        beta_frame = math.sqrt(1.0 - 1.0 / gamma_frame**2)
        u_particle = math.sqrt(gamma_particle**2 - 1.0)
        v_particle = u_particle / gamma_particle
        omega = 1.0 / gamma_particle
        radius = v_particle / omega

        def x_moving_frame(t):
            return radius * (1.0 - math.cos(omega * t))

        def vx_moving_frame(t):
            return v_particle * math.sin(omega * t)

        def vy_moving_frame(t):
            return v_particle * math.cos(omega * t)

        moving_time = lab_time / gamma_frame
        for _ in range(8):
            residual = gamma_frame * (
                moving_time + beta_frame * x_moving_frame(moving_time)
            ) - lab_time
            derivative = gamma_frame * (
                1.0 + beta_frame * vx_moving_frame(moving_time)
            )
            moving_time -= residual / derivative

        vx_m = vx_moving_frame(moving_time)
        vy_m = vy_moving_frame(moving_time)
        denominator = 1.0 + beta_frame * vx_m
        return (
            (vx_m + beta_frame) / denominator,
            vy_m / (gamma_frame * denominator),
            0.0,
        )

    def test_lorentz_boosted_crossed_field_gyration_matches_exact_velocity(self):
        # Section 3.3: gamma_P = 10 and gamma_M = 5.  Fields are the Lorentz
        # transform of a pure magnetic field in the moving frame, so the exact
        # lab-frame velocity is obtained by transforming the circular orbit.
        gamma_particle = 10.0
        gamma_frame = 5.0
        beta_frame = math.sqrt(1.0 - 1.0 / gamma_frame**2)
        v_particle = math.sqrt(1.0 - 1.0 / gamma_particle**2)
        E = (0.0, beta_frame * gamma_frame, 0.0)
        B = (0.0, 0.0, gamma_frame)
        v0 = (beta_frame, v_particle / gamma_frame, 0.0)
        gyration_period = 2.0 * math.pi * gamma_particle * gamma_frame
        dt = 1.0e-3 * gyration_period
        steps = 20

        exact_v = self._boosted_gyration_exact_velocity(steps * dt, gamma_particle, gamma_frame)
        higuera_cary_v = self._advance(higuera_cary_single_particle, v0, E, B, dt, steps)
        boris_v = self._advance(relativistic_boris_single_particle, v0, E, B, dt, steps)

        higuera_cary_error = math.sqrt(sum((higuera_cary_v[i] - exact_v[i]) ** 2 for i in range(3)))
        boris_error = math.sqrt(sum((boris_v[i] - exact_v[i]) ** 2 for i in range(3)))

        self.assertLess(higuera_cary_error, 1.0e-6)
        self.assertLess(boris_error, 1.0e-3)

    def test_oscillating_parallel_electric_field_returns_to_initial_energy(self):
        # Section 3.7: B = B0 zhat and E = E0 zhat cos(omega0 t), with
        # gamma_perp = 1.1, omega0 = omega_perp / 2, and
        # E0 = 10 m omega0 c / q.  A midpoint field sample preserves the
        # oscillatory impulse over five electric-field periods.
        gamma_perp = 1.1
        u_perp = math.sqrt(gamma_perp**2 - 1.0)
        v0 = self._velocity_from_u((u_perp, 0.0, 0.0))
        B0 = 1.0
        omega_perp = B0 / gamma_perp
        omega0 = 0.5 * omega_perp
        E0 = 10.0 * omega0
        electric_period = 2.0 * math.pi / omega0
        steps_per_period = 100
        dt = electric_period / steps_per_period
        steps = 5 * steps_per_period

        for pusher in (relativistic_boris_single_particle, higuera_cary_single_particle):
            v = v0
            for step in range(steps):
                t_mid = (step + 0.5) * dt
                E = (0.0, 0.0, E0 * math.cos(omega0 * t_mid))
                B = (0.0, 0.0, B0)
                v = self._step(pusher, v, E, B, dt)

            self.assertAlmostEqual(v[2], 0.0, delta=1.0e-10)
            self.assertAlmostEqual(self._gamma(v), gamma_perp, delta=1.0e-10)


if __name__ == "__main__":
    unittest.main()
