"""Physical moment and staggering checks for the BSSN matter source."""

import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    electromagnetic_moments,
)
from tests.kernel_fixtures import kernel_parameters

try:
    from JAX_BSSN.bssn.variables import BSSNVariables
    from PyPIC3D.relativity.dynamical_relativity import build_bssn_metric
except ModuleNotFoundError as error:
    if error.name != "JAX_BSSN":
        raise
    BSSNVariables = None


@unittest.skipIf(BSSNVariables is None, "requires the optional JAX BSSN dependency")
class TestEnergyMomentumTensor(unittest.TestCase):
    def assertClose(self, actual, expected):
        tolerance = 300 * jnp.finfo(jnp.asarray(actual).dtype).eps
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)))

    def parameters(self, **kwargs):
        defaults = dict(
            Nx=8, Ny=1, Nz=1, x_wind=8.0, y_wind=1.0, z_wind=1.0,
            x_min=0.0, y_min=0.0, z_min=0.0, guard_cells=3,
            solver="static_metric", particle_pusher="hybrid_boris_geodesic",
            current_deposition="esirkepov", particle_batch_size=2,
        )
        defaults.update(kwargs)
        return kernel_parameters(**defaults)

    def bssn(self, shape=(8, 1, 1), W=1.0, conformal=None):
        scalar = jnp.ones(shape)
        if conformal is None:
            conformal = jnp.eye(3)
        gamma = conformal.reshape((3, 3, 1, 1, 1)) * scalar
        return BSSNVariables(
            gamma, W * scalar, jnp.zeros_like(gamma), jnp.zeros_like(scalar),
            jnp.zeros((3,) + shape), scalar, jnp.zeros((3,) + shape),
        )

    def particles(self, positions, momenta, *, mass=(1.0,), weight=(1.0,), active=None):
        positions = jnp.asarray(positions).reshape(1, 1, 1, len(mass), -1, 3)
        momenta = jnp.asarray(momenta).reshape(positions.shape)
        if active is None:
            active = jnp.ones(positions.shape[:-1], dtype=bool)
        else:
            active = jnp.asarray(active, dtype=bool).reshape(positions.shape[:-1])
        particles = TiledParticles(positions, momenta, active)
        species = SpeciesConfig(
            jnp.zeros(len(mass)), jnp.asarray(mass), jnp.asarray(weight),
            jnp.zeros((len(mass), 3), dtype=bool),
        )
        return particles, species

    def test_gaussian_uniform_electric_magnetic_and_null_wave(self):
        static, dynamic = self.parameters()
        state = self.bssn()
        metric = build_bssn_metric(state, static, dynamic)
        for D, B in (((2., 0., 0.), (0., 0., 0.)),
                     ((0., 0., 0.), (0., 0., 3.)),
                     ((1., 0., 0.), (0., 1., 0.))):
            with self.subTest(D=D, B=B):
                displacement = tuple(jnp.full_like(metric.D[i].lapse, D[i]) for i in range(3))
                magnetic = tuple(jnp.full_like(metric.B[i].lapse, B[i]) for i in range(3))
                moments = electromagnetic_moments(displacement, magnetic, static)
                rho, momentum, stress = compute_electromagnetic_energy_momentum(moments, state)
                d, b = jnp.asarray(D), jnp.asarray(B)
                energy = (jnp.dot(d, d) + jnp.dot(b, b)) / (8 * jnp.pi)
                expected_stress = energy * jnp.eye(3) - (jnp.outer(d, d) + jnp.outer(b, b)) / (4 * jnp.pi)
                self.assertClose(rho, energy)
                self.assertClose(momentum, jnp.cross(d, b)[:, None, None, None] / (4 * jnp.pi))
                self.assertClose(stress, expected_stress[:, :, None, None, None])
                self.assertClose(jnp.einsum("ii...->...", stress), rho)

    def test_alternating_native_field_retains_energy(self):
        static, dynamic = self.parameters()
        state = self.bssn()
        metric = build_bssn_metric(state, static, dynamic)
        template = metric.D[0].lapse
        alternating = jnp.broadcast_to((-1.0) ** jnp.arange(template.shape[3])[None, None, None, :, None, None], template.shape)
        zero = jnp.zeros_like(template)
        moments = electromagnetic_moments((alternating, zero, zero), (zero, zero, zero), static)
        self.assertClose(moments[0][0, 0], 1.0)
        self.assertClose(compute_electromagnetic_energy_momentum(moments, state)[0], 1 / (8 * jnp.pi))

    def test_anisotropic_metric_uses_covariant_momentum_and_stress(self):
        static, dynamic = self.parameters()
        gamma = jnp.diag(jnp.asarray([4.0, 9.0, 16.0]))
        sqrt_gamma = jnp.asarray(24.0)
        W = sqrt_gamma ** (-1 / 3)
        state = self.bssn(W=W, conformal=gamma * W**2)
        metric = build_bssn_metric(state, static, dynamic)
        d, b = jnp.asarray([0.3, -0.2, 0.1]), jnp.asarray([0.0, 0.4, -0.2])
        displacement = tuple(d[i] * metric.D[i].sqrt_gamma for i in range(3))
        magnetic = tuple(b[i] * metric.B[i].sqrt_gamma for i in range(3))
        rho, momentum, stress = compute_electromagnetic_energy_momentum(
            electromagnetic_moments(displacement, magnetic, static), state,
        )
        d_cov, b_cov = gamma @ d, gamma @ b
        energy = (d @ d_cov + b @ b_cov) / (8 * jnp.pi)
        self.assertClose(rho, energy)
        self.assertClose(momentum, (sqrt_gamma * jnp.cross(d, b))[:, None, None, None] / (4 * jnp.pi))
        expected_stress = energy * gamma - (jnp.outer(d_cov, d_cov) + jnp.outer(b_cov, b_cov)) / (4 * jnp.pi)
        self.assertClose(stress, expected_stress[:, :, None, None, None])

    def test_particle_integrals_include_neutral_rest_mass_and_ignore_inactive_nan(self):
        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                static, dynamic = self.parameters(
                    shape_factor=shape_factor, x_wind=4.0, y_wind=0.5, z_wind=0.5,
                )
                state = self.bssn()
                metric = build_bssn_metric(state, static, dynamic)
                particles, species = self.particles(
                    [[0.1, 0., 0.], [1.7, 0., 0.], [jnp.nan] * 3,
                     [3.9, 0., 0.], [2.1, 0., 0.], [jnp.nan] * 3],
                    [[0.3, 0.4, 0.], [-0.2, 0.1, 0.5], [jnp.nan] * 3,
                     [0., 0., 0.], [0.1, -0.3, 0.2], [jnp.nan] * 3],
                    mass=(2.0, 3.0), weight=(0.4, 0.7), active=[True, True, False, True, True, False],
                )
                rho, momentum, stress = compute_particle_energy_momentum(particles, species, metric, static, dynamic)
                u = jnp.where(particles.active[..., None], particles.u, 0.0)[0, 0, 0]
                Gamma = jnp.sqrt(1 + jnp.sum(u * u, axis=-1))
                mw = species.mass[:, None] * species.weight[:, None] * particles.active[0, 0, 0]
                volume = dynamic.dx * dynamic.dy * dynamic.dz
                self.assertClose(jnp.sum(rho) * volume, jnp.sum(mw * Gamma))
                self.assertClose(jnp.sum(momentum, axis=(1, 2, 3)) * volume, jnp.sum(mw[..., None] * u, axis=(0, 1)))
                self.assertClose(jnp.sum(stress, axis=(2, 3, 4)) * volume, jnp.einsum("sp,spi,spj->ij", mw / Gamma, u, u))
                self.assertTrue(bool(jnp.all(jnp.isfinite(rho))))

    def test_particle_proper_volume_is_applied_at_destination_nodes(self):
        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                static, dynamic = self.parameters(shape_factor=shape_factor)
                W = (0.8 + 0.1 * jnp.cos(2 * jnp.pi * jnp.arange(8) / 8))[:, None, None]
                state = self.bssn(W=W)
                metric = build_bssn_metric(state, static, dynamic)
                particles, species = self.particles([[0.2, 0., 0.], [7.9, 0., 0.]], [[0., 0., 0.]] * 2, mass=(2.,), weight=(0.3,))
                rho, momentum, stress = compute_particle_energy_momentum(particles, species, metric, static, dynamic)
                self.assertClose(jnp.sum(rho * W**-3), 1.2)
                self.assertClose(momentum, 0.0)
                self.assertClose(stress, 0.0)

    def test_particle_gamma_changes_at_each_bssn_stage(self):
        static, dynamic = self.parameters()
        u_cov = jnp.asarray([0.5, 0.3, -0.2])
        conformal = jnp.diag(jnp.asarray([4.0, 0.5, 0.5]))
        particles, species = self.particles([[2.2, 0., 0.]], u_cov, mass=(2.,), weight=(0.4,))
        moments = tuple(jnp.zeros((3, 3, 8, 1, 1)) for _ in range(3))
        for W in (1.0, 0.5):
            with self.subTest(W=W):
                state = self.bssn(W=W, conformal=conformal)
                rho, momentum, stress = compute_energy_momentum_tensor(particles, species, moments, state, static, dynamic)
                Gamma = jnp.sqrt(1 + W**2 * jnp.sum(u_cov**2 / jnp.diag(conformal)))
                self.assertClose(jnp.sum(rho) * W**-3, 0.8 * Gamma)
                self.assertClose(jnp.sum(momentum, axis=(1, 2, 3)) * W**-3, 0.8 * u_cov)
                self.assertClose(jnp.sum(stress, axis=(2, 3, 4)) * W**-3, 0.8 * jnp.outer(u_cov, u_cov) / Gamma)

    def test_no_active_particles_returns_finite_zero_sources(self):
        static, dynamic = self.parameters()
        state = self.bssn()
        metric = build_bssn_metric(state, static, dynamic)
        particles, species = self.particles([[jnp.nan] * 3], [[jnp.nan] * 3], active=[False])
        sources = compute_particle_energy_momentum(particles, species, metric, static, dynamic)
        for value in sources:
            self.assertClose(value, 0.0)
        empty = TiledParticles(jnp.zeros((1, 1, 1, 0, 0, 3)), jnp.zeros((1, 1, 1, 0, 0, 3)), jnp.zeros((1, 1, 1, 0, 0), dtype=bool))
        sources = compute_particle_energy_momentum(empty, species, metric, static, dynamic)
        for value in sources:
            self.assertClose(value, 0.0)

    @unittest.skipIf(len(jax.devices()) < 2, "requires two CPU devices for tile exchange")
    def test_periodic_two_tile_sources_equal_one_tile(self):
        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                single, single_dynamic = self.parameters(shape_factor=shape_factor)
                tiled, tiled_dynamic = self.parameters(shape_factor=shape_factor, tile_shape=(4, 1, 1))
                state = self.bssn()
                single_metric = build_bssn_metric(state, single, single_dynamic)
                tiled_metric = build_bssn_metric(state, tiled, tiled_dynamic)
                particles, species = self.particles(
                    [[0.1, 0., 0.], [3.9, 0., 0.], [4.1, 0., 0.], [7.9, 0., 0.]],
                    [[0.2, 0.1, -0.3]] * 4, mass=(2.,), weight=(0.4,),
                )
                tiled_particles = TiledParticles(
                    particles.x.reshape(2, 1, 1, 1, 2, 3),
                    particles.u.reshape(2, 1, 1, 1, 2, 3),
                    particles.active.reshape(2, 1, 1, 1, 2),
                )
                expected = compute_particle_energy_momentum(particles, species, single_metric, single, single_dynamic)
                actual = compute_particle_energy_momentum(tiled_particles, species, tiled_metric, tiled, tiled_dynamic)
                for value, reference in zip(actual, expected):
                    self.assertClose(jax.device_put(value, reference.sharding), reference)


if __name__ == "__main__":
    unittest.main()
