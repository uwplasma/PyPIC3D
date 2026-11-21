"""Physical moment and staggering checks for the BSSN matter source."""

import unittest

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

    def test_disabled_backreaction_hands_bssn_exact_zeros(self):
        """The flag gates what BSSN integrates, not what the leaves measure."""
        static, dynamic = self.parameters()
        u_cov = jnp.asarray([0.5, 0.3, -0.2])
        particles, species = self.particles([[2.2, 0., 0.]], u_cov, mass=(2.,), weight=(0.4,))
        state = self.bssn()
        template = build_bssn_metric(state, static, dynamic).D[0].lapse
        alternating = jnp.broadcast_to(
            (-1.0) ** jnp.arange(template.shape[3])[None, None, None, :, None, None],
            template.shape)
        zero = jnp.zeros_like(template)
        moments = electromagnetic_moments((alternating, zero, zero), (zero, zero, zero), static)

        coupled = compute_energy_momentum_tensor(
            particles, species, moments, state, static, dynamic)
        self.assertGreater(float(jnp.max(jnp.abs(coupled[0]))), 0.)

        vacuum = compute_energy_momentum_tensor(
            particles, species, moments, state,
            static._replace(metric_backreaction=False), dynamic)
        for value, reference in zip(vacuum, coupled):
            self.assertEqual(value.shape, reference.shape)
            self.assertEqual(float(jnp.max(jnp.abs(value))), 0.)

        # The leaf functions stay physical so diagnostics keep measuring energy.
        self.assertClose(compute_electromagnetic_energy_momentum(moments, state)[0],
                         coupled[0] - compute_particle_energy_momentum(
                             particles, species, build_bssn_metric(state, static, dynamic),
                             static, dynamic)[0])

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


if __name__ == "__main__":
    unittest.main()
