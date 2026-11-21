"""Physical moment and staggering checks for the BSSN matter source."""

import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_particle_energy_momentum,
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
