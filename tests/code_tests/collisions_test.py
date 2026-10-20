import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.collisions import apply_collisions, load_collisions_from_toml
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles

jax.config.update("jax_enable_x64", True)
DYN = SimpleNamespace(dx=1.0, dy=1.0, dz=1.0, dt=1.0e-9, eps=8.854e-12, kb=1.380649e-23)


def make_particles(n=401, species=2, cells=4, scale=1.0e5, seed=0):
    kx, ku = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.uniform(kx, (1, 1, 1, species, n, 3), maxval=cells)
    u = scale * jax.random.normal(ku, (1, 1, 1, species, n, 3))
    active = jnp.ones((1, 1, 1, species, n), bool).at[..., -7:].set(False)
    return TiledParticles(x=x, u=u, active=active, key=jax.random.PRNGKey(seed + 1))


def make_config(mass=(9.1e-31, 3.0e-30), charge=(-1.6e-19, 1.6e-19)):
    n = len(mass)
    return SpeciesConfig(charge=jnp.asarray(charge), mass=jnp.asarray(mass), weight=jnp.full(n, 1.0e10),
                         update_x=jnp.ones((n, 3), bool))


def run(specs, particles, config, steps=1):
    step = jax.jit(lambda p: apply_collisions(p, config, SimpleNamespace(collisions=specs), DYN))
    for _ in range(steps):
        particles, overflow = step(particles)
    return particles, overflow


class TestCollisions(unittest.TestCase):
    def moments(self, particles, config):
        m = config.mass[None, None, None, :, None, None] * particles.active[..., None]
        return jnp.sum(m * particles.u, axis=(0, 1, 2, 3, 4)), jnp.sum(m * particles.u**2)

    def test_binary_conserves_momentum_and_energy(self):
        config = make_config()
        particles = make_particles()
        specs = (("binary", 0, 0, 10.0), ("binary", 0, 1, 10.0), ("binary", 1, 1, 10.0))
        after, _ = run(specs, particles, config, steps=3)
        p0, e0 = self.moments(particles, config)
        p1, e1 = self.moments(after, config)
        self.assertTrue(jnp.allclose(p0, p1, rtol=1e-10, atol=1e-30 + 1e-10 * float(jnp.abs(p0).max())))
        self.assertAlmostEqual(float(e1 / e0), 1.0, places=10)
        self.assertFalse(jnp.allclose(after.u, particles.u))
        self.assertTrue(jnp.array_equal(after.u[..., -7:, :], particles.u[..., -7:, :]))

    def test_mcc_heavy_elastic_preserves_speed_and_charge_exchange_takes_neutral(self):
        config = make_config()
        particles = make_particles(species=1)
        elastic = (("mcc", 0, 1.0e30, 0.0, 1.0e10, (("elastic", 0.0, (0.0, 1e30), (1e-19, 1e-19), -1, -1),)),)
        after, _ = run(elastic, particles, make_config(mass=(9.1e-31,), charge=(-1.6e-19,)))
        speed = lambda p: jnp.linalg.norm(p.u, axis=-1)
        self.assertTrue(jnp.allclose(speed(after), speed(particles), rtol=1e-12))
        self.assertFalse(jnp.allclose(after.u, particles.u))
        cx = (("mcc", 0, 1.0e30, 0.0, 1.0e-26, (("charge_exchange", 0.0, (0.0, 1e30), (1e-19, 1e-19), -1, -1),)),)
        after, _ = run(cx, particles, make_config(mass=(1e-26,), charge=(1.6e-19,)))
        self.assertTrue(jnp.allclose(after.u[after.active], 0.0))

    def test_ionization_creates_pairs_in_free_slots(self):
        config = make_config()
        particles = make_particles(scale=1.0e7)
        particles = particles._replace(active=particles.active.at[0, 0, 0, 1].set(False))
        ion = (("mcc", 0, 1.0e30, 0.0, 6.6e-26, (("ionization", 15.0, (0.0, 15.0, 1e9), (0.0, 1e-19, 1e-19), 0, 1),)),)
        after, overflow = run(ion, particles, config)
        new_e = int(after.active[..., 0, :].sum() - particles.active[..., 0, :].sum())
        self.assertEqual(int(after.active[..., 1, :].sum()), int(particles.active[..., 0, :].sum()))
        self.assertEqual(new_e, 7)
        # every electron ionizes; ions fit, but only 7 free electron slots exist
        self.assertTrue(bool(overflow))

    def test_toml_loader(self):
        config = {"collisions": {"ee": {"type": "binary", "species": ["e", "e"], "coulomb_log": 12.0},
                                 "en": {"type": "background_mcc", "species": "e", "background_density": 1e20,
                                        "background_mass": 6.6e-26, "elastic": {"cross_section": 1e-19},
                                        "ionization": {"cross_section": [[15.0, 0.0], [100.0, 1e-20]],
                                                       "energy": 15.0, "product_species": ["e", "i"]}}}}
        specs = load_collisions_from_toml(config, ("e", "i"))
        self.assertEqual(specs[0], ("binary", 0, 0, 12.0))
        self.assertEqual([p[0] for p in specs[1][5]], ["elastic", "ionization"])
        self.assertEqual(specs[1][5][1][4:], (0, 1))


if __name__ == "__main__":
    unittest.main()
