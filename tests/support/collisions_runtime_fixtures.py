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


