"""Pair birth, capacity, and compiled scheduling on the production GR path."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.relativity.pair_injection import insert_pairs, birth_covariant_momentum
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update('jax_enable_x64', True)


def setup(tile_shape=(4, 4, 4), slots=6, cells=(4, 4, 4)):
    static, dynamic = kernel_parameters(
        Nx=cells[0], Ny=cells[1], Nz=cells[2], tile_shape=tile_shape, dt=.01,
        solver='static_metric', current_deposition='esirkepov',
        particle_pusher='hybrid_boris_geodesic')
    metric = initialize_flat_cartesian_metric(static, dynamic)
    zero = empty_tiled_vector(static, dynamic)
    fields = (zero, zero, zero, zero[0], zero[0], (zero, zero), metric,
              (zero, zero), jnp.asarray(False))
    # Each candidate lies strictly inside its own tile, including y and z tiles.
    center = jnp.stack([axis[..., static.guard_cells] + (axis[..., 1]-axis[..., 0])*.4
                        for axis in dynamic.grids.tiled_center_grid], axis=-1)
    shape = center.shape[:3] + (3, slots)
    x = jnp.broadcast_to(center[..., None, None, :], shape+(3,))
    particles = TiledParticles(x, jnp.zeros_like(x), jnp.zeros(shape, bool))
    species = SpeciesConfig(jnp.array([-1., 2., 1.]), jnp.ones(3),
                            jnp.ones(3), jnp.ones((3, 3), bool))
    positions = jnp.broadcast_to(center[..., None, :], center.shape[:3]+(3, 3))
    momenta = jnp.broadcast_to(jnp.array([.1, .2, -.1]), positions.shape)
    return static, dynamic, particles, species, fields, positions, momenta


def source(positions, momenta):
    def inject(particles, species, fields, static, dynamic, key, event):
        next_key, sample_key = jax.random.split(key)
        sample_key = jax.random.fold_in(sample_key, event)
        # A random momentum makes event/key progression observable.
        u = momenta * jax.random.uniform(sample_key, (), minval=.5, maxval=1.)
        requested = jnp.ones(particles.active.shape[:3], jnp.int32)
        result, counts = insert_pairs(
            particles, species, positions, u, requested, fields[0], fields[1],
            fields[6], static, dynamic, species_indices=(0, 2))
        return result, next_key, counts
    return inject


