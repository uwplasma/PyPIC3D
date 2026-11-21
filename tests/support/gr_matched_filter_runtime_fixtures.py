"""Periodic GR particle--grid filtering and its discrete conservation contract."""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence, electromagnetic_constraints
from PyPIC3D.initialization import _validate_current_filter_contract, initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.core import build_yee_metric
from PyPIC3D.relativity.filtering import filter_gr_grid, particle_metric, validate_gr_filter
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.metrics.flat import initialize_flat_cartesian_metric
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import empty_tiled_scalar, empty_tiled_vector, kernel_parameters


def parameters(choice="bilinear", **kwargs):
    settings = dict(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., dt=.05, guard_cells=3,
        solver="static_metric", metric="numerical", current_deposition="esirkepov",
        particle_pusher="hybrid_boris_geodesic", current_filter=choice,
        alpha=.6, particle_batch_size=4,
    )
    settings.update(kwargs)
    return kernel_parameters(**settings)


def varying_metric(position):
    phase = 2 * jnp.pi * position[0] / 8.
    lapse = 1. + .08 * jnp.cos(phase)
    shift = jnp.array([.03 * jnp.sin(phase), .02 * jnp.cos(phase), 0.])
    gamma = jnp.array([
        [1.4 + .2 * jnp.cos(phase), .04 * jnp.sin(phase), 0.],
        [.04 * jnp.sin(phase), 1.1 + .1 * jnp.sin(phase), 0.],
        [0., 0., 1.2 + .15 * jnp.cos(phase)],
    ])
    return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))


def particles_at(position, tile_counts=(1, 1, 1)):
    x = jnp.broadcast_to(jnp.asarray(position, dtype=jnp.float64), tile_counts + (1, 1, 3))
    active = jnp.zeros(tile_counts + (1, 1), bool).at[0, 0, 0].set(True)
    return TiledParticles(x, jnp.zeros_like(x), active)


def species():
    return SpeciesConfig(jnp.array([1.]), jnp.array([1.]), jnp.array([.7]),
                         jnp.ones((1, 3), bool))


