"""Shared-device BSSN/PIC transfers and coupled time-centering regressions.

Run with JAX_NUM_CPU_DEVICES=2 on a CPU host.
"""

import unittest
from unittest.mock import patch
import importlib

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNVariables
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_tile_layout, bssn_to_tiles, tiles_to_bssn, build_bssn_metric,
    initialize_dynamical_relativity,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    electromagnetic_moments, compute_energy_momentum_tensor,
)
from PyPIC3D.diagnostics.gr_work_energy_theorem import sample_energy, volume_integral, surface_integral
from PyPIC3D.diagnostics.static_metric import bssn_constraint_norms
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from tests.kernel_fixtures import kernel_parameters

jax.config.update("jax_enable_x64", True)


def coupled_example(tiles, *, particles=False, current_filter="none", crossing=False):
    static, dynamic = kernel_parameters(
        Nx=16, Ny=1, Nz=1, x_wind=16., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., tile_shape=(16 // tiles, 1, 1),
        dt=.002, guard_cells=3, solver="dynamic_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        particle_batch_size=4, current_filter=current_filter)
    static = static._replace(metric_backreaction=True)
    phase = 2 * jnp.pi * jnp.arange(16)[:, None, None] / 16
    zero = jnp.zeros_like(phase)
    identity = jnp.eye(3)[:, :, None, None, None] * jnp.ones_like(phase)
    vector = jnp.zeros((3,) + phase.shape)
    state = BSSNVariables(identity, 1. + .003 * jnp.cos(phase), identity * 0,
                          .001 * jnp.sin(phase), vector, 1. + .002 * jnp.sin(phase),
                          vector.at[0].set(.01 * jnp.cos(phase)))
    static, bssn = build_bssn_parameters(static, dynamic, state=state,
                                         evolution_parameters={"nu": .002, "kappa": .03})
    dynamic = dynamic._replace(bssn_parameters=bssn)
    D = tuple(bssn_to_tiles(value, static) for value in
              (.03 + .002 * jnp.sin(phase), .01 * jnp.cos(phase), zero))
    B = tuple(bssn_to_tiles(value, static) for value in
              (zero, .02 + .001 * jnp.sin(phase), .01 * jnp.cos(phase)))
    zeros = tuple(jnp.zeros_like(value) for value in D)
    if particles:
        position_values = [.2, 7.99999, 8.00001, 15.8] if crossing else [.2, 7.9, 8.1, 15.8]
        positions = jnp.array(position_values).reshape(tiles, 1, 1, 1, 4 // tiles)
        x = jnp.zeros(positions.shape + (3,)).at[..., 0].set(positions)
        u = jnp.broadcast_to(jnp.array([.1, -.03, .02]), x.shape)
        if crossing:
            u = u.at[..., 0].set(jnp.array([.1, .2, -.2, -.1]).reshape(positions.shape))
        active = jnp.ones(positions.shape, dtype=bool)
        species = SpeciesConfig(jnp.array([.3]), jnp.array([1.]), jnp.array([.01]),
                                jnp.ones((1, 3), dtype=bool))
    else:
        x = jnp.zeros((tiles, 1, 1, 0, 0, 3))
        u, active = x, jnp.zeros(x.shape[:-1], dtype=bool)
        empty = jnp.zeros((0,))
        species = SpeciesConfig(empty, empty, empty, jnp.zeros((0, 3), dtype=bool))
    particle_state = shard_tiled_particles(TiledParticles(x, u, active), static)
    fields = (D, B, zeros, zeros[0], zeros[0], (zeros, zeros), None, (D, B), jnp.asarray(False))
    particle_state, fields, dynamic = initialize_dynamical_relativity(
        particle_state, species, fields, static, dynamic)
    return static, dynamic, particle_state, species, fields


def gather_bssn(value, static):
    return tiles_to_bssn(value, static._replace(guard_cells=bssn_tile_layout(static).guard_cells))


