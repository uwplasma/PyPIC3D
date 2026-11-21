"""Geometry transfer and leapfrog initialization for native BSSN coupling."""

import unittest
import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed") from error
    raise

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_to_tiles, tiles_to_bssn, build_bssn_metric,
    initialize_dynamical_relativity, synchronize_particles, rephase_pic_state,
)
from tests.kernel_fixtures import kernel_parameters


def runtime(boundaries=(0, 0, 0)):
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=2., y_wind=.25, z_wind=.25,
        x_min=0., y_min=0., z_min=0., dt=.01, guard_cells=3,
        solver="dynamic_metric", metric="numerical", current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic", boundary_conditions=boundaries)
    static, parameters = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    dynamic = dynamic._replace(bssn_parameters=parameters)
    scalar = jnp.zeros((1, 1, 1, 14, 7, 7))
    zero = (scalar,) * 3
    electric = (scalar + .04, scalar, scalar)
    fields = (electric, zero, zero, scalar, scalar, (zero, zero), None, (zero, zero), jnp.asarray(False))
    position = jnp.array([.7, 0., 0.]).reshape(1, 1, 1, 1, 1, 3)
    momentum = jnp.array([.2, .05, 0.]).reshape(position.shape)
    particles = TiledParticles(position, momentum, jnp.ones(position.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.array([.4]), jnp.array([1.]), jnp.array([.01]), jnp.ones((1, 3), dtype=bool))
    return static, dynamic, particles, species, fields


