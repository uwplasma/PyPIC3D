"""Proca PEC images, PML state contracts, and public runtime integration.

Run with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=4.
"""
import contextlib
import io
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import openpmd_api as pmd

from PyPIC3D.__main__ import run_PyPIC3D
from PyPIC3D.boundary_conditions.PML import load_pml_from_toml
from PyPIC3D.diagnostics.diagnostic_quantities import compute_dark_energy
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, scalar_field_for_output
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.solvers.dark_matter_yee.boundaries import dark_field_boundaries
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    initialize_dark_photon_fields, synchronized_dark_fields, dark_divergence,
)
from PyPIC3D.solvers.dark_matter_yee.pml import initialize_dark_pml, advance_dark_pml, stretch_dark_derivatives
from PyPIC3D.solvers.dark_matter_yee.time_loop import time_loop_dark_photon
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.utilities.parameters import build_static_parameters
from tests.kernel_fixtures import kernel_parameters
from tests.support.dark_photon_fixtures import evolve, interior
from tests.support import dark_photon_fixtures as periodic_tests


def wall_mode(s, d, location, parity):
    """A separable mode with known PEC images on every guard plane."""
    value = 1.
    for axis, (length, lower) in enumerate(zip((d.x_wind, d.y_wind, d.z_wind), (-d.x_wind/2, -d.y_wind/2, -d.z_wind/2))):
        grid = d.grids.tiled_center_grid if location[axis] == "C" else d.grids.tiled_vertex_grid
        phase = np.pi * (grid[axis] - lower) / length
        if s.boundary_conditions[axis] == 0:
            factor = jnp.cos(2 * phase)
        else:
            factor = jnp.sin(phase) if parity[axis] == -1 else jnp.cos(phase)
        shape = list(factor.shape[:3]) + [1, 1, 1]
        shape[axis + 3] = factor.shape[-1]
        value = value * factor.reshape(shape)
    return value


def pml_evolve(fields, state, s, d, steps):
    J = tuple(jnp.zeros_like(v) for v in fields[0])
    return jax.jit(lambda f, p: jax.lax.fori_loop(
        0, steps, lambda _, pair: advance_dark_pml(pair[0], J, pair[1], s, d), (f, p),
    ))(fields, state)


class DarkBoundariesFixtures:
    def assert_tree_close(self, a, b, atol=2e-12):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=atol)
