"""Continuum GR energy budgets on synchronized native Maxwell/PIC states."""

import unittest

import jax
import jax.numpy as jnp
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT, BC_CONDUCTING
from PyPIC3D.diagnostics.gr_work_energy_theorem import (
    EnergySample,
    advance_tracker,
    tracker_values,
    initialize_tracker,
    start_tracker,
    sample_energy,
    surface_integral,
    volume_integral,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.utilities.parameters import build_static_parameters, dynamic_parameters_for_output
from PyPIC3D.utilities.toml_helpers import update_parameters_from_toml
from tests.kernel_fixtures import kernel_parameters

try:
    from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
    from PyPIC3D.relativity.dynamical_relativity import (
        build_bssn_metric,
        initialize_dynamical_relativity,
        tiles_to_bssn,
    )
except ModuleNotFoundError as error:
    if error.name != "JAX_BSSN":
        raise
    build_bssn_parameters = None


jax.config.update("jax_enable_x64", True)


def parameters(**kwargs):
    values = dict(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., tile_shape=(8, 1, 1),
        guard_cells=3, solver="dynamic_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="none", particle_batch_size=4,
    )
    values.update(kwargs)
    return kernel_parameters(**values)


def native_runtime(static, dynamic, *, gamma=None, curvature=None):
    """Initialize genuine native histories from common-time uniform EM data."""
    static, bssn = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    state = bssn.state
    if gamma is not None:
        W = jnp.linalg.det(gamma) ** (-1. / 6.)
        conformal = gamma * W**2
        state = state._replace(
            conformal_factor=jnp.full_like(state.conformal_factor, W),
            conformal_metric=conformal[:, :, None, None, None]
            * jnp.ones_like(state.conformal_factor),
        )
    if curvature is not None:
        trace = jnp.einsum("ij,ij->", jnp.linalg.inv(gamma), curvature)
        traceless = W**2 * (curvature - gamma * trace / 3.)
        state = state._replace(
            trace_K=jnp.full_like(state.trace_K, trace),
            traceless_K=traceless[:, :, None, None, None]
            * jnp.ones_like(state.conformal_factor),
        )
    bssn = bssn._replace(state=state)
    dynamic = dynamic._replace(bssn_parameters=bssn)
    metric = build_bssn_metric(state, static, dynamic)
    electric = tuple(jnp.full_like(metric.D[i].lapse, (.5, -.3, .2)[i]) for i in range(3))
    magnetic = tuple(jnp.full_like(metric.B[i].lapse, (.1, .4, -.2)[i]) for i in range(3))
    zero = tuple(jnp.zeros_like(value) for value in electric)
    counts = metric.center.lapse.shape[:3]
    x = jnp.zeros(counts + (0, 0, 3))
    particles = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
    vector = jnp.zeros((0,))
    species = SpeciesConfig(vector, vector, vector, jnp.zeros((0, 3), dtype=bool))
    fields = (electric, magnetic, zero, zero[0], zero[0], (zero, zero), None,
              (electric, magnetic), jnp.asarray(False))
    particles, fields, dynamic = initialize_dynamical_relativity(
        particles, species, fields, static, dynamic)
    return particles, species, fields, static, dynamic


class WorkEnergyAssertions(unittest.TestCase):
    def assertClose(self, actual, expected, *, rtol=2.e-12, atol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol)),
                        f"actual={actual}, expected={expected}")


