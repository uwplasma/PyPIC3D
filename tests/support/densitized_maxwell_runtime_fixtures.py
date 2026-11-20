"""Regression contracts for persistent Maxwell densities and physical adapters."""
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.ghost_cells import owned_nodes
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, field_map_for_output
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, build_yee_metric
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from PyPIC3D.relativity.field_state import (
    densitize_vector, physical_vector, densitize_fields, physical_fields,
)
from PyPIC3D.solvers.GR_yee.curved_maxwell import compute_covariant_E, compute_covariant_H, update_D, update_B
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.support.pec_projector_fixtures import coupled_setup


def random_vectors(metric):
    rng = np.random.default_rng(2026)
    return [tuple(jnp.asarray(rng.normal(size=metric.center.lapse.shape))
                  for _ in range(3)) for _ in range(3)]


def curl(vector, dynamic, forward):
    """Independent cyclic finite-difference reference."""
    spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
    def derivative(value, axis):
        shift = -1 if forward else 1
        return (jnp.roll(value, shift, axis+3)-value)*(1 if forward else -1)/spacing[axis]
    return tuple(derivative(vector[(i+2)%3], (i+1)%3)
                 - derivative(vector[(i+1)%3], (i+2)%3) for i in range(3))


def signed_volume_metric(position):
    """Identity gamma with a volume that changes sign along periodic y but never vanishes on a node."""
    volume = (1.+.5*position[0])*jnp.sin(2*jnp.pi*(position[1]+.1))
    return 1., jnp.zeros(3), jnp.eye(3), jnp.eye(3), volume
