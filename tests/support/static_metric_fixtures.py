import unittest
from unittest.mock import patch
import importlib
import itertools

from PyPIC3D.relativity.field_state import densitize_fields, densitize_vector, physical_vector

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import BC_TYPE_PARTICLE
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from PyPIC3D.initialization import (
    _encode_current_calculation,
    _validate_tiled_yee_configuration,
    validate_field_solver,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
import PyPIC3D.pusher.hybrid_boris_geodesic as hybrid_pusher
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    coordinate_velocity,
    geodesic_acceleration,
    hybrid_boris_geodesic_push,
    magnetic_boris_rotation,
)
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
    location_grid,
)
from PyPIC3D.relativity.interpolate_metric import ParticleMetric, interpolate_metric
from PyPIC3D.relativity.metrics.flat import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
)
from PyPIC3D.relativity.metrics.kerr_schild import (
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
)
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E,
    compute_covariant_H,
    update_D,
)
from PyPIC3D.utilities.filters import tiled_bilinear_filter_vector, tiled_digital_filter_vector
from tests.kernel_fixtures import active_interior, empty_tiled_vector, kernel_parameters


def _single_particle_state(static_parameters, dynamic_parameters, u):
    x = jnp.zeros((1, 1, 1, 1, 1, 3))
    u = jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3))
    active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
    particles = TiledParticles(x=x, u=u, active=active)
    species = SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )
    return particles, species


def _constant_tiled_vector(static_parameters, dynamic_parameters, values):
    vector = empty_tiled_vector(static_parameters, dynamic_parameters)
    return tuple(vector[i].at[:, :, :, :, :, :].set(values[i]) for i in range(3))


def _replace_lapse_shift(metric, lapse, shift):
    def replace_one(metric_at_location):
        shift_array = jnp.zeros_like(metric_at_location.shift)
        for i, value in enumerate(shift):
            shift_array = shift_array.at[..., i].set(value)
        return metric_at_location._replace(
            lapse=jnp.full_like(metric_at_location.lapse, lapse),
            shift=shift_array,
        )

    return metric._replace(
        D=tuple(replace_one(metric_at_location) for metric_at_location in metric.D),
        B=tuple(replace_one(metric_at_location) for metric_at_location in metric.B),
        center=replace_one(metric.center),
        vertex=replace_one(metric.vertex),
    )


def _metric_locations_with_grids(metric, dynamic_parameters):
    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    metric_locations = (
        tuple(zip(metric.D, D_FIELD_LOCATIONS))
        + tuple(zip(metric.B, B_FIELD_LOCATIONS))
        + ((metric.center, ("C", "C", "C")),)
        + ((metric.vertex, ("V", "V", "V")),)
    )

    return tuple(
        (
            metric_at_location,
            location_grid(center_grid, vertex_grid, location),
        )
        for metric_at_location, location in metric_locations
    )


class StaticMetricTestCase:
    def assertAllClose(self, actual, expected, **kwargs):
        self.assertTrue(bool(jnp.allclose(jnp.asarray(actual), jnp.asarray(expected), **kwargs)))


class StaticMetricInitializationFixtures(StaticMetricTestCase):
    pass


class ConstitutiveFieldsFixtures(StaticMetricTestCase):
    pass


class HybridPusherFixtures(StaticMetricTestCase):
    pass


class GRDirectDepositionFixtures(StaticMetricTestCase):
    pass


class StaticMetricTimeLoopFixtures(StaticMetricTestCase):
    pass


class StaticMetricDispatchFixtures(StaticMetricTestCase):
    pass
