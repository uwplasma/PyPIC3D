"""Hermite reconstruction of the supplied grid metric at particle positions."""

import unittest
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import coordinate_velocity, hybrid_boris_geodesic_push
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.relativity import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
)
from PyPIC3D.relativity.field_state import densitize_vector
from PyPIC3D.relativity.interpolate_metric import (
    interpolate_hermite,
    interpolate_metric,
    particle_metric_valid,
    positive_definite_3x3,
)
from PyPIC3D.utilities.parameters import build_static_parameters, static_parameters_for_output
from tests.kernel_fixtures import kernel_parameters
from tests.support.particle_metric_fixtures import (
    make_runtime,
    manufactured,
    sample_metric,
    sampled_rotation_checks,
    consumer_runtime,
)


class HermiteFixtures:
    pass


class ParticleMetricFixtures:
    pass


class ParticleMetricConsumersFixtures:
    def tearDown(self):
        jax.clear_caches()

    def assert_same_tree(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-14)


class CheckedSamplingFixtures:
    pass


