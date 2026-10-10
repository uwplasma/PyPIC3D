"""Fixed-resolution layout equivalence for the curved-metric field solver."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.diagnostics.output_adapters import scalar_field_for_output
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_state import densitize_vector
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E, compute_covariant_H, update_B, update_D,
)
from tests.kernel_fixtures import vector_tiles_from_global
from tests.support.pec_projector_fixtures import coupled_setup


@partial(jax.jit, static_argnames="static")
def _field_step(D, B, J, metric, static, dynamic):
    D = refresh_fields(densitize_vector(D, metric.D), static, D_FIELD_LOCATIONS, "D", metric)
    B = refresh_fields(densitize_vector(B, metric.B), static, B_FIELD_LOCATIONS, "B", metric)
    J = densitize_vector(J, metric.D)
    B = update_B(compute_covariant_E(D, B, metric), B, metric, static, dynamic, .003)
    D = update_D(D, compute_covariant_H(D, B, metric), J, metric, static, dynamic, .003)
    return D, B


class TestStaticMetricFieldLayout(unittest.TestCase):
    def test_curved_metric_field_step_matches_one_tile(self):
        # One spatially varying, non-diagonal metric and one resolution suffice
        # here; numerical/convergence matrices belong to the ordinary phase.
        rng = np.random.default_rng(812)
        fields = tuple(tuple(jnp.asarray(rng.normal(size=(10, 8, 6))) for _ in range(3))
                       for _ in range(3))
        results = []
        for tile_shape in ((8, 6, 4), (4, 6, 4)):
            static, dynamic, metric = coupled_setup((8, 6, 4), tile_shape, (0, 0, 0))
            tiled = tuple(vector_tiles_from_global(field, static, dynamic) for field in fields)
            result = _field_step(*tiled, metric, static, dynamic)
            results.append(tuple(np.asarray(scalar_field_for_output(value, static))
                                 for vector in result for value in vector))
        for single, distributed in zip(*results):
            np.testing.assert_allclose(distributed, single, rtol=1.e-12, atol=1.e-12)


if __name__ == "__main__":
    unittest.main()
