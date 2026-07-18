import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.parameters import (
    DynamicParameters,
    GridParameters,
    StaticParameters,
    build_dynamic_parameters,
    build_static_parameters,
)


class TestKernelParameters(unittest.TestCase):
    def test_static_and_dynamic_parameters_split_kernel_contract(self):
        world = {
            "dt": 0.1,
            "dx": 0.25,
            "dy": 0.5,
            "dz": 1.0,
            "Nx": 4,
            "Ny": 2,
            "Nz": 1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "guard_cells": 2,
            "tile_shape": (4, 2, 1),
            "current_deposition": "direct",
            "current_filter": "none",
            "particle_species_names": ("electrons",),
            "particle_species_metadata": ({"name": "electrons"},),
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 1, "z": 2},
            "grids": {
                "vertex": (jnp.arange(6.0), jnp.arange(4.0), jnp.arange(3.0)),
                "center": (jnp.arange(6.0), jnp.arange(4.0), jnp.arange(3.0)),
                "tiled_vertex_grid": (jnp.zeros((1, 1, 1, 8, 6, 5)),) * 3,
                "tiled_center_grid": (jnp.zeros((1, 1, 1, 8, 6, 5)),) * 3,
            },
            "field_mesh": object(),
        }
        constants = {"C": 1.0, "eps": 2.0, "mu": 3.0, "kb": 4.0, "alpha": 0.5}

        static_parameters = build_static_parameters({
            **world,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(world, constants)

        self.assertIsInstance(static_parameters, StaticParameters)
        self.assertEqual(static_parameters.current_deposition, "direct")
        self.assertEqual(static_parameters.current_filter, "none")
        self.assertEqual(static_parameters.particle_pusher, "boris")
        self.assertEqual(static_parameters.tile_shape, (4, 2, 1))
        self.assertEqual(static_parameters.boundary_conditions, (0, 0, 0))
        self.assertEqual(static_parameters.particle_boundary_conditions, (0, 1, 2))
        self.assertIs(static_parameters.field_mesh, world["field_mesh"])
        self.assertNotIn("particle_species_names", static_parameters._asdict())
        self.assertNotIn("particle_species_metadata", static_parameters._asdict())
        self.assertIsInstance(hash(static_parameters), int)
        with self.assertRaises(TypeError):
            static_parameters["current_deposition"]

        self.assertIsInstance(dynamic_parameters, DynamicParameters)
        self.assertIsInstance(dynamic_parameters.grids, GridParameters)
        self.assertNotIn("current_deposition", dynamic_parameters._asdict())
        self.assertNotIn("current_filter", dynamic_parameters._asdict())
        self.assertNotIn("field_mesh", dynamic_parameters._asdict())
        self.assertAlmostEqual(float(dynamic_parameters.dt), 0.1)
        self.assertAlmostEqual(float(dynamic_parameters.C), 1.0)
        self.assertIs(dynamic_parameters.grids.tiled_center_grid, world["grids"]["tiled_center_grid"])
        with self.assertRaises(TypeError):
            dynamic_parameters["dt"]

        flattened, _ = jax.tree_util.tree_flatten(dynamic_parameters)
        self.assertTrue(all(hasattr(leaf, "shape") for leaf in flattened))

    def test_public_deposition_methods_are_jitted_static_parameter_boundaries(self):
        self.assertTrue(hasattr(J_from_rhov, "lower"))
        self.assertTrue(hasattr(Esirkepov_current, "lower"))
        self.assertTrue(hasattr(compute_rho, "lower"))


if __name__ == "__main__":
    unittest.main()
