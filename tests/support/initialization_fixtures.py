import unittest
import contextlib
import io
import tempfile
import os
from unittest.mock import patch

import numpy as np
import toml
from PyPIC3D.relativity.field_state import physical_fields

import jax
import jax.numpy as jnp
from PyPIC3D.initialization import (
    _available_cpu_threads,
    _encode_field_bc,
    _encode_particle_bc,
    _resolve_particle_batch_size,
    default_parameters,
    initialize_simulation,
    setup_write_dir,
    validate_field_solver,
)
from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
)
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.utilities.parameters import build_static_parameters
from PyPIC3D.utilities.toml_helpers import update_parameters_from_toml
from tests.kernel_fixtures import kernel_parameters


class InitializationFunctionsFixtures:


    def setUp(self):
        self.plotting_parameters, self.simulation_parameters, self.dynamic_values = default_parameters()
        self.simulation_parameters['output_dir'] = 'test_output'
        # check the  default parameters are set correctly


        # check that the output directory is created


        # check that the default parameters contain expected keys

    @staticmethod
    def _particles_with_active(active):
        active = jnp.asarray(active, dtype=bool)
        state_shape = active.shape + (3,)
        return TiledParticles(
            x=jnp.zeros(state_shape),
            u=jnp.zeros(state_shape),
            active=active,
        )
