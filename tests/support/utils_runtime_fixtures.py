import unittest
import os
import tempfile
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import toml
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.kernel_fixtures import (
    build_tiled_particles,
    particle_parameters_from_tile_values,
    particle_species,
)
from PyPIC3D.diagnostics import plotting
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.utilities.plasma_quantities import (
    T_to_vth,
    build_plasma_parameters_dict,
    check_stability,
    vth_to_T,
)
from PyPIC3D.utilities.simulation_helpers import (
    courant_condition,
    particle_sanity_check,
)
from PyPIC3D.utilities.toml_helpers import (
    dump_parameters_to_toml,
    load_external_fields_from_toml,
)
from tests.kernel_fixtures import kernel_parameters_from_values


def _field_parameter_values(Nx, Ny, Nz):
        return {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "tile_shape": (Nx, Ny, Nz),
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _active_interior(parameter_set):
    g = int(parameter_set["guard_cells"])
    return (
        0,
        0,
        0,
        slice(g, g + int(parameter_set["Nx"])),
        slice(g, g + int(parameter_set["Ny"])),
        slice(g, g + int(parameter_set["Nz"])),
    )


