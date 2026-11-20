import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.supergaussian import (
    SUPERGAUSSIAN_WALLS,
    apply_tiled_supergaussian_absorber,
    build_supergaussian_envelope,
    load_supergaussian_from_toml,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_CONSTANT, BC_PERIODIC
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.yee.first_order_yee import update_B, update_E
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from tests.kernel_fixtures import kernel_parameters_from_values, tile_vector_field


def _empty_global_fields(parameter_set):
    shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
    E = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    B = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    J = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    return E, B, J


def _base_parameter_values(nx=24, ny=1, nz=1):
    parameter_set = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx": 1.0 / nx,
        "dy": 1.0 if ny == 1 else 1.0 / ny,
        "dz": 1.0 if nz == 1 else 1.0 / nz,
        "dt": 0.5 / nx,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }
    center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
    parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
    return parameter_set


def _dynamic_parameters(parameter_set, dynamic_values=None):
    if dynamic_values is None:
        dynamic_values = {}
    return SimpleNamespace(
        Nx=parameter_set["Nx"],
        Ny=parameter_set["Ny"],
        Nz=parameter_set["Nz"],
        dx=parameter_set["dx"],
        dy=parameter_set["dy"],
        dz=parameter_set["dz"],
        dt=parameter_set["dt"],
        x_wind=parameter_set["x_wind"],
        y_wind=parameter_set["y_wind"],
        z_wind=parameter_set["z_wind"],
        C=parameter_set.get("C", dynamic_values.get("C", 1.0)),
        eps=parameter_set.get("eps", dynamic_values.get("eps", 1.0)),
        mu=parameter_set.get("mu", dynamic_values.get("mu", 1.0)),
        alpha=parameter_set.get("alpha", dynamic_values.get("alpha", 1.0)),
        grids=SimpleNamespace(**parameter_set["grids"]),
    )


def _load_supergaussian(raw, parameter_set, dynamic_values=None):
    return load_supergaussian_from_toml(raw, _dynamic_parameters(parameter_set, dynamic_values))


def _empty_config(tmpdir, solver="electrodynamic_yee", supergaussian=None):
    sim = {
        "name": "supergaussian init test",
        "output_dir": tmpdir,
        "solver": solver,
        "Nx": 8,
        "Ny": 1,
        "Nz": 1,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "Nt": 1,
        "dt": 1e-10,
    }
    config = {"simulation_parameters": sim, "plotting": {"plotting": False}}
    if supergaussian is not None:
        config["supergaussian"] = supergaussian
    return config


