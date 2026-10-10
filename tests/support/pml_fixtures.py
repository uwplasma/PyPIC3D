import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    PML_WALLS,
    build_pml,
    initialize_pml_state,
    initialize_tiled_pml_state,
    load_pml_from_toml,
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
    tile_pml_profiles,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.support.compiled_yee import (
    assemble_yee_curl,
    update_B,
    update_E,
)
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from tests.kernel_fixtures import kernel_parameters_from_values, tile_vector_field


def _update_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field,
    )
    return field


def _update_ghost_cells_from_parameters(field, parameter_set):
    bc_x = parameter_set["boundary_conditions"]["x"]
    bc_y = parameter_set["boundary_conditions"]["y"]
    bc_z = parameter_set["boundary_conditions"]["z"]

    return _update_ghost_cells(field, bc_x, bc_y, bc_z)


def _empty_global_fields(parameter_set):
    shape = (parameter_set["Nx"] + 2, parameter_set["Ny"] + 2, parameter_set["Nz"] + 2)
    E = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    B = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    J = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    return E, B, J


def _legacy_stretch_derivatives(derivatives, memory, sigma, dt, derivative_axes):
    stretched = []
    memory_new = []
    for derivative, previous, axis in zip(derivatives, memory, derivative_axes):
        b = jnp.exp(-sigma[axis] * dt)
        updated = b * previous + (b - 1.0) * derivative
        stretched.append(derivative + updated)
        memory_new.append(updated)
    return tuple(stretched), tuple(memory_new)


def _base_parameter_values(nx=24, ny=1, nz=1):
    parameter_set = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx": 1.0 / nx,
        "dy": 1.0,
        "dz": 1.0,
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


def _empty_config(tmpdir, solver="electrodynamic_yee", pml=None):
    sim = {
        "name": "pml init test",
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
    if pml is not None:
        config["pml"] = pml
    return config


def _load_pml(raw_pml, parameter_set, dynamic_values):
    dynamic_parameters = _dynamic_parameters(parameter_set, dynamic_values)
    return load_pml_from_toml(raw_pml, None, dynamic_parameters)


def _initialize_pml_state(parameter_set, dynamic_values=None):
    dynamic_parameters = _dynamic_parameters(parameter_set, dynamic_values)
    return initialize_pml_state(dynamic_parameters, parameter_set["pml"][-1])


def _initialize_tiled_pml_state(parameter_set, tile_shape, dynamic_values=None):
    parameter_set = {**parameter_set, "tile_shape": tile_shape}
    static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
    return initialize_tiled_pml_state(static_parameters, dynamic_parameters, parameter_set["pml"][-1], tile_shape)


def _tile_pml_profiles(parameter_set, tile_shape, dynamic_values=None):
    parameter_set = {**parameter_set, "tile_shape": tile_shape}
    static_parameters, _ = kernel_parameters_from_values(parameter_set, dynamic_values)
    return tile_pml_profiles(static_parameters, parameter_set["pml"][-1], tile_shape)


class PMLConfigurationFixtures:
    pass


class PMLInitializationFixtures:
    pass


class PMLFDTDBehaviorFixtures:
    pass
