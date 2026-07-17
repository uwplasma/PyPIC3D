import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh


def _axis_tuple(axis_values):
    return (
        int(axis_values["x"]),
        int(axis_values["y"]),
        int(axis_values["z"]),
    )


def _tile_shape(static_config):
    if "tile_shape" in static_config:
        return tuple(int(width) for width in static_config["tile_shape"])

    return (
        int(static_config["particle_tile_nx"]),
        int(static_config["particle_tile_ny"]),
        int(static_config["particle_tile_nz"]),
    )


def _field_mesh(static_config, tile_shape):
    if static_config.get("field_mesh") is not None:
        return static_config["field_mesh"]

    tile_grid_shape = (
        int(static_config["Nx"]) // int(tile_shape[0]),
        int(static_config["Ny"]) // int(tile_shape[1]),
        int(static_config["Nz"]) // int(tile_shape[2]),
    )
    return make_field_mesh(tile_grid_shape)


def build_static_parameters(static_config):
    """
    Collect compile-time PIC choices for the timestep kernels.

    These values select numerical branches, tile layout, and boundary behavior.
    They are intended to be closed over by a jitted driver or passed through
    non-jitted Python dispatch, not traced as dynamic arrays.
    """

    tile_shape = _tile_shape(static_config)

    return {
        "solver": static_config.get("solver", "electrodynamic_yee"),
        "electrostatic": bool(static_config.get("electrostatic", False)),
        "relativistic": bool(static_config.get("relativistic", True)),
        "particle_pusher": static_config.get("particle_pusher", "boris"),
        "current_deposition": static_config.get("current_deposition", "direct"),
        "current_filter": static_config.get("current_filter", "none"),
        "shape_factor": int(static_config["shape_factor"]),
        "guard_cells": int(static_config["guard_cells"]),
        "tile_shape": tile_shape,
        "boundary_conditions": _axis_tuple(static_config["boundary_conditions"]),
        "particle_boundary_conditions": _axis_tuple(
            static_config.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})
        ),
        "field_mesh": _field_mesh(static_config, tile_shape),
    }


def build_dynamic_parameters(dynamic_config, extra_dynamic_config=None):
    """
    Collect scalar/grid data that can move through JAX as a PyTree.
    """

    if extra_dynamic_config is None:
        extra_dynamic_config = {}

    return {
        "dt": jnp.asarray(dynamic_config["dt"]),
        "dx": jnp.asarray(dynamic_config["dx"]),
        "dy": jnp.asarray(dynamic_config["dy"]),
        "dz": jnp.asarray(dynamic_config["dz"]),
        "Nx": jnp.asarray(dynamic_config["Nx"]),
        "Ny": jnp.asarray(dynamic_config["Ny"]),
        "Nz": jnp.asarray(dynamic_config["Nz"]),
        "x_wind": jnp.asarray(dynamic_config["x_wind"]),
        "y_wind": jnp.asarray(dynamic_config["y_wind"]),
        "z_wind": jnp.asarray(dynamic_config["z_wind"]),
        "C": jnp.asarray(dynamic_config.get("C", extra_dynamic_config.get("C", 1.0))),
        "eps": jnp.asarray(dynamic_config.get("eps", extra_dynamic_config.get("eps", 1.0))),
        "mu": jnp.asarray(dynamic_config.get("mu", extra_dynamic_config.get("mu", 1.0))),
        "kb": jnp.asarray(dynamic_config.get("kb", extra_dynamic_config.get("kb", 1.0))),
        "alpha": jnp.asarray(dynamic_config.get("alpha", extra_dynamic_config.get("alpha", 1.0))),
        "grids": dynamic_config["grids"],
    }


def boundary_dict(static_parameters):
    bc_x, bc_y, bc_z = static_parameters["boundary_conditions"]
    return {"x": bc_x, "y": bc_y, "z": bc_z}


def particle_boundary_dict(static_parameters):
    bc_x, bc_y, bc_z = static_parameters["particle_boundary_conditions"]
    return {"x": bc_x, "y": bc_y, "z": bc_z}
