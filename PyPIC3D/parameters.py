import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh


def _axis_tuple(axis_dict):
    return (
        int(axis_dict["x"]),
        int(axis_dict["y"]),
        int(axis_dict["z"]),
    )


def _field_mesh(world, tile_shape):
    if world.get("field_mesh") is not None:
        return world["field_mesh"]

    tile_grid_shape = (
        int(world["Nx"]) // int(tile_shape[0]),
        int(world["Ny"]) // int(tile_shape[1]),
        int(world["Nz"]) // int(tile_shape[2]),
    )
    return make_field_mesh(tile_grid_shape)


def build_static_parameters(
    world,
    solver,
    electrostatic,
    relativistic,
    particle_pusher,
):
    """
    Collect compile-time PIC choices for the jitted timestep kernels.

    These values select numerical branches, tile layout, and boundary behavior.
    They should be closed over by a jitted driver or passed only to non-jitted
    Python dispatch code.
    """

    tile_shape = tuple(int(width) for width in world["tile_shape"])

    return {
        "solver": solver,
        "electrostatic": bool(electrostatic),
        "relativistic": bool(relativistic),
        "particle_pusher": particle_pusher,
        "current_deposition": world.get("current_deposition", "direct"),
        "current_filter": world.get("current_filter", "none"),
        "shape_factor": int(world["shape_factor"]),
        "guard_cells": int(world["guard_cells"]),
        "tile_shape": tile_shape,
        "boundary_conditions": _axis_tuple(world["boundary_conditions"]),
        "particle_boundary_conditions": _axis_tuple(
            world.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})
        ),
        "field_mesh": _field_mesh(world, tile_shape),
    }


def build_dynamic_parameters(world, constants):
    """
    Collect scalar/grid data that can move through JAX as a PyTree.
    """

    return {
        "dt": jnp.asarray(world["dt"]),
        "dx": jnp.asarray(world["dx"]),
        "dy": jnp.asarray(world["dy"]),
        "dz": jnp.asarray(world["dz"]),
        "Nx": jnp.asarray(world["Nx"]),
        "Ny": jnp.asarray(world["Ny"]),
        "Nz": jnp.asarray(world["Nz"]),
        "x_wind": jnp.asarray(world["x_wind"]),
        "y_wind": jnp.asarray(world["y_wind"]),
        "z_wind": jnp.asarray(world["z_wind"]),
        "C": jnp.asarray(constants.get("C", 1.0)),
        "eps": jnp.asarray(constants.get("eps", 1.0)),
        "mu": jnp.asarray(constants.get("mu", 1.0)),
        "kb": jnp.asarray(constants.get("kb", 1.0)),
        "alpha": jnp.asarray(constants.get("alpha", 1.0)),
        "grids": world["grids"],
    }


def static_parameters_from_world(
    world,
    solver="electrodynamic_yee",
    electrostatic=False,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Compatibility helper for tests and notebook code still constructing world.
    """

    return build_static_parameters(
        world,
        solver=solver,
        electrostatic=electrostatic,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )


def dynamic_parameters_from_world(world, constants):
    """
    Compatibility helper for tests and notebook code still constructing constants.
    """

    return build_dynamic_parameters(world, constants)


def kernel_parameters_from_inputs(
    static_or_world,
    dynamic_or_constants,
    solver="electrodynamic_yee",
    electrostatic=False,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Accept either the new split parameters or the legacy world/constants pair.
    """

    if "dt" in static_or_world:
        static_parameters = build_static_parameters(
            static_or_world,
            solver=solver,
            electrostatic=electrostatic,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )
        dynamic_parameters = build_dynamic_parameters(static_or_world, dynamic_or_constants)
        return static_parameters, dynamic_parameters

    return static_or_world, dynamic_or_constants


def boundary_dict(static_parameters):
    bc_x, bc_y, bc_z = static_parameters["boundary_conditions"]
    return {"x": bc_x, "y": bc_y, "z": bc_z}


def particle_boundary_dict(static_parameters):
    bc_x, bc_y, bc_z = static_parameters["particle_boundary_conditions"]
    return {"x": bc_x, "y": bc_y, "z": bc_z}


def world_from_parameters(static_parameters, dynamic_parameters):
    """
    Build the minimal world-shaped view used by existing numerical helpers.
    """

    return {
        "dt": dynamic_parameters["dt"],
        "dx": dynamic_parameters["dx"],
        "dy": dynamic_parameters["dy"],
        "dz": dynamic_parameters["dz"],
        "Nx": dynamic_parameters["Nx"],
        "Ny": dynamic_parameters["Ny"],
        "Nz": dynamic_parameters["Nz"],
        "x_wind": dynamic_parameters["x_wind"],
        "y_wind": dynamic_parameters["y_wind"],
        "z_wind": dynamic_parameters["z_wind"],
        "shape_factor": static_parameters["shape_factor"],
        "guard_cells": static_parameters["guard_cells"],
        "tile_shape": static_parameters["tile_shape"],
        "current_deposition": static_parameters["current_deposition"],
        "current_filter": static_parameters["current_filter"],
        "boundary_conditions": boundary_dict(static_parameters),
        "particle_boundary_conditions": particle_boundary_dict(static_parameters),
        "field_mesh": static_parameters["field_mesh"],
        "grids": dynamic_parameters["grids"],
    }


def constants_from_parameters(dynamic_parameters):
    """
    Build the minimal constants-shaped view used by existing numerical helpers.
    """

    return {
        "C": dynamic_parameters["C"],
        "eps": dynamic_parameters["eps"],
        "mu": dynamic_parameters["mu"],
        "kb": dynamic_parameters["kb"],
        "alpha": dynamic_parameters["alpha"],
    }
