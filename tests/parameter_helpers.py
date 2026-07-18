from types import SimpleNamespace

from PyPIC3D.parameters import build_dynamic_parameters, build_static_parameters


def _axis_tuple(axis_values):
    if isinstance(axis_values, tuple):
        return axis_values
    return (
        int(axis_values["x"]),
        int(axis_values["y"]),
        int(axis_values["z"]),
    )


def split_test_parameters(world, constants=None):
    """
    Build the split kernel parameter dictionaries used by tiled tests.

    Most older test fixtures still construct one combined ``world`` dictionary.
    Production kernels now take static simulation choices separately from
    dynamic scalar/grid values, so tests split that fixture at the call site.
    """

    if constants is None:
        constants = {}

    if "grids" in world and "particle_tile_nx" in world and "shape_factor" in world:
        return build_static_parameters(world), build_dynamic_parameters(world, constants)

    static_config = dict(world)
    static_config["boundary_conditions"] = _axis_tuple(world["boundary_conditions"])
    static_config["particle_boundary_conditions"] = _axis_tuple(
        world.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})
    )

    dynamic_parameters = {
        key: value
        for key, value in world.items()
        if key not in ("boundary_conditions", "particle_boundary_conditions", "field_mesh", "tile_shape")
    }
    dynamic_parameters.update(constants)

    return build_static_parameters(static_config), build_dynamic_parameters(dynamic_parameters, constants)


def field_initialization_parameters(world, constants=None):
    """
    Build the direct-attribute parameter objects needed by field-array setup.

    These lightweight fixtures intentionally cover only the storage geometry
    used by ``initialize_fields`` and ``build_tiled_array``.
    """

    if constants is None:
        constants = {}

    grids = world.get("grids", {})
    grid_parameters = SimpleNamespace(
        vertex=grids.get("vertex"),
        center=grids.get("center"),
        tiled_vertex_grid=grids.get("tiled_vertex_grid"),
        tiled_center_grid=grids.get("tiled_center_grid"),
    )
    static_parameters = SimpleNamespace(
        tile_shape=tuple(int(width) for width in world["tile_shape"]),
        guard_cells=int(world.get("guard_cells", 2)),
        boundary_conditions=_axis_tuple(world.get("boundary_conditions", {"x": 0, "y": 0, "z": 0})),
        field_mesh=world.get("field_mesh"),
        shape_factor=world.get("shape_factor", "cloud-in-cell"),
        particle_tile_capacity_factor=float(world.get("particle_tile_capacity_factor", 1.0)),
    )
    dynamic_parameters = SimpleNamespace(
        Nx=world["Nx"],
        Ny=world["Ny"],
        Nz=world["Nz"],
        dx=world.get("dx"),
        dy=world.get("dy"),
        dz=world.get("dz"),
        dt=world.get("dt"),
        x_wind=world.get("x_wind"),
        y_wind=world.get("y_wind"),
        z_wind=world.get("z_wind"),
        C=world.get("C", constants.get("C", 1.0)),
        eps=world.get("eps", constants.get("eps", 1.0)),
        mu=world.get("mu", constants.get("mu", 1.0)),
        kb=world.get("kb", constants.get("kb", 1.0)),
        alpha=world.get("alpha", constants.get("alpha", 1.0)),
        grids=grid_parameters,
    )

    return static_parameters, dynamic_parameters
