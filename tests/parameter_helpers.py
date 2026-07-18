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

    static_parameters = dict(world)
    static_parameters["boundary_conditions"] = _axis_tuple(world["boundary_conditions"])
    static_parameters["particle_boundary_conditions"] = _axis_tuple(
        world.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})
    )

    dynamic_parameters = {
        key: value
        for key, value in world.items()
        if key not in ("boundary_conditions", "particle_boundary_conditions", "field_mesh", "tile_shape")
    }
    dynamic_parameters.update(constants)

    return static_parameters, dynamic_parameters
