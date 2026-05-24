import math

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


PML_WALLS = ["-x", "+x", "-y", "+y", "-z", "+z"]

_AXIS_FOR_WALL = {"-x": "x", "+x": "x", "-y": "y", "+y": "y", "-z": "z", "+z": "z"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_SPACING_KEY = {"x": "dx", "y": "dy", "z": "dz"}
_COUNT_KEY = {"x": "Nx", "y": "Ny", "z": "Nz"}


def _normalize_raw_pml(raw_pml):
    if raw_pml is None:
        return []
    if isinstance(raw_pml, dict):
        return [raw_pml]
    return list(raw_pml)


def parse_pml_config(raw_pml, world, constants):
    """
    Validate user PML config and return normalized wall layer dictionaries.
    """
    entries = []
    seen = set()
    for raw in _normalize_raw_pml(raw_pml):
        wall = raw.get("wall")
        if wall not in PML_WALLS:
            raise ValueError(f"Invalid PML wall: {wall}. Expected one of {PML_WALLS}")
        if wall in seen:
            raise ValueError(f"Duplicate PML wall: {wall}")
        seen.add(wall)

        axis = _AXIS_FOR_WALL[wall]
        active_cells = int(world[_COUNT_KEY[axis]])
        thickness = int(raw.get("thickness", 0))
        if thickness <= 0:
            raise ValueError(f"PML thickness for {wall} must be positive")
        if thickness > active_cells:
            raise ValueError(
                f"PML thickness for {wall} exceeds active cells on {axis}: {thickness} > {active_cells}"
            )

        order = float(raw.get("order", 3.0))
        if order <= 0.0:
            raise ValueError(f"PML order for {wall} must be positive")

        target_reflection = float(raw.get("target_reflection", 1e-8))
        if target_reflection <= 0.0 or target_reflection >= 1.0:
            raise ValueError(f"PML target_reflection for {wall} must be between 0 and 1")

        if "sigma_max" in raw:
            sigma_max = float(raw["sigma_max"])
        else:
            layer_width = thickness * float(world[_SPACING_KEY[axis]])
            sigma_max = -((order + 1.0) * float(constants["C"]) * math.log(target_reflection)) / (
                2.0 * layer_width
            )
        if sigma_max <= 0.0:
            raise ValueError(f"PML sigma_max for {wall} must be positive")

        entries.append(
            {
                "wall": wall,
                "axis": axis,
                "thickness": thickness,
                "order": order,
                "target_reflection": target_reflection,
                "sigma_max": sigma_max,
            }
        )
    return entries


def _ghost_shape(world):
    return (int(world["Nx"]) + 2, int(world["Ny"]) + 2, int(world["Nz"]) + 2)


def build_pml_profiles(world, pml_layers):
    """
    Build ghost-celled sigma profiles for x, y, and z derivative directions.
    """
    shape = _ghost_shape(world)
    profiles = {
        "sigma_x": jnp.zeros(shape),
        "sigma_y": jnp.zeros(shape),
        "sigma_z": jnp.zeros(shape),
    }

    for layer in pml_layers:
        wall = layer["wall"]
        axis = layer["axis"]
        axis_index = _AXIS_INDEX[axis]
        sigma_name = f"sigma_{axis}"
        thickness = int(layer["thickness"])
        order = float(layer["order"])
        sigma_max = float(layer["sigma_max"])

        for i in range(thickness):
            ramp = sigma_max * ((i + 1) / thickness) ** order
            index = 1 + i if wall[0] == "-" else shape[axis_index] - 2 - (thickness - 1 - i)
            slices = [slice(None), slice(None), slice(None)]
            slices[axis_index] = index
            profiles[sigma_name] = profiles[sigma_name].at[tuple(slices)].set(ramp)

    return profiles


def _zero_like_profiles(world):
    shape = _ghost_shape(world)
    return {
        "sigma_x": jnp.zeros(shape),
        "sigma_y": jnp.zeros(shape),
        "sigma_z": jnp.zeros(shape),
    }


def build_pml_metadata(raw_pml, world, constants):
    layers = parse_pml_config(raw_pml, world, constants)
    pml_axes = {
        "x": any(layer["axis"] == "x" for layer in layers),
        "y": any(layer["axis"] == "y" for layer in layers),
        "z": any(layer["axis"] == "z" for layer in layers),
    }
    return {
        "active": bool(layers),
        "pml_x": pml_axes["x"],
        "pml_y": pml_axes["y"],
        "pml_z": pml_axes["z"],
        "profiles": build_pml_profiles(world, layers) if layers else _zero_like_profiles(world),
    }


def _zeros_for_terms(world, names):
    shape = (int(world["Nx"]), int(world["Ny"]), int(world["Nz"]))
    return {name: jnp.zeros(shape) for name in names}


def initialize_pml_state(world):
    """
    Return zero-valued auxiliary source memory for PML derivative terms.
    """
    e_terms = ("dBz_dy", "dBy_dz", "dBx_dz", "dBz_dx", "dBy_dx", "dBx_dy")
    b_terms = ("dEz_dy", "dEy_dz", "dEx_dz", "dEz_dx", "dEy_dx", "dEx_dy")
    return {"E": _zeros_for_terms(world, e_terms), "B": _zeros_for_terms(world, b_terms)}


def has_pml(world):
    pml = world.get("pml", None)
    return bool(pml is not None and pml.get("active", False))


def _sigma_for_term(world, term):
    profiles = world["pml"]["profiles"]
    if term.endswith("_dx"):
        return profiles["sigma_x"][1:-1, 1:-1, 1:-1]
    if term.endswith("_dy"):
        return profiles["sigma_y"][1:-1, 1:-1, 1:-1]
    if term.endswith("_dz"):
        return profiles["sigma_z"][1:-1, 1:-1, 1:-1]
    raise ValueError(f"Unsupported PML derivative term: {term}")


def _apply_auxiliary_source(derivative, memory, sigma, dt):
    b = jnp.exp(-sigma * dt)
    memory_new = b * memory + (b - 1.0) * derivative
    return derivative + memory_new, memory_new


def apply_pml_to_e_curl(derivatives, world, pml_state):
    dt = world["dt"]
    memory = pml_state["E"]
    corrected = {}
    memory_new = {}
    for name, derivative in derivatives.items():
        corrected[name], memory_new[name] = _apply_auxiliary_source(
            derivative, memory[name], _sigma_for_term(world, name), dt
        )

    curl_x = corrected["dBz_dy"] - corrected["dBy_dz"]
    curl_y = corrected["dBx_dz"] - corrected["dBz_dx"]
    curl_z = corrected["dBy_dx"] - corrected["dBx_dy"]
    return (curl_x, curl_y, curl_z), {"E": memory_new, "B": pml_state["B"]}


def apply_pml_to_b_curl(derivatives, world, pml_state):
    dt = world["dt"]
    memory = pml_state["B"]
    corrected = {}
    memory_new = {}
    for name, derivative in derivatives.items():
        corrected[name], memory_new[name] = _apply_auxiliary_source(
            derivative, memory[name], _sigma_for_term(world, name), dt
        )

    curl_x = corrected["dEz_dy"] - corrected["dEy_dz"]
    curl_y = corrected["dEx_dz"] - corrected["dEz_dx"]
    curl_z = corrected["dEy_dx"] - corrected["dEx_dy"]
    return (curl_x, curl_y, curl_z), {"E": pml_state["E"], "B": memory_new}


def update_ghost_cells_for_pml(field, world):
    """
    Fill ghost cells while suppressing periodic wrap on axes that contain PML.
    """
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    pml = world["pml"]
    bc_x = jnp.where((pml["pml_x"]) & (bc_x == BC_PERIODIC), BC_CONDUCTING, bc_x)
    bc_y = jnp.where((pml["pml_y"]) & (bc_y == BC_PERIODIC), BC_CONDUCTING, bc_y)
    bc_z = jnp.where((pml["pml_z"]) & (bc_z == BC_PERIODIC), BC_CONDUCTING, bc_z)
    return update_ghost_cells(field, bc_x, bc_y, bc_z)
