import math

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


PML_WALLS = ["-x", "+x", "-y", "+y", "-z", "+z"]

_AXIS_FOR_WALL = {"-x": "x", "+x": "x", "-y": "y", "+y": "y", "-z": "z", "+z": "z"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_SPACING_KEY = {"x": "dx", "y": "dy", "z": "dz"}
_COUNT_KEY = {"x": "Nx", "y": "Ny", "z": "Nz"}


def load_pml_from_toml(raw_pml, world, constants):
    """
    Read PML wall layers from TOML-style config and build the stretch profiles.

    The returned tuple is

        active, pml_x, pml_y, pml_z, profiles

    where `profiles = (sigma_x, sigma_y, sigma_z)`.  The layer tuples built here
    are kept local because callers only need the finished coordinate-stretch
    profiles, not a second configuration object.
    """
    raw_layers = [] if raw_pml is None else raw_pml
    raw_layers = [raw_layers] if isinstance(raw_layers, dict) else list(raw_layers)

    pml_layers = []
    seen_walls = set()
    for raw in raw_layers:
        wall = raw.get("wall")
        if wall not in PML_WALLS:
            raise ValueError(f"Invalid PML wall: {wall}. Expected one of {PML_WALLS}")
        if wall in seen_walls:
            raise ValueError(f"Duplicate PML wall: {wall}")
        seen_walls.add(wall)

        axis = _AXIS_FOR_WALL[wall]
        thickness = int(raw.get("thickness", 0))
        active_cells = int(world[_COUNT_KEY[axis]])
        if thickness <= 0:
            raise ValueError(f"PML thickness for {wall} must be positive")
        if thickness > active_cells:
            raise ValueError(
                f"PML thickness for {wall} exceeds active cells on {axis}: {thickness} > {active_cells}"
            )

        order = float(raw.get("order", 3.0))
        target_reflection = float(raw.get("target_reflection", 1.0e-8))
        if "sigma_max" in raw:
            sigma_max = float(raw["sigma_max"])
        else:
            layer_width = thickness * float(world[_SPACING_KEY[axis]])
            sigma_max = -((order + 1.0) * float(constants["C"]) * math.log(target_reflection)) / (
                2.0 * layer_width
            )

        pml_layers.append((wall, axis, thickness, order, sigma_max))

    pml_x = any(axis == "x" for _, axis, _, _, _ in pml_layers)
    pml_y = any(axis == "y" for _, axis, _, _, _ in pml_layers)
    pml_z = any(axis == "z" for _, axis, _, _, _ in pml_layers)

    return bool(pml_layers), pml_x, pml_y, pml_z, build_pml(world, tuple(pml_layers))


def build_pml(world, pml_layers):
    """
    Build ghost-celled damping profiles for the coordinate stretch.

    `sigma_x`, `sigma_y`, and `sigma_z` are the real damping strengths in the
    frequency-space stretch factors.  The ramp is weak at the interface with the
    physical domain and grows toward the outer wall, so the outgoing wave sees a
    gradual coordinate stretch rather than a sharp jump.
    """
    shape = (int(world["Nx"]) + 2, int(world["Ny"]) + 2, int(world["Nz"]) + 2)

    sigma_x = jnp.zeros(shape)
    sigma_y = jnp.zeros(shape)
    sigma_z = jnp.zeros(shape)
    profiles = (sigma_x, sigma_y, sigma_z)

    for wall, axis, thickness, order, sigma_max in pml_layers:
        axis_index = _AXIS_INDEX[axis]
        profile_index = _AXIS_INDEX[axis]
        sigma_profile = profiles[profile_index]

        for i in range(int(thickness)):
            sigma = float(sigma_max) * ((i + 1) / int(thickness)) ** float(order)
            if wall[0] == "-":
                index = int(thickness) - i
            else:
                index = shape[axis_index] - 1 - int(thickness) + i

            slices = [slice(None), slice(None), slice(None)]
            slices[axis_index] = index
            sigma_profile = sigma_profile.at[tuple(slices)].set(sigma)

        profiles = profiles[:profile_index] + (sigma_profile,) + profiles[profile_index + 1 :]

    return profiles


def initialize_pml_state(world):
    """
    Allocate ADE memory for stretched derivatives.

    `pml_state = (E_memory, B_memory)`.

    E_memory stores B-derivative history in this order:
        dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy

    B_memory stores E-derivative history in this order:
        dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy

    Each memory array has physical-interior shape `(Nx, Ny, Nz)`, matching the
    finite-difference arrays that enter the Yee curl.
    """
    shape = (int(world["Nx"]), int(world["Ny"]), int(world["Nz"]))

    e_memory = tuple(jnp.zeros(shape) for _ in range(6))
    b_memory = tuple(jnp.zeros(shape) for _ in range(6))

    return e_memory, b_memory


def stretch_spatial_derivative(derivative, memory, sigma, dt):
    """
    Apply one coordinate-stretched PML derivative.

    In frequency space, a PML is a complex coordinate stretch.  For an x
    derivative,

        d_dx -> (1 / sigma(w)) d_dx

    where `sigma(w)` is the frequency-domain stretch factor.  Equivalently,
    one often writes `s_x(w) = 1 + sigma_x / (i w)`, so the derivative becomes
    `(1 / s_x) d_dx`.  The real array `sigma` below is the local damping profile
    inside the PML.  The memory term is the time-domain bookkeeping that applies
    this stretched derivative without storing the whole field history.
    """
    b = jnp.exp(-sigma * dt)
    memory_new = b * memory + (b - 1.0) * derivative
    return derivative + memory_new, memory_new


def apply_pml_to_e_curl(derivatives, world, pml_state):
    """
    Stretch the B derivatives before assembling the curl used in Ampere's law.
    """
    dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy = derivatives
    # The PML state is (E_memory, B_memory), but the E curl only needs the B memory.
    e_memory, b_memory = pml_state
    (
        memory_dBz_dy,
        memory_dBy_dz,
        memory_dBx_dz,
        memory_dBz_dx,
        memory_dBy_dx,
        memory_dBx_dy,
    ) = e_memory
    # unpack the memory in the same order as the derivatives, so we can apply stretch_spatial_derivative

    _, _, _, _, profiles = world["pml"]
    sigma_x, sigma_y, sigma_z = profiles
    sigma_x = sigma_x[1:-1, 1:-1, 1:-1]
    sigma_y = sigma_y[1:-1, 1:-1, 1:-1]
    sigma_z = sigma_z[1:-1, 1:-1, 1:-1]
    dt = world["dt"]
    # unpack the profiles and select only interior cells (no ghost cells)

    dBz_dy, memory_dBz_dy = stretch_spatial_derivative(dBz_dy, memory_dBz_dy, sigma_y, dt)
    dBy_dz, memory_dBy_dz = stretch_spatial_derivative(dBy_dz, memory_dBy_dz, sigma_z, dt)
    dBx_dz, memory_dBx_dz = stretch_spatial_derivative(dBx_dz, memory_dBx_dz, sigma_z, dt)
    dBz_dx, memory_dBz_dx = stretch_spatial_derivative(dBz_dx, memory_dBz_dx, sigma_x, dt)
    dBy_dx, memory_dBy_dx = stretch_spatial_derivative(dBy_dx, memory_dBy_dx, sigma_x, dt)
    dBx_dy, memory_dBx_dy = stretch_spatial_derivative(dBx_dy, memory_dBx_dy, sigma_y, dt)
    # apply the stretch to each derivative and update the memory

    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy
    # assemble the curl from the stretched derivatives

    e_memory = (
        memory_dBz_dy,
        memory_dBy_dz,
        memory_dBx_dz,
        memory_dBz_dx,
        memory_dBy_dx,
        memory_dBx_dy,
    )
    # pack the updated memory in the same order as the derivatives

    return (curl_x, curl_y, curl_z), (e_memory, b_memory)
    # return the curl and the updated PML state (with the new E memory and unchanged B memory)


def apply_pml_to_b_curl(derivatives, world, pml_state):
    """
    Stretch the E derivatives before assembling the curl used in Faraday's law.
    """
    dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy = derivatives
    e_memory, b_memory = pml_state
    (
        memory_dEz_dy,
        memory_dEy_dz,
        memory_dEx_dz,
        memory_dEz_dx,
        memory_dEy_dx,
        memory_dEx_dy,
    ) = b_memory
    # unpack the memory in the same order as the derivatives, so we can apply stretch_spatial_derivative

    _, _, _, _, profiles = world["pml"]
    sigma_x, sigma_y, sigma_z = profiles
    sigma_x = sigma_x[1:-1, 1:-1, 1:-1]
    sigma_y = sigma_y[1:-1, 1:-1, 1:-1]
    sigma_z = sigma_z[1:-1, 1:-1, 1:-1]
    dt = world["dt"]
    # unpack the profiles and select only interior cells (no ghost cells)

    dEz_dy, memory_dEz_dy = stretch_spatial_derivative(dEz_dy, memory_dEz_dy, sigma_y, dt)
    dEy_dz, memory_dEy_dz = stretch_spatial_derivative(dEy_dz, memory_dEy_dz, sigma_z, dt)
    dEx_dz, memory_dEx_dz = stretch_spatial_derivative(dEx_dz, memory_dEx_dz, sigma_z, dt)
    dEz_dx, memory_dEz_dx = stretch_spatial_derivative(dEz_dx, memory_dEz_dx, sigma_x, dt)
    dEy_dx, memory_dEy_dx = stretch_spatial_derivative(dEy_dx, memory_dEy_dx, sigma_x, dt)
    dEx_dy, memory_dEx_dy = stretch_spatial_derivative(dEx_dy, memory_dEx_dy, sigma_y, dt)
    # apply the stretch to each derivative and update the memory

    curl_x = dEz_dy - dEy_dz
    curl_y = dEx_dz - dEz_dx
    curl_z = dEy_dx - dEx_dy
    # assemble the curl from the stretched derivatives

    b_memory = (
        memory_dEz_dy,
        memory_dEy_dz,
        memory_dEx_dz,
        memory_dEz_dx,
        memory_dEy_dx,
        memory_dEx_dy,
    )

    return (curl_x, curl_y, curl_z), (e_memory, b_memory)


def update_ghost_cells_for_pml(field, world):
    """
    Fill ghost cells without periodically wrapping across a PML wall.

    The PML damping happens in the stretched derivatives above.  This ghost-cell
    rule only prevents a periodic stencil from feeding a wave back through the
    opposite side of a PML-active axis.
    """
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    _, pml_x, pml_y, pml_z, _ = world["pml"]

    bc_x = jnp.where((pml_x) & (bc_x == BC_PERIODIC), BC_CONDUCTING, bc_x)
    bc_y = jnp.where((pml_y) & (bc_y == BC_PERIODIC), BC_CONDUCTING, bc_y)
    bc_z = jnp.where((pml_z) & (bc_z == BC_PERIODIC), BC_CONDUCTING, bc_z)

    return update_ghost_cells(field, bc_x, bc_y, bc_z)
