import math

import jax
import jax.numpy as jnp

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


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("PML tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def _tile_scalar_profile(profile, tile_shape, num_guard_cells=2):
    """
    Split one ghost-celled PML profile into compact tile-local profile arrays.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    Nx = int(profile.shape[0]) - 2
    Ny = int(profile.shape[1]) - 2
    Nz = int(profile.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)
    g = int(num_guard_cells)

    if g != 1:
        profile_tiles = jnp.zeros(
            (ntx, nty, ntz, tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g),
            dtype=profile.dtype,
        )
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = tx * tile_nx
                    iy = ty * tile_ny
                    iz = tz * tile_nz
                    tile_with_one_guard = profile[ix:ix + tile_nx + 2, iy:iy + tile_ny + 2, iz:iz + tile_nz + 2]
                    profile_tiles = profile_tiles.at[
                        tx,
                        ty,
                        tz,
                        g - 1:g + tile_nx + 1,
                        g - 1:g + tile_ny + 1,
                        g - 1:g + tile_nz + 1,
                    ].set(tile_with_one_guard)
        return profile_tiles

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(profile, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def tile_pml_profiles(world, tile_shape):
    """
    Tile the ghost-celled PML conductivity profiles.

    The leading tile axes match tiled field storage.  The final three axes use
    the same guard-cell depth as the field tiles.
    """

    _, _, _, _, profiles = world["pml"]
    g = int(world.get("guard_cells", 2))
    return tuple(_tile_scalar_profile(profile, tile_shape, num_guard_cells=g) for profile in profiles)


def initialize_tiled_pml_state(world, tile_shape):
    """
    Allocate tile-local ADE memory for stretched tiled Yee derivatives.

    The memory arrays have leading tile axes followed by the tile physical
    interior.  They intentionally do not store halos; the ADE terms are attached
    to the derivatives after the tile halo exchange has already supplied the
    stencil values.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx = _tile_axis_count(int(world["Nx"]), tile_nx)
    nty = _tile_axis_count(int(world["Ny"]), tile_ny)
    ntz = _tile_axis_count(int(world["Nz"]), tile_nz)
    shape = (ntx, nty, ntz, tile_nx, tile_ny, tile_nz)

    e_memory = tuple(jnp.zeros(shape) for _ in range(6))
    b_memory = tuple(jnp.zeros(shape) for _ in range(6))
    tiled_profiles = tile_pml_profiles(world, tile_shape)

    return e_memory, b_memory, tiled_profiles


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


def apply_tiled_pml_to_e_curl(derivatives, world, pml_state):
    """
    Stretch tile-local B derivatives before assembling Ampere's-law curls.
    """

    dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy = derivatives
    e_memory, b_memory, tiled_profiles = pml_state
    (
        memory_dBz_dy,
        memory_dBy_dz,
        memory_dBx_dz,
        memory_dBz_dx,
        memory_dBy_dx,
        memory_dBx_dy,
    ) = e_memory

    sigma_x, sigma_y, sigma_z = tiled_profiles
    g = int(world.get("guard_cells", 2))
    sigma_x = sigma_x[:, :, :, g:-g, g:-g, g:-g]
    sigma_y = sigma_y[:, :, :, g:-g, g:-g, g:-g]
    sigma_z = sigma_z[:, :, :, g:-g, g:-g, g:-g]
    dt = world["dt"]

    dBz_dy, memory_dBz_dy = stretch_spatial_derivative(dBz_dy, memory_dBz_dy, sigma_y, dt)
    dBy_dz, memory_dBy_dz = stretch_spatial_derivative(dBy_dz, memory_dBy_dz, sigma_z, dt)
    dBx_dz, memory_dBx_dz = stretch_spatial_derivative(dBx_dz, memory_dBx_dz, sigma_z, dt)
    dBz_dx, memory_dBz_dx = stretch_spatial_derivative(dBz_dx, memory_dBz_dx, sigma_x, dt)
    dBy_dx, memory_dBy_dx = stretch_spatial_derivative(dBy_dx, memory_dBy_dx, sigma_x, dt)
    dBx_dy, memory_dBx_dy = stretch_spatial_derivative(dBx_dy, memory_dBx_dy, sigma_y, dt)

    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy

    e_memory = (
        memory_dBz_dy,
        memory_dBy_dz,
        memory_dBx_dz,
        memory_dBz_dx,
        memory_dBy_dx,
        memory_dBx_dy,
    )

    return (curl_x, curl_y, curl_z), (e_memory, b_memory, tiled_profiles)


def apply_tiled_pml_to_b_curl(derivatives, world, pml_state):
    """
    Stretch tile-local E derivatives before assembling Faraday-law curls.
    """

    dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy = derivatives
    e_memory, b_memory, tiled_profiles = pml_state
    (
        memory_dEz_dy,
        memory_dEy_dz,
        memory_dEx_dz,
        memory_dEz_dx,
        memory_dEy_dx,
        memory_dEx_dy,
    ) = b_memory

    sigma_x, sigma_y, sigma_z = tiled_profiles
    g = int(world.get("guard_cells", 2))
    sigma_x = sigma_x[:, :, :, g:-g, g:-g, g:-g]
    sigma_y = sigma_y[:, :, :, g:-g, g:-g, g:-g]
    sigma_z = sigma_z[:, :, :, g:-g, g:-g, g:-g]
    dt = world["dt"]

    dEz_dy, memory_dEz_dy = stretch_spatial_derivative(dEz_dy, memory_dEz_dy, sigma_y, dt)
    dEy_dz, memory_dEy_dz = stretch_spatial_derivative(dEy_dz, memory_dEy_dz, sigma_z, dt)
    dEx_dz, memory_dEx_dz = stretch_spatial_derivative(dEx_dz, memory_dEx_dz, sigma_z, dt)
    dEz_dx, memory_dEz_dx = stretch_spatial_derivative(dEz_dx, memory_dEz_dx, sigma_x, dt)
    dEy_dx, memory_dEy_dx = stretch_spatial_derivative(dEy_dx, memory_dEy_dx, sigma_x, dt)
    dEx_dy, memory_dEx_dy = stretch_spatial_derivative(dEx_dy, memory_dEx_dy, sigma_y, dt)

    curl_x = dEz_dy - dEy_dz
    curl_y = dEx_dz - dEz_dx
    curl_z = dEy_dx - dEx_dy

    b_memory = (
        memory_dEz_dy,
        memory_dEy_dz,
        memory_dEx_dz,
        memory_dEz_dx,
        memory_dEy_dx,
        memory_dEx_dy,
    )

    return (curl_x, curl_y, curl_z), (e_memory, b_memory, tiled_profiles)
