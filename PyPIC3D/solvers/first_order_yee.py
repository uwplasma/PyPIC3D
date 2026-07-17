from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.parameters import (
    constants_from_parameters,
    kernel_parameters_from_inputs,
    world_from_parameters,
)
from PyPIC3D.utilities.filters import digital_filter_vector


def update_E(E_tiles, B_tiles, J_tiles, world, constants, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after B halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    static_parameters, dynamic_parameters = kernel_parameters_from_inputs(world, constants)
    world = world_from_parameters(static_parameters, dynamic_parameters)
    constants = constants_from_parameters(dynamic_parameters)

    Ex, Ey, Ez = E_tiles
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    g = int(world["guard_cells"])
    # get the tile information

    active = slice(g, -g)
    # build interior slice for active axes
    forward = slice(g + 1, None if g == 1 else -g + 1)
    # build forward slice used for forward differences

    Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells(B_tiles, world, g)
    Jx, Jy, Jz = J_tiles
    current = slice(g, -g) #_active_slice(g)

    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]
    C = constants["C"]
    eps = constants["eps"]

    # Forward differences use each tile's + side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dBz_dy = (Bz[:, :, :, active, forward, active] - Bz[:, :, :, active, active, active]) / dy
    dBy_dz = (By[:, :, :, active, active, forward] - By[:, :, :, active, active, active]) / dz
    dBx_dz = (Bx[:, :, :, active, active, forward] - Bx[:, :, :, active, active, active]) / dz
    dBx_dy = (Bx[:, :, :, active, forward, active] - Bx[:, :, :, active, active, active]) / dy
    dBz_dx = (Bz[:, :, :, forward, active, active] - Bz[:, :, :, active, active, active]) / dx
    dBy_dx = (By[:, :, :, forward, active, active] - By[:, :, :, active, active, active]) / dx

    if pml_state is None:
        curl_x = dBz_dy - dBy_dz
        curl_y = dBx_dz - dBz_dx
        curl_z = dBy_dx - dBx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_e_curl(
            (dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy),
            world,
            pml_state,
        )

    Ex = Ex.at[:, :, :, active, active, active].set(
        Ex[:, :, :, active, active, active]
        + (C**2 * curl_x - Jx[:, :, :, current, current, current] / eps) * dt
    )
    Ey = Ey.at[:, :, :, active, active, active].set(
        Ey[:, :, :, active, active, active]
        + (C**2 * curl_y - Jy[:, :, :, current, current, current] / eps) * dt
    )
    Ez = Ez.at[:, :, :, active, active, active].set(
        Ez[:, :, :, active, active, active]
        + (C**2 * curl_z - Jz[:, :, :, current, current, current] / eps) * dt
    )

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Ex, Ey, Ez = digital_filter_vector((Ex, Ey, Ez), constants.get("alpha", 1.0), num_guard_cells=g)

    Ex, Ey, Ez = ghost_cells.apply_tiled_conducting_bc((Ex, Ey, Ez), world, num_guard_cells=g)

    return ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g), pml_state


def update_B(E_tiles, B_tiles, world, constants, pml_state=None):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    static_parameters, dynamic_parameters = kernel_parameters_from_inputs(world, constants)
    world = world_from_parameters(static_parameters, dynamic_parameters)
    constants = constants_from_parameters(dynamic_parameters)

    Bx, By, Bz = B_tiles
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    tile_nx, tile_ny, tile_nz = tile_shape
    g = int(world["guard_cells"])
    g = int(g)
    active = slice(g, -g)
    # build interior slice for active axes
    backward = slice(g - 1, -g - 1)
    # build backward slice for active axes, used for backward differences

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, world, g)
    dt = world["dt"]
    dx, dy, dz = world["dx"], world["dy"], world["dz"]

    # Backward differences use each tile's - side guard cell.  Those guards now
    # contain the adjacent tile's interior value, including periodic wrap at
    # the global edge.
    dEz_dy = (Ez[:, :, :, active, active, active] - Ez[:, :, :, active, backward, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, active] - Ey[:, :, :, active, active, backward]) / dz
    dEx_dz = (Ex[:, :, :, active, active, active] - Ex[:, :, :, active, active, backward]) / dz
    dEx_dy = (Ex[:, :, :, active, active, active] - Ex[:, :, :, active, backward, active]) / dy
    dEz_dx = (Ez[:, :, :, active, active, active] - Ez[:, :, :, backward, active, active]) / dx
    dEy_dx = (Ey[:, :, :, active, active, active] - Ey[:, :, :, backward, active, active]) / dx

    if pml_state is None:
        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_b_curl(
            (dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy),
            world,
            pml_state,
        )

    Bx = Bx.at[:, :, :, active, active, active].set(Bx[:, :, :, active, active, active] - dt * curl_x)
    By = By.at[:, :, :, active, active, active].set(By[:, :, :, active, active, active] - dt * curl_y)
    Bz = Bz.at[:, :, :, active, active, active].set(Bz[:, :, :, active, active, active] - dt * curl_z)

    Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), world, g)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Bx, By, Bz = digital_filter_vector((Bx, By, Bz), constants.get("alpha", 1.0), num_guard_cells=g)

    return ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), world, g), pml_state
