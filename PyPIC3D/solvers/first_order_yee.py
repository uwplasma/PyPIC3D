from PyPIC3D.boundary_conditions.PML import (
    apply_tiled_pml_to_b_curl,
    apply_tiled_pml_to_e_curl,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.utilities.filters import digital_filter_vector


def update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after B halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Ex, Ey, Ez = E_tiles
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = static_parameters.guard_cells
    # get the tile information
    del tile_shape

    active = slice(g, -g)
    # build interior slice for active axes
    backward = slice(g - 1, -g - 1)
    # build backward slice used for differences from vertex fields to center fields

    Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells(B_tiles, static_parameters, g)
    Jx, Jy, Jz = J_tiles
    current = slice(g, -g) #_active_slice(g)

    dt = dynamic_parameters.dt
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz
    C = dynamic_parameters.C
    eps = dynamic_parameters.eps

    # Backward differences map staggered B components onto same-index E/J
    # locations under the legacy center=collocated, vertex=staggered contract.
    dBz_dy = (Bz[:, :, :, active, active, active] - Bz[:, :, :, active, backward, active]) / dy
    dBy_dz = (By[:, :, :, active, active, active] - By[:, :, :, active, active, backward]) / dz
    dBx_dz = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, active, backward]) / dz
    dBx_dy = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, backward, active]) / dy
    dBz_dx = (Bz[:, :, :, active, active, active] - Bz[:, :, :, backward, active, active]) / dx
    dBy_dx = (By[:, :, :, active, active, active] - By[:, :, :, backward, active, active]) / dx

    if pml_state is None:
        curl_x = dBz_dy - dBy_dz
        curl_y = dBx_dz - dBz_dx
        curl_z = dBy_dx - dBx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_e_curl(
            (dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy),
            static_parameters,
            dynamic_parameters,
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

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), static_parameters, g)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Ex, Ey, Ez = digital_filter_vector((Ex, Ey, Ez), dynamic_parameters.alpha, num_guard_cells=g)

    bc_x, bc_y, bc_z = static_parameters.boundary_conditions
    if int(bc_x) == BC_CONDUCTING:
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=0, num_guard_cells=g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=0, num_guard_cells=g)
    if int(bc_y) == BC_CONDUCTING:
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=1, num_guard_cells=g)
        Ez = ghost_cells.apply_tiled_zero_boundary(Ez, static_parameters, axis=1, num_guard_cells=g)
    if int(bc_z) == BC_CONDUCTING:
        Ex = ghost_cells.apply_tiled_zero_boundary(Ex, static_parameters, axis=2, num_guard_cells=g)
        Ey = ghost_cells.apply_tiled_zero_boundary(Ey, static_parameters, axis=2, num_guard_cells=g)
    # conducting walls zero tangential E components on the physical boundary
    # planes; the shared scalar helper refreshes halos through ppermute.

    return ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), static_parameters, g), pml_state


def update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state=None):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Bx, By, Bz = B_tiles
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    tile_nx, tile_ny, tile_nz = tile_shape
    g = static_parameters.guard_cells
    g = int(g)
    del tile_nx, tile_ny, tile_nz
    active = slice(g, -g)
    # build interior slice for active axes
    forward = slice(g + 1, None if g == 1 else -g + 1)
    # build forward slice used for differences from center fields to vertex fields

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g)
    dt = dynamic_parameters.dt
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    # Forward differences map same-index E/J components onto staggered B
    # locations under the legacy center=collocated, vertex=staggered contract.
    dEz_dy = (Ez[:, :, :, active, forward, active] - Ez[:, :, :, active, active, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, forward] - Ey[:, :, :, active, active, active]) / dz
    dEx_dz = (Ex[:, :, :, active, active, forward] - Ex[:, :, :, active, active, active]) / dz
    dEx_dy = (Ex[:, :, :, active, forward, active] - Ex[:, :, :, active, active, active]) / dy
    dEz_dx = (Ez[:, :, :, forward, active, active] - Ez[:, :, :, active, active, active]) / dx
    dEy_dx = (Ey[:, :, :, forward, active, active] - Ey[:, :, :, active, active, active]) / dx

    if pml_state is None:
        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy
    else:
        (curl_x, curl_y, curl_z), pml_state = apply_tiled_pml_to_b_curl(
            (dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy),
            static_parameters,
            dynamic_parameters,
            pml_state,
        )

    Bx = Bx.at[:, :, :, active, active, active].set(Bx[:, :, :, active, active, active] - dt * curl_x)
    By = By.at[:, :, :, active, active, active].set(By[:, :, :, active, active, active] - dt * curl_y)
    Bz = Bz.at[:, :, :, active, active, active].set(Bz[:, :, :, active, active, active] - dt * curl_z)

    Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), static_parameters, g)
    # refresh tile halos before the digital field filter, matching the global
    # ghost-cell order in the standard Yee solver.

    Bx, By, Bz = digital_filter_vector((Bx, By, Bz), dynamic_parameters.alpha, num_guard_cells=g)

    return ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), static_parameters, g), pml_state
