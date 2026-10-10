from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.PML import (
    stretch_tiled_pml_b_derivatives,
    stretch_tiled_pml_e_derivatives,
)
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.boundary_conditions import ghost_cells


def yee_derivatives_e_to_b(E_tiles, static_parameters, dynamic_parameters):
    """
    Return the six forward Yee derivatives from centered E to staggered B.

    Output order is
    ``dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy``.
    """

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)

    Ex, Ey, Ez = ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g, locations=D_FIELD_LOCATIONS)
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    dEz_dy = (Ez[:, :, :, active, forward, active] - Ez[:, :, :, active, active, active]) / dy
    dEy_dz = (Ey[:, :, :, active, active, forward] - Ey[:, :, :, active, active, active]) / dz
    dEx_dz = (Ex[:, :, :, active, active, forward] - Ex[:, :, :, active, active, active]) / dz
    dEz_dx = (Ez[:, :, :, forward, active, active] - Ez[:, :, :, active, active, active]) / dx
    dEy_dx = (Ey[:, :, :, forward, active, active] - Ey[:, :, :, active, active, active]) / dx
    dEx_dy = (Ex[:, :, :, active, forward, active] - Ex[:, :, :, active, active, active]) / dy

    return dEz_dy, dEy_dz, dEx_dz, dEz_dx, dEy_dx, dEx_dy


def yee_derivatives_b_to_e(B_tiles, static_parameters, dynamic_parameters):
    """
    Return the six backward Yee derivatives from staggered B to centered E.

    Output order is
    ``dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy``.
    """

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    backward = slice(g - 1, -g - 1)

    Bx, By, Bz = ghost_cells.update_tiled_vector_ghost_cells(B_tiles, static_parameters, g, locations=B_FIELD_LOCATIONS)
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    dBz_dy = (Bz[:, :, :, active, active, active] - Bz[:, :, :, active, backward, active]) / dy
    dBy_dz = (By[:, :, :, active, active, active] - By[:, :, :, active, active, backward]) / dz
    dBx_dz = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, active, backward]) / dz
    dBz_dx = (Bz[:, :, :, active, active, active] - Bz[:, :, :, backward, active, active]) / dx
    dBy_dx = (By[:, :, :, active, active, active] - By[:, :, :, backward, active, active]) / dx
    dBx_dy = (Bx[:, :, :, active, active, active] - Bx[:, :, :, active, backward, active]) / dy

    return dBz_dy, dBy_dz, dBx_dz, dBz_dx, dBy_dx, dBx_dy


def assemble_yee_curl(derivatives):
    """Assemble a Yee curl from its six directional derivatives."""

    d1, d2, d3, d4, d5, d6 = derivatives
    return d1 - d2, d3 - d4, d5 - d6


def yee_curl_e_to_b(E_tiles, static_parameters, dynamic_parameters):
    """Apply the canonical forward Yee curl from centered E to staggered B."""

    derivatives = yee_derivatives_e_to_b(E_tiles, static_parameters, dynamic_parameters)
    return assemble_yee_curl(derivatives)


def yee_curl_b_to_e(B_tiles, static_parameters, dynamic_parameters):
    """Apply the explicit backward Yee curl from staggered B to centered E."""

    derivatives = yee_derivatives_b_to_e(
        B_tiles,
        static_parameters,
        dynamic_parameters,
    )
    return assemble_yee_curl(derivatives)


def update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state=None):
    """
    Update compact tiled electric fields without assembling a global field.

    The B-to-E curl uses explicit backward differences after refreshing B
    halos from neighboring tiles or field boundary conditions.
    """

    Ex, Ey, Ez = E_tiles
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    # build interior slice for active axes
    Jx, Jy, Jz = J_tiles

    dt = dynamic_parameters.dt
    C = dynamic_parameters.C
    eps = dynamic_parameters.eps

    derivatives = yee_derivatives_b_to_e(
        B_tiles,
        static_parameters,
        dynamic_parameters,
    )
    if pml_state is not None:
        derivatives, pml_state = stretch_tiled_pml_e_derivatives(
            derivatives,
            static_parameters,
            dynamic_parameters,
            pml_state,
        )
    curl_x, curl_y, curl_z = assemble_yee_curl(derivatives)

    Ex = Ex.at[:, :, :, active, active, active].set(
        Ex[:, :, :, active, active, active]
        + (C**2 * curl_x - Jx[:, :, :, active, active, active] / eps) * dt
    )
    Ey = Ey.at[:, :, :, active, active, active].set(
        Ey[:, :, :, active, active, active]
        + (C**2 * curl_y - Jy[:, :, :, active, active, active] / eps) * dt
    )
    Ez = Ez.at[:, :, :, active, active, active].set(
        Ez[:, :, :, active, active, active]
        + (C**2 * curl_z - Jz[:, :, :, active, active, active] / eps) * dt
    )

    Ex, Ey, Ez = ghost_cells.apply_tiled_pec_boundary((Ex, Ey, Ez), static_parameters)

    E_tiles = apply_tiled_supergaussian_absorber(
        (Ex, Ey, Ez),
        static_parameters,
        dynamic_parameters,
        dynamic_parameters.dt,
        locations=D_FIELD_LOCATIONS,
    )
    # A supergaussian layer is a field-only sponge: it damps the evolved fields
    # after the Maxwell update without changing the deposited current.

    return ghost_cells.update_tiled_vector_ghost_cells(E_tiles, static_parameters, g, locations=D_FIELD_LOCATIONS), pml_state


def update_B(E_tiles, B_tiles, static_parameters, dynamic_parameters, pml_state=None):
    """
    Update compact tiled magnetic fields without assembling a global field.

    The Yee curl is evaluated on each tile's physical interior after E halos
    have been refreshed from neighbor tiles or field boundary conditions.
    """

    Bx, By, Bz = B_tiles
    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    # build interior slice for active axes
    dt = dynamic_parameters.dt / 2  # half timestep for B update

    derivatives = yee_derivatives_e_to_b(
        E_tiles,
        static_parameters,
        dynamic_parameters,
    )
    if pml_state is not None:
        derivatives, pml_state = stretch_tiled_pml_b_derivatives(
            derivatives,
            static_parameters,
            dynamic_parameters,
            pml_state,
        )
    curl_x, curl_y, curl_z = assemble_yee_curl(derivatives)

    Bx = Bx.at[:, :, :, active, active, active].set(Bx[:, :, :, active, active, active] - dt * curl_x)
    By = By.at[:, :, :, active, active, active].set(By[:, :, :, active, active, active] - dt * curl_y)
    Bz = Bz.at[:, :, :, active, active, active].set(Bz[:, :, :, active, active, active] - dt * curl_z)

    Bx, By, Bz = apply_tiled_supergaussian_absorber(
        (Bx, By, Bz),
        static_parameters,
        dynamic_parameters,
        dt,
        locations=B_FIELD_LOCATIONS,
    )
    # The B update is split into half steps, so the sponge uses the same half
    # timestep as Faraday's-law update here.

    return ghost_cells.update_tiled_vector_ghost_cells((Bx, By, Bz), static_parameters, g, locations=B_FIELD_LOCATIONS), pml_state
