from jax import jit, lax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import BC_CONDUCTING, BC_PERIODIC

# Christopher Woolford, Oct 22 2024
# This file contains functions that apply boundary conditions to a field.


@jit
def update_ghost_cells(field, bc_x, bc_y, bc_z):
    """
    Fill ghost cells of a (Nx+2, Ny+2, Nz+2) field based on boundary conditions.

    For periodic boundaries, ghost cells are copied from the opposite interior edge.
    For conducting boundaries, ghost cells are set to zero.

    Args:
        field (ndarray): 3D field array with shape (Nx+2, Ny+2, Nz+2).
        bc_x, bc_y, bc_z (int): Boundary condition codes per axis (0=periodic, 1=conducting).

    Returns:
        ndarray: Field with ghost cells updated.
    """
    # x-axis ghost cells
    field = lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field
    )

    # y-axis ghost cells
    field = lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field
    )

    # z-axis ghost cells
    field = lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field
    )

    return field


@jit
def fold_ghost_cells(field, bc_x, bc_y, bc_z):
    """
    Fold ghost cell deposits back into the interior after deposition.

    For periodic boundaries, ghost cell values are added to the corresponding
    interior cells on the opposite side, then the ghost cells are cleared.
    For conducting boundaries, ghost cells are simply cleared.

    Args:
        field (ndarray): 3D field array with shape (Nx+2, Ny+2, Nz+2).
        bc_x, bc_y, bc_z (int): Boundary condition codes per axis (0=periodic, 1=conducting).

    Returns:
        ndarray: Field with ghost cell deposits folded back into the interior.
    """
    # x-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[-2, :, :].add(f[-1, :, :]).at[1, :, :].add(f[0, :, :]),
        lambda f: f,
        operand=field
    )
    field = field.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    # clear x ghost cells

    # y-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, -2, :].add(f[:, -1, :]).at[:, 1, :].add(f[:, 0, :]),
        lambda f: f,
        operand=field
    )
    field = field.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    # clear y ghost cells

    # z-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, -2].add(f[:, :, -1]).at[:, :, 1].add(f[:, :, 0]),
        lambda f: f,
        operand=field
    )
    field = field.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    # clear z ghost cells

    return field


@jit
def apply_conducting_bc(E, bc_x, bc_y, bc_z):
    """
    Enforce conducting boundary conditions by zeroing tangential E at boundaries.

    On a conducting wall normal to x: Ey and Ez are zero at x boundaries.
    On a conducting wall normal to y: Ex and Ez are zero at y boundaries.
    On a conducting wall normal to z: Ex and Ey are zero at z boundaries.

    Args:
        E (tuple): Electric field components (Ex, Ey, Ez), each (Nx+2, Ny+2, Nz+2).
        bc_x, bc_y, bc_z (int): Boundary condition codes per axis (0=periodic, 1=conducting).

    Returns:
        tuple: Electric field components with conducting BCs applied.
    """
    Ex, Ey, Ez = E

    ### x-boundary conducting: zero tangential Ey, Ez ###
    Ey = lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[1, :, :].set(0.0).at[-2, :, :].set(0.0),
        lambda f: f,
        operand=Ey
    )
    Ez = lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[1, :, :].set(0.0).at[-2, :, :].set(0.0),
        lambda f: f,
        operand=Ez
    )

    ### y-boundary conducting: zero tangential Ex, Ez ###
    Ex = lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 1, :].set(0.0).at[:, -2, :].set(0.0),
        lambda f: f,
        operand=Ex
    )
    Ez = lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 1, :].set(0.0).at[:, -2, :].set(0.0),
        lambda f: f,
        operand=Ez
    )

    ### z-boundary conducting: zero tangential Ex, Ey ###
    Ex = lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 1].set(0.0).at[:, :, -2].set(0.0),
        lambda f: f,
        operand=Ex
    )
    Ey = lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 1].set(0.0).at[:, :, -2].set(0.0),
        lambda f: f,
        operand=Ey
    )

    return (Ex, Ey, Ez)


@jit
def apply_scalar_conducting_bc(field, bc_x, bc_y, bc_z):
    """
    Enforce conducting boundary conditions on a scalar field (e.g. phi).

    Sets the field to zero at each conducting boundary face.

    Args:
        field (ndarray): 3D scalar field with shape (Nx+2, Ny+2, Nz+2).
        bc_x, bc_y, bc_z (int): Boundary condition codes per axis (0=periodic, 1=conducting).

    Returns:
        ndarray: Scalar field with conducting BCs applied.
    """
    field = lax.cond(
        bc_x == BC_CONDUCTING,
        lambda f: f.at[1, :, :].set(0.0).at[-2, :, :].set(0.0),
        lambda f: f,
        operand=field
    )
    field = lax.cond(
        bc_y == BC_CONDUCTING,
        lambda f: f.at[:, 1, :].set(0.0).at[:, -2, :].set(0.0),
        lambda f: f,
        operand=field
    )
    field = lax.cond(
        bc_z == BC_CONDUCTING,
        lambda f: f.at[:, :, 1].set(0.0).at[:, :, -2].set(0.0),
        lambda f: f,
        operand=field
    )
    return field


def apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength):
    """
    Apply Super-Gaussian absorbing boundary conditions to the given field.

    Args:
        field (ndarray): The field to which the Super-Gaussian boundary condition is applied.
        boundary_thickness (int): The thickness of the absorbing boundary layer.
        order (int): The order of the Super-Gaussian function.
        strength (float): The strength of the absorption.

    Returns:
        ndarray: The field with Super-Gaussian boundary conditions applied.
    """
    def supergaussian_factor(x, thickness, order, strength):
        return jnp.exp(-strength * (x / thickness)**order)

    nx, ny, nz = field.shape
    for i in range(boundary_thickness):
        factor = supergaussian_factor(i, boundary_thickness, order, strength)
        field = field.at[i, :, :].mul(factor)
        field = field.at[nx - 1 - i, :, :].mul(factor)
        field = field.at[:, i, :].mul(factor)
        field = field.at[:, ny - 1 - i, :].mul(factor)
        field = field.at[:, :, i].mul(factor)
        field = field.at[:, :, nz - 1 - i].mul(factor)

    return field

@jit
def apply_zero_boundary_condition(field):
    """
    Apply zero boundary conditions to the given field.

    Args:
        field (ndarray): The field to which the zero boundary condition is applied.

    Returns:
        ndarray: The field with zero boundary conditions applied.
    """
    field = field.at[0, :, :].set(0)
    field = field.at[-1, :, :].set(0)
    field = field.at[:, 0, :].set(0)
    field = field.at[:, -1, :].set(0)
    field = field.at[:, :, 0].set(0)
    field = field.at[:, :, -1].set(0)

    return field
