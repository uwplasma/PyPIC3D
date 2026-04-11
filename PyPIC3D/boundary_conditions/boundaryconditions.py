# Christopher Woolford, April 6th 2026
# Field-level boundary condition operations for ghost-celled 3D arrays.

from jax import jit, lax
import jax.numpy as jnp
# import external libraries

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
# import internal modules


@jit
def update_ghost_cells(field, bc_x, bc_y, bc_z):
    """
    Fill ghost cells of a 3D field from its interior based on boundary conditions.

    Ghost cells provide the halo data needed by finite-difference stencils
    that reach across the domain boundary. For periodic BCs, the ghost
    layer mirrors the opposite interior edge. For conducting BCs, ghost
    cells are zeroed (Dirichlet condition).

    The field is assumed to have shape (Nx+2, Ny+2, Nz+2) with one ghost
    cell on each side per axis. Indices [0] and [-1] are ghost cells;
    indices [1] and [-2] are the first and last interior cells.

    Args:
        field (jnp.ndarray): 3D field array with shape (Nx+2, Ny+2, Nz+2).
        bc_x (int): Boundary condition code for the x-axis (BC_PERIODIC or BC_CONDUCTING).
        bc_y (int): Boundary condition code for the y-axis.
        bc_z (int): Boundary condition code for the z-axis.

    Returns:
        jnp.ndarray: Field with ghost cells filled.
    """
    # x-axis ghost cells
    # Periodic: ghost[0] <- interior[-2], ghost[-1] <- interior[1]
    # Conducting: ghost[0] = ghost[-1] = 0
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
    Fold ghost cell deposits back into the interior after particle deposition.

    During charge/current deposition, particles near the boundary may
    deposit into ghost cells. This function adds those ghost deposits to
    the corresponding interior cells and then clears the ghost layer.

    For periodic BCs, ghost deposits are added to the opposite interior
    edge (ghost[0] -> interior[1], ghost[-1] -> interior[-2]).

    For conducting BCs with reflecting particles, ghost deposits are reflected 
    back into the domain with opposite sign.

    In both cases, ghost cells are cleared to zero after folding.

    Args:
        field (jnp.ndarray): 3D field array with shape (Nx+2, Ny+2, Nz+2).
        bc_x (int): Boundary condition code for the x-axis (BC_PERIODIC or BC_CONDUCTING).
        bc_y (int): Boundary condition code for the y-axis.
        bc_z (int): Boundary condition code for the z-axis.

    Returns:
        jnp.ndarray: Field with ghost cell deposits folded back into the interior.
    """
    # x-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[1, :, :].add(f[-1, :, :]).at[-2, :, :].add(f[0, :, :]),
        # add the ghost deposits to the opposite inner cell.
        lambda f: f.at[1, :, :].add(-1 * f[0, :, :]).at[-2, :, :].add(-1 * f[-1, :, :]),
        # for conducting BCs, the ghost deposits should be reflected back with opposite sign.
        operand=field
    )
    # if its periodic, the ghost deposits should be added to the other side.
    # if its conducting, the ghost deposits should be reflected back with opposite sign.
    # Clear x ghost cells after folding (both periodic and conducting)
    field = field.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)

    # y-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 1, :].add(f[:, -1, :]).at[:, -2, :].add(f[:, 0, :]),
        lambda f: f.at[:, 1, :].add(-1 * f[:, 0, :]).at[:, -2, :].add(-1 * f[:, -1, :]),
        operand=field
    )
    # Clear y ghost cells after folding
    field = field.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)

    # z-axis: fold ghost deposits back to interior
    field = lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 1].add(f[:, :, -1]).at[:, :, -2].add(f[:, :, 0]),
        lambda f: f.at[:, :, 1].add(-1 * f[:, :, 0]).at[:, :, -2].add(-1 * f[:, :, -1]),
        operand=field
    )
    # Clear z ghost cells after folding
    field = field.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    return field


@jit
def apply_conducting_bc(E, bc_x, bc_y, bc_z):
    """
    Enforce conducting boundary conditions by zeroing tangential E at boundaries.

    On a perfectly conducting wall, the tangential electric field must vanish.
    On a ghost-celled (Nx+2, Ny+2, Nz+2) grid, the first physical face is
    at index [1] and the last at index [-2].

    For a conducting wall normal to a given axis, the two tangential
    components are zeroed at both boundary faces along that axis:
        - Wall normal to x: zero Ey, Ez at x-boundaries [1] and [-2].
        - Wall normal to y: zero Ex, Ez at y-boundaries [1] and [-2].
        - Wall normal to z: zero Ex, Ey at z-boundaries [1] and [-2].

    Args:
        E (tuple): Electric field components (Ex, Ey, Ez), each with shape (Nx+2, Ny+2, Nz+2).
        bc_x (int): Boundary condition code for the x-axis (BC_PERIODIC or BC_CONDUCTING).
        bc_y (int): Boundary condition code for the y-axis.
        bc_z (int): Boundary condition code for the z-axis.

    Returns:
        tuple: Electric field components (Ex, Ey, Ez) with conducting BCs applied.
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
    Enforce conducting boundary conditions on a scalar field (e.g. electric potential phi).

    Sets the field to zero at each conducting boundary face. On a
    ghost-celled grid, boundary faces are at interior indices [1] and [-2].

    Args:
        field (jnp.ndarray): 3D scalar field with shape (Nx+2, Ny+2, Nz+2).
        bc_x (int): Boundary condition code for the x-axis (BC_PERIODIC or BC_CONDUCTING).
        bc_y (int): Boundary condition code for the y-axis.
        bc_z (int): Boundary condition code for the z-axis.

    Returns:
        jnp.ndarray: Scalar field with conducting BCs applied.
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


# No @jit: the Python for-loop is unrolled during JAX tracing by any
# JIT-compiled caller, so explicit JIT here would cause tracing errors.
def apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength):
    """
    Apply a super-Gaussian absorbing boundary layer to damp fields near domain edges.

    Multiplies the field by an exponential damping profile that increases
    toward the boundary: exp(-strength * (distance / thickness)^order).
    This is applied symmetrically on all six faces of the 3D domain.

    Args:
        field (jnp.ndarray): 3D field array with shape (Nx, Ny, Nz).
        boundary_thickness (int): Number of cells in the absorbing layer.
        order (int): Exponent of the super-Gaussian profile (higher = sharper transition).
        strength (float): Amplitude of the damping (dimensionless).

    Returns:
        jnp.ndarray: Field with absorbing boundary damping applied.
    """
    def supergaussian_factor(x, thickness, order, strength):
        """Compute the damping factor at distance x from the boundary."""
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
    Zero all six boundary faces of a 3D field.

    Sets the outermost slice on each axis to zero. This is a simple
    Dirichlet-zero condition applied directly to the array boundaries,
    without distinguishing ghost cells from physical cells.

    Args:
        field (jnp.ndarray): 3D field array of any shape.

    Returns:
        jnp.ndarray: Field with all boundary faces set to zero.
    """
    field = field.at[0, :, :].set(0)
    field = field.at[-1, :, :].set(0)
    field = field.at[:, 0, :].set(0)
    field = field.at[:, -1, :].set(0)
    field = field.at[:, :, 0].set(0)
    field = field.at[:, :, -1].set(0)

    return field
