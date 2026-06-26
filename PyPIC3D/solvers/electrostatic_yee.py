import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.rho_tiled import compute_tiled_rho_from_tiled_particles
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient
from PyPIC3D.utils import digital_filter
from PyPIC3D.boundary_conditions.boundaryconditions import (
    update_ghost_cells, apply_scalar_conducting_bc
)
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_scalar_field,
    tile_scalar_field,
    update_tiled_ghost_cells,
    update_tiled_vector_ghost_cells,
)


@partial(jit, static_argnames=("tol", "max_iter"))
def solve_poisson_with_conjugate_gradient(rho, phi, constants, world, tol=1e-12, max_iter=5000):
    """
    Solve Poisson's equation using matrix-free conjugate gradient.

    Uses ghost-cell slicing for the Laplacian stencil instead of jnp.roll.

    Args:
        rho (ndarray): Charge density field with shape (Nx+2, Ny+2, Nz+2).
        phi (ndarray): Initial guess for potential with shape (Nx+2, Ny+2, Nz+2).
        constants (dict): Physical constants with key ``eps``.
        world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.
        tol (float): Residual tolerance.
        max_iter (int): Maximum number of CG iterations.

    Returns:
        ndarray: Electrostatic potential field with shape (Nx+2, Ny+2, Nz+2).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
    # get the boundary condition codes for each axis

    def lapl(field):
        # Laplacian using ghost-cell neighbors instead of jnp.roll
        dfdx2 = (field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1]) / (dx * dx)
        dfdy2 = (field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1]) / (dy * dy)
        dfdz2 = (field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] - 2.0 * field[1:-1, 1:-1, 1:-1]) / (dz * dz)
        return dfdx2 + dfdy2 + dfdz2
    # compute the laplacian on the interior using ghost cell neighbors

    def apply_bc(field):
        field = apply_scalar_conducting_bc(field, bc_x, bc_y, bc_z)
        field = update_ghost_cells(field, bc_x, bc_y, bc_z)
        return field
    # apply boundary conditions and update ghost cells

    def body_fun(state):
        phi, r, p, k = state

        lapl_p = -lapl(p)
        alpha = jnp.sum(r * r) / jnp.sum(p[1:-1, 1:-1, 1:-1] * lapl_p)
        # compute the optimal step size in the direction of the residual
        phi_next = phi.at[1:-1, 1:-1, 1:-1].add(alpha * p[1:-1, 1:-1, 1:-1])
        # update the potential guess
        phi_next = apply_bc(phi_next)
        # apply boundary conditions to the new potential guess

        r_next = r - alpha * lapl_p
        # compute the new residual after stepping in the direction of p
        beta = jnp.sum(r_next * r_next) / jnp.sum(r * r)
        # compute the optimal scaling for the new search direction
        p_next = p.at[1:-1, 1:-1, 1:-1].set(r_next + beta * p[1:-1, 1:-1, 1:-1])
        p_next = apply_bc(p_next)
        # compute the new search direction

        return phi_next, r_next, p_next, k + 1
    # perform one iteration of the conjugate gradient method

    def cond_fun(state):
        phi, r, p, k = state

        norm_r = jnp.sum(r * r)
        # compute the squared L2 norm of the residual for convergence checking
        return jnp.logical_and(k < max_iter, norm_r > tol**2)
    # if the residual norm is above the tolerance and we haven't exceeded max iterations, continue iterating

    phi = apply_bc(phi)
    residual = rho[1:-1, 1:-1, 1:-1] / eps + lapl(phi)
    # compute the error in the Poisson equation with the current potential guess

    p0 = jnp.zeros_like(phi)
    p0 = p0.at[1:-1, 1:-1, 1:-1].set(residual)
    p0 = apply_bc(p0)
    # initialize the search direction with ghost cells

    phi, residual, p, iteration = lax.while_loop(
        cond_fun,
        body_fun,
        (phi, residual, p0, 0),
    )
    # run the conjugate gradient iterations until convergence or max iterations reached

    return apply_bc(phi)


@partial(jit, static_argnames=("solver", "bc"))
def calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, bc):
    """
    Compute electrostatic fields from charge deposition and Poisson solve.

    All field arrays have shape (Nx+2, Ny+2, Nz+2) with ghost cells.

    Args:
        world (dict): Simulation world dictionary.
        particles (list): Particle species list.
        constants (dict): Physical constants.
        rho (ndarray): Charge density array with shape (Nx+2, Ny+2, Nz+2).
        phi (ndarray): Potential array with shape (Nx+2, Ny+2, Nz+2).
        solver (str): Electrostatic solver mode.
        bc (str): Boundary condition for finite-difference gradient.

    Returns:
        tuple: ``((Ex, Ey, Ez), phi, rho)``.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']

    rho = compute_rho(particles, rho, world, constants)

    phi = solve_poisson_with_conjugate_gradient(rho, phi, constants, world)

    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # refresh ghost cells before any stencil-based post-processing

    alpha = constants['alpha']
    phi = digital_filter(phi, alpha)
    phi = apply_scalar_conducting_bc(phi, bc_x, bc_y, bc_z)
    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # update ghost cells after filtering

    Ex, Ey, Ez = centered_finite_difference_gradient(-1 * phi[1:-1, 1:-1, 1:-1], dx, dy, dz, bc)
    # compute gradient on the interior

    # Place the gradient results into ghost-celled arrays
    Ex_full = jnp.zeros_like(phi)
    Ey_full = jnp.zeros_like(phi)
    Ez_full = jnp.zeros_like(phi)
    Ex_full = Ex_full.at[1:-1, 1:-1, 1:-1].set(Ex)
    Ey_full = Ey_full.at[1:-1, 1:-1, 1:-1].set(Ey)
    Ez_full = Ez_full.at[1:-1, 1:-1, 1:-1].set(Ez)
    Ex_full = update_ghost_cells(Ex_full, bc_x, bc_y, bc_z)
    Ey_full = update_ghost_cells(Ey_full, bc_x, bc_y, bc_z)
    Ez_full = update_ghost_cells(Ez_full, bc_x, bc_y, bc_z)

    return (Ex_full, Ey_full, Ez_full), phi, rho


def _centered_tiled_electrostatic_gradient(phi_tiles, world):
    """
    Compute ``E = -grad(phi)`` on compact scalar tiles.

    The potential halos must already contain neighboring tile/global boundary
    values.  This mirrors ``centered_finite_difference_gradient`` on the
    assembled physical interior, then refreshes vector halos for particle
    interpolation.
    """

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]

    phi_tiles = update_tiled_ghost_cells(phi_tiles, world)

    Ex = jnp.zeros_like(phi_tiles)
    Ey = jnp.zeros_like(phi_tiles)
    Ez = jnp.zeros_like(phi_tiles)

    Ex = Ex.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        -1.0 * (phi_tiles[:, :, :, 2:, 1:-1, 1:-1] - phi_tiles[:, :, :, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    )
    Ey = Ey.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        -1.0 * (phi_tiles[:, :, :, 1:-1, 2:, 1:-1] - phi_tiles[:, :, :, 1:-1, :-2, 1:-1]) / (2.0 * dy)
    )
    Ez = Ez.at[:, :, :, 1:-1, 1:-1, 1:-1].set(
        -1.0 * (phi_tiles[:, :, :, 1:-1, 1:-1, 2:] - phi_tiles[:, :, :, 1:-1, 1:-1, :-2]) / (2.0 * dz)
    )

    return update_tiled_vector_ghost_cells((Ex, Ey, Ez), world)


def calculate_tiled_electrostatic_fields(world, particles, constants, rho_tiles, phi_tiles, solver, bc, tile_shape):
    """
    Compute electrostatic fields from tiled rho deposition and a global Poisson solve.

    Rho is deposited into tile-major scalar storage, assembled for the existing
    Poisson solver, and the solved global potential is tiled again before the
    electric field is differentiated on tile-local halos.
    """

    del bc

    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    rho_tiles = compute_tiled_rho_from_tiled_particles(particles, rho_tiles, world, constants)
    rho = assemble_tiled_scalar_field(rho_tiles, world, tile_shape)
    phi = assemble_tiled_scalar_field(phi_tiles, world, tile_shape)

    phi = solve_poisson_with_conjugate_gradient(rho, phi, constants, world)

    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # refresh ghost cells before filtering and tiled differentiation

    alpha = constants["alpha"]
    phi = digital_filter(phi, alpha)
    phi = apply_scalar_conducting_bc(phi, bc_x, bc_y, bc_z)
    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # keep the same global phi post-processing order as the untiled solver

    phi_tiles = tile_scalar_field(phi, world, tile_shape)
    E_tiles = _centered_tiled_electrostatic_gradient(phi_tiles, world)

    return E_tiles, phi_tiles, rho_tiles
