import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient
from PyPIC3D.solvers.pstd import spectral_gradient
from PyPIC3D.utils import digital_filter
from PyPIC3D.boundary_conditions.boundaryconditions import (
    update_ghost_cells, apply_scalar_conducting_bc
)


@jit
def solve_poisson_with_fft(rho, constants, world):
    """
    Solve Poisson's equation in a periodic domain using FFTs.

    Operates on the interior of the ghost-celled rho array and writes the
    result back into the interior of phi.

    Args:
        rho (ndarray): Charge density field with shape (Nx+2, Ny+2, Nz+2).
        constants (dict): Physical constants with key ``eps``.
        world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.

    Returns:
        ndarray: Electrostatic potential field with shape (Nx+2, Ny+2, Nz+2).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    rho_interior = rho[1:-1, 1:-1, 1:-1]
    # extract the physical interior for the FFT solve

    rho_hat = jnp.fft.fftn(rho_interior)
    nx, ny, nz = rho_hat.shape

    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')

    k_squared = kx**2 + ky**2 + kz**2
    k_squared = k_squared.at[0, 0, 0].set(1.0)

    phi_hat = rho_hat / (eps * k_squared)
    phi_hat = phi_hat.at[0, 0, 0].set(0.0)
    phi_interior = jnp.fft.ifftn(phi_hat).real

    phi = jnp.zeros_like(rho)
    phi = phi.at[1:-1, 1:-1, 1:-1].set(phi_interior)
    # write the solution back into the interior of the ghost-celled array

    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # fill ghost cells for the potential

    return phi


@partial(jit, static_argnames=("tol", "max_iter"))
def solve_poisson_with_conjugate_gradient(rho, phi, constants, world, tol=1e-6, max_iter=5000):
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

        lapl_p = lapl(p)
        alpha = -1*jnp.sum(r * r) / jnp.sum(p[1:-1, 1:-1, 1:-1] * lapl_p)
        # compute the optimal step size in the direction of the residual
        phi_next = phi.at[1:-1, 1:-1, 1:-1].add(alpha * p[1:-1, 1:-1, 1:-1])
        # update the potential guess
        phi_next = apply_bc(phi_next)
        # apply boundary conditions to the new potential guess

        r_next = r + alpha * lapl_p
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

    phi = lax.cond(
        solver == "spectral",
        lambda _: solve_poisson_with_fft(rho, constants, world),
        lambda _: solve_poisson_with_conjugate_gradient(rho, phi, constants, world),
        operand=None,
    )

    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # refresh ghost cells before any stencil-based post-processing

    alpha = constants['alpha']
    phi = digital_filter(phi, alpha)
    phi = apply_scalar_conducting_bc(phi, bc_x, bc_y, bc_z)
    phi = update_ghost_cells(phi, bc_x, bc_y, bc_z)
    # update ghost cells after filtering

    Ex, Ey, Ez = lax.cond(
        solver == "spectral",
        lambda _: spectral_gradient(-1 * phi[1:-1, 1:-1, 1:-1], world),
        lambda _: centered_finite_difference_gradient(-1 * phi[1:-1, 1:-1, 1:-1], dx, dy, dz, bc),
        operand=None,
    )
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
