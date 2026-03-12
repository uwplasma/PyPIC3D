import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient
from PyPIC3D.solvers.pstd import spectral_gradient
from PyPIC3D.utils import digital_filter

BC_PERIODIC = 0
BC_CONDUCTING = 1


def _get_field_bc_code(world, axis):
    """
    Get the boundary condition code for an axis, supporting legacy and refactored world schemas.
    """
    if 'boundary_conditions' in world and axis in world['boundary_conditions']:
        bc = world['boundary_conditions'][axis]
    else:
        legacy_key = f"{axis}_bc"
        bc = world[legacy_key] if legacy_key in world else BC_PERIODIC

    if isinstance(bc, str):
        return BC_CONDUCTING if bc == "conducting" else BC_PERIODIC
    return bc


@jit
def solve_poisson_with_fft(rho, constants, world):
    """
    Solve Poisson's equation in a periodic domain using FFTs.

    Args:
        rho (ndarray): Charge density field.
        constants (dict): Physical constants with key ``eps``.
        world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.

    Returns:
        ndarray: Electrostatic potential field with zero-mean gauge.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    rho_hat = jnp.fft.fftn(rho)
    nx, ny, nz = rho_hat.shape

    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')

    k_squared = kx**2 + ky**2 + kz**2
    k_squared = k_squared.at[0, 0, 0].set(1.0)

    phi_hat = rho_hat / (eps * k_squared)
    phi_hat = phi_hat.at[0, 0, 0].set(0.0)
    phi = jnp.fft.ifftn(phi_hat).real
    return phi


@partial(jit, static_argnames=("tol", "max_iter"))
def solve_poisson_with_conjugate_gradient(rho, phi, constants, world, tol=1e-6, max_iter=5000):
    """
    Solve Poisson's equation in a periodic domain with matrix-free conjugate gradient.

    The discrete system solved is:
        ``(-∇²) phi = rho / eps``

    Args:
        rho (ndarray): Charge density field.
        phi (ndarray): Initial guess for potential.
        constants (dict): Physical constants with key ``eps``.
        world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.
        tol (float): Residual tolerance.
        max_iter (int): Maximum number of CG iterations.

    Returns:
        ndarray: Electrostatic potential field with zero-mean gauge.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    x_bc = _get_field_bc_code(world, 'x')
    y_bc = _get_field_bc_code(world, 'y')
    z_bc = _get_field_bc_code(world, 'z')
    # get the boundary condition codes for each axis, supporting legacy and refactored world schemas

    def lapl(field):
        dfdx2 = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2.0 * field) / (dx * dx)
        dfdy2 = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2.0 * field) / (dy * dy)
        dfdz2 = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2.0 * field) / (dz * dz)
        return dfdx2 + dfdy2 + dfdz2
    # compute the laplacian of the potential

    def apply_x_bc(field):
        return lax.cond(
            x_bc == BC_CONDUCTING,
            lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
            lambda f: f,
            field
        )

    def apply_y_bc(field):
        return lax.cond(
            y_bc == BC_CONDUCTING,
            lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
            lambda f: f,
            field
        )

    def apply_z_bc(field):
        return lax.cond(
            z_bc == BC_CONDUCTING,
            lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
            lambda f: f,
            field
        )


    def body_fun(state):
        phi, r, p, k = state

        alpha = -1*jnp.sum(r * r) / jnp.sum(p * lapl(p))
        # compute the optimal step size in the direction of the residual
        phi_next = phi + alpha * p
        # update the potential guess
        phi_next = apply_x_bc(phi_next)
        phi_next = apply_y_bc(phi_next)
        phi_next = apply_z_bc(phi_next)
        # apply boundary conditions to the new potential guess

        r_next = r + alpha * lapl(p)
        # compute the new residual after stepping in the direction of p
        beta = jnp.sum(r_next * r_next) / jnp.sum(r * r)
        # compute the optimal scaling for the new search direction
        p_next = r_next + beta * p
        # compute the new search direction as a combination of the new residual and the previous search direction

        return phi_next, r_next, p_next, k + 1
    # perform one iteration of the conjugate gradient method

    def cond_fun(state):
        phi, r, p, k = state

        norm_r = jnp.sum(r * r)
        # compute the squared L2 norm of the residual for convergence checking
        return jnp.logical_and(k < max_iter, norm_r > tol**2)
    # if the residual norm is above the tolerance and we haven't exceeded max iterations, continue iterating


    residual = rho / eps + lapl(phi)
    # compute the error in the Poisson equation with the current potential guess

    phi, residual, p, iteration = lax.while_loop(
        cond_fun,
        body_fun,
        (phi, residual, residual, 0),
    )
    # run the conjugate gradient iterations until convergence or max iterations reached


    return phi


@partial(jit, static_argnames=("solver", "bc"))
def calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, bc):
    """
    Compute electrostatic fields from charge deposition and Poisson solve.

    Args:
        world (dict): Simulation world dictionary.
        particles (list): Particle species list.
        constants (dict): Physical constants.
        rho (ndarray): Charge density array.
        phi (ndarray): Potential array.
        solver (str): Electrostatic solver mode.
        bc (str): Boundary condition for finite-difference gradient.

    Returns:
        tuple: ``((Ex, Ey, Ez), phi, rho)``.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    rho = compute_rho(particles, rho, world, constants)

    phi = lax.cond(
        solver == "spectral",
        lambda _: solve_poisson_with_fft(rho, constants, world),
        lambda _: solve_poisson_with_conjugate_gradient(rho, phi, constants, world),
        operand=None,
    )

    alpha = constants['alpha']
    phi = digital_filter(phi, alpha)

    Ex, Ey, Ez = lax.cond(
        solver == "spectral",
        lambda _: spectral_gradient(-1 * phi, world),
        lambda _: centered_finite_difference_gradient(-1 * phi, dx, dy, dz, bc),
        operand=None,
    )

    return (Ex, Ey, Ez), phi, rho
