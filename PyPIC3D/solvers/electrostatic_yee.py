import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial

from PyPIC3D.rho import compute_rho
from PyPIC3D.solvers.fdtd import centered_finite_difference_gradient
from PyPIC3D.solvers.pstd import spectral_gradient
from PyPIC3D.utils import digital_filter


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

    rhs = rho / eps
    rhs = rhs - jnp.mean(rhs)
    phi = phi - jnp.mean(phi)

    tiny = jnp.asarray(1e-30, dtype=phi.dtype)
    tolerance_squared = jnp.asarray(tol, dtype=phi.dtype) ** 2

    def apply_negative_laplacian(field):
        laplacian_x = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2.0 * field) / (dx * dx)
        laplacian_y = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2.0 * field) / (dy * dy)
        laplacian_z = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2.0 * field) / (dz * dz)
        return -(laplacian_x + laplacian_y + laplacian_z)

    residual = rhs - apply_negative_laplacian(phi)
    direction = residual
    residual_norm_squared = jnp.sum(residual * residual)

    def cond_fun(state):
        _, _, _, residual_norm_squared, iteration = state
        return jnp.logical_and(iteration < max_iter, residual_norm_squared > tolerance_squared)

    def body_fun(state):
        phi, residual, direction, residual_norm_squared, iteration = state

        operator_direction = apply_negative_laplacian(direction)
        denominator = jnp.sum(direction * operator_direction)
        denominator = jnp.where(jnp.abs(denominator) < tiny, tiny, denominator)

        alpha = residual_norm_squared / denominator
        phi_next = phi + alpha * direction
        residual_next = residual - alpha * operator_direction
        residual_norm_squared_next = jnp.sum(residual_next * residual_next)

        beta_denominator = jnp.where(residual_norm_squared < tiny, tiny, residual_norm_squared)
        beta = residual_norm_squared_next / beta_denominator
        direction_next = residual_next + beta * direction

        return phi_next, residual_next, direction_next, residual_norm_squared_next, iteration + 1

    phi, _, _, _, _ = lax.while_loop(
        cond_fun,
        body_fun,
        (phi, residual, direction, residual_norm_squared, 0),
    )

    phi = phi - jnp.mean(phi)
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
