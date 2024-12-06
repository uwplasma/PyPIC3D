import jax
import jax.numpy as jnp


#### DISCLAIMER: I have not tested this method and I have a gut feeling that it will not work as intended.

def solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=1.5, tol=1e-6, max_iter=10000):
    """
    Solve Poisson's equation using Successive Over-Relaxation (SOR) method.

    Parameters:
    phi (jax.numpy.ndarray): Initial guess for the potential.
    rho (jax.numpy.ndarray): Charge density.
    omega (float): Relaxation factor.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.

    Returns:
    jax.numpy.ndarray: Solution for the potential.
    """
    phi = jnp.array(phi)
    rho = jnp.array(rho)
    b = dx * dy * dz / jnp.pi

    for _ in range(max_iter):
        phi_old = phi.copy()
        phi = phi.at[1:-1, 1:-1, 1:-1].set(
            (1 - omega) * phi[1:-1, 1:-1, 1:-1] + omega / 6 * (
                phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
                phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
                phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:] -
                b * rho[1:-1, 1:-1, 1:-1] / eps
            )
        )
        if jnp.linalg.norm(phi - phi_old) < tol:
            break

    return phi