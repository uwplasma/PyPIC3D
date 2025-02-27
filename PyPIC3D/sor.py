import jax
import jax.numpy as jnp


#### DISCLAIMER: I have not tested this method and I have a gut feeling that it will not work as intended.

#@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=1.5, tol=1e-6, max_iter=10000):
    """
    Solve Poisson's equation using Successive Over-Relaxation (SOR) method.

    Args:
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

    def cond_fun(val):
        _, norm_diff, iter_count = val
        return (norm_diff >= tol) & (iter_count < max_iter)

    def body_fun(val):
        phi, _, iter_count = val
        phi_old = phi.copy()
        phi = phi.at[1:-1, 1:-1, 1:-1].set(
            (1 - omega) * phi[1:-1, 1:-1, 1:-1] + omega / 6 * (
                phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
                phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
                phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:] -
                b * rho[1:-1, 1:-1, 1:-1] / eps
            )
        )
        return phi, jnp.linalg.norm(phi - phi_old), iter_count + 1

    phi, _, _ = jax.lax.while_loop(cond_fun, body_fun, (phi, jnp.inf, 0))

    return phi