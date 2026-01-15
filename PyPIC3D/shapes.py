from jax import jit
import jax.numpy as jnp


@jit
def get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the second-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

    Sx1 = (1/2) * ((1/2) + (deltax/dx))**2
    Sy1 = (1/2) * ((1/2) + (deltay/dy))**2
    Sz1 = (1/2) * ((1/2) + (deltaz/dz))**2

    Sx_minus1 = (1/2) * ((1/2) - (deltax/dx))**2
    Sy_minus1 = (1/2) * ((1/2) - (deltay/dy))**2
    Sz_minus1 = (1/2) * ((1/2) - (deltaz/dz))**2
    # second order weights

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights

@jit
def get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the first-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = jnp.asarray(1 - deltax / dx)
    Sy0 = jnp.asarray(1 - deltay / dy)
    Sz0 = jnp.asarray(1 - deltaz / dz)

    Sx1 = jnp.asarray(deltax / dx)
    Sy1 = jnp.asarray(deltay / dy)
    Sz1 = jnp.asarray(deltaz / dz)

    Sx_minus1 = jnp.zeros_like(Sx0)
    Sy_minus1 = jnp.zeros_like(Sy0)
    Sz_minus1 = jnp.zeros_like(Sz0)
    # No second-order weights for first-order weighting

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights