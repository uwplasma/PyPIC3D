import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.utils import digital_filter, wrap_around
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights


@jit
def compute_rho(particles, rho, world, constants):
    """Compute the charge density (rho) on the vertex grid."""
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    grid = world["grids"]["vertex"]
    Nx, Ny, Nz = rho.shape

    rho = jnp.zeros_like(rho)

    for species in particles:
        shape_factor = species.get_shape()
        q = species.get_charge()
        dq = q / dx / dy / dz
        x, y, z = species.get_position()

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round((x - grid[0][0]) / dx).astype(int),
            operand=None,
        )

        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round((y - grid[1][0]) / dy).astype(int),
            operand=None,
        )

        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round((z - grid[2][0]) / dz).astype(int),
            operand=None,
        )

        deltax = x - (x0 * dx + grid[0][0])
        deltay = y - (y0 * dy + grid[1][0])
        deltaz = z - (z0 * dz + grid[2][0])

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)

        x1 = wrap_around(x0 + 1, Nx)
        y1 = wrap_around(y0 + 1, Ny)
        z1 = wrap_around(z0 + 1, Nz)

        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    rho = rho.at[xpts[i], ypts[j], zpts[k]].add(
                        dq * x_weights[i] * y_weights[j] * z_weights[k], mode="drop"
                    )

    alpha = constants["alpha"]
    rho = digital_filter(rho, alpha)

    return rho