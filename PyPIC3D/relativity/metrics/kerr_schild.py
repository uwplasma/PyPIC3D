from functools import partial

import jax.numpy as jnp

from PyPIC3D.relativity.core import build_yee_metric


def _kerr_schild_cartesian_metric_at_position(position, mass=1.0, spin=0.0):
    x, y, z = position
    dtype = x.dtype
    eye = jnp.eye(3, dtype=dtype)

    a = spin
    rho_squared = x**2 + y**2 + z**2
    r_squared = 0.5 * (
        rho_squared
        - a**2
        + jnp.sqrt((rho_squared - a**2) ** 2 + 4.0 * a**2 * z**2)
    )
    r = jnp.sqrt(r_squared)

    denominator = r_squared + a**2
    ell = jnp.stack(
        (
            (r * x + a * y) / denominator,
            (r * y - a * x) / denominator,
            z / r,
        ),
        axis=-1,
    )

    H = mass * r**3 / (r**4 + a**2 * z**2)
    two_H = 2.0 * H
    lapse = 1.0 / jnp.sqrt(1.0 + two_H)
    shift = (two_H / (1.0 + two_H)) * ell

    inverse_factor = two_H / (1.0 + two_H)
    gamma = eye + two_H * ell[:, jnp.newaxis] * ell[jnp.newaxis, :]
    gamma_inv = eye - inverse_factor * ell[:, jnp.newaxis] * ell[jnp.newaxis, :]
    sqrt_gamma = jnp.sqrt(1.0 + two_H)

    return lapse, shift, gamma, gamma_inv, sqrt_gamma


def _kerr_schild_spherical_metric_at_position(position, mass=1.0, spin=0.0):
    r, theta, _ = position
    dtype = r.dtype

    a = spin
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    rho_squared = r**2 + a**2 * cos_theta**2
    xi = 1.0 + 2.0 * mass * r / rho_squared

    lapse = xi**-0.5
    shift = jnp.zeros(3, dtype=dtype)
    shift = shift.at[0].set((xi - 1.0) / xi)

    gamma = jnp.zeros((3, 3), dtype=dtype)
    gamma = gamma.at[0, 0].set(xi)
    gamma = gamma.at[1, 1].set(rho_squared)
    gamma = gamma.at[2, 2].set(
        sin_theta**2 * (rho_squared + a**2 * xi * sin_theta**2)
    )
    gamma = gamma.at[0, 2].set(-a * xi * sin_theta**2)
    gamma = gamma.at[2, 0].set(gamma[0, 2])

    gamma_inv = jnp.zeros((3, 3), dtype=dtype)
    gamma_inv = gamma_inv.at[0, 0].set(1.0 / xi + a**2 * sin_theta**2 / rho_squared)
    gamma_inv = gamma_inv.at[1, 1].set(1.0 / rho_squared)
    gamma_inv = gamma_inv.at[2, 2].set(1.0 / (rho_squared * sin_theta**2))
    gamma_inv = gamma_inv.at[0, 2].set(a / rho_squared)
    gamma_inv = gamma_inv.at[2, 0].set(gamma_inv[0, 2])

    sqrt_gamma = rho_squared * jnp.sqrt(xi) * sin_theta

    return lapse, shift, gamma, gamma_inv, sqrt_gamma


def initialize_kerr_schild_cartesian_metric(static_parameters, dynamic_parameters, mass=1.0, spin=0.0):
    """
    Build the ingoing Cartesian Kerr-Schild 3+1 metric on the tiled Yee grid.
    """

    del static_parameters
    return build_yee_metric(
        dynamic_parameters,
        partial(_kerr_schild_cartesian_metric_at_position, mass=mass, spin=spin),
    )


def initialize_kerr_schild_spherical_metric(static_parameters, dynamic_parameters, mass=1.0, spin=0.0):
    """
    Build the spherical Kerr-Schild 3+1 metric on the tiled Yee grid.
    """

    del static_parameters
    return build_yee_metric(
        dynamic_parameters,
        partial(_kerr_schild_spherical_metric_at_position, mass=mass, spin=spin),
    )
