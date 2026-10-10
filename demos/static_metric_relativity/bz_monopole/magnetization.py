"""Positive proper number density and metric-aware staggered-grid magnetization."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import location_interpolate as _location_interpolate


class Magnetization(NamedTuple):
    number_density: jax.Array  # (species, tile_x, tile_y, tile_z, r, theta, phi)
    magnetic_squared: jax.Array
    sigma: jax.Array
    valid: jax.Array


def deposit_number_density(particles, species, template, metric, static, dynamic):
    """Reuse production shape deposition/halo folding with unit species charge."""
    result = []
    for s in range(species.charge.shape[0]):
        config = species._replace(charge=jnp.arange(species.charge.shape[0]) == s)
        conformal = compute_rho(particles, config, template, static, dynamic)
        result.append(conformal / metric.center.sqrt_gamma)
    return jnp.stack(result)


def collocate_magnetic_field(B):
    """Collocate physical components at cell centers.

    Never contract components sampled at different Yee positions.
    """
    return jnp.stack([_location_interpolate(B[i], loc, ("C",)*3)
                      for i, loc in enumerate(B_FIELD_LOCATIONS)], axis=-1)


def magnetization_from_density(B, number_density, masses, metric):
    field = collocate_magnetic_field(B)
    b2 = jnp.einsum("...i,...ij,...j->...", field, metric.center.gamma, field)
    mass_density = jnp.einsum("s,s...->...", masses, number_density)
    valid = (jnp.all(jnp.isfinite(number_density) & (number_density >= 0), axis=0)
             & jnp.isfinite(b2) & (b2 >= 0))
    sigma = jnp.where(mass_density > 0, b2/(4*jnp.pi*jnp.where(mass_density>0, mass_density, 1)),
                      jnp.where(b2 > 0, jnp.inf, 0.))
    return Magnetization(number_density, b2, jnp.where(valid, sigma, jnp.nan), valid)


def measure_magnetization(particles, species, B, metric, static, dynamic):
    n = deposit_number_density(particles, species, jnp.zeros_like(B[0]), metric, static, dynamic)
    return magnetization_from_density(B, n, species.mass, metric)