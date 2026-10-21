"""Exact local relaxation about a fixed native density, with midpoint forcing."""

import jax.numpy as jnp


def sponge_advance(value, rhs, dt, sigma, reference):
    """Integrate dF/dt = rhs - sigma*(F-reference), freezing midpoint rhs.

    This exponential midpoint step is second order for smooth forcing. The
    zero-rate branch retains the original Yee update, including its rounding.
    Negative dt is supported for initializing staggered field histories.
    """
    decay = jnp.exp(-sigma * dt)
    weight = -jnp.expm1(-sigma * dt) / jnp.where(sigma != 0, sigma, 1.0)
    relaxed = reference + decay * (value - reference) + weight * rhs
    return jnp.where(sigma != 0, relaxed, value + dt * rhs)
