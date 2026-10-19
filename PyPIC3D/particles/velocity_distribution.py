"""Thermal three-velocity sampling in the plasma rest frame."""

from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.utilities.plasma_quantities import T_to_vth


@partial(jax.jit, static_argnames=("N_particles",))
def maxwellian(N_particles, mass, Tx, Ty, Tz, kb, key):
    """Return ``(vx, vy, vz)`` with component variances ``kb*T_i/mass``.

    Temperatures are nonnegative; mass and kb are positive. Each returned
    array has shape ``(N_particles,)``. The caller supplies a fresh JAX key.
    """
    key_x, key_y, key_z = jax.random.split(key, 3)
    dtype = jnp.result_type(mass, Tx, Ty, Tz, kb, 1.0)

    std_x = T_to_vth(Tx, mass, kb)
    std_y = T_to_vth(Ty, mass, kb)
    std_z = T_to_vth(Tz, mass, kb)
    vx = std_x * jax.random.normal(key_x, (N_particles,), dtype=dtype)
    vy = std_y * jax.random.normal(key_y, (N_particles,), dtype=dtype)
    vz = std_z * jax.random.normal(key_z, (N_particles,), dtype=dtype)

    return vx, vy, vz


def _sample_kinetic_energy(key, theta):
    """Draw x = gamma - 1 with the modified Canfield four-Gamma envelope.

    See Zenitani & Nakano (2022), section IV:
    https://arxiv.org/abs/2210.15118
    """
    a, b = 0.56, 0.35
    sqrt_pi = jnp.sqrt(jnp.pi)
    sqrt_two = jnp.sqrt(2.0)

    # Integrated envelope weights for shapes 3/2, 2, 5/2, 3, all at scale theta.
    weights = jnp.array([
        sqrt_pi,
        a * jnp.sqrt(2.0 * theta),
        1.5 * b * sqrt_pi * theta,
        (2.0 * theta) ** 1.5,
    ], dtype=theta.dtype)
    cumulative_probability = jnp.cumsum(weights / jnp.sum(weights))
    gamma_shapes = jnp.array([1.5, 2.0, 2.5, 3.0], dtype=theta.dtype)

    def attempt(state):
        key, _, _ = state
        key, mixture_key, gamma_key, acceptance_key = jax.random.split(key, 4)
        mixture_uniform = jax.random.uniform(mixture_key, dtype=theta.dtype)
        component = jnp.sum(mixture_uniform >= cumulative_probability[:-1])
        x = theta * jax.random.gamma(gamma_key, gamma_shapes[component], dtype=theta.dtype)

        sqrt_x = jnp.sqrt(x)
        acceptance_probability = (
            (1.0 + x) * jnp.sqrt(x + 2.0)
            / (sqrt_two + a * sqrt_x + b * sqrt_two * x + x * sqrt_x)
        )
        accepted = jax.random.uniform(acceptance_key, dtype=theta.dtype) < acceptance_probability
        return key, x, accepted

    # The cold distribution is a point mass at zero and needs no proposals.
    initial_state = key, jnp.zeros_like(theta), theta == 0
    _, x, _ = jax.lax.while_loop(lambda state: ~state[2], attempt, initial_state)
    return x


@partial(jax.jit, static_argnames=("N_particles",))
def maxwell_juttner(theta, N_particles, key, c=1.0):
    """Return isotropic Maxwell-Juttner three-velocities ``(vx, vy, vz)``.

    ``theta = kb*T/(mass*c**2)`` is a nonnegative scalar, and c is positive.
    Each component has shape ``(N_particles,)`` and units set by c. Sampling
    uses four-Gamma rejection, without a momentum cutoff or inverse-CDF table.
    These are rest-frame three-velocities, not covariant four-velocities.
    The caller supplies a fresh JAX key; zero theta returns zero velocities.
    """
    theta = jnp.asarray(theta, dtype=jnp.result_type(theta, c, 1.0))
    momentum_key, mu_key, azimuth_key = jax.random.split(key, 3)
    particle_keys = jax.random.split(momentum_key, N_particles)
    x = jax.vmap(_sample_kinetic_energy, in_axes=(0, None))(particle_keys, theta)

    # p_hat = |p|/(m*c); using x avoids cancellation in gamma**2 - 1.
    p_hat = jnp.sqrt(x * (x + 2.0))
    speed = c * p_hat / (1.0 + x)

    mu = jax.random.uniform(mu_key, (N_particles,), dtype=theta.dtype, minval=-1.0, maxval=1.0)
    azimuth = jax.random.uniform(azimuth_key, (N_particles,), dtype=theta.dtype, maxval=2.0 * jnp.pi)
    sin_polar = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu**2))
    vx = speed * sin_polar * jnp.cos(azimuth)
    vy = speed * sin_polar * jnp.sin(azimuth)
    vz = speed * mu

    return vx, vy, vz
