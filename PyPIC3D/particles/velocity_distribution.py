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
def maxwell_juttner(N_particles, mass, Tx, Ty, Tz, kb, key, c=1.0):
    """Return rest-frame Maxwell-Juttner three-velocities ``(vx, vy, vz)``.

    The first seven arguments match ``maxwellian``: Tx, Ty, Tz are scalar
    physical temperatures; mass, kb, and c are positive finite scalars. Each
    returned component has shape ``(N_particles,)`` and the units of c.
    Supply a fresh JAX key for each sample; use ``jax.vmap`` for batches.

    With theta_i = kb*T_i/(mass*c**2) and theta0 = mean(theta_i), the
    dimensionless momentum u_i = p_i/(mass*c) has density proportional to
    exp(-sqrt(1 + sum(theta0/theta_i * u_i**2))/theta0). The existing
    Canfield four-Gamma rejection method samples an isotropic momentum q at
    theta0, then u_i = q_i*sqrt(theta_i/theta0). Equal temperatures recover
    the isotropic distribution. These directional parameters describe its
    shape, not measured relativistic pressure temperatures.

    Velocities are c*u_i/sqrt(1 + sum(u_i**2)), using the Lorentz factor
    after rescaling. They are not covariant four-velocities. No bulk boost
    or leapfrog centering is applied.

    Temperatures must be positive and finite, except that all three may be
    zero for the cold limit. Invalid temperatures (including partly zero
    inputs) return NaNs, also under JIT and vmap. Extreme temperatures can
    exceed the floating-point range or make speeds round to c; no velocity
    clipping or momentum cutoff is applied.

    See Livadiotis (2016), Ann. Geophys. 34, 1145-1158:
    https://doi.org/10.5194/angeo-34-1145-2016
    """
    if any(jnp.ndim(value) != 0 for value in (mass, Tx, Ty, Tz, kb, c)):
        raise ValueError("mass, Tx, Ty, Tz, kb, and c must be scalars; use jax.vmap for batches")
    dtype = jnp.result_type(mass, Tx, Ty, Tz, kb, c, 1.0)
    temperature = jnp.asarray([Tx, Ty, Tz], dtype=dtype)
    theta = kb * temperature / (mass * c**2)
    theta0 = jnp.mean(theta)

    # Invalid proposals could keep the rejection loop running indefinitely.
    # Route them through the existing cold limit, then mark the output NaN.
    valid_temperature = (
        jnp.all(jnp.isfinite(theta)) & jnp.isfinite(theta0)
        & (jnp.all(theta > 0) | jnp.all(temperature == 0))
    )
    theta = jnp.where(valid_temperature, theta, 0.0)
    theta0 = jnp.where(valid_temperature, theta0, 0.0)

    momentum_key, mu_key, azimuth_key = jax.random.split(key, 3)
    particle_keys = jax.random.split(momentum_key, N_particles)
    x = jax.vmap(_sample_kinetic_energy, in_axes=(0, None))(particle_keys, theta0)

    # p_hat = |p|/(m*c); using x avoids cancellation in gamma**2 - 1.
    p_hat = jnp.sqrt(x * (x + 2.0))

    mu = jax.random.uniform(mu_key, (N_particles,), dtype=theta.dtype, minval=-1.0, maxval=1.0)
    azimuth = jax.random.uniform(azimuth_key, (N_particles,), dtype=theta.dtype, maxval=2.0 * jnp.pi)
    sin_polar = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu**2))
    # Deform momentum in the rest frame before computing the physical gamma.
    scale = jnp.sqrt(theta / jnp.where(theta0 > 0, theta0, 1.0))
    ux = p_hat * sin_polar * jnp.cos(azimuth) * scale[0]
    uy = p_hat * sin_polar * jnp.sin(azimuth) * scale[1]
    uz = p_hat * mu * scale[2]
    gamma = jnp.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    vx = jnp.where(valid_temperature, c * ux / gamma, jnp.nan)
    vy = jnp.where(valid_temperature, c * uy / gamma, jnp.nan)
    vz = jnp.where(valid_temperature, c * uz / gamma, jnp.nan)

    return vx, vy, vz
