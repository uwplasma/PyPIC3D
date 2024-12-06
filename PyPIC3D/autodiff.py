
# Christopher Woolford Dec 5, 2024
# This script contains various functions for auto-differentiation in the 3D PIC code using Jax.

import jax
import jax.numpy as jnp

def kinetic_energy_grad(species):
    """
    Compute the gradient of the kinetic energy with respect to the particle velocities.

    Returns:
    - grad_v1, grad_v2, grad_v3 (tuple): The gradients of the kinetic energy with respect to v1, v2, and v3.
    """
    mass = species.get_mass()
    v1, v2, v3 = species.get_velocity()
    ke_fn = lambda v1, v2, v3: 0.5 * mass * jnp.sum(v1**2 + v2**2 + v3**2)
    grad_fn = jax.grad(ke_fn, argnums=(0, 1, 2))
    grad_v1, grad_v2, grad_v3 = grad_fn(v1, v2, v3)
    return grad_v1, grad_v2, grad_v3