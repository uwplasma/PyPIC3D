# Christopher Woolford Jan 17 2025
# This script evolve particle drifts under the influence of a magnetic field

import jax.numpy as jnp
from jax import jit

@jit
def general_drift(particles, Fx, Fy, Fz, Bx, By, Bz):
    """
    Calculate the general drift velocity of particles in the presence of electric and magnetic fields.

    Args:
        particles (Particles): An object representing the particles, which must have methods to get and set charge and velocity.
        Fx (float): The x-component of the electric field.
        Fy (float): The y-component of the electric field.
        Fz (float): The z-component of the electric field.
        Bx (float): The x-component of the magnetic field.
        By (float): The y-component of the magnetic field.
        Bz (float): The z-component of the magnetic field.

    Returns:
        Particles: An object representing the updated particles with new velocities
    """
    q = particles.get_charge()
    # get the charge of the particles
    vd = (1/q) * jnp.cross(jnp.array([Fx, Fy, Fz]), jnp.array([Bx, By, Bz])) / (Bx**2 + By**2 + Bz**2)
    # calculate the drift velocity

    particles.set_velocity(particles.get_velocity() + vd)
    # update the particle velocity

    return particles