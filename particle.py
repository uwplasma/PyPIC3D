import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK


def initial_particles(N_particles, x_wind, y_wind, z_wind, mass, T, kb, key):
    """
    Initializes the velocities and positions of the particles.

    Parameters:
    - N_particles (int): The number of particles.
    - x_wind (float): The maximum value for the x-coordinate of the particles' positions.
    - y_wind (float): The maximum value for the y-coordinate of the particles' positions.
    - z_wind (float): The maximum value for the z-coordinate of the particles' positions.
    - mass (float): The mass of the particles.
    - T (float): The temperature of the system.
    - kb (float): The Boltzmann constant.
    - key (jax.random.PRNGKey): The random key for generating random numbers.

    Returns:
    - x (jax.numpy.ndarray): The x-coordinates of the particles' positions.
    - y (jax.numpy.ndarray): The y-coordinates of the particles' positions.
    - z (jax.numpy.ndarray): The z-coordinates of the particles' positions.
    - v_x (numpy.ndarray): The x-component of the particles' velocities.
    - v_y (numpy.ndarray): The y-component of the particles' velocities.
    - v_z (numpy.ndarray): The z-component of the particles' velocities.
    """

    x = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.0025 * x_wind)
    y = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.0025 * y_wind)
    z = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.0025 * z_wind)
    # initialize the positions of the particles
    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z

@jit
def update_position(x, y, z, vx, vy, vz, dt):
    """
    Update the position of the particles.

    Parameters:
    x (float): The current x-coordinate of the particle.
    y (float): The current y-coordinate of the particle.
    z (float): The current z-coordinate of the particle.
    vx (float): The velocity in the x-direction of the particle.
    vy (float): The velocity in the y-direction of the particle.
    vz (float): The velocity in the z-direction of the particle.
    dt (float): The time step for the update.

    Returns:
    tuple: A tuple containing the updated x, y, and z coordinates of the particle.
    """
    
    x = x + vx*dt/2
    y = y + vy*dt/2
    z = z + vz*dt/2
    return x, y, z