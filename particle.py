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
    initial_wind = 0.0025
    # what is the initial window for the particles (as a fraction of spatial window)
    x = jax.random.uniform(key, shape = (N_particles,), minval=-initial_wind*x_wind/2, maxval=initial_wind*x_wind/2)
    y = jax.random.uniform(key, shape = (N_particles,), minval=-initial_wind*y_wind/2, maxval=initial_wind*y_wind/2)
    z = jax.random.uniform(key, shape = (N_particles,), minval=-initial_wind*z_wind/2, maxval=initial_wind*z_wind/2)
    # initialize the positions of the particles
    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z

@jit
def periodic_boundary_condition(x_wind, y_wind, z_wind, x, y, z):
    """
    Implement periodic boundary conditions for the particles.

    Returns:
    - x (jax.numpy.ndarray): The x-coordinates of the particles' positions.
    - y (jax.numpy.ndarray): The y-coordinates of the particles' positions.
    - z (jax.numpy.ndarray): The z-coordinates of the particles' positions.
    """
    x = jnp.where(x > x_wind/2, x - x_wind, x)
    x = jnp.where(x < -x_wind/2, x + x_wind, x)
    y = jnp.where(y > y_wind/2, y - y_wind, y)
    y = jnp.where(y < -y_wind/2, y + y_wind, y)
    z = jnp.where(z > z_wind/2, z - z_wind, z)
    z = jnp.where(z < -z_wind/2, z + z_wind, z)
    return x, y, z    

@jit
def update_position(x, y, z, vx, vy, vz, dt, x_wind, y_wind, z_wind):
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
    # update the position of the particles
    x, y, z = periodic_boundary_condition(x_wind, y_wind, z_wind, x, y, z)
    # apply periodic boundary conditions
    return x, y, z

@jit
def total_KE(m, vx, vy, vz):
    """
    Calculate the total kinetic energy of the particles.

    Parameters:
    - m (float): The mass of the particle.
    - v (jax.numpy.ndarray): The velocity of the particle.

    Returns:
    - float: The total kinetic energy of the particle.
    """
    return 0.5 * m * jnp.sum( vx**2 + vy**2 + vz**2 )