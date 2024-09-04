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
# this method initializes the velocties and the positions of the particles
    x = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * x_wind)
    y = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * y_wind)
    z = jax.random.uniform(key, shape = (N_particles,), minval=0, maxval=0.05 * z_wind)
    # initialize the positions of the particles
    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z

@jit
def update_position(x, y, z, vx, vy, vz, dt):
    # update the position of the particles
    x = x + vx*dt/2
    y = y + vy*dt/2
    z = z + vz*dt/2
    return x, y, z