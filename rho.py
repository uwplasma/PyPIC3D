import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

@jit
def index_particles(particle, positions, ds):
    """
    Calculate the index of a particle in a given position array.

    Parameters:
    - particle: int
        The index of the particle.
    - positions (array-like): The position array containing the particle positions.
    - ds: float
        The grid spacing.

    Returns:
    - index: int
        The index of the particle in the position array, rounded down to the nearest integer.
    """
    return (positions.at[particle].get() / ds).astype(int)

@jit
def particle_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind):
    """
    Distributes the charge of a particle to the surrounding grid points.

    Parameters:
    q (float): Charge of the particle.
    x (float): x-coordinate of the particle.
    y (float): y-coordinate of the particle.
    z (float): z-coordinate of the particle.
    rho (ndarray): Charge density array.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    x_wind (float): Window in the x-direction.
    y_wind (float): Window in the y-direction.
    z_wind (float): Window in the z-direction.

    Returns:
    ndarray: Updated charge density array.
    """


    x0, y0, z0 = ((x + x_wind/2)/dx).astype(int), ((y + y_wind/2)/dy).astype(int), ((z + z_wind/2)/dz).astype(int)
    deltax, deltay, deltaz = (x + x_wind/2) - x0*dx, (y + y_wind/2) - y0*dy, (z + z_wind/2) - z0*dz
    # calculate the difference between x and its nearest grid point
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    # calculate the index of the next grid point

    wx = jnp.select( [x0 == 0, deltax == 0, deltax != 0], [0, 0, deltax/( x + x_wind/2 )] )
    wy = jnp.select( [y0 == 0, deltay == 0, deltay != 0], [0, 0, deltay/( y + y_wind/2 )] )
    wz = jnp.select( [z0 == 0, deltaz == 0, deltaz != 0], [0, 0, deltaz/( z + z_wind/2 )] )
    # calculate the weights for the surrounding grid points

    dv = dx*dy*dz
    # calculate the volume of each grid point

    rho = rho.at[x0, y0, z0].add( (q/dv)*(1 - wx)*(1 - wy)*(1 - wz), mode='drop' )
    rho = rho.at[x1, y0, z0].add( (q/dv)*wx*(1 - wy)*(1 - wz)      , mode='drop')
    rho = rho.at[x0, y1, z0].add( (q/dv)*(1 - wx)*wy*(1 - wz)      , mode='drop')
    rho = rho.at[x0, y0, z1].add( (q/dv)*(1 - wx)*(1 - wy)*wz      , mode='drop')
    # distribute the charge of the particle to the surrounding grid points

    return rho

@jit
def update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, x_wind, y_wind, z_wind, rho):
    """
    Update the charge density (rho) based on the positions of particles.
    Parameters:
    Nparticles (int): Number of particles.
    particlex (array-like): Array containing the x-coordinates of particles.
    particley (array-like): Array containing the y-coordinates of particles.
    particlez (array-like): Array containing the z-coordinates of particles.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    q (float): Charge of a single particle.
    x_wind (array-like): Window function in the x-direction.
    y_wind (array-like): Window function in the y-direction.
    z_wind (array-like): Window in the z-direction.
    rho (array-like): Initial charge density array to be updated.
    Returns:
    array-like: Updated charge density array.
    """


    # def addto_rho(particle, rho):
    #     x = particlex.at[particle].get()
    #     y = particley.at[particle].get()
    #     z = particlez.at[particle].get()
    #     rho = particle_weighting(q, x, y, z, rho, dx, dy, dz, x_wind, y_wind, z_wind)
    #     return rho

    def addto_rho(particle, rho):
        x = index_particles(particle, particlex, dx)
        y = index_particles(particle, particley, dy)
        z = index_particles(particle, particlez, dz)
        rho = rho.at[x, y, z].add( q / (dx*dy*dz) )
        return rho
    
    return jax.lax.fori_loop(0, Nparticles-1, addto_rho, rho )
