
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

from PyPIC3D.utils import interpolate_and_stagger_field, interpolate_field, use_gpu_if_set
from PyPIC3D.particle import particle_species

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

@use_gpu_if_set
def compute_current_density(particles, Jx, Jy, Jz, world, GPUs):
    """
    Computes the current density for a given set of particles in a simulation world.

    Parameters:
    particles (list): A list of particle species, each containing methods to get the number of particles, 
                      their positions, velocities, and charge.
    Jx, Jy, Jz (numpy.ndarray): The current density arrays to be updated.
    world (dict): A dictionary containing the simulation world parameters such as grid spacing (dx, dy, dz) 
                  and window dimensions (x_wind, y_wind, z_wind).
    GPUs (bool): A flag indicating whether to use GPU acceleration for the computation.

    Returns:
    tuple: The updated current density arrays (Jx, Jy, Jz).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']

    for species in particles:
        N_particles = species.get_number_of_particles()
        charge = species.get_charge()
        if N_particles > 0:
            particle_x, particle_y, particle_z = species.get_position()
            particle_vx, particle_vy, particle_vz = species.get_velocity()
            Jx, Jy, Jz = update_current_density(N_particles, particle_x, particle_y, particle_z, particle_vx, particle_vy, particle_vz, dx, dy, dz, charge, x_wind, y_wind, z_wind, Jx, Jy, Jz, GPUs)
    return Jx, Jy, Jz

@use_gpu_if_set
@jit
def update_current_density(Nparticles, particlex, particley, particlez, particlevx, particlevy, particlevz, dx, dy, dz, q, x_wind, y_wind, z_wind, Jx, Jy, Jz, GPUs=False):
    def addto_J(particle, J):
        Jx, Jy, Jz = J
        x = index_particles(particle, particlex, dx)
        y = index_particles(particle, particley, dy)
        z = index_particles(particle, particlez, dz)
        vx = particlevx.at[particle].get()
        vy = particlevy.at[particle].get()
        vz = particlevz.at[particle].get()
        Jx = Jx.at[x, y, z].add(q * vx / (dx * dy * dz))
        Jy = Jy.at[x, y, z].add(q * vy / (dx * dy * dz))
        Jz = Jz.at[x, y, z].add(q * vz / (dx * dy * dz))
        return Jx, Jy, Jz

    return jax.lax.fori_loop(0, Nparticles, addto_J, (Jx, Jy, Jz))