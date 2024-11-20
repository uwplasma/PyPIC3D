
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
        x = particlex.at[particle].get()
        y = particley.at[particle].get()
        z = particlez.at[particle].get()
        vx = particlevx.at[particle].get()
        vy = particlevy.at[particle].get()
        vz = particlevz.at[particle].get()

        # Calculate the nearest grid points
        x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
        y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
        z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)

        # Calculate the difference between the particle position and the nearest grid point
        deltax = (x + x_wind / 2) - x0 * dx
        deltay = (y + y_wind / 2) - y0 * dy
        deltaz = (z + z_wind / 2) - z0 * dz

        # Calculate the index of the next grid point
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Calculate the weights for the surrounding grid points
        wx = deltax / (x + x_wind/2)
        wy = deltay / (y + y_wind/2)
        wz = deltaz / (z + z_wind/2)

        # Calculate the volume of each grid point
        dv = dx * dy * dz

        # Distribute the current density to the surrounding grid points
        Jx = Jx.at[x0, y0, z0].add(q * vx * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jx = Jx.at[x1, y0, z0].add(q * vx * wx * (1 - wy) * (1 - wz) / dv)
        Jx = Jx.at[x0, y1, z0].add(q * vx * (1 - wx) * wy * (1 - wz) / dv)
        Jx = Jx.at[x0, y0, z1].add(q * vx * (1 - wx) * (1 - wy) * wz / dv)
        Jx = Jx.at[x1, y1, z0].add(q * vx * wx * wy * (1 - wz) / dv)
        Jx = Jx.at[x1, y0, z1].add(q * vx * wx * (1 - wy) * wz / dv)
        Jx = Jx.at[x0, y1, z1].add(q * vx * (1 - wx) * wy * wz / dv)
        Jx = Jx.at[x1, y1, z1].add(q * vx * wx * wy * wz / dv)

        Jy = Jy.at[x0, y0, z0].add(q * vy * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jy = Jy.at[x1, y0, z0].add(q * vy * wx * (1 - wy) * (1 - wz) / dv)
        Jy = Jy.at[x0, y1, z0].add(q * vy * (1 - wx) * wy * (1 - wz) / dv)
        Jy = Jy.at[x0, y0, z1].add(q * vy * (1 - wx) * (1 - wy) * wz / dv)
        Jy = Jy.at[x1, y1, z0].add(q * vy * wx * wy * (1 - wz) / dv)
        Jy = Jy.at[x1, y0, z1].add(q * vy * wx * (1 - wy) * wz / dv)
        Jy = Jy.at[x0, y1, z1].add(q * vy * (1 - wx) * wy * wz / dv)
        Jy = Jy.at[x1, y1, z1].add(q * vy * wx * wy * wz / dv)

        Jz = Jz.at[x0, y0, z0].add(q * vz * (1 - wx) * (1 - wy) * (1 - wz) / dv)
        Jz = Jz.at[x1, y0, z0].add(q * vz * wx * (1 - wy) * (1 - wz) / dv)
        Jz = Jz.at[x0, y1, z0].add(q * vz * (1 - wx) * wy * (1 - wz) / dv)
        Jz = Jz.at[x0, y0, z1].add(q * vz * (1 - wx) * (1 - wy) * wz / dv)
        Jz = Jz.at[x1, y1, z0].add(q * vz * wx * wy * (1 - wz) / dv)
        Jz = Jz.at[x1, y0, z1].add(q * vz * wx * (1 - wy) * wz / dv)
        Jz = Jz.at[x0, y1, z1].add(q * vz * (1 - wx) * wy * wz / dv)
        Jz = Jz.at[x1, y1, z1].add(q * vz * wx * wy * wz / dv)

        return Jx, Jy, Jz

    return jax.lax.fori_loop(0, Nparticles-1, addto_J, (Jx, Jy, Jz))