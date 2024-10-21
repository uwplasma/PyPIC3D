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
def interpolate_field(field, grid, x, y, z):
    """
    Interpolates the given field at the specified (x, y, z) coordinates using a regular grid interpolator.
    Args:
        field (array-like): The field values to be interpolated.
        grid (tuple of array-like): The grid points for each dimension (x, y, z).
        x (array-like): The x-coordinates where interpolation is desired.
        y (array-like): The y-coordinates where interpolation is desired.
        z (array-like): The z-coordinates where interpolation is desired.
    Returns:
        array-like: Interpolated values at the specified (x, y, z) coordinates.
    """

    interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, field, fill_value=0)
    # create the interpolator
    points = jnp.stack([x, y, z], axis=-1)
    return interpolate(points)

def particle_push(particles, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt):
    """
    Updates the velocities of particles using the Boris algorithm.

    Args:
        particles (Particles): The particles to be updated.
        Ex (array-like): Electric field component in the x-direction.
        Ey (array-like): Electric field component in the y-direction.
        Ez (array-like): Electric field component in the z-direction.
        Bx (array-like): Magnetic field component in the x-direction.
        By (array-like): Magnetic field component in the y-direction.
        Bz (array-like): Magnetic field component in the z-direction.
        grid (Grid): The grid on which the fields are defined.
        staggered_grid (Grid): The staggered grid for field interpolation.
        dt (float): The time step for the update.

    Returns:
        Particles: The particles with updated velocities.
    """
    q = particles.get_charge()
    m = particles.get_mass()
    x, y, z = particles.get_position()
    vx, vy, vz = particles.get_velocity()
    # get the charge, mass, position, and velocity of the particles
    newvx, newvy, newvz = boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt)
    # use the boris algorithm to update the velocities
    particles.set_velocity(newvx, newvy, newvz)
    # set the new velocities of the particles
    return particles


@jit
def boris(q, m, x, y, z, vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt):
    """
    Perform the Boris algorithm to update the velocity of a charged particle in an electromagnetic field.

    Parameters:
    q (float): Charge of the particle.
    m (float): Mass of the particle.
    x (float): x-coordinate of the particle's position.
    y (float): y-coordinate of the particle's position.
    z (float): z-coordinate of the particle's position.
    vx (float): x-component of the particle's velocity.
    vy (float): y-component of the particle's velocity.
    vz (float): z-component of the particle's velocity.
    Ex (ndarray): x-component of the electric field array.
    Ey (ndarray): y-component of the electric field array.
    Ez (ndarray): z-component of the electric field array.
    Bx (ndarray): x-component of the magnetic field array.
    By (ndarray): y-component of the magnetic field array.
    Bz (ndarray): z-component of the magnetic field array.
    grid (ndarray): Grid for the electric field.
    staggered_grid (ndarray): Staggered grid for the magnetic field.
    dt (float): Time step for the update.

    Returns:
    tuple: Updated velocity components (newvx, newvy, newvz).
    """


    efield_atx = interpolate_field(Ex, grid, x, y, z)
    efield_aty = interpolate_field(Ey, grid, x, y, z)
    efield_atz = interpolate_field(Ez, grid, x, y, z)
    # interpolate the electric field component arrays and calculate the e field at the particle positions
    bfield_atx = interpolate_field(Bx, staggered_grid, x, y, z)
    bfield_aty = interpolate_field(By, staggered_grid, x, y, z)
    bfield_atz = interpolate_field(Bz, staggered_grid, x, y, z)
    # interpolate the magnetic field component arrays and calculate the b field at the particle positions

    vxminus = vx + q*dt/(2*m)*efield_atx
    vyminus = vy + q*dt/(2*m)*efield_aty
    vzminus = vz + q*dt/(2*m)*efield_atz
    # calculate the v minus vector used in the boris push algorithm
    tx = q*dt/(2*m)*bfield_atx
    ty = q*dt/(2*m)*bfield_aty
    tz = q*dt/(2*m)*bfield_atz

    vprimex = vxminus + (vyminus*tz - vzminus*ty)
    vprimey = vyminus + (vzminus*tx - vxminus*tz)
    vprimez = vzminus + (vxminus*ty - vyminus*tx)
    # vprime = vminus + vminus cross t

    smag = 2 / (1 + tx*tx + ty*ty + tz*tz)
    sx = smag * tx
    sy = smag * ty
    sz = smag * tz
    # calculate the scaled rotation vector

    vxplus = vxminus + (vprimey*sz - vprimez*sy)
    vyplus = vyminus + (vprimez*sx - vprimex*sz)
    vzplus = vzminus + (vprimex*sy - vprimey*sx)

    newvx = vxplus + q*dt/(2*m)*efield_atx
    newvy = vyplus + q*dt/(2*m)*efield_aty
    newvz = vzplus + q*dt/(2*m)*efield_atz
    # calculate the new velocity

    return newvx, newvy, newvz