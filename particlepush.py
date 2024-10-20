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
    interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, field, fill_value=0)
    # create the interpolator
    points = jnp.stack([x, y, z], axis=-1)
    return interpolate(points)

@jit
def boris(q, Ex, Ey, Ez, Bx, By, Bz, x, y, z, vx, vy, vz, grid, staggered_grid, dt, m):
    """
    Perform Boris push algorithm to update the velocity of a charged particle in an electromagnetic field.

    Parameters:
    q (float): Charge of the particle.
    Ex (ndarray): Electric field component array in the x-direction.
    Ey (ndarray): Electric field component array in the y-direction.
    Ez (ndarray): Electric field component array in the z-direction.
    Bx (ndarray): Magnetic field component array in the x-direction.
    By (ndarray): Magnetic field component array in the y-direction.
    Bz (ndarray): Magnetic field component array in the z-direction.
    x (ndarray): Particle position array in the x-direction.
    y (ndarray): Particle position array in the y-direction.
    z (ndarray): Particle position array in the z-direction.
    vx (ndarray): Particle velocity array in the x-direction.
    vy (ndarray): Particle velocity array in the y-direction.
    vz (ndarray): Particle velocity array in the z-direction.
    dt (float): Time step size.
    m (float): Mass of the particle.

    Returns:
    tuple: Updated velocity of the particle in the x, y, and z directions.
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