import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK


@jit
def boris(q, Ex, Ey, Ez, Bx, By, Bz, x, y, z, vx, vy, vz, dt, m):
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


    efield_atx = jax.scipy.ndimage.map_coordinates(Ex, [x, y, z], order=1)
    efield_aty = jax.scipy.ndimage.map_coordinates(Ey, [x, y, z], order=1)
    efield_atz = jax.scipy.ndimage.map_coordinates(Ez, [x, y, z], order=1)
    # interpolate the electric field component arrays and calculate the e field at the particle positions
    bfield_atx = jax.scipy.ndimage.map_coordinates(Bx, [x, y, z], order=1)
    bfield_aty = jax.scipy.ndimage.map_coordinates(By, [x, y, z], order=1)
    bfield_atz = jax.scipy.ndimage.map_coordinates(Bz, [x, y, z], order=1)
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

@jit
def curlx(yfield, zfield, dy, dz):
    """
    Calculate the curl of a vector field in the x-direction.

    Parameters:
    - yfield (ndarray): The y-component of the vector field.
    - zfield (ndarray): The z-component of the vector field.
    - dy (float): The spacing between y-values.
    - dz (float): The spacing between z-values.

    Returns:
    - ndarray: The x-component of the curl of the vector field.
    """
    delZdely = (jnp.roll(zfield, shift=1, axis=1) + jnp.roll(zfield, shift=-1, axis=1) - 2*zfield)/(dy*dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) + jnp.roll(yfield, shift=-1, axis=2) - 2*yfield)/(dz*dz)
    return delZdely - delYdelz

@jit
def curly(xfield, zfield, dx, dz):
    """
    Calculates the curl of a vector field in 2D.

    Parameters:
    - xfield (ndarray): The x-component of the vector field.
    - zfield (ndarray): The z-component of the vector field.
    - dx (float): The spacing between grid points in the x-direction.
    - dz (float): The spacing between grid points in the z-direction.

    Returns:
    - ndarray: The curl of the vector field.

    """
    delXdelz = (jnp.roll(xfield, shift=1, axis=2) + jnp.roll(xfield, shift=-1, axis=2) - 2*xfield)/(dz*dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) + jnp.roll(zfield, shift=-1, axis=0) - 2*zfield)/(dx*dx)
    return delXdelz - delZdelx

@jit
def curlz(yfield, xfield, dx, dy):
    """
    Calculate the curl of a 2D vector field in the z-direction.

    Parameters:
    - yfield (ndarray): The y-component of the vector field.
    - xfield (ndarray): The x-component of the vector field.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.

    Returns:
    - ndarray: The z-component of the curl of the vector field.
    """
    delYdelx = (jnp.roll(yfield, shift=1, axis=0) + jnp.roll(yfield, shift=-1, axis=0) - 2*yfield)/(dx*dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) + jnp.roll(xfield, shift=-1, axis=1) - 2*xfield)/(dy*dy)
    return delYdelx - delXdely

@jit
def update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt):
    """
    Update the magnetic field components Bx, By, and Bz based on the electric field components Ex, Ey, and Ez.

    Parameters:
    - Bx (float): The x-component of the magnetic field.
    - By (float): The y-component of the magnetic field.
    - Bz (float): The z-component of the magnetic field.
    - Ex (float): The x-component of the electric field.
    - Ey (float): The y-component of the electric field.
    - Ez (float): The z-component of the electric field.
    - dx (float): The spacing in the x-direction.
    - dy (float): The spacing in the y-direction.
    - dz (float): The spacing in the z-direction.
    - dt (float): The time step.

    Returns:
    - Bx (float): The updated x-component of the magnetic field.
    - By (float): The updated y-component of the magnetic field.
    - Bz (float): The updated z-component of the magnetic field.
    """
    Bx = Bx - dt*curlx(Ey, Ez, dy, dz)
    By = By - dt*curly(Ex, Ez, dx, dz)
    Bz = Bz - dt*curlz(Ex, Ey, dx, dy)
    return Bx, By, Bz