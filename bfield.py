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
    delZdely = (jnp.roll(zfield, shift=1, axis=1) + jnp.roll(zfield, shift=-1, axis=1) - 2*zfield)/(dy*dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) + jnp.roll(yfield, shift=-1, axis=2) - 2*yfield)/(dz*dz)
    return delZdely - delYdelz

@jit
def curly(xfield, zfield, dx, dz):
    delXdelz = (jnp.roll(xfield, shift=1, axis=2) + jnp.roll(xfield, shift=-1, axis=2) - 2*xfield)/(dz*dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) + jnp.roll(zfield, shift=-1, axis=0) - 2*zfield)/(dx*dx)
    return delXdelz - delZdelx

@jit
def curlz(yfield, xfield, dx, dy):
    delYdelx = (jnp.roll(yfield, shift=1, axis=0) + jnp.roll(yfield, shift=-1, axis=0) - 2*yfield)/(dx*dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) + jnp.roll(xfield, shift=-1, axis=1) - 2*xfield)/(dy*dy)
    return delYdelx - delXdely

@jit
def update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt):
    Bx = Bx - dt*curlx(Ey, Ez, dy, dz)
    By = By - dt*curly(Ex, Ez, dx, dz)
    Bz = Bz - dt*curlz(Ex, Ey, dx, dy)
    return Bx, By, Bz