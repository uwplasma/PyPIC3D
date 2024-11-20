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


def psatd(Ex, Ey, Ez, Bx, By, Bz, q, m, dx, dy, dz, dt, x, y, z, vx, vy, vz):
    """
    Perform the PSATD algorithm to update the electric and magnetic fields.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.
    - q (float): The charge of the particle.
    - m (float): The mass of the particle.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - dt (float): The time step.
    - x (ndarray): The x-coordinates of the particles.
    - y (ndarray): The y-coordinates of the particles.
    - z (ndarray): The z-coordinates of the particles.
    - vx (ndarray): The x-component of the velocity of the particles.
    - vy (ndarray): The y-component of the velocity of the particles.
    - vz (ndarray): The z-component of the velocity of the particles.

    Returns:
    - Ex (ndarray): The updated x-component of the electric field.
    - Ey (ndarray): The updated y-component of the electric field.
    - Ez (ndarray): The updated z-component of the electric field.
    - Bx (ndarray): The updated x-component of the magnetic field.
    - By (ndarray): The updated y-component of the magnetic field.
    - Bz (ndarray): The updated z-component of the magnetic field.
    """
    # C = 1.0
    # # speed of light
    # Ex, Ey, Ez = spectralEsolve(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, dt, C)
    # # update the electric field
    # Bx, By, Bz = spectralBsolve(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)
    # # update the magnetic field
    # return Ex, Ey, Ez, Bx, By, Bz

    # Build Fourier Mesh
    Nx, Ny, Nz = Ex.shape
    kx = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz, dz) * 2 * jnp.pi
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    # create 3D meshgrid of wavenumbers
    
    # FFT of the fields
    Ex_hat = jnp.fft.fftn(Ex)
    Ey_hat = jnp.fft.fftn(Ey)
    Ez_hat = jnp.fft.fftn(Ez)
    Bx_hat = jnp.fft.fftn(Bx)
    By_hat = jnp.fft.fftn(By)
    Bz_hat = jnp.fft.fftn(Bz)

    return