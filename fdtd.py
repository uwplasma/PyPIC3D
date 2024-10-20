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
def periodic_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field using 2nd order finite difference with Periodic boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)
    return x_comp + y_comp + z_comp


@jit
def neumann_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field with Neumann boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The boundary condition.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """


    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx) 
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)

    x_comp = x_comp.at[0, :, :].set(0)
    x_comp = x_comp.at[-1, :, :].set(0)
    y_comp = y_comp .at[:, 0, :].set(0)
    y_comp = y_comp.at[:, -1, :].set(0)
    z_comp = z_comp.at[:, :, 0].set(0)
    z_comp = z_comp.at[:, :, -1].set(0)

    return x_comp + y_comp + z_comp


@jit
def dirichlet_laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field with Dirichlet boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - bc: str
        The boundary condition.

    Returns:
    - numpy.ndarray
        The Laplacian of the field.
    """
    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field)/(dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field)/(dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field)/(dz*dz)

    x_comp = x_comp.at[0, :, :].set((jnp.roll(field, shift=1, axis=0) - 2*field).at[0, :, :].get()/(dx*dx))
    x_comp = x_comp.at[-1, :, :].set((jnp.roll(field, shift=-1, axis=0) - 2*field).at[-1, :, :].get()/(dx*dx))
    y_comp = y_comp.at[:, 0, :].set((jnp.roll(field, shift=1, axis=1) - 2*field).at[:, 0, :].get()/(dy*dy))
    y_comp = y_comp.at[:, -1, :].set((jnp.roll(field, shift=-1, axis=1) - 2*field).at[:, -1, :].get()/(dy*dy))
    z_comp = z_comp.at[:, :, 0].set((jnp.roll(field, shift=1, axis=2) - 2*field).at[:, :, 0].get()/(dz*dz))
    z_comp = z_comp.at[:, :, -1].set((jnp.roll(field, shift=-1, axis=2) - 2*field).at[:, :, -1].get()/(dz*dz))

    return x_comp + y_comp + z_comp

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
    Bx = Bx - dt/2*curlx(Ey, Ez, dy, dz)
    By = By - dt/2*curly(Ex, Ez, dx, dz)
    Bz = Bz - dt/2*curlz(Ex, Ey, dx, dy)
    return Bx, By, Bz


@jit
def update_E(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, dt, C):
    """
    Update the electric field components Ex, Ey, and Ez based on the magnetic field components Bx, By, and Bz.

    Parameters:
    - Ex (float): The x-component of the electric field.
    - Ey (float): The y-component of the electric field.
    - Ez (float): The z-component of the electric field.
    - Bx (float): The x-component of the magnetic field.
    - By (float): The y-component of the magnetic field.
    - Bz (float): The z-component of the magnetic field.
    - dx (float): The spacing in the x-direction.
    - dy (float): The spacing in the y-direction.
    - dz (float): The spacing in the z-direction.
    - dt (float): The time step.
    - C (float): The Courant number.

    Returns:
    - Ex (float): The updated x-component of the electric field.
    - Ey (float): The updated y-component of the electric field.
    - Ez (float): The updated z-component of the electric field.
    """
    Ex = Ex + C**2*curlx(By, Bz, dy, dz)*dt/2
    Ey = Ey + C**2*curly(Bx, Bz, dx, dz)*dt/2
    Ez = Ez + C**2*curlz(Bx, By, dx, dy)*dt/2
    return Ex, Ey, Ez