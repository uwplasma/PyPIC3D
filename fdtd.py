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

from utils import interpolate_and_stagger_field

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
def periodic_divergence(field_x, field_y, field_z, dx, dy, dz):
    """
    Computes the divergence of a vector field using centered finite differencing with periodic boundary conditions.

    Parameters:
    - field_x: numpy.ndarray
        The x-component of the vector field.
    - field_y: numpy.ndarray
        The y-component of the vector field.
    - field_z: numpy.ndarray
        The z-component of the vector field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.

    Returns:
    - numpy.ndarray
        The divergence of the vector field.
    """
    div_x = (jnp.roll(field_x, shift=-1, axis=0) - jnp.roll(field_x, shift=1, axis=0)) / (2 * dx)
    div_y = (jnp.roll(field_y, shift=-1, axis=1) - jnp.roll(field_y, shift=1, axis=1)) / (2 * dy)
    div_z = (jnp.roll(field_z, shift=-1, axis=2) - jnp.roll(field_z, shift=1, axis=2)) / (2 * dz)
    return div_x + div_y + div_z

@jit
def neumann_divergence(field_x, field_y, field_z, dx, dy, dz):
    """
    Computes the divergence of a vector field using centered finite differencing with Neumann boundary conditions.

    Parameters:
    - field_x: numpy.ndarray
        The x-component of the vector field.
    - field_y: numpy.ndarray
        The y-component of the vector field.
    - field_z: numpy.ndarray
        The z-component of the vector field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.

    Returns:
    - numpy.ndarray
        The divergence of the vector field.
    """
    div_x = (jnp.roll(field_x, shift=-1, axis=0) - jnp.roll(field_x, shift=1, axis=0)) / (2 * dx)
    div_y = (jnp.roll(field_y, shift=-1, axis=1) - jnp.roll(field_y, shift=1, axis=1)) / (2 * dy)
    div_z = (jnp.roll(field_z, shift=-1, axis=2) - jnp.roll(field_z, shift=1, axis=2)) / (2 * dz)

    div_x = div_x.at[0, :, :].set(0)
    div_x = div_x.at[-1, :, :].set(0)
    div_y = div_y.at[:, 0, :].set(0)
    div_y = div_y.at[:, -1, :].set(0)
    div_z = div_z.at[:, :, 0].set(0)
    div_z = div_z.at[:, :, -1].set(0)

    return div_x + div_y + div_z

@jit
def dirichlet_divergence(field_x, field_y, field_z, dx, dy, dz):
    """
    Computes the divergence of a vector field using centered finite differencing with Dirichlet boundary conditions.

    Parameters:
    - field_x: numpy.ndarray
        The x-component of the vector field.
    - field_y: numpy.ndarray
        The y-component of the vector field.
    - field_z: numpy.ndarray
        The z-component of the vector field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.

    Returns:
    - numpy.ndarray
        The divergence of the vector field.
    """
    div_x = (jnp.roll(field_x, shift=-1, axis=0) - jnp.roll(field_x, shift=1, axis=0)) / (2 * dx)
    div_y = (jnp.roll(field_y, shift=-1, axis=1) - jnp.roll(field_y, shift=1, axis=1)) / (2 * dy)
    div_z = (jnp.roll(field_z, shift=-1, axis=2) - jnp.roll(field_z, shift=1, axis=2)) / (2 * dz)

    div_x = div_x.at[0, :, :].set((jnp.roll(field_x, shift=-1, axis=0) - field_x).at[0, :, :].get() / dx)
    div_x = div_x.at[-1, :, :].set((field_x - jnp.roll(field_x, shift=1, axis=0)).at[-1, :, :].get() / dx)
    div_y = div_y.at[:, 0, :].set((jnp.roll(field_y, shift=-1, axis=1) - field_y).at[:, 0, :].get() / dy)
    div_y = div_y.at[:, -1, :].set((field_y - jnp.roll(field_y, shift=1, axis=1)).at[:, -1, :].get() / dy)
    div_z = div_z.at[:, :, 0].set((jnp.roll(field_z, shift=-1, axis=2) - field_z).at[:, :, 0].get() / dz)
    div_z = div_z.at[:, :, -1].set((field_z - jnp.roll(field_z, shift=1, axis=2)).at[:, :, -1].get() / dz)

    return div_x + div_y + div_z

@jit
def curlx_periodic(yfield, zfield, dy, dz):
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
def curly_periodic(xfield, zfield, dx, dz):
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
def curlz_periodic(yfield, xfield, dx, dy):
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
def curlx_neumann(yfield, zfield, dy, dz):
    """
    Compute the curl of a vector field using Neumann boundary conditions.
    This function calculates the x-component of the curl of a vector field
    given its y and z components. The Neumann boundary conditions are applied
    to ensure that the derivative at the boundaries is zero.

    Parameters:
    yfield (ndarray): The y-component of the vector field.
    zfield (ndarray): The z-component of the vector field.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    Returns:
    ndarray: The x-component of the curl of the vector field.
    """

    delZdely = (jnp.roll(zfield, shift=1, axis=1) + jnp.roll(zfield, shift=-1, axis=1) - 2*zfield)/(dy*dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) + jnp.roll(yfield, shift=-1, axis=2) - 2*yfield)/(dz*dz)

    delZdely = delZdely.at[:, 0, :].set(0)
    delZdely = delZdely.at[:, -1, :].set(0)
    delYdelz = delYdelz.at[:, :, 0].set(0)
    delYdelz = delYdelz.at[:, :, -1].set(0)

    return delZdely - delYdelz

@jit
def curly_neumann(xfield, zfield, dx, dz):
    """
    Compute the Curly Neumann boundary conditions for the given fields.
    This function calculates the second-order central differences for the 
    xfield and zfield along the z and x directions, respectively, and applies 
    Neumann boundary conditions by setting the boundary values to zero.

    Parameters:
    xfield (jnp.ndarray): The field values along the x-direction.
    zfield (jnp.ndarray): The field values along the z-direction.
    dx (float): The grid spacing in the x-direction.
    dz (float): The grid spacing in the z-direction.
    Returns:
    jnp.ndarray: The result of the Curly Neumann boundary condition applied 
                 to the input fields.
    """

    delXdelz = (jnp.roll(xfield, shift=1, axis=2) + jnp.roll(xfield, shift=-1, axis=2) - 2*xfield)/(dz*dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) + jnp.roll(zfield, shift=-1, axis=0) - 2*zfield)/(dx*dx)

    delXdelz = delXdelz.at[:, :, 0].set(0)
    delXdelz = delXdelz.at[:, :, -1].set(0)
    delZdelx = delZdelx.at[0, :, :].set(0)
    delZdelx = delZdelx.at[-1, :, :].set(0)

    return delXdelz - delZdelx

@jit
def curlz_neumann(yfield, xfield, dx, dy):
    """
    Compute the z-component of the curl of a 2D vector field using Neumann boundary conditions.

    This function calculates the z-component of the curl of a 2D vector field (yfield, xfield)
    using central differences and Neumann boundary conditions. The Neumann boundary conditions
    are applied by setting the derivative at the boundaries to zero.

    Parameters:
    yfield (ndarray): The y-component of the vector field.
    xfield (ndarray): The x-component of the vector field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.

    Returns:
    ndarray: The z-component of the curl of the input vector field.
    """

    delYdelx = (jnp.roll(yfield, shift=1, axis=0) + jnp.roll(yfield, shift=-1, axis=0) - 2*yfield)/(dx*dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) + jnp.roll(xfield, shift=-1, axis=1) - 2*xfield)/(dy*dy)

    delYdelx = delYdelx.at[0, :, :].set(0)
    delYdelx = delYdelx.at[-1, :, :].set(0)
    delXdely = delXdely.at[:, 0, :].set(0)
    delXdely = delXdely.at[:, -1, :].set(0)

    return delYdelx - delXdely

@jit
def curlx_dirichlet(yfield, zfield, dy, dz):
    """
    Compute the x-component of the curl of a vector field using Dirichlet boundary conditions.

    Parameters:
    yfield (ndarray): The y-component of the vector field.
    zfield (ndarray): The z-component of the vector field.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    ndarray: The x-component of the curl of the vector field.
    """
    delZdely = (jnp.roll(zfield, shift=1, axis=1) - jnp.roll(zfield, shift=-1, axis=1)) / (2 * dy)
    delYdelz = (jnp.roll(yfield, shift=1, axis=2) - jnp.roll(yfield, shift=-1, axis=2)) / (2 * dz)

    delZdely = delZdely.at[:, 0, :].set((zfield[:, 1, :] - zfield[:, 0, :]) / dy)
    delZdely = delZdely.at[:, -1, :].set((zfield[:, -1, :] - zfield[:, -2, :]) / dy)
    delYdelz = delYdelz.at[:, :, 0].set((yfield[:, :, 1] - yfield[:, :, 0]) / dz)
    delYdelz = delYdelz.at[:, :, -1].set((yfield[:, :, -1] - yfield[:, :, -2]) / dz)

    return delZdely - delYdelz

@jit
def curly_dirichlet(xfield, zfield, dx, dz):
    """
    Compute the y-component of the curl of a vector field using Dirichlet boundary conditions.

    Parameters:
    xfield (ndarray): The x-component of the vector field.
    zfield (ndarray): The z-component of the vector field.
    dx (float): The grid spacing in the x-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    ndarray: The y-component of the curl of the vector field.
    """
    delXdelz = (jnp.roll(xfield, shift=1, axis=2) - jnp.roll(xfield, shift=-1, axis=2)) / (2 * dz)
    delZdelx = (jnp.roll(zfield, shift=1, axis=0) - jnp.roll(zfield, shift=-1, axis=0)) / (2 * dx)

    delXdelz = delXdelz.at[:, :, 0].set((xfield[:, :, 1] - xfield[:, :, 0]) / dz)
    delXdelz = delXdelz.at[:, :, -1].set((xfield[:, :, -1] - xfield[:, :, -2]) / dz)
    delZdelx = delZdelx.at[0, :, :].set((zfield[1, :, :] - zfield[0, :, :]) / dx)
    delZdelx = delZdelx.at[-1, :, :].set((zfield[-1, :, :] - zfield[-2, :, :]) / dx)

    return delXdelz - delZdelx

@jit
def curlz_dirichlet(yfield, xfield, dx, dy):
    """
    Compute the z-component of the curl of a vector field using Dirichlet boundary conditions.

    Parameters:
    yfield (ndarray): The y-component of the vector field.
    xfield (ndarray): The x-component of the vector field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.

    Returns:
    ndarray: The z-component of the curl of the vector field.
    """
    delYdelx = (jnp.roll(yfield, shift=1, axis=0) - jnp.roll(yfield, shift=-1, axis=0)) / (2 * dx)
    delXdely = (jnp.roll(xfield, shift=1, axis=1) - jnp.roll(xfield, shift=-1, axis=1)) / (2 * dy)

    delYdelx = delYdelx.at[0, :, :].set((yfield[1, :, :] - yfield[0, :, :]) / dx)
    delYdelx = delYdelx.at[-1, :, :].set((yfield[-1, :, :] - yfield[-2, :, :]) / dx)
    delXdely = delXdely.at[:, 0, :].set((xfield[:, 1, :] - xfield[:, 0, :]) / dy)
    delXdely = delXdely.at[:, -1, :].set((xfield[:, -1, :] - xfield[:, -2, :]) / dy)

    return delYdelx - delXdely

@jit
def update_B(grid, staggered_grid, Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt, boundary_condition):
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
    - boundary_condition (str): The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - Bx (float): The updated x-component of the magnetic field.
    - By (float): The updated y-component of the magnetic field.
    - Bz (float): The updated z-component of the magnetic field.
    """
    if boundary_condition == 'periodic':
        curlx_func = curlx_periodic
        curly_func = curly_periodic
        curlz_func = curlz_periodic
    elif boundary_condition == 'neumann':
        curlx_func = curlx_neumann
        curly_func = curly_neumann
        curlz_func = curlz_neumann
    elif boundary_condition == 'dirichlet':
        curlx_func = curlx_dirichlet
        curly_func = curly_dirichlet
        curlz_func = curlz_dirichlet
    else:
        raise ValueError("Invalid boundary condition")

    curlx = curlx_func(Ey, Ez, dy, dz)
    curly = curly_func(Ex, Ez, dx, dz)
    curlz = curlz_func(Ex, Ey, dx, dy)
    # calculate the curl of the electric field
    # curlx = interpolate_and_stagger_field(curlx, grid, staggered_grid)
    # curly = interpolate_and_stagger_field(curly, grid, staggered_grid)
    # curlz = interpolate_and_stagger_field(curlz, grid, staggered_grid)
    # interpolate the curl of the electric field to the cell faces
    Bx = Bx - dt/2*curlx
    By = By - dt/2*curly
    Bz = Bz - dt/2*curlz
    # update the magnetic field
    return Bx, By, Bz


@jit
def update_E(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dx, dy, dz, dt, C, eps, boundary_condition):
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
    - boundary_condition (str): The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - Ex (float): The updated x-component of the electric field.
    - Ey (float): The updated y-component of the electric field.
    - Ez (float): The updated z-component of the electric field.
    """
    if boundary_condition == 'periodic':
        curlx_func = curlx_periodic
        curly_func = curly_periodic
        curlz_func = curlz_periodic
    elif boundary_condition == 'neumann':
        curlx_func = curlx_neumann
        curly_func = curly_neumann
        curlz_func = curlz_neumann
    elif boundary_condition == 'dirichlet':
        curlx_func = curlx_dirichlet
        curly_func = curly_dirichlet
        curlz_func = curlz_dirichlet
    else:
        raise ValueError("Invalid boundary condition")

    curlx = curlx_func(By, Bz, dy, dz)
    curly = curly_func(Bx, Bz, dx, dz)
    curlz = curlz_func(Bx, By, dx, dy)
    # calculate the curl of the magnetic field
    # curlx = interpolate_and_stagger_field(curlx, staggered_grid, grid)
    # curly = interpolate_and_stagger_field(curly, staggered_grid, grid)
    # curlz = interpolate_and_stagger_field(curlz, staggered_grid, grid)
    # interpolate the curl of the magnetic field to the cell centers
    Ex = Ex + ( C**2 * curlx - 1/eps * Jx) * dt / 2
    Ey = Ey + ( C**2 * curly - 1/eps * Jy) * dt / 2
    Ez = Ez + ( C**2 * curlz - 1/eps * Jz) * dt / 2
    return Ex, Ey, Ez