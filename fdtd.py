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
from boundaryconditions import apply_zero_boundary_condition


def centered_finite_difference_laplacian(field, dx, dy, dz, bc):
    """
    Calculates the Laplacian of a given field using centered finite difference and applies the specified boundary conditions.

    Parameters:
    - field: numpy.ndarray
        The input field.
    - dx: float
        The spacing between grid points in the x-direction.
    - dy: float
        The spacing between grid points in the y-direction.
    - dz: float
        The spacing between grid points in the z-direction.
    - boundary_condition: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - numpy.ndarray
        The Laplacian of the field with the specified boundary conditions applied.
    """

    if bc == 'dirichlet':
        field = apply_zero_boundary_condition(field)
        # apply zero boundary condition at the edges of the field

    x_comp = (jnp.roll(field, shift=1, axis=0) + jnp.roll(field, shift=-1, axis=0) - 2*field) / (dx*dx)
    y_comp = (jnp.roll(field, shift=1, axis=1) + jnp.roll(field, shift=-1, axis=1) - 2*field) / (dy*dy)
    z_comp = (jnp.roll(field, shift=1, axis=2) + jnp.roll(field, shift=-1, axis=2) - 2*field) / (dz*dz)
    # calculate the Laplacian of the field using centered finite difference

    if bc == 'neumann':
        x_comp = apply_zero_boundary_condition(x_comp)
        y_comp = apply_zero_boundary_condition(y_comp)
        z_comp = apply_zero_boundary_condition(z_comp)

    return x_comp + y_comp + z_comp

def centered_finite_difference_curl(field_x, field_y, field_z, dx, dy, dz, bc):
    """
    Computes the curl of a vector field using centered finite differencing and applies the specified boundary conditions.

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
    - bc: str
        The type of boundary condition ('periodic', 'neumann', 'dirichlet').

    Returns:
    - tuple of numpy.ndarray
        The curl components (curl_x, curl_y, curl_z) of the vector field with the specified boundary conditions applied.
    """
    if bc == 'dirichlet':
        field_x = apply_zero_boundary_condition(field_x)
        field_y = apply_zero_boundary_condition(field_y)
        field_z = apply_zero_boundary_condition(field_z)

    curl_x = (jnp.roll(field_z, shift=-1, axis=1) - jnp.roll(field_z, shift=1, axis=1)) / (2 * dy) - \
                (jnp.roll(field_y, shift=-1, axis=2) - jnp.roll(field_y, shift=1, axis=2)) / (2 * dz)
    curl_y = (jnp.roll(field_x, shift=-1, axis=2) - jnp.roll(field_x, shift=1, axis=2)) / (2 * dz) - \
                (jnp.roll(field_z, shift=-1, axis=0) - jnp.roll(field_z, shift=1, axis=0)) / (2 * dx)
    curl_z = (jnp.roll(field_y, shift=-1, axis=0) - jnp.roll(field_y, shift=1, axis=0)) / (2 * dx) - \
                (jnp.roll(field_x, shift=-1, axis=1) - jnp.roll(field_x, shift=1, axis=1)) / (2 * dy)

    if bc == 'neumann':
        curl_x = apply_zero_boundary_condition(curl_x)
        curl_y = apply_zero_boundary_condition(curl_y)
        curl_z = apply_zero_boundary_condition(curl_z)

    return curl_x, curl_y, curl_z

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

    curlx, curly, curlz = centered_finite_difference_curl(Ex, Ey, Ez, dx, dy, dz, boundary_condition)
    Bx = Bx - dt/2*curlx
    By = By - dt/2*curly
    Bz = Bz - dt/2*curlz
    # update the magnetic field
    return Bx, By, Bz

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

    curlx, curly, curlz = centered_finite_difference_curl(Bx, By, Bz, dx, dy, dz, boundary_condition)
    Ex = Ex + ( C**2 * curlx - 1/eps * Jx) * dt / 2
    Ey = Ey + ( C**2 * curly - 1/eps * Jy) * dt / 2
    Ez = Ez + ( C**2 * curlz - 1/eps * Jz) * dt / 2
    # update the electric field
    return Ex, Ey, Ez