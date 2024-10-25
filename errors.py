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

from spectral import spectral_laplacian, spectral_divergence
from fdtd import centered_finite_difference_laplacian, centered_finite_difference_divergence


def compute_pe(phi, rho, eps, dx, dy, dz, bc='periodic'):
    """
    Compute the relative percentage difference of the Poisson solver.

    Parameters:
    phi (ndarray): The potential field.
    rho (ndarray): The charge density.
    eps (float): The permittivity.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    bc (str): The boundary condition.

    Returns:
    float: The relative percentage difference of the Poisson solver.
    """
    if bc == 'spectral':
        x = spectral_laplacian(phi, dx, dy, dz)
    else:
        x = centered_finite_difference_laplacian(phi, dx, dy, dz, bc)
    poisson_error = x + rho/eps
    index         = jnp.argmax(poisson_error)
    return 200 * jnp.abs( jnp.ravel(poisson_error)[index]) / ( jnp.abs(jnp.ravel(rho/eps)[index])+ jnp.abs(jnp.ravel(x)[index]) )
    # this method computes the relative percentage difference of poisson solver

def compute_magnetic_divergence_error(Bx, By, Bz, dx, dy, dz, bc='periodic'):
    """
    Compute the error in the divergence of the magnetic field for different solvers.

    Parameters:
    Bx (ndarray): The x-component of the magnetic field.
    By (ndarray): The y-component of the magnetic field.
    Bz (ndarray): The z-component of the magnetic field.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    bc (str): The boundary condition.

    Returns:
    float: The error in the divergence of the magnetic field.
    """
    if bc == 'spectral':
        divB = spectral_divergence(Bx, By, Bz, dx, dy, dz)
    else:
        divB = centered_finite_difference_divergence(Bx, By, Bz, dx, dy, dz, bc)
    
    divergence_error = jnp.sum(jnp.abs(divB))
    return divergence_error

def compute_electric_divergence_error(Ex, Ey, Ez, rho, eps, dx, dy, dz, bc='periodic'):
    """
    Compute the error in the divergence of the electric field using the charge density and the components of the electric field.

    Parameters:
    Ex (ndarray): The x-component of the electric field.
    Ey (ndarray): The y-component of the electric field.
    Ez (ndarray): The z-component of the electric field.
    rho (ndarray): The charge density.
    eps (float): The permittivity.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    bc (str): The boundary condition.

    Returns:
    float: The error in the divergence of the electric field.
    """
    if bc == 'spectral':
        divE = spectral_divergence(Ex, Ey, Ez, dx, dy, dz)
    else:
        divE = centered_finite_difference_divergence(Ex, Ey, Ez, dx, dy, dz, bc)
    
    divergence_error = jnp.sum(jnp.abs(divE - rho / eps))
    return divergence_error