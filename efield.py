import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools

@jit
def laplacian(field, dx, dy, dz):
    """
    Calculates the Laplacian of a given field.

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
def index_particles(particle, positions, ds):
    """
    Calculate the index of a particle in a given position array.

    Parameters:
    - particle: int
        The index of the particle.
    - positions: pandas.DataFrame
        The position array containing the particle positions.
    - ds: float
        The grid spacing.

    Returns:
    - index: int
        The index of the particle in the position array, rounded down to the nearest integer.
    """
    return (positions.at[particle].get() / ds).astype(int)

@jit
def update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, q, rho):
    """
    Update the charge density field based on the positions of particles.
    Parameters:
    - Nparticles (int): The number of particles.
    - particlex (array-like): The x-coordinates of the particles.
    - particley (array-like): The y-coordinates of the particles.
    - particlez (array-like): The z-coordinates of the particles.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - q (float): The charge of each particle.
    - rho (array-like): The charge density field.
    Returns:
    - rho (array-like): The updated charge density field.
    """

    def addto_rho(particle, rho):
        x = index_particles(particle, particlex, dx)
        y = index_particles(particle, particley, dy)
        z = index_particles(particle, particlez, dz)
        rho = rho.at[x, y, z].add( q / (dx*dy*dz) )
        return rho
    
    return jax.lax.fori_loop(0, Nparticles-1, addto_rho, rho )

@jit
def solve_poisson(rho, eps, dx, dy, dz, phi, M = None):
    """
    Solve the Poisson equation for electrostatic potential.

    Parameters:
    - rho (ndarray): Charge density.
    - eps (float): Permittivity.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - dz (float): Grid spacing in the z-direction.
    - phi (ndarray): Initial guess for the electrostatic potential.
    - M (ndarray, optional): Preconditioner matrix for the conjugate gradient solver.

    Returns:
    - phi (ndarray): Solution to the Poisson equation.
    """
    lapl = functools.partial(laplacian, dx=dx, dy=dy, dz=dz)
    phi, exitcode = jax.scipy.sparse.linalg.cg(lapl, rho/eps, phi, tol=1e-6, maxiter=20000, M=M)
    return phi



def compute_pe(phi, rho, eps, dx, dy, dz):
    """
    Compute the relative percentage difference of the Poisson solver.

    Parameters:
    phi (ndarray): The potential field.
    rho (ndarray): The charge density.
    eps (float): The permittivity.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.

    Returns:
    float: The relative percentage difference of the Poisson solver.
    """
    x = laplacian(phi, dx, dy, dz)
    y = rho/eps
    poisson_error = x - y
    index         = jnp.argmax(poisson_error)
    return 200 * jnp.abs( jnp.ravel(x)[index] - jnp.ravel(y)[index] ) / ( (jnp.ravel(x)[index]) + (jnp.ravel(y)[index]) )

    # this method computes the relative percentage difference of poisson solver