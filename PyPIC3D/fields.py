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

from PyPIC3D.pstd import spectral_poisson_solve, spectral_laplacian, spectralBsolve, spectralEsolve, spectral_gradient
from PyPIC3D.fdtd import centered_finite_difference_laplacian, centered_finite_difference_gradient
from PyPIC3D.rho import update_rho
from PyPIC3D.cg import conjugate_grad
from PyPIC3D.sor import solve_poisson_sor
from PyPIC3D.errors import compute_pe
from PyPIC3D.utils import use_gpu_if_set
# import internal libraries

def initialize_fields(world):
    """
    Initializes the electric and magnetic field arrays, as well as the electric potential and charge density arrays.

    Parameters:
    - Nx (int): Number of grid points in the x-direction.
    - Ny (int): Number of grid points in the y-direction.
    - Nz (int): Number of grid points in the z-direction.

    Returns:
    - Ex (ndarray): Electric field array in the x-direction.
    - Ey (ndarray): Electric field array in the y-direction.
    - Ez (ndarray): Electric field array in the z-direction.
    - Bx (ndarray): Magnetic field array in the x-direction.
    - By (ndarray): Magnetic field array in the y-direction.
    - Bz (ndarray): Magnetic field array in the z-direction.
    - phi (ndarray): Electric potential array.
    - rho (ndarray): Charge density array.
    """
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']
    
    Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric field arrays as 0
    Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the magnetic field arrays as 0

    phi = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric potential and charge density arrays as 0

    return Ex, Ey, Ez, Bx, By, Bz, phi, rho

@use_gpu_if_set
def solve_poisson(rho, eps, dx, dy, dz, phi, solver, bc='periodic', M = None, GPUs = False):
    """
    Solve the Poisson equation for electrostatic potential.

    Parameters:
    - rho (ndarray): Charge density.
    - eps (float): Permittivity.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - dz (float): Grid spacing in the z-direction.
    - phi (ndarray): Initial guess for the electrostatic potential.
    - bc (str): Boundary condition.
    - M (ndarray, optional): Preconditioner matrix for the conjugate gradient solver.

    Returns:
    - phi (ndarray): Solution to the Poisson equation.
    """

    if solver == 'spectral':
        phi = spectral_poisson_solve(rho, eps, dx, dy, dz)
    elif solver == 'fdtd':
        lapl = functools.partial(centered_finite_difference_laplacian, dx=dx, dy=dy, dz=dz, bc=bc)
        lapl = jit(lapl)
        # define the laplacian operator using finite difference method
        phi = conjugate_grad(lapl, -rho/eps, phi, tol=1e-8, maxiter=100000, M=M)
        #phi = solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=0.25, tol=1e-6, max_iter=100000)
    return phi

def calculateE(world, particles, constants, rho, phi, M, t, solver, bc, verbose, GPUs):
    """
    Calculates the electric field components (Ex, Ey, Ez), electric potential (phi), and charge density (rho) based on the given parameters.

    Parameters:
    - electrons (object): Object containing information about the electrons.
    - ions (object): Object containing information about the ions.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - dz (float): Grid spacing in the z-direction.
    - q_e (float): Charge of an electron.
    - q_i (float): Charge of an ion.
    - rho (array-like): Initial charge density.
    - eps (float): Permittivity of the medium.
    - phi (array-like): Initial electric potential.
    - t (int): Time step.
    - M (array-like): Matrix for solving Poisson's equation.
    - Nx (int): Number of grid points in the x-direction.
    - Ny (int): Number of grid points in the y-direction.
    - Nz (int): Number of grid points in the z-direction.
    - bc (str): Boundary condition.
    - verbose (bool): Whether to print additional information.
    - GPUs (bool): Whether to use GPUs for Poisson solver.

    Returns:
    - Ex (array-like): x-component of the electric field.
    - Ey (array-like): y-component of the electric field.
    - Ez (array-like): z-component of the electric field.
    - phi (array-like): Updated electric potential.
    - rho (array-like): Updated charge density.
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']

    eps = constants['eps']

    if solver == 'spectral' or solver == 'fdtd':
        for species in particles:
            N_particles = species.get_number_of_particles()
            charge = species.get_charge()
            if N_particles > 0:
                particle_x, particle_y, particle_z = species.get_position()
                rho = update_rho(N_particles, particle_x, particle_y, particle_z, dx, dy, dz, charge, x_wind, y_wind, z_wind, rho, GPUs)


    if verbose:
        print(f"Calculating Charge Density, Max Value: {jnp.max(rho)}")

    if solver == 'spectral' or solver == 'fdtd':
        if t == 0:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, solver=solver, bc=bc, M=None, GPUs=GPUs)
        else:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, solver=solver, bc=bc, M=M, GPUs=GPUs)


    if verbose:
        print(f"Calculating Electric Potential, Max Value: {jnp.max(phi)}")
        print(f"Potential Error: {compute_pe(phi, rho, constants, world, solver, bc='periodic')}%")

    if solver == 'spectral':
        Ex, Ey, Ez = spectral_gradient(phi, dx, dy, dz)
        Ex = -Ex
        Ey = -Ey
        Ez = -Ez
    elif solver == 'fdtd':
        Ex, Ey, Ez = centered_finite_difference_gradient(phi, dx, dy, dz, bc)
        Ex = -Ex
        Ey = -Ey
        Ez = -Ez

    return Ex, Ey, Ez, phi, rho
