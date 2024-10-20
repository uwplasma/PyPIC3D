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

from spectral import spectral_poisson_solve, spectral_laplacian, spectralBsolve, spectralEsolve
from fdtd import periodic_laplacian, neumann_laplacian, dirichlet_laplacian
from rho import update_rho
from cg import conjugate_grad
# import internal libraries

def initialize_fields(Nx, Ny, Nz):
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

def solve_poisson(rho, eps, dx, dy, dz, phi, bc='periodic', M = None):
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

    if bc == 'periodic':
        lapl = functools.partial(periodic_laplacian, dx=dx, dy=dy, dz=dz)
    elif bc == 'dirichlet':
        lapl = functools.partial(dirichlet_laplacian, dx=dx, dy=dy, dz=dz)
    elif bc == 'neumann':
        lapl = functools.partial(neumann_laplacian, dx=dx, dy=dy, dz=dz)
    #phi = conjugate_grad(lapl, rho/eps, phi, tol=1e-6, maxiter=10000, M=M)
    #phi, exitcode = jax.scipy.sparse.linalg.cg(lapl, -rho/eps, phi, tol=1e-6, maxiter=20000, M=M)
    if bc == 'spectral':
        phi = spectral_poisson_solve(rho, eps, dx, dy, dz)
    else:
        phi = conjugate_grad(lapl, -rho/eps, phi, tol=1e-6, maxiter=20000, M=M)
    return phi

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
    elif bc == 'periodic':
        x = periodic_laplacian(phi, dx, dy, dz)
    elif bc == 'dirichlet':
        x = dirichlet_laplacian(phi, dx, dy, dz)
    elif bc == 'neumann':
        x = neumann_laplacian(phi, dx, dy, dz)
    poisson_error = x + rho/eps
    index         = jnp.argmax(poisson_error)
    return 200 * jnp.abs( jnp.ravel(poisson_error)[index]) / ( jnp.abs(jnp.ravel(rho/eps)[index])+ jnp.abs(jnp.ravel(x)[index]) )
    # this method computes the relative percentage difference of poisson solver


def calculateE(N_electrons, electron_x, electron_y, electron_z, \
               N_ions, ion_x, ion_y, ion_z,                     \
               dx, dy, dz, q_e, q_i, rho, eps, phi, t, M, Nx, Ny, Nz, x_wind, y_wind, z_wind, bc, verbose, GPUs):
    """
                Calculates the electric field components (Ex, Ey, Ez), electric potential (phi), and charge density (rho) based on the given parameters.

                Parameters:
                - N_electrons (int): Number of electrons.
                - electron_x (array-like): x-coordinates of electrons.
                - electron_y (array-like): y-coordinates of electrons.
                - electron_z (array-like): z-coordinates of electrons.
                - N_ions (int): Number of ions.
                - ion_x (array-like): x-coordinates of ions.
                - ion_y (array-like): y-coordinates of ions.
                - ion_z (array-like): z-coordinates of ions.
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
    
    if GPUs:
        with jax.default_device(jax.devices('gpu')[0]):
                rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
                # reset value of charge density
                rho = update_rho(N_electrons, electron_x, electron_y, electron_z, dx, dy, dz, q_e, x_wind, y_wind, z_wind, rho)
                if N_ions > 0:
                    rho = update_rho(N_ions, ion_x, ion_y, ion_z, dx, dy, dz, q_i, x_wind, y_wind, z_wind, rho)
                    # update the charge density field
    else:
        rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
        # reset value of charge density
        rho = update_rho(N_electrons, electron_x, electron_y, electron_z, dx, dy, dz, q_e, x_wind, y_wind, z_wind, rho)
        if N_ions > 0:
            rho = update_rho(N_ions, ion_x, ion_y, ion_z, dx, dy, dz, q_i, x_wind, y_wind, z_wind, rho)
            # update the charge density field

    if verbose: print(f"Calculating Charge Density, Max Value: {jnp.max(rho)}")
    # print the maximum value of the charge density

    if GPUs:
        with jax.default_device(jax.devices('gpu')[0]):
                if t == 0:
                    phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, bc=bc, M=None)
                else:
                    phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, bc=bc, M=M)
    else:
        if t == 0:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, bc=bc, M=None)
        else:
            phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, bc=bc, M=M)
    # solve the poisson equation for the electric potential

    if verbose: print(f"Calculating Electric Potential, Max Value: {jnp.max(phi)}")
    # print the maximum value of the electric potential
    #if verbose: print( f'Poisson Relative Percent Difference: {compute_pe(phi, rho, eps, dx, dy, dz)}%')
    # Use conjugated gradients to calculate the electric potential from the charge density

    E_fields = jnp.gradient(phi)
    Ex       = -1*E_fields[0]
    Ey       = -1*E_fields[1]
    Ez       = -1*E_fields[2]
    # Calculate the E field using the gradient of the potential

    return Ex, Ey, Ez, phi, rho