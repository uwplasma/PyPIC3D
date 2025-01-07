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

from PyPIC3D.pstd import spectral_poisson_solve, spectral_laplacian, spectral_gradient, spectral_marder_correction
from PyPIC3D.fdtd import centered_finite_difference_laplacian, centered_finite_difference_gradient
from PyPIC3D.rho import update_rho, compute_rho
from PyPIC3D.cg import conjugated_gradients
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
    # get the number of grid points in each direction
    Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric field arrays as 0
    Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the magnetic field arrays as 0

    Jx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jy = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the current density arrays as 0

    phi = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric potential and charge density arrays as 0

    return Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho

@partial(jit, static_argnums=(4, 5, 7))
def solve_poisson(rho, constants, world, phi, solver, bc='periodic', M = None, GPUs = False):
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
        phi = spectral_poisson_solve(rho, constants, world)
    elif solver == 'fdtd':
        eps = constants['eps']
        dx = world['dx']
        dy = world['dy']
        dz = world['dz']
        # lapl = functools.partial(centered_finite_difference_laplacian, dx=dx, dy=dy, dz=dz, bc=bc)
        # lapl = jit(lapl)
        # #define the laplacian operator using finite difference method
        # #phi = conjugate_grad(lapl, -rho/eps, phi, tol=1e-9, maxiter=5000, M=M)

        # phi = conjugated_gradients(lapl, -rho/eps, phi, tol=1e-9, maxiter=1000)
        sor = functools.partial(solve_poisson_sor, dx=dx, dy=dy, dz=dz, eps=eps, omega=0.15, tol=1e-15, max_iter=15000)
        phi = sor(phi, rho)
        #phi = jax.scipy.sparse.linalg.cg(lapl, -rho/eps, x0=phi, tol=1e-6, maxiter=40000, M=M)[0]
        #phi = solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=0.25, tol=1e-6, max_iter=100000)
    return phi

@partial(jit, static_argnums=(9, 10, 11, 12, 13, 14))
def calculateE(Ex, Ey, Ez, world, particles, constants, rho, phi, M, t, solver, bc, verbose, GPUs, electrostatic):
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
    - electrostatic (bool): Whether to use electrostatic solver.

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

    if solver == 'spectral':

        if electrostatic or t < 1:
            rho = compute_rho(particles, rho, world, GPUs)
            # calculate the charge density based on the particle positions

            if verbose:
                jax.debug.print("Calculating Charge Density, Max Value: {}", jnp.max(jnp.abs(rho)))

            if t == 0:
                phi = solve_poisson(rho, constants, world, phi=rho, solver=solver, bc=bc, M=None, GPUs=GPUs)
            else:
                phi = solve_poisson(rho, constants, world, phi=phi, solver=solver, bc=bc, M=M, GPUs=GPUs)

            if verbose:
                jax.debug.print("Calculating Electric Potential, Max Value: {}", jnp.max(phi))
                jax.debug.print("Potential Error: {}%", compute_pe(phi, rho, constants, world, solver, bc='periodic'))

            Ex, Ey, Ez = spectral_gradient(phi, world)
            # compute the gradient of the electric potential to get the electric field
            Ex = -Ex
            Ey = -Ey
            Ez = -Ez
            # multiply by -1 to get the correct direction of the electric field


        # Ex, Ey, Ez = spectral_marder_correction(Ex, Ey, Ez, rho, world, constants)
        # # apply Marder correction to the electric field

    elif solver == 'fdtd':

        if electrostatic or t < 1:
            rho = compute_rho(particles, rho, world, GPUs)
            # calculate the charge density based on the particle positions

            if verbose:
                jax.debug.print("Calculating Charge Density, Max Value: {}", jnp.max(jnp.abs(rho)))

            if t == 0:
                phi = solve_poisson(rho, constants, world, phi=rho, solver=solver, bc=bc, M=None, GPUs=GPUs)
            else:
                phi = solve_poisson(rho, constants, world, phi=phi, solver=solver, bc=bc, M=M, GPUs=GPUs)

            if verbose:
                jax.debug.print("Calculating Electric Potential, Max Value: {}", jnp.max(phi))
                jax.debug.print("Potential Error: {}%", compute_pe(phi, rho, constants, world, solver, bc='periodic'))

            Ex, Ey, Ez = centered_finite_difference_gradient(phi, dx, dy, dz, bc)
            Ex = -Ex
            Ey = -Ey
            Ez = -Ez
            # multiply by -1 to get the correct direction of the electric field

    return Ex, Ey, Ez, phi, rho


@partial(jit, static_argnums=(7))
def update_E(grid, staggered_grid, E, B, J, world, constants, curl_func):
    """
    Update the electric field components (Ex, Ey, Ez) based on the given parameters.

    Parameters:
    grid (object): The grid object containing the simulation grid.
    staggered_grid (object): The staggered grid object for the simulation.
    E (tuple): A tuple containing the electric field components (Ex, Ey, Ez).
    B (tuple): A tuple containing the magnetic field components (Bx, By, Bz).
    J (tuple): A tuple containing the current density components (Jx, Jy, Jz).
    world (dict): A dictionary containing the world parameters such as 'dx', 'dy', 'dz', and 'dt'.
    constants (dict): A dictionary containing the physical constants such as 'C' (speed of light) and 'eps' (permittivity).
    curl_func (function): A function to calculate the curl of the magnetic field.

    Returns:
    tuple: Updated electric field components (Ex, Ey, Ez).
    """

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']
    C = constants['C']
    eps = constants['eps']
    mu = constants['mu']

    jax.debug.print("dx: {}", dx)
    jax.debug.print("dy: {}", dy)
    jax.debug.print("dz: {}", dz)
    jax.debug.print("dt: {}", dt)
    jax.debug.print("C: {}", C)
    jax.debug.print("eps: {}", eps)
    jax.debug.print("mu: {}", mu)

    curlx, curly, curlz = curl_func(Bx, By, Bz)
    # calculate the curl of the magnetic field
    jax.debug.print("Curl B Mean Magnitude: {}", jnp.mean(jnp.sqrt(curlx**2 + curly**2 + curlz**2)))
    jax.debug.print("Current Density Mean Magnitude: {}", jnp.mean(jnp.sqrt(Jx**2 + Jy**2 + Jz**2)))

    Ex = Ex.at[:,:,:].add( ( C**2 * curlx - Jx / eps ) * dt )
    Ey = Ey.at[:,:,:].add( ( C**2 * curly - Jy / eps ) * dt )
    Ez = Ez.at[:,:,:].add( ( C**2 * curlz - Jz / eps ) * dt )

    return Ex, Ey, Ez


@partial(jit, static_argnums=(6))
def update_B(grid, staggered_grid, E, B, world, constants, curl_func):
    """
    Update the magnetic field components (Bx, By, Bz) using the curl of the electric field.

    Parameters:
    grid (ndarray): The grid on which the fields are defined.
    staggered_grid (ndarray): The staggered grid for field calculations.
    E (tuple): The electric field components (Ex, Ey, Ez).
    B (tuple): The magnetic field components (Bx, By, Bz).
    world (dict): Dictionary containing simulation parameters such as 'dx', 'dy', 'dz', and 'dt'.
    constants (dict): Dictionary containing physical constants.
    curl_func (function): Function to calculate the curl of the electric field.

    Returns:
    tuple: Updated magnetic field components (Bx, By, Bz).
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']

    Ex, Ey, Ez = E
    Bx, By, Bz = B

    curlx, curly, curlz = curl_func(Ex, Ey, Ez)
    jax.debug.print("Curl E Mean Magnitude: {}", jnp.mean(jnp.sqrt(curlx**2 + curly**2 + curlz**2)))
    # calculate the curl of the electric field
    Bx = Bx.at[:,:,:].add(-1*dt*curlx)
    By = By.at[:,:,:].add(-1*dt*curly)
    Bz = Bz.at[:,:,:].add(-1*dt*curlz)

    return Bx, By, Bz