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
import toml
# import external libraries

from rho import update_rho

def debugprint(value):
    """
    Prints the given value using JAX's debug print functionality.

    Args:
        value: The value to be printed. Can be of any type that is supported by JAX's debug print.
    """
    jax.debug.print('{x}', x=value)

def update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters):
    """
    Update the simulation parameters with values from a TOML config file.

    Parameters:
    - config_file (str): Path to the TOML config file.
    - simulation_parameters (dict): Dictionary of default simulation parameters.
    - plotting_parameters (dict): Dictionary of default plotting parameters.

    Returns:
    - tuple: Updated simulation parameters and plotting parameters.
    """
    # Load the TOML config file
    config = toml.load(config_file)
    
    # Update the simulation parameters with values from the config file
    for key, value in config["simulation_parameters"].items():
        if key in simulation_parameters:
            simulation_parameters[key] = value

    for key, value in config["plotting"].items():
        if key in plotting_parameters:
            plotting_parameters[key] = value
    
    return simulation_parameters, plotting_parameters

def read_simulation_parameters(file_path):
    """
    Reads simulation parameters from a TOML file.

    Args:
        file_path (str): Path to the TOML file.

    Returns:
        dict: Dictionary containing simulation parameters.
    """
    with open(file_path, 'r') as file:
        parameters = toml.load(file)
    return parameters


def courant_condition(courant_number, dx, dy, dz, C):
    """
    Calculate the Courant condition for a given grid spacing and wave speed.

    The Courant condition is a stability criterion for numerical solutions of partial differential equations. 
    It ensures that the numerical domain of dependence contains the true domain of dependence.

    Parameters:
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    C (float): Wave speed or Courant number.

    Returns:
    float: The maximum allowable time step for stability.
    """
    return 1 / (C * ( (1/dx) + (1/dy) + (1/dz) ) )
# calculate the courant condition

def plasma_frequency(N_electrons, x_wind, y_wind, z_wind, eps, me, q_e):
    """
    Calculate the theoretical frequency of a system based on the given parameters.

    Parameters:
    N_electrons (float): Number of electrons in the system.
    x_wind (float): Width of the system in the x-dimension.
    y_wind (float): Width of the system in the y-dimension.
    z_wind (float): Width of the system in the z-dimension.
    eps (float): Permittivity of the medium.
    me (float): Mass of an electron.
    q_e (float): Charge of an electron.

    Returns:
    float: Theoretical frequency of the system.
    """
    ne = N_electrons / (x_wind*y_wind*z_wind)
    return jnp.sqrt(  ne * q_e**2  / (eps*me)  )
# calculate the expected plasma frequency from analytical theory

def debye_length(eps, Te, N_electrons, x_wind, y_wind, z_wind, q_e, kb):
    """
    Calculate the Debye length of a system based on the given parameters.

    Parameters:
    eps (float): Permittivity of the medium.
    T (float): Temperature of the system.
    N_electrons (float): Number of electrons in the system.
    q_e (float): Charge of an electron.

    Returns:
    float: Debye length of the system.
    """

    ne = N_electrons / (x_wind*y_wind*z_wind)
    return jnp.sqrt( eps * kb * Te / (ne * q_e**2) )

def probe(fieldx, fieldy, fieldz, x, y, z):
    """
    Probe the value of a vector field at a given point.

    Parameters:
    - fieldx (ndarray): The x-component of the vector field.
    - fieldy (ndarray): The y-component of the vector field.
    - fieldz (ndarray): The z-component of the vector field.
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.

    Returns:
    - tuple: The value of the vector field at the given point.
    """
    return fieldx.at[x, y, z].get(), fieldy.at[x, y, z].get(), fieldz.at[x, y, z].get()


def magnitude_probe(fieldx, fieldy, fieldz, x, y, z):
    """
    Probe the magnitude of a vector field at a given point.

    Parameters:
    - fieldx (ndarray): The x-component of the vector field.
    - fieldy (ndarray): The y-component of the vector field.
    - fieldz (ndarray): The z-component of the vector field.
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - z (float): The z-coordinate of the point.

    Returns:
    - float: The magnitude of the vector field at the given point.
    """
    return jnp.sqrt(fieldx.at[x, y, z].get()**2 + fieldy.at[x, y, z].get()**2 + fieldz.at[x, y, z].get()**2)
@jit
def number_density(n, Nparticles, particlex, particley, particlez, dx, dy, dz, Nx, Ny, Nz):
    """
    Calculate the number density of particles at each grid point.

    Parameters:
    - n (array-like): The initial number density array.
    - Nparticles (int): The number of particles.
    - particlex (array-like): The x-coordinates of the particles.
    - particley (array-like): The y-coordinates of the particles.
    - particlez (array-like): The z-coordinates of the particles.
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.

    Returns:
    - ndarray: The number density of particles at each grid point.
    """
    x_wind = (Nx * dx).astype(int)
    y_wind = (Ny * dy).astype(int)
    z_wind = (Nz * dz).astype(int)
    n = update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, 1, x_wind, y_wind, z_wind, n)

    return n

def freq(n, Nelectrons, ex, ey, ez, Nx, Ny, Nz, dx, dy, dz):
    """
    Calculate the plasma frequency based on the given parameters.
    Parameters:
    n (array-like): Input array representing the electron distribution.
    Nelectrons (int): Total number of electrons.
    ex (float): Electric field component in the x-direction.
    ey (float): Electric field component in the y-direction.
    ez (float): Electric field component in the z-direction.
    Nx (int): Number of grid points in the x-direction.
    Ny (int): Number of grid points in the y-direction.
    Nz (int): Number of grid points in the z-direction.
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    Returns:
    float: The calculated plasma frequency.
    """

    ne = jnp.ravel(number_density(n, Nelectrons, ex, ey, ez, dx, dy, dz, Nx, Ny, Nz))
    # compute the number density of the electrons
    eps = 8.854e-12
    # permitivity of freespace
    q_e = -1.602e-19
    # charge of electron
    me = 9.1093837e-31 # Kg
    # mass of the electron
    c1 = q_e**2 / (eps*me)

    mask = jnp.where(  ne  > 0  )[0]
    # Calculate mean using the mask
    electron_density = jnp.mean(ne[mask])
    freq = jnp.sqrt( c1 * electron_density )
    return freq
# computes the average plasma frequency over the middle 75% of the world volume

def freq_probe(n, x, y, z, Nelectrons, ex, ey, ez, Nx, Ny, Nz, dx, dy, dz):
    """
    Calculate the plasma frequency at a given point in a 3D grid.
    Parameters:
    n (ndarray): The electron density array.
    x (float): The x-coordinate of the probe point.
    y (float): The y-coordinate of the probe point.
    z (float): The z-coordinate of the probe point.
    Nelectrons (int): The total number of electrons.
    ex (float): The extent of the grid in the x-direction.
    ey (float): The extent of the grid in the y-direction.
    ez (float): The extent of the grid in the z-direction.
    Nx (int): The number of grid points in the x-direction.
    Ny (int): The number of grid points in the y-direction.
    Nz (int): The number of grid points in the z-direction.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    dz (float): The grid spacing in the z-direction.
    Returns:
    float: The plasma frequency at the specified point.
    """

    ne = number_density(n, Nelectrons, ex, ey, ez, dx, dy, dz, Nx, Ny, Nz)
    # compute the number density of the electrons
    eps = 8.854e-12
    # permitivity of freespace
    q_e = -1.602e-19
    # charge of electron
    me = 9.1093837e-31 # Kg
    # mass of the electron
    xi, yi, zi = int(x/dx + Nx/2), int(y/dy + Ny/2), int(z/dz + Nz/2)
    # get the array spacings for x, y, and z
    c1 = q_e**2 / (eps*me)
    freq = jnp.sqrt( c1 * ne.at[xi,yi,zi].get() )    # calculate the plasma frequency at the array point: x, y, z
    return freq


def totalfield_energy(Ex, Ey, Ez, Bx, By, Bz, mu, eps):
    """
    Calculate the total field energy of the electric and magnetic fields.

    Parameters:
    - Ex (ndarray): The x-component of the electric field.
    - Ey (ndarray): The y-component of the electric field.
    - Ez (ndarray): The z-component of the electric field.
    - Bx (ndarray): The x-component of the magnetic field.
    - By (ndarray): The y-component of the magnetic field.
    - Bz (ndarray): The z-component of the magnetic field.

    Returns:
    - float: The total field energy.
    """

    total_magnetic_energy = (0.5/mu)*jnp.sum(Bx**2 + By**2 + Bz**2)
    total_electric_energy = (0.5*eps)*jnp.sum(Ex**2 + Ey**2 + Ez**2)
    return total_magnetic_energy + total_electric_energy