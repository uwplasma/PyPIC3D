import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK

def plot_rho(rho, t, name, dx, dy, dz):
    """
    Plot the density field.

    Parameters:
    - rho (ndarray): The density field.
    - t (int): The time step.
    - name (str): The name of the plot.
    - dx (float): The spacing in the x-direction.
    - dy (float): The spacing in the y-direction.
    - dz (float): The spacing in the z-direction.

    Returns:
    None
    """
    Nx = rho.shape[0]
    Ny = rho.shape[1]
    Nz = rho.shape[2]
    x = np.linspace(0, Nx, Nx) * dx
    y = np.linspace(0, Ny, Ny) * dy
    z = np.linspace(0, Nz, Nz) * dz
    gridToVTK(f"./plots/rho/{name}_{t:09}", x, y, z,   \
            cellData = {f"{name}" : np.asarray(rho)}) 
# plot the charge density in the vtk file format

def plot_fields(fieldx, fieldy, fieldz, t, name, dx, dy, dz):
    """
    Plot the fields in a 3D grid.

    Parameters:
    - fieldx (ndarray): Array representing the x-component of the field.
    - fieldy (ndarray): Array representing the y-component of the field.
    - fieldz (ndarray): Array representing the z-component of the field.
    - t (float): Time value.
    - name (str): Name of the field.
    - dx (float): Spacing between grid points in the x-direction.
    - dy (float): Spacing between grid points in the y-direction.
    - dz (float): Spacing between grid points in the z-direction.

    Returns:
    None
    """
    Nx = fieldx.shape[0]
    Ny = fieldx.shape[1]
    Nz = fieldx.shape[2]
    x = np.linspace(0, Nx, Nx) * dx
    y = np.linspace(0, Ny, Ny) * dy
    z = np.linspace(0, Nz, Nz) * dz
    gridToVTK(f"./plots/fields/{name}_{t:09}", x, y, z,   \
            cellData = {f"{name}_x" : np.asarray(fieldx), \
             f"{name}_y" : np.asarray(fieldy), f"{name}_z" : np.asarray(fieldz)}) 
# plot the electric fields in the vtk file format

def plot_positions(x, y, z, t, x_wind, y_wind, z_wind):
    """
    Makes a 3D plot of the positions of the particles.

    Parameters:
    x (array-like): The x-coordinates of the particles.
    y (array-like): The y-coordinates of the particles.
    z (array-like): The z-coordinates of the particles.
    t (float): The time value.
    x_wind (float): The x-axis wind limit.
    y_wind (float): The y-axis wind limit.
    z_wind (float): The z-axis wind limit.

    Returns:
    None
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim( -x_wind, x_wind )
    ax.set_ylim( -y_wind, y_wind )
    ax.set_zlim( -z_wind, z_wind )
    ax.scatter(x, y, z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Particle Positions")
    plt.savefig(f"plots/positions/particles.{t:09}.png", dpi=300)
    plt.close()