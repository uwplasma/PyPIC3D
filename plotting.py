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
    ax.set_xlim( -(2/3)*x_wind, (2/3)*x_wind )
    ax.set_ylim( -(2/3)*y_wind, (2/3)*y_wind )
    ax.set_zlim( -(2/3)*z_wind, (2/3)*z_wind )
    ax.scatter(x, y, z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Particle Positions")
    plt.savefig(f"plots/positions/particles.{t:09}.png", dpi=300)
    plt.close()

def plot_velocity_histogram(vx, vy, vz, t, nbins=50):
    """
    Plots the histogram of the velocities of the particles.

    Parameters:
    vx (array-like): The x-component of the velocities of the particles.
    vy (array-like): The y-component of the velocities of the particles.
    vz (array-like): The z-component of the velocities of the particles.
    t (float): The time value.

    Returns:
    None
    """
    fig, axs = plt.subplots(3)
    fig.suptitle('Particle Velocities')
    axs[0].hist(jnp.abs(vx), bins=nbins)
    axs[0].set_title('X-Component')
    axs[1].hist(jnp.abs(vy), bins=nbins)
    axs[1].set_title('Y-Component')
    axs[2].hist(jnp.abs(vz), bins=nbins)
    axs[2].set_title('Z-Component')
    plt.savefig(f"plots/velocity_histograms/velocities.{t:09}.png", dpi=300)
    plt.close()

def plot_velocities(x, y, z, vx, vy, vz, t, x_wind, y_wind, z_wind):
    """
    Makes a 3D plot of the velocities of the particles.

    Parameters:
    vx (array-like): The x-component of the velocities of the particles.
    vy (array-like): The y-component of the velocities of the particles.
    vz (array-like): The z-component of the velocities of the particles.
    t (float): The time value.
    x_wind (float): The x-axis wind limit.
    y_wind (float): The y-axis wind limit.
    z_wind (float): The z-axis wind limit.

    Returns:
    None
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim( -x_wind, x_wind )
    #ax.set_ylim( -y_wind, y_wind )
    #ax.set_zlim( -z_wind, z_wind )
    ax.quiver(x, y, z, vx, vy, vz, length=x_wind/1000)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Particle Velocities")
    plt.savefig(f"plots/velocities/velocities.{t:09}.png", dpi=300)
    plt.close()

def plot_KE(KE, t):
    """
    Plots the kinetic energy of the particles.

    Parameters:
    KE (array-like): The kinetic energy of the particles.
    t (array-like): The time value.

    Returns:
    None
    """
    plt.plot(t, KE)
    plt.xlabel("Time")
    plt.ylabel("Kinetic Energy")
    plt.title("Kinetic Energy vs. Time")
    plt.savefig("plots/KE.png", dpi=300)
    plt.close()


def plot_probe(probe, name):
    """
    Plots a probe.

    Parameters:
    probe (array-like): The probe.

    Returns:
    None
    """
    plt.plot(probe)
    plt.xlabel("Time")
    plt.ylabel(f"{name}")
    plt.title(f"{name} vs. Time")
    plt.savefig(f"plots/{name}_probe.png", dpi=300)
    plt.close()