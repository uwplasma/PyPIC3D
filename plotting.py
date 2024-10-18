import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import scipy
import os

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

    # Create directory if it doesn't exist
    directory = "./plots/rho"
    if not os.path.exists(directory):
        os.makedirs(directory)


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

    # Create directory if it doesn't exist
    directory = "./plots/fields"
    if not os.path.exists(directory):
        os.makedirs(directory)

    gridToVTK(f"./plots/fields/{name}_{t:09}", x, y, z,   \
            cellData = {f"{name}_x" : np.asarray(fieldx), \
             f"{name}_y" : np.asarray(fieldy), f"{name}_z" : np.asarray(fieldz)}) 
# plot the electric fields in the vtk file format

def plot_1dposition(x, name, particle):
    """
    Plot the 1D position of a particle.

    Parameters:
    - x (ndarray): The x-coordinates of the particle.
    - name (str): The name of the plot.

    Returns:
    None
    """
    plt.plot(x)
    plt.title(f"{name} Position")
    plt.xlabel("Time")
    plt.ylabel("Position")

    if not os.path.exists(f"plots/{name}"):
        os.makedirs(f"plots/{name}")

    plt.savefig(f"plots/{name}/{particle}_position.png", dpi=300)
    plt.close()

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

    if not os.path.exists("plots/positions"):
        os.makedirs("plots/positions")

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

    if not os.path.exists("plots/velocity_histograms"):
        os.makedirs("plots/velocity_histograms")

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

    if not os.path.exists("plots/velocities"):
        os.makedirs("plots/velocities")

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

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig("plots/KE.png", dpi=300)
    plt.close()


def plot_probe(probe, name, savename):
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
    plt.title(f"{name}")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{savename}_probe.png", dpi=300)
    plt.close()

def fft(signal, dt):
    """
    Perform a Fast Fourier Transform (FFT) on the given signal.

    Parameters:
    - signal: The input signal to be transformed. It can be a list or a numpy array.
    - dt: The time interval between samples in the signal.

    Returns:
    - xf: The frequency index for the FFT.
    - yf: The transformed signal after FFT.

    Note:
    - The input signal will be converted to a numpy array if it is a list.
    """
    if type(signal) is list:
        signal = np.asarray(signal)

    N = signal.shape[0]
    # get the total length of the signal
    yf = scipy.fft.fft(signal)[:int(N/2)]
    # do a fast fourier transform
    xf = scipy.fft.fftfreq(N, dt)[:int(N/2)]
    # get the frequency index for the fast fourier transform
    return xf, yf


def plot_fft(signal, dt, name, savename):
    xf, yf = fft(signal, dt)

    plt.plot(xf, np.abs(yf))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of {name}")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{savename}.png", dpi=300)
    plt.close()
    return xf[ np.argmax(np.abs(yf)[1:]) ]
    # plot the fft of a signal

def phase_space(x, vx, t, name):
    """
    Plot the phase space of the particles.

    Parameters:
    - x (ndarray): The x-coordinates of the particles.
    - vx (ndarray): The x-component of the velocities of the particles.
    - t (ndarray): The time values.
    - name (str): The name of the plot.

    Returns:
    None
    """
    plt.scatter(x, vx)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"{name} Phase Space")

    if not os.path.exists(f"plots/phase_space/{name}"):
        os.makedirs(f"plots/phase_space/{name}")

    plt.savefig(f"plots/phase_space/{name}/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()


def multi_phase_space(x1, x2, vx1, vx2, t, species1, species2, name, x_wind):
    """
    Plot the phase space of the particles.

    Parameters:
    - x1 (ndarray): The x-coordinates of the particles in the first species.
    - x2 (ndarray): The x-coordinates of the particles in the second species.
    - vx1 (ndarray): The x-component of the velocities of the particles in the first species.
    - vx2 (ndarray): The x-component of the velocities of the particles in the second species.
    - t (ndarray): The time values.
    - name (str): The name of the plot.

    Returns:
    None
    """


    ax = plt.figure().add_subplot()
    ax.set_xlim( -(2/3)*x_wind, (2/3)*x_wind )
    ax.scatter(x1, vx1, c='r', label=f'{species1}')
    ax.scatter(x2, vx2, c='b', label=f'{species2}')
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"{name} Phase Space")
    ax.legend(loc='upper right')

    if not os.path.exists(f"plots/phase_space/{name}"):
        os.makedirs(f"plots/phase_space/{name}")
        
    plt.savefig(f"plots/phase_space/{name}/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()