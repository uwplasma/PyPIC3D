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
from src.rho import update_rho

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

def plot_positions(particles, t, x_wind, y_wind, z_wind):
    """
    Makes a 3D plot of the positions of the particles.

    Parameters:
    particles (list): A list of ParticleSpecies objects containing positions.
    t (float): The time value.
    x_wind (float): The x-axis wind limit.
    y_wind (float): The y-axis wind limit.
    z_wind (float): The z-axis wind limit.

    Returns:
    None
    """
    x, y, z = [], [], []
    for species in particles:
        x.append(species.get_position()[0])
        y.append(species.get_position()[1])
        z.append(species.get_position()[2])
    x = jnp.concatenate(x)
    y = jnp.concatenate(y)
    z = jnp.concatenate(z)

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

def plot_velocities(particles, t, x_wind, y_wind, z_wind):
    """
    Makes a 3D plot of the velocities of the particles.

    Parameters:
    particles (list): A list of ParticleSpecies objects containing positions and velocities.
    t (float): The time value.
    x_wind (float): The x-axis wind limit.
    y_wind (float): The y-axis wind limit.
    z_wind (float): The z-axis wind limit.

    Returns:
    None
    """
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    for species in particles:
        pos = species.get_position()
        vel = species.get_velocity()
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        vx.append(vel[0])
        vy.append(vel[1])
        vz.append(vel[2])
    
    x = jnp.concatenate(x)
    y = jnp.concatenate(y)
    z = jnp.concatenate(z)
    vx = jnp.concatenate(vx)
    vy = jnp.concatenate(vy)
    vz = jnp.concatenate(vz)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
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


def plot_probe(probe, t, name, savename):
    """
    Plots a probe.

    Parameters:
    probe (array-like): The probe.

    Returns:
    None
    """
    plt.plot(t, probe)
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


# def particles_phase_space(particles, t, name):
#     """
#     Plot the phase space of the particles.

#     Parameters:
#     - particles (Particles): The particles to be plotted.
#     - t (ndarray): The time values.
#     - name (str): The name of the plot.

#     Returns:
#     None
#     """

#     total_x, total_vx = [], []
#     total_y, total_vy = [], []
#     total_z, total_vz = [], []

#     colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
#     idx = 0
#     for species in particles:
#         x, y, z = species.get_position()
#         vx, vy, vz = species.get_velocity()

#         total_x.append(x)
#         total_vx.append(vx)
#         total_y.append(y)
#         total_vy.append(vy)
#         total_z.append(z)
#         total_vz.append(vz)

#     if not os.path.exists(f"plots/phase_space/x/{name}"):
#         os.makedirs(f"plots/phase_space/x/{name}")
#     if not os.path.exists(f"plots/phase_space/y/{name}"):
#         os.makedirs(f"plots/phase_space/y/{name}")
#     if not os.path.exists(f"plots/phase_space/z/{name}"):
#         os.makedirs(f"plots/phase_space/z/{name}")
#     if not os.path.exists(f"plots/phase_space/magnitude/{name}"):
#         os.makedirs(f"plots/phase_space/magnitude/{name}")

#     v_magnitude = jnp.sqrt(jnp.square(jnp.concatenate(total_vx)) + jnp.square(jnp.concatenate(total_vy)) + jnp.square(jnp.concatenate(total_vz)))

#     plt.scatter(jnp.concatenate(total_x), v_magnitude)
#     plt.xlabel("Position")
#     plt.ylabel("Velocity Magnitude")
#     plt.title(f"{name} Phase Space (Magnitude)")
#     plt.savefig(f"plots/phase_space/magnitude/{name}_phase_space.{t:09}.png", dpi=300)
#     plt.close()

#     x = jnp.concatenate(total_x)
#     vx = jnp.concatenate(total_vx)
#     plt.scatter(x, vx)
#     plt.xlabel("Position")
#     plt.ylabel("Velocity")
#     plt.title(f"{name} Phase Space")
#     plt.savefig(f"plots/phase_space/x/{name}_phase_space.{t:09}.png", dpi=300)
#     plt.close()

#     y = jnp.concatenate(total_y)
#     vy = jnp.concatenate(total_vy)
#     plt.scatter(y, vy)
#     plt.xlabel("Position")
#     plt.ylabel("Velocity")
#     plt.title(f"{name} Phase Space")
#     plt.savefig(f"plots/phase_space/y/{name}_phase_space.{t:09}.png", dpi=300)
#     plt.close()

#     z = jnp.concatenate(total_z)
#     vz = jnp.concatenate(total_vz)
#     plt.scatter(z, vz)
#     plt.xlabel("Position")
#     plt.ylabel("Velocity")
#     plt.title(f"{name} Phase Space")
#     plt.savefig(f"plots/phase_space/z/{name}_phase_space.{t:09}.png", dpi=300)
#     plt.close()


def particles_phase_space(particles, t, name):
    """
    Plot the phase space of the particles.

    Parameters:
    - particles (Particles): The particles to be plotted.
    - t (ndarray): The time values.
    - name (str): The name of the plot.

    Returns:
    None
    """

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    idx = 0
    order = 10
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(x, vx, c=colors[idx], zorder=order)
        idx += 1
        order -= 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    #plt.ylim(-1e10, 1e10)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"plots/phase_space/x/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(y, vy, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"{name} Phase Space")
    plt.savefig(f"plots/phase_space/y/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(z, vz, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"{name} Phase Space")
    plt.savefig(f"plots/phase_space/z/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        v_magnitude = jnp.sqrt(jnp.square(vx) + jnp.square(vy) + jnp.square(vz))
        plt.scatter(x, v_magnitude, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity Magnitude")
    plt.title(f"{name} Phase Space (Magnitude)")
    plt.savefig(f"plots/phase_space/magnitude/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()



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


def dispersion_relation(field, direction, dx, dy, dz):
    """
    Calculate the dispersion relation of a field along a specified direction.

    Parameters:
    - field (ndarray): The field to analyze.
    - direction (str): The direction along which to calculate the dispersion relation ('x', 'y', or 'z').
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.

    Returns:
    - k (ndarray): The wave numbers.
    - omega (ndarray): The angular frequencies.
    """
    if direction == 'x':
        axis = 0
        spacing = dx
    elif direction == 'y':
        axis = 1
        spacing = dy
    elif direction == 'z':
        axis = 2
        spacing = dz
    else:
        raise ValueError("Invalid direction. Choose from 'x', 'y', or 'z'.")

    # Perform FFT along the specified direction
    field_fft = np.fft.fftshift(np.fft.fftn(field, axes=[axis]))
    N = field.shape[axis]
    k = np.fft.fftshift(np.fft.fftfreq(N, spacing)) * 2 * np.pi

    # Calculate the angular frequencies
    omega = np.abs(field_fft)

    return k, omega

def plot_dispersion_relation(k, omega, name):
    """
    Plots the dispersion relation between wave number (k) and angular frequency (omega).

    Parameters:
    k (array-like): Array of wave numbers.
    omega (array-like): Array of angular frequencies corresponding to the wave numbers.
    name (str): Name of the dispersion relation, used for the plot title and saved file name.

    Saves the plot as a PNG file in the 'plots' directory with the filename format '{name}_dispersion_relation.png'.
    """

    plt.plot(k, omega)
    plt.xlabel("Wave Number")
    plt.ylabel("Angular Frequency")
    plt.title(f"{name} Dispersion Relation")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{name}_dispersion_relation.png", dpi=300)
    plt.close()


def dominant_modes(field, direction, dx, dy, dz, num_modes=5):
    """
    Calculate the dominant wavenumber modes of a field along a specified direction.

    Parameters:
    - field (ndarray): The field to analyze.
    - direction (str): The direction along which to calculate the wavenumber modes ('x', 'y', or 'z').
    - dx (float): The grid spacing in the x-direction.
    - dy (float): The grid spacing in the y-direction.
    - dz (float): The grid spacing in the z-direction.
    - num_modes (int): The number of dominant modes to return.

    Returns:
    - dominant_modes (ndarray): The dominant wavenumber modes.
    """
    if direction == 'x':
        axis = 0
        spacing = dx
    elif direction == 'y':
        axis = 1
        spacing = dy
    elif direction == 'z':
        axis = 2
        spacing = dz
    else:
        raise ValueError("Invalid direction. Choose from 'x', 'y', or 'z'.")

    # Perform FFT along the specified direction
    field_fft = np.fft.fftshift(np.fft.fftn(field, axes=[axis]))
    N = field.shape[axis]
    k = np.fft.fftshift(np.fft.fftfreq(N, spacing)) * 2 * np.pi

    # Calculate the power spectrum
    power_spectrum = np.abs(field_fft)**2

    # Find the indices of the dominant modes
    dominant_indices = np.argsort(power_spectrum, axis=axis)[-num_modes:]

    # Extract the dominant wavenumber modes
    dominant_modes = k[dominant_indices]

    # Extract the dominant wavenumber modes
    dominant_modes = np.sum(dominant_modes, axis=(1, 2))

    return dominant_modes

def plot_dominant_modes(dominant_modes, t, name, savename):
    """
    Plots the dominant wavenumber modes of a field.

    Parameters:
    - dominant_modes (ndarray): The dominant wavenumber modes to plot.
    - t (int): The time value.
    - name (str): The name of the field.

    Saves the plot as a PNG file in the 'plots' directory with the filename format '{name}_dominant_modes.png'.
    """
    plt.plot(t, dominant_modes)
    plt.xlabel("Time")
    plt.ylabel("K")
    plt.title(f"{name}")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{savename}_dominant_modes.png", dpi=300)
    plt.close()
def write_probe(probe_data, t, filename):
    """
    Writes probe data and timestep to a file.

    Parameters:
    probe_data (any): The data collected by the probe to be written to the file.
    t (int or float): The timestep at which the probe data was collected.
    filename (str): The name of the file where the probe data will be written.

    Returns:
    None
    """
    with open(filename, 'a+') as file:
        file.write(f"{t}\t{probe_data}\n")