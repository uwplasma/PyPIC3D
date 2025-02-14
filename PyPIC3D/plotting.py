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
import plotly.graph_objects as go
from PyPIC3D.rho import update_rho
import vtk
import vtk.util.numpy_support as vtknp

from PyPIC3D.errors import (
    compute_electric_divergence_error, compute_magnetic_divergence_error
)

from PyPIC3D.J import compute_current_density

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

def save_vector_field_as_vtk(fieldx, fieldy, fieldz, grid, file_path):
    """
    Save a 3D vector field as a VTK rectilinear grid file.
    Parameters:
    fieldx (numpy.ndarray): The x-component of the vector field.
    fieldy (numpy.ndarray): The y-component of the vector field.
    fieldz (numpy.ndarray): The z-component of the vector field.
    grid (tuple of numpy.ndarray): A tuple containing the grid coordinates (x, y, z).
    file_path (str): The file path where the VTK file will be saved.
    Returns:
    None
    """

    # Create a new vtkRectilinearGrid
    rectilinear_grid = vtk.vtkRectilinearGrid()
    rectilinear_grid.SetDimensions(grid[0].size, grid[1].size, grid[2].size)

    # Set the coordinates for the grid
    rectilinear_grid.SetXCoordinates(vtknp.numpy_to_vtk(grid[0]))
    rectilinear_grid.SetYCoordinates(vtknp.numpy_to_vtk(grid[1]))
    rectilinear_grid.SetZCoordinates(vtknp.numpy_to_vtk(grid[2]))

    # Combine the components into a single vector field
    vector_field = np.stack((fieldx, fieldy, fieldz), axis=-1)

    # Convert the numpy vector field to vtk format
    vtk_vector_field = vtknp.numpy_to_vtk(vector_field.reshape(-1, 3), deep=True)
    vtk_vector_field.SetName("VectorField")

    # Add the vector field to the grid
    rectilinear_grid.GetPointData().SetVectors(vtk_vector_field)

    # Write the grid to a file
    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(rectilinear_grid)
    writer.Write()

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


def plot_positions(particles, t, x_wind, y_wind, z_wind, path):
    """
    Makes an interactive 3D plot of the positions of the particles using Plotly.

    Parameters:
    particles (list): A list of ParticleSpecies objects containing positions.
    t (float): The time value.
    x_wind (float): The x-axis wind limit.
    y_wind (float): The y-axis wind limit.
    z_wind (float): The z-axis wind limit.

    Returns:
    None
    """
    fig = go.Figure()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan']
    for idx, species in enumerate(particles):
        x, y, z = species.get_position()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=2, color=colors[idx % len(colors)]),
            name=f'Species {idx + 1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-(2/3)*x_wind, (2/3)*x_wind]),
            yaxis=dict(range=[-(2/3)*y_wind, (2/3)*y_wind]),
            zaxis=dict(range=[-(2/3)*z_wind, (2/3)*z_wind]),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        title="Particle Positions"
    )

    if not os.path.exists(f"{path}/data/positions"):
        os.makedirs(f"{path}/data/positions")

    fig.write_html(f"{path}/data/positions/particles.{t:09}.html")


def plot_velocity_histogram(species, t, output_dir, nbins=50):
    """
    Plots the histogram of the magnitudes of the velocities of the particles.

    Parameters:
    vx (array-like): The x-component of the velocities of the particles.
    vy (array-like): The y-component of the velocities of the particles.
    vz (array-like): The z-component of the velocities of the particles.
    t (float): The time value.

    Returns:
    None
    """
    vx, vy, vz = species.get_velocity()
    species_name = species.get_name().replace(" ", "")
    v_magnitude = jnp.sqrt(vx**2 + vy**2 + vz**2)
    plt.hist(v_magnitude, bins=nbins)
    plt.title('Particle Velocity Magnitudes')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Number of Particles')

    if not os.path.exists(f"{output_dir}/data/velocity_histograms/{species_name}"):
        os.makedirs(f"{output_dir}/data/velocity_histograms/{species_name}")

    plt.savefig(f"{output_dir}/data/velocity_histograms/{species_name}/velocities.{t:09}.png", dpi=200)
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
    yf = jnp.fft.fftn(signal)[:int(N/2)]
    # do a fast fourier transform
    xf = scipy.fft.fftfreq(N, dt)[:int(N/2)]
    # get the frequency index for the fast fourier transform
    return xf, yf


def plot_fft(signal, dt, name, path):
    xf, yf = fft(signal, dt)

    plt.plot(xf, np.abs(yf))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of {name}")

    if not os.path.exists(f"{path}/data/fft"):
        os.makedirs(f"{path}/data/fft")

    savename = name.replace(" ", "_")
    plt.savefig(f"{path}/data/fft/{savename}.png", dpi=300)
    plt.close()
    return xf[ np.argmax(np.abs(yf)[1:]) ]
    # plot the fft of a signal


def particles_phase_space(particles, world, t, name, path):
    """
    Plot the phase space of the particles.

    Parameters:
    - particles (Particles): The particles to be plotted.
    - t (ndarray): The time values.
    - name (str): The name of the plot.

    Returns:
    None
    """

    if not os.path.exists(f"{path}/data/phase_space/x"):
        os.makedirs(f"{path}/data/phase_space/x")
    if not os.path.exists(f"{path}/data/phase_space/y"):
        os.makedirs(f"{path}/data/phase_space/y")
    if not os.path.exists(f"{path}/data/phase_space/z"):
        os.makedirs(f"{path}/data/phase_space/z")

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    idx = 0
    order = 10
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(x, vx, c=colors[idx], zorder=order, s=1)
        idx += 1
        order -= 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    #plt.ylim(-1e10, 1e10)
    plt.xlim(-(2/3)*x_wind, (2/3)*x_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/x/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(y, vy, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(-(2/3)*y_wind, (2/3)*y_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/y/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(z, vz, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(-(2/3)*z_wind, (2/3)*z_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/z/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()

def center_of_mass(particles):
    """
    Calculate the center of mass of a particle species.

    Parameters:
    - particles (ParticleSpecies): The particle species object containing positions and masses.

    Returns:
    - ndarray: The center of mass coordinates.
    """
    if particles.get_number_of_particles() == 0:
        return 0, 0, 0
    
    total_mass = particles.get_mass() * particles.get_number_of_particles()
    # get the total mass of the particles
    x_com = jnp.sum( particles.get_mass() * particles.get_position()[0] ) / total_mass
    y_com = jnp.sum( particles.get_mass() * particles.get_position()[1] ) / total_mass
    z_com = jnp.sum( particles.get_mass() * particles.get_position()[2] ) / total_mass
    # calculate the center of mass
    return x_com, y_com, z_com

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

    max_amp = np.argsort(field_fft, axis=None)[-num_modes:]
    max_amp = np.unravel_index(max_amp, field_fft.shape)
    max_k   = k[max_amp[axis]]
    # get the k values of the dominant modes

    return max_k

def plot_dominant_modes(dominant_modes, t, name, savename):
    """
    Plots the dominant wavenumber modes of a field.

    Parameters:
    - dominant_modes (ndarray): The dominant wavenumber modes to plot.
    - t (int): The time value.
    - name (str): The name of the field.

    Saves the plot as a PNG file in the 'plots' directory with the filename format '{name}_dominant_modes.png'.
    """
    for mode in range(dominant_modes.shape[1]):
        plt.plot(t, dominant_modes[:, mode], label=f"Mode {mode + 1}")
    #plt.plot(t, dominant_modes)
    plt.xlabel("Time")
    plt.ylabel("K")
    plt.title(f"{name}")
    plt.legend()

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
    with open(filename, 'a') as file:
        file.write(f"{t}\t{probe_data}\n")

def plot_initial_KE(particles, path):
    """
    Plots the initial kinetic energy of the particles.

    Parameters:
    - particles (Particles): The particles to be plotted.
    - t (ndarray): The time values.
    - name (str): The name of the plot.

    Returns:
    None
    """
    for particle in particles:
        particle_name = particle.get_name().replace(" ", "")
        vx, vy, vz = particle.get_velocity()
        vmag = jnp.sqrt(vx**2 + vy**2 + vz**2)
        KE = 0.5 * particle.get_mass() * vmag**2
        plt.hist(KE, bins=50)
        plt.xlabel("Kinetic Energy")
        plt.ylabel("Number of Particles")
        plt.title("Initial Kinetic Energy")
        plt.savefig(f"{path}/data/{particle_name}_initialKE.png", dpi=300)
        plt.close()

def plot_slice(field_slice, t, name, path, world, dt):
    """
    Plots a 2D slice of a field and saves the plot as a PNG file.

    Parameters:
    field_slice (2D array): The 2D array representing the field slice to be plotted.
    t (int): The time step index.
    name (str): The name of the field being plotted.
    world (dict): A dictionary containing the dimensions of the world with keys 'x_wind' and 'y_wind'.
    dt (float): The time step duration.

    Returns:
    None
    """

    if not os.path.exists(f"{path}/data/{name}_slice"):
        os.makedirs(f'{path}/data/{name}_slice')
    # Create directory if it doesn't exist
    
    plt.title(f'{name} at t={t*dt:.2e}s')
    plt.imshow(field_slice, origin='lower', extent=[-world['x_wind']/2, world['x_wind']/2, -world['y_wind']/2, world['y_wind']/2])
    plt.colorbar(label=name)
    plt.tight_layout()
    plt.savefig(f'{path}/data/{name}_slice/{name}_slice_{t:09}.png', dpi=300)
    plt.close()

def write_data(filename, time, data):
    """
    Write the given time and data to a file.

    Parameters:
    filename (str): The name of the file to write to.
    time (float): The time value to write.
    data (any): The data to write.

    Returns:
    None
    """
    with open(filename, "a") as f:
        f.write(f"{time}, {data}\n")


def save_datas(t, dt, particles, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz, E_grid, B_grid, plotting_parameters, world, constants, solver, bc, output_dir):
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']

    #Eline = jnp.sqrt(Ex**2 + Ey**2 + Ez**2)[175, :, 0]
    # select a slice of the E field along the y-axis
    #_ = plot_fft(Eline, dt, f"E along Tungsten {t}", output_dir)

    if plotting_parameters['plotenergy']:
        E2_integral = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(Ex**2 + Ey**2 + Ez**2, dx=dz), dx=dy), dx=dx)
        B2_integral = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(Bx**2 + By**2 + Bz**2, dx=dz), dx=dy), dx=dx)
        #integral of E^2 and B^2 over the entire grid
        e_energy = 0.5*constants['eps']*E2_integral
        b_energy = 0.5/constants['mu']*B2_integral
        #electric and magnetic field energy
        kinetic_energy = sum(particle.kinetic_energy() for particle in particles)
        #kinetic energy of the particles
        write_data(f"{output_dir}/data/total_energy.txt", t*dt, e_energy + b_energy + kinetic_energy)
        write_data(f"{output_dir}/data/electric_field_energy.txt", t*dt, e_energy)
        write_data(f"{output_dir}/data/magnetic_field_energy.txt", t*dt, b_energy)
        write_data(f"{output_dir}/data/kinetic_energy.txt", t*dt, kinetic_energy)
        # write the total energy to a file

    if plotting_parameters['plotvelocities']:
        for species in particles:
            plot_velocity_histogram(species, t, output_dir, nbins=50)

    if plotting_parameters['plotcurrent']:
        write_data(f"{output_dir}/data/Jx_mean.txt", t*dt, jnp.mean(Jx))
        write_data(f"{output_dir}/data/Jy_mean.txt", t*dt, jnp.mean(Jy))
        write_data(f"{output_dir}/data/Jz_mean.txt", t*dt, jnp.mean(Jz))
    # write the mean current corrections to a file

    if plotting_parameters['plotpositions']:
        plot_positions(particles, t, world['x_wind'], world['y_wind'], world['z_wind'], output_dir)

    if plotting_parameters['phaseSpace']:
        particles_phase_space([particles[0]], world, t, "Particles", output_dir)

    if plotting_parameters['plotfields']:
        # save_vector_field_as_vtk(Ex, Ey, Ez, E_grid, f"{output_dir}/fields/E_{t*dt:09}.vtr")
        # save_vector_field_as_vtk(Bx, By, Bz, B_grid, f"{output_dir}/fields/B_{t*dt:09}.vtr")
        # plot_rho(rho, t, "rho", dx, dy, dz)
        write_data(f"{output_dir}/data/averageE.txt", t*dt, jnp.mean(jnp.sqrt(Ex**2 + Ey**2 + Ez**2)))
        write_data(f"{output_dir}/data/averageB.txt", t*dt, jnp.mean(jnp.sqrt(Bx**2 + By**2 + Bz**2)))
        write_data(f"{output_dir}/data/Eprobe.txt", t*dt, magnitude_probe(Ex, Ey, Ez, Nx-1, Ny-1, Nz-1))
        write_data(f"{output_dir}/data/centerE.txt", t*dt, magnitude_probe(Ex, Ey, Ez, Nx//2, Ny//2, Nz//2))
        plot_slice(jnp.sqrt(Ex**2 + Ey**2 + Ez**2)[:, :, int(Nz/2)], t, 'E', output_dir, world, dt)
        plot_slice(rho[:, :, int(Nz/2)], t, 'rho', output_dir, world, dt)

    if plotting_parameters['plot_errors']:
        write_data(f"{output_dir}/data/electric_divergence_errors.txt", t*dt, compute_electric_divergence_error(Ex, Ey, Ez, rho, constants, world, solver, bc))
        write_data(f"{output_dir}/data/magnetic_divergence_errors.txt", t*dt, compute_magnetic_divergence_error(Bx, By, Bz, world, solver, bc))

def save_avg_positions(t, dt, particles, output_dir):
    avg_x_path = os.path.join(output_dir, "data/avg_x.txt")
    avg_y_path = os.path.join(output_dir, "data/avg_y.txt")
    avg_z_path = os.path.join(output_dir, "data/avg_z.txt")

    xcom, ycom, zcom = center_of_mass(particles[0])
    with open(avg_x_path, "a") as f_avg_x:
        f_avg_x.write(f"{t*dt}, {  xcom  }\n")
    with open(avg_y_path, "a") as f_avg_y:
        f_avg_y.write(f"{t*dt}, {  ycom }\n")
    with open(avg_z_path, "a") as f_avg_z:
        f_avg_z.write(f"{t*dt}, {  zcom }\n")

def save_total_momentum(t, dt, particles, output_dir):
    p0 = sum(particle.momentum() for particle in particles)
    filename = os.path.join(output_dir, "data/total_momentum.txt")
    with open(filename, "a") as f_momentum:
        f_momentum.write(f"{t*dt}, {p0}\n")

def continuity_error(rho, old_rho, particles, world, divergence_func, GPUs):
    """
    Calculate the continuity error in a plasma simulation.

    Parameters:
    rho (array-like): Current charge density.
    old_rho (array-like): Previous charge density.
    particles (array-like): Particle data.
    world (dict): Simulation world parameters, including time step 'dt'.
    divergence_func (function): Function to compute the divergence of the current density.
    GPUs (bool): Flag to indicate if GPUs are used for computation.

    Returns:
    float: The mean absolute continuity error.
    """

    dpdt = (rho - old_rho) / world['dt']
    # calculate the change in charge density over time
    Jx, Jy, Jz = jnp.zeros_like(rho), jnp.zeros_like(rho), jnp.zeros_like(rho)
    Jx, Jy, Jz = compute_current_density(particles, Jx, Jy, Jz, world, GPUs)
    # calculate the current density
    divJ = divergence_func(Jx, Jy, Jz)
    # calculate the divergence of the current density
    continuity_error = dpdt + divJ
    # calculate the continuity error
    return jnp.mean(jnp.abs(continuity_error))


def plotter(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters, solver, bc):
    """
    Plots and saves various simulation data at specified intervals.
    Parameters:
    particles (list): List of particle objects in the simulation.
    Ex, Ey, Ez (ndarray): Electric field components in the x, y, and z directions.
    Bx, By, Bz (ndarray): Magnetic field components in the x, y, and z directions.
    Jx, Jy, Jz (ndarray): Current density components in the x, y, and z directions.
    rho (ndarray): Charge density.
    phi (ndarray): Electric potential.
    E_grid, B_grid (ndarray): Grids for electric and magnetic fields.
    world (dict): Dictionary containing world parameters such as time step 'dt'.
    constants (dict): Dictionary containing physical constants.
    plotting_parameters (dict): Dictionary containing parameters for plotting, including 'plotting_interval'.
    M (object): Mass matrix or related object.
    solver (object): Solver object for the simulation.
    bc (object): Boundary conditions object.
    electrostatic (bool): Flag indicating if the simulation is electrostatic.
    verbose (bool): Flag for verbose output.
    GPUs (list): List of GPUs used in the simulation.
    Returns:
    None
    """

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']

    if plotting_parameters['plotting']:
        if t % plotting_parameters['plotting_interval'] == 0:
            save_avg_positions(t, dt, particles, output_dir)
            save_total_momentum(t, dt, particles, output_dir)
            save_datas(t, dt, particles, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz, E_grid, B_grid, plotting_parameters, world, constants, solver, bc, output_dir)