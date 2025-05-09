import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
from pyevtk.hl import gridToVTK
import os
import plotly.graph_objects as go
from PyPIC3D.rho import update_rho
import jax
from functools import partial

def plot_rho(rho, t, name, dx, dy, dz):
    """
    Plot the density field.

    Args:
        rho (ndarray): The density field.
        t (int): The time step.
        name (str): The name of the plot.
        dx (float): The spacing in the x-direction.
        dy (float): The spacing in the y-direction.
        dz (float): The spacing in the z-direction.

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

    Args:
        fieldx (ndarray): Array representing the x-component of the field.
        fieldy (ndarray): Array representing the y-component of the field.
        fieldz (ndarray): Array representing the z-component of the field.
        t (float): Time value.
        name (str): Name of the field.
        dx (float): Spacing between grid points in the x-direction.
        dy (float): Spacing between grid points in the y-direction.
        dz (float): Spacing between grid points in the z-direction.

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

    Args:
        x (ndarray): The x-coordinates of the particle.
        name (str): The name of the plot.

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

    Args:
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

def write_particles_phase_space(particles, t, path):
    """
    Write the phase space of the particles to a file.

    Args:
        particles (Particles): The particles to be written.
        t (ndarray): The time values.
        name (str): The name of the plot.

    Returns:
        None
    """
    if not os.path.exists(f"{path}/data/phase_space/x"):
        os.makedirs(f"{path}/data/phase_space/x")
    if not os.path.exists(f"{path}/data/phase_space/y"):
        os.makedirs(f"{path}/data/phase_space/y")
    if not os.path.exists(f"{path}/data/phase_space/z"):
        os.makedirs(f"{path}/data/phase_space/z")
    # Create directory if it doesn't exist

    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        name = species.get_name().replace(" ", "")

        x_phase_space = jnp.stack((x, vx), axis=-1)
        y_phase_space = jnp.stack((y, vy), axis=-1)
        z_phase_space = jnp.stack((z, vz), axis=-1)

        jnp.save(f"{path}/data/phase_space/x/{name}_phase_space.{t:09}.npy", x_phase_space)
        jnp.save(f"{path}/data/phase_space/y/{name}_phase_space.{t:09}.npy", y_phase_space)
        jnp.save(f"{path}/data/phase_space/z/{name}_phase_space.{t:09}.npy", z_phase_space)
    # write the phase space of the particles to a file

def particles_phase_space(particles, world, t, name, path):
    """
    Plot the phase space of the particles.

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        name (str): The name of the plot.

    Returns:
        None
    """

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


def plot_initial_histograms(particle_species, world, name, path):
    """
    Generates and saves histograms for the initial positions and velocities 
    of particles in a simulation.

    Parameters:
        particle_species (object): An object representing the particle species, 
                                   which provides methods `get_position()` and 
                                   `get_velocity()` to retrieve particle positions 
                                   (x, y, z) and velocities (vx, vy, vz).
        world (dict): A dictionary containing the simulation world parameters, 
                      specifically the wind dimensions with keys 'x_wind', 
                      'y_wind', and 'z_wind'.
        name (str): A string representing the name of the particle species or 
                    simulation, used in the titles of the histograms and filenames.
        path (str): The directory path where the histogram images will be saved.

    Saves:
        Six histogram images:
            - Initial X, Y, and Z position histograms.
            - Initial X, Y, and Z velocity histograms.
        The images are saved in the specified `path` directory with filenames 
        formatted as "{name}_initial_<property>_histogram.png".
    """

    x, y, z = particle_species.get_position()
    vx, vy, vz = particle_species.get_velocity()

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']


    plt.hist(x, bins=50)
    plt.xlabel("X")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*x_wind, (2/3)*x_wind)
    plt.title(f"{name} Initial X Position Histogram")
    plt.savefig(f"{path}/{name}_initial_x_histogram.png", dpi=150)
    plt.close()

    plt.hist(y, bins=50)
    plt.xlabel("Y")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*y_wind, (2/3)*y_wind)
    plt.title(f"{name} Initial Y Position Histogram")
    plt.savefig(f"{path}/{name}_initial_y_histogram.png", dpi=150)
    plt.close()

    plt.hist(z, bins=50)
    plt.xlabel("Z")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*z_wind, (2/3)*z_wind)
    plt.title(f"{name} Initial Z Position Histogram")
    plt.savefig(f"{path}/{name}_initial_z_histogram.png", dpi=150)
    plt.close()

    plt.hist(vx, bins=50)
    plt.xlabel("X Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial X Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_x_velocity_histogram.png", dpi=150)
    plt.close()

    plt.hist(vy, bins=50)
    plt.xlabel("Y Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial Y Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_y_velocity_histogram.png", dpi=150)
    plt.close()

    plt.hist(vz, bins=50)
    plt.xlabel("Z Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial Z Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_z_velocity_histogram.png", dpi=150)
    plt.close()

@jit
def number_density(n, Nparticles, particlex, particley, particlez, dx, dy, dz, Nx, Ny, Nz):
    """
    Calculate the number density of particles at each grid point.

    Args:
        n (array-like): The initial number density array.
        Nparticles (int): The number of particles.
        particlex (array-like): The x-coordinates of the particles.
        particley (array-like): The y-coordinates of the particles.
        particlez (array-like): The z-coordinates of the particles.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        dz (float): The grid spacing in the z-direction.

    Returns:
        ndarray: The number density of particles at each grid point.
    """
    x_wind = (Nx * dx).astype(int)
    y_wind = (Ny * dy).astype(int)
    z_wind = (Nz * dz).astype(int)
    n = update_rho(Nparticles, particlex, particley, particlez, dx, dy, dz, 1, x_wind, y_wind, z_wind, n)

    return n

def magnitude_probe(fieldx, fieldy, fieldz, x, y, z):
    """
    Probe the magnitude of a vector field at a given point.

    Args:
        fieldx (ndarray): The x-component of the vector field.
        fieldy (ndarray): The y-component of the vector field.
        fieldz (ndarray): The z-component of the vector field.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.

    Returns:
        float: The magnitude of the vector field at the given point.
    """
    return jnp.sqrt(fieldx.at[x, y, z].get()**2 + fieldy.at[x, y, z].get()**2 + fieldz.at[x, y, z].get()**2)

def freq(n, Nelectrons, ex, ey, ez, Nx, Ny, Nz, dx, dy, dz):
    """
    Calculate the plasma frequency based on the given parameters.

    Args:
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

    Args:
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

def write_probe(probe_data, t, filename):
    """
    Writes probe data and timestep to a file.

    Args:
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

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        name (str): The name of the plot.

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


@partial(jax.jit, static_argnums=(2, 3))
def plot_slice(field_slice, t, name, path, world, dt):
    """
    Plots a 2D slice of a field and saves the plot as a PNG file using JAX's debug callback.

    Args:
        field_slice (2D array): The 2D array representing the field slice to be plotted.
        t (int): The time step index.
        name (str): The name of the field being plotted.
        path (str): The directory path where the plot will be saved.
        world (dict): A dictionary containing the dimensions of the world with keys 'x_wind' and 'y_wind'.
        dt (float): The time step duration.

    Returns:
        None
    """

    def plot_and_save(field_slice, t, name, path, world, dt):
        if not os.path.exists(f"{path}/data/{name}_slice"):
            os.makedirs(f"{path}/data/{name}_slice")
        # Create directory if it doesn't exist

        plt.title(f'{name} at t={t*dt:.2e}s')
        plt.imshow(jnp.swapaxes(field_slice, 0, 1), origin='lower', 
                   extent=[-world['x_wind']/2, world['x_wind']/2, 
                           -world['y_wind']/2, world['y_wind']/2])
        plt.colorbar(label=name)
        plt.tight_layout()
        plt.savefig(f'{path}/data/{name}_slice/{name}_slice_{t:09}.png', dpi=300)
        plt.clf()  # Clear the current figure
        plt.close('all')  # Close all figures to free up memory

    # Use JAX debug callback to execute the plotting function
    return jax.debug.callback(plot_and_save, field_slice, t, name, path, world, dt, ordered=True)


def write_slice(field_slice, x1, x2, t, name, path, dt):
    """
    Plots a slice of a field and saves it in VTK format.
    Parameters:
    field_slice (numpy.ndarray): The 2D slice of the field to be plotted.
    x1 (numpy.ndarray): The x-coordinates of the field slice.
    x2 (numpy.ndarray): The y-coordinates of the field slice.
    t (int): The time step or index for the slice.
    name (str): The name of the field or slice.
    path (str): The directory path where the VTK file will be saved.
    dt (float): The time step size (not used in the function but included in parameters).
    Returns:
    None
    """

    x3 = np.zeros(1)

    field_slice = jnp.expand_dims(field_slice, axis=-1)

    gridToVTK(f"{path}/data/{name}_slice/{name}_slice_{t:09}", x1, x2, x3,  \
            cellData = {f"{name}" : field_slice})
    # plot the slice of the field in the vtk file format


@partial(jax.jit, static_argnums=(0))
def write_data(filename, time, data):
    """
    Write the given time and data to a file using JAX's callback mechanism.
    This function is designed to be used with JAX's just-in-time compilation (jit) to optimize performance.

    Args:
        filename (str): The name of the file to write to.
        time (float): The time value to write.
        data (any): The data to write.

    Returns:
        None
    """

    def write_to_file(filename, time, data):
        with open(filename, "a") as f:
            f.write(f"{time}, {data}\n")
    # Open the file in append mode
    # Write the time and data to the file
    # Close the file
    return jax.debug.callback(write_to_file, filename, time, data, ordered=True)


@partial(jax.jit, static_argnums=(18) )
def save_datas(t, dt, particles, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz, E_grid, B_grid, plotting_parameters, world, constants, output_dir):
    """
    Save and process simulation data, including energy, field, and current density information.

    It uses conditional logic to determine which data to process and save based on the provided
    plotting parameters.

    Args:
        t (int): Current time step.
        dt (float): Time step size.
        particles (list): List of particle objects, each containing methods for computing properties like kinetic energy.
        Ex, Ey, Ez (ndarray): Electric field components on the grid.
        Bx, By, Bz (ndarray): Magnetic field components on the grid.
        rho (ndarray): Charge density on the grid.
        Jx, Jy, Jz (ndarray): Current density components on the grid.
        E_grid (tuple): Electric field grid dimensions.
        B_grid (tuple): Magnetic field grid dimensions.
        plotting_parameters (dict): Dictionary of flags controlling which data to process and save.
            Keys include:
            - 'plotenergy': Whether to compute and save energy data.
            - 'plotfields': Whether to compute and save field-related data.
            - 'plotcurrent': Whether to compute and save current density data.
        world (dict): Dictionary containing simulation world parameters, including:
            - 'dx', 'dy', 'dz': Grid spacings in x, y, and z directions.
            - 'Nx', 'Ny', 'Nz': Number of grid points in x, y, and z directions.
        constants (dict): Dictionary of physical constants, including:
            - 'eps': Permittivity of free space.
            - 'mu': Permeability of free space.
        output_dir (str): Directory where output data files will be saved.

    Returns:
        None
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']
    # Get the grid dimensions

    ##########################################################################################################
    def compute_and_write_energy():
        E2_integral = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(Ex**2 + Ey**2 + Ez**2, dx=dz), dx=dy), dx=dx)
        B2_integral = jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(Bx**2 + By**2 + Bz**2, dx=dz), dx=dy), dx=dx)
        # Integral of E^2 and B^2 over the entire grid
        e_energy = 0.5 * constants['eps'] * E2_integral
        b_energy = 0.5 / constants['mu'] * B2_integral
        # Electric and magnetic field energy

        #PE = jnp.sum( jnp.sum( jnp.sum( Ex**2 + Ey**2 + Ez**2 ) ) ) * 0.5 * constants['eps']
        # sum field using Lubos method


        kinetic_energy = sum(particle.kinetic_energy() for particle in particles)
        # Kinetic energy of the particles
        write_data(f"{output_dir}/data/total_energy.txt", t * dt, e_energy + b_energy + kinetic_energy)
        write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
        #write_data(f"{output_dir}/data/efield2.txt", t * dt, PE)
        # Write the electric field energy to a file
        write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
        write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)
        # Write the total energy to a file

    jax.lax.cond(
        plotting_parameters['plotenergy'],
        lambda _: compute_and_write_energy(),
        lambda _: None,
        operand=None
    )
    # Compute and write the energy of the system

    ##########################################################################################################
    def plot_fields(t, dt, Ex, Ey, Ez, Bx, By, Bz, Nx, Ny, Nz, output_dir):
        """
        Plot and save the electric and magnetic fields, as well as their averages.

        Args:
            t (int): Current time step.
            dt (float): Time step size.
            Ex, Ey, Ez (ndarray): Electric field components.
            Bx, By, Bz (ndarray): Magnetic field components.
            Nx, Ny, Nz (int): Grid dimensions.
            output_dir (str): Directory to save the data.

        Returns:
            None
        """
        write_data(f"{output_dir}/data/averageEx.txt", t*dt, jnp.mean(jnp.abs(Ex)))
        write_data(f"{output_dir}/data/averageEy.txt", t*dt, jnp.mean(jnp.abs(Ey)))
        write_data(f"{output_dir}/data/averageEz.txt", t*dt, jnp.mean(jnp.abs(Ez)))
        write_data(f"{output_dir}/data/averageBx.txt", t*dt, jnp.mean(jnp.abs(Bx)))
        write_data(f"{output_dir}/data/averageBy.txt", t*dt, jnp.mean(jnp.abs(By)))
        write_data(f"{output_dir}/data/averageBz.txt", t*dt, jnp.mean(jnp.abs(Bz)))
        # write_data(f"{output_dir}/data/averageE.txt", t*dt, jnp.mean(jnp.sqrt(Ex**2 + Ey**2 + Ez**2)))
        # write_data(f"{output_dir}/data/averageB.txt", t*dt, jnp.mean(jnp.sqrt(Bx**2 + By**2 + Bz**2)))
        write_data(f"{output_dir}/data/Eprobe.txt", t*dt, magnitude_probe(Ex, Ey, Ez, Nx-1, Ny-1, Nz-1))
        write_data(f"{output_dir}/data/centerE.txt", t*dt, magnitude_probe(Ex, Ey, Ez, Nx//2, Ny//2, Nz//2))

        # write_slice(Ex[:, :, Nz//2], E_grid[0], E_grid[1], t, 'Ex', output_dir, dt)
        # write_slice(Ey[:, :, Nz//2], E_grid[0], E_grid[1], t, 'Ey', output_dir, dt)
        # write_slice(Ez[:, :, Nz//2], E_grid[0], E_grid[1], t, 'Ez', output_dir, dt)

        plot_slice(Ex[:, :, Nz//2], t, 'Ex', output_dir, world, dt)
        plot_slice(Ey[:, :, Nz//2], t, 'Ey', output_dir, world, dt)
        plot_slice(Ez[:, :, Nz//2], t, 'Ez', output_dir, world, dt)

    jax.lax.cond(
        plotting_parameters['plotfields'],
        lambda _: plot_fields(t, dt, Ex, Ey, Ez, Bx, By, Bz, Nx, Ny, Nz, output_dir),
        lambda _: None,
        operand=None
    )
    # Plot and save energy-related data


    ##############################################################################################################################
    def write_current_data(t, dt, Jx, Jy, Jz, output_dir):
        """
        Write the mean current density components to files.

        Args:
            t (int): Current time step.
            dt (float): Time step size.
            Jx, Jy, Jz (ndarray): Current density components.
            output_dir (str): Directory to save the data.

        Returns:
            None
        """
        write_data(f"{output_dir}/data/Jx_mean.txt", t*dt, jnp.mean(Jx))
        write_data(f"{output_dir}/data/Jy_mean.txt", t*dt, jnp.mean(Jy))
        write_data(f"{output_dir}/data/Jz_mean.txt", t*dt, jnp.mean(Jz))

    jax.lax.cond(
        plotting_parameters['plotcurrent'],
        lambda _: write_current_data(t, dt, Jx, Jy, Jz, output_dir),
        lambda _: None,
        operand=None
    )

    
# @partial(jax.jit, static_argnums=(12))
def plotter(t, particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters):
    """
    Plots and saves various simulation data at specified intervals.

    Parameters:
    t (int): Current time step.
    particles (list): List of particles in the simulation.
    Ex, Ey, Ez (ndarray): Electric field components.
    Bx, By, Bz (ndarray): Magnetic field components.
    Jx, Jy, Jz (ndarray): Current density components.
    rho (ndarray): Charge density.
    phi (ndarray): Electric potential.
    E_grid (ndarray): Electric field grid.
    B_grid (ndarray): Magnetic field grid.
    world (dict): Dictionary containing world parameters.
    constants (dict): Dictionary containing physical constants.
    plotting_parameters (dict): Dictionary containing plotting parameters.
    simulation_parameters (dict): Dictionary containing simulation parameters.
    solver (object): Solver object used in the simulation.
    bc (object): Boundary conditions object.

    Returns:
    None
    """

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']

    jax.lax.cond(
        plotting_parameters['plotting'],

        lambda _: jax.lax.cond(
            t % plotting_parameters['plotting_interval'] == 0,
            lambda _: save_datas(t, dt, particles, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz, E_grid, B_grid, plotting_parameters, world, constants, output_dir),
            lambda _: None,
            operand=None
        ),

        lambda _: None,
        operand=None
    )


    if t % plotting_parameters['plotting_interval'] == 0:
        if plotting_parameters['phaseSpace']:
                #particles_phase_space([particles[0]], world, t, "Particles", output_dir)
                write_particles_phase_space(particles, t, output_dir)
    # Really need to make this jit compatible

    # if plotting_parameters['plotting']:
    #     if t % plotting_parameters['plotting_interval'] == 0:
    #         save_datas(t, dt, particles, Ex, Ey, Ez, Bx, By, Bz, rho, Jx, Jy, Jz, E_grid, B_grid, plotting_parameters, world, constants, output_dir)
 