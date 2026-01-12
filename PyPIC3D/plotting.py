import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
from pyevtk.hl import gridToVTK, pointsToVTK
import os
import plotly.graph_objects as go
import jax
from functools import partial
import vtk
from vtk.util import numpy_support
import openpmd_api as io
import importlib.metadata

from PyPIC3D.utils import compute_energy

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

    return jax.debug.callback(write_to_file, filename, time, data, ordered=True)


def plot_vtk_particles(particles, t, path):
    """
    Plot the particles in VTK format.

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        path (str): The path to save the plot.

    Returns:
        None
    """
    if not os.path.exists(f"{path}/data/particles"):
        os.makedirs(f"{path}/data/particles")

    particle_names = [species.get_name().replace(" ", "") for species in particles]
    for species in particles:
        name = species.get_name().replace(" ", "")
        x, y, z = map(np.asarray, species.get_position())
        vx, vy, vz = map(np.asarray, species.get_velocity())
        # Get the position and velocity of the particles
        q = np.asarray( species.get_charge() * np.ones_like(vx) )
        # Get the charge of the particles

        pointsToVTK(f"{path}/data/particles/{name}.{t:09}", x, y, z, \
                {"v": (vx, vy, vz), "q": q})
        # save the particles in the vtk file format


def plot_field_slice_vtk(field_slices, field_names, slice, grid, t, name, path, world):
    """
    Plot a slice of a field in VTK format using Python VTK library.

    Args:
        field_slices (list): List of 2D field slices to be plotted.
        field_names (list): List of field names corresponding to the slices.
        slice (int): Slice direction (0=x-slice, 1=y-slice, 2=z-slice, 3=full 3D).
        grid (tuple): The grid dimensions (x, y, z).
        t (int): The time step.
        name (str): The name of the field.
        path (str): The path to save the plot.
        world (dict): World parameters containing grid information.

    Returns:
        None
    """

    x, y, z = grid
    nx, ny, nz = world['Nx'], world['Ny'], world['Nz']
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    if not os.path.exists(f"{path}/data/{name}_slice"):
        os.makedirs(f"{path}/data/{name}_slice")
    # Create directory if it doesn't exist

    # Create VTK structured grid
    structured_grid = vtk.vtkStructuredGrid()

    if slice == 0:
        x = np.asarray([x[nx//2]])
    elif slice == 1:
        y = np.asarray([y[ny//2]])
    elif slice == 2:
        z = np.asarray([z[nz//2]])

    structured_grid.SetDimensions(x.shape[0], y.shape[0], z.shape[0])
    # Set the dimensions of the structured grid based on the slice type

    # Efficiently create all grid points using numpy meshgrid and bulk insert
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    structured_grid.SetPoints(points)
    # Create points for the structured grid based on the slice type

    for idx, (field_slice, field_name) in enumerate(zip(field_slices, field_names)):
        # Ensure field_slice is 2D and handle VTK ordering
        field_data = np.asarray(field_slice)
        if field_data.ndim == 2:
            # VTK expects data in (k,j,i) order, but our slice is typically (j,k)
            field_data = field_data.T.flatten()
        else:
            field_data = field_data.flatten()

        vtk_array = numpy_support.numpy_to_vtk(field_data)
        vtk_array.SetName(field_name)
        structured_grid.GetPointData().AddArray(vtk_array)

    # Write the VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(f"{path}/data/{name}_slice/{name}_slice_{t:09}.vtk")
    writer.SetInputData(structured_grid)
    writer.Write()


def plot_vectorfield_slice_vtk(field_slices, field_names, slice, grid, t, name, path, world):
    """
    Plot a slice of a field in VTK format as vector data using Python VTK library.

    Args:
        field_slices (list): List of 2D field slices to be plotted. Should be [Fx, Fy, Fz] for vector fields.
        field_names (list): List of field names corresponding to the slices (e.g., ['Ex', 'Ey', 'Ez']).
        slice (int): Slice direction (0=x-slice, 1=y-slice, 2=z-slice, 3=full 3D).
        grid (tuple): The grid dimensions (x, y, z).
        t (int): The time step.
        name (str): The name of the field.
        path (str): The path to save the plot.
        world (dict): World parameters containing grid information.

    Returns:
        None
    """

    x, y, z = grid
    nx, ny, nz = world['Nx'], world['Ny'], world['Nz']
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    if not os.path.exists(f"{path}/data/{name}_slice"):
        os.makedirs(f"{path}/data/{name}_slice")

    # Handle slicing
    if slice == 0:
        x = np.asarray([x[nx//2]])
    elif slice == 1:
        y = np.asarray([y[ny//2]])
    elif slice == 2:
        z = np.asarray([z[nz//2]])

    # Create VTK structured grid
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(x.shape[0], y.shape[0], z.shape[0])

    # Create grid points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    structured_grid.SetPoints(points)

    # Stack field slices as vector data
    # Each field_slice should be 2D, shape (len(x), len(y)) or similar
    # We flatten and stack them as (N, 3) for VTK vector
    for field_slice, field_name in zip(field_slices, field_names):
        field_arrays = []
        for comp in field_slice:
            field_data = np.asarray(comp)
            # convert to np array
            if field_data.ndim == 2:
                field_data = field_data.T.flatten()  # VTK expects Fortran order
            else:
                field_data = field_data.flatten()
            field_arrays.append(field_data)
        # Stack as (N, 3)
        vector_data = np.stack(field_arrays, axis=-1)
        # If shape is (N, 3), convert to VTK
        vtk_vector_array = numpy_support.numpy_to_vtk(vector_data, deep=True)
        vtk_vector_array.SetName(f"{field_name}_vector")
        structured_grid.GetPointData().AddArray(vtk_vector_array)

    # Write the VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(f"{path}/data/{name}_slice/{name}_slice_{t:09}.vtk")
    writer.SetInputData(structured_grid)
    writer.Write()


def write_openpmd_initial_particles(particles, world, output_dir, filename="initial_particles.h5"):
    """
    Write the initial particle states to separate openPMD files, one per species.

    Args:
        particles (list): List of particle species.
        world (dict): Dictionary containing the simulation world parameters.
        output_dir (str): Base output directory for the simulation.
        filename (str): Base name of the openPMD output file (species name is prepended).
    """
    if not particles:
        return

    output_path = os.path.join(output_dir, "data", "openpmd")
    os.makedirs(output_path, exist_ok=True)

    def make_array_writable(arr):
        arr = np.array(arr, dtype=np.float64, copy=True, order="C")
        arr.setflags(write=True)
        return arr

    for species in particles:
        species_name = species.get_name().replace(" ", "_")
        series_filename = f"{species_name}_{filename}"
        series_path = os.path.join(output_path, series_filename)

        series = io.Series(series_path, io.Access.create)
        series.set_attribute("software", "PyPIC3D")
        series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))

        iteration = series.iterations[0]
        iteration.time = 0.0
        iteration.dt = float(world["dt"])
        iteration.time_unit_SI = 1.0

        species_group = iteration.particles[species_name]

        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()

        x = make_array_writable(x)
        y = make_array_writable(y)
        z = make_array_writable(z)
        vx = make_array_writable(vx)
        vy = make_array_writable(vy)
        vz = make_array_writable(vz)

        num_particles = x.shape[0]
        particle_mass = float(species.mass)
        particle_charge = float(species.charge)

        position = species_group["position"]
        for component, data in zip(("x", "y", "z"), (x, y, z)):
            record_component = position[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

        # positionOffset: required by openPMD consumers (WarpX expects it)
        pos_off = species_group["positionOffset"]
        zeros = np.zeros(num_particles, dtype=np.float64)
        for comp in ("x", "y", "z"):
            rc = pos_off[comp]
            rc.reset_dataset(io.Dataset(zeros.dtype, [num_particles]))
            rc.store_chunk(zeros, [0], [num_particles])
            rc.unit_SI = 1.0

        momentum = species_group["momentum"]
        for component, data in zip(("x", "y", "z"), (vx, vy, vz)):
            record_component = momentum[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data * particle_mass, [0], [num_particles])
            record_component.unit_SI = 1.0

        weighting = species_group["weighting"]
        weights = np.full(num_particles, float(species.weight), dtype=np.float64)
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charges = np.full(num_particles, particle_charge, dtype=np.float64)
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charges, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        masses = np.full(num_particles, particle_mass, dtype=np.float64)
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(masses, [0], [num_particles])
        mass.unit_SI = 1.0

        series.flush()
        series.close()