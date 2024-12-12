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
import os, sys
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import vtk
import vtkmodules.util.numpy_support as vtknp
from jax.tree_util import tree_map
# import external libraries

from PyPIC3D.particle import initial_particles, particle_species

def particle_sanity_check(particles):
    """
    Perform a sanity check on the particles to ensure consistency in their attributes.

    This function iterates over each species in the particles list and checks that the 
    number of particles matches the shape of their position and velocity arrays.

    Parameters:
    particles (list): A list of species objects, where each species object must have 
                      the following methods:
                      - get_number_of_particles(): returns the number of particles (int)
                      - get_position(): returns a tuple of numpy arrays (x, y, z) representing 
                        the positions of the particles
                      - get_velocity(): returns a tuple of numpy arrays (vx, vy, vz) representing 
                        the velocities of the particles

    Raises:
    AssertionError: If the shapes of the position and velocity arrays do not match the 
                    number of particles.
    """

    for species in particles:
        N = species.get_number_of_particles()
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        assert x.shape == y.shape == z.shape == vx.shape == vy.shape == vz.shape == (N,)


def print_stats(world):
    """
    Print the statistics of the simulation world.
    Parameters:
    world (dict): A dictionary containing the following keys:
        - 'Nt' (int): Number of time steps.
        - 'dx' (float): Resolution in the x-direction.
        - 'dy' (float): Resolution in the y-direction.
        - 'dz' (float): Resolution in the z-direction.
        - 'dt' (float): Time step size.
        - 'x_wind' (float): Size of the window in the x-direction.
        - 'y_wind' (float): Size of the window in the y-direction.
        - 'z_wind' (float): Size of the window in the z-direction.
    Prints:
    The time window, x window, y window, z window, and resolution details (dx, dy, dz, dt, Nt).
    """

    Nt = world['Nt']
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    t_wind = Nt*dt
    print(f'time window: {t_wind}')
    print(f'x window: {x_wind}')
    print(f'y window: {y_wind}')
    print(f'z window: {z_wind}')
    print(f"\nResolution")
    print(f'dx: {dx}')
    print(f'dy: {dy}')
    print(f'dz: {dz}')
    print(f'dt:          {dt}')
    print(f'Nt:          {Nt}')

def check_stability(world, constants, electrons, dt):
    """
    Check the stability of the simulation based on various physical parameters.

    Parameters:
    world (dict): A dictionary containing the spatial resolution and wind parameters.
        - 'dx' (float): Spatial resolution in the x-direction.
        - 'dy' (float): Spatial resolution in the y-direction.
        - 'dz' (float): Spatial resolution in the z-direction.
        - 'x_wind' (float): Wind speed in the x-direction.
        - 'y_wind' (float): Wind speed in the y-direction.
        - 'z_wind' (float): Wind speed in the z-direction.
    constants (dict): A dictionary containing physical constants.
        - 'eps' (float): Permittivity of free space.
        - 'kb' (float): Boltzmann constant.
    electrons (object): An object representing the electrons in the simulation.
        - get_charge() (method): Returns the charge of an electron.
        - get_mass() (method): Returns the mass of an electron.
        - get_temperature() (method): Returns the temperature of the electrons.
        - get_number_of_particles() (method): Returns the number of electrons.
    dt (float): Time step of the simulation.

    Prints:
    - Warnings about numerical stability if the number of electrons is low or if the Debye length is less than the spatial resolution.
    - Theoretical plasma frequency.
    - Debye length.
    - Thermal velocity.
    - Number of electrons.
    """
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
    q_e = electrons.get_charge()
    me = electrons.get_mass()
    eps = constants['eps']
    Te = electrons.get_temperature()
    kb = constants['kb']

    theoretical_freq = plasma_frequency(electrons, world, constants)
    if theoretical_freq * dt > 2.0:
        print(f"# of Electrons is Low and may introduce numerical stability")
        print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")

    debye = debye_length(electrons, world, constants)
    if debye < dx:
        print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")

    print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")
    print(f"Debye Length: {debye} m")
    thermal_velocity = jnp.sqrt(3*kb*Te/me)
    print(f"Thermal Velocity: {thermal_velocity}\n")
    print(f"Number of Electrons: {electrons.get_number_of_particles()}")
    return theoretical_freq, debye, thermal_velocity

def convert_to_jax_compatible(data):
    """
    Convert a dictionary to a JAX-compatible PyTree.

    Parameters:
    - data (dict): The dictionary to convert.

    Returns:
    - dict: The JAX-compatible PyTree.
    """
    return tree_map(lambda x: jnp.array(x) if isinstance(x, (int, float, list, tuple)) else x, data)


def load_rectilinear_grid(file_path):
    """
    Load a rectilinear grid from a VTK file and extract the vector field components.
    Parameters:
    file_path (str): The path to the VTK file containing the rectilinear grid.
    Returns:
    tuple: A tuple containing three numpy arrays (field_x, field_y, field_z) representing
           the x, y, and z components of the vector field, respectively. Each array is
           reshaped to match the dimensions of the grid.
    """
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    rectilinear_grid = reader.GetOutput()
    x = vtknp.vtk_to_numpy(rectilinear_grid.GetXCoordinates())
    y = vtknp.vtk_to_numpy(rectilinear_grid.GetYCoordinates())
    z = vtknp.vtk_to_numpy(rectilinear_grid.GetZCoordinates())
    data = vtknp.vtk_to_numpy(rectilinear_grid.GetPointData().GetVectors())
    field_x = data[:, 0].reshape(len(x), len(y), len(z))
    field_y = data[:, 1].reshape(len(x), len(y), len(z))
    field_z = data[:, 2].reshape(len(x), len(y), len(z))
    return field_x, field_y, field_z


def check_nyquist_criterion(Ex, Ey, Ez, Bx, By, Bz, world):
    """
    Check if the E and B fields meet the Nyquist criterion.

    Parameters:
    Ex (ndarray): The electric field component in the x-direction.
    Ey (ndarray): The electric field component in the y-direction.
    Ez (ndarray): The electric field component in the z-direction.
    Bx (ndarray): The magnetic field component in the x-direction.
    By (ndarray): The magnetic field component in the y-direction.
    Bz (ndarray): The magnetic field component in the z-direction.
    world (dict): A dictionary containing the spatial resolution parameters.
        - 'dx' (float): Spatial resolution in the x-direction.
        - 'dy' (float): Spatial resolution in the y-direction.
        - 'dz' (float): Spatial resolution in the z-direction.

    Returns:
    bool: True if the fields meet the Nyquist criterion, False otherwise.
    """
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    nx, ny, nz = Ex.shape

    # Calculate the maximum wavenumber that can be resolved
    kx_max = jnp.pi / dx
    ky_max = jnp.pi / dy
    kz_max = jnp.pi / dz

    # Calculate the wavenumber components of the fields
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=dy) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz, d=dz) * 2 * jnp.pi

    # Check if the wavenumber components exceed the maximum wavenumber
    for field_name, field in {'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Bx': Bx, 'By': By, 'Bz': Bz}.items():
        kx_field = jnp.fft.fftn(field, axes=(0,))
        ky_field = jnp.fft.fftn(field, axes=(1,))
        kz_field = jnp.fft.fftn(field, axes=(2,))
        if jnp.any(jnp.abs(kx_field) > kx_max) or jnp.any(jnp.abs(ky_field) > ky_max) or jnp.any(jnp.abs(kz_field) > kz_max):
            print(f"Warning: The {field_name} field does not meet the Nyquist criterion. FFT may introduce aliasing.")


# Define the function to read the TOML file and convert it to a DataFrame
def read_toml_to_dataframe(toml_file):
    """
    Reads a TOML file and converts it to a pandas DataFrame.

    Parameters:
    - toml_file (str): Path to the TOML file.

    Returns:
    - pd.DataFrame: DataFrame containing the TOML data.
    """
    # Read the TOML file
    data = toml.load(toml_file)

    # Convert the TOML data to a pandas DataFrame
    df = pd.json_normalize(data, sep='_')
    # Transpose the DataFrame to swap rows and columns
    df = df.transpose()
    return df

def cylindrical_to_cartesian_matrix(r, theta, z):
    """
    Create a transformation matrix to convert cylindrical coordinates (r, theta, z) to Cartesian coordinates (x, y, z).

    Parameters:
    r (float): Radial distance in cylindrical coordinates.
    theta (float): Angle in radians in cylindrical coordinates.
    z (float): Height in cylindrical coordinates.

    Returns:
    ndarray: A 3x3 transformation matrix.
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    transformation_matrix = jnp.array([
        [cos_theta, -r * sin_theta, 0],
        [sin_theta, r * cos_theta, 0],
        [0, 0, 1]
    ])

    return transformation_matrix

def cartesian_to_cylindrical_vector_field(vector_field, x, y, z):
    """
    Transform a vector field from Cartesian coordinates (x, y, z) to cylindrical coordinates (r, theta, z).

    Parameters:
    vector_field (ndarray): The vector field in Cartesian coordinates.
    x (ndarray): The x-coordinates.
    y (ndarray): The y-coordinates.
    z (ndarray): The z-coordinates.

    Returns:
    ndarray: The vector field in cylindrical coordinates.
    """
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Transformation matrix from Cartesian to cylindrical coordinates
    transformation_matrix = jnp.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Apply the transformation to the vector field
    cylindrical_vector_field = jnp.einsum('ij,j...->i...', transformation_matrix, vector_field)

    return cylindrical_vector_field

def convert_spatial_resolution(dx1, dx2, dx3, from_system, to_system):
    """
    Convert the spatial resolution parameters from one coordinate system to another.

    Parameters:
    dx (float): Grid spacing in the x-direction.
    dy (float): Grid spacing in the y-direction.
    dz (float): Grid spacing in the z-direction.
    from_system (str): The original coordinate system ('cartesian' or 'cylindrical').
    to_system (str): The target coordinate system ('cartesian' or 'cylindrical').

    Returns:
    tuple: Converted spatial resolution parameters (dx, dy, dz) in the target coordinate system.
    """
    if from_system == to_system:
        return dx1, dx2, dx3

    if from_system == 'cartesian' and to_system == 'cylindrical':
        dr = jnp.sqrt(dx1**2 + dx2**2)
        dtheta = jnp.arctan2(dx2, dx1)
        return dr, dtheta, dx3

    if from_system == 'cylindrical' and to_system == 'cartesian':
        dx = dx1 * jnp.cos(dx2)
        dy = dx2 * jnp.sin(dx2)
        dz = dx3
        return dx, dy, dz

    raise ValueError("Invalid coordinate system conversion")

def build_coallocated_grid(world):
    """
    Builds a co-allocated grid based on the provided world parameters.
    Parameters:
    world (dict): A dictionary containing the following keys:
        - 'dx' (float): The grid spacing in the x-direction.
        - 'dy' (float): The grid spacing in the y-direction.
        - 'dz' (float): The grid spacing in the z-direction.
        - 'x_wind' (float): The extent of the grid in the x-direction.
        - 'y_wind' (float): The extent of the grid in the y-direction.
        - 'z_wind' (float): The extent of the grid in the z-direction.
    Returns:
    tuple: A tuple containing two elements:
        - grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.
        - grid (tuple): A duplicate of the first grid tuple.
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the grid parameters
    grid = jnp.arange(-x_wind/2, x_wind/2, dx), jnp.arange(-y_wind/2, y_wind/2, dy), jnp.arange(-z_wind/2, z_wind/2, dz)
    # create the grid space
    return grid, grid

def build_yee_grid(world):
    """
    Builds a Yee grid and a staggered Yee grid based on the provided world parameters.
    Parameters:
    world (dict): A dictionary containing the following keys:
        - 'dx' (float): Grid spacing in the x-direction.
        - 'dy' (float): Grid spacing in the y-direction.
        - 'dz' (float): Grid spacing in the z-direction.
        - 'x_wind' (float): Extent of the grid in the x-direction.
        - 'y_wind' (float): Extent of the grid in the y-direction.
        - 'z_wind' (float): Extent of the grid in the z-direction.
    Returns:
    tuple: A tuple containing two elements:
        - grid (tuple of jnp.ndarray): The Yee grid with three arrays representing the x, y, and z coordinates.
        - staggered_grid (tuple of jnp.ndarray): The staggered Yee grid with three arrays representing the x, y, and z coordinates.
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the grid parameters
    grid = jnp.arange(-x_wind/2, x_wind/2, dx), jnp.arange(-y_wind/2, y_wind/2, dy), jnp.arange(-z_wind/2, z_wind/2, dz)
    staggered_grid = jnp.arange(-x_wind/2 + dx/2, x_wind/2 + dx/2, dx), jnp.arange(-y_wind/2 + dy/2, y_wind/2 + dy/2, dy), jnp.arange(-z_wind/2 + dz/2, z_wind/2 + dz/2, dz)
    # create the grid space
    return grid, staggered_grid

def precondition(NN, phi, rho, model=None):
    """
    Precondition the Poisson equation using a neural network model.

    Parameters:
    NN (callable): The neural network model to be used for preconditioning.
    phi (ndarray): The potential field.
    rho (ndarray): The charge density.
    model (callable): The neural network model to be used for preconditioning.

    Returns:
    ndarray: The preconditioned potential field.
    """
    if model is None:
        return None
    else:
        return model(phi, rho)

def use_gpu_if_set(func):
    """
    Decorator to run a function on GPU using JAX if the `GPUs` flag is set to True.

    Parameters:
    func (callable): The function to be decorated.

    Returns:
    callable: The decorated function that runs on GPU if `GPUs` is True.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_gpu = kwargs.pop('GPUs', False)
        if use_gpu:
            with jax.default_device(jax.devices('gpu')[0]):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def fix_bc_and_jit_compile(func, bc_value):
    """
    Fixes the boundary condition argument of a function using functools.partial and then JIT compiles the new function.

    Parameters:
    func (callable): The original function that takes a boundary condition argument.
    bc_value (any): The value to fix for the boundary condition argument.

    Returns:
    callable: The JIT compiled function with the boundary condition argument fixed.
    """
    fixed_bc_func = partial(func, bc=bc_value)
    jit_compiled_func = jit(fixed_bc_func)
    return jit_compiled_func

def grab_particle_keys(config):
    """
    Extracts and returns a list of keys from the given configuration dictionary
    that start with the prefix 'particle'.
    Args:
        config (dict): A dictionary containing configuration keys and values.
    Returns:
        list: A list of keys from the configuration dictionary that start with 'particle'.
    """
    particle_keys = []
    for key in config.keys():
        if key[:8] == 'particle':
            particle_keys.append(key)
    return particle_keys

def grab_field_keys(config):
    """
    Extracts and returns a list of keys from the given configuration dictionary
    that start with the prefix 'field'.
    Args:
        config (dict): A dictionary containing configuration keys and values.
    Returns:
        list: A list of keys from the config dictionary that start with 'field'.
    """
    field_keys = []
    for key in config.keys():
        if key[:5] == 'field':
            field_keys.append(key)
    return field_keys

def load_external_fields_from_toml(fields, toml_file):
    """
    Load external fields from a TOML file.

    Parameters:
    - toml_file (str): Path to the TOML file.

    Returns:
    - dict: Dictionary containing the external fields.
    """
    config = toml.load(toml_file)

    field_keys = grab_field_keys(config)

    for toml_key in field_keys:
        field_name = config[toml_key]['name']
        field_type = config[toml_key]['type']
        field_path = config[toml_key]['path']
        print(f"Loading field: {field_name} from {field_path}")

        external_field = jnp.load(field_path)

        fields[field_type] += external_field
        print(f"Field loaded successfully: {field_name}")

    return fields

def load_particles_from_toml(toml_file, simulation_parameters, world, constants):
    """
    Load particle data from a TOML file and initialize particle species.
    Args:
        toml_file (str): Path to the TOML file containing particle configuration.
        simulation_parameters (dict): Dictionary containing simulation parameters.
        world (dict): Dictionary containing world parameters such as 'x_wind', 'y_wind', 'z_wind', 'dx', 'dy', 'dz'.
        constants (dict): Dictionary containing constants such as 'kb'.
    Returns:
        list: A list of particle_species objects initialized with the data from the TOML file.
    The function reads particle configuration from the provided TOML file, initializes particle properties such as
    position, velocity, charge, mass, and temperature. It also handles loading initial positions and velocities from
    external sources if specified in the TOML file. The particles are then appended to a list and returned.
    """

    config = toml.load(toml_file)

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    kb = constants['kb']
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    i = 0
    particles = []
    particle_keys = grab_particle_keys(config)

    for toml_key in particle_keys:
        key1, key2, key3 = jax.random.PRNGKey(i), jax.random.PRNGKey(i+1), jax.random.PRNGKey(i+2)
        i += 3
        # build the particle random number generator keys
        particle_name = config[toml_key]['name']
        N_particles=config[toml_key]['N_particles']
        charge=config[toml_key]['charge']
        mass=config[toml_key]['mass']
        T=config[toml_key]['temperature']
        x, y, z, vx, vy, vz = initial_particles(N_particles, x_wind, y_wind, z_wind, mass, T, kb, key1, key2, key3)

        if 'initial_x' in config[toml_key]:
            print(f"Loading initial_x from external source: {config[toml_key]['initial_x']}")
            x = jnp.load(config[toml_key]['initial_x'])
        if 'initial_y' in config[toml_key]:
            print(f"Loading initial_y from external source: {config[toml_key]['initial_y']}")
            y = jnp.load(config[toml_key]['initial_y'])
        if 'initial_z' in config[toml_key]:
            print(f"Loading initial_z from external source: {config[toml_key]['initial_z']}")
            z = jnp.load(config[toml_key]['initial_z'])
        if 'initial_vx' in config[toml_key]:
            print(f"Loading initial_vx from external source: {config[toml_key]['initial_vx']}")
            vx = jnp.load(config[toml_key]['initial_vx'])
        if 'initial_vy' in config[toml_key]:
            print(f"Loading initial_vy from external source: {config[toml_key]['initial_vy']}")
            vy = jnp.load(config[toml_key]['initial_vy'])
        if 'initial_vz' in config[toml_key]:
            print(f"Loading initial_vz from external source: {config[toml_key]['initial_vz']}")
            vz = jnp.load(config[toml_key]['initial_vz'])
        print('\n')

        update_pos = True
        update_v   = True

        if update_pos in config[toml_key]:
            update_pos = config[toml_key]['update_pos']
        if update_v in config[toml_key]:
            update_v = config[toml_key]['update_v']

        zeta1 = ( x + x_wind/2 ) % dx
        zeta2 = zeta1
        eta1  = ( y + y_wind/2 ) % dy
        eta2  = eta1
        xi1   = ( z + z_wind/2 ) % dz
        xi2   = xi1
        subcells = zeta1, zeta2, eta1, eta2, xi1, xi2

        particle = particle_species(
            name=particle_name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            T=T,
            x1=x,
            x2=y,
            x3=z,
            v1=vx,
            v2=vy,
            v3=vz,
            subcells=subcells,
            xwind=x_wind,
            ywind=y_wind,
            zwind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            bc='periodic',
            update_pos=update_pos,
            update_v=update_v
        )
        particles.append(particle)

    return particles

def debugprint(value):
    """
    Prints the given value using JAX's debug print functionality.

    Args:
        value: The value to be printed. Can be of any type that is supported by JAX's debug print.
    """
    jax.debug.print('{x}', x=value)

def update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters, constants):
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

    if "constants" in config:
        for key, value in config["constants"].items():
            if key in constants:
                constants[key] = value

    return simulation_parameters, plotting_parameters, constants

def dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters, constants, particles):
    """
    Dump the simulation, plotting parameters, and particle species into an output TOML file.

    Parameters:
    - simulation_stats (dict): Dictionary of simulation statistics.
    - simulation_parameters (dict): Dictionary of simulation parameters.
    - plotting_parameters (dict): Dictionary of plotting parameters.
    - constants (dict): Dictionary of constants.
    - particles (list): List of particle species.
    """

    output_path = simulation_parameters["output_dir"]
    output_file = os.path.join(output_path, "output.toml")

    config = {
        "simulation_stats": simulation_stats,
        "simulation_parameters": simulation_parameters,
        "plotting": plotting_parameters,
        "constants": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, constants),
        "particles": []
    }

    for particle in particles:
        particle_dict = {
            "name": particle.name,
            "N_particles": particle.N_particles,
            "charge": particle.charge,
            "mass": particle.mass,
            "temperature": particle.T,
            "update_pos": particle.update_pos,
            "update_v": particle.update_v
        }
        config["particles"].append(particle_dict)

    with open(output_file, 'w') as f:
        toml.dump(config, f)


@jit
def interpolate_field(field, grid, x, y, z):
    """
    Interpolates the given field at the specified (x, y, z) coordinates using a regular grid interpolator.
    Args:
        field (array-like): The field values to be interpolated.
        grid (tuple of array-like): The grid points for each dimension (x, y, z).
        x (array-like): The x-coordinates where interpolation is desired.
        y (array-like): The y-coordinates where interpolation is desired.
        z (array-like): The z-coordinates where interpolation is desired.
    Returns:
        array-like: Interpolated values at the specified (x, y, z) coordinates.
    """

    interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, field, fill_value=0)
    # create the interpolator
    points = jnp.stack([x, y, z], axis=-1)
    return interpolate(points)

@jit
def interpolate_and_stagger_field(field, grid, staggered_grid):
    """
    Interpolates a given field defined on a grid to a staggered grid.

    Parameters:
    field (array-like): The field values defined on the original grid.
    grid (array-like): The coordinates of the original grid.
    staggered_grid (array-like): The coordinates of the staggered grid where the field needs to be interpolated.

    Returns:
    array-like: The interpolated field values on the staggered grid.
    """

    interpolate = jax.scipy.interpolate.RegularGridInterpolator(grid, field, fill_value=0)
    # create the interpolator
    mesh = jnp.meshgrid( *staggered_grid, indexing='ij')
    points = jnp.stack(mesh, axis=-1)
    # get the points for the interpolation
    staggered_field = interpolate( points)
    # interpolate the field to the staggered grid
    return staggered_field

def courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants):
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
    C = constants['C']
    return courant_number / (C * ( (1/dx) + (1/dy) + (1/dz) ) )
# calculate the courant condition

def plasma_frequency(electrons, world, constants):
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

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    q_e = electrons.get_charge()
    N_electrons = electrons.get_number_of_particles()
    me = electrons.get_mass()
    eps = constants['eps']
    ne = N_electrons / (x_wind*y_wind*z_wind)
    return jnp.sqrt(  ne * q_e**2  / (eps*me)  )
# calculate the expected plasma frequency from analytical theory

def debye_length(electrons, world, constants):
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

    q_e = electrons.get_charge()
    N_electrons = electrons.get_number_of_particles()
    Te = electrons.get_temperature()
    eps = constants['eps']
    kb = constants['kb']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    ne = N_electrons / (x_wind*y_wind*z_wind)
    return jnp.sqrt( eps * kb * Te / (ne * q_e**2) )