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
# import external libraries

from PyPIC3D.particle import initial_particles, particle_species

def load_rectilinear_grid(file_path):
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

def build_grid(world):
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
    particle_keys = []
    for key in config.keys():
        if key[:8] == 'particle':
            particle_keys.append(key)
    return particle_keys

def load_particles_from_toml(toml_file, simulation_parameters, world, constants):
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
        
        update_pos = True
        update_v   = True

        if update_pos in config[toml_key]:
            update_pos = config[toml_key]['update_pos']
        if update_v in config[toml_key]:
            update_v = config[toml_key]['update_v']

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
            dx=dx,
            dy=dy,
            dz=dz,
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
        "constants": constants,
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

def courant_condition(courant_number, world, simulation_parameters, constants):
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
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
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