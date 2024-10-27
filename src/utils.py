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
# import external libraries

from src.particle import initial_particles, particle_species

def build_grid(x_wind, y_wind, z_wind, dx, dy, dz):
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

def load_particles_from_toml(toml_file, simulation_parameters, dx, dy, dz):
    config = toml.load(toml_file)

    x_wind = simulation_parameters['x_wind']
    y_wind = simulation_parameters['y_wind']
    z_wind = simulation_parameters['z_wind']
    kb = simulation_parameters['kb']

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
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
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

def dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters):
    """
    Dump the simulation and plotting parameters into an output TOML file.

    Parameters:
    - output_file (str): Path to the output TOML file.
    - simulation_parameters (dict): Dictionary of simulation parameters.
    - plotting_parameters (dict): Dictionary of plotting parameters.
    """

    output_path = simulation_parameters["output_dir"]
    output_file = os.path.join(output_path, "output.toml")

    config = {
        "simulation_stats": simulation_stats,
        "simulation_parameters": simulation_parameters,
        "plotting": plotting_parameters
    }
    
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
