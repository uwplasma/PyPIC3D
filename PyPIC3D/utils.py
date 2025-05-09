import jax
import jaxdecomp
import plotly
import tqdm
import pyevtk
from jax import jit
import argparse
import jax.numpy as jnp
import functools
from functools import partial
import toml
import os
#import pandas as pd
from jax.tree_util import tree_map
from datetime import datetime
import importlib.metadata
# import external libraries

def make_dir(path):
    """
    Create a directory if it does not exist.
    Args:
        path (str): The path to the directory to be created.
    """
    
    if not os.path.exists(path):
        os.makedirs(path)


def vth_to_T(vth, m, kb):
    """
    Convert thermal velocity to temperature.

    Args:
        vth (float): Thermal velocity.
        m (float): Mass of the particle.
        kb (float): Boltzmann constant.

    Returns:
        float: Temperature.
    """
    return m * vth**2 / (kb)

def T_to_vth(T, m, kb):
    """
    Convert temperature to thermal velocity.

    Args:
        T (float): Temperature.
        m (float): Mass of the particle.
        kb (float): Boltzmann constant.

    Returns:
        float: Thermal velocity.
    """
    return jnp.sqrt(kb * T / m)

def load_config_file():
    """
    Parses command-line arguments to get the path to a configuration file,
    loads the configuration file in TOML format, and returns its contents.

    Returns:
        dict: The contents of the configuration file as a dictionary.

    Raises:
        SystemExit: If the command-line arguments are not provided correctly.
        FileNotFoundError: If the specified configuration file does not exist.
        toml.TomlDecodeError: If the configuration file is not a valid TOML file.
    """
    parser = argparse.ArgumentParser(description="3D PIC code using Jax")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    # argument parser for the configuration file
    config_file = args.config
    # path to the configuration file
    print(f"Using Configuration File: {config_file}")
    toml_file = toml.load(config_file)
    # load the configuration file
    return toml_file

def if_verbose_print(verbose, string):
    """
    Conditionally prints a string based on the verbosity flag.

    Args:
        verbose (bool): A flag indicating whether to print the string.
        string (str): The string to be printed if verbose is True.

    Returns:
        None
    """
    jax.lax.cond(
        verbose,
        lambda _: print(string),
        lambda _: None,
        operand=None
    )

def particle_sanity_check(particles):
    """
    Perform a sanity check on the particles to ensure consistency in their attributes.

    This function iterates over each species in the particles list and checks that the 
    number of particles matches the shape of their position and velocity arrays.

    Args:
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
    
    Args:
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
    print(f'\ntime window: {t_wind} s with {Nt} time steps of {dt} s')
    print(f'x window: {x_wind} m with dx: {dx}')
    print(f'y window: {y_wind} m with dy: {dy}')
    print(f'z window: {z_wind} m with dz: {dz}\n')

def check_stability(plasma_parameters, dt):
    """
    Check the stability of the simulation based on various physical parameters.

    Args:
        plasma_parameters (dict): A dictionary containing various plasma parameters.
            - "Theoretical Plasma Frequency" (float): Theoretical plasma frequency.
            - "Debye Length" (float): Debye length.
            - "Thermal Velocity" (float): Thermal velocity.
            - "Number of Electrons" (int): Number of electrons.
            - "Temperature of Electrons" (float): Temperature of electrons.
            - "DebyePerDx" (float): Debye length per dx.
            - "DebyePerDy" (float): Debye length per dy.
            - "DebyePerDz" (float): Debye length per dz.
        dt (float): Time step of the simulation.

    Prints:
        Warnings about numerical stability if the number of electrons is low or if the Debye length is less than the spatial resolution.
        Theoretical plasma frequency.
        Debye length.
        Thermal velocity.
        Number of electrons.
    """
    theoretical_freq = plasma_parameters["Theoretical Plasma Frequency"]
    debye = plasma_parameters["Debye Length"]
    thermal_velocity = plasma_parameters["Thermal Velocity"]
    num_electrons = plasma_parameters["Number of Electrons"]
    dxperDebye = plasma_parameters["dx per debye length"]

    if theoretical_freq * dt > 2.0:
        print(f"# of Electrons is Low and may introduce numerical stability")
        # print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")

    if dxperDebye < 1:
        print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")

    print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")
    print(f"Debye Length: {debye} m")
    print(f"Thermal Velocity: {thermal_velocity}")
    print(f'Dx Per Debye Length: {dxperDebye}')
    print(f"Number of Electrons: {num_electrons}\n")


def build_plasma_parameters_dict(world, constants, electrons, dt):
    """
    Build a dictionary containing various plasma parameters.

    Args:
        world (dict): A dictionary containing the spatial resolution and wind parameters.
        constants (dict): A dictionary containing physical constants.
        electrons (object): An object representing the electrons in the simulation.
        dt (float): Time step of the simulation.

    Returns:
        dict: A dictionary containing the plasma parameters.
    """

    me = electrons.get_mass()
    Te = electrons.get_temperature()
    kb = constants['kb']
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    theoretical_freq = plasma_frequency(electrons, world, constants)
    debye = debye_length(electrons, world, constants)
    thermal_velocity = jnp.sqrt(3*kb*Te/me)

    plasma_parameters = {
        "Theoretical Plasma Frequency": theoretical_freq,
        "Debye Length": debye,
        "Thermal Velocity": thermal_velocity,
        "Number of Electrons": electrons.get_number_of_particles(),
        "Temperature of Electrons": electrons.get_temperature(),
        "dx per debye length": debye/dx,
        "dy per debye length": debye/dy,
        "dz per debye length": debye/dz,
    }

    return plasma_parameters

def convert_to_jax_compatible(data):
    """
    Convert a dictionary to a JAX-compatible PyTree.

    Args:
        data (dict): The dictionary to convert.

    Returns:
        dict: The JAX-compatible PyTree.
    """
    return tree_map(lambda x: jnp.array(x) if isinstance(x, (int, float, list, tuple)) else x, data)


def build_coallocated_grid(world):
    """
    Builds a co-allocated grid based on the provided world parameters.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
        func (callable): The original function that takes a boundary condition argument.
        bc_value (any): The value to fix for the boundary condition argument.

    Returns:
        callable: The JIT compiled function with the boundary condition argument fixed.
    """
    fixed_bc_func = partial(func, bc=bc_value)
    jit_compiled_func = jit(fixed_bc_func)
    return jit_compiled_func


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

def load_external_fields_from_toml(fields, config):
    """
    Load external fields from a TOML file.

    Args:
        fields (dict): Dictionary containing the external fields.
        config (dict): Dictionary containing the configuration values.

    Returns:
        dict: Dictionary containing the external fields.
    """

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

def debugprint(value):
    """
    Prints the given value using JAX's debug print functionality.

    Args:
        value: The value to be printed. Can be of any type that is supported by JAX's debug print.

    Returns:
        None
    """
    jax.debug.print('{x}', x=value)

def update_parameters_from_toml(config, simulation_parameters, plotting_parameters, constants):
    """
    Update the simulation parameters with values from a TOML config file.

    Args:
        config (dict): Dictionary containing the configuration values.
        simulation_parameters (dict): Dictionary of default simulation parameters.
        plotting_parameters (dict): Dictionary of default plotting parameters.

    Returns:
        tuple: Updated simulation parameters and plotting parameters.
    """

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

def dump_parameters_to_toml(simulation_stats, simulation_parameters, plasma_parameters, plotting_parameters, constants, particles):
    """
    Dump the simulation, plotting parameters, and particle species into an output TOML file.

    Args:
        simulation_stats (dict): Dictionary of simulation statistics.
        simulation_parameters (dict): Dictionary of simulation parameters.
        plotting_parameters (dict): Dictionary of plotting parameters.
        constants (dict): Dictionary of constants.
        particles (list): List of particle species.
    """

    output_path = simulation_parameters["output_dir"]
    output_file = os.path.join(output_path, "data/output.toml")

    config = {
        "simulation_stats": simulation_stats,
        "simulation_parameters": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, simulation_parameters),
        'plasma_parameters': jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plasma_parameters),
        "plotting": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plotting_parameters),
        "constants": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, constants),
        "particles": []
    }

    for particle in particles:
        particle_dict = {
            "name": particle.name,
            "N_particles": float(particle.N_particles),
            "weight": float(particle.weight),
            "charge": float(particle.charge),
            "mass": float(particle.mass),
            "temperature": float(particle.T),
            "scaled mass": float(particle.get_mass()),
            "scaled charge": float(particle.get_charge()),
            "update_pos": particle.update_pos,
            "update_v": particle.update_v
        }
        config["particles"].append(particle_dict)

    config["version"] = {
        "PyPIC3D_version": importlib.metadata.version('PyPIC3D'),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    # Get the versions of all the packages being imported
    package_versions = {
        "jax": jax.__version__,
        "jaxdecomp" : jaxdecomp.__version__,
        "toml": toml.__version__,
        "plotly": plotly.__version__,
        "tqdm": tqdm.__version__,
        "pyevtk": pyevtk.__version__,
    }

    config["package_versions"] = package_versions

    # print("Simulation Stats:", simulation_stats)
    # print("Simulation Parameters:", simulation_parameters)
    # print("Plasma Parameters:", plasma_parameters)
    # print("Plotting Parameters:", plotting_parameters)
    # print("Constants:", constants)

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



def create_trilinear_interpolator(field, grid):
    """
    Create a trilinear interpolation function for a given 3D field and grid.

    Args:
        field (ndarray): The 3D field to interpolate.
        grid (tuple): A tuple of three arrays representing the grid points in the x, y, and z directions.

    Returns:
        function: A function that takes (x, y, z) coordinates and returns the interpolated values.
    """
    x_grid, y_grid, z_grid = grid

    @jit
    def interpolator(x, y, z):
        x_idx = jnp.clip(jnp.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2)
        y_idx = jnp.clip(jnp.searchsorted(y_grid, y) - 1, 0, len(y_grid) - 2)
        z_idx = jnp.clip(jnp.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)

        x0, x1 = x_grid[x_idx], x_grid[x_idx + 1]
        y0, y1 = y_grid[y_idx], y_grid[y_idx + 1]
        z0, z1 = z_grid[z_idx], z_grid[z_idx + 1]

        xd = (x - x0) / (x1 - x0)
        yd = (y - y0) / (y1 - y0)
        zd = (z - z0) / (z1 - z0)

        c00 = field[x_idx, y_idx, z_idx] * (1 - xd) + field[x_idx + 1, y_idx, z_idx] * xd
        c01 = field[x_idx, y_idx, z_idx + 1] * (1 - xd) + field[x_idx + 1, y_idx, z_idx + 1] * xd
        c10 = field[x_idx, y_idx + 1, z_idx] * (1 - xd) + field[x_idx + 1, y_idx + 1, z_idx] * xd
        c11 = field[x_idx, y_idx + 1, z_idx + 1] * (1 - xd) + field[x_idx + 1, y_idx + 1, z_idx + 1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd

    return interpolator


def courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants):
    """
    Calculate the Courant condition for a given grid spacing and wave speed.

    The Courant condition is a stability criterion for numerical solutions of partial differential equations. 
    It ensures that the numerical domain of dependence contains the true domain of dependence.

    Args:
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        C (float): Wave speed or Courant number.

    Returns:
        float: The maximum allowable time step for stability.
    """

    solver = simulation_parameters['solver']

    #if solver == 'fdtd':
    C = constants['C']
    return courant_number / (C * ( (1/dx) + (1/dy) + (1/dz) ) )


def plasma_frequency(particle_species, world, constants):
    """
    Calculate the plasma frequency.

    The plasma frequency is calculated using the properties of the electrons,
    the dimensions of the world, and physical constants.

    Args:
        particle_species (object): An object representing the particle species
                    with the following methods:
                    - get_charge(): returns the charge of the particle.
                    - get_number_of_particles(): returns the number of particles.
                    - get_mass(): returns the mass of the particle.
                    - get_temperature(): returns the temperature of the particle.
        world (dict): A dictionary containing the dimensions of the world with keys:
                    'x_wind', 'y_wind', and 'z_wind'.
        constants (dict): A dictionary containing physical constants with key 'eps'.

    Returns:
        float: The calculated plasma frequency.
    """


    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    q = particle_species.get_charge() / particle_species.weight
    N = particle_species.get_number_of_particles()
    m = particle_species.get_mass() / particle_species.weight
    eps = constants['eps']

    # these values are so small that I was having issues calculating
    # the plasma frequency with floating point precision so I had to
    # break it down into smaller parts
    sqrt_dv = jnp.sqrt( x_wind * y_wind * z_wind )
    sqrt_ne = jnp.sqrt( N * particle_species.weight) / sqrt_dv
    sqrt_eps = jnp.sqrt( eps )
    sqrt_me = jnp.sqrt( m )
    pf = sqrt_ne * jnp.abs(q) / (sqrt_eps * sqrt_me)

    # pf = jnp.sqrt( N * particle_species.weight * q**2 ) / jnp.sqrt(m) / jnp.sqrt(eps) / jnp.sqrt(x_wind)

    #plasma_frequency = jnp.sqrt(number_pseudoelectrons * weight * charge_electron**2)/jnp.sqrt(mass_electron)/jnp.sqrt(epsilon_0)/jnp.sqrt(length)


    return pf
# calculate the expected plasma frequency from analytical theory

def debye_length(particle_species, world, constants):
    """
    Calculate the Debye length of a system based on the given parameters.

    Args:
        particle_species (object): An object representing the particle species
                    with the following methods:
                    - get_charge(): returns the charge of the particle.
                    - get_number_of_particles(): returns the number of particles.
                    - get_temperature(): returns the temperature of the particle.
        world (dict): A dictionary containing the dimensions of the world with keys:
                    'x_wind', 'y_wind', and 'z_wind'.
        constants (dict): A dictionary containing physical constants with keys 'eps' and 'kb'.

    Returns:
        float: Debye length of the system.
    """

    q = particle_species.get_charge()
    N_particles = particle_species.get_number_of_particles()
    T = particle_species.get_temperature()
    eps = constants['eps']
    kb = constants['kb']
    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    n = particle_species.weight * N_particles / (x_wind * y_wind * z_wind)
    return jnp.sqrt(eps * kb * T / (n * ( q / particle_species.weight )**2))