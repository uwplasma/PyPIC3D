import jax
import plotly
import tqdm
from jax import jit
import argparse
import jax.numpy as jnp
import functools
from functools import partial
import toml
import os
from jax.tree_util import tree_map
from datetime import datetime
import importlib.metadata
from scipy import stats
# import external libraries

from PyPIC3D.parameters import dynamic_parameters_for_output, static_parameters_for_output

def setup_pmd_files(file_path, name, extension=".bp"):
    """
    Set up the openPMD file structure for storing simulation data.

    Args:
        file_path (str): The path where the openPMD files will be stored.
        name (str): The base name for the openPMD files.
    Returns:
        None
    """

    file = os.path.join(file_path, name + ".pmd")
    with open(file, 'w') as f:
        f.write(f"{name}{extension}\n")
    # create the openPMD file structure

@jit
def wrap_around(ix, size):
    """Wrap around index (scalar or 1D array) to ensure it is within bounds."""
    return jnp.mod(ix, size)


def mae(x, y):
    """
    Calculates the mean absolute error (MAE) between two arrays.

    Parameters:
        x (array-like): First input array.
        y (array-like): Second input array, must be broadcastable to the shape of x.

    Returns:
        jnp.ndarray: The square root of the mean of squared differences between x and y.

    Note:
        Despite the function name 'mae' (mean absolute error), this function actually computes the root mean squared error (RMSE).
    """

    return jnp.sqrt( jnp.mean( (x-y)**2 ) )

def mse(x, y):
    """
    Calculates the mean squared error (MSE) between two arrays.

    Parameters:
        x (array-like): First input array.
        y (array-like): Second input array, must be broadcastable to the shape of x.

    Returns:
        jnp.ndarray: The mean of squared differences between x and y.
    """

    return jnp.mean( (x-y)**2 )


def convergence_test(func):
    """
    Computes the order of convergence for a numerical method by measuring the error at increasing grid resolutions.

    Args:
        func (callable): A function that takes an integer `nx` (number of grid points) as input and returns a tuple `(error, dx)`,
                         where `error` is the error at that resolution and `dx` is the grid spacing.

    Returns:
        float: The absolute value of the slope from a linear regression of log(error) vs. log(dx), representing the order of convergence.
    """

    nxs = [10*i + 30 for i in range(20)]
    # build list of different number of grid points

    errors = []
    dxs    = []
    # initialize the error and resolution lists

    for nx in nxs:
        error, dx = func(nx)
        errors.append( error )
        dxs.append( dx )
    # measure the error for increasing resolutions

    dxs = jnp.asarray(dxs)
    errors = jnp.asarray(errors)
    # convert the result lists to ndarrays

    res = stats.linregress( jnp.log(dxs), jnp.log(errors) + 3*jnp.log(dxs) )
    slope = jnp.abs( res.slope )
    # compute the order of the convergence using a line fit of the log(y)/log(x)

    return slope

def compute_energy(particles, E, B, static_parameters, dynamic_parameters=None, species_config=None):
    """
    Compute the total energy of the system, including electric field energy, magnetic field energy, and kinetic energy of particles.

    Args:
        particles (TiledParticles): Tile-major particle storage.
        E (tuple): Electric field components (Ex, Ey, Ez).
        B (tuple): Magnetic field components (Bx, By, Bz).
        static_parameters (dict): Compile-time geometry and branch choices.
        dynamic_parameters (dict): Scalar values and grids.

    Returns:
        None
    """

    if dynamic_parameters is None:
        dynamic_parameters = static_parameters
    elif "dx" not in dynamic_parameters:
        dynamic_parameters = {**static_parameters, **dynamic_parameters}

    dx = dynamic_parameters['dx']
    dy = dynamic_parameters['dy']
    dz = dynamic_parameters['dz']
    # get the resolution of the grid

    Nx = dynamic_parameters['Nx']
    Ny = dynamic_parameters['Ny']
    Nz = dynamic_parameters['Nz']
    # get the number of grid points in each direction

    def nd_trapezoid(arr, dxs):
        # arr: ndarray to integrate
        # dxs: list/tuple of grid spacings for each axis
        for axis, dx in enumerate(dxs):
            arr = jnp.trapezoid( jnp.squeeze(arr), dx=dx, axis=-1)
        return arr

    # Build dxs tuple with only components that are not 1
    dxs = tuple(d for d in (dz, dy, dx) if d != 1)

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # use physical interior slices to exclude ghost cells from the energy
    # integral.  Tiled fields have leading tile axes followed by local
    # ghost-celled Yee arrays.
    if Ex.ndim == 6:
        g = int(static_parameters.get("guard_cells", 2))
        interior = (
            slice(None),
            slice(None),
            slice(None),
            slice(g, -g),
            slice(g, -g),
            slice(g, -g),
        )
    else:
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    dV = dx * dy * dz
    # calculate the volume element
    E2_integral = jnp.sum(Ex[interior]**2 + Ey[interior]**2 + Ez[interior]**2) * dV
    B2_integral = jnp.sum(Bx[interior]**2 + By[interior]**2 + Bz[interior]**2) * dV
    # Integrate E^2 and B^2 over the grid using trapezoidal rule
    e_energy = 0.5 * dynamic_parameters['eps'] * E2_integral
    b_energy = 0.5 / dynamic_parameters['mu'] * B2_integral
    # Electric and magnetic field energy
    C = dynamic_parameters['C']
    # speed of light
    vx = particles.u[..., 0]
    vy = particles.u[..., 1]
    vz = particles.u[..., 2]
    v2 = vx**2 + vy**2 + vz**2

    active = particles.active.astype(v2.dtype)
    species_mass = species_config.mass * species_config.weight
    mass = jnp.broadcast_to(
        species_mass.reshape((1, 1, 1, species_mass.shape[0], 1)),
        particles.active.shape,
    )
    gamma = 1.0 / jnp.sqrt(1 - v2 / C**2)
    momentum2 = jnp.square(mass * gamma) * v2
    kinetic_energy = jnp.sum(active * (jnp.sqrt(momentum2 * C**2 + mass**2 * C**4) - mass * C**2))


    # Kinetic energy of particles
    return e_energy, b_energy, kinetic_energy


def compute_total_momentum(particles, species_config=None):
    """
    Compute the scalar momentum diagnostic for tiled particles.
    """

    vmag = jnp.sqrt(particles.u[..., 0]**2 + particles.u[..., 1]**2 + particles.u[..., 2]**2)
    active = particles.active.astype(vmag.dtype)
    species_mass = species_config.mass * species_config.weight
    mass = jnp.broadcast_to(
        species_mass.reshape((1, 1, 1, species_mass.shape[0], 1)),
        particles.active.shape,
    )
    return jnp.sum(active * vmag * mass)


def add_external_fields(E, B, external_fields):
    """
    Add prescribed external fields to the self-consistent fields.

    Maxwell updates should use E and B by themselves. Particle pushes and total
    energy diagnostics should use the returned totals, because those are the
    fields particles actually see.
    """
    external_E, external_B = external_fields
    total_E = tuple(e + ext_e for e, ext_e in zip(E, external_E))
    total_B = tuple(b + ext_b for b, ext_b in zip(B, external_B))
    return total_E, total_B

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
    Perform a basic shape check for tiled particle storage.

    Args:
        particles (TiledParticles): Tile-major particle storage.

    Raises:
        AssertionError: If the tiled position, velocity, and active arrays are inconsistent.
    """

    assert particles.x.shape == particles.u.shape
    assert particles.x.shape[-1] == 3
    assert particles.active.shape == particles.x.shape[:-1]


def print_stats(static_parameters, dynamic_parameters):
    """
    Print the statistics of the simulation.
    
    Args:
        static_parameters (dict): Compile-time run settings.
        dynamic_parameters (dict): Spatial and temporal scalar values.
        
    Prints:
        The time window, x window, y window, z window, and resolution details (dx, dy, dz, dt, Nt).
    """

    Nt = static_parameters['Nt']
    dx = dynamic_parameters['dx']
    dy = dynamic_parameters['dy']
    dz = dynamic_parameters['dz']
    dt = dynamic_parameters['dt']
    x_wind = dynamic_parameters['x_wind']
    y_wind = dynamic_parameters['y_wind']
    z_wind = dynamic_parameters['z_wind']
    t_wind = Nt*dt
    print(f'\ntime window: {t_wind} s with {Nt} time steps of {dt} s')
    print(f'x window: {x_wind} m with dx: {dx} m')
    print(f'y window: {y_wind} m with dy: {dy} m')
    print(f'z window: {z_wind} m with dz: {dz} m\n')

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


def build_plasma_parameters_dict(static_parameters, dynamic_parameters, electrons):
    """
    Build a dictionary containing various plasma parameters.

    Args:
        static_parameters (dict): Compile-time run settings.
        dynamic_parameters (dict): Scalar values and grids.
        electrons (dict): Metadata for the electron species from particle initialization.
        dt (float): Time step of the simulation.

    Returns:
        dict: A dictionary containing the plasma parameters.
    """

    me = electrons["mass"]
    Te = electrons["temperature"]
    N = electrons["N_particles"]
    q = electrons["charge"]
    weight = electrons["weight"]
    kb = dynamic_parameters['kb']
    dx, dy, dz = dynamic_parameters['dx'], dynamic_parameters['dy'], dynamic_parameters['dz']

    volume = dynamic_parameters["x_wind"] * dynamic_parameters["y_wind"] * dynamic_parameters["z_wind"]
    density = weight * N / volume
    theoretical_freq = jnp.sqrt(density) * jnp.abs(q) / jnp.sqrt(dynamic_parameters["eps"] * me)
    debye = jnp.sqrt(dynamic_parameters["eps"] * kb * Te / (density * q**2))
    thermal_velocity = jnp.sqrt(3*kb*Te/me)

    plasma_parameters = {
        "Theoretical Plasma Frequency": theoretical_freq,
        "Debye Length": debye,
        "Thermal Velocity": thermal_velocity,
        "Number of Electrons": N,
        "Temperature of Electrons": Te,
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

def _add_external_field_to_tiled_component(component, external_field, static_parameters, dynamic_parameters, field_name):
    """
    Add one physical field array into the active interiors of a tiled component.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters["tile_shape"]]
    g = int(static_parameters["guard_cells"])
    ntx, nty, ntz = component.shape[:3]
    interior_shape = (int(dynamic_parameters["Nx"]), int(dynamic_parameters["Ny"]), int(dynamic_parameters["Nz"]))
    if external_field.shape != interior_shape:
        raise ValueError(
            f"Shape mismatch for field '{field_name}': external field shape {external_field.shape} "
            f"does not match expected interior shape {interior_shape}"
        )

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                ix = tx * tile_nx
                iy = ty * tile_ny
                iz = tz * tile_nz
                block = external_field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                component = component.at[
                    tx, ty, tz,
                    g:g + tile_nx,
                    g:g + tile_ny,
                    g:g + tile_nz,
                ].add(block)

    return component


def load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters):
    """
    Load external fields from a TOML file.

    Args:
        fields (list): Flattened list of evolved E, B, and J field components.
        external_fields (tuple): External-only E and B field tuples.
        config (dict): Dictionary containing the configuration values.
        static_parameters (dict): Tile shape and guard-cell depth.
        dynamic_parameters (dict): Grid size and scalar values.

    Returns:
        tuple: Updated evolved fields list and external field tuple.
    """

    field_keys = grab_field_keys(config)
    external_E, external_B = external_fields

    for toml_key in field_keys:
        field_name = config[toml_key]['name']
        field_type = config[toml_key]['type']
        field_path = config[toml_key]['path']
        evolve = config[toml_key].get('evolve', True)
        print(f"Loading field: {field_name} from {field_path}")

        external_field = jnp.load(field_path)

        if not evolve and (field_type < 0 or field_type > 5):
            raise ValueError("External-only fields must be electric or magnetic field components with type 0 through 5")

        if evolve:
            # Evolved fields are part of the self-consistent Maxwell solve.
            # This is the original behavior and remains the default.
            fields[field_type] = _add_external_field_to_tiled_component(
                fields[field_type],
                external_field,
                static_parameters,
                dynamic_parameters,
                field_name,
            )
        else:
            # External-only E/B fields are invisible to Maxwell's equations.
            # They are added back only for particle pushes and diagnostics.
            if field_type < 3:
                external_E = list(external_E)
                external_E[field_type] = _add_external_field_to_tiled_component(
                    external_E[field_type],
                    external_field,
                    static_parameters,
                    dynamic_parameters,
                    field_name,
                )
                external_E = tuple(external_E)
            else:
                external_B = list(external_B)
                b_index = field_type - 3
                external_B[b_index] = _add_external_field_to_tiled_component(
                    external_B[b_index],
                    external_field,
                    static_parameters,
                    dynamic_parameters,
                    field_name,
                )
                external_B = tuple(external_B)

        print(f"Field loaded successfully: {field_name}")

    return fields, (external_E, external_B)

def debugprint(value):
    """
    Prints the given value using JAX's debug print functionality.

    Args:
        value: The value to be printed. Can be of any type that is supported by JAX's debug print.

    Returns:
        None
    """
    jax.debug.print('{x}', x=value)

def update_parameters_from_toml(config, static_parameters, dynamic_parameters, plotting_parameters):
    """
    Update run parameters with values from a TOML config file.

    Args:
        config (dict): Dictionary containing the configuration values.
        static_parameters (dict): Dictionary of default compile-time/run parameters.
        dynamic_parameters (dict): Dictionary of default scalar/grid parameters.
        plotting_parameters (dict): Dictionary of default plotting parameters.

    Returns:
        tuple: Updated static, dynamic, and plotting parameters.
    """

    for key, value in config.get("simulation_parameters", {}).items():
        if key in static_parameters:
            static_parameters[key] = value
        if key in dynamic_parameters:
            dynamic_parameters[key] = value

    for key, value in config.get("static_parameters", {}).items():
        if key in static_parameters:
            static_parameters[key] = value

    for key, value in config.get("dynamic_parameters", {}).items():
        if key in dynamic_parameters:
            dynamic_parameters[key] = value

    for key, value in config.get("plotting", {}).items():
        if key in plotting_parameters:
            plotting_parameters[key] = value

    return static_parameters, dynamic_parameters, plotting_parameters

def dump_parameters_to_toml(simulation_stats, static_parameters, dynamic_parameters, plasma_parameters, plotting_parameters, particles):
    """
    Dump run, plotting, and particle species data into an output TOML file.

    Args:
        simulation_stats (dict): Dictionary of simulation statistics.
        static_parameters (dict): Compile-time/run parameters.
        dynamic_parameters (dict): Scalar/grid parameters.
        plotting_parameters (dict): Dictionary of plotting parameters.
        particles (TiledParticles): Tile-major particle storage.
    """

    output_path = static_parameters["output_dir"]
    output_file = os.path.join(output_path, "data/output.toml")
    plotting_parameters_for_output = {
        key: value
        for key, value in plotting_parameters.items()
        if key not in ("particle_species_names", "particle_species_metadata")
    }

    config = {
        "simulation_stats": simulation_stats,
        "static_parameters": static_parameters_for_output(static_parameters),
        "dynamic_parameters": dynamic_parameters_for_output(dynamic_parameters),
        'plasma_parameters': jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plasma_parameters),
        "plotting": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plotting_parameters_for_output),
        "particles": []
    }

    n_species = particles.active.shape[3]
    species_names = plotting_parameters.get("particle_species_names")
    species_metadata = plotting_parameters.get("particle_species_metadata")
    tile_shape = jax.tree_util.tree_map(
        lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x,
        static_parameters.get("tile_shape", ()),
    )

    for species_index in range(n_species):
        if species_names is None:
            name = f"species_{species_index}"
        else:
            name = species_names[species_index]

        active_particles = int(jnp.sum(particles.active[:, :, :, species_index, :]))
        if species_metadata is None:
            particle_dict = {"name": name}
        else:
            particle_dict = dict(
                jax.tree_util.tree_map(
                    lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x,
                    species_metadata[species_index],
                )
            )

        particle_dict["storage"] = "tiled"
        particle_dict["active_particles"] = active_particles
        particle_dict["tile_shape"] = tile_shape
        config["particles"].append(particle_dict)

    config["version"] = {
        "PyPIC3D_version": importlib.metadata.version('PyPIC3D'),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    # Get the versions of all the packages being imported
    package_versions = {
        "jax": jax.__version__,
        "toml": toml.__version__,
        "plotly": plotly.__version__,
        "tqdm": tqdm.__version__,
    }

    config["package_versions"] = package_versions

    # print("Simulation Stats:", simulation_stats)
    # print("Static Parameters:", static_parameters)
    # print("Dynamic Parameters:", dynamic_parameters)
    # print("Plasma Parameters:", plasma_parameters)
    # print("Plotting Parameters:", plotting_parameters)

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


def courant_condition(courant_number, dx, dy, dz, dynamic_parameters):
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

    C = dynamic_parameters['C']


    Nx = dynamic_parameters["Nx"]
    Ny = dynamic_parameters["Ny"]
    Nz = dynamic_parameters["Nz"]
    # get the number of grid points in each direction
    Ns  = [Nx, Ny, Nz]
    dxs = [dx, dy, dz]
    # build a list of the spatial steps in each direction and the number of grid points in each direction

    dx_inv = []
    for d, N in zip(dxs, Ns):
        if N > 1:
            dx_inv.append(1/d)
    # only add the inverse spatial steps for dimensions with more than one grid point

    dx_inv = sum(dx_inv)
    # sum all the inverse spatial steps

    dt = courant_number / (C * dx_inv)
    # compute the maximum allowed timestep

    return dt
