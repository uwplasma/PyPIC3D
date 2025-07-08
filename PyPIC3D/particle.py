import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from PyPIC3D.utils import vth_to_T, plasma_frequency, debye_length, T_to_vth

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

def load_particles_from_toml(config, simulation_parameters, world, constants):
    """
    Load particle data from a TOML file and initialize particle species.
    Args:
        config (dict): Dictionary containing configuration keys and values.
        simulation_parameters (dict): Dictionary containing simulation parameters.
        world (dict): Dictionary containing world parameters such as 'x_wind', 'y_wind', 'z_wind', 'dx', 'dy', 'dz'.
        constants (dict): Dictionary containing constants such as 'kb'.
    Returns:
        list: A list of particle_species objects initialized with the data from the TOML file.

    The function reads particle configuration from the provided TOML file, initializes particle properties such as
    position, velocity, charge, mass, and temperature. It also handles loading initial positions and velocities from
    external sources if specified in the TOML file. The particles are then appended to a list and returned.
    """

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the world dimensions
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']
    # get spatial and temporal resolution
    kb = constants['kb']
    eps = constants['eps']
    C   = constants['C']
    # get the constants

    i = 0
    # initialize the random number generator key
    # this is used to generate random numbers for the initial positions and velocities of the particles
    # it is incremented by 3 for each particle species to ensure different random numbers for each species
    particles = []
    particle_keys = grab_particle_keys(config)
    # get the particle keys from the config dictionary

    weight = compute_macroparticle_weight(config, particle_keys, simulation_parameters, world, constants)
    # scale the particle weight by the debye length to prevent numerical heating
    # this is done by computing the total debye length of the plasma and scaling the particle weight accordingly


    for toml_key in particle_keys:
        key1, key2, key3 = jax.random.key(i), jax.random.key(i+1), jax.random.key(i+2)
        i += 3
        # build the particle random number generator keys
        particle_name = config[toml_key]['name']
        charge=config[toml_key]['charge']
        mass=config[toml_key]['mass']

        if 'N_particles' in config[toml_key]:
            N_particles=config[toml_key]['N_particles']
            N_per_cell = N_particles / (world['Nx'] * world['Ny'] * world['Nz'])
        elif "N_per_cell" in config[toml_key]:
            N_per_cell = config[toml_key]["N_per_cell"]
            N_particles = int(N_per_cell * world['Nx'] * world['Ny'] * world['Nz'])
        # set the number of particles in the species

        if 'temperature' in config[toml_key]:
            T=config[toml_key]['temperature']
            vth = T_to_vth(T, mass, kb)
        elif 'vth' in config[toml_key]:
            vth = config[toml_key]['vth']
            T = vth_to_T(vth, mass, kb)
        else:
            T = 1.0
            vth = T_to_vth(T, mass, kb)
        # set the temperature of the particle species

        xmin = read_value('xmin', toml_key, config, -x_wind / 2)
        xmax = read_value('xmax', toml_key, config, x_wind / 2)
        ymin = read_value('ymin', toml_key, config, -y_wind / 2)
        ymax = read_value('ymax', toml_key, config, y_wind / 2)
        zmin = read_value('zmin', toml_key, config, -z_wind / 2)
        zmax = read_value('zmax', toml_key, config, z_wind / 2)
        # set the bounds for the particle species
        x, y, z, vx, vy, vz = initial_particles(N_per_cell, N_particles, xmin, xmax, ymin, ymax, zmin, zmax, mass, T, kb, key1, key2, key3)
        # initialize the positions and velocities of the particles

        bc = 'periodic'
        if 'bc' in config[toml_key]:
            bc = config[toml_key]['bc']
        # set the boundary condition

        x = load_initial_positions('initial_x', config, toml_key, x, N_particles, dx, key1)
        y = load_initial_positions('initial_y', config, toml_key, y, N_particles, dy, key2)
        z = load_initial_positions('initial_z', config, toml_key, z, N_particles, dz, key3)
        # load the initial positions of the particles from the toml file, if specified
        # otherwise, use the initialized positions
        vx = load_initial_velocities('initial_vx', config, toml_key, vx, N_particles)
        vy = load_initial_velocities('initial_vy', config, toml_key, vy, N_particles)
        vz = load_initial_velocities('initial_vz', config, toml_key, vz, N_particles)
        # load the initial velocities of the particles from the toml file, if specified
        # otherwise, use the initialized velocities

        if "weight" in config[toml_key]:
            weight = config[toml_key]['weight']
            # set the weight of the particles, if specified in the toml file
        elif 'ds_per_debye' in config[toml_key]: # assuming dx = dy = dz
            ds_per_debye = config[toml_key]['ds_per_debye']
            weight = (x_wind*y_wind*z_wind * eps * kb * T)  / (N_particles * charge**2 * ds_per_debye**2 * dx*dx)
            # weight the particles by the debye length and the number of particles

        update_pos = read_value('update_pos', toml_key, config, True)
        update_v = read_value('update_v', toml_key, config, True)
        update_vx = read_value('update_vx', toml_key, config, True)
        update_vy = read_value('update_vy', toml_key, config, True)
        update_vz = read_value('update_vz', toml_key, config, True)
        update_x = read_value('update_x', toml_key, config, True)
        update_y = read_value('update_y', toml_key, config, True)
        update_z = read_value('update_z', toml_key, config, True)

        zeta1 = ( x + x_wind/2 ) % dx
        zeta2 = zeta1
        eta1  = ( y + y_wind/2 ) % dy
        eta2  = eta1
        xi1   = ( z + z_wind/2 ) % dz
        xi2   = xi1
        subcells = zeta1, zeta2, eta1, eta2, xi1, xi2
        # calculate the subcell positions for charge conservation algorithm

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
            weight=weight,
            bc=bc,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_pos=update_pos,
            update_v=update_v,
            shape=simulation_parameters['shape_factor'],
            dt=dt
        )
        particles.append(particle)

        pf = plasma_frequency(particle, world, constants)
        dl = debye_length(particle, world, constants)
        print(f"\nInitializing particle species: {particle_name}")
        print(f"Number of particles: {N_particles}")
        print(f"Number of particles per cell: {N_per_cell}")
        print(f"Charge: {charge}")
        print(f"Mass: {mass}")
        print(f"Temperature: {T}")
        print(f"Thermal Velocity: {vth}")
        print(f"Particle Kinetic Energy: {particle.kinetic_energy()}")
        print(f"Particle Species Plasma Frequency: {pf}")
        print(f"Time Steps Per Plasma Period: {(1 / (dt * pf) )}")
        print(f"Particle Species Debye Length: {dl}")
        print(f"Particle Weight: {weight}")
        print(f"Particle Species Scaled Charge: {particle.get_charge()}")
        print(f"Particle Species Scaled Mass: {particle.get_mass()}")

    return particles


def read_value(param, key, config, default_value):
    """
    Reads a value from a nested dictionary structure and returns it if it exists;
    otherwise, returns a default value.

    Args:
        param (str): The parameter name to look for in the nested dictionary.
        key (str): The key in the outer dictionary where the nested dictionary is located.
        config (dict): The configuration dictionary containing nested dictionaries.
        default_value (Any): The value to return if the parameter is not found.

    Returns:
        Any: The value associated with `param` in `config[key]` if it exists,
             otherwise `default_value`.
    """
    if param in config[key]:
        print(f'Reading user defined {param}')
        return config[key][param]
    else:
        return default_value


def load_initial_positions(param, config, key, default, N_particles, ds, key1):
    """
    Load initial positions for particles based on the provided configuration.

    This function checks if a specific parameter exists in the configuration
    under the given key. If the parameter exists and is a string, it loads
    the data from an external source. If the parameter exists and is a number,
    it creates an array filled with that value. If the parameter does not
    exist, it returns the default value.

    Args:
        param (str): The name of the parameter to look for in the configuration.
        config (dict): The configuration dictionary containing parameters and values.
        key (str): The key in the configuration dictionary under which the parameter is stored.
        default (Any): The default value to return if the parameter is not found.
        N_particles (int): The number of particles, used to determine the size of the array.
        ds (float): The spatial resolution, used to add noise to the particle positions.
        key1 (jax.random.PRNGKey): The random key for generating random numbers.

    Returns:
        jax.numpy.ndarray or Any: An array of particle positions if the parameter is found,
        or the default value if the parameter is not found.
    """
    if param in config[key]:
        if isinstance(config[key][param], str):
            print(f"Loading {param} from external source: {config[key][param]}")
            return jnp.load(config[key][param])
            # if the value is a string, load it from an external source
        else:
            #return jnp.full(N_particles, config[key][param])
            val = config[key][param]
            return jax.random.uniform(key1, shape = (N_particles,), minval=val-(ds/2), maxval=val+(ds/2))
            # if the value is a number, fill the array with that value with some noise in the subcell position
    else:
        return default
        # return the default value if the parameter is not found

def load_initial_velocities(param, config, key, default, N_particles):
    """
    Load initial velocities for particles based on the provided configuration.

    This function checks if a specific parameter exists in the configuration
    dictionary under the given key. Depending on the type of the parameter's
    value, it either loads data from an external source or initializes an array
    with a specified value. If the parameter is not found, a default value is returned.

    Args:
        param (str): The name of the parameter to look for in the configuration.
        config (dict): A dictionary containing configuration data.
        key (str): The key in the configuration dictionary where the parameter is located.
        default (float or jnp.ndarray): The default value to return if the parameter is not found.
        N_particles (int): The number of particles, used to determine the size of the array.

    Returns:
        jnp.ndarray: An array of initial velocities for the particles. If the parameter
        is a string, the array is loaded from an external source. If the parameter is
        a number, the array is filled with that value plus the default. If the parameter
        is not found, the default value is returned.
    """
    if param in config[key]:
        if isinstance(config[key][param], str):
            print(f"Loading {param} from external source: {config[key][param]}")
            return jnp.load(config[key][param])
            # if the value is a string, load it from an external source
        else:
            return jnp.full(N_particles, config[key][param]) + default
            # if the value is a number, fill the array with that value
    else:
        return default
        # return the default value if the parameter is not found

def compute_macroparticle_weight(config, particle_keys, simulation_parameters, world, constants):

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']
    # get the world dimensions
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    # get the world resolution
    kb = constants['kb']
    eps = constants['eps']
    # get the constants

    if simulation_parameters['ds_per_debye']: # scale the particle weight by the debye length to prevent numerical heating
        ds_per_debye = simulation_parameters['ds_per_debye']
        # get the number of grid points per debye length
        inverse_total_debye = 0

        for toml_key in particle_keys:
            N_particles = config[toml_key]['N_particles']
            charge = config[toml_key]['charge']
            mass = config[toml_key]['mass']
            # get the charge and mass of the particle species
            if 'temperature' in config[toml_key]:
                T=config[toml_key]['temperature']
            elif 'vth' in config[toml_key]:
                T = vth_to_T(config[toml_key]['vth'], mass, kb)
            # get the temperature of the particle species

            inverse_total_debye += jnp.sqrt( N_particles / (x_wind * y_wind * z_wind) / (eps * kb * T) ) * jnp.abs(charge)
            # get the inverse debye length before macroparticle weighting

        weight = 1 / (dx**2) / (ds_per_debye**2) / inverse_total_debye
        # weight the particles by the total debye length of the plasma

    else:
        weight = 1.0 # default to single particle weight

    return weight

def initial_particles(N_per_cell, N_particles, minx, maxx, miny, maxy, minz, maxz, mass, T, kb, key1, key2, key3):
    """
    Initializes the velocities and positions of the particles.

    Args:
        N_particles (int): The number of particles.
        minx (float): The minimum value for the x-coordinate of the particles' positions.
        maxx (float): The maximum value for the x-coordinate of the particles' positions.
        miny (float): The minimum value for the y-coordinate of the particles' positions.
        maxy (float): The maximum value for the y-coordinate of the particles' positions.
        minz (float): The minimum value for the z-coordinate of the particles' positions.
        maxz (float): The maximum value for the z-coordinate of the particles' positions.
        mass (float): The mass of the particles.
        T (float): The temperature of the system.
        kb (float): The Boltzmann constant.
        key (jax.random.PRNGKey): The random key for generating random numbers.

    Returns:
        x (jax.numpy.ndarray): The x-coordinates of the particles' positions.
        y (jax.numpy.ndarray): The y-coordinates of the particles' positions.
        z (jax.numpy.ndarray): The z-coordinates of the particles' positions.
        v_x (numpy.ndarray): The x-component of the particles' velocities.
        v_y (numpy.ndarray): The y-component of the particles' velocities.
        v_z (numpy.ndarray): The z-component of the particles' velocities.
    """

    # if N_per_cell < 1:
    x = jax.random.uniform(key1, shape = (N_particles,), minval=minx, maxval=maxx)
    y = jax.random.uniform(key2, shape = (N_particles,), minval=miny, maxval=maxy)
    z = jax.random.uniform(key3, shape = (N_particles,), minval=minz, maxval=maxz)
        # initialize the positions of the particles
    # else:
        # x = jnp.repeat(jax.random.uniform(key1, shape=(N_particles // N_per_cell,), minval=minx, maxval=maxx), N_per_cell)
        # y = jnp.repeat(jax.random.uniform(key2, shape=(N_particles // N_per_cell,), minval=miny, maxval=maxy), N_per_cell)
        # z = jnp.repeat(jax.random.uniform(key3, shape=(N_particles // N_per_cell,), minval=minz, maxval=maxz), N_per_cell)
        # initialize the positions of the particles, giving every N_per_cell particles the same position
    #std = jnp.sqrt( kb * T / mass )
    std = T_to_vth( T, mass, kb )
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z

@jit
def compute_index(x, dx, window):
    """
    Compute the index of a position in a discretized space.

    Args:
        x (float or ndarray): The position(s) to compute the index for.
        dx (float): The discretization step size.

    Returns:
        int or ndarray: The computed index/indices as integer(s).
    """
    scaled_x = x + window/2
    return jnp.floor( scaled_x / dx).astype(int)

@register_pytree_node_class
class particle_species:
    """
    Class representing a species of particles in a simulation.

    Attributes:
        name (str): Name of the particle species.
        N_particles (int): Number of particles in the species.
        charge (float): Charge of each particle.
        mass (float): Mass of each particle.
        weight (float): Weighting factor for the particles.
        T (float): Temperature of the particle species.
        v1, v2, v3 (array-like): Velocity components of the particles.
        x1, x2, x3 (array-like): Position components of the particles.
        dx, dy, dz (float): Spatial resolution in each dimension.
        x_wind, y_wind, z_wind (float): Domain size in each dimension.
        zeta1, zeta2, eta1, eta2, xi1, xi2 (float): Subcell positions for charge conservation.
        bc (str): Boundary condition type ('periodic' or 'reflecting').
        update_x, update_y, update_z (bool): Flags to update position in respective dimensions.
        update_vx, update_vy, update_vz (bool): Flags to update velocity in respective dimensions.
        update_pos (bool): Flag to update particle positions.
        update_v (bool): Flag to update particle velocities.
        shape (int): Shape factor for the particles (1 for first order, 2 for second order)

    Methods:
        get_name(): Returns the name of the particle species.
        get_charge(): Returns the total charge of the particles.
        get_number_of_particles(): Returns the number of particles in the species.
        get_temperature(): Returns the temperature of the particle species.
        get_velocity(): Returns the velocity components of the particles.
        get_position(): Returns the position components of the particles.
        get_mass(): Returns the total mass of the particles.
        get_subcell_position(): Returns the subcell positions for charge conservation.
        get_resolution(): Returns the spatial resolution in each dimension.
        get_shape(): Returns the shape factor of the particles.
        get_index(): Computes and returns the particle indices in the grid.
        set_velocity(v1, v2, v3): Sets the velocity components of the particles.
        set_position(x1, x2, x3): Sets the position components of the particles.
        set_mass(mass): Sets the mass of the particles.
        set_weight(weight): Sets the weight of the particles.
        calc_subcell_position(): Calculates and returns the subcell positions.
        kinetic_energy(): Computes and returns the kinetic energy of the particles.
        momentum(): Computes and returns the momentum of the particles.
        periodic_boundary_condition(x_wind, y_wind, z_wind): Applies periodic boundary conditions.
        reflecting_boundary_condition(x_wind, y_wind, z_wind): Applies reflecting boundary conditions.
        update_position(dt): Updates the positions of the particles based on their velocities and boundary conditions.
        tree_flatten(): Flattens the object for serialization.
        tree_unflatten(aux_data, children): Reconstructs the object from flattened data.
    """



    def __init__(self, name, N_particles, charge, mass, T, v1, v2, v3, x1, x2, x3, subcells, \
            xwind, ywind, zwind, dx, dy, dz, weight=1, bc='periodic', update_x=True, update_y=True, update_z=True, \
                update_vx=True, update_vy=True, update_vz=True, update_pos=True, update_v=True, shape=1, dt = 0):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
        self.weight = weight
        self.T = T
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x_wind = xwind
        self.y_wind = ywind
        self.z_wind = zwind
        self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2 = subcells
        self.bc = bc
        self.update_x = update_x
        self.update_y = update_y
        self.update_z = update_z
        self.update_vx = update_vx
        self.update_vy = update_vy
        self.update_vz = update_vz
        self.update_pos = update_pos
        self.update_v   = update_v
        self.shape = shape
        self.dt = dt

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        self.x1_back = self.x1 - self.v1 * self.dt / 2
        self.x2_back = self.x2 - self.v2 * self.dt / 2
        self.x3_back = self.x3 - self.v3 * self.dt / 2

        self.x1_forward = self.x1 + self.v1 * self.dt / 2
        self.x2_forward = self.x2 + self.v2 * self.dt / 2
        self.x3_forward = self.x3 + self.v3 * self.dt / 2

        self.old_x1 = self.x1
        self.old_x2 = self.x2
        self.old_x3 = self.x3


    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge*self.weight

    def get_number_of_particles(self):
        return self.N_particles

    def get_temperature(self):
        return self.T

    def get_velocity(self):
        return self.v1, self.v2, self.v3

    def get_backward_position(self):
        return self.x1_back, self.x2_back, self.x3_back

    def get_forward_position(self):
        return self.x1_forward, self.x2_forward, self.x3_forward

    def get_position(self):
        return self.x1, self.x2, self.x3

    def get_mass(self):
        return self.mass*self.weight

    def get_subcell_position(self):
        return self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2

    def get_old_position(self):
        return self.old_x1, self.old_x2, self.old_x3

    def get_resolution(self):
        return self.dx, self.dy, self.dz

    def get_shape(self):
        return self.shape

    def get_index(self):
        return compute_index(self.x1, self.dx, self.x_wind), compute_index(self.x2, self.dy, self.y_wind), compute_index(self.x3, self.dz, self.z_wind)

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            if self.update_vx:
                self.v1 = v1
            if self.update_vy:
                self.v2 = v2
            if self.update_vz:
                self.v3 = v3

    def set_backward_position(self, x1, x2, x3):
        if self.update_pos:
            if self.update_x:
                self.x1_back = x1
            if self.update_y:
                self.x2_back = x2
            if self.update_z:
                self.x3_back = x3

    def set_forward_position(self, x1, x2, x3):
        if self.update_pos:
            if self.update_x:
                self.x1_forward = x1
            if self.update_y:
                self.x2_forward = x2
            if self.update_z:
                self.x3_forward = x3

    def set_position(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def set_mass(self, mass):
        self.mass = mass

    def set_weight(self, weight):
        self.weight = weight

    def calc_subcell_position(self):
        newzeta = (self.x1 + self.x_wind / 2) % self.dx
        neweta  = (self.x2 + self.y_wind / 2) % self.dy
        newxi   = (self.x3 + self.z_wind / 2) % self.dz

        return newzeta, neweta, newxi

    def kinetic_energy(self):
        return 0.5 * self.weight * self.mass *  (  jnp.abs( jnp.sum( self.v1**2)) + jnp.abs( jnp.sum( self.v2**2)) + jnp.abs( jnp.sum( self.v3**2)) )

    def momentum(self):
        return self.mass * self.weight * jnp.sum(jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2))

    def periodic_boundary_condition(self):
        self.x1 = jnp.where(self.x1 > self.x_wind/2,  self.x1 - self.x_wind, \
                            jnp.where(self.x1 < -self.x_wind/2, self.x1 + self.x_wind, self.x1))
        self.x2 = jnp.where(self.x2 > self.y_wind/2,  self.x2 - self.y_wind, \
                            jnp.where(self.x2 < -self.y_wind/2, self.x2 + self.y_wind, self.x2))
        self.x3 = jnp.where(self.x3 > self.z_wind/2,  self.x3 - self.z_wind, \
                            jnp.where(self.x3 < -self.z_wind/2, self.x3 + self.z_wind, self.x3))

    def reflecting_boundary_condition(self):

        self.v1 = jnp.where((self.x1 > self.x_wind/2) | (self.x1 < -self.x_wind/2), -self.v1, self.v1)
        self.x1 = jnp.where(self.x1 > self.x_wind/2, self.x_wind/2, self.x1)
        self.x1 = jnp.where(self.x1 < -self.x_wind/2, -self.x_wind/2, self.x1)

        self.v2 = jnp.where((self.x2 > self.y_wind/2) | (self.x2 < -self.y_wind/2), -self.v2, self.v2)
        self.x2 = jnp.where(self.x2 > self.y_wind/2, self.y_wind/2, self.x2)
        self.x2 = jnp.where(self.x2 < -self.y_wind/2, -self.y_wind/2, self.x2)

        self.v3 = jnp.where((self.x3 > self.z_wind/2) | (self.x3 < -self.z_wind/2), -self.v3, self.v3)
        self.x3 = jnp.where(self.x3 > self.z_wind/2, self.z_wind/2, self.x3)
        self.x3 = jnp.where(self.x3 < -self.z_wind/2, -self.z_wind/2, self.x3)


    def update_position(self):
        if self.update_pos:
            if self.update_x:
                self.x1_back = self.x1_forward
                self.x1_forward = self.x1_forward + self.v1 * self.dt / 2
                self.old_x1 = self.x1
                # store the old position of the particles
                self.x1 = self.x1_forward - self.v1 * self.dt / 2
                # update the x position of the particles

            if self.update_y:
                self.x2_back = self.x2_forward
                self.x2_forward = self.x2_forward + self.v2 * self.dt / 2
                self.old_x2 = self.x2
                # store the old position of the particles
                self.x2 = self.x2_forward - self.v2 * self.dt / 2
                # update the y position of the particles

            if self.update_z:
                self.x3_back = self.x3_forward
                self.x3_forward = self.x3_forward + self.v3 * self.dt / 2
                self.old_x3 = self.x3
                # store the old position of the particles
                self.x3 = self.x3_forward - self.v3 * self.dt / 2
                # update the z position of the particles

        if self.bc == 'periodic':
            self.periodic_boundary_condition()
            # apply periodic boundary conditions to the particles
        elif self.bc == 'reflecting':
            self.reflecting_boundary_condition()
            # apply reflecting boundary conditions to the particles

        self.zeta1 = self.zeta2
        self.eta1  = self.eta2
        self.xi1   = self.xi2
        self.zeta2, self.eta2, self.xi2 = self.calc_subcell_position()
        # update the subcell positions for charge conservation algorithm

    def tree_flatten(self):
        children = (
            self.v1, self.v2, self.v3, \
            self.x1, self.x2, self.x3, \
            self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2, \
            self.x1_back, self.x2_back, self.x3_back, \
            self.x1_forward, self.x2_forward, self.x3_forward, \
            self.old_x1, self.old_x2, self.old_x3
        )

        aux_data = (
            self.name, self.N_particles, self.charge, self.mass, self.T, \
            self.x_wind, self.y_wind, self.z_wind, self.dx, self.dy, self.dz, \
            self.weight, self.bc, self.update_pos, self.update_v, self.update_x, self.update_y, \
            self.update_z, self.update_vx, self.update_vy, self.update_vz, self.shape, self.dt
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        v1, v2, v3, x1, x2, x3, zeta1, zeta2, eta1, eta2, xi1, xi2, x1_back, x2_back, x3_back, x1_forward, x2_forward, x3_forward, old_x1, old_x2, old_x3 = children


        name, N_particles, charge, mass, T, x_wind, y_wind, z_wind, dx, dy, \
            dz, weight, bc, update_pos, update_v, update_x, update_y, update_z, \
                update_vx, update_vy, update_vz, shape, dt  = aux_data

        subcells = zeta1, zeta2, eta1, eta2, xi1, xi2

        obj = cls(
            name=name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            T=T,
            x1=x1,
            x2=x2,
            x3=x3,
            v1=v1,
            v2=v2,
            v3=v3,
            subcells=subcells,
            xwind=x_wind,
            ywind=y_wind,
            zwind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            weight=weight,
            bc=bc,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            update_pos=update_pos,
            update_v=update_v,
            shape=shape,
            dt=dt
        )

        return obj