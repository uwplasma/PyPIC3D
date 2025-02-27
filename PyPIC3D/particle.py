import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from PyPIC3D.utils import vth_to_T

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
        #T=config[toml_key]['temperature']


        if 'temperature' in config[toml_key]:
            T=config[toml_key]['temperature']
        elif 'vth' in config[toml_key]:
            T = vth_to_T(config[toml_key]['vth'], mass, kb)
        else:
            T = 1.0
        # set the temperature of the particle species

        w = jnp.zeros((3,3))
        g = jnp.zeros((3,3))
        # initialize the frequency and damping matrices for the particle species


        xmin = -x_wind/2
        xmax = x_wind/2
        ymin = -y_wind/2
        ymax = y_wind/2
        zmin = -z_wind/2
        zmax = z_wind/2
        # set the default bounds for the particle species

        bounded = False
        if 'bounded' in config[toml_key]:
            bounded = config[toml_key]['bounded']
        # check if the particle species is bounded or not
        if bounded:
            print(f"Initializing bounded particle species: {particle_name}")
            w = jnp.array(config[toml_key]['w']) * jnp.identity(3)
            g = jnp.array(config[toml_key]['g']) * jnp.identity(3)
            print(f"Using w:\n {w}")
            print(f"Using g:\n {g}")
            try:
                xmin = config[toml_key]['xmin']
                xmax = config[toml_key]['xmax']
                ymin = config[toml_key]['ymin']
                ymax = config[toml_key]['ymax']
                zmin = config[toml_key]['zmin']
                zmax = config[toml_key]['zmax']
            except:
                print(f"Error: Bounded particle species {particle_name} requires xmin, xmax, ymin, ymax, zmin, zmax")
                exit(1)
            if "fermi_energy" in config[toml_key]:
                fermi_energy = config[toml_key]['fermi_energy']
            else:
                fermi_energy = 1.0
            
            x, y, z, vx, vy, vz = initial_bound_particles(N_particles, xmin, xmax, ymin, ymax, zmin, zmax, mass, T, fermi_energy, kb, key1, key2, key3)
        else:
            print(f"Initializing unbounded particle species: {particle_name}")
            if 'xmin' in config[toml_key]:
                xmin = config[toml_key]['xmin']
            if 'xmax' in config[toml_key]:
                xmax = config[toml_key]['xmax']
            if 'ymin' in config[toml_key]:
                ymin = config[toml_key]['ymin']
            if 'ymax' in config[toml_key]:
                ymax = config[toml_key]['ymax']
            if 'zmin' in config[toml_key]:
                zmin = config[toml_key]['zmin']
            if 'zmax' in config[toml_key]:
                zmax = config[toml_key]['zmax']
            # set the bounds for the particle species if specified
            x, y, z, vx, vy, vz = initial_particles(N_particles, xmin, xmax, ymin, ymax, zmin, zmax, mass, T, kb, key1, key2, key3)
        # initialize the positions and velocities of the particles

        bc = 'periodic'
        if 'bc' in config[toml_key]:
            bc = config[toml_key]['bc']
        # set the boundary condition
        if 'initial_x' in config[toml_key]:
            if isinstance(config[toml_key]['initial_x'], str):
                print(f"Loading initial_x from external source: {config[toml_key]['initial_x']}")
                x = jnp.load(config[toml_key]['initial_x'])
            else:
                x = jnp.full(N_particles, config[toml_key]['initial_x'])
        if 'initial_y' in config[toml_key]:
            if isinstance(config[toml_key]['initial_y'], str):
                print(f"Loading initial_y from external source: {config[toml_key]['initial_y']}")
                y = jnp.load(config[toml_key]['initial_y'])
            else:
                y = jnp.full(N_particles, config[toml_key]['initial_y'])
        if 'initial_z' in config[toml_key]:
            if isinstance(config[toml_key]['initial_z'], str):
                print(f"Loading initial_z from external source: {config[toml_key]['initial_z']}")
                z = jnp.load(config[toml_key]['initial_z'])
            else:
                z = jnp.full(N_particles, config[toml_key]['initial_z'])
        if 'initial_vx' in config[toml_key]:
            if isinstance(config[toml_key]['initial_vx'], str):
                print(f"Loading initial_vx from external source: {config[toml_key]['initial_vx']}")
                vx = jnp.load(config[toml_key]['initial_vx'])
            else:
                vx = jnp.full(N_particles, config[toml_key]['initial_vx'])
        if 'initial_vy' in config[toml_key]:
            if isinstance(config[toml_key]['initial_vy'], str):
                print(f"Loading initial_vy from external source: {config[toml_key]['initial_vy']}")
                vy = jnp.load(config[toml_key]['initial_vy'])
            else:
                vy = jnp.full(N_particles, config[toml_key]['initial_vy'])
        if 'initial_vz' in config[toml_key]:
            if isinstance(config[toml_key]['initial_vz'], str):
                print(f"Loading initial_vz from external source: {config[toml_key]['initial_vz']}")
                vz = jnp.load(config[toml_key]['initial_vz'])
            else:
                vz = jnp.full(N_particles, config[toml_key]['initial_vz'])
        print('\n')

        update_pos = True
        update_v   = True
        update_vx  = True
        update_vy  = True
        update_vz  = True
        update_x   = True
        update_y   = True
        update_z   = True

        weight = 1.0
        if weight in config[toml_key]:
            weight = config[toml_key]['weight']

        if 'update_pos' in config[toml_key]:
            update_pos = config[toml_key]['update_pos']
            print(f"update_pos: {update_pos}")
        if 'update_v' in config[toml_key]:
            update_v = config[toml_key]['update_v']
            print(f"update_v: {update_v}")
        if 'update_vx' in config[toml_key]:
            update_vx = config[toml_key]['update_vx']
            print(f"update_vx: {update_vx}")
        if 'update_vy' in config[toml_key]:
            update_vy = config[toml_key]['update_vy']
            print(f"update_vy: {update_vy}")
        if 'update_vz' in config[toml_key]:
            update_vz = config[toml_key]['update_vz']
            print(f"update_vz: {update_vz}")
        if 'update_x' in config[toml_key]:
            update_x = config[toml_key]['update_x']
            print(f"update_x: {update_x}")
        if 'update_y' in config[toml_key]:
            update_y = config[toml_key]['update_y']
            print(f"update_y: {update_y}")
        if 'update_z' in config[toml_key]:
            update_z = config[toml_key]['update_z']
            print(f"update_z: {update_z}")


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
            bound=bounded,
            w=w,
            g=g,
        )
        particles.append(particle)

    return particles

def fermi_dirac_distribution(energy, fermi_energy, T, kb):
    """
    Calculate the Fermi-Dirac distribution for a given energy, Fermi energy, temperature, and Boltzmann constant.

    Args:
        energy (float): The energy of the particle.
        fermi_energy (float): The Fermi energy of the particle.
        T (float): The temperature of the system.
        kb (float): The Boltzmann constant.
    Returns:
        float: The value of the Fermi-Dirac distribution at the given energy.

    """
    return 1 / (jnp.exp( (energy-fermi_energy)/kb*T) + 1)

def initial_bound_particles(N_particles, minx, maxx, miny, maxy, minz, maxz, mass, T, fermi_energy, kb, key1, key2, key3):
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
    x = jax.random.uniform(key1, shape = (N_particles,), minval=minx, maxval=maxx)
    y = jax.random.uniform(key2, shape = (N_particles,), minval=miny, maxval=maxy)
    z = jax.random.uniform(key3, shape = (N_particles,), minval=minz, maxval=maxz)
    # initialize the positions of the particles
    
    uniformdist = jax.random.uniform(key1, shape=(N_particles,), minval=1e-3, maxval=1)
    # Generate random numbers from a uniform distribution
    energy = fermi_energy - kb * T * jnp.log(uniformdist)
    # print(energy)
    # print(kb*T)
    # print(fermi_energy)
    vmag = jnp.sqrt(2*energy/mass)
    # get the magnitude of the velocity from the fermi dirac distribution

    # plt.hist(energy, bins=30, alpha=0.75, edgecolor='black')
    # plt.title('Histogram of Velocity Magnitude')
    # plt.xlabel('Velocity Magnitude')
    # plt.ylabel('Frequency')
    # plt.show()
    # exit()

    random_numbers = jax.random.dirichlet(key3, jnp.ones(3))
    # Generate three random numbers that sum to 1 from the dirichlet distribution for the velocity components
    v_x = vmag * random_numbers[0]
    v_y = vmag * random_numbers[1]
    v_z = vmag * random_numbers[2]
    # initialize the velocity components using the fermi dirac distribution and the random numbers

    return x, y, z, v_x, v_y, v_z



def initial_particles(N_particles, minx, maxx, miny, maxy, minz, maxz, mass, T, kb, key1, key2, key3):
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
    x = jax.random.uniform(key1, shape = (N_particles,), minval=minx, maxval=maxx)
    y = jax.random.uniform(key2, shape = (N_particles,), minval=miny, maxval=maxy)
    z = jax.random.uniform(key3, shape = (N_particles,), minval=minz, maxval=maxz)
    # initialize the positions of the particles
    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z


@jit
def total_KE(particle_species_list):
    """
    Calculate the total kinetic energy of all particle species.

    Args:
        particle_species_list (list): A list of particle_species objects.

    Returns:
        float: The total kinetic energy of all particle species.
    """
    total_ke = 0.0
    for species in particle_species_list:
        vx, vy, vz = species.get_velocity()
        total_ke += 0.5 * species.mass * jnp.sum(vx**2 + vy**2 + vz**2)
    return total_ke

@jit
def total_momentum(m, vx, vy, vz):
    """
    Calculate the total momentum of the particles.

    Args:
        m (float): The mass of the particle.
        v (jax.numpy.ndarray): The velocity of the particle.

    Returns:
        float: The total momentum of the particle.
    """
    return m * jnp.sum( jnp.sqrt( vx**2 + vy**2 + vz**2 ) )

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
    A class to represent a species of particles in a simulation.
    Attributes
    ----------
    name : str
        The name of the particle species.
    N_particles : int
        The number of particles in the species.
    charge : float
        The charge of each particle.
    mass : float
        The mass of each particle.
    weight : float, optional
        The weight of the particles (default is 1).
    T : float
        The temperature of the particles.
    v1, v2, v3 : array-like
        The velocity components of the particles.
    x1, x2, x3 : array-like
        The position components of the particles.
    dx, dy, dz : float
        The resolution of the grid in each dimension.
    x_wind, y_wind, z_wind : float
        The dimensions of the simulation box.
    zeta1, zeta2, eta1, eta2, xi1, xi2 : array-like
        The subcell positions for charge conservation.
    bc : str, optional
        The boundary condition type (default is 'periodic').
    update_x, update_y, update_z : bool, optional
        Flags to update the position components (default is True).
    update_vx, update_vy, update_vz : bool, optional
        Flags to update the velocity components (default is True).
    update_pos, update_v : bool, optional
        Flags to update the position and velocity (default is True).
    w, g : array-like, optional
        Frequency and damping matrices for bound species (default is zero matrices).
    Methods
    -------
    get_name():
        Returns the name of the particle species.
    get_charge():
        Returns the charge of the particles multiplied by their weight.
    get_number_of_particles():
        Returns the number of particles in the species.
    get_temperature():
        Returns the temperature of the particles.
    get_velocity():
        Returns the velocity components of the particles.
    get_position():
        Returns the position components of the particles.
    get_mass():
        Returns the mass of the particles multiplied by their weight.
    get_subcell_position():
        Returns the subcell positions for charge conservation.
    get_resolution():
        Returns the resolution of the grid in each dimension.
    get_index():
        Returns the grid indices of the particle positions.
    get_freqmatrix():
        Returns the frequency matrix.
    get_dampingmatrix():
        Returns the damping matrix.
    set_velocity(v1, v2, v3):
        Sets the velocity components of the particles.
    set_position(x1, x2, x3):
        Sets the position components of the particles.
    set_mass(mass):
        Sets the mass of the particles.
    set_weight(weight):
        Sets the weight of the particles.
    calc_subcell_position():
        Calculates the subcell positions for charge conservation.
    kinetic_energy():
        Calculates the kinetic energy of the particles.
    momentum():
        Calculates the momentum of the particles.
    periodic_boundary_condition(x_wind, y_wind, z_wind):
        Applies periodic boundary conditions to the particle positions.
    reflecting_boundary_condition(x_wind, y_wind, z_wind):
        Applies reflecting boundary conditions to the particle positions and velocities.
    update_position(dt, x_wind, y_wind, z_wind):
        Updates the position of the particles based on their velocities and boundary conditions.
    tree_flatten():
        Flattens the particle species object for JAX transformations.
    tree_unflatten(aux_data, children):
        Unflattens the particle species object for JAX transformations.
    """


    def __init__(self, name, N_particles, charge, mass, T, v1, v2, v3, x1, x2, x3, subcells, \
            xwind, ywind, zwind, dx, dy, dz, weight=1, bc='periodic', update_x=True, update_y=True, update_z=True, \
                update_vx=True, update_vy=True, update_vz=True, update_pos=True, update_v=True, w=jnp.zeros((3,3)), \
                    g=jnp.zeros((3,3)), bound=False):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
        self.weight = weight
        self.T = T
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
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

        self.bound = bound
        self.w = w
        self.g = g
        # matricies for bounded species

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

    def get_position(self):
        return self.x1, self.x2, self.x3

    def get_mass(self):
        return self.mass*self.weight

    def get_subcell_position(self):
        return self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2

    def get_resolution(self):
        return self.dx, self.dy, self.dz

    def get_index(self):
        return compute_index(self.x1, self.dx, self.x_wind), compute_index(self.x2, self.dy, self.y_wind), compute_index(self.x3, self.dz, self.z_wind)

    def is_bounded(self):
        return self.bound

    def get_freqmatrix(self):
        return self.w
    def get_dampingmatrix(self):
        return self.g
    
    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            if self.update_vx:
                self.v1 = v1
            if self.update_vy:
                self.v2 = v2
            if self.update_vz:
                self.v3 = v3

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
        return 0.5 * self.weight * self.mass * jnp.sum(self.v1**2 + self.v2**2 + self.v3**2)

    def momentum(self):
        return self.mass * self.weight * jnp.sum(jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2))

    def periodic_boundary_condition(self, x_wind, y_wind, z_wind):
        self.x1 = jnp.where(self.x1 > x_wind/2, -x_wind/2, self.x1)
        self.x1 = jnp.where(self.x1 < -x_wind/2,  x_wind/2, self.x1)
        self.x2 = jnp.where(self.x2 > y_wind/2, -y_wind/2, self.x2)
        self.x2 = jnp.where(self.x2 < -y_wind/2,  y_wind/2, self.x2)
        self.x3 = jnp.where(self.x3 > z_wind/2, -z_wind/2, self.x3)
        self.x3 = jnp.where(self.x3 < -z_wind/2,  z_wind/2, self.x3)

    def reflecting_boundary_condition(self, x_wind, y_wind, z_wind):

        self.v1 = jnp.where((self.x1 > x_wind/2) | (self.x1 < -x_wind/2), -self.v1, self.v1)
        self.x1 = jnp.where(self.x1 > x_wind/2, x_wind/2, self.x1)
        self.x1 = jnp.where(self.x1 < -x_wind/2, -x_wind/2, self.x1)

        self.v2 = jnp.where((self.x2 > y_wind/2) | (self.x2 < -y_wind/2), -self.v2, self.v2)
        self.x2 = jnp.where(self.x2 > y_wind/2, y_wind/2, self.x2)
        self.x2 = jnp.where(self.x2 < -y_wind/2, -y_wind/2, self.x2)

        self.v3 = jnp.where((self.x3 > z_wind/2) | (self.x3 < -z_wind/2), -self.v3, self.v3)
        self.x3 = jnp.where(self.x3 > z_wind/2, z_wind/2, self.x3)
        self.x3 = jnp.where(self.x3 < -z_wind/2, -z_wind/2, self.x3)


    def update_position(self, dt, x_wind, y_wind, z_wind):
        if self.update_pos:
            if self.update_x:
                self.x1 = self.x1 + self.v1 * dt
            if self.update_y:
                self.x2 = self.x2 + self.v2 * dt
            if self.update_z:
                self.x3 = self.x3 + self.v3 * dt
            # update the position of the particles
            if self.bc == 'periodic':
                #print('Using periodic boundary conditions')
                self.periodic_boundary_condition(x_wind, y_wind, z_wind)
            elif self.bc == 'reflecting':
                #print('Using reflecting boundary conditions')
                self.reflecting_boundary_condition(x_wind, y_wind, z_wind)
            # self.x1, self.x2, self.x3, self.v1, self.v2, self.v3 = self.boundary_condition()
            # apply boundary conditions

            self.zeta1 = self.zeta2
            self.eta1  = self.eta2
            self.xi1   = self.xi2
            self.zeta2, self.eta2, self.xi2 = self.calc_subcell_position()
            # update the subcell positions for charge conservation algorithm


    def tree_flatten(self):
        children = (
            self.v1, self.v2, self.v3, \
            self.x1, self.x2, self.x3, \
            self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2
        )

        aux_data = (
            self.name, self.N_particles, self.charge, self.mass, self.T, \
            self.x_wind, self.y_wind, self.z_wind, self.dx, self.dy, self.dz, \
            self.weight, self.bc, self.update_pos, self.update_v, self.update_x, self.update_y, \
            self.update_z, self.update_vx, self.update_vy, self.update_vz, self.w, self.g
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        v1, v2, v3, x1, x2, x3, zeta1, zeta2, eta1, eta2, xi1, xi2 = children

        name, N_particles, charge, mass, T, x_wind, y_wind, z_wind, dx, dy, \
            dz, weight, bc, update_pos, update_v, update_x, update_y, update_z, \
                update_vx, update_vy, update_vz, w, g = aux_data

        subcells = zeta1, zeta2, eta1, eta2, xi1, xi2

        return cls(
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
            w=w,
            g=g
        )