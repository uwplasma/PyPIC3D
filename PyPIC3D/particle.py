import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
from jax import tree_util
from functools import partial


def initial_particles(N_particles, x_wind, y_wind, z_wind, mass, T, kb, key1, key2, key3):
    """
    Initializes the velocities and positions of the particles.

    Parameters:
    - N_particles (int): The number of particles.
    - x_wind (float): The maximum value for the x-coordinate of the particles' positions.
    - y_wind (float): The maximum value for the y-coordinate of the particles' positions.
    - z_wind (float): The maximum value for the z-coordinate of the particles' positions.
    - mass (float): The mass of the particles.
    - T (float): The temperature of the system.
    - kb (float): The Boltzmann constant.
    - key (jax.random.PRNGKey): The random key for generating random numbers.

    Returns:
    - x (jax.numpy.ndarray): The x-coordinates of the particles' positions.
    - y (jax.numpy.ndarray): The y-coordinates of the particles' positions.
    - z (jax.numpy.ndarray): The z-coordinates of the particles' positions.
    - v_x (numpy.ndarray): The x-component of the particles' velocities.
    - v_y (numpy.ndarray): The y-component of the particles' velocities.
    - v_z (numpy.ndarray): The z-component of the particles' velocities.
    """
    initial_wind = 1.0
    # what is the initial window for the particles (as a fraction of spatial window)
    x = jax.random.uniform(key1, shape = (N_particles,), minval=-initial_wind*x_wind/2, maxval=initial_wind*x_wind/2)
    y = jax.random.uniform(key2, shape = (N_particles,), minval=-initial_wind*y_wind/2, maxval=initial_wind*y_wind/2)
    z = jax.random.uniform(key3, shape = (N_particles,), minval=-initial_wind*z_wind/2, maxval=initial_wind*z_wind/2)
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

    Parameters:
    - particle_species_list (list): A list of particle_species objects.

    Returns:
    - float: The total kinetic energy of all particle species.
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

    Parameters:
    - m (float): The mass of the particle.
    - v (jax.numpy.ndarray): The velocity of the particle.

    Returns:
    - float: The total momentum of the particle.
    """
    return m * jnp.sum( jnp.sqrt( vx**2 + vy**2 + vz**2 ) )

@jit
def compute_index(x, dx, window):
    """
    Compute the index of a position in a discretized space.

    Parameters:
    x (float or ndarray): The position(s) to compute the index for.
    dx (float): The discretization step size.

    Returns:
    int or ndarray: The computed index/indices as integer(s).
    """
    scaled_x = x + window/2
    return jnp.floor( scaled_x / dx).astype(int)

class particle_species:
    """
    A class to represent a species of particles in a simulation.

    Attributes:
    -----------
    name : str
        The name of the particle species.
    N_particles : int
        The number of particles in the species.
    charge : float
        The charge of each particle.
    mass : float
        The mass of each particle.
    v1, v2, v3 : array-like
        The velocity components of the particles.
    x1, x2, x3 : array-like
        The position components of the particles.
    dx, dy, dz : float
        The resolution of the grid in each dimension.
    zeta1, zeta2, eta1, eta2, xi1, xi2 : array-like
        The subcell positions for charge conservation.
    bc : str, optional
        The boundary condition type (default is 'periodic').
    update_pos : bool, optional
        Flag to determine if positions should be updated (default is True).
    update_v : bool, optional
        Flag to determine if velocities should be updated (default is True).

    Methods:
    --------
    get_name():
        Returns the name of the particle species.
    get_charge():
        Returns the charge of the particles.
    get_number_of_particles():
        Returns the number of particles.
    get_velocity():
        Returns the velocity components of the particles.
    get_position():
        Returns the position components of the particles.
    get_mass():
        Returns the mass of the particles.
    get_subcell_position():
        Returns the subcell positions for charge conservation.
    get_resolution():
        Returns the grid resolution in each dimension.
    get_index():
        Returns the grid indices of the particle positions.
    set_velocity(v1, v2, v3):
        Sets the velocity components of the particles.
    set_position(x1, x2, x3):
        Sets the position components of the particles.
    update_subcell_position():
        Updates the subcell positions for charge conservation.
    set_mass(mass):
        Sets the mass of the particles.
    kinetic_energy():
        Returns the kinetic energy of the particles.
    momentum():
        Returns the momentum of the particles.
    periodic_boundary_condition(x_wind, y_wind, z_wind):
        Applies periodic boundary conditions to the particle positions.
    update_position(dt, x_wind, y_wind, z_wind):
        Updates the positions of the particles using Euler's method and applies boundary conditions.
    """

    def __init__(self, name, N_particles, charge, mass, T, v1, v2, v3, x1, x2, x3, xwind, \
                ywind, zwind, dx, dy, dz, bc='periodic', update_pos=True, update_v=True):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
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
        self.zeta1 = ( self.x1 + self.x_wind/2 ) % self.dx
        self.zeta2 = self.zeta1
        self.eta1  = ( self.x2 + self.y_wind/2 ) % self.dy
        self.eta2  = self.eta1
        self.xi1   = ( self.x3 + self.z_wind/2 ) % self.dz
        self.xi2   = self.xi1
        self.bc = bc
        self.update_pos = update_pos
        self.update_v   = update_v

    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge

    def get_number_of_particles(self):
        return self.N_particles

    def get_temperature(self):
        return self.T

    def get_velocity(self):
        return self.v1, self.v2, self.v3

    def get_position(self):
        return self.x1, self.x2, self.x3

    def get_mass(self):
        return self.mass

    def get_subcell_position(self):
        return self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2

    def get_resolution(self):
        return self.dx, self.dy, self.dz

    def get_index(self):
        return compute_index(self.x1, self.dx, self.x_wind), compute_index(self.x2, self.dy, self.y_wind), compute_index(self.x3, self.dz, self.z_wind)

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            self.v1 = v1
            self.v2 = v2
            self.v3 = v3

    def set_position(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def calc_subcell_position(self):
        newzeta = (self.x1 + self.x_wind / 2) % self.dx
        neweta  = (self.x2 + self.y_wind / 2) % self.dy
        newxi   = (self.x3 + self.z_wind / 2) % self.dz

        return newzeta, neweta, newxi

    def set_mass(self, mass):
        self.mass = mass

    def kinetic_energy(self):
        return 0.5 * self.mass * jnp.sum(self.v1**2 + self.v2**2 + self.v3**2)

    def momentum(self):
        return self.mass * jnp.sum(jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2))

    def periodic_boundary_condition(self, x_wind, y_wind, z_wind):
        self.x1 = jnp.where(self.x1 > x_wind/2, -x_wind/2, self.x1)
        self.x1 = jnp.where(self.x1 < -x_wind/2,  x_wind/2, self.x1)
        self.x2 = jnp.where(self.x2 > y_wind/2, -y_wind/2, self.x2)
        self.x2 = jnp.where(self.x2 < -y_wind/2,  y_wind/2, self.x2)
        self.x3 = jnp.where(self.x3 > z_wind/2, -z_wind/2, self.x3)
        self.x3 = jnp.where(self.x3 < -z_wind/2,  z_wind/2, self.x3)

    def update_position(self, dt, x_wind, y_wind, z_wind):
        if self.update_pos:
            self.x1 = self.x1 + self.v1*dt
            self.x2 = self.x2 + self.v2*dt
            self.x3 = self.x3 + self.v3*dt
            # update the position of the particles
            if self.bc == 'periodic':
                self.periodic_boundary_condition(x_wind, y_wind, z_wind)
            # apply periodic boundary conditions

            self.zeta1 = self.zeta2
            self.eta1  = self.eta2
            self.xi1   = self.xi2
            self.zeta2, self.eta2, self.xi2 = self.calc_subcell_position()
            # update the subcell positions for charge conservation algorithm

    def _tree_flatten(self):
        return ((self.v1, self.v2, self.v3, self.x1, self.x2, self.x3, self.zeta1, self.zeta2, self.eta1, self.eta2, self.xi1, self.xi2),
                (self.name, self.N_particles, self.charge, self.mass, self.T, self.x_wind, self.y_wind, self.z_wind, self.dx, self.dy, self.dz, self.bc, self.update_pos, self.update_v))

    @classmethod
    def _tree_unflatten(cls, data, children):
        return cls(data[0], data[1], data[2], data[3], data[4], *children[:3], *children[3:6], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13])

# Register the particle_species class as a PyTree
tree_util.register_pytree_node(
    particle_species,
    particle_species._tree_flatten,
    particle_species._tree_unflatten
)