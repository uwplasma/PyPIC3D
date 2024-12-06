import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
from jax.experimental import checkify
import math
from pyevtk.hl import gridToVTK
from jax import tree_util
from jax.tree_util import register_pytree_node_class
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
    T : float
        The temperature of the particle species.
    v1, v2, v3 : array-like
        The velocity components of the particles.
    x1, x2, x3 : array-like
        The position components of the particles.
    dx, dy, dz : float
        The resolution of the grid in each dimension.
    x_wind, y_wind, z_wind : float
        The wind components in each dimension.
    zeta1, zeta2, eta1, eta2, xi1, xi2 : float
        The subcell positions for charge conservation.
    bc : str, optional
        The boundary condition type (default is 'periodic').
    update_pos : bool, optional
        Flag to update the position of particles (default is True).
    update_v : bool, optional
        Flag to update the velocity of particles (default is True).

    Methods
    -------
    get_name():
        Returns the name of the particle species.
    get_charge():
        Returns the charge of the particles.
    get_number_of_particles():
        Returns the number of particles in the species.
    get_temperature():
        Returns the temperature of the particle species.
    get_velocity():
        Returns the velocity components of the particles.
    get_position():
        Returns the position components of the particles.
    get_mass():
        Returns the mass of the particles.
    get_subcell_position():
        Returns the subcell positions for charge conservation.
    get_resolution():
        Returns the resolution of the grid in each dimension.
    get_index():
        Computes and returns the index of the particles in the grid.
    set_velocity(v1, v2, v3):
        Sets the velocity components of the particles.
    set_position(x1, x2, x3):
        Sets the position components of the particles.
    calc_subcell_position():
        Calculates and returns the new subcell positions.
    set_mass(mass):
        Sets the mass of the particles.
    kinetic_energy():
        Computes and returns the kinetic energy of the particles.
    momentum():
        Computes and returns the momentum of the particles.
    periodic_boundary_condition(x_wind, y_wind, z_wind):
        Applies periodic boundary conditions to the particle positions.
    update_position(dt, x_wind, y_wind, z_wind):
        Updates the position of the particles based on their velocity and time step.
    tree_flatten():
        Flattens the particle species object for serialization.
    tree_unflatten(aux_data, children):
        Unflattens the particle species object from serialized data.
    """


    def __init__(self, name, N_particles, charge, mass, T, v1, v2, v3, x1, x2, x3, subcells, \
            xwind, ywind, zwind, dx, dy, dz, bc='periodic', update_x=True, update_y=True, update_z=True, \
                update_vx=True, update_vy=True, update_vz=True, update_pos=True, update_v=True):
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
            if self.update_x:
                self.x1 = self.x1 + self.v1 * dt
            if self.update_y:
                self.x2 = self.x2 + self.v2 * dt
            if self.update_z:
                self.x3 = self.x3 + self.v3 * dt
            # update the position of the particles
            if self.bc == 'periodic':
                self.periodic_boundary_condition(x_wind, y_wind, z_wind)
            # apply periodic boundary conditions

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
            self.bc, self.update_pos, self.update_v, self.update_x, self.update_y, \
            self.update_z, self.update_vx, self.update_vy, self.update_vz
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        v1, v2, v3, x1, x2, x3, zeta1, zeta2, eta1, eta2, xi1, xi2 = children

        name, N_particles, charge, mass, T, x_wind, y_wind, z_wind, dx, dy, \
            dz, bc, update_pos, update_v, update_x, update_y, update_z, update_vx, update_vy, update_vz = aux_data

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
            bc=bc,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            update_pos=update_pos,
            update_v=update_v
        )