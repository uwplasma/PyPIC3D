import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK


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


def cold_start_init(start, N_particles, x_wind, y_wind, z_wind, mass, T, kb, key1, key2, key3):
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
    x = start * jnp.ones(N_particles)
    y = start * jnp.ones(N_particles)
    z = start * jnp.ones(N_particles)
    # initialize the positions of the particles

    std = kb * T / mass
    v_x = np.random.normal(0, std, N_particles)
    v_y = np.random.normal(0, std, N_particles)
    v_z = np.random.normal(0, std, N_particles)
    # initialize the particles with a maxwell boltzmann distribution.
    return x, y, z, v_x, v_y, v_z


@jit
def periodic_boundary_condition(x_wind, y_wind, z_wind, x, y, z):
    """
    Implement periodic boundary conditions for the particles.

    Returns:
    - x (jax.numpy.ndarray): The x-coordinates of the particles' positions.
    - y (jax.numpy.ndarray): The y-coordinates of the particles' positions.
    - z (jax.numpy.ndarray): The z-coordinates of the particles' positions.
    """
    x = jnp.where(x > x_wind/2, -x_wind/2, x)
    x = jnp.where(x < -x_wind/2,  x_wind/2, x)
    y = jnp.where(y > y_wind/2, -y_wind/2, y)
    y = jnp.where(y < -y_wind/2,  y_wind/2, y)
    z = jnp.where(z > z_wind/2, -z_wind/2, z)
    z = jnp.where(z < -z_wind/2,  z_wind/2, z)
    return x, y, z    

@jit
def euler_update(s, v, dt):
    """
    Update the position of the particles using the Euler method.

    Parameters:
    - s (jax.numpy.ndarray): The current position of the particles.
    - v (jax.numpy.ndarray): The velocity of the particles.
    - dt (float): The time step for the update.

    Returns:
    - jax.numpy.ndarray: The updated position of the particles.
    """
    return s + v * dt


def update_position(x, y, z, vx, vy, vz, dt, x_wind, y_wind, z_wind, bc='periodic'):
    """
    Update the position of the particles.

    Parameters:
    x (float): The current x-coordinate of the particle.
    y (float): The current y-coordinate of the particle.
    z (float): The current z-coordinate of the particle.
    vx (float): The velocity in the x-direction of the particle.
    vy (float): The velocity in the y-direction of the particle.
    vz (float): The velocity in the z-direction of the particle.
    dt (float): The time step for the update.

    Returns:
    tuple: A tuple containing the updated x, y, and z coordinates of the particle.
    """

    x = euler_update(x, vx, dt)
    y = euler_update(y, vy, dt)
    z = euler_update(z, vz, dt)
    # update the position of the particles

    if bc == 'periodic':
        x, y, z = periodic_boundary_condition(x_wind, y_wind, z_wind, x, y, z)
    # apply periodic boundary conditions
    return x, y, z

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
        The charge of the particles.
    mass : float
        The mass of the particles.
    vx : array-like
        The velocity of the particles in the x-direction.
    vy : array-like
        The velocity of the particles in the y-direction.
    vz : array-like
        The velocity of the particles in the z-direction.
    x : array-like
        The position of the particles in the x-direction.
    y : array-like
        The position of the particles in the y-direction.
    z : array-like
        The position of the particles in the z-direction.
    bc : str, optional
        The boundary condition type (default is 'periodic').
    Methods:
    --------
    get_charge():
        Returns the charge of the particles.
    get_number_of_particles():
        Returns the number of particles.
    get_velocity():
        Returns the velocity of the particles as a tuple (vx, vy, vz).
    get_position():
        Returns the position of the particles as a tuple (x, y, z).
    get_mass():
        Returns the mass of the particles.
    set_velocity(vx, vy, vz):
        Sets the velocity of the particles.
    set_position(x, y, z):
        Sets the position of the particles.
    set_mass(mass):
        Sets the mass of the particles.
    kinetic_energy():
        Calculates and returns the kinetic energy of the particles.
    momentum():
        Calculates and returns the momentum of the particles.
    periodic_boundary_condition(x_wind, y_wind, z_wind):
        Applies periodic boundary conditions to the particles' positions.
    update_position(dt, x_wind, y_wind, z_wind):
        Updates the position of the particles based on their velocity and time step, 
        and applies periodic boundary conditions if specified.
    """
    def __init__(self, name, N_particles, charge, mass, vx, vy, vz, x, y, z, bc='periodic'):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.x = x
        self.y = y
        self.z = z
        self.bc = bc

    def get_name(self):
        return self.name
    
    def get_charge(self):
        return self.charge
    
    def get_number_of_particles(self):
        return self.N_particles
    
    def get_velocity(self):
        return self.vx, self.vy, self.vz
    
    def get_position(self):
        return self.x, self.y, self.z
    
    def get_mass(self):
        return self.mass
    
    def set_velocity(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_mass(self, mass):
        self.mass = mass

    def kinetic_energy(self):
        return 0.5 * self.mass * jnp.sum(self.vx**2 + self.vy**2 + self.vz**2)
    
    def momentum(self):
        return self.mass * jnp.sum(jnp.sqrt(self.vx**2 + self.vy**2 + self.vz**2))
    
    def periodic_boundary_condition(self, x_wind, y_wind, z_wind):
        self.x, self.y, self.z = periodic_boundary_condition(x_wind, y_wind, z_wind, self.x, self.y, self.z)

    def update_position(self, dt, x_wind, y_wind, z_wind):
        self.x = euler_update(self.x, self.vx, dt)
        self.y = euler_update(self.y, self.vy, dt)
        self.z = euler_update(self.z, self.vz, dt)
        
        if self.bc == 'periodic':
            self.periodic_boundary_condition(x_wind, y_wind, z_wind)
        # apply periodic boundary conditions