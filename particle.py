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
def total_KE(m, vx, vy, vz):
    """
    Calculate the total kinetic energy of the particles.

    Parameters:
    - m (float): The mass of the particle.
    - v (jax.numpy.ndarray): The velocity of the particle.

    Returns:
    - float: The total kinetic energy of the particle.
    """
    return 0.5 * m * jnp.sum( vx**2 + vy**2 + vz**2 )

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
class particle:
    """
    A class to represent a particle in 3D space.
    Attributes
    ----------
    mass : float
        Mass of the particle.
    vx : float
        Velocity of the particle along the x-axis.
    vy : float
        Velocity of the particle along the y-axis.
    vz : float
        Velocity of the particle along the z-axis.
    x : float
        Position of the particle along the x-axis.
    y : float
        Position of the particle along the y-axis.
    z : float
        Position of the particle along the z-axis.
    Methods
    -------
    get_velocity():
        Returns the velocity components (vx, vy, vz) of the particle.
    get_position():
        Returns the position components (x, y, z) of the particle.
    get_mass():
        Returns the mass of the particle.
    set_velocity(vx, vy, vz):
        Sets the velocity components (vx, vy, vz) of the particle.
    set_position(x, y, z):
        Sets the position components (x, y, z) of the particle.
    set_mass(mass):
        Sets the mass of the particle.
    kinetic_energy():
        Calculates and returns the kinetic energy of the particle.
    momentum():
        Calculates and returns the momentum of the particle.
    """

    def __init__(self, mass, vx, vy, vz, x, y, z):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
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
        return 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)
    def momentum(self):
        return self.mass * jnp.sqrt( self.vx**2 + self.vy**2 + self.vz**2 )

@jit
class particle_species:
    """
    A class to represent a species of particles.
    Attributes
    ----------
    N_particles : int
        Number of particles in the species.
    mass : float or jnp.array
        Mass of the particles. Can be a single value or an array of values.
    particles : list
        List of particle objects.
    bc : str
        Boundary condition type. Default is 'periodic'.
    Methods
    -------
    get_velocity():
        Returns the velocities of all particles.
    get_position():
        Returns the positions of all particles.
    get_mass():
        Returns the masses of all particles.
    set_velocity(vx, vy, vz):
        Sets the velocities of all particles.
    set_position(x, y, z):
        Sets the positions of all particles.
    set_mass(mass):
        Sets the masses of all particles.
    kinetic_energy():
        Returns the total kinetic energy of all particles.
    momentum():
        Returns the total momentum of all particles.
    periodic_boundary_condition(x_wind, y_wind, z_wind):
        Applies periodic boundary conditions to the particles.
    update_position(dt, x_wind, y_wind, z_wind):
        Updates the positions of the particles using Euler's method and applies boundary conditions.
    """
    def __init__(self, N_particles, mass, vx, vy, vz, x, y, z, bc='periodic'):
        self.N_particles = N_particles
        self.mass = mass
        if isinstance(self.mass, jnp.array):
            self.particles = [particle(mass[i], vx[i], vy[i], vz[i], x[i], y[i], z[i]) for i in range(N_particles)]
        else:
            self.particles = [particle(mass, vx[i], vy[i], vz[i], x[i], y[i], z[i]) for i in range(N_particles)]
        # build a list of particles
        self.bc = bc

    def get_velocity(self):
        return [p.get_velocity() for p in self.particles]
    def get_position(self):
        return [p.get_position() for p in self.particles]
    def get_mass(self):
        return [p.get_mass() for p in self.particles]
    def set_velocity(self, vx, vy, vz):
        for i in range(self.N_particles):
            self.particles[i].set_velocity(vx[i], vy[i], vz[i])
    def set_position(self, x, y, z):
        for i in range(self.N_particles):
            self.particles[i].set_position(x[i], y[i], z[i])
    def set_mass(self, mass):
        for i in range(self.N_particles):
            self.particles[i].set_mass(mass[i])
    def kinetic_energy(self):
        return sum([p.kinetic_energy() for p in self.particles])
    def momentum(self):
        return sum([p.momentum() for p in self.particles])
    
    def periodic_boundary_condition(self, x_wind, y_wind, z_wind):
        for i in range(self.N_particles):
            x, y, z = self.particles[i].get_position()
            x, y, z = periodic_boundary_condition(x_wind, y_wind, z_wind, x, y, z)
            self.particles[i].set_position(x, y, z)

    def update_position(self, dt, x_wind, y_wind, z_wind):
        for i in range(self.N_particles):
            x, y, z = self.particles[i].get_position()
            vx, vy, vz = self.particles[i].get_velocity()
            x = euler_update(x, vx, dt)
            y = euler_update(y, vy, dt)
            z = euler_update(z, vz, dt)
            # update the position of the particles
            self.particles[i].set_position(x, y, z)
        if self.bc == 'periodic':
            self.periodic_boundary_condition(x_wind, y_wind, z_wind)
        # apply periodic boundary conditions