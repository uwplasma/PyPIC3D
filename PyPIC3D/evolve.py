# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.

#from memory_profiler import profile
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from PyPIC3D.fields import (
    calculateE, update_B, update_E
)

from PyPIC3D.J import (
    VB_correction
)

from PyPIC3D.boris import (
    particle_push
)

#@profile

@partial(jit, static_argnums=(10, 11, 12))
def time_loop_electrostatic(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, curl_func, solver, bc):
    """
    Perform a time loop for an electrostatic simulation.

    Args:
        t (float): Current time step.
        particles (list): List of particle objects.
        E (tuple): Electric field components (Ex, Ey, Ez).
        B (tuple): Magnetic field components (Bx, By, Bz).
        J (tuple): Current density components (Jx, Jy, Jz).
        rho (array): Charge density.
        phi (array): Electric potential.
        E_grid (array): Electric field grid.
        B_grid (array): Magnetic field grid.
        world (dict): Simulation world parameters including 'dt', 'x_wind', 'y_wind', 'z_wind'.
        constants (dict): Physical constants.
        pecs (list): List of PEC (Perfect Electric Conductor) boundary conditions.
        lasers (list): List of laser objects for injecting fields.
        surfaces (list): List of material surface objects.
        curl_func (function): Function to calculate the curl of a field.
        M (array): Matrix for solving the Poisson equation.
        solver (object): Solver object for the Poisson equation.
        bc (dict): Boundary conditions.
        verbose (bool): Flag for verbose output.
        GPUs (list): List of GPU devices.

    Returns:
        tuple: Updated particles, electric field components (Ex, Ey, Ez), magnetic field components (Bx, By, Bz), 
            current density components (Jx, Jy, Jz), electric potential (phi), and charge density (rho).
    """

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'])
        # use boris push for particle velocities

        particles[i].update_position(world['dt'])
        # update the particle positions

    ############### SOLVE E FIELD ############################################################################################
    E, phi, rho = calculateE(world, particles, constants, rho, phi, solver, bc)
    # calculate the electric field using the Poisson equation

    return particles, E, B, J, phi, rho

@partial(jit, static_argnums=(10, 11, 12))
def time_loop_electrodynamic(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, curl_func, solver, bc):
    """
    Perform a time loop for electrodynamic simulation.

    Args:
        t (float): Current time step.
        particles (list): List of particle objects.
        E (tuple): Electric field components (Ex, Ey, Ez).
        B (tuple): Magnetic field components (Bx, By, Bz).
        J (tuple): Current density components (Jx, Jy, Jz).
        rho (array): Charge density.
        phi (array): Electric potential.
        E_grid (array): Electric field grid.
        B_grid (array): Magnetic field grid.
        world (dict): Dictionary containing simulation parameters such as 'dt', 'Nx', 'Ny', 'Nz', 'x_wind', 'y_wind', 'z_wind'.
        constants (dict): Dictionary containing physical constants.
        pecs (list): List of PEC (Perfect Electric Conductor) boundary condition objects.
        lasers (list): List of laser pulse objects.
        surfaces (list): List of material surface objects.
        curl_func (function): Function to compute the curl of a field.
        M (array): Mass matrix.
        solver (object): Solver object for field equations.
        bc (object): Boundary condition object.
        verbose (bool): Flag to enable verbose output.
        GPUs (list): List of GPU devices.

    Returns:
        tuple: Updated particles, electric field components (Ex, Ey, Ez), magnetic field components (Bx, By, Bz),
            current density components (Jx, Jy, Jz), electric potential (phi), and charge density (rho).
    """

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'])
        # use boris push for particle velocities

        particles[i].update_position(world['dt'])
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    J = VB_correction(particles, J, constants)
    # calculate the corrections for charge conservation using villasenor buneamn 1991
    E = update_E(E, B, J, world, constants, curl_func)
    # update the electric field using the curl of the magnetic field
    B = update_B(E, B, world, constants, curl_func)
    # update the magnetic field using the curl of the electric field

    return particles, E, B, J, phi, rho