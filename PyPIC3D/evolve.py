# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.

import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from PyPIC3D.fields import (
    calculateE, update_B, update_E
)

from PyPIC3D.boris import (
    particle_push
)

from PyPIC3D.vector_potential import (
    E_from_A, B_from_A, update_vector_potential
)

@partial(jit, static_argnames=("curl_func", "J_func", "solver", "bc"))
def time_loop_electrostatic(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, bc):
    """
    Advances the simulation by one time step for an electrostatic Particle-In-Cell (PIC) loop.

    This function performs the following steps:
    1. Pushes all particles using the current electric and magnetic fields (typically with the Boris algorithm).
    2. Updates particle positions.
    3. Solves for the new electric field using the Poisson equation, updating the field and potential.
    4. Packs the updated fields and returns the new particle and field states.

    Args:
        particles (list): List of particle objects to be advanced.
        fields (tuple): Tuple containing the field arrays (E, B, J, rho, phi).
        E_grid (ndarray): Grid representing the electric field.
        B_grid (ndarray): Grid representing the magnetic field.
        world (dict): Dictionary containing simulation parameters (e.g., time step 'dt').
        constants (dict): Dictionary of physical constants used in the simulation.
        curl_func (callable): Function to compute the curl of a field (not used in this function).
        J_func (callable): Function to compute the current density (not used in this function).
        solver (callable): Function or object to solve the Poisson equation for the electric field.
        bc (object): Boundary condition object or function for the field solver.

    Returns:
        tuple: Updated particles list and fields tuple (E, B, J, rho, phi).
    """

    E, B, J, rho, phi = fields
    # unpack the fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'], constants)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ############### SOLVE E FIELD ############################################################################################
    E, phi, rho = calculateE(world, particles, constants, rho, phi, solver, bc)
    # calculate the electric field using the Poisson equation

    fields = (E, B, J, rho, phi)
    # pack the fields into a tuple

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "bc"))
def time_loop_electrodynamic(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, bc):
    """
    Advances the simulation by one time step using the electrodynamic Particle-In-Cell (PIC) method.

    This function performs the following steps:
        1. Pushes all particles using the Boris algorithm and updates their positions.
        2. Computes the current density from the updated particle positions and velocities.
        3. Updates the electric and magnetic fields using Maxwell's equations and the computed current density.

    Args:
        particles (list): List of particle objects to be updated.
        fields (tuple): Tuple containing the field arrays (E, B, J, rho, phi).
        E_grid (ndarray): Grid for the electric field.
        B_grid (ndarray): Grid for the magnetic field.
        world (dict): Dictionary containing simulation parameters (e.g., time step 'dt').
        constants (dict): Dictionary of physical constants used in the simulation.
        curl_func (callable): Function to compute the curl of a field.
        J_func (callable): Function to compute the current density from particles.
        solver (object): Field solver object (not used directly in this function).
        bc (object): Boundary condition handler (not used directly in this function).

    Returns:
        tuple: Updated particles list and fields tuple (E, B, J, rho, phi).
    """

    E, B, J, rho, phi = fields
    # unpack the fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'], constants)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    J = J_func(particles, J, constants, world, E_grid)
    # calculate the current density based on the selected method
    E = update_E(E, B, J, world, constants, curl_func)
    # update the electric field using the curl of the magnetic field
    B = update_B(E, B, world, constants, curl_func)
    # update the magnetic field using the curl of the electric field

    fields = (E, B, J, rho, phi)
    # pack the fields into a tuple

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "bc"))
def time_loop_vector_potential(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, bc):
    """
    Advances the simulation by one time step using the vector potential formulation.

    This function performs a single iteration of the main time loop for a particle-in-cell (PIC) simulation
    using the vector potential approach. It updates particle positions and velocities, computes current density,
    updates the vector potential, and recalculates electric and magnetic fields.

    Args:
        particles (list): List of particle objects to be updated.
        fields (tuple): Tuple containing field arrays (E, B, J, rho, phi, A2, A1, A0).
        E_grid (ndarray): Grid for the electric field.
        B_grid (ndarray): Grid for the magnetic field.
        world (dict): Simulation parameters, including time step ('dt') and grid information.
        constants (dict): Physical constants used in the simulation.
        curl_func (callable): Function to compute the curl of a field.
        J_func (callable): Function to compute the current density from particles.
        solver (callable): Function to solve field equations (not used directly in this function).
        bc (object): Boundary condition handler (not used directly in this function).

    Returns:
        tuple: Updated particles and fields after one time step.
    """

    E, B, J, rho, phi, A2, A1, A0 = fields
    # unpack the fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'], constants)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    J = J_func(particles, J, constants, world, E_grid)
    # calculate the current density using the selected method

    A0 = A1
    A1 = A2
    # update the vector potential for the next iteration
    A2 = update_vector_potential(J, world, constants, A1, A0)
    # update the vector potential based on the current density J

    E = E_from_A(A2, A0, world)
    # calculate the electric field from the vector potential using centered finite difference
    B = B_from_A(A1, world)
    # calculate the magnetic field from the vector potential using centered finite difference


    fields = (E, B, J, rho, phi, A2, A1, A0)
    # pack the fields into a tuple

    return particles, fields