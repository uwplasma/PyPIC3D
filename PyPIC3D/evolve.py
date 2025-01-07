# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.


import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.fields import (
    calculateE, update_B, update_E
)

from PyPIC3D.J import (
    VB_correction
)


from PyPIC3D.utils import (
    dump_parameters_to_toml
)

from PyPIC3D.pstd import (
     check_nyquist_criterion
)


from PyPIC3D.boris import (
    particle_push
)

#@partial(jit, static_argnums=(0, 18, 19, 20, 21, 22, 23, 24))
def time_loop(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, curl_func, M, solver, bc, electrostatic, verbose, GPUs):
    """
    Perform a single time step in the simulation loop.

    Parameters:
    t (float): Current time step.
    particles (list): List of particle objects in the simulation.
    Ex, Ey, Ez (array): Electric field components.
    Bx, By, Bz (array): Magnetic field components.
    Jx, Jy, Jz (array): Current density components.
    rho (array): Charge density.
    phi (array): Electric potential.
    E_grid, B_grid (array): Grids for electric and magnetic fields.
    world (dict): Dictionary containing world parameters such as grid size and wind components.
    constants (dict): Dictionary containing physical constants.
    plotting_parameters (dict): Parameters for plotting (not used in the function).
    M (array): Preconditioner matrix.
    solver (object): Solver object for field equations.
    bc (object): Boundary conditions object.
    electrostatic (bool): Flag to indicate if the simulation is electrostatic.
    verbose (bool): Flag to enable verbose output.
    GPUs (bool): Flag to indicate if GPUs are used.

    Returns:
    tuple: Updated particles, electric field components (Ex, Ey, Ez), magnetic field components (Bx, By, Bz),
           current density components (Jx, Jy, Jz), electric potential (phi), and charge density (rho).
    """


    ############### SOLVE E FIELD ############################################################################################
    Ex, Ey, Ez, phi, rho = calculateE(Ex, Ey, Ez, world, particles, constants, rho, phi, M, t, solver, bc, verbose, GPUs, electrostatic)
    if verbose: print(f"Calculating Electric Field, Max Value: {jnp.max(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
    # print the maximum value of the electric field

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):
        if particles[i].get_number_of_particles() > 0:
            if verbose: print(f'Updating {particles[i].get_name()}')
            particles[i] = particle_push(particles[i], Ex, Ey, Ez, Bx, By, Bz, E_grid, B_grid, world['dt'], GPUs)
            # use boris push for particle velocities
            if verbose: print(f"Calculating {particles[i].get_name()} Velocities, Mean Value: {jnp.mean(jnp.abs(particles[i].get_velocity()[0]))}")
            x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
            particles[i].update_position(world['dt'], x_wind, y_wind, z_wind)
            if verbose: print(f"Calculating {particles[i].get_name()} Positions, Mean Value: {jnp.mean(jnp.abs(particles[i].get_position()[0]))}")
            # update the particle positions
    ################ FIELD UPDATE ################################################################################################
    if not electrostatic:
        Nx, Ny, Nz = world['Nx'], world['Ny'], world['Nz']
        Jx, Jy, Jz = VB_correction(particles, Jx, Jy, Jz)
        # calculate the corrections for charge conservation using villasenor buneamn 1991
        if verbose: print(f"Calculating Current Density, Max Value: {jnp.max(jnp.sqrt(Jx**2 + Jy**2 + Jz**2))}")
        Ex, Ey, Ez = update_E(E_grid, B_grid, (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), world, constants, curl_func)
        # update the electric field using the curl of the magnetic field
        Bx, By, Bz = update_B(E_grid, B_grid, (Bx, By, Bz), (Ex, Ey, Ez), world, constants, curl_func)
        # update the magnetic field using the curl of the electric field
        if solver == 'spectral':
            check_nyquist_criterion(Ex, Ey, Ez, Bx, By, Bz, world)
            # check if the spectral solver can resolve the highest frequencies in the fields


    return particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho