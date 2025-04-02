# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.

#from memory_profiler import profile
import jax.numpy as jnp
import jax

from PyPIC3D.fields import (
    calculateE, update_B, update_E
)

from PyPIC3D.J import (
    VB_correction
)

from PyPIC3D.boris import (
    particle_push
)


def time_loop_electrostatic(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, pecs, lasers, surfaces, curl_func, M, solver, bc, verbose, GPUs):
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

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    # unpack the electric and magnetic fields

    #if_verbose_print(verbose, f"Calculating Electric Field, Max Value: {jnp.max(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
    #if_verbose_print(verbose, f"Calculating Magnetic Field, Max Value: {jnp.max(jnp.sqrt(Bx**2 + By**2 + Bz**2))}")

    # for pec in pecs:
    #     Ex, Ey, Ez = pec.apply_pec(Ex, Ey, Ez)
    #     # apply any PEC boundary conditions to the electric field

    # for laser in lasers:
    #     Ex, Ey, Ez = laser.inject_incident_fields(Ex, Ey, Ez, t)
    #     # inject any laser pulses into the electric field

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):
        ######################### Material Surfaces ############################################
        barrier_x = jnp.zeros_like(Ex)
        barrier_y = jnp.zeros_like(Ey)
        barrier_z = jnp.zeros_like(Ez)
        # for surface in surfaces:
        #     barrier_x += surface.get_barrier_x()
        #     barrier_y += surface.get_barrier_y()
        #     barrier_z += surface.get_barrier_z()
            # get the boundaries of the material surfaces

        total_Ex = Ex + barrier_x
        total_Ey = Ey + barrier_y
        total_Ez = Ez + barrier_z
        # add the boundaries as background fields
        ########################################################################################

        #if_verbose_print(verbose, f'Updating {particles[i].get_name()}')

        particles[i] = particle_push(particles[i], total_Ex, total_Ey, total_Ez, Bx, By, Bz, E_grid, B_grid, world['dt'], GPUs)
        # use boris push for particle velocities

        #if_verbose_print(verbose, f"Calculating {particles[i].get_name()} Velocities, Mean Value: {jnp.mean(jnp.abs(particles[i].get_velocity()[0]))}")

        x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
        particles[i].update_position(world['dt'], x_wind, y_wind, z_wind)
        # update the particle positions

        #if_verbose_print(verbose, f"Calculating {particles[i].get_name()} Positions, Mean Value: {jnp.mean(jnp.abs(particles[i].get_position()[0]))}")

    ############### SOLVE E FIELD ############################################################################################
    #jax.debug.print("Max value of rho before calc E: {}", jnp.max(rho))

    E, phi, rho = calculateE(world, particles, constants, rho, phi, M, solver, bc)
    # calculate the electric field using the Poisson equation
    
    #jax.debug.print("Max value of rho after calc E: {}", jnp.max(rho))

    return particles, E, B, J, phi, rho


def time_loop_electrodynamic(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, pecs, lasers, surfaces, curl_func, M, solver, bc, verbose, GPUs):
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

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    #if_verbose_print(verbose, f"Calculating Electric Field, Max Value: {jnp.max(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
    #if_verbose_print(verbose, f"Calculating Magnetic Field, Max Value: {jnp.max(jnp.sqrt(Bx**2 + By**2 + Bz**2))}")
    # print the maximum value of the electric and magnetic fields

    # for pec in pecs:
    #     Ex, Ey, Ez = pec.apply_pec(Ex, Ey, Ez)
    #     # apply any PEC boundary conditions to the electric field

    # for laser in lasers:
    #     Ex, Ey, Ez, Bx, By, Bz = laser.inject_incident_fields(Ex, Ey, Ez, Bx, By, Bz, t)
    #     #inject any laser pulses into the electric and magnetic fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):
        ######################### Material Surfaces ############################################
        barrier_x = jnp.zeros_like(Ex)
        barrier_y = jnp.zeros_like(Ey)
        barrier_z = jnp.zeros_like(Ez)
        # for surface in surfaces:
        #     barrier_x += surface.get_barrier_x()
        #     barrier_y += surface.get_barrier_y()
        #     barrier_z += surface.get_barrier_z()
            # get the boundaries of the material surfaces

        total_Ex = Ex + barrier_x
        total_Ey = Ey + barrier_y
        total_Ez = Ez + barrier_z
        # add the boundaries as background fields
        ########################################################################################

        #if_verbose_print(verbose, f'Updating {particles[i].get_name()}')

        particles[i] = particle_push(particles[i], total_Ex, total_Ey, total_Ez, Bx, By, Bz, E_grid, B_grid, world['dt'], GPUs)
        # use boris push for particle velocities

        #if_verbose_print(verbose, f"Calculating {particles[i].get_name()} Velocities, Mean Value: {jnp.mean(jnp.abs(particles[i].get_velocity()[0]))}")

        x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
        particles[i].update_position(world['dt'], x_wind, y_wind, z_wind)
        # update the particle positions

        #if_verbose_print(verbose, f"Calculating {particles[i].get_name()} Positions, Mean Value: {jnp.mean(jnp.abs(particles[i].get_position()[0]))}")

    ################ FIELD UPDATE ################################################################################################
    Jx, Jy, Jz = VB_correction(particles, Jx, Jy, Jz, constants)
    # calculate the corrections for charge conservation using villasenor buneamn 1991

    #if_verbose_print(verbose, f"Calculating Current Density, Max Value: {jnp.max(jnp.sqrt(Jx**2 + Jy**2 + Jz**2))}")

    E = update_E(E_grid, B_grid, (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), world, constants, curl_func)
    # update the electric field using the curl of the magnetic field
    B = update_B(E_grid, B_grid, (Ex, Ey, Ez), (Bx, By, Bz), world, constants, curl_func)
    # update the magnetic field using the curl of the electric field

    return particles, E, B, (Jx, Jy, Jz), phi, rho