# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.

import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from PyPIC3D.pusher.particle_push import (
    particle_push
)

from PyPIC3D.solvers.first_order_yee import (
    update_E, update_B
)

from PyPIC3D.solvers.electrostatic_yee import (
    calculate_electrostatic_fields
)

from PyPIC3D.utils import add_external_fields

@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def time_loop_electrostatic(particles, fields, world, constants, curl_func, J_func, solver, relativistic=True, particle_pusher="boris"):
    """
    Advances the simulation by one time step for an electrostatic Particle-In-Cell (PIC) loop.

    This function performs the following steps:
    1. Pushes all particles using the current electric and magnetic fields with the selected particle pusher.
    2. Updates particle positions.
    3. Solves for the new electric field using the Poisson equation, updating the field and potential.
    4. Packs the updated fields and returns the new particle and field states.

    Args:
        particles (list): List of particle objects to be advanced.
        fields (tuple): Tuple containing the field arrays (E, B, J, rho, phi, external_fields).
        world (dict): Dictionary containing simulation parameters (e.g., time step 'dt').
        constants (dict): Dictionary of physical constants used in the simulation.
        curl_func (callable): Function to compute the curl of a field (not used in this function).
        J_func (callable): Function to compute the current density (not used in this function).
        solver (callable): Function or object to solve the Poisson equation for the electric field.

    Returns:
        tuple: Updated particles list and fields tuple (E, B, J, rho, phi, external_fields).
    """

    E, B, J, rho, phi, external_fields = fields
    # unpack the fields
    center_grid = world['grids']['center']
    vertex_grid = world['grids']['vertex']
    push_E, push_B = add_external_fields(E, B, external_fields)
    # particles see evolved fields plus external-only fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], push_E, push_B, center_grid, vertex_grid, world, constants, relativistic=relativistic, particle_pusher=particle_pusher)
        # use the selected particle pusher for particle velocities

        particles[i].update_position()
        # update the particle positions

        particles[i].boundary_conditions(world)
        # electrostatic rho is deposited from the current in-domain particle positions

    ############### SOLVE E FIELD ############################################################################################
    E, phi, rho = calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, 'periodic')
    # calculate the electric field using the Poisson equation

    fields = (E, B, J, rho, phi, external_fields)
    # pack the fields into a tuple

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def time_loop_electrodynamic(particles, fields, world, constants, curl_func, J_func, solver, relativistic=True, particle_pusher="boris"):
    """
    Advance an electrodynamic Particle-In-Cell (PIC) system by one time step.
    This routine performs, in order:
    1) Particle push using the selected particle pusher,
    2) Particle position update,
    3) Current deposition onto the grid via the provided `J_func`,
    4) Field update (E then B) using curl operators and boundary conditions,
    5) Particle boundary condition enforcement.
    Parameters
    ----------
    particles : Sequence[object]
        Collection of particle objects. Each particle is expected to be compatible
        with `particle_push(...)`, provide an `update_position()` method, and a
        `boundary_conditions()` method.
    fields : tuple
        Tuple of field arrays/objects in the form
        `(E, B, J, rho, phi, external_fields, pml_state)`.
        `pml_state` is `None` when the ordinary Yee update is used.
    world : dict
        Simulation configuration. Must contain `world['dt']` (time step).
    constants : object or dict
        Physical/normalization constants required by push, deposition, and updates.
    curl_func : callable
        Function/operator used by field update routines to compute curls.
    J_func : callable
        Current deposition function with signature like
        `J_func(particles, J, constants, world) -> J`.
    solver : object
        Reserved/placeholder for a solver interface (currently unused).
    relativistic : bool, optional
        If True, perform a relativistic Boris push; otherwise use the
        non-relativistic variant.
    Returns
    -------
    particles : Sequence[object]
        Updated particle collection after push, position update, and boundary
        conditions.
    fields : tuple
        Updated fields tuple `(E, B, J, rho, phi, external_fields, pml_state)`
        with new E, B, and J. `rho`, `phi`, and `external_fields` are passed
        through unchanged.
    Notes
    -----
    - The update order is: deposit J -> update E -> update B.
    - `solver` is accepted for API compatibility but is not used in this function.
    """


    E, B, J, rho, phi, external_fields, pml_state = fields
    # unpack the fields
    center_grid = world['grids']['center']
    vertex_grid = world['grids']['vertex']
    push_E, push_B = add_external_fields(E, B, external_fields)
    # particles see evolved fields plus external-only fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], push_E, push_B, center_grid, vertex_grid, world, constants, relativistic=relativistic, particle_pusher=particle_pusher)
        # use the selected particle pusher for particle velocities

        particles[i].update_position()
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    J = J_func(particles, J, constants, world)
    # calculate the current density based on the selected method
    E, pml_state = update_E(E, B, J, world, constants, curl_func, pml_state)
    # update the electric field using the curl of the magnetic field
    B, pml_state = update_B(E, B, world, constants, curl_func, pml_state)
    # update the magnetic field using the curl of the electric field

    for i in range(len(particles)):
        particles[i].boundary_conditions(world)
        # apply boundary conditions to the particles

    fields = (E, B, J, rho, phi, external_fields, pml_state)
    # pack the fields into a tuple
    

    return particles, fields
