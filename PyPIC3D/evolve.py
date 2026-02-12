# Christopher Woolford Dec 5, 2024
# This contains the evolution loop for the 3D PIC code that calculates the electric and magnetic fields and updates the particles.

import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from PyPIC3D.boris import (
    particle_push
)

from PyPIC3D.solvers.first_order_yee import (
    update_E, update_B, calculateE
)

from PyPIC3D.solvers.vector_potential import (
    E_from_A, B_from_A, update_vector_potential
)

@partial(jit, static_argnames=("curl_func", "J_func", "solver", "x_bc", "y_bc", "z_bc", "relativistic"))
def time_loop_electrostatic(particles, fields, vertex_grid, center_grid, world, constants, curl_func, J_func, solver, x_bc, y_bc, z_bc, relativistic=True):
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
        vertex_grid (ndarray): Grid representing the electric field.
        center_grid (ndarray): Grid representing the magnetic field.
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

        particles[i] = particle_push(particles[i], E, B, vertex_grid, center_grid, world['dt'], constants, relativistic=relativistic)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ############### SOLVE E FIELD ############################################################################################
    E, phi, rho = calculateE(world, particles, constants, rho, phi, solver, bc)
    # calculate the electric field using the Poisson equation

    fields = (E, B, J, rho, phi)
    # pack the fields into a tuple

    for i in range(len(particles)):
        particles[i].boundary_conditions()
        # apply boundary conditions to the particles

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "x_bc", "y_bc", "z_bc", "relativistic"))
def time_loop_electrodynamic(particles, fields, vertex_grid, center_grid, world, constants, curl_func, J_func, solver, x_bc, y_bc, z_bc, relativistic=True):
    """
    Advance an electrodynamic Particle-In-Cell (PIC) system by one time step.
    This routine performs, in order:
    1) Particle push using the Boris algorithm (optionally relativistic),
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
        Tuple of field arrays/objects in the form `(E, B, J, rho, phi)`.
    vertex_grid : object
        Grid definition for vertex-centered quantities; passed through to the
        particle pusher.
    center_grid : object
        Grid definition for cell/centered quantities; used by the particle pusher
        and current deposition.
    world : dict
        Simulation configuration. Must contain `world['dt']` (time step).
    constants : object or dict
        Physical/normalization constants required by push, deposition, and updates.
    curl_func : callable
        Function/operator used by field update routines to compute curls.
    J_func : callable
        Current deposition function with signature like
        `J_func(particles, J, constants, world, center_grid) -> J`.
    solver : object
        Reserved/placeholder for a solver interface (currently unused).
    x_bc, y_bc, z_bc : object
        Boundary condition specifications for x, y, and z directions; forwarded to
        `update_E` and `update_B`.
    relativistic : bool, optional
        If True, perform a relativistic Boris push; otherwise use the
        non-relativistic variant.
    Returns
    -------
    particles : Sequence[object]
        Updated particle collection after push, position update, and boundary
        conditions.
    fields : tuple
        Updated fields tuple `(E, B, J, rho, phi)` with new E, B, and J. `rho` and
        `phi` are passed through unchanged.
    Notes
    -----
    - The update order is: deposit J -> update E -> update B.
    - `solver` is accepted for API compatibility but is not used in this function.
    """


    E, B, J, rho, phi = fields
    # unpack the fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, center_grid, vertex_grid, world['dt'], constants, relativistic=relativistic)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    J = J_func(particles, J, constants, world, center_grid)
    # calculate the current density based on the selected method
    E = update_E(E, B, J, world, constants, curl_func, x_bc, y_bc, z_bc)
    # update the electric field using the curl of the magnetic field
    B = update_B(E, B, world, constants, curl_func, x_bc, y_bc, z_bc)
    # update the magnetic field using the curl of the electric field

    for i in range(len(particles)):
        particles[i].boundary_conditions()
        # apply boundary conditions to the particles

    fields = (E, B, J, rho, phi)
    # pack the fields into a tuple
    

    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "x_bc", "y_bc", "z_bc", "relativistic"))
def time_loop_vector_potential(particles, fields, vertex_grid, center_grid, world, constants, curl_func, J_func, solver, x_bc, y_bc, z_bc, relativistic=True):
    """
    Advance a PIC (Particle-In-Cell) simulation by one time step using a
    vector-potential formulation for the electromagnetic fields.

    This routine:
    1) Pushes particle velocities/positions using the current E and B fields.
    2) Updates the vector potential A via the current density J.
    3) Recomputes E and B from the updated vector potential.
    4) Recomputes J from particle motion and applies particle boundary conditions.

    Parameters
    ----------
    particles : Sequence
        Iterable of particle objects. Each particle is expected to be compatible
        with `particle_push(...)`, provide `update_position()`, and
        `boundary_conditions()` methods.
    fields : tuple
        Field tuple in the order `(E, B, J, rho, phi, A2, A1, A0)`, where `A2`
        denotes the newest vector potential, `A1` the previous, and `A0` the
        older one (used for time differencing).
    vertex_grid : Any
        Grid/mesh information at vertices (used for field computations and/or
        interpolation).
    center_grid : Any
        Grid/mesh information at cell centers (used for particle push and current
        deposition).
    world : dict
        Simulation parameters. Must at least contain `'dt'` (time step).
    constants : Any
        Physical/constants container passed through to lower-level routines.
    curl_func : Callable
        Reserved/unused in the current implementation (kept for API compatibility).
    J_func : Callable
        Function to compute/update current density, called as:
        `J_func(particles, J, constants, world, center_grid)`.
    solver : Any
        Reserved/unused in the current implementation (kept for API compatibility).
    x_bc, y_bc, z_bc : Any
        Reserved/unused in the current implementation (boundary conditions are
        currently applied via each particle's `boundary_conditions()` method).
    relativistic : bool, optional
        If True, the particle pusher is invoked in relativistic mode.

    Returns
    -------
    particles : Sequence
        Updated particle collection after push, position update, and boundary
        conditions.
    fields : tuple
        Updated field tuple `(E, B, J, rho, phi, A2, A1, A0)` after advancing A
        and recomputing E, B, and J.

    Notes
    -----
    - The vector potential history is shifted each step (`A0 <- A1`, `A1 <- A2`),
      then `A2` is computed from the current density via `update_vector_potential`.
    - Electric and magnetic fields are recomputed using `E_from_A` and `B_from_A`
      with `interpolation_order=2`.
    - Several parameters (`curl_func`, `solver`, `x_bc`, `y_bc`, `z_bc`) are
      currently not used inside this function but may be part of a broader API.
    """

    E, B, J, rho, phi, A2, A1, A0 = fields
    # unpack the fields

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):

        particles[i] = particle_push(particles[i], E, B, center_grid, vertex_grid, world['dt'], constants, relativistic=relativistic)
        # use boris push for particle velocities

        particles[i].update_position()
        # update the particle positions

    ################ FIELD UPDATE ################################################################################################
    A0 = A1
    A1 = A2
    # update the vector potential for the next iteration
    A2 = update_vector_potential(J, world, constants, A1, A0)
    # update the vector potential based on the current density J

    E = E_from_A(A2, A1, A0, world, center_grid, vertex_grid, interpolation_order=2)
    # calculate the electric field from the vector potential using centered finite difference
    B = B_from_A(A1, world, center_grid, vertex_grid, interpolation_order=2)
    # calculate the magnetic field from the vector potential using centered finite difference
    J = J_func(particles, J, constants, world, center_grid)
    # calculate the current density using the selected method

    for i in range(len(particles)):
        particles[i].boundary_conditions()
        # apply boundary conditions to the particles

    fields = (E, B, J, rho, phi, A2, A1, A0)
    # pack the fields into a tuple


    return particles, fields