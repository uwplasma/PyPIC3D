# Christopher Woolford Dec 5, 2024
# This contains the public evolution-loop names for the 3D PIC code.

from functools import partial

from jax import jit

from PyPIC3D.electrodynamic_tiled import time_loop_electrodynamic_tiled as time_loop_electrodynamic
from PyPIC3D.electrostatic_tiled import time_loop_electrostatic_tiled as time_loop_electrostatic
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.solvers.electrostatic_yee import calculate_electrostatic_fields
from PyPIC3D.solvers.first_order_yee import update_B, update_E
from PyPIC3D.utils import add_external_fields


__all__ = ["time_loop_electrodynamic", "time_loop_electrostatic"]


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def _time_loop_electrostatic_global_reference(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Old global electrostatic loop retained as a reference path for tests.
    """

    E, B, J, rho, phi, external_fields = fields
    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]
    push_E, push_B = add_external_fields(E, B, external_fields)

    for i in range(len(particles)):
        particles[i] = particle_push(
            particles[i],
            push_E,
            push_B,
            center_grid,
            vertex_grid,
            world,
            constants,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )
        particles[i].update_position()
        particles[i].boundary_conditions(world)

    E, phi, rho = calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, "periodic")

    fields = (E, B, J, rho, phi, external_fields)
    return particles, fields


@partial(jit, static_argnames=("curl_func", "J_func", "solver", "relativistic", "particle_pusher"))
def _time_loop_electrodynamic_global_reference(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Old global electrodynamic loop retained as a reference path for tests.
    """

    E, B, J, rho, phi, external_fields, pml_state = fields
    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]
    push_E, push_B = add_external_fields(E, B, external_fields)

    for i in range(len(particles)):
        particles[i] = particle_push(
            particles[i],
            push_E,
            push_B,
            center_grid,
            vertex_grid,
            world,
            constants,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )
        particles[i].update_position()

    J = J_func(particles, J, constants, world)
    E, pml_state = update_E(E, B, J, world, constants, curl_func, pml_state)
    B, pml_state = update_B(E, B, world, constants, curl_func, pml_state)

    for i in range(len(particles)):
        particles[i].boundary_conditions(world)

    fields = (E, B, J, rho, phi, external_fields, pml_state)
    return particles, fields
