import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_PERIODIC,
)
from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh
from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import kernel_parameters_from_values


def add_single_tile_grids_to_parameters(parameter_set):
    tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
    g = int(parameter_set["guard_cells"])
    parameter_set["tile_shape"] = tile_shape
    parameter_set["field_mesh"] = make_field_mesh((1, 1, 1))
    static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g,
                                        field_mesh=parameter_set["field_mesh"])
    dynamic_parameters = SimpleNamespace(
        dx=parameter_set["dx"],
        dy=parameter_set["dy"],
        dz=parameter_set["dz"],
        grids=SimpleNamespace(vertex=parameter_set["grids"]["vertex"], center=parameter_set["grids"]["center"]),
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
    parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
    parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
    # make tiled versions of the grids for the parallelized particle pushers to use


def one_slot_tiled_particles(
    x,
    u,
    charge=1.0,
    mass=1.0,
    weight=1.0,
    active=True,
    update_x=(True, True, True),
):
    particles = TiledParticles(
        x=jnp.asarray(x, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        u=jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.asarray([[[[[active]]]]], dtype=bool),
    )
    # create a single tile of particles with one particle in it, and a single species config
    species_config = SpeciesConfig(
        charge=jnp.asarray([charge], dtype=float),
        mass=jnp.asarray([mass], dtype=float),
        weight=jnp.asarray([weight], dtype=float),
        update_x=jnp.asarray([update_x], dtype=bool),
    )
    return particles, species_config


        # ensure that the particle has become inactive after crossing the absorbing boundary
