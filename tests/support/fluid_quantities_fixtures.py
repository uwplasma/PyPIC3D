import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_PARTICLE,
    fold_tiled_ghost_cells,
    particle_vector_reflecting_parity,
)
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.fluid_quantities import (
    compute_velocity_field,
    fluid_velocity,
)
from PyPIC3D.diagnostics.output_adapters import (
    assemble_tiled_scalar_field,
    build_field_output_map,
)
from tests.kernel_fixtures import build_tiled_particles, field_tiles_from_global, kernel_parameters, particle_species


class TiledFluidQuantitiesFixtures:
    def _build_parameters(self, shape_factor=2, tile_shape=None):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if tile_shape is None:
            tile_shape = (8, 6, 4)

        return kernel_parameters(
            Nx=8,
            Ny=6,
            Nz=4,
            x_wind=x_wind,
            y_wind=y_wind,
            z_wind=z_wind,
            dx=x_wind / 8,
            dy=y_wind / 6,
            dz=z_wind / 4,
            dt=0.08,
            shape_factor=shape_factor,
            guard_cells=2,
            tile_shape=tile_shape,
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

    def _empty_scalar(self, dynamic_parameters):
        return jnp.zeros(
            (
                int(dynamic_parameters.Nx) + 2,
                int(dynamic_parameters.Ny) + 2,
                int(dynamic_parameters.Nz) + 2,
            )
        )

    def _scalar_tiles(self, static_parameters, dynamic_parameters):
        return field_tiles_from_global(
            self._empty_scalar(dynamic_parameters),
            static_parameters,
            dynamic_parameters,
            num_guard_cells=int(static_parameters.guard_cells),
        )

    def _assemble_scalar(self, field_tiles, static_parameters):
        return assemble_tiled_scalar_field(
            field_tiles,
            static_parameters,
            static_parameters.tile_shape,
            num_guard_cells=int(static_parameters.guard_cells),
        )

    def _weighted_average_particles(self):
        electrons = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=1.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([2.0]),
            v2=jnp.array([-4.0]),
            v3=jnp.array([0.5]),
        )
        ions = particle_species(
            name="ions",
            charge=1.0,
            mass=4.0,
            weight=3.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([10.0]),
            v2=jnp.array([4.0]),
            v3=jnp.array([-1.5]),
        )
        return [electrons, ions]

    def _spread_particles(self):
        electrons = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
        )
        ions = particle_species(
            name="ions",
            charge=2.0,
            mass=5.0,
            weight=0.25,
            x1=jnp.array([-1.25, -0.20, 0.75]),
            x2=jnp.array([1.15, -0.75, 0.45]),
            x3=jnp.array([0.35, -0.45, 0.85]),
            v1=jnp.array([-0.1, 0.2, -0.25]),
            v2=jnp.array([0.3, -0.05, 0.15]),
            v3=jnp.array([0.1, 0.05, -0.2]),
            active_mask=jnp.array([True, False, True]),
        )
        return [electrons, ions]
