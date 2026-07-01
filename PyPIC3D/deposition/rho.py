import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells
from PyPIC3D.deposition.rho_tiled import compute_rho_from_tiled_particles, compute_tiled_rho_from_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.utils import digital_filter


def compute_rho(particles, rho, world, constants, species_config=None):
    """Compute charge density for flat or tile-major particle storage."""
    if isinstance(particles, TiledParticles):
        if species_config is None:
            species_config = world["species_config"]

        if getattr(rho, "ndim", 0) == 6:
            return compute_tiled_rho_from_tiled_particles(
                particles,
                species_config,
                rho,
                world,
                constants,
                tile_shape=tuple(int(width) for width in world["tile_shape"]),
                g=int(world["guard_cells"]),
            )

        return compute_rho_from_tiled_particles(particles, species_config, rho, world, constants)

    return _compute_rho_flat(particles, rho, world, constants)


@jit
def _compute_rho_flat(particles, rho, world, constants):
    """Compute the charge density (rho) on the vertex grid."""
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    grid = world["grids"]["vertex"]
    Nx, Ny, Nz = rho.shape
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    rho = jnp.zeros_like(rho)

    for species in particles:
        q = species.get_charge()
        dq = q / dx / dy / dz
        x, y, z = species.get_forward_position()
        active = species.get_active_mask().astype(x.dtype)

        x, _, deltax, xpts = prepare_particle_axis_stencil(
            x,
            grid[0],
            Nx,
            shape_factor,
            bc_x,
            wind=world["x_wind"],
            ghost_cells=True,
        )
        y, _, deltay, ypts = prepare_particle_axis_stencil(
            y,
            grid[1],
            Ny,
            shape_factor,
            bc_y,
            wind=world["y_wind"],
            ghost_cells=True,
        )
        z, _, deltaz, zpts = prepare_particle_axis_stencil(
            z,
            grid[2],
            Nz,
            shape_factor,
            bc_z,
            wind=world["z_wind"],
            ghost_cells=True,
        )

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights = jnp.asarray(x_weights)
        y_weights = jnp.asarray(y_weights)
        z_weights = jnp.asarray(z_weights)

        xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx, ghost_cells=True)
        ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny, ghost_cells=True)
        zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz, ghost_cells=True)

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    rho = rho.at[xpts[i], ypts[j], zpts[k]].add(
                        active * dq * x_weights[i] * y_weights[j] * z_weights[k], mode="drop"
                    )

    rho = fold_ghost_cells(rho, bc_x, bc_y, bc_z)
    # fold ghost cell deposits back into interior
    rho = update_ghost_cells(rho, bc_x, bc_y, bc_z)
    # refresh ghost cells before any stencil-based post-processing

    alpha = constants["alpha"]
    rho = digital_filter(rho, alpha)
    rho = update_ghost_cells(rho, bc_x, bc_y, bc_z)

    return rho
