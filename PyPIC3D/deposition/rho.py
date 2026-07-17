from functools import partial

import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.filters import digital_filter


def _species_scalar_to_slots(tiled_particles, species_value):
    return jnp.broadcast_to(
        species_value.reshape((1, 1, 1, species_value.shape[0], 1)),
        tiled_particles.active.shape,
    )


def _species_slot_counts(tiled_particles):
    occupied = tiled_particles.active | jnp.any(tiled_particles.x != 0.0, axis=-1) | jnp.any(tiled_particles.u != 0.0, axis=-1)
    species_counts = jnp.sum(occupied.astype(tiled_particles.x.dtype), axis=(0, 1, 2, 4))
    return jnp.where(species_counts > 0, species_counts, 1.0)


def _diagnostic_mass_position(tiled_particles, world):
    """
    Mirror the ordinary scalar-diagnostic particle position for tiled particles.
    """

    x = tiled_particles.x[..., 0] - tiled_particles.u[..., 0] * world["dt"] / 2
    y = tiled_particles.x[..., 1] - tiled_particles.u[..., 1] * world["dt"] / 2
    z = tiled_particles.x[..., 2] - tiled_particles.u[..., 2] * world["dt"] / 2

    x = jax.lax.cond(
        world["boundary_conditions"]["x"] == BC_PERIODIC,
        lambda value: jnp.where(
            value > world["x_wind"] / 2,
            value - world["x_wind"],
            jnp.where(value < -world["x_wind"] / 2, value + world["x_wind"], value),
        ),
        lambda value: value,
        x,
    )
    y = jax.lax.cond(
        world["boundary_conditions"]["y"] == BC_PERIODIC,
        lambda value: jnp.where(
            value > world["y_wind"] / 2,
            value - world["y_wind"],
            jnp.where(value < -world["y_wind"] / 2, value + world["y_wind"], value),
        ),
        lambda value: value,
        y,
    )
    z = jax.lax.cond(
        world["boundary_conditions"]["z"] == BC_PERIODIC,
        lambda value: jnp.where(
            value > world["z_wind"] / 2,
            value - world["z_wind"],
            jnp.where(value < -world["z_wind"] / 2, value + world["z_wind"], value),
        ),
        lambda value: value,
        z,
    )

    return x, y, z


def _deposit_tiled_scalar_moment_to_tiles(
    tiled_particles,
    rho_tiles,
    world,
    grid,
    scalar_weight,
    tile_shape,
    g,
    position=None,
    divide_by_volume=True,
):
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(g)
    Nx = int(grid[0].shape[0]) - 2
    Ny = int(grid[1].shape[0]) - 2
    Nz = int(grid[2].shape[0]) - 2

    if position is None:
        x = tiled_particles.x[..., 0].reshape(-1)
        y = tiled_particles.x[..., 1].reshape(-1)
        z = tiled_particles.x[..., 2].reshape(-1)
    else:
        x = position[0].reshape(-1)
        y = position[1].reshape(-1)
        z = position[2].reshape(-1)
    active = tiled_particles.active.reshape(-1).astype(x.dtype)
    scalar_density = scalar_weight.reshape(-1)
    if divide_by_volume:
        scalar_density = scalar_density / (dx * dy * dz)

    x, _, deltax, xpts = prepare_particle_axis_stencil(
        x,
        grid[0],
        Nx + 2,
        shape_factor,
        bc_x,
        wind=world["x_wind"],
        ghost_cells=True,
    )
    y, _, deltay, ypts = prepare_particle_axis_stencil(
        y,
        grid[1],
        Ny + 2,
        shape_factor,
        bc_y,
        wind=world["y_wind"],
        ghost_cells=True,
    )
    z, _, deltaz, zpts = prepare_particle_axis_stencil(
        z,
        grid[2],
        Nz + 2,
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

    xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx + 2, ghost_cells=True)
    ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny + 2, ghost_cells=True)
    zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz + 2, ghost_cells=True)

    rho_tiles = jnp.zeros_like(rho_tiles)

    def folded_owner(points, n_cells, cells_per_tile, bc):
        folded = jax.lax.cond(
            bc == BC_PERIODIC,
            lambda p: jnp.where(p == 0, n_cells, jnp.where(p == n_cells + 1, 1, p)),
            lambda p: jnp.where(p == 0, 1, jnp.where(p == n_cells + 1, n_cells, p)),
            points,
        )
        sign = jax.lax.cond(
            bc == BC_PERIODIC,
            lambda p: jnp.ones_like(p, dtype=x.dtype),
            lambda p: jnp.where((p == 0) | (p == n_cells + 1), -1.0, 1.0).astype(x.dtype),
            points,
        )
        valid = (points >= 0) & (points <= n_cells + 1)
        folded = jnp.clip(folded, 1, n_cells)
        tile = (folded - 1) // cells_per_tile
        local = g + (folded - 1) - tile * cells_per_tile
        return tile, local, sign, valid

    for i in range(xpts.shape[0]):
        tx, lx, sx, valid_x = folded_owner(xpts[i], Nx, tile_nx, bc_x)
        for j in range(ypts.shape[0]):
            ty, ly, sy, valid_y = folded_owner(ypts[j], Ny, tile_ny, bc_y)
            for k in range(zpts.shape[0]):
                tz, lz, sz, valid_z = folded_owner(zpts[k], Nz, tile_nz, bc_z)
                valid = valid_x & valid_y & valid_z
                value = (
                    valid.astype(x.dtype)
                    * active
                    * sx
                    * sy
                    * sz
                    * scalar_density
                    * x_weights[i]
                    * y_weights[j]
                    * z_weights[k]
                )
                rho_tiles = rho_tiles.at[tx, ty, tz, lx, ly, lz].add(value, mode="drop")

    rho_tiles = update_tiled_ghost_cells(rho_tiles, world, g)

    return rho_tiles


def _filter_tiled_scalar_with_halos(field_tiles, alpha, world, g):
    g = int(g)
    field_tiles = update_tiled_ghost_cells(field_tiles, world, g)
    return digital_filter(field_tiles, alpha, num_guard_cells=g)


def compute_rho(
    particles,
    species_config,
    rho_tiles,
    constants,
    world,
):
    """Compute charge density into tile-major vertex-grid scalar arrays."""

    if not isinstance(particles, TiledParticles):
        raise ValueError("Public compute_rho requires TiledParticles.")

    tile_shape = tuple(int(width) for width in world["tile_shape"])
    g = int(world["guard_cells"])
    grid = world["grids"]["vertex"]

    scalar_weight = _species_scalar_to_slots(
        particles,
        species_config.charge * species_config.weight,
    )
    rho_tiles = _deposit_tiled_scalar_moment_to_tiles(
        particles,
        rho_tiles,
        world,
        grid,
        scalar_weight,
        tile_shape,
        g,
    )

    alpha = constants["alpha"]
    rho_tiles = _filter_tiled_scalar_with_halos(rho_tiles, alpha, world, g)
    rho_tiles = update_tiled_ghost_cells(rho_tiles, world, g)

    return rho_tiles


def compute_tiled_mass_density_from_tiled_particles(tiled_particles, species_config, rho_tiles, world, grid=None, tile_shape=None, g=2):
    """Compute mass density into tile-major vertex-grid scalar arrays."""
    if grid is None:
        grid = world["grids"]["vertex"]

    scalar_weight = _species_scalar_to_slots(
        tiled_particles,
        species_config.mass * species_config.weight,
    )
    position = _diagnostic_mass_position(tiled_particles, world)
    return _deposit_tiled_scalar_moment_to_tiles(
        tiled_particles,
        rho_tiles,
        world,
        grid,
        scalar_weight,
        tile_shape,
        g,
        position=position,
    )


def compute_tiled_velocity_field_from_tiled_particles(
    tiled_particles,
    species_config,
    field_tiles,
    direction,
    world,
    grid=None,
    tile_shape=None,
    g=2,
):
    """Compute the legacy scalar velocity diagnostic into tile-major arrays."""
    if grid is None:
        grid = world["grids"]["vertex"]

    species_counts = _species_slot_counts(tiled_particles)
    species_counts = species_counts.reshape((1, 1, 1, species_counts.shape[0], 1))
    scalar_weight = tiled_particles.u[..., direction] / species_counts
    position = _diagnostic_mass_position(tiled_particles, world)

    return _deposit_tiled_scalar_moment_to_tiles(
        tiled_particles,
        field_tiles,
        world,
        grid,
        scalar_weight,
        tile_shape,
        g,
        position=position,
        divide_by_volume=False,
    )


def compute_tiled_pressure_field_from_tiled_particles(
    tiled_particles,
    species_config,
    field_tiles,
    velocity_field_tiles,
    direction,
    world,
    grid=None,
    tile_shape=None,
    g=2,
):
    """Compute the legacy scalar pressure diagnostic into tile-major arrays."""
    if grid is None:
        grid = world["grids"]["vertex"]
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(g)
    Nx = int(grid[0].shape[0]) - 2
    Ny = int(grid[1].shape[0]) - 2
    Nz = int(grid[2].shape[0]) - 2

    position = _diagnostic_mass_position(tiled_particles, world)
    x = position[0].reshape(-1)
    y = position[1].reshape(-1)
    z = position[2].reshape(-1)
    v = tiled_particles.u[..., direction].reshape(-1)
    active = tiled_particles.active.reshape(-1).astype(x.dtype)

    x, _, deltax, xpts = prepare_particle_axis_stencil(
        x,
        grid[0],
        Nx + 2,
        shape_factor,
        bc_x,
        wind=world["x_wind"],
        ghost_cells=True,
    )
    y, _, deltay, ypts = prepare_particle_axis_stencil(
        y,
        grid[1],
        Ny + 2,
        shape_factor,
        bc_y,
        wind=world["y_wind"],
        ghost_cells=True,
    )
    z, _, deltaz, zpts = prepare_particle_axis_stencil(
        z,
        grid[2],
        Nz + 2,
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

    xpts, x_weights = collapse_axis_stencil(xpts, x_weights, Nx + 2, ghost_cells=True)
    ypts, y_weights = collapse_axis_stencil(ypts, y_weights, Ny + 2, ghost_cells=True)
    zpts, z_weights = collapse_axis_stencil(zpts, z_weights, Nz + 2, ghost_cells=True)

    field_tiles = jnp.zeros_like(field_tiles)

    def folded_owner(points, n_cells, cells_per_tile, bc):
        folded = jax.lax.cond(
            bc == BC_PERIODIC,
            lambda p: jnp.where(p == 0, n_cells, jnp.where(p == n_cells + 1, 1, p)),
            lambda p: jnp.where(p == 0, 1, jnp.where(p == n_cells + 1, n_cells, p)),
            points,
        )
        sign = jax.lax.cond(
            bc == BC_PERIODIC,
            lambda p: jnp.ones_like(p, dtype=x.dtype),
            lambda p: jnp.where((p == 0) | (p == n_cells + 1), -1.0, 1.0).astype(x.dtype),
            points,
        )
        valid = (points >= 0) & (points <= n_cells + 1)
        folded = jnp.clip(folded, 1, n_cells)
        tile = (folded - 1) // cells_per_tile
        local = g + (folded - 1) - tile * cells_per_tile
        return tile, local, sign, valid

    for i in range(xpts.shape[0]):
        tx, lx, sx, valid_x = folded_owner(xpts[i], Nx, tile_nx, bc_x)
        for j in range(ypts.shape[0]):
            ty, ly, sy, valid_y = folded_owner(ypts[j], Ny, tile_ny, bc_y)
            for k in range(zpts.shape[0]):
                tz, lz, sz, valid_z = folded_owner(zpts[k], Nz, tile_nz, bc_z)
                valid = valid_x & valid_y & valid_z
                velocity_value = velocity_field_tiles[tx, ty, tz, lx, ly, lz]
                vbar = v - velocity_value
                value = (
                    valid.astype(x.dtype)
                    * active
                    * sx
                    * sy
                    * sz
                    * vbar**2
                    * x_weights[i]
                    * y_weights[j]
                    * z_weights[k]
                )
                field_tiles = field_tiles.at[tx, ty, tz, lx, ly, lz].add(value, mode="drop")

    field_tiles = update_tiled_ghost_cells(field_tiles, world, g)
    return field_tiles
