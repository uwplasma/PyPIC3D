import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.solvers.yee_tiled import (
    fold_tiled_ghost_cells,
    tile_scalar_field,
    update_tiled_ghost_cells,
)
from PyPIC3D.utils import digital_filter


def _deposit_tiled_scalar_moment(
    tiled_particles,
    rho,
    world,
    grid,
    scalar_weight,
    position=None,
):
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    Nx, Ny, Nz = rho.shape
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    if position is None:
        x = tiled_particles.x[..., 0].reshape(-1)
        y = tiled_particles.x[..., 1].reshape(-1)
        z = tiled_particles.x[..., 2].reshape(-1)
    else:
        x, y, z = position
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
    active = tiled_particles.active.reshape(-1).astype(x.dtype)
    scalar_density = scalar_weight.reshape(-1) / (dx * dy * dz)

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

    rho = jnp.zeros_like(rho)

    for i in range(xpts.shape[0]):
        for j in range(ypts.shape[0]):
            for k in range(zpts.shape[0]):
                rho = rho.at[xpts[i], ypts[j], zpts[k]].add(
                    active * scalar_density * x_weights[i] * y_weights[j] * z_weights[k],
                    mode="drop",
                )

    rho = fold_ghost_cells(rho, bc_x, bc_y, bc_z)
    rho = update_ghost_cells(rho, bc_x, bc_y, bc_z)

    return rho


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


def _tile_axis(axis, tile_index, cells_per_tile):
    start = tile_index * cells_per_tile
    return jax.lax.dynamic_slice(axis, (start,), (cells_per_tile + 2,))


def _deposit_tiled_scalar_moment_to_tiles(
    tiled_particles,
    rho_tiles,
    world,
    grid,
    scalar_weight,
):
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    tile_nx = rho_tiles.shape[3] - 2
    tile_ny = rho_tiles.shape[4] - 2
    tile_nz = rho_tiles.shape[5] - 2
    ntx, nty, ntz = rho_tiles.shape[:3]

    def deposit_one_tile(tx, ty, tz, x_tile, u_tile, active_tile, scalar_weight_tile):
        local_grid = (
            _tile_axis(grid[0], tx, tile_nx),
            _tile_axis(grid[1], ty, tile_ny),
            _tile_axis(grid[2], tz, tile_nz),
        )

        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        active = active_tile.reshape(-1).astype(x.dtype)
        scalar_density = scalar_weight_tile.reshape(-1) / (dx * dy * dz)

        x, _, deltax, xpts = prepare_particle_axis_stencil(
            x,
            local_grid[0],
            tile_nx + 2,
            shape_factor,
            bc_x,
            wind=world["x_wind"],
            ghost_cells=True,
        )
        y, _, deltay, ypts = prepare_particle_axis_stencil(
            y,
            local_grid[1],
            tile_ny + 2,
            shape_factor,
            bc_y,
            wind=world["y_wind"],
            ghost_cells=True,
        )
        z, _, deltaz, zpts = prepare_particle_axis_stencil(
            z,
            local_grid[2],
            tile_nz + 2,
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

        xpts, x_weights = collapse_axis_stencil(xpts, x_weights, tile_nx + 2, ghost_cells=True)
        ypts, y_weights = collapse_axis_stencil(ypts, y_weights, tile_ny + 2, ghost_cells=True)
        zpts, z_weights = collapse_axis_stencil(zpts, z_weights, tile_nz + 2, ghost_cells=True)

        rho_tile = jnp.zeros((tile_nx + 2, tile_ny + 2, tile_nz + 2), dtype=rho_tiles.dtype)

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    rho_tile = rho_tile.at[xpts[i], ypts[j], zpts[k]].add(
                        active * scalar_density * x_weights[i] * y_weights[j] * z_weights[k],
                        mode="drop",
                    )

        return rho_tile

    tx = jnp.arange(ntx)
    ty = jnp.arange(nty)
    tz = jnp.arange(ntz)
    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(None, 0, None, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, None, None, 0, 0, 0, 0), out_axes=0)

    rho_tiles = deposit_tiles(
        tx,
        ty,
        tz,
        tiled_particles.x,
        tiled_particles.u,
        tiled_particles.active,
        scalar_weight,
    )

    rho_tiles = fold_tiled_ghost_cells(rho_tiles, world)
    rho_tiles = update_tiled_ghost_cells(rho_tiles, world)

    return rho_tiles


def _digital_filter_tiled_scalar(field_tiles, alpha, world):
    field_tiles = update_tiled_ghost_cells(field_tiles, world)

    filter_tiles = digital_filter
    filter_tiles = jax.vmap(filter_tiles, in_axes=(0, None), out_axes=0)
    filter_tiles = jax.vmap(filter_tiles, in_axes=(0, None), out_axes=0)
    filter_tiles = jax.vmap(filter_tiles, in_axes=(0, None), out_axes=0)

    return filter_tiles(field_tiles, alpha)


@jit
def compute_rho_from_tiled_particles(tiled_particles, rho, world, constants, grid=None):
    """Compute charge density on the vertex grid from tile-major particles."""
    if grid is None:
        grid = world["grids"]["vertex"]

    scalar_weight = tiled_particles.charge * tiled_particles.weight
    rho = _deposit_tiled_scalar_moment(
        tiled_particles,
        rho,
        world,
        grid,
        scalar_weight,
    )

    alpha = constants["alpha"]
    rho = digital_filter(rho, alpha)
    rho = update_ghost_cells(
        rho,
        world["boundary_conditions"]["x"],
        world["boundary_conditions"]["y"],
        world["boundary_conditions"]["z"],
    )

    return rho


@jit
def compute_tiled_rho_from_tiled_particles(tiled_particles, rho_tiles, world, constants, grid=None):
    """Compute charge density into tile-major vertex-grid scalar arrays."""
    if grid is None:
        grid = world["grids"]["vertex"]

    scalar_weight = tiled_particles.charge * tiled_particles.weight
    rho_tiles = _deposit_tiled_scalar_moment_to_tiles(
        tiled_particles,
        rho_tiles,
        world,
        grid,
        scalar_weight,
    )

    alpha = constants["alpha"]
    rho_tiles = _digital_filter_tiled_scalar(rho_tiles, alpha, world)
    rho_tiles = update_tiled_ghost_cells(rho_tiles, world)

    return rho_tiles


@jit
def compute_mass_density_from_tiled_particles(tiled_particles, rho, world, grid=None):
    """Compute mass density on the vertex grid from tile-major particles."""
    if grid is None:
        grid = world["grids"]["vertex"]

    scalar_weight = tiled_particles.mass * tiled_particles.weight
    position = _diagnostic_mass_position(tiled_particles, world)
    return _deposit_tiled_scalar_moment(
        tiled_particles,
        rho,
        world,
        grid,
        scalar_weight,
        position=position,
    )


@jit
def compute_tiled_mass_density_from_tiled_particles(tiled_particles, rho_tiles, world, grid=None):
    """Compute mass density into tile-major vertex-grid scalar arrays."""
    if grid is None:
        grid = world["grids"]["vertex"]

    scalar_weight = tiled_particles.mass * tiled_particles.weight
    position = _diagnostic_mass_position(tiled_particles, world)
    tile_shape = tuple(int(width) - 2 for width in rho_tiles.shape[3:])
    Nx = int(rho_tiles.shape[0]) * tile_shape[0]
    Ny = int(rho_tiles.shape[1]) * tile_shape[1]
    Nz = int(rho_tiles.shape[2]) * tile_shape[2]
    rho = jnp.zeros(
        (
            Nx + 2,
            Ny + 2,
            Nz + 2,
        ),
        dtype=rho_tiles.dtype,
    )
    rho = _deposit_tiled_scalar_moment(
        tiled_particles,
        rho,
        world,
        grid,
        scalar_weight,
        position=position,
    )

    return tile_scalar_field(rho, world, tile_shape)
