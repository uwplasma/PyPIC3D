import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.pusher.boris import (
    boris_single_particle,
    interpolate_field_to_particles,
    relativistic_boris_single_particle,
)


def _tile_axis(axis, tile_index, cells_per_tile):
    start = tile_index * cells_per_tile
    return jax.lax.dynamic_slice(axis, (start,), (cells_per_tile + 2,))


@partial(jit, static_argnames=("tile_shape", "relativistic"))
def tiled_particle_push(tiled_particles, E_tiles, B_tiles, world, constants, tile_shape, relativistic=True):
    """
    Push tile-major particles with Boris using compact tiled Yee fields.

    Particles are assumed to live in the tile that owns their current forward
    position.  The one-cell field halos on each compact tile provide the
    neighboring Yee data needed by the interpolation stencil near tile faces.
    """

    tile_nx, tile_ny, tile_nz = tile_shape
    dt = world["dt"]
    shape_factor = world["shape_factor"]

    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]

    Ex_tiles, Ey_tiles, Ez_tiles = E_tiles
    Bx_tiles, By_tiles, Bz_tiles = B_tiles

    ntx, nty, ntz = tiled_particles.x.shape[:3]

    def push_one_tile(tx, ty, tz, x_tile, u_tile, charge_tile, mass_tile, active_tile, update_u1, update_u2, update_u3,
                      Ex_tile, Ey_tile, Ez_tile, Bx_tile, By_tile, Bz_tile):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        q = charge_tile.reshape(-1)
        m = mass_tile.reshape(-1)

        center_x = _tile_axis(center_grid[0], tx, tile_nx)
        center_y = _tile_axis(center_grid[1], ty, tile_ny)
        center_z = _tile_axis(center_grid[2], tz, tile_nz)
        vertex_x = _tile_axis(vertex_grid[0], tx, tile_nx)
        vertex_y = _tile_axis(vertex_grid[1], ty, tile_ny)
        vertex_z = _tile_axis(vertex_grid[2], tz, tile_nz)

        Ex_grid = vertex_x, center_y, center_z
        Ey_grid = center_x, vertex_y, center_z
        Ez_grid = center_x, center_y, vertex_z
        Bx_grid = center_x, vertex_y, vertex_z
        By_grid = vertex_x, center_y, vertex_z
        Bz_grid = vertex_x, vertex_y, center_z

        efield_atx = interpolate_field_to_particles(Ex_tile, x, y, z, Ex_grid, shape_factor, ghost_cells=True)
        efield_aty = interpolate_field_to_particles(Ey_tile, x, y, z, Ey_grid, shape_factor, ghost_cells=True)
        efield_atz = interpolate_field_to_particles(Ez_tile, x, y, z, Ez_grid, shape_factor, ghost_cells=True)

        bfield_atx = interpolate_field_to_particles(Bx_tile, x, y, z, Bx_grid, shape_factor, ghost_cells=True)
        bfield_aty = interpolate_field_to_particles(By_tile, x, y, z, By_grid, shape_factor, ghost_cells=True)
        bfield_atz = interpolate_field_to_particles(Bz_tile, x, y, z, Bz_grid, shape_factor, ghost_cells=True)

        boris_vmap = jax.vmap(
            boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )
        relativistic_boris_vmap = jax.vmap(
            relativistic_boris_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )

        new_vx, new_vy, new_vz = jax.lax.cond(
            relativistic,
            lambda _: relativistic_boris_vmap(
                vx, vy, vz,
                efield_atx, efield_aty, efield_atz,
                bfield_atx, bfield_aty, bfield_atz,
                q, m, dt, constants,
            ),
            lambda _: boris_vmap(
                vx, vy, vz,
                efield_atx, efield_aty, efield_atz,
                bfield_atx, bfield_aty, bfield_atz,
                q, m, dt, constants,
            ),
            operand=None,
        )

        active = active_tile.reshape(-1)
        update_u1 = update_u1.reshape(-1)
        update_u2 = update_u2.reshape(-1)
        update_u3 = update_u3.reshape(-1)

        new_u = u_tile.reshape(-1, 3)
        new_u = new_u.at[:, 0].set(jnp.where(active & update_u1, new_vx, vx))
        new_u = new_u.at[:, 1].set(jnp.where(active & update_u2, new_vy, vy))
        new_u = new_u.at[:, 2].set(jnp.where(active & update_u3, new_vz, vz))

        return new_u.reshape(u_tile.shape)

    push_tiles = push_one_tile
    tx = jnp.arange(ntx)
    ty = jnp.arange(nty)
    tz = jnp.arange(ntz)

    push_tiles = jax.vmap(push_tiles, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(None, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)

    new_u = push_tiles(
        tx, ty, tz,
        tiled_particles.x,
        tiled_particles.u,
        tiled_particles.charge,
        tiled_particles.mass,
        tiled_particles.active,
        tiled_particles.update_u1,
        tiled_particles.update_u2,
        tiled_particles.update_u3,
        Ex_tiles,
        Ey_tiles,
        Ez_tiles,
        Bx_tiles,
        By_tiles,
        Bz_tiles,
    )

    return TiledParticles(
        x=tiled_particles.x,
        u=new_u,
        charge=tiled_particles.charge,
        mass=tiled_particles.mass,
        weight=tiled_particles.weight,
        active=tiled_particles.active,
        update_x1=tiled_particles.update_x1,
        update_x2=tiled_particles.update_x2,
        update_x3=tiled_particles.update_x3,
        update_u1=tiled_particles.update_u1,
        update_u2=tiled_particles.update_u2,
        update_u3=tiled_particles.update_u3,
    )
