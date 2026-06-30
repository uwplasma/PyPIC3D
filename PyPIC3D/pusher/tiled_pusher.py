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
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle


def _tile_axis(axis, tile_index, cells_per_tile, num_guard_cells, d):
    local_n = cells_per_tile + 2 * num_guard_cells
    offsets = jnp.arange(local_n, dtype=axis.dtype)
    return axis[0] + (offsets + tile_index * cells_per_tile - (num_guard_cells - 1)) * d


@partial(jit, static_argnames=("tile_shape", "g", "relativistic", "particle_pusher"))
def tiled_particle_push(tiled_particles, species_config, E_tiles, B_tiles, world, constants, tile_shape, g, relativistic=True, particle_pusher="boris"):
    """
    Push tile-major particles with the selected pusher using compact tiled Yee fields.

    Particles are assumed to live in the tile that owns their current forward
    position.  The configured field halos on each compact tile provide the
    neighboring Yee data needed by the interpolation stencil near tile faces.
    """

    tile_nx, tile_ny, tile_nz = tile_shape
    dt = world["dt"]
    shape_factor = world["shape_factor"]

    center_grid = world["grids"]["center"]
    vertex_grid = world["grids"]["vertex"]

    Ex_tiles, Ey_tiles, Ez_tiles = E_tiles
    Bx_tiles, By_tiles, Bz_tiles = B_tiles
    g = int(g)

    ntx, nty, ntz = tiled_particles.x.shape[:3]

    def push_one_tile(tx, ty, tz, x_tile, u_tile, active_tile, charge_species, mass_species, update_u_species,
                      Ex_tile, Ey_tile, Ez_tile, Bx_tile, By_tile, Bz_tile):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        q = jnp.broadcast_to(charge_species[:, jnp.newaxis], active_tile.shape).reshape(-1)
        m = jnp.broadcast_to(mass_species[:, jnp.newaxis], active_tile.shape).reshape(-1)

        center_x = _tile_axis(center_grid[0], tx, tile_nx, g, world["dx"])
        center_y = _tile_axis(center_grid[1], ty, tile_ny, g, world["dy"])
        center_z = _tile_axis(center_grid[2], tz, tile_nz, g, world["dz"])
        vertex_x = _tile_axis(vertex_grid[0], tx, tile_nx, g, world["dx"])
        vertex_y = _tile_axis(vertex_grid[1], ty, tile_ny, g, world["dy"])
        vertex_z = _tile_axis(vertex_grid[2], tz, tile_nz, g, world["dz"])

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
        higuera_cary_vmap = jax.vmap(
            higuera_cary_single_particle,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )

        if particle_pusher == "boris":
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
        elif particle_pusher == "higuera_cary":
            new_vx, new_vy, new_vz = higuera_cary_vmap(
                vx, vy, vz,
                efield_atx, efield_aty, efield_atz,
                bfield_atx, bfield_aty, bfield_atz,
                q, m, dt, constants,
            )
        else:
            raise ValueError(f"Unknown particle_pusher: {particle_pusher}")

        active = active_tile.reshape(-1)
        update_u1 = jnp.broadcast_to(update_u_species[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_u2 = jnp.broadcast_to(update_u_species[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_u3 = jnp.broadcast_to(update_u_species[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)

        new_u = u_tile.reshape(-1, 3)
        new_u = new_u.at[:, 0].set(jnp.where(active & update_u1, new_vx, vx))
        new_u = new_u.at[:, 1].set(jnp.where(active & update_u2, new_vy, vy))
        new_u = new_u.at[:, 2].set(jnp.where(active & update_u3, new_vz, vz))

        return new_u.reshape(u_tile.shape)

    push_tiles = push_one_tile
    tx = jnp.arange(ntx)
    ty = jnp.arange(nty)
    tz = jnp.arange(ntz)

    push_tiles = jax.vmap(push_tiles, in_axes=(None, None, 0, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(None, 0, None, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(0, None, None, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0), out_axes=0)

    new_u = push_tiles(
        tx, ty, tz,
        tiled_particles.x,
        tiled_particles.u,
        tiled_particles.active,
        species_config.charge,
        species_config.mass,
        species_config.update_u,
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
        active=tiled_particles.active,
    )
