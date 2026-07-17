import jax
import jax.numpy as jnp

from PyPIC3D.parameters import (
    constants_from_parameters,
    kernel_parameters_from_inputs,
    world_from_parameters,
)
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.pusher.boris import (
    boris_single_particle,
    interpolate_field_to_particles,
    relativistic_boris_single_particle,
)
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle


def particle_push(particles, species_config, E_tiles, B_tiles, world, constants, relativistic=True, particle_pusher="boris"):
    """
    Push tile-major particles with the selected pusher using compact tiled Yee fields.

    Particles are assumed to live in the tile that owns their current forward
    position.  The configured field halos on each compact tile provide the
    neighboring Yee data needed by the interpolation stencil near tile faces.
    """

    static_parameters, dynamic_parameters = kernel_parameters_from_inputs(
        world,
        constants,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )
    world = world_from_parameters(static_parameters, dynamic_parameters)
    constants = constants_from_parameters(dynamic_parameters)
    relativistic = static_parameters["relativistic"]
    particle_pusher = static_parameters["particle_pusher"]

    tile_shape = tuple(int(width) for width in world["tile_shape"])
    tile_nx, tile_ny, tile_nz = tile_shape
    g = int(world["guard_cells"])
    dt = world["dt"]
    shape_factor = world["shape_factor"]

    tiled_center_grid = world["grids"]["tiled_center_grid"]
    tiled_vertex_grid = world["grids"]["tiled_vertex_grid"]

    Ex_tiles, Ey_tiles, Ez_tiles = E_tiles
    Bx_tiles, By_tiles, Bz_tiles = B_tiles

    ntx, nty, ntz = particles.x.shape[:3]
    active_axes = (
        int(ntx) * int(tile_nx) > 1,
        int(nty) * int(tile_ny) > 1,
        int(ntz) * int(tile_nz) > 1,
    )
    inactive_axis_indices = (g, g, g)

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

        center_x = tiled_center_grid[0][tx, ty, tz]
        center_y = tiled_center_grid[1][tx, ty, tz]
        center_z = tiled_center_grid[2][tx, ty, tz]
        vertex_x = tiled_vertex_grid[0][tx, ty, tz]
        vertex_y = tiled_vertex_grid[1][tx, ty, tz]
        vertex_z = tiled_vertex_grid[2][tx, ty, tz]

        Ex_grid = vertex_x, center_y, center_z
        Ey_grid = center_x, vertex_y, center_z
        Ez_grid = center_x, center_y, vertex_z
        Bx_grid = center_x, vertex_y, vertex_z
        By_grid = vertex_x, center_y, vertex_z
        Bz_grid = vertex_x, vertex_y, center_z

        efield_atx = interpolate_field_to_particles(
            Ex_tile, x, y, z, Ex_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )
        efield_aty = interpolate_field_to_particles(
            Ey_tile, x, y, z, Ey_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )
        efield_atz = interpolate_field_to_particles(
            Ez_tile, x, y, z, Ez_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )

        bfield_atx = interpolate_field_to_particles(
            Bx_tile, x, y, z, Bx_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )
        bfield_aty = interpolate_field_to_particles(
            By_tile, x, y, z, By_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )
        bfield_atz = interpolate_field_to_particles(
            Bz_tile, x, y, z, Bz_grid, shape_factor, ghost_cells=True,
            active_axes=active_axes, inactive_axis_indices=inactive_axis_indices
        )

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
            if relativistic:
                new_vx, new_vy, new_vz = relativistic_boris_vmap(
                    vx, vy, vz,
                    efield_atx, efield_aty, efield_atz,
                    bfield_atx, bfield_aty, bfield_atz,
                    q, m, dt, constants,
                )
            else:
                new_vx, new_vy, new_vz = boris_vmap(
                    vx, vy, vz,
                    efield_atx, efield_aty, efield_atz,
                    bfield_atx, bfield_aty, bfield_atz,
                    q, m, dt, constants,
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
        particles.x,
        particles.u,
        particles.active,
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
        x=particles.x,
        u=new_u,
        active=particles.active,
    )
