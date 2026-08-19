import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.particles.particle_batching import (
    number_of_particle_batches,
    particle_batch_indices,
    prepare_particle_batches,
)
from PyPIC3D.pusher.boris import (
    boris_single_particle,
    interpolate_field_to_particles,
    relativistic_boris_single_particle,
)
from PyPIC3D.pusher.higuera_cary import higuera_cary_single_particle


def particle_push(particles, species_config, E_tiles, B_tiles, static_parameters, dynamic_parameters):
    """
    Push tile-major particles with the selected pusher using compact tiled Yee fields.

    Particles are assumed to live in the tile that owns their current forward
    position.  The configured field halos on each compact tile provide the
    neighboring Yee data needed by the interpolation stencil near tile faces.
    """

    relativistic = static_parameters.relativistic
    particle_pusher = static_parameters.particle_pusher

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    tile_nx, tile_ny, tile_nz = tile_shape
    g = int(static_parameters.guard_cells)
    dt = dynamic_parameters.dt
    shape_factor = static_parameters.shape_factor

    tiled_center_grid = dynamic_parameters.grids.tiled_center_grid
    tiled_vertex_grid = dynamic_parameters.grids.tiled_vertex_grid

    Ex_tiles, Ey_tiles, Ez_tiles = E_tiles
    Bx_tiles, By_tiles, Bz_tiles = B_tiles

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

    ntx, nty, ntz = particles.x.shape[:3]
    active_axes = (
        int(ntx) * int(tile_nx) > 1,
        int(nty) * int(tile_ny) > 1,
        int(ntz) * int(tile_nz) > 1,
    )
    inactive_axis_indices = (g, g, g)

    def push_one_tile(tx, ty, tz, x_tile, u_tile, active_tile, charge_species, mass_species, update_x_species,
                      Ex_tile, Ey_tile, Ez_tile, Bx_tile, By_tile, Bz_tile):
        particle_capacity, batch_size, active_indices, n_active = prepare_particle_batches(
            active_tile,
            static_parameters.particle_batch_size,
        )
        if particle_capacity == 0:
            return u_tile

        slots_per_species = active_tile.shape[-1]

        x_flat = x_tile.reshape(-1, 3)
        u_flat = u_tile.reshape(-1, 3)

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

        n_batches = number_of_particle_batches(n_active, batch_size)

        def push_batch(batch_state):
            batch_index, current_u = batch_state
            particle_indices, valid = particle_batch_indices(
                active_indices,
                n_active,
                batch_index,
                batch_size,
            )

            x_batch = x_flat[particle_indices]
            old_u_batch = current_u[particle_indices]
            x, y, z = x_batch[:, 0], x_batch[:, 1], x_batch[:, 2]
            vx, vy, vz = old_u_batch[:, 0], old_u_batch[:, 1], old_u_batch[:, 2]

            species_indices = particle_indices // slots_per_species
            q = charge_species[species_indices]
            m = mass_species[species_indices]
            update_x_batch = update_x_species[species_indices]

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

            if particle_pusher == "boris":
                if relativistic:
                    new_vx, new_vy, new_vz = relativistic_boris_vmap(
                        vx, vy, vz,
                        efield_atx, efield_aty, efield_atz,
                        bfield_atx, bfield_aty, bfield_atz,
                        q, m, dt, dynamic_parameters,
                    )
                else:
                    new_vx, new_vy, new_vz = boris_vmap(
                        vx, vy, vz,
                        efield_atx, efield_aty, efield_atz,
                        bfield_atx, bfield_aty, bfield_atz,
                        q, m, dt, dynamic_parameters,
                    )
            elif particle_pusher == "higuera_cary":
                new_vx, new_vy, new_vz = higuera_cary_vmap(
                    vx, vy, vz,
                    efield_atx, efield_aty, efield_atz,
                    bfield_atx, bfield_aty, bfield_atz,
                    q, m, dt, dynamic_parameters,
                )
            else:
                raise ValueError(f"Unknown particle_pusher: {particle_pusher}")

            new_u_batch = jnp.stack((new_vx, new_vy, new_vz), axis=-1)
            apply_update = valid[:, jnp.newaxis] & update_x_batch
            delta_u = jnp.where(apply_update, new_u_batch - old_u_batch, 0.0)
            current_u = current_u.at[particle_indices].add(delta_u)

            return batch_index + 1, current_u

        def batches_remaining(batch_state):
            batch_index, _ = batch_state
            return batch_index < n_batches

        _, new_u = jax.lax.while_loop(
            batches_remaining,
            push_batch,
            (jnp.asarray(0), u_flat),
        )

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
        species_config.update_x,
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
