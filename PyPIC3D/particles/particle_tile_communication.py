import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import wrap_periodic_position
from PyPIC3D.particles.particle_class import TiledParticles


def _apply_tiled_axis_boundary(x, u, active, wind, bc):
    half_wind = 0.5 * wind
    periodic = bc == 0
    reflecting = bc == 1
    absorbing = bc == 2

    periodic_x = wrap_periodic_position(x, wind)
    reflected_x = jnp.where(
        x > half_wind,
        2.0 * half_wind - x,
        jnp.where(x < -half_wind, -2.0 * half_wind - x, x),
    )
    reflected_u = jnp.where((x >= half_wind) | (x <= -half_wind), -u, u)

    x_out = jnp.where(periodic, periodic_x, jnp.where(reflecting, reflected_x, x))
    u_out = jnp.where(reflecting, reflected_u, u)
    active_out = jnp.where(absorbing, active & (x <= half_wind) & (x >= -half_wind), active)

    return x_out, u_out, active_out


def _particle_tile_indices(x, y, z, static_parameters, dynamic_parameters, tile_counts):
    tile_shape = static_parameters["tile_shape"]
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx, nty, ntz = tile_counts

    x_cell = jnp.floor((x + 0.5 * dynamic_parameters["x_wind"]) / dynamic_parameters["dx"]).astype(int)
    y_cell = jnp.floor((y + 0.5 * dynamic_parameters["y_wind"]) / dynamic_parameters["dy"]).astype(int)
    z_cell = jnp.floor((z + 0.5 * dynamic_parameters["z_wind"]) / dynamic_parameters["dz"]).astype(int)

    x_cell = jnp.clip(x_cell, 0, dynamic_parameters["Nx"] - 1)
    y_cell = jnp.clip(y_cell, 0, dynamic_parameters["Ny"] - 1)
    z_cell = jnp.clip(z_cell, 0, dynamic_parameters["Nz"] - 1)

    return (
        jnp.clip(x_cell // tile_nx, 0, ntx - 1),
        jnp.clip(y_cell // tile_ny, 0, nty - 1),
        jnp.clip(z_cell // tile_nz, 0, ntz - 1),
    )


def update_tiled_particle_positions(tiled_particles, species_config, dt):
    """
    Advance tile-major particle positions without changing tile ownership.
    """

    active = tiled_particles.active.astype(tiled_particles.x.dtype)
    update_x = species_config.update_x.reshape((1, 1, 1, species_config.update_x.shape[0], 1, 3))

    dx = active * tiled_particles.u[..., 0] * dt
    dy = active * tiled_particles.u[..., 1] * dt
    dz = active * tiled_particles.u[..., 2] * dt

    x = tiled_particles.x
    x = x.at[..., 0].set(jnp.where(tiled_particles.active & update_x[..., 0], x[..., 0] + dx, x[..., 0]))
    x = x.at[..., 1].set(jnp.where(tiled_particles.active & update_x[..., 1], x[..., 1] + dy, x[..., 1]))
    x = x.at[..., 2].set(jnp.where(tiled_particles.active & update_x[..., 2], x[..., 2] + dz, x[..., 2]))

    return tiled_particles._replace(x=x)


def _movement_offsets(count):
    if count == 1:
        return (0,)
    return (1, 0, -1)


def _adjacent_tile_offset(dest_tile, source_tile, tile_count):
    """
    Signed adjacent offset from the source tile to the destination tile.

    The tiled particle step assumes particles move by at most one cell, so tile
    ownership can only change by one neighboring tile along any active axis.
    Periodic end points are represented with the physical crossing direction:
    first -> last is -1, last -> first is +1.
    """

    if tile_count == 1:
        return jnp.zeros_like(dest_tile)

    offset = dest_tile - source_tile
    if tile_count == 2:
        return offset

    offset = jnp.where(offset == tile_count - 1, -1, offset)
    offset = jnp.where(offset == -(tile_count - 1), 1, offset)

    return offset



def _bounded_state_and_tile_offsets(tiled_particles, static_parameters, dynamic_parameters):
    """
    Apply physical particle boundaries and identify the adjacent tile offset.
    """

    ntx, nty, ntz, n_species, n_slots = tiled_particles.active.shape
    tile_counts = (ntx, nty, ntz)

    particle_bc = static_parameters["particle_boundary_conditions"]
    bounded_x = tiled_particles.x
    bounded_u = tiled_particles.u
    bounded_active = tiled_particles.active

    x1, u1, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 0],
        bounded_u[..., 0],
        bounded_active,
        dynamic_parameters["x_wind"],
        particle_bc[0],
    )
    x2, u2, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 1],
        bounded_u[..., 1],
        bounded_active,
        dynamic_parameters["y_wind"],
        particle_bc[1],
    )
    x3, u3, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 2],
        bounded_u[..., 2],
        bounded_active,
        dynamic_parameters["z_wind"],
        particle_bc[2],
    )

    bounded_x = bounded_x.at[..., 0].set(x1)
    bounded_x = bounded_x.at[..., 1].set(x2)
    bounded_x = bounded_x.at[..., 2].set(x3)
    bounded_u = bounded_u.at[..., 0].set(u1)
    bounded_u = bounded_u.at[..., 1].set(u2)
    bounded_u = bounded_u.at[..., 2].set(u3)

    dest_tx, dest_ty, dest_tz = _particle_tile_indices(
        bounded_x[..., 0],
        bounded_x[..., 1],
        bounded_x[..., 2],
        static_parameters,
        dynamic_parameters,
        tile_counts,
    )

    tx = jnp.arange(ntx).reshape((ntx, 1, 1, 1, 1))
    ty = jnp.arange(nty).reshape((1, nty, 1, 1, 1))
    tz = jnp.arange(ntz).reshape((1, 1, ntz, 1, 1))

    offset_x = _adjacent_tile_offset(dest_tx, tx, ntx)
    offset_y = _adjacent_tile_offset(dest_ty, ty, nty)
    offset_z = _adjacent_tile_offset(dest_tz, tz, ntz)

    return bounded_x, bounded_u, bounded_active, offset_x, offset_y, offset_z


def _fill_incoming_particles(stay_x, stay_u, stay_active, incoming_x, incoming_u, incoming_active):
    """
    Fill inactive destination slots with incoming neighbor particles.

    The slot layout remains fixed.  Tiles without incoming particles keep their
    stay-particle slots untouched; tiles with incoming particles use the first
    available inactive slots and report overflow when the incoming stream is
    larger than the local free capacity.
    """

    leading_shape = stay_active.shape[:-1]
    n_slots = stay_active.shape[-1]
    n_candidates = incoming_active.shape[-1]

    flat_stay_x = stay_x.reshape((-1, n_slots, 3))
    flat_stay_u = stay_u.reshape((-1, n_slots, 3))
    flat_stay_active = stay_active.reshape((-1, n_slots))
    flat_incoming_x = incoming_x.reshape((-1, n_candidates, 3))
    flat_incoming_u = incoming_u.reshape((-1, n_candidates, 3))
    flat_incoming_active = incoming_active.reshape((-1, n_candidates))

    slot_ids = jnp.arange(n_slots)

    def fill_one(local_x, local_u, local_active, incoming_x_in, incoming_u_in, incoming_active_in):
        free = ~local_active
        free_rank = jnp.cumsum(free.astype(int)) - 1
        safe_free_rank = jnp.where(free, free_rank, 0)
        slot_for_rank = jnp.zeros(n_slots, dtype=slot_ids.dtype)
        slot_for_rank = slot_for_rank.at[safe_free_rank].add(jnp.where(free, slot_ids, 0))

        incoming_rank = jnp.cumsum(incoming_active_in.astype(int)) - 1
        n_free = jnp.sum(free.astype(int))
        fits = incoming_active_in & (incoming_rank < n_free)
        overflow = jnp.any(incoming_active_in & (incoming_rank >= n_free))

        safe_rank = jnp.where(fits, incoming_rank, 0)
        selected_slots = slot_for_rank[safe_rank]
        valid = fits.astype(local_x.dtype)

        incoming_count = jnp.zeros(n_slots, dtype=int)
        local_x = local_x.at[selected_slots].add(valid[:, None] * incoming_x_in)
        local_u = local_u.at[selected_slots].add(valid[:, None] * incoming_u_in)
        incoming_count = incoming_count.at[selected_slots].add(fits.astype(int))
        local_active = local_active | (incoming_count > 0)

        return local_x, local_u, local_active, overflow

    flat_x, flat_u, flat_active, flat_overflow = jax.vmap(fill_one)(
        flat_stay_x,
        flat_stay_u,
        flat_stay_active,
        flat_incoming_x,
        flat_incoming_u,
        flat_incoming_active,
    )

    new_x = flat_x.reshape(leading_shape + (n_slots, 3))
    new_u = flat_u.reshape(leading_shape + (n_slots, 3))
    new_active = flat_active.reshape(leading_shape + (n_slots,))
    overflow = jnp.any(flat_overflow)

    return new_x, new_u, new_active, overflow


def _refresh_tiled_particle_tiles_sparse(tiled_particles, static_parameters, dynamic_parameters):
    """
    Move active particles into owning tiles using neighbor-only incoming streams.
    """

    bounded_x, bounded_u, bounded_active, offset_x, offset_y, offset_z = _bounded_state_and_tile_offsets(
        tiled_particles,
        static_parameters,
        dynamic_parameters,
    )

    ntx, nty, ntz, n_species, n_slots = bounded_active.shape
    moving = bounded_active & ((offset_x != 0) | (offset_y != 0) | (offset_z != 0))
    stay_active = bounded_active & ~moving
    stay_x = jnp.where(stay_active[..., None], bounded_x, 0.0)
    stay_u = jnp.where(stay_active[..., None], bounded_u, 0.0)

    incoming_x = []
    incoming_u = []
    incoming_active = []

    for ox in _movement_offsets(ntx):
        for oy in _movement_offsets(nty):
            for oz in _movement_offsets(ntz):
                if ox == 0 and oy == 0 and oz == 0:
                    continue

                stream_active = (
                    moving
                    & (offset_x == ox)
                    & (offset_y == oy)
                    & (offset_z == oz)
                )
                stream_x = jnp.where(stream_active[..., None], bounded_x, 0.0)
                stream_u = jnp.where(stream_active[..., None], bounded_u, 0.0)

                stream_x = jnp.roll(stream_x, shift=ox, axis=0)
                stream_x = jnp.roll(stream_x, shift=oy, axis=1)
                stream_x = jnp.roll(stream_x, shift=oz, axis=2)
                stream_u = jnp.roll(stream_u, shift=ox, axis=0)
                stream_u = jnp.roll(stream_u, shift=oy, axis=1)
                stream_u = jnp.roll(stream_u, shift=oz, axis=2)
                stream_active = jnp.roll(stream_active, shift=ox, axis=0)
                stream_active = jnp.roll(stream_active, shift=oy, axis=1)
                stream_active = jnp.roll(stream_active, shift=oz, axis=2)

                incoming_x.append(stream_x)
                incoming_u.append(stream_u)
                incoming_active.append(stream_active)

    if len(incoming_active) == 0:
        overflow = jnp.asarray(False)
        return TiledParticles(x=stay_x, u=stay_u, active=stay_active), overflow

    incoming_x = jnp.concatenate(incoming_x, axis=-2)
    incoming_u = jnp.concatenate(incoming_u, axis=-2)
    incoming_active = jnp.concatenate(incoming_active, axis=-1)

    new_x, new_u, new_active, overflow = _fill_incoming_particles(
        stay_x,
        stay_u,
        stay_active,
        incoming_x,
        incoming_u,
        incoming_active,
    )

    refreshed = TiledParticles(
        x=new_x,
        u=new_u,
        active=new_active,
    )

    return refreshed, overflow


def refresh_tiled_particle_tiles(tiled_particles, static_parameters, dynamic_parameters):
    """
    Move active particles into their owning tiles while preserving static shape.

    The refresh assumes particles move by at most one cell in a timestep, so each
    particle either stays in its current tile or moves to an adjacent tile.
    Particles that do not fit in the destination tile capacity are dropped from
    the returned active mask and reported through the overflow flag.
    """

    return _refresh_tiled_particle_tiles_sparse(tiled_particles, static_parameters, dynamic_parameters)
