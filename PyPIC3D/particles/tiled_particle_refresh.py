import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import wrap_periodic_position
from PyPIC3D.particles.tiled_particles import TiledParticles


def _particle_boundary_conditions(world):
    if "particle_boundary_conditions" in world:
        return world["particle_boundary_conditions"]
    return {"x": 0, "y": 0, "z": 0}


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


def _particle_tile_indices(x, y, z, world, tile_shape, tile_counts):
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx, nty, ntz = tile_counts

    x_cell = jnp.floor((x + 0.5 * world["x_wind"]) / world["dx"]).astype(int)
    y_cell = jnp.floor((y + 0.5 * world["y_wind"]) / world["dy"]).astype(int)
    z_cell = jnp.floor((z + 0.5 * world["z_wind"]) / world["dz"]).astype(int)

    x_cell = jnp.clip(x_cell, 0, world["Nx"] - 1)
    y_cell = jnp.clip(y_cell, 0, world["Ny"] - 1)
    z_cell = jnp.clip(z_cell, 0, world["Nz"] - 1)

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


def _compact_tile_candidates(candidate_x, candidate_u, candidate_active, n_slots):
    """
    Repack fixed-size incoming streams into the destination tile slot axis.
    """

    leading_shape = candidate_active.shape[:-1]
    n_candidates = candidate_active.shape[-1]

    flat_x = candidate_x.reshape((-1, n_candidates, 3))
    flat_u = candidate_u.reshape((-1, n_candidates, 3))
    flat_active = candidate_active.reshape((-1, n_candidates))

    def compact_one(local_x_in, local_u_in, local_active_in):
        rank = jnp.cumsum(local_active_in.astype(int)) - 1
        fits = local_active_in & (rank < n_slots)
        safe_rank = jnp.where(fits, rank, 0)

        local_x = jnp.zeros((n_slots, 3), dtype=local_x_in.dtype)
        local_u = jnp.zeros((n_slots, 3), dtype=local_u_in.dtype)
        local_active_count = jnp.zeros(n_slots, dtype=int)

        valid = fits.astype(local_x_in.dtype)
        local_x = local_x.at[safe_rank].add(valid[:, None] * local_x_in)
        local_u = local_u.at[safe_rank].add(valid[:, None] * local_u_in)
        local_active_count = local_active_count.at[safe_rank].add(fits.astype(int))

        return local_x, local_u, local_active_count > 0, jnp.sum(local_active_in) > n_slots

    flat_new_x, flat_new_u, flat_new_active, flat_overflow = jax.vmap(compact_one)(
        flat_x,
        flat_u,
        flat_active,
    )

    new_x = flat_new_x.reshape(leading_shape + (n_slots, 3))
    new_u = flat_new_u.reshape(leading_shape + (n_slots, 3))
    new_active = flat_new_active.reshape(leading_shape + (n_slots,))
    overflow = jnp.any(flat_overflow)

    return new_x, new_u, new_active, overflow


def refresh_tiled_particle_tiles(tiled_particles, world, tile_shape):
    """
    Move active particles into their owning tiles while preserving static shape.

    The refresh assumes particles move by at most one cell in a timestep, so each
    particle either stays in its current tile or moves to an adjacent tile.
    Particles that do not fit in the destination tile capacity are dropped from
    the returned active mask and reported through the overflow flag.
    """

    ntx, nty, ntz, n_species, n_slots = tiled_particles.active.shape
    tile_counts = (ntx, nty, ntz)

    particle_bc = _particle_boundary_conditions(world)
    bounded_x = tiled_particles.x
    bounded_u = tiled_particles.u
    bounded_active = tiled_particles.active

    x1, u1, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 0],
        bounded_u[..., 0],
        bounded_active,
        world["x_wind"],
        particle_bc["x"],
    )
    x2, u2, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 1],
        bounded_u[..., 1],
        bounded_active,
        world["y_wind"],
        particle_bc["y"],
    )
    x3, u3, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 2],
        bounded_u[..., 2],
        bounded_active,
        world["z_wind"],
        particle_bc["z"],
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
        world,
        tile_shape,
        tile_counts,
    )

    tx = jnp.arange(ntx).reshape((ntx, 1, 1, 1, 1))
    ty = jnp.arange(nty).reshape((1, nty, 1, 1, 1))
    tz = jnp.arange(ntz).reshape((1, 1, ntz, 1, 1))

    offset_x = _adjacent_tile_offset(dest_tx, tx, ntx)
    offset_y = _adjacent_tile_offset(dest_ty, ty, nty)
    offset_z = _adjacent_tile_offset(dest_tz, tz, ntz)

    x_offsets = _movement_offsets(ntx)
    y_offsets = _movement_offsets(nty)
    z_offsets = _movement_offsets(ntz)

    candidate_x = []
    candidate_u = []
    candidate_active = []

    for ox in x_offsets:
        for oy in y_offsets:
            for oz in z_offsets:
                stream_active = (
                    bounded_active
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

                candidate_x.append(stream_x)
                candidate_u.append(stream_u)
                candidate_active.append(stream_active)

    candidate_x = jnp.concatenate(candidate_x, axis=-2)
    candidate_u = jnp.concatenate(candidate_u, axis=-2)
    candidate_active = jnp.concatenate(candidate_active, axis=-1)

    new_x, new_u, new_active, overflow = _compact_tile_candidates(
        candidate_x,
        candidate_u,
        candidate_active,
        n_slots,
    )

    refreshed = TiledParticles(
        x=new_x,
        u=new_u,
        active=new_active,
    )

    return refreshed, overflow
