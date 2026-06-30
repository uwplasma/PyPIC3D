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


def _neighbor_offsets(count):
    offsets = []
    seen = set()
    for offset in (-1, 0, 1):
        key = offset % count
        if key not in seen:
            offsets.append(offset)
            seen.add(key)
    return tuple(offsets)


def refresh_tiled_particle_tiles(tiled_particles, world, tile_shape):
    """
    Move active particles into their owning tiles while preserving static shape.

    The refresh assumes particles move by at most one cell in a timestep, so each
    destination tile only reads candidate particles from its neighboring tiles.
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

    new_x = jnp.zeros_like(tiled_particles.x)
    new_u = jnp.zeros_like(tiled_particles.u)
    new_active = jnp.zeros_like(tiled_particles.active)
    overflow = jnp.asarray(False)

    x_offsets = _neighbor_offsets(ntx)
    y_offsets = _neighbor_offsets(nty)
    z_offsets = _neighbor_offsets(ntz)

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                for species in range(n_species):
                    candidate_x = []
                    candidate_u = []
                    candidate_active = []
                    candidate_dest_tx = []
                    candidate_dest_ty = []
                    candidate_dest_tz = []

                    for ox in x_offsets:
                        sx = (tx + ox) % ntx
                        for oy in y_offsets:
                            sy = (ty + oy) % nty
                            for oz in z_offsets:
                                sz = (tz + oz) % ntz
                                index = (sx, sy, sz, species)
                                candidate_x.append(bounded_x[index])
                                candidate_u.append(bounded_u[index])
                                candidate_active.append(bounded_active[index])
                                candidate_dest_tx.append(dest_tx[index])
                                candidate_dest_ty.append(dest_ty[index])
                                candidate_dest_tz.append(dest_tz[index])

                    candidate_x = jnp.concatenate(candidate_x, axis=0)
                    candidate_u = jnp.concatenate(candidate_u, axis=0)
                    candidate_active = jnp.concatenate(candidate_active, axis=0)
                    candidate_dest_tx = jnp.concatenate(candidate_dest_tx, axis=0)
                    candidate_dest_ty = jnp.concatenate(candidate_dest_ty, axis=0)
                    candidate_dest_tz = jnp.concatenate(candidate_dest_tz, axis=0)

                    keep = (
                        candidate_active
                        & (candidate_dest_tx == tx)
                        & (candidate_dest_ty == ty)
                        & (candidate_dest_tz == tz)
                    )
                    rank = jnp.cumsum(keep.astype(int)) - 1
                    fits = keep & (rank < n_slots)
                    safe_rank = jnp.where(fits, rank, 0)

                    local_x = jnp.zeros_like(new_x[tx, ty, tz, species])
                    local_u = jnp.zeros_like(new_u[tx, ty, tz, species])
                    local_active_count = jnp.zeros(n_slots, dtype=int)

                    valid = fits.astype(candidate_x.dtype)
                    local_x = local_x.at[safe_rank].add(valid[:, None] * candidate_x)
                    local_u = local_u.at[safe_rank].add(valid[:, None] * candidate_u)
                    local_active_count = local_active_count.at[safe_rank].add(fits.astype(int))

                    index = (tx, ty, tz, species)
                    new_x = new_x.at[index].set(local_x)
                    new_u = new_u.at[index].set(local_u)
                    new_active = new_active.at[index].set(local_active_count > 0)
                    overflow = overflow | (jnp.sum(keep) > n_slots)

    refreshed = TiledParticles(
        x=new_x,
        u=new_u,
        active=new_active,
    )

    return refreshed, overflow
