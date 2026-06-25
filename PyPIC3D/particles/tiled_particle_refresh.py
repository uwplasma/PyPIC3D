import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import wrap_periodic_position
from PyPIC3D.particles.tiled_particles import TiledParticles


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


def update_tiled_particle_positions(tiled_particles, world):
    """
    Advance tile-major particle positions without changing tile ownership.
    """

    dt = world["dt"]
    active = tiled_particles.active.astype(tiled_particles.x.dtype)

    dx = active * tiled_particles.u[..., 0] * dt
    dy = active * tiled_particles.u[..., 1] * dt
    dz = active * tiled_particles.u[..., 2] * dt

    x = tiled_particles.x
    x = x.at[..., 0].set(jnp.where(tiled_particles.active & tiled_particles.update_x1, x[..., 0] + dx, x[..., 0]))
    x = x.at[..., 1].set(jnp.where(tiled_particles.active & tiled_particles.update_x2, x[..., 1] + dy, x[..., 1]))
    x = x.at[..., 2].set(jnp.where(tiled_particles.active & tiled_particles.update_x3, x[..., 2] + dz, x[..., 2]))

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

    wrapped_x = tiled_particles.x
    wrapped_x = wrapped_x.at[..., 0].set(wrap_periodic_position(wrapped_x[..., 0], world["x_wind"]))
    wrapped_x = wrapped_x.at[..., 1].set(wrap_periodic_position(wrapped_x[..., 1], world["y_wind"]))
    wrapped_x = wrapped_x.at[..., 2].set(wrap_periodic_position(wrapped_x[..., 2], world["z_wind"]))

    dest_tx, dest_ty, dest_tz = _particle_tile_indices(
        wrapped_x[..., 0],
        wrapped_x[..., 1],
        wrapped_x[..., 2],
        world,
        tile_shape,
        tile_counts,
    )

    new_x = jnp.zeros_like(tiled_particles.x)
    new_u = jnp.zeros_like(tiled_particles.u)
    new_charge = jnp.zeros_like(tiled_particles.charge)
    new_mass = jnp.zeros_like(tiled_particles.mass)
    new_weight = jnp.zeros_like(tiled_particles.weight)
    new_active = jnp.zeros_like(tiled_particles.active)
    new_update_x1 = jnp.zeros_like(tiled_particles.update_x1)
    new_update_x2 = jnp.zeros_like(tiled_particles.update_x2)
    new_update_x3 = jnp.zeros_like(tiled_particles.update_x3)
    new_update_u1 = jnp.zeros_like(tiled_particles.update_u1)
    new_update_u2 = jnp.zeros_like(tiled_particles.update_u2)
    new_update_u3 = jnp.zeros_like(tiled_particles.update_u3)
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
                    candidate_charge = []
                    candidate_mass = []
                    candidate_weight = []
                    candidate_active = []
                    candidate_update_x1 = []
                    candidate_update_x2 = []
                    candidate_update_x3 = []
                    candidate_update_u1 = []
                    candidate_update_u2 = []
                    candidate_update_u3 = []
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
                                candidate_x.append(wrapped_x[index])
                                candidate_u.append(tiled_particles.u[index])
                                candidate_charge.append(tiled_particles.charge[index])
                                candidate_mass.append(tiled_particles.mass[index])
                                candidate_weight.append(tiled_particles.weight[index])
                                candidate_active.append(tiled_particles.active[index])
                                candidate_update_x1.append(tiled_particles.update_x1[index])
                                candidate_update_x2.append(tiled_particles.update_x2[index])
                                candidate_update_x3.append(tiled_particles.update_x3[index])
                                candidate_update_u1.append(tiled_particles.update_u1[index])
                                candidate_update_u2.append(tiled_particles.update_u2[index])
                                candidate_update_u3.append(tiled_particles.update_u3[index])
                                candidate_dest_tx.append(dest_tx[index])
                                candidate_dest_ty.append(dest_ty[index])
                                candidate_dest_tz.append(dest_tz[index])

                    candidate_x = jnp.concatenate(candidate_x, axis=0)
                    candidate_u = jnp.concatenate(candidate_u, axis=0)
                    candidate_charge = jnp.concatenate(candidate_charge, axis=0)
                    candidate_mass = jnp.concatenate(candidate_mass, axis=0)
                    candidate_weight = jnp.concatenate(candidate_weight, axis=0)
                    candidate_active = jnp.concatenate(candidate_active, axis=0)
                    candidate_update_x1 = jnp.concatenate(candidate_update_x1, axis=0)
                    candidate_update_x2 = jnp.concatenate(candidate_update_x2, axis=0)
                    candidate_update_x3 = jnp.concatenate(candidate_update_x3, axis=0)
                    candidate_update_u1 = jnp.concatenate(candidate_update_u1, axis=0)
                    candidate_update_u2 = jnp.concatenate(candidate_update_u2, axis=0)
                    candidate_update_u3 = jnp.concatenate(candidate_update_u3, axis=0)
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
                    local_charge = jnp.zeros_like(new_charge[tx, ty, tz, species])
                    local_mass = jnp.zeros_like(new_mass[tx, ty, tz, species])
                    local_weight = jnp.zeros_like(new_weight[tx, ty, tz, species])
                    local_active_count = jnp.zeros(n_slots, dtype=int)
                    local_update_x1_count = jnp.zeros(n_slots, dtype=int)
                    local_update_x2_count = jnp.zeros(n_slots, dtype=int)
                    local_update_x3_count = jnp.zeros(n_slots, dtype=int)
                    local_update_u1_count = jnp.zeros(n_slots, dtype=int)
                    local_update_u2_count = jnp.zeros(n_slots, dtype=int)
                    local_update_u3_count = jnp.zeros(n_slots, dtype=int)

                    valid = fits.astype(candidate_x.dtype)
                    local_x = local_x.at[safe_rank].add(valid[:, None] * candidate_x)
                    local_u = local_u.at[safe_rank].add(valid[:, None] * candidate_u)
                    local_charge = local_charge.at[safe_rank].add(valid * candidate_charge)
                    local_mass = local_mass.at[safe_rank].add(valid * candidate_mass)
                    local_weight = local_weight.at[safe_rank].add(valid * candidate_weight)
                    local_active_count = local_active_count.at[safe_rank].add(fits.astype(int))
                    local_update_x1_count = local_update_x1_count.at[safe_rank].add((fits & candidate_update_x1).astype(int))
                    local_update_x2_count = local_update_x2_count.at[safe_rank].add((fits & candidate_update_x2).astype(int))
                    local_update_x3_count = local_update_x3_count.at[safe_rank].add((fits & candidate_update_x3).astype(int))
                    local_update_u1_count = local_update_u1_count.at[safe_rank].add((fits & candidate_update_u1).astype(int))
                    local_update_u2_count = local_update_u2_count.at[safe_rank].add((fits & candidate_update_u2).astype(int))
                    local_update_u3_count = local_update_u3_count.at[safe_rank].add((fits & candidate_update_u3).astype(int))

                    index = (tx, ty, tz, species)
                    new_x = new_x.at[index].set(local_x)
                    new_u = new_u.at[index].set(local_u)
                    new_charge = new_charge.at[index].set(local_charge)
                    new_mass = new_mass.at[index].set(local_mass)
                    new_weight = new_weight.at[index].set(local_weight)
                    new_active = new_active.at[index].set(local_active_count > 0)
                    new_update_x1 = new_update_x1.at[index].set(local_update_x1_count > 0)
                    new_update_x2 = new_update_x2.at[index].set(local_update_x2_count > 0)
                    new_update_x3 = new_update_x3.at[index].set(local_update_x3_count > 0)
                    new_update_u1 = new_update_u1.at[index].set(local_update_u1_count > 0)
                    new_update_u2 = new_update_u2.at[index].set(local_update_u2_count > 0)
                    new_update_u3 = new_update_u3.at[index].set(local_update_u3_count > 0)
                    overflow = overflow | (jnp.sum(keep) > n_slots)

    refreshed = TiledParticles(
        x=new_x,
        u=new_u,
        charge=new_charge,
        mass=new_mass,
        weight=new_weight,
        active=new_active,
        update_x1=new_update_x1,
        update_x2=new_update_x2,
        update_x3=new_update_x3,
        update_u1=new_update_u1,
        update_u2=new_update_u2,
        update_u3=new_update_u3,
    )

    return refreshed, overflow
