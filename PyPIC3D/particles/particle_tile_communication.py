import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from PyPIC3D.boundary_conditions.grid_and_stencil import wrap_periodic_position
from PyPIC3D.boundary_conditions.ghost_cells import MESH_AXES
from PyPIC3D.particles.particle_class import TiledParticles


PARTICLE_STATE_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None, None)
PARTICLE_ACTIVE_TILE_SPEC = P("tile_x", "tile_y", "tile_z", None, None)


def _validate_particle_tile_topology(tiled_particles, mesh):
    tile_grid_shape = tuple(int(width) for width in tiled_particles.active.shape[:3])
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    if tile_grid_shape != mesh_shape:
        raise ValueError(
            "Tiled particle communication requires one logical particle tile per device: "
            f"particle tile topology {tile_grid_shape} does not match device mesh {mesh_shape}."
        )


def shard_tiled_particles(tiled_particles, static_parameters):
    """
    Place tile-major particle arrays on the same one-tile-per-device mesh as fields.
    """

    mesh = static_parameters.field_mesh
    _validate_particle_tile_topology(tiled_particles, mesh)
    state_sharding = NamedSharding(mesh, PARTICLE_STATE_TILE_SPEC)
    active_sharding = NamedSharding(mesh, PARTICLE_ACTIVE_TILE_SPEC)

    return TiledParticles(
        x=jax.device_put(tiled_particles.x, state_sharding),
        u=jax.device_put(tiled_particles.u, state_sharding),
        active=jax.device_put(tiled_particles.active, active_sharding),
    )


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
    tile_shape = static_parameters.tile_shape
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx, nty, ntz = tile_counts

    x_cell = jnp.floor((x + 0.5 * dynamic_parameters.x_wind) / dynamic_parameters.dx).astype(int)
    y_cell = jnp.floor((y + 0.5 * dynamic_parameters.y_wind) / dynamic_parameters.dy).astype(int)
    z_cell = jnp.floor((z + 0.5 * dynamic_parameters.z_wind) / dynamic_parameters.dz).astype(int)

    x_cell = jnp.clip(x_cell, 0, dynamic_parameters.Nx - 1)
    y_cell = jnp.clip(y_cell, 0, dynamic_parameters.Ny - 1)
    z_cell = jnp.clip(z_cell, 0, dynamic_parameters.Nz - 1)

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


def _send_positive_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == 0:
        return tuple((i, (i + 1) % axis_size) for i in range(axis_size))
    return tuple((i, i + 1) for i in range(axis_size - 1))


def _send_negative_permutation(axis_size, boundary_condition):
    axis_size = int(axis_size)
    if boundary_condition == 0:
        return tuple((i, (i - 1) % axis_size) for i in range(axis_size))
    return tuple((i, i - 1) for i in range(1, axis_size))


def _send_axis_stream(stream, offset, axis_name, axis_size, boundary_condition):
    if axis_size == 1 or offset == 0:
        return stream
    if offset == 1:
        return jax.lax.ppermute(
            stream,
            axis_name,
            _send_positive_permutation(axis_size, boundary_condition),
        )
    return jax.lax.ppermute(
        stream,
        axis_name,
        _send_negative_permutation(axis_size, boundary_condition),
    )


def _send_particle_stream(stream, offset_x, offset_y, offset_z, mesh_shape, particle_boundary_conditions):
    stream = _send_axis_stream(stream, offset_x, MESH_AXES[0], mesh_shape[0], particle_boundary_conditions[0])
    stream = _send_axis_stream(stream, offset_y, MESH_AXES[1], mesh_shape[1], particle_boundary_conditions[1])
    stream = _send_axis_stream(stream, offset_z, MESH_AXES[2], mesh_shape[2], particle_boundary_conditions[2])
    return stream


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


def _bounded_local_state_and_tile_offsets(local_x, local_u, local_active, static_parameters, dynamic_parameters, mesh_shape):
    """
    Apply physical particle boundaries on one local tile and identify neighbor offsets.
    """

    particle_bc = static_parameters.particle_boundary_conditions
    bounded_x = local_x
    bounded_u = local_u
    bounded_active = local_active

    x1, u1, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 0],
        bounded_u[..., 0],
        bounded_active,
        dynamic_parameters.x_wind,
        particle_bc[0],
    )
    x2, u2, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 1],
        bounded_u[..., 1],
        bounded_active,
        dynamic_parameters.y_wind,
        particle_bc[1],
    )
    x3, u3, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 2],
        bounded_u[..., 2],
        bounded_active,
        dynamic_parameters.z_wind,
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
        mesh_shape,
    )

    tx = jax.lax.axis_index(MESH_AXES[0])
    ty = jax.lax.axis_index(MESH_AXES[1])
    tz = jax.lax.axis_index(MESH_AXES[2])

    offset_x = _adjacent_tile_offset(dest_tx, tx, mesh_shape[0])
    offset_y = _adjacent_tile_offset(dest_ty, ty, mesh_shape[1])
    offset_z = _adjacent_tile_offset(dest_tz, tz, mesh_shape[2])

    return bounded_x, bounded_u, bounded_active, offset_x, offset_y, offset_z


def make_distributed_particle_refresher(static_parameters):
    mesh = static_parameters.field_mesh
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    particle_boundary_conditions = tuple(int(bc) for bc in static_parameters.particle_boundary_conditions)

    def local_refresh(local_x_tiles, local_u_tiles, local_active_tiles, dynamic_parameters):
        local_x = local_x_tiles[0, 0, 0]
        local_u = local_u_tiles[0, 0, 0]
        local_active = local_active_tiles[0, 0, 0]

        bounded_x, bounded_u, bounded_active, offset_x, offset_y, offset_z = _bounded_local_state_and_tile_offsets(
            local_x,
            local_u,
            local_active,
            static_parameters,
            dynamic_parameters,
            mesh_shape,
        )

        nonlocal_offset = (offset_x != 0) | (offset_y != 0) | (offset_z != 0)
        invalid_offset = (
            (jnp.abs(offset_x) > 1)
            | (jnp.abs(offset_y) > 1)
            | (jnp.abs(offset_z) > 1)
        )
        moving = bounded_active & nonlocal_offset & ~invalid_offset
        stay_active = bounded_active & ~moving & ~invalid_offset
        stay_x = jnp.where(stay_active[..., None], bounded_x, 0.0)
        stay_u = jnp.where(stay_active[..., None], bounded_u, 0.0)

        incoming_x = []
        incoming_u = []
        incoming_active = []

        for ox in _movement_offsets(mesh_shape[0]):
            for oy in _movement_offsets(mesh_shape[1]):
                for oz in _movement_offsets(mesh_shape[2]):
                    if ox == 0 and oy == 0 and oz == 0:
                        continue

                    stream_active = moving & (offset_x == ox) & (offset_y == oy) & (offset_z == oz)
                    stream_x = jnp.where(stream_active[..., None], bounded_x, 0.0)
                    stream_u = jnp.where(stream_active[..., None], bounded_u, 0.0)

                    incoming_x.append(
                        _send_particle_stream(
                            stream_x,
                            ox,
                            oy,
                            oz,
                            mesh_shape,
                            particle_boundary_conditions,
                        )
                    )
                    incoming_u.append(
                        _send_particle_stream(
                            stream_u,
                            ox,
                            oy,
                            oz,
                            mesh_shape,
                            particle_boundary_conditions,
                        )
                    )
                    incoming_active.append(
                        _send_particle_stream(
                            stream_active,
                            ox,
                            oy,
                            oz,
                            mesh_shape,
                            particle_boundary_conditions,
                        )
                    )

        if len(incoming_active) == 0:
            overflow = jnp.any(bounded_active & invalid_offset)
            overflow = jax.lax.pmax(overflow, MESH_AXES)
            return (
                stay_x[jnp.newaxis, jnp.newaxis, jnp.newaxis],
                stay_u[jnp.newaxis, jnp.newaxis, jnp.newaxis],
                stay_active[jnp.newaxis, jnp.newaxis, jnp.newaxis],
                overflow,
            )

        incoming_x = jnp.concatenate(incoming_x, axis=-2)
        incoming_u = jnp.concatenate(incoming_u, axis=-2)
        incoming_active = jnp.concatenate(incoming_active, axis=-1)

        new_x, new_u, new_active, capacity_overflow = _fill_incoming_particles(
            stay_x,
            stay_u,
            stay_active,
            incoming_x,
            incoming_u,
            incoming_active,
        )
        overflow = capacity_overflow | jnp.any(bounded_active & invalid_offset)
        overflow = jax.lax.pmax(overflow, MESH_AXES)

        return (
            new_x[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            new_u[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            new_active[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            overflow,
        )

    def refresh(tiled_particles, dynamic_parameters):
        _validate_particle_tile_topology(tiled_particles, mesh)
        mapped_refresh = jax.shard_map(
            local_refresh,
            mesh=mesh,
            in_specs=(
                PARTICLE_STATE_TILE_SPEC,
                PARTICLE_STATE_TILE_SPEC,
                PARTICLE_ACTIVE_TILE_SPEC,
                None,
            ),
            out_specs=(
                PARTICLE_STATE_TILE_SPEC,
                PARTICLE_STATE_TILE_SPEC,
                PARTICLE_ACTIVE_TILE_SPEC,
                P(),
            ),
            check_vma=False,
        )
        x, u, active, overflow = mapped_refresh(
            tiled_particles.x,
            tiled_particles.u,
            tiled_particles.active,
            dynamic_parameters,
        )
        return TiledParticles(x=x, u=u, active=active), overflow

    return refresh


def _refresh_tiled_particle_tiles_sparse(tiled_particles, static_parameters, dynamic_parameters):
    """
    Move active particles into owning tiles using neighbor-only incoming streams.
    """

    refresher = make_distributed_particle_refresher(
        static_parameters,
    )
    return refresher(tiled_particles, dynamic_parameters)


def refresh_tiled_particle_tiles(tiled_particles, static_parameters, dynamic_parameters):
    """
    Move active particles into their owning tiles while preserving static shape.

    The refresh assumes particles move by at most one cell in a timestep, so each
    particle either stays in its current tile or moves to an adjacent tile.  Tile
    transfers are performed inside the field device mesh with ppermute streams;
    no device owns local copies of other particle tiles.  Particles that do not
    fit in the destination tile capacity, or that would require a non-adjacent
    tile jump, are dropped from the returned active mask and reported through the
    overflow flag.
    """

    return _refresh_tiled_particle_tiles_sparse(tiled_particles, static_parameters, dynamic_parameters)
