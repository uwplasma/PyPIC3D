import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_PERIODIC,
    wrap_periodic_position,
)
from PyPIC3D.boundary_conditions.ghost_cells import MESH_AXES
from PyPIC3D.boundary_conditions.halo_exchange import (
    _send_negative_permutation,
    _send_positive_permutation,
)
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import grid_domain_bounds


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


def _apply_tiled_axis_boundary(x, u, active, axis_min, axis_max, bc):
    wind = axis_max - axis_min
    center = 0.5 * (axis_min + axis_max)
    periodic = bc == BC_PERIODIC
    reflecting = bc == BC_CONDUCTING
    absorbing = bc == BC_ABSORBING

    periodic_x = wrap_periodic_position(x - center, wind) + center
    reflected_x = jnp.where(
        x > axis_max,
        2.0 * axis_max - x,
        jnp.where(x < axis_min, 2.0 * axis_min - x, x),
    )
    reflected_u = jnp.where((x >= axis_max) | (x <= axis_min), -u, u)

    x_out = jnp.where(periodic, periodic_x, jnp.where(reflecting, reflected_x, x))
    u_out = jnp.where(reflecting, reflected_u, u)
    active_out = jnp.where(absorbing, active & (x <= axis_max) & (x >= axis_min), active)

    return x_out, u_out, active_out


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


def _particle_axis_tile_index(position, axis_min, cell_width, tile_width, tile_count):
    """Return the owner tile along one axis for bounded particle positions."""

    cell = jnp.floor((position - axis_min) / cell_width).astype(int)
    cell = jnp.clip(cell, 0, int(tile_width) * int(tile_count) - 1)
    return jnp.clip(cell // int(tile_width), 0, int(tile_count) - 1)


def _particle_face_packet_capacity(n_slots, tile_shape, axis, n_directions=1):
    """
    Choose a static mover capacity from the tile slot density and face area.

    The packet holds the capacity-equivalent population of one boundary-cell
    layer per direction.  Concentrated boundary populations can exceed this
    estimate, so every pack operation also returns an overflow flag.
    """

    n_slots = int(n_slots)
    tile_shape = tuple(int(width) for width in tile_shape)
    tile_cells = int(tile_shape[0] * tile_shape[1] * tile_shape[2])
    face_cells = tile_cells // tile_shape[int(axis)]
    slots_per_cell = (n_slots + tile_cells - 1) // tile_cells
    return min(n_slots, int(n_directions) * face_cells * slots_per_cell)


def _pack_particle_packet(x, u, moving, packet_capacity):
    """Compact per-species movers into one homogeneous ``[x, u, valid]`` packet."""

    packet_capacity = int(packet_capacity)
    packet_lanes = jnp.arange(packet_capacity)

    def pack_species(species_x, species_u, species_moving):
        mover_count = jnp.sum(species_moving.astype(int))
        mover_indices = jnp.flatnonzero(
            species_moving,
            size=packet_capacity,
            fill_value=0,
        )
        valid = packet_lanes < mover_count
        packet_x = jnp.where(valid[:, None], species_x[mover_indices], 0.0)
        packet_u = jnp.where(valid[:, None], species_u[mover_indices], 0.0)
        packet_valid = valid[:, None].astype(species_x.dtype)
        packet = jnp.concatenate((packet_x, packet_u, packet_valid), axis=-1)
        return packet, mover_count > packet_capacity

    packet, overflow = jax.vmap(pack_species)(x, u, moving)
    return packet, jnp.any(overflow)


def _unpack_particle_packet(packet):
    """Split a homogeneous mover packet into the local particle representation."""

    return packet[..., :3], packet[..., 3:6], packet[..., 6] != 0.0


def _exchange_particle_axis(
    x,
    u,
    active,
    *,
    axis,
    axis_min,
    cell_width,
    tile_width,
    tile_count,
    packet_layout,
    negative_permutation,
    positive_permutation,
):
    """Move particles across the two immediate faces of one mesh axis."""

    tile_count = int(tile_count)
    if tile_count == 1:
        return x, u, active, jnp.asarray(False)

    axis_name = MESH_AXES[int(axis)]
    source_tile = jax.lax.axis_index(axis_name)
    destination_tile = _particle_axis_tile_index(
        x[..., int(axis)],
        axis_min,
        cell_width,
        tile_width,
        tile_count,
    )
    offset = _adjacent_tile_offset(destination_tile, source_tile, tile_count)
    invalid = active & (jnp.abs(offset) > 1)
    moving_negative = active & (offset == -1)
    moving_positive = active & (offset == 1)
    moving = (moving_negative | moving_positive) & ~invalid

    stay_active = active & ~moving & ~invalid
    stay_x = jnp.where(stay_active[..., None], x, 0.0)
    stay_u = jnp.where(stay_active[..., None], u, 0.0)

    n_slots = active.shape[-1]
    if tile_count == 2:
        packet_capacity = _particle_face_packet_capacity(
            n_slots,
            *packet_layout,
        )
        packet, packet_overflow = _pack_particle_packet(
            x,
            u,
            moving,
            packet_capacity,
        )
        incoming_packet = jax.lax.ppermute(
            packet,
            axis_name,
            ((0, 1), (1, 0)),
        )
    else:
        packet_capacity = _particle_face_packet_capacity(
            n_slots,
            *packet_layout,
        )
        negative_packet, negative_overflow = _pack_particle_packet(
            x,
            u,
            moving_negative & ~invalid,
            packet_capacity,
        )
        positive_packet, positive_overflow = _pack_particle_packet(
            x,
            u,
            moving_positive & ~invalid,
            packet_capacity,
        )
        incoming_negative = jax.lax.ppermute(
            negative_packet,
            axis_name,
            negative_permutation,
        )
        incoming_positive = jax.lax.ppermute(
            positive_packet,
            axis_name,
            positive_permutation,
        )
        incoming_packet = jnp.concatenate(
            (incoming_negative, incoming_positive),
            axis=-2,
        )
        packet_overflow = negative_overflow | positive_overflow

    incoming_x, incoming_u, incoming_active = _unpack_particle_packet(incoming_packet)
    new_x, new_u, new_active, capacity_overflow = _fill_incoming_particles(
        stay_x,
        stay_u,
        stay_active,
        incoming_x,
        incoming_u,
        incoming_active,
    )
    overflow = packet_overflow | capacity_overflow | jnp.any(invalid)
    return new_x, new_u, new_active, overflow


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


def _apply_local_particle_boundaries(local_x, local_u, local_active, static_parameters, dynamic_parameters):
    """
    Apply global particle boundary conditions before staged owner migration.
    """

    particle_bc = static_parameters.particle_boundary_conditions
    bounded_x = local_x
    bounded_u = local_u
    bounded_active = local_active
    (x_bounds, y_bounds, z_bounds) = grid_domain_bounds(dynamic_parameters)

    x1, u1, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 0],
        bounded_u[..., 0],
        bounded_active,
        x_bounds[0],
        x_bounds[1],
        particle_bc[0],
    )
    x2, u2, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 1],
        bounded_u[..., 1],
        bounded_active,
        y_bounds[0],
        y_bounds[1],
        particle_bc[1],
    )
    x3, u3, bounded_active = _apply_tiled_axis_boundary(
        bounded_x[..., 2],
        bounded_u[..., 2],
        bounded_active,
        z_bounds[0],
        z_bounds[1],
        particle_bc[2],
    )

    bounded_x = bounded_x.at[..., 0].set(x1)
    bounded_x = bounded_x.at[..., 1].set(x2)
    bounded_x = bounded_x.at[..., 2].set(x3)
    bounded_u = bounded_u.at[..., 0].set(u1)
    bounded_u = bounded_u.at[..., 1].set(u2)
    bounded_u = bounded_u.at[..., 2].set(u3)

    return bounded_x, bounded_u, bounded_active


def _build_distributed_particle_refresher(static_parameters):
    """Build one mapped refresher from static communication topology."""

    mesh = static_parameters.field_mesh
    mesh_shape = tuple(int(width) for width in mesh.devices.shape)
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    particle_boundary_conditions = tuple(int(bc) for bc in static_parameters.particle_boundary_conditions)
    axis_communication = tuple(
        (
            axis,
            tile_shape[axis],
            mesh_shape[axis],
            (
                tile_shape,
                axis,
                2 if mesh_shape[axis] == 2 else 1,
            ),
            _send_negative_permutation(mesh_shape[axis], particle_boundary_conditions[axis]),
            _send_positive_permutation(mesh_shape[axis], particle_boundary_conditions[axis]),
        )
        for axis in range(3)
    )

    def local_refresh(local_x_tiles, local_u_tiles, local_active_tiles, dynamic_parameters):
        local_x = local_x_tiles[0, 0, 0]
        local_u = local_u_tiles[0, 0, 0]
        local_active = local_active_tiles[0, 0, 0]

        bounded_x, bounded_u, bounded_active = _apply_local_particle_boundaries(
            local_x,
            local_u,
            local_active,
            static_parameters,
            dynamic_parameters,
        )

        bounds = grid_domain_bounds(dynamic_parameters)
        cell_widths = (
            dynamic_parameters.dx,
            dynamic_parameters.dy,
            dynamic_parameters.dz,
        )
        x, u, active = bounded_x, bounded_u, bounded_active
        overflow = jnp.asarray(False)
        for (
            axis,
            tile_width,
            tile_count,
            packet_layout,
            negative_permutation,
            positive_permutation,
        ) in axis_communication:
            x, u, active, axis_overflow = _exchange_particle_axis(
                x,
                u,
                active,
                axis=axis,
                axis_min=bounds[axis][0],
                cell_width=cell_widths[axis],
                tile_width=tile_width,
                tile_count=tile_count,
                packet_layout=packet_layout,
                negative_permutation=negative_permutation,
                positive_permutation=positive_permutation,
            )
            overflow = overflow | axis_overflow

        overflow = jax.lax.pmax(overflow, MESH_AXES)

        return (
            x[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            u[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            active[jnp.newaxis, jnp.newaxis, jnp.newaxis],
            overflow,
        )

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

    def refresh(tiled_particles, dynamic_parameters):
        _validate_particle_tile_topology(tiled_particles, mesh)
        x, u, active, overflow = mapped_refresh(
            tiled_particles.x,
            tiled_particles.u,
            tiled_particles.active,
            dynamic_parameters,
        )
        return TiledParticles(x=x, u=u, active=active), overflow

    return refresh


@functools.lru_cache(maxsize=16)
def _cached_distributed_particle_refresher(static_parameters):
    return _build_distributed_particle_refresher(static_parameters)


def make_distributed_particle_refresher(static_parameters):
    try:
        hash(static_parameters)
    except TypeError:
        return _build_distributed_particle_refresher(static_parameters)
    return _cached_distributed_particle_refresher(static_parameters)


def refresh_tiled_particle_tiles(tiled_particles, static_parameters, dynamic_parameters):
    """
    Move active particles into their owning tiles while preserving static shape.

    The refresh assumes particles move by at most one cell in a timestep, so each
    particle either stays in its current tile or moves to an adjacent tile.  It
    compacts actual movers into fixed-capacity face packets and routes them in
    x/y/z stages.  A corner particle can therefore traverse two or three faces
    without creating separate edge and corner communication streams.  Particles
    that exceed a face-packet or destination capacity, or that require a
    non-adjacent jump, are dropped and reported through the overflow flag.
    """

    refresher = make_distributed_particle_refresher(static_parameters)
    return refresher(tiled_particles, dynamic_parameters)
