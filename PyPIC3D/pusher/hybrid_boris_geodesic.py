from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.particles.particle_batching import (
    prepare_particle_batches, particle_batch_indices, number_of_particle_batches,
)
from PyPIC3D.pusher.boris import interpolate_field_to_particles
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    covariant_lorentz_factor,
    location_grid,
    lower_vector,
)
from PyPIC3D.relativity.interpolate_metric import (
    interpolate_metric,
    particle_active_axes,
    safe_inactive_positions,
)


def coordinate_velocity(u_cov, metric):
    """
    Coordinate velocity dx^i/dt = alpha gamma^ij u_j / Gamma - beta^i.
    """

    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    u_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_cov)
    return metric.lapse[..., jnp.newaxis] * u_con / Gamma[..., jnp.newaxis] - metric.shift


def geodesic_acceleration(u_cov, metric):
    """
    Geodesic source du_i/dt = -Gamma d_i alpha + u_j d_i beta^j
    - alpha/(2 Gamma) u_l u_m d_i gamma^lm.
    """

    Gamma = covariant_lorentz_factor(u_cov, metric.gamma_inv)
    lapse_force = -Gamma[..., jnp.newaxis] * metric.grad_lapse
    shift_force = jnp.einsum("...j,...ji->...i", u_cov, metric.grad_shift)
    metric_force = (-0.5 * metric.lapse / Gamma)[..., jnp.newaxis] * jnp.einsum(
        "...l,...m,...ilm->...i", u_cov, u_cov, metric.grad_gamma_inv
    )
    return lapse_force + shift_force + metric_force


def magnetic_boris_rotation(u_minus, B_con, metric, q_over_m, dt):
    """
    Boris rotation of covariant u_i about the contravariant magnetic field B^i.
    """

    Gamma_minus = covariant_lorentz_factor(u_minus, metric.gamma_inv)
    u0_bar = Gamma_minus / metric.lapse
    t_con = (q_over_m * dt / (2.0 * u0_bar))[..., jnp.newaxis] * B_con
    t_cov = lower_vector(t_con, metric.gamma)
    t_norm = jnp.einsum("...i,...i->...", t_con, t_cov)
    s_con = 2.0 * t_con / (1.0 + t_norm)[..., jnp.newaxis]

    sqrt_gamma = metric.sqrt_gamma[..., jnp.newaxis]
    u_minus_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_minus)
    u_prime = u_minus + sqrt_gamma * jnp.cross(u_minus_con, t_con)
    u_prime_con = jnp.einsum("...ij,...j->...i", metric.gamma_inv, u_prime)
    return u_minus + sqrt_gamma * jnp.cross(u_prime_con, s_con)


def gather_vector(field, locations, position, center_grid, vertex_grid,
                  shape_factor, active_axes, inactive_axis_indices):
    """
    Gather a staggered tile-local vector field to ``position[..., 3]``.

    ``field[i]`` lives on the Yee location ``locations[i]``; the grids are the
    cell-centered and vertex axes of the same tile.
    """

    x, y, z = (position[..., axis].reshape(-1) for axis in range(3))
    components = [
        interpolate_field_to_particles(
            field[i], x, y, z,
            location_grid(center_grid, vertex_grid, locations[i]),
            shape_factor,
            ghost_cells=True,
            active_axes=active_axes,
            inactive_axis_indices=inactive_axis_indices,
        )
        for i in range(3)
    ]
    return jnp.stack(components, axis=-1).reshape(position.shape)


class TileFields(NamedTuple):
    """One tile's coordinate axes, cell-centered metric and physical D^i/B^i."""

    center_grid: tuple
    vertex_grid: tuple
    metric: object
    D: tuple
    B: tuple


def tile_fields(tx, ty, tz, metric, D_tiles, B_tiles, dynamic_parameters):
    """
    Slice the tiled grids, metric and native field densities down to one tile.

    ``D_tiles``/``B_tiles`` are native densities ``sqrt(gamma) V^i``; each
    component is divided by its own grid-node ``sqrt_gamma`` here, so the
    returned tile carries the physical contravariant field the particles see.
    """

    return TileFields(
        center_grid=tuple(axis[tx, ty, tz] for axis in dynamic_parameters.grids.tiled_center_grid),
        vertex_grid=tuple(axis[tx, ty, tz] for axis in dynamic_parameters.grids.tiled_vertex_grid),
        metric=jax.tree.map(lambda array: array[tx, ty, tz], metric.center),
        D=tuple(component[tx, ty, tz] / sample.sqrt_gamma[tx, ty, tz]
                for component, sample in zip(D_tiles, metric.D)),
        B=tuple(component[tx, ty, tz] / sample.sqrt_gamma[tx, ty, tz]
                for component, sample in zip(B_tiles, metric.B)),
    )


@partial(jax.jit, static_argnames=("static_parameters", "advance_positions"))
def hybrid_boris_geodesic_push(
    particles,
    species_config,
    D_tiles,
    B_tiles,
    metric,
    static_parameters,
    dynamic_parameters,
    *,
    advance_positions=True,
):
    """
    Explicit Strang-split second-order 3+1 particle push.

    Particle positions are contravariant coordinates.  ``particles.u`` stores
    covariant spatial velocity components ``u_i``.  ``D_tiles``/``B_tiles``
    are native densities ``sqrt(gamma) V^i`` (including any external field);
    each component is divided by its own grid-node ``sqrt_gamma`` before the
    gather, so particles see the physical contravariant field.

    This is a staggered leapfrog.  The incoming ``particles.u`` is
    ``u^{n-1/2}``; the velocity operator ``EM(dt/2) . geodesic(dt) . EM(dt/2)``
    is applied with every field and metric quantity sampled at ``x^n``, giving
    ``u^{n+1/2}``, and the position is only advanced afterwards with a
    midpoint rule.  A run must therefore start from ``u^{-1/2}``, which
    :func:`PyPIC3D.pusher.particle_push.seed_leapfrog_velocity` provides;
    starting from the physical ``u(0)`` costs a full order of accuracy.

    Returns the full-step particles ``(x^{n+1}, u^{n+1/2})`` and the centred
    particles ``(x^{n+1/2}, u^{n+1/2})`` used for the current deposition.
    ``advance_positions=False`` applies only the momentum operator, for
    initializing and synchronizing leapfrog momenta in a changing metric.
    """

    g = int(static_parameters.guard_cells)
    dt = dynamic_parameters.dt
    shape_factor = static_parameters.shape_factor
    metric_name = static_parameters.metric
    ntx, nty, ntz = particles.active.shape[:3]
    active_axes = particle_active_axes((ntx, nty, ntz), static_parameters.tile_shape)
    inactive_axis_indices = (g, g, g)
    species_q_over_m = species_config.charge / species_config.mass

    def sample_metric(tile, position, derivatives=True):
        """Hermite metric sample at one particle position in this tile."""

        return interpolate_metric(
            tile.metric, position, tile.center_grid, metric_name,
            active_axes, inactive_axis_indices, derivatives=derivatives,
        )

    def gather(field, locations, tile, position):
        """Gather one staggered tile-local vector field to a particle position."""

        return gather_vector(field, locations, position, tile.center_grid, tile.vertex_grid,
                             shape_factor, active_axes, inactive_axis_indices)

    def geodesic_step(x_n, u_old, active, update_x, q_over_m, tile):
        """
        Advance one particle from ``(x^n, u^{n-1/2})`` to ``(x^{n+1}, u^{n+1/2})``.

        ``x_n``, ``u_old`` and ``update_x`` are single vectors; ``active`` and
        ``q_over_m`` are scalars.  Also returns the centred ``x^{n+1/2}``.
        """

        # update_x freezes individual velocity and coordinate components per species
        moving = active & update_x
        x_n = safe_inactive_positions(x_n, active, tile.center_grid, active_axes, g)
        u_old = jnp.where(active, u_old, 0.0)

        # Every force in the velocity update is sampled once, at x^n.
        metric_n = sample_metric(tile, x_n)
        D_con = gather(tile.D, D_FIELD_LOCATIONS, tile, x_n)
        B_con = gather(tile.B, B_FIELD_LOCATIONS, tile, x_n)
        E_cov = lower_vector(D_con, metric_n.gamma)
        # EM(dt/2) is a quarter-step electric kick, a half-step rotation, and a quarter-step kick.
        electric_kick = q_over_m * dt / 4.0 * metric_n.lapse * E_cov

        def electromagnetic_half_step(u):
            u = u + electric_kick
            u = magnetic_boris_rotation(u, B_con, metric_n, q_over_m, dt / 2.0)
            return u + electric_kick

        u_after_first_em = jnp.where(moving, electromagnetic_half_step(u_old), u_old)

        # midpoint rule for the geodesic source over the full step
        u_geo_mid = u_after_first_em + 0.5 * dt * geodesic_acceleration(u_after_first_em, metric_n)
        u_after_geodesic = u_after_first_em + dt * geodesic_acceleration(u_geo_mid, metric_n)
        u_after_geodesic = jnp.where(moving, u_after_geodesic, u_old)

        u_new = jnp.where(moving, electromagnetic_half_step(u_after_geodesic), u_old)

        # midpoint rule for the position; x^{n+1/2} is also the deposition position
        if advance_positions:
            x_half = jnp.where(moving, x_n + 0.5 * dt * coordinate_velocity(u_new, metric_n), x_n)
            metric_half = sample_metric(tile, x_half, derivatives=False)
            x_new = jnp.where(moving, x_n + dt * coordinate_velocity(u_new, metric_half), x_n)
        else:
            x_half = x_n
            x_new = x_n

        return x_new, u_new, x_half

    push_particles = jax.vmap(geodesic_step, in_axes=(0, 0, 0, 0, 0, None))

    def push_one_tile(tx, ty, tz, x_tile, u_tile, active_tile):
        particle_capacity, batch_size, active_indices, n_active = prepare_particle_batches(
            active_tile,
            static_parameters.particle_batch_size,
        )
        if particle_capacity == 0:
            return x_tile, u_tile, x_tile

        slots_per_species = active_tile.shape[-1]

        x_flat = x_tile.reshape(-1, 3)
        u_flat = u_tile.reshape(-1, 3)

        tile = tile_fields(tx, ty, tz, metric, D_tiles, B_tiles, dynamic_parameters)

        n_batches = number_of_particle_batches(n_active, batch_size)

        def push_batch(batch_state):
            # Each active particle appears in exactly one batch, so every batch
            # reads the tile's original slots rather than the loop carry.
            batch_index, current_x, current_u, current_half = batch_state
            particle_indices, valid = particle_batch_indices(
                active_indices,
                n_active,
                batch_index,
                batch_size,
            )

            species_indices = particle_indices // slots_per_species

            new_x, new_u, mid_x = push_particles(
                x_flat[particle_indices],
                u_flat[particle_indices],
                valid,
                species_config.update_x[species_indices],
                species_q_over_m[species_indices],
                tile,
            )

            # Padded lanes repeat a live source index, so their writes are dropped.
            destination = jnp.where(valid, particle_indices, particle_capacity)

            return (batch_index + 1,
                    current_x.at[destination].set(new_x, mode="drop"),
                    current_u.at[destination].set(new_u, mode="drop"),
                    current_half.at[destination].set(mid_x, mode="drop"))

        def batches_remaining(batch_state):
            batch_index, _, _, _ = batch_state
            return batch_index < n_batches

        _, new_x, new_u, new_half = jax.lax.while_loop(
            batches_remaining,
            push_batch,
            (jnp.asarray(0), x_flat, u_flat, x_flat),
        )

        return (new_x.reshape(x_tile.shape),
                new_u.reshape(u_tile.shape),
                new_half.reshape(x_tile.shape))

    push_tiles = push_one_tile
    tx = jnp.arange(ntx)
    ty = jnp.arange(nty)
    tz = jnp.arange(ntz)

    push_tiles = jax.vmap(push_tiles, in_axes=(None, None, 0, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(None, 0, None, 0, 0, 0), out_axes=0)
    push_tiles = jax.vmap(push_tiles, in_axes=(0, None, None, 0, 0, 0), out_axes=0)

    x_new, u_new, x_half = push_tiles(
        tx, ty, tz,
        particles.x,
        particles.u,
        particles.active,
    )

    full_step = TiledParticles(x=x_new, u=u_new, active=particles.active)
    half_step = TiledParticles(x=x_half, u=u_new, active=particles.active)
    return full_step, half_step
