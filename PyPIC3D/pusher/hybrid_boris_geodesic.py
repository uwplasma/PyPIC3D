from functools import partial

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
    check_particle_samples,
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


@partial(jax.jit, static_argnames="static_parameters")
def hybrid_boris_geodesic_push(
    particles,
    species_config,
    D_tiles,
    B_tiles,
    metric,
    static_parameters,
    dynamic_parameters,
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
    """

    g = int(static_parameters.guard_cells)
    dt = dynamic_parameters.dt
    shape_factor = static_parameters.shape_factor
    metric_name = static_parameters.metric
    ntx, nty, ntz = particles.active.shape[:3]
    active_axes = particle_active_axes((ntx, nty, ntz), static_parameters.tile_shape)
    inactive_axis_indices = (g, g, g)
    q_over_m = species_config.charge / species_config.mass
    cell_size = jnp.array([dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz])

    def push_active_batch(x_n, u_old, active, q_over_m_batch, update_x, tx, ty, tz):
        tile = jnp.array([tx, ty, tz])
        center_grid = tuple(axis[tx, ty, tz] for axis in dynamic_parameters.grids.tiled_center_grid)
        vertex_grid = tuple(axis[tx, ty, tz] for axis in dynamic_parameters.grids.tiled_vertex_grid)
        tile_metric = jax.tree.map(lambda array: array[tx, ty, tz], metric.center)
        D_tile = tuple(component[tx, ty, tz] / sample.sqrt_gamma[tx, ty, tz]
                       for component, sample in zip(D_tiles, metric.D))
        B_tile = tuple(component[tx, ty, tz] / sample.sqrt_gamma[tx, ty, tz]
                       for component, sample in zip(B_tiles, metric.B))

        live = active[..., jnp.newaxis]
        # update_x freezes individual velocity and coordinate components per species
        moving = live & update_x
        x_n = safe_inactive_positions(x_n, active, center_grid, active_axes, g)
        u_old = jnp.where(live, u_old, 0.0)

        # Every force in the velocity update is sampled once, at x^n.
        metric_n = interpolate_metric(
            tile_metric, x_n, center_grid, metric_name, active_axes, inactive_axis_indices
        )
        D_con = gather_vector(D_tile, D_FIELD_LOCATIONS, x_n, center_grid, vertex_grid,
                              shape_factor, active_axes, inactive_axis_indices)
        B_con = gather_vector(B_tile, B_FIELD_LOCATIONS, x_n, center_grid, vertex_grid,
                              shape_factor, active_axes, inactive_axis_indices)
        E_cov = lower_vector(D_con, metric_n.gamma)
        # EM(dt/2) is a quarter-step electric kick, a half-step rotation, and a quarter-step kick.
        electric_kick = (q_over_m_batch * dt / 4.0)[..., jnp.newaxis] * metric_n.lapse[..., jnp.newaxis] * E_cov

        def electromagnetic_half_step(u):
            u = u + electric_kick
            u = magnetic_boris_rotation(u, B_con, metric_n, q_over_m_batch, dt / 2.0)
            return u + electric_kick

        u_after_first_em = jnp.where(moving, electromagnetic_half_step(u_old), u_old)

        # midpoint rule for the geodesic source over the full step
        u_geo_mid = u_after_first_em + 0.5 * dt * geodesic_acceleration(u_after_first_em, metric_n)
        u_after_geodesic = u_after_first_em + dt * geodesic_acceleration(u_geo_mid, metric_n)
        u_after_geodesic = jnp.where(moving, u_after_geodesic, u_old)

        u_new = jnp.where(moving, electromagnetic_half_step(u_after_geodesic), u_old)

        # midpoint rule for the position; x^{n+1/2} is also the deposition position
        x_half = jnp.where(moving, x_n + 0.5 * dt * coordinate_velocity(u_new, metric_n), x_n)
        metric_half = interpolate_metric(
            tile_metric, x_half, center_grid, metric_name, active_axes, inactive_axis_indices,
            derivatives=False,
        )
        x_new = jnp.where(moving, x_n + dt * coordinate_velocity(u_new, metric_half), x_n)

        for stage, value in (("first magnetic half-step", u_after_first_em),
                             ("geodesic midpoint", u_geo_mid),
                             ("geodesic full-step", u_after_geodesic),
                             ("second magnetic half-step", u_new),
                             ("position midpoint", x_half),
                             ("position full-step", x_new)):
            check_particle_samples(~active | jnp.all(jnp.isfinite(value), axis=-1), x_n, stage, tile)
        displacement = jnp.abs(x_new - x_n) / cell_size
        within_one_cell = jnp.all(jnp.where(jnp.array(active_axes), displacement <= 1, True), axis=-1)
        check_particle_samples(~active | within_one_cell, x_n, "one-cell displacement bound", tile)

        return x_new, u_new, x_half

    capacity = particles.active.shape[-2] * particles.active.shape[-1]
    if capacity == 0:
        return particles, particles
    batch_size = min(int(static_parameters.particle_batch_size), capacity)
    slots = particles.active.shape[-1]
    tile_shape = particles.active.shape[:3]
    x_flat = particles.x.reshape(tile_shape + (capacity, 3))
    u_flat = particles.u.reshape(tile_shape + (capacity, 3))

    def map_tiles(function):
        if tile_shape == (1, 1, 1):
            # Keep the direct single-tile dispatch used by active-only batching.
            def one_tile(*args):
                result = function(*(value[0, 0, 0] for value in args))
                return jax.tree.map(lambda value: value[None, None, None], result)
            return one_tile
        for _ in range(3):
            function = jax.vmap(function)
        return function

    def prepare(active):
        _, _, indices, count = prepare_particle_batches(active, static_parameters.particle_batch_size)
        return indices, count

    indices, counts = map_tiles(prepare)(particles.active)
    n_batches = jnp.max(number_of_particle_batches(counts, batch_size))
    tx, ty, tz = jnp.meshgrid(jnp.arange(ntx), jnp.arange(nty), jnp.arange(ntz), indexing="ij")

    def push_selected(index_array, count, x, u, tx, ty, tz, batch_index):
        selected, valid = particle_batch_indices(index_array, count, batch_index, batch_size)
        species = selected // slots
        new_x, new_u, mid_x = push_active_batch(
            x[selected], u[selected], valid, q_over_m[species],
            species_config.update_x[species], tx, ty, tz,
        )
        return jnp.where(valid, selected, capacity), new_x, new_u, mid_x

    def write_tile(destination, values, original):
        # Padded lanes must not overwrite their repeated live source index.
        return original.at[destination].set(values, mode="drop")

    def advance_batch(state):
        batch_index, full_x, full_u, half_x = state
        destination, new_x, new_u, mid_x = map_tiles(push_selected)(
            indices, counts, x_flat, u_flat, tx, ty, tz,
            jnp.broadcast_to(batch_index, tile_shape),
        )
        return (batch_index + 1,
                map_tiles(write_tile)(destination, new_x, full_x),
                map_tiles(write_tile)(destination, new_u, full_u),
                map_tiles(write_tile)(destination, mid_x, half_x))

    # Keep the dynamic loop outside vmap: checkify rejects vmap-of-while.
    # Tiles share the maximum active batch count; padded lanes use safe metrics.
    _, x_new, u_new, x_half = jax.lax.while_loop(
        lambda state: state[0] < n_batches,
        advance_batch, (jnp.asarray(0), x_flat, u_flat, x_flat),
    )
    x_new = x_new.reshape(particles.x.shape)
    u_new = u_new.reshape(particles.u.shape)
    x_half = x_half.reshape(particles.x.shape)

    full_step = TiledParticles(x=x_new, u=u_new, active=particles.active)
    half_step = TiledParticles(x=x_half, u=u_new, active=particles.active)
    return full_step, half_step
