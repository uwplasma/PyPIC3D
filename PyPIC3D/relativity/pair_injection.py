"""Neutral pair insertion into fixed-capacity storage for static-metric PIC.

Source models supply tile-owned candidate positions and orthonormal spatial
four-velocities at the birth time. All field inputs are native D/B densities.
"""
import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.interpolate_metric import (
    interpolate_metric, particle_active_axes, safe_inactive_positions,
)


def birth_covariant_momentum(momentum, positions, active, metric, grid, static,
                             active_axes):
    """Use the production metric reconstruction, safely masking unused lanes."""
    g = static.guard_cells
    evaluation = safe_inactive_positions(positions, active, grid, active_axes, g)
    sampled = interpolate_metric(
        metric, evaluation, grid, static.metric, active_axes, (g, g, g),
        derivatives=False)
    L = jnp.linalg.cholesky(sampled.gamma)
    covariant = jnp.einsum("...ij,...j->...i", L, momentum)
    return jnp.where(active[..., None], covariant, 0.)


def insert_pairs(particles, species, positions, momenta, requested, D, B, metric,
                 static, dynamic, species_indices=(0, 1)):
    """Insert complete neutral pairs and seed only their backward half step.

    positions/momenta have shape (ntx, nty, ntz, candidates, 3); requested has
    shape (ntx, nty, ntz). Candidates are ordered by priority within each tile.
    The source supplies tile-owned positions and physical orthonormal u at birth.
    Both partners receive the same position and momentum before charge-dependent
    leapfrog seeding. Other species and existing particles are unchanged.

    Equal positive masses/weights and opposite nonzero charges are required.
    Incompatible species reject every requested pair, including under JIT.
    Returns (particles, (requested, inserted)), with per-tile int32 pair counts.
    The caller must fail on the host if any requested - inserted counts are nonzero.
    Counts include requests beyond both candidate and particle storage capacity.
    """
    if static.solver != "static_metric":
        raise ValueError("Pair injection requires solver='static_metric'")
    s0, s1 = species_indices
    if s0 == s1 or min(s0, s1) < 0 or max(s0, s1) >= particles.active.shape[-2]:
        raise ValueError("Pair injection requires two distinct existing species")
    tile_shape = particles.active.shape[:3]
    if (positions.shape != momenta.shape or positions.shape[:3] != tile_shape
            or positions.ndim != 5 or positions.shape[-1] != 3
            or requested.shape != tile_shape):
        raise ValueError("Pair candidates and requested counts must match the particle tiles")

    slots = particles.active.shape[-1]
    capacity = min(slots, positions.shape[-2])
    positions, momenta = positions[..., :capacity, :], momenta[..., :capacity, :]
    requested = jnp.asarray(requested, dtype=jnp.int32)
    compatible = ((species.charge[s0] == -species.charge[s1])
                  & (species.charge[s0] != 0)
                  & (species.mass[s0] == species.mass[s1]) & (species.mass[s0] > 0)
                  & (species.weight[s0] == species.weight[s1]) & (species.weight[s0] > 0))
    # The species need to be the anti-particles of each other, with equal mass and weight.
    
    active_axes = particle_active_axes(tile_shape, static.tile_shape)
    ranks = jnp.arange(capacity)

    def one_tile(x, u, active, positions, momenta, requested, primitive_metric, grid):
        free = jnp.sum(~active, axis=-1, dtype=jnp.int32)
        inserted = jnp.minimum(jnp.minimum(free[s0], free[s1]),
                               jnp.minimum(requested, capacity))
        inserted = jnp.where(compatible, inserted, 0)
        covariant = birth_covariant_momentum(
            momenta, positions, ranks < inserted, primitive_metric, grid, static,
            active_axes)
        for s in species_indices:
            indices = jnp.nonzero(~active[s], size=capacity, fill_value=slots)[0]
            indices = jnp.where(ranks < inserted, indices, slots)
            x = x.at[s, indices].set(positions, mode="drop")
            u = u.at[s, indices].set(covariant, mode="drop")
            active = active.at[s, indices].set(True, mode="drop")
        return x, u, active, inserted

    # Nested maps retain all three tile axes and their existing sharding.
    mapped = jax.vmap(jax.vmap(jax.vmap(one_tile)))
    x, u, active, inserted = mapped(
        particles.x, particles.u, particles.active, positions, momenta, requested,
        metric.center, dynamic.grids.tiled_center_grid)
    # Insertion only activates slots; identify births before any push or migration.
    newborn = active & ~particles.active
    result = TiledParticles(x, u, active)
    seeded = seed_leapfrog_velocity(
        result._replace(active=newborn), species, D, B, static, dynamic, metric)
    result = result._replace(u=jnp.where(newborn[..., None], seeded.u, result.u))
    return result, (requested, inserted)
