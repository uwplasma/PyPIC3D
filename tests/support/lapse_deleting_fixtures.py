"""A supplied affine lapse with a known deletion surface at x=0."""

import jax.numpy as jnp

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.relativity.core import build_yee_metric
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters


def lapse_runtime(*, tile_width=8, scheme="esirkepov", boundary=0,
                  lapse_slope=-0.01, dt=0.2):
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=8.0, tile_shape=(tile_width, 1, 1), dt=dt,
        solver="static_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition=scheme,
        particle_boundary_conditions=(boundary, 0, 0), particle_batch_size=4,
    )
    static = static._replace(lapse_deleting=True, lapse_threshold=0.1)

    def metric_at_position(position):
        lapse = 0.1 + lapse_slope * position[0]
        return lapse, jnp.zeros(3), jnp.eye(3), jnp.eye(3), jnp.asarray(1.0)

    metric = build_yee_metric(dynamic, metric_at_position)
    zero = empty_tiled_vector(static, dynamic)
    fields = (zero, zero, zero, zero[0], zero[0], (zero, zero), metric,
              (zero, zero), jnp.asarray(False))
    species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1),
                            jnp.ones((1, 3), dtype=bool))
    return static, dynamic, fields, species


def lapse_particles(static, positions, momenta, *, owner_positions=None):
    """Place one species in its original tiles, including pre-migration states."""
    if owner_positions is None:
        owner_positions = positions
    tile_count = 8 // static.tile_shape[0]
    shape = (tile_count, 1, 1, 1, len(positions))
    x = jnp.zeros(shape + (3,))
    u = jnp.zeros_like(x)
    active = jnp.zeros(shape, dtype=bool)
    for slot, (position, momentum, owner) in enumerate(zip(positions, momenta, owner_positions)):
        tile = min(max(int((owner + 4.0) // static.tile_shape[0]), 0), tile_count - 1)
        index = (tile, 0, 0, 0, slot)
        x = x.at[index + (0,)].set(position)
        u = u.at[index + (0,)].set(momentum)
        active = active.at[index].set(True)
    return shard_tiled_particles(TiledParticles(x, u, active), static)


def active_particle_rows(particles):
    active = particles.active.reshape(-1)
    positions = particles.x.reshape(-1, 3)[active]
    momenta = particles.u.reshape(-1, 3)[active]
    order = jnp.argsort(positions[:, 0])
    return positions[order], momenta[order]


def interior_line(component, static):
    """Concatenate physical x nodes from the one-dimensional tiled grid."""
    guard = static.guard_cells
    return component[:, 0, 0, guard:-guard, guard, guard].reshape(-1)
