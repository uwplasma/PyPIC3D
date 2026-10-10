import jax
from typing import NamedTuple


class SpeciesConfig(NamedTuple):
    """
    Per-species particle metadata shared by all slots in the species block.
    """

    charge: jax.Array # (species,)
    mass: jax.Array # (species,)
    weight: jax.Array # (species,)
    update_x: jax.Array # (species, 3)


class TiledParticles(NamedTuple):
    """
    Tile-major particle storage on the shared tiled Yee mesh.

    The leading ``(ntx, nty, ntz)`` axes match the field-tile layout.  Each
    tile/species block has a fixed slot capacity chosen during initialization;
    inactive slots remain present so the tiled arrays keep a static shape.

    ``x`` and ``u`` are staggered in time.  Every solver runs a leapfrog, so
    ``x`` holds ``x^n`` while ``u`` holds ``u^{n-1/2}``, half a step behind.
    ``PyPIC3D.pusher.particle_push.seed_leapfrog_velocity`` puts the configured
    initial velocity onto that half step at startup.

    For the ``static_metric`` solver ``u`` is the covariant spatial
    four-velocity ``u_i``, not a three-velocity, and it carries the index
    placement of the chart in use.
    """

    x: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # contravariant positions x^i at integer time steps
    u: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # velocities at half time steps, u^{n-1/2}; covariant u_i for static_metric
    active: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # boolean array indicating whether the particle is active or not
