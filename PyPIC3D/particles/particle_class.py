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
    update_u: jax.Array # (species, 3)


class TiledParticles(NamedTuple):
    """
    Tile-major particle storage on the shared tiled Yee mesh.

    The leading ``(ntx, nty, ntz)`` axes match the field-tile layout.  Each
    tile/species block has a fixed slot capacity chosen during initialization;
    inactive slots remain present so the tiled arrays keep a static shape.
    """

    x: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # positions
    u: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # velocities
    active: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # boolean array indicating whether the particle is active or not
