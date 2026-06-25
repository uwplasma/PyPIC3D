import jax
import jax.numpy as jnp
from typing import NamedTuple

class TiledParticles(NamedTuple):
    x: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # positions
    u: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile, 3)
    # velocities
    charge: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    mass: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # charge and mass of the particles
    weight: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # macroparticle weight of the particles
    active: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # boolean array indicating whether the particle is active or not
    update_x1: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    update_x2: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    update_x3: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    update_u1: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    update_u2: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    update_u3: jax.Array # (ntx, nty, ntz, species, max_particles_per_tile)
    # boolean array indicating whether the particle's position or velocity should be updated or not