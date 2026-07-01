from PyPIC3D.deposition.direct_deposition_tiled import _direct_J_from_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles


def J_from_rhov(
    particles,
    J,
    constants,
    world,
    grid=None,
    filter="bilinear",
    species_config=None,
    tile_shape=None,
    g=None,
):
    """Compute current density from velocity-weighted particle charge."""

    if isinstance(particles, TiledParticles):
        if species_config is None:
            species_config = J
            J = constants
            constants = world
            world = grid
            grid = None
        # Tiled runtime callers pass ``species_config`` as the second positional
        # argument so the public current name can be used without global current
        # assembly in the deposition path.

        if tile_shape is None:
            tile_shape = tuple(int(width) for width in world["tile_shape"])
        if g is None:
            g = int(world["guard_cells"])

        return _direct_J_from_tiled_particles(
            particles,
            species_config,
            J,
            constants,
            world,
            grid=grid,
            filter=filter,
            tile_shape=tile_shape,
            g=int(g),
        )

    raise ValueError("Public J_from_rhov requires TiledParticles.")
