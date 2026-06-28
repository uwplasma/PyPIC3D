from PyPIC3D.particles.tiled_particle_diagnostics import flatten_tiled_particles_by_species
from PyPIC3D.solvers.yee_tiled import assemble_tiled_scalar_field, assemble_tiled_vector_field


def _is_tiled_scalar(field):
    return getattr(field, "ndim", 0) == 6


def _is_tiled_vector(field):
    return (
        isinstance(field, (list, tuple))
        and len(field) == 3
        and _is_tiled_scalar(field[0])
    )


def _tile_shape_from_field(field):
    return tuple(int(width) - 2 for width in field.shape[3:])


def _tile_shape_from_world_or_field(world, field):
    if "tile_shape" in world:
        return tuple(int(width) for width in world["tile_shape"])
    return _tile_shape_from_field(field)


def _guard_depth_from_field(field, tile_shape):
    guard_depths = tuple((int(local_width) - int(tile_width)) // 2 for local_width, tile_width in zip(field.shape[3:], tile_shape))
    if any(int(local_width) != int(tile_width) + 2 * guard_depth for local_width, tile_width, guard_depth in zip(field.shape[3:], tile_shape, guard_depths)):
        raise ValueError("Tiled field local shape is incompatible with world['tile_shape'].")
    if guard_depths[0] != guard_depths[1] or guard_depths[1] != guard_depths[2]:
        raise ValueError("Output assembly requires the same guard depth on each local field axis.")
    return guard_depths[0]


def scalar_field_for_output(field, world):
    """
    Return an ordinary ghost-celled scalar field for file formats.

    Runtime diagnostics can operate on tile-major arrays, but openPMD and VTK
    mesh writers still expect one global ghost-celled array.
    """

    if not _is_tiled_scalar(field):
        return field

    tile_shape = _tile_shape_from_world_or_field(world, field)
    return assemble_tiled_scalar_field(field, world, tile_shape)


def vector_field_for_output(field, world):
    """
    Return ordinary ghost-celled vector components for file formats.
    """

    if not _is_tiled_vector(field):
        return field

    tile_shape = _tile_shape_from_world_or_field(world, field[0])
    num_guard_cells = _guard_depth_from_field(field[0], tile_shape)
    return assemble_tiled_vector_field(field, world, tile_shape, num_guard_cells=num_guard_cells)


def fields_for_output(fields, world):
    """
    Assemble tile-major fields at the I/O boundary.

    The live solver state is left untouched.  The particle-retile overflow flag
    is a Python-driver diagnostic, not a physical field, so it is not included
    in the returned output tuple.
    """

    E, B, J, rho, phi, external_fields, *rest = fields
    external_E, external_B = external_fields

    output_fields = (
        vector_field_for_output(E, world),
        vector_field_for_output(B, world),
        vector_field_for_output(J, world),
        scalar_field_for_output(rho, world),
        scalar_field_for_output(phi, world),
        (
            vector_field_for_output(external_E, world),
            vector_field_for_output(external_B, world),
        ),
    )

    if not rest:
        return output_fields

    pml_state = rest[0]
    return output_fields + (pml_state,)


def particles_for_output(particles, species_names=None, world=None):
    """
    Flatten fixed-capacity tiled particle storage for diagnostics.
    """

    return flatten_tiled_particles_by_species(
        particles,
        species_names=species_names,
        world=world,
    )
