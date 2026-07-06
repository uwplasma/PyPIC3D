import jax.numpy as jnp
from typing import NamedTuple

from PyPIC3D.particles.particle_class import TiledParticles


class ParticleOutputRecord(NamedTuple):
    name: str
    species_index: int
    x: jnp.ndarray
    x_diagnostic: jnp.ndarray
    u: jnp.ndarray
    charge: jnp.ndarray
    mass: jnp.ndarray
    weight: jnp.ndarray


def _is_tiled_scalar(field):
    return getattr(field, "ndim", 0) == 6


def _is_tiled_vector(field):
    return (
        isinstance(field, (list, tuple))
        and len(field) == 3
        and _is_tiled_scalar(field[0])
    )


def _tile_shape_from_world(world):
    return tuple(int(width) for width in world["tile_shape"])


def _guard_depth_from_world(world):
    return int(world["guard_cells"])


def assemble_tiled_scalar_field(field_tiles, world, tile_shape, num_guard_cells=2):
    """
    Assemble compact field tiles back into one global ghost-celled field.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    ntx, nty, ntz = field_tiles.shape[:3]
    Nx = int(ntx) * tile_nx
    Ny = int(nty) * tile_ny
    Nz = int(ntz) * tile_nz

    field = jnp.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=field_tiles.dtype)

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                tile_with_one_guard = field_tiles[
                    tx,
                    ty,
                    tz,
                    g - 1:g + tile_nx + 1,
                    g - 1:g + tile_ny + 1,
                    g - 1:g + tile_nz + 1,
                ]
                ix = tx * tile_nx
                iy = ty * tile_ny
                iz = tz * tile_nz
                field = field.at[ix:ix + tile_nx + 2, iy:iy + tile_ny + 2, iz:iz + tile_nz + 2].set(tile_with_one_guard)

    return field


def assemble_tiled_vector_field(field_tiles, world, tile_shape, num_guard_cells=2):
    """
    Assemble tiled vector-field components into ordinary ghost-celled arrays.
    """

    return tuple(assemble_tiled_scalar_field(component, world, tile_shape, num_guard_cells) for component in field_tiles)


def scalar_field_for_output(field, world):
    """
    Return an ordinary ghost-celled scalar field for file formats.

    Runtime diagnostics can operate on tile-major arrays, but openPMD
    mesh writers still expect one global ghost-celled array.
    """

    if not _is_tiled_scalar(field):
        return field

    tile_shape = _tile_shape_from_world(world)
    g = _guard_depth_from_world(world)
    return assemble_tiled_scalar_field(field, world, tile_shape, num_guard_cells=g)


def vector_field_for_output(field, world):
    """
    Return ordinary ghost-celled vector components for file formats.
    """

    if not _is_tiled_vector(field):
        return field

    tile_shape = _tile_shape_from_world(world)
    g = _guard_depth_from_world(world)
    return assemble_tiled_vector_field(field, world, tile_shape, num_guard_cells=g)


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


def _axis_diagnostic_position(x, u, dt, wind, bc):
    x_diagnostic = x - u * dt / 2

    if int(jnp.asarray(bc).item()) == 0:
        half_wind = wind / 2
        x_diagnostic = jnp.where(
            x_diagnostic > half_wind,
            x_diagnostic - wind,
            jnp.where(x_diagnostic < -half_wind, x_diagnostic + wind, x_diagnostic),
        )

    return x_diagnostic


def _diagnostic_position(x, u, world):
    if world is None:
        return x

    particle_bc = world.get("particle_boundary_conditions", {})
    dt = world["dt"]
    x_diagnostic = _axis_diagnostic_position(x[:, 0], u[:, 0], dt, world["x_wind"], particle_bc.get("x", 0))
    y_diagnostic = _axis_diagnostic_position(x[:, 1], u[:, 1], dt, world["y_wind"], particle_bc.get("y", 0))
    z_diagnostic = _axis_diagnostic_position(x[:, 2], u[:, 2], dt, world["z_wind"], particle_bc.get("z", 0))

    return jnp.stack((x_diagnostic, y_diagnostic, z_diagnostic), axis=-1)


def particles_for_output(particles, species_config=None, species_names=None, world=None):
    """
    Flatten fixed-capacity tiled particle storage for diagnostics.
    """

    if not isinstance(particles, TiledParticles):
        raise TypeError("Particle output requires TiledParticles runtime storage.")

    n_species = particles.active.shape[3]
    output_particles = []

    for species_index in range(n_species):
        active = particles.active[:, :, :, species_index, :].reshape(-1)

        x = particles.x[:, :, :, species_index, :, :].reshape(-1, 3)[active]
        u = particles.u[:, :, :, species_index, :, :].reshape(-1, 3)[active]
        n_active = int(jnp.sum(active))
        charge = jnp.full((n_active,), species_config.charge[species_index])
        mass = jnp.full((n_active,), species_config.mass[species_index])
        weight = jnp.full((n_active,), species_config.weight[species_index])

        if species_names is None:
            name = f"species_{species_index}"
        else:
            name = species_names[species_index]

        output_particles.append(
            ParticleOutputRecord(
                name=name,
                species_index=species_index,
                x=x,
                x_diagnostic=_diagnostic_position(x, u, world),
                u=u,
                charge=charge,
                mass=mass,
                weight=weight,
            )
        )

    return output_particles
