from typing import NamedTuple

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.fluid_quantities import compute_velocity_field
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.grids import grid_domain_bounds


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


def _tile_shape_from_static_parameters(static_parameters):
    return tuple(int(width) for width in static_parameters.tile_shape)


def _guard_depth_from_static_parameters(static_parameters):
    return int(static_parameters.guard_cells)


def assemble_tiled_scalar_field(field_tiles, static_parameters, tile_shape, num_guard_cells=2):
    """
    Assemble compact field tiles back into one global ghost-celled field.

    This is a diagnostic/output boundary.  Distributed runtime fields may be
    sharded across devices, so gather them here before constructing the
    ordinary global array expected by tests and file writers.
    """

    field_tiles = jnp.asarray(jax.device_get(field_tiles))
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


def assemble_tiled_vector_field(field_tiles, static_parameters, tile_shape, num_guard_cells=2):
    """
    Assemble tiled vector-field components into ordinary ghost-celled arrays.
    """

    return tuple(assemble_tiled_scalar_field(component, static_parameters, tile_shape, num_guard_cells) for component in field_tiles)


def scalar_field_for_output(field, static_parameters):
    """
    Return an ordinary ghost-celled scalar field for file formats.

    Runtime diagnostics can operate on tile-major arrays, but openPMD
    mesh writers still expect one global ghost-celled array.
    """

    if not _is_tiled_scalar(field):
        return field

    tile_shape = _tile_shape_from_static_parameters(static_parameters)
    g = _guard_depth_from_static_parameters(static_parameters)
    return assemble_tiled_scalar_field(field, static_parameters, tile_shape, num_guard_cells=g)


def vector_field_for_output(field, static_parameters):
    """
    Return ordinary ghost-celled vector components for file formats.
    """

    if not _is_tiled_vector(field):
        return field

    tile_shape = _tile_shape_from_static_parameters(static_parameters)
    g = _guard_depth_from_static_parameters(static_parameters)
    return assemble_tiled_vector_field(field, static_parameters, tile_shape, num_guard_cells=g)


def build_field_output_map(
        fields,
        particles,
        species_config,
        static_parameters,
        dynamic_parameters,
        include_fluid_velocity=False,
        include_charge_density=False,
):
    """
    Select the tiled mesh quantities written by field diagnostics.

    Electrostatic output contains rho, phi, and E. Other solvers output E, B,
    and J by default. Charge density for non-electrostatic solvers and fluid
    velocity are particle diagnostics calculated only when requested.
    """

    E, B, J, rho, phi, *_rest = fields
    electrostatic = static_parameters.solver == "electrostatic"

    if electrostatic:
        rho_for_output = compute_rho(
            particles,
            species_config,
            rho,
            static_parameters,
            dynamic_parameters,
        )
        field_map = {
            "rho": rho_for_output,
            "phi": phi,
            "E": E,
        }
    else:
        field_map = {
            "E": E,
            "B": B,
            "J": J,
        }

        if include_charge_density:
            field_map["rho"] = compute_rho(
                particles,
                species_config,
                rho,
                static_parameters,
                dynamic_parameters,
            )

    if include_fluid_velocity:
        velocity_template = J[0]
        field_map["fluid_velocity"] = tuple(
            compute_velocity_field(
                particles,
                velocity_template,
                direction,
                static_parameters,
                dynamic_parameters,
                species_config=species_config,
            )
            for direction in range(3)
        )

    return field_map


def field_map_for_output(field_map, static_parameters):
    """
    Assemble only the selected tile-major fields at the synchronous I/O boundary.
    """

    output_map = {}

    for name, field in field_map.items():
        is_vector = isinstance(field, (list, tuple)) and len(field) == 3
        if is_vector:
            output_map[name] = vector_field_for_output(field, static_parameters)
        else:
            output_map[name] = scalar_field_for_output(field, static_parameters)

    return output_map


def fields_for_output(fields, static_parameters):
    """
    Assemble tile-major fields at the I/O boundary.

    The live solver state is left untouched.  The particle-retile overflow flag
    is a Python-driver diagnostic, not a physical field, so it is not included
    in the returned output tuple.
    """

    E, B, J, rho, phi, external_fields, *rest = fields
    external_E, external_B = external_fields

    output_fields = (
        vector_field_for_output(E, static_parameters),
        vector_field_for_output(B, static_parameters),
        vector_field_for_output(J, static_parameters),
        scalar_field_for_output(rho, static_parameters),
        scalar_field_for_output(phi, static_parameters),
        (
            vector_field_for_output(external_E, static_parameters),
            vector_field_for_output(external_B, static_parameters),
        ),
    )

    if not rest:
        return output_fields

    if getattr(static_parameters, "solver", None) == "static_metric":
        return output_fields

    pml_state = rest[0]
    return output_fields + (pml_state,)


def _axis_diagnostic_position(x, u, dt, axis_min, axis_max, bc):
    x_diagnostic = x - u * dt / 2

    if int(jnp.asarray(bc).item()) == 0:
        wind = axis_max - axis_min
        x_diagnostic = jnp.where(
            x_diagnostic > axis_max,
            x_diagnostic - wind,
            jnp.where(x_diagnostic < axis_min, x_diagnostic + wind, x_diagnostic),
        )

    return x_diagnostic


def _diagnostic_position(x, u, static_parameters, dynamic_parameters):
    particle_bc = static_parameters.particle_boundary_conditions
    dt = dynamic_parameters.dt
    x_bounds, y_bounds, z_bounds = grid_domain_bounds(dynamic_parameters)
    x_diagnostic = _axis_diagnostic_position(x[:, 0], u[:, 0], dt, x_bounds[0], x_bounds[1], particle_bc[0])
    y_diagnostic = _axis_diagnostic_position(x[:, 1], u[:, 1], dt, y_bounds[0], y_bounds[1], particle_bc[1])
    z_diagnostic = _axis_diagnostic_position(x[:, 2], u[:, 2], dt, z_bounds[0], z_bounds[1], particle_bc[2])

    return jnp.stack((x_diagnostic, y_diagnostic, z_diagnostic), axis=-1)


def particles_for_output(particles, species_config=None, species_names=None, static_parameters=None, dynamic_parameters=None):
    """
    Flatten fixed-capacity tiled particle storage for diagnostics.
    """

    if not isinstance(particles, TiledParticles):
        raise TypeError("Particle output requires TiledParticles runtime storage.")
    use_diagnostic_positions = static_parameters is not None and dynamic_parameters is not None

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
                x_diagnostic=(
                    _diagnostic_position(x, u, static_parameters, dynamic_parameters)
                    if use_diagnostic_positions
                    else x
                ),
                u=u,
                charge=charge,
                mass=mass,
                weight=weight,
            )
        )

    return output_particles
