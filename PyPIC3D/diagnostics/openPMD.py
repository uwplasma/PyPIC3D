
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import importlib.metadata
import os

import jax
import numpy as np
import openpmd_api as io

from PyPIC3D.diagnostics.output_adapters import field_map_for_output, particles_for_output
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor
from PyPIC3D.utilities.grids import grid_domain_bounds


# SI base-dimension exponents for mesh records written by PyPIC3D. Keys use
# openPMD's L, M, T, I symbols: length, mass, time, and electric current.
MESH_UNIT_EXPONENTS = {
    "E": {"L": 1.0, "M": 1.0, "T": -3.0, "I": -1.0},
    "B": {"M": 1.0, "T": -2.0, "I": -1.0},
    "J": {"L": -2.0, "I": 1.0},
    "rho": {"L": -3.0, "T": 1.0, "I": 1.0},
    "phi": {"L": 2.0, "M": 1.0, "T": -3.0, "I": -1.0},
    "fluid_velocity": {"L": 1.0, "T": -1.0},
    "dark_E": {"L": 1.0, "M": 1.0, "T": -3.0, "I": -1.0},
    "dark_B": {"M": 1.0, "T": -2.0, "I": -1.0},
    "dark_A": {"L": 1.0, "M": 1.0, "T": -2.0, "I": -1.0},
    "dark_phi": {"L": 2.0, "M": 1.0, "T": -3.0, "I": -1.0},
}

DARK_MESH_LOCATIONS = {
    "dark_E": D_FIELD_LOCATIONS,
    "dark_A": D_FIELD_LOCATIONS,
    "dark_B": B_FIELD_LOCATIONS,
    "dark_phi": (("C", "C", "C"),),
}
# Yee locations of the integer-time dark photon outputs


@dataclass(frozen=True)
class TiledMeshLayout:
    """
    Metadata needed to place tile-local interiors into a global openPMD mesh.
    """

    global_shape: Tuple[int, int, int]
    tile_shape: Tuple[int, int, int]
    guard_cells: Union[int, Tuple[int, int, int]] = 1
    active_dims: Tuple[int, int, int] = (1, 1, 1)
    dtype: Any = np.float64


def _ensure_openpmd_array(data, dtype=np.float64, squeeze=False):
    """Finish the device-to-host boundary without uploading host snapshots."""
    arr = np.asarray(jax.device_get(data), dtype=dtype)
    if squeeze:
        arr = np.squeeze(arr)
    if not arr.flags.c_contiguous or not arr.flags.writeable:
        arr = arr.copy(order="C")
    return arr


def _open_openpmd_series(output_path, filename, file_extension=".bp"):
    filename = "_".join(filename.split()) + file_extension
    # add file extension
    series_path = os.path.join(output_path, filename)
    access_mode = io.Access.append if os.path.exists(series_path) else io.Access.create
    series = io.Series(series_path, access_mode)
    series.set_attribute("software", "PyPIC3D")
    series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))
    return series

def _configure_openpmd_mesh(
    mesh,
    dynamic_parameters,
    active_dims=(1, 1, 1),
    quantity_name=None,
):
    mesh.geometry = io.Geometry.cartesian
    # openpmd-api 0.16+ removed io.Data_Order; mesh.data_order accepts a string.
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"

    axes    = []
    ds      = []
    offsets = []
    # initialize lists for axes, spacings, and offsets
    x_bounds, y_bounds, z_bounds = grid_domain_bounds(dynamic_parameters)
    if active_dims[0]:
        axes.append("x")
        ds.append(float(dynamic_parameters.dx))
        offsets.append(float(x_bounds[0]))
    if active_dims[1]:
        axes.append("y")
        ds.append(float(dynamic_parameters.dy))
        offsets.append(float(y_bounds[0]))
    if active_dims[2]:
        axes.append("z")
        ds.append(float(dynamic_parameters.dz))
        offsets.append(float(z_bounds[0]))
    # determine the active axes being used and set them

    mesh.axis_labels = axes
    mesh.grid_spacing = ds
    mesh.grid_global_offset = offsets

    mesh.unit_SI = 1.0
    if quantity_name in DARK_MESH_LOCATIONS:
        mesh.time_offset = 0.0  # reconstructed integer-time output
    unit_exponents = MESH_UNIT_EXPONENTS.get(quantity_name)
    if unit_exponents is not None:
        mesh.unit_dimension = {
            getattr(io.Unit_Dimension, symbol): exponent
            for symbol, exponent in unit_exponents.items()
        }


def _set_dark_mesh_position(record, name, component_index, active_dims):
    """Record dark Yee positions; a 'V' location sits half a cell past the base C node."""
    if name in DARK_MESH_LOCATIONS:
        location = DARK_MESH_LOCATIONS[name][component_index]
        record.position = [0.5 if point == "V" else 0.0 for point, active in zip(location, active_dims) if active]


def _write_openpmd_scalar_mesh(iteration, name, data, dynamic_parameters, active_dims=(1,1,1)):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(
        mesh,
        dynamic_parameters,
        active_dims,
        quantity_name=name,
    )
    array = _ensure_openpmd_array(data)
    record = mesh[io.Mesh_Record_Component.SCALAR]
    record.reset_dataset(io.Dataset(array.dtype, array.shape))
    record.store_chunk(array, [0] * array.ndim, array.shape)
    record.unit_SI = 1.0
    _set_dark_mesh_position(record, name, 0, active_dims)


def _write_openpmd_vector_mesh(iteration, name, components, dynamic_parameters, active_dims=(1,1,1)):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(
        mesh,
        dynamic_parameters,
        active_dims,
        quantity_name=name,
    )
    for component_index, (component_name, component_data) in enumerate(zip(("x", "y", "z"), components)):
        array = _ensure_openpmd_array(component_data)
        record = mesh[component_name]
        record.reset_dataset(io.Dataset(array.dtype, array.shape))
        record.store_chunk(array, [0] * array.ndim, array.shape)
        record.unit_SI = 1.0
        _set_dark_mesh_position(record, name, component_index, active_dims)


def _field_map_to_interior(field_map):
    """Strip ghost cells from the selected global fields before openPMD output."""
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    interior_map = {}

    for name, field in field_map.items():
        is_vector = isinstance(field, (list, tuple)) and len(field) == 3
        if is_vector:
            interior_map[name] = tuple(component[interior] for component in field)
        else:
            interior_map[name] = field[interior]

    return interior_map


def _as_3tuple(value):
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    return (int(value), int(value), int(value))


def _slice_start(index_entry):
    if isinstance(index_entry, slice):
        return 0 if index_entry.start is None else int(index_entry.start)
    return int(index_entry)


def _tile_interior(tile, guard_cells):
    gx, gy, gz = guard_cells
    sx = slice(gx, -gx) if gx > 0 else slice(None)
    sy = slice(gy, -gy) if gy > 0 else slice(None)
    sz = slice(gz, -gz) if gz > 0 else slice(None)
    return tile[sx, sy, sz]


def _iter_tile_chunks_from_host_shard(shard_index, shard_data, *, layout):
    """
    Yield ``(global_mesh_offset, tile_interior)`` chunks from one host shard.
    """
    if shard_data.ndim != 6:
        raise ValueError(
            "Expected tiled scalar field shard with shape "
            "(ntx, nty, ntz, nx+2g, ny+2g, nz+2g). "
            f"Got {shard_data.shape}."
        )

    tile_shape = tuple(int(width) for width in layout.tile_shape)
    guard_cells = _as_3tuple(layout.guard_cells)

    tx0 = _slice_start(shard_index[0])
    ty0 = _slice_start(shard_index[1])
    tz0 = _slice_start(shard_index[2])
    ntx_local, nty_local, ntz_local = shard_data.shape[:3]

    for ltx in range(ntx_local):
        for lty in range(nty_local):
            for ltz in range(ntz_local):
                tx = tx0 + ltx
                ty = ty0 + lty
                tz = tz0 + ltz

                raw_tile = shard_data[ltx, lty, ltz]
                interior = _tile_interior(raw_tile, guard_cells)
                if tuple(interior.shape) != tile_shape:
                    raise ValueError(
                        f"Tile interior has shape {interior.shape}, expected {tile_shape}. "
                        f"Check guard_cells={guard_cells} and tile_shape={tile_shape}."
                    )

                offset = [
                    tx * tile_shape[0],
                    ty * tile_shape[1],
                    tz * tile_shape[2],
                ]
                yield offset, interior


def _reset_scalar_mesh_record(iteration, name, *, dynamic_parameters, layout):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(
        mesh,
        dynamic_parameters,
        layout.active_dims,
        quantity_name=name,
    )
    record = mesh[io.Mesh_Record_Component.SCALAR]
    record.reset_dataset(io.Dataset(np.dtype(layout.dtype), list(layout.global_shape)))
    record.unit_SI = 1.0
    _set_dark_mesh_position(record, name, 0, layout.active_dims)
    return record


def _reset_vector_mesh_record(iteration, name, component_name, *, dynamic_parameters, layout):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(
        mesh,
        dynamic_parameters,
        layout.active_dims,
        quantity_name=name,
    )
    record = mesh[component_name]
    record.reset_dataset(io.Dataset(np.dtype(layout.dtype), list(layout.global_shape)))
    record.unit_SI = 1.0
    _set_dark_mesh_position(record, name, "xyz".index(component_name), layout.active_dims)
    return record


def write_tiled_scalar_field_chunks_to_iteration(iteration, name, host_shards, *, dynamic_parameters, layout):
    record = _reset_scalar_mesh_record(iteration, name, dynamic_parameters=dynamic_parameters, layout=layout)

    for shard_index, shard_data in host_shards:
        for offset, tile in _iter_tile_chunks_from_host_shard(
            shard_index,
            shard_data,
            layout=layout,
        ):
            tile = _ensure_openpmd_array(tile, dtype=layout.dtype)
            record.store_chunk(tile, offset, list(tile.shape))


def write_tiled_vector_field_chunks_to_iteration(
    iteration,
    name,
    component_host_shards,
    *,
    dynamic_parameters,
    layout,
    component_names=("x", "y", "z"),
):
    for component_name, host_shards in zip(component_names, component_host_shards):
        record = _reset_vector_mesh_record(
            iteration,
            name,
            component_name,
            dynamic_parameters=dynamic_parameters,
            layout=layout,
        )

        for shard_index, shard_data in host_shards:
            for offset, tile in _iter_tile_chunks_from_host_shard(
                shard_index,
                shard_data,
                layout=layout,
            ):
                tile = _ensure_openpmd_array(tile, dtype=layout.dtype)
                record.store_chunk(tile, offset, list(tile.shape))


def write_tiled_field_snapshot_openpmd(
    snapshot,
    *,
    output_dir,
    filename,
    dynamic_parameters,
    layout,
    file_extension=".bp",
):
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(dynamic_parameters.dt)
        iteration.time_unit_SI = 1.0

        for name, value in snapshot.fields.items():
            is_vector = isinstance(value, tuple) and len(value) == 3
            if is_vector:
                write_tiled_vector_field_chunks_to_iteration(
                    iteration,
                    name,
                    value,
                    dynamic_parameters=dynamic_parameters,
                    layout=layout,
                )
            else:
                write_tiled_scalar_field_chunks_to_iteration(
                    iteration,
                    name,
                    value,
                    dynamic_parameters=dynamic_parameters,
                    layout=layout,
                )

        series.flush()
    finally:
        series.close()


def write_openpmd_fields_to_iteration(iteration, field_map, dynamic_parameters, active_dims=(1,1,1)):
    for name, data in field_map.items():
        is_vector = isinstance(data, (list, tuple)) and len(data) == 3
        if is_vector:
            _write_openpmd_vector_mesh(iteration, name, data, dynamic_parameters, active_dims)
        else:
            _write_openpmd_scalar_mesh(iteration, name, data, dynamic_parameters, active_dims)


def _axis_diagnostic_position_array(x, u, dt, axis_min, axis_max, bc):
    x_diagnostic = x - u * dt / 2.0

    if int(np.asarray(bc)) == 0:
        wind = axis_max - axis_min
        x_diagnostic = np.where(
            x_diagnostic > axis_max,
            x_diagnostic - wind,
            np.where(x_diagnostic < axis_min, x_diagnostic + wind, x_diagnostic),
        )

    return x_diagnostic


def _diagnostic_position_array(x, u, static_parameters, dynamic_parameters):
    particle_bc = static_parameters.particle_boundary_conditions
    dt = float(dynamic_parameters.dt)
    x_bounds, y_bounds, z_bounds = grid_domain_bounds(dynamic_parameters)

    return np.stack(
        (
            _axis_diagnostic_position_array(x[:, 0], u[:, 0], dt, float(x_bounds[0]), float(x_bounds[1]), particle_bc[0]),
            _axis_diagnostic_position_array(x[:, 1], u[:, 1], dt, float(y_bounds[0]), float(y_bounds[1]), particle_bc[1]),
            _axis_diagnostic_position_array(x[:, 2], u[:, 2], dt, float(z_bounds[0]), float(z_bounds[1]), particle_bc[2]),
        ),
        axis=-1,
    )


def _count_snapshot_particles_by_species(snapshot):
    counts = [0] * len(snapshot.species_names)

    for active_index, active_chunk in snapshot.active_shards:
        if active_chunk.ndim != 5:
            raise ValueError(
                "Expected active particle shard with shape "
                "(ntx, nty, ntz, nspecies, max_particles_per_tile). "
                f"Got {active_chunk.shape}."
            )
        s0 = _slice_start(active_index[3])
        for local_s in range(active_chunk.shape[3]):
            counts[s0 + local_s] += int(np.count_nonzero(active_chunk[:, :, :, local_s, :]))

    return counts


def _iter_snapshot_particle_chunks(snapshot, static_parameters, dynamic_parameters):
    if len(snapshot.x_shards) != len(snapshot.u_shards) or len(snapshot.x_shards) != len(snapshot.active_shards):
        raise ValueError("Particle snapshot x, u, and active shard lists must have the same length.")

    C = float(dynamic_parameters.C)
    species_charge = np.asarray(snapshot.species_charge, dtype=np.float64)
    species_mass = np.asarray(snapshot.species_mass, dtype=np.float64)
    species_weight = np.asarray(snapshot.species_weight, dtype=np.float64)

    static_metric = getattr(snapshot, "gamma_shards", None) is not None
    # static-metric snapshots carry Gamma computed from covariant u_i and the metric
    gamma_shards = snapshot.gamma_shards if static_metric else [(None, None)] * len(snapshot.x_shards)
    if static_metric:
        if len(gamma_shards) != len(snapshot.x_shards):
            raise ValueError("Particle snapshot gamma and x shard lists must have the same length.")
        for (x_index, _), (gamma_index, _) in zip(snapshot.x_shards, gamma_shards):
            if tuple(gamma_index) != tuple(x_index[:5]):
                raise ValueError("Particle snapshot gamma shards must follow the x shard layout.")

    for (x_index, x_chunk), (_u_index, u_chunk), (_active_index, active_chunk), (_gamma_index, gamma_chunk) in zip(
        snapshot.x_shards,
        snapshot.u_shards,
        snapshot.active_shards,
        gamma_shards,
    ):
        if x_chunk.ndim != 6:
            raise ValueError(
                "Expected particle x shard with shape "
                "(ntx, nty, ntz, nspecies, max_particles_per_tile, 3). "
                f"Got {x_chunk.shape}."
            )
        if u_chunk.shape != x_chunk.shape:
            raise ValueError(f"Particle u shard shape {u_chunk.shape} must match x shape {x_chunk.shape}.")
        if active_chunk.shape != x_chunk.shape[:-1]:
            raise ValueError(f"Particle active shard shape {active_chunk.shape} must match x.shape[:-1] {x_chunk.shape[:-1]}.")

        s0 = _slice_start(x_index[3])
        ntx, nty, ntz, ns_local, _slots, _coords = x_chunk.shape

        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    for local_s in range(ns_local):
                        species_index = s0 + local_s
                        active = np.asarray(active_chunk[tx, ty, tz, local_s], dtype=bool)
                        n_active = int(np.count_nonzero(active))
                        if n_active == 0:
                            continue

                        x_live = np.asarray(x_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]
                        u_live = np.asarray(u_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]

                        charge = np.full(n_active, float(species_charge[species_index]), dtype=np.float64)
                        mass = np.full(n_active, float(species_mass[species_index]), dtype=np.float64)
                        weight = np.full(n_active, float(species_weight[species_index]), dtype=np.float64)

                        if static_metric:
                            gamma = np.asarray(gamma_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]
                            momentum = u_live * mass[:, None]
                            # covariant u_i already includes Gamma; positions are written as stored
                            yield species_index, x_live, momentum, charge, mass, weight, gamma
                        else:
                            x_diagnostic = _diagnostic_position_array(x_live, u_live, static_parameters, dynamic_parameters)
                            gamma = 1.0 / np.sqrt(1.0 - np.sum(u_live * u_live, axis=1) / C**2)
                            momentum = u_live * (mass * gamma)[:, None]
                            yield species_index, x_diagnostic, momentum, charge, mass, weight, gamma


def _reset_particle_species_records(species_group, dtype, num_particles):
    shape = [int(num_particles)]
    dtype = np.dtype(dtype)

    for record_name in ("position", "positionOffset", "momentum"):
        record = species_group[record_name]
        for component in ("x", "y", "z"):
            record_component = record[component]
            record_component.reset_dataset(io.Dataset(dtype, shape))
            record_component.unit_SI = 1.0

    for record_name in ("weighting", "charge", "mass", "gamma"):
        record_component = species_group[record_name]
        record_component.reset_dataset(io.Dataset(dtype, shape))
        record_component.unit_SI = 1.0


def _store_particle_record_chunk(species_group, offset, x, momentum, charge, mass, weight, gamma, dtype=np.float64):
    num_particles = int(x.shape[0])
    if num_particles == 0:
        return int(offset)

    start = [int(offset)]
    extent = [num_particles]

    x = _ensure_openpmd_array(x, dtype=dtype)
    momentum = _ensure_openpmd_array(momentum, dtype=dtype)
    charge = _ensure_openpmd_array(charge, dtype=dtype)
    mass = _ensure_openpmd_array(mass, dtype=dtype)
    weight = _ensure_openpmd_array(weight, dtype=dtype)
    gamma = _ensure_openpmd_array(gamma, dtype=dtype)

    for component, data in zip(("x", "y", "z"), (x[:, 0], x[:, 1], x[:, 2])):
        species_group["position"][component].store_chunk(_ensure_openpmd_array(data, dtype=dtype), start, extent)

    zeros = _ensure_openpmd_array(np.zeros(num_particles, dtype=np.dtype(dtype)), dtype=dtype)
    for component in ("x", "y", "z"):
        species_group["positionOffset"][component].store_chunk(zeros, start, extent)

    for component, data in zip(("x", "y", "z"), (momentum[:, 0], momentum[:, 1], momentum[:, 2])):
        species_group["momentum"][component].store_chunk(_ensure_openpmd_array(data, dtype=dtype), start, extent)

    species_group["weighting"].store_chunk(weight, start, extent)
    species_group["charge"].store_chunk(charge, start, extent)
    species_group["mass"].store_chunk(mass, start, extent)
    species_group["gamma"].store_chunk(gamma, start, extent)

    return int(offset) + num_particles


def write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static_parameters, dynamic_parameters, dtype=np.float64):
    counts = _count_snapshot_particles_by_species(snapshot)

    for species_index, species_name in enumerate(snapshot.species_names):
        species_group = iteration.particles[species_name.replace(" ", "_")]
        _reset_particle_species_records(
            species_group,
            dtype=dtype,
            num_particles=int(counts[species_index]),
        )

    offsets = [0] * len(snapshot.species_names)

    for species_index, x, momentum, charge, mass, weight, gamma in _iter_snapshot_particle_chunks(snapshot, static_parameters, dynamic_parameters):
        species_group = iteration.particles[snapshot.species_names[species_index].replace(" ", "_")]
        offsets[species_index] = _store_particle_record_chunk(
            species_group,
            int(offsets[species_index]),
            x,
            momentum,
            charge,
            mass,
            weight,
            gamma,
            dtype=dtype,
        )

    for species_name, stored, expected in zip(snapshot.species_names, offsets, counts):
        if int(stored) != int(expected):
            raise RuntimeError(f"Stored {stored} particles for {species_name}, expected {expected}.")


def write_tiled_particle_snapshot_openpmd(
    snapshot,
    *,
    output_dir,
    filename,
    static_parameters,
    dynamic_parameters,
    file_extension=".bp",
    dtype=np.float64,
):
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(dynamic_parameters.dt)
        iteration.time_unit_SI = 1.0

        write_tiled_particle_snapshot_to_iteration(
            iteration,
            snapshot,
            static_parameters,
            dynamic_parameters,
            dtype=dtype,
        )

        series.flush()
    finally:
        series.close()


def write_openpmd_initial_particles(
    particles,
    static_parameters,
    dynamic_parameters,
    output_dir,
    filename="initial_particles.h5",
    species_config=None,
    species_names=None,
    metric=None,
):
    """
    Write the initial particle states to separate openPMD files, one per species.

    Args:
        particles (list): List of particle species.
        static_parameters (dict): Compile-time/run parameters.
        dynamic_parameters (dict): Scalar/grid parameters.
        output_dir (str): Base output directory for the simulation.
        filename (str): Base name of the openPMD output file (species name is prepended).
        metric (YeeMetric): Static-metric runs pass the grid metric; particle u is then covariant u_i.
    """

    static_metric_gamma = None
    if metric is not None:
        static_metric_gamma = particle_lorentz_factor(particles, metric, static_parameters, dynamic_parameters)
        # flattened per species below with the same active mask as particles_for_output
        active_tiles = particles.active

    particles = particles_for_output(
        particles,
        species_config=species_config,
        species_names=species_names,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
    )
    # Initial particle dumps may receive tile-major runtime storage.  The
    # openPMD file still contains ordinary per-species particle records.

    if not particles:
        return
    
    C = float(dynamic_parameters.C)
    # speed of light

    output_path = os.path.join(output_dir, "data", "initial_particles")
    os.makedirs(output_path, exist_ok=True)

    for species in particles:
        species_name = species.name.replace(" ", "_")
        series_filename = f"{species_name}_{filename}"
        series_path = os.path.join(output_path, series_filename)

        series = io.Series(series_path, io.Access.create)
        series.set_attribute("software", "PyPIC3D")
        series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))

        iteration = series.iterations[0]
        iteration.time = 0.0
        iteration.dt = float(dynamic_parameters.dt)
        iteration.time_unit_SI = 1.0

        species_group = iteration.particles[species_name]

        positions = _ensure_openpmd_array(species.x)
        velocities = _ensure_openpmd_array(species.u)
        x, y, z = positions.T
        if static_metric_gamma is None:
            gamma = 1 / np.sqrt(1.0 - np.sum(velocities**2, axis=1) / C**2)
            # compute the Lorentz factor
            momentum_per_mass = velocities * gamma[:, None]
        else:
            s = species.species_index
            active = np.asarray(active_tiles[:, :, :, s, :]).reshape(-1)
            gamma = np.asarray(static_metric_gamma[:, :, :, s, :]).reshape(-1)[active]
            momentum_per_mass = velocities
            # covariant u_i already includes the Lorentz factor

        x = _ensure_openpmd_array(x)
        y = _ensure_openpmd_array(y)
        z = _ensure_openpmd_array(z)
        gamma = _ensure_openpmd_array(gamma)

        num_particles = x.shape[0]
        particle_mass = species.mass
        particle_charge = species.charge
        particle_weight = species.weight

        if np.ndim(particle_weight) == 0:
            weights = _ensure_openpmd_array(np.full(num_particles, float(particle_weight), dtype=np.float64))
        else:
            weights = _ensure_openpmd_array(particle_weight, squeeze=True)

        if np.ndim(particle_mass) == 0:
            masses = _ensure_openpmd_array(np.full(num_particles, float(particle_mass), dtype=np.float64))
        else:
            masses = _ensure_openpmd_array(particle_mass, squeeze=True)

        if np.ndim(particle_charge) == 0:
            charges = _ensure_openpmd_array(np.full(num_particles, float(particle_charge), dtype=np.float64))
        else:
            charges = _ensure_openpmd_array(particle_charge, squeeze=True)

        position = species_group["position"]
        for component, data in zip(("x", "y", "z"), (x, y, z)):
            record_component = position[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

        # positionOffset: required by openPMD consumers (WarpX expects it)
        pos_off = species_group["positionOffset"]
        zeros = _ensure_openpmd_array(np.zeros(num_particles, dtype=np.float64))
        for comp in ("x", "y", "z"):
            rc = pos_off[comp]
            rc.reset_dataset(io.Dataset(zeros.dtype, [num_particles]))
            rc.store_chunk(zeros, [0], [num_particles])
            rc.unit_SI = 1.0

        momentum = species_group["momentum"]
        for component, data in zip(("x", "y", "z"), momentum_per_mass.T):
            data = _ensure_openpmd_array(data * masses)
            record_component = momentum[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

        gamma_record = species_group["gamma"]
        gamma_record.reset_dataset(io.Dataset(gamma.dtype, [num_particles]))
        gamma_record.store_chunk(gamma, [0], [num_particles])
        gamma_record.unit_SI = 1.0

        weighting = species_group["weighting"]
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charges, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(masses, [0], [num_particles])
        mass.unit_SI = 1.0

        series.flush()
        series.close()

def write_openpmd_initial_fields(field_map, static_parameters, dynamic_parameters, output_dir, filename="initial_fields.h5"):
    """
    Write the initial field states to an openPMD file.

    Args:
        field_map (dict): Selected scalar and vector mesh quantities.
        static_parameters (dict): Compile-time/run parameters.
        dynamic_parameters (dict): Scalar/grid parameters.
        output_dir (str): Base output directory for the simulation.
        filename (str): openPMD file name.
    """

    field_map = field_map_for_output(field_map, static_parameters)
    field_map = _field_map_to_interior(field_map)
    # extract physical interior (strip ghost cells)

    active_dims = (1, 1, 1)
    # keep singleton mesh axes so thin 2D runs stay in physical x-y-z coordinates


    output_path = os.path.join(output_dir, "data", "initial_fields")
    os.makedirs(output_path, exist_ok=True)
    series_path = os.path.join(output_path, filename)
    series = io.Series(series_path, io.Access.create)
    series.set_attribute("software", "PyPIC3D")
    series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))
    # create the openPMD series

    iteration = series.iterations[0]
    iteration.time = 0.0
    iteration.dt = float(dynamic_parameters.dt)
    iteration.time_unit_SI = 1.0
    write_openpmd_fields_to_iteration(iteration, field_map, dynamic_parameters, active_dims)
    series.flush()
    series.close()
