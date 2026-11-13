
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import importlib.metadata
import os
import traceback

import jax
import numpy as np
import openpmd_api as io

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


def _abort_parallel_write(comm):
    """Stop peers before a rank-local failure leaves them in collective I/O."""
    if comm is not None:
        traceback.print_exc()
        comm.Abort(1)


def _open_openpmd_series(output_path, filename, file_extension=".bp", comm=None, access_mode=None):
    if file_extension is not None:
        filename = "_".join(filename.split()) + file_extension
    series_path = os.path.join(output_path, filename)
    try:
        if comm is None:
            os.makedirs(output_path, exist_ok=True)
            if access_mode is None:
                access_mode = io.Access.append if os.path.exists(series_path) else io.Access.create
            series = io.Series(series_path, access_mode)
        else:
            if not io.variants.get("mpi", False):
                raise RuntimeError("Shared distributed output requires MPI-enabled openpmd-api and parallel HDF5")
            if comm.rank == 0:
                os.makedirs(output_path, exist_ok=True)
            comm.Barrier()
            if access_mode is None:
                exists = comm.bcast(os.path.exists(series_path) if comm.rank == 0 else None, root=0)
                access_mode = io.Access.append if exists else io.Access.create
            # Ranks may own different numbers of chunks, including no particles.
            # Metadata, flushes and close still execute collectively.
            series = io.Series(series_path, access_mode, comm,
                               '{"hdf5":{"independent_stores":true}}')
        series.set_attribute("software", "PyPIC3D")
        series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))
        return series
    except BaseException:
        _abort_parallel_write(comm)
        raise


def _close_openpmd_series(series, comm=None):
    try:
        series.close()
    except BaseException:
        _abort_parallel_write(comm)
        raise

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
    if getattr(dynamic_parameters, "bssn_parameters", None) is not None:
        mesh.set_attribute("normalization", "Gaussian, c=1")
        # D and the reconstructed B are at the common time. The retained
        # current is centered in the last h=dt/2 particle substep.
        mesh.time_offset = -float(dynamic_parameters.dt) / 4 if quantity_name == "J" else 0.0
        return
    if quantity_name in DARK_MESH_LOCATIONS:
        mesh.time_offset = 0.0  # reconstructed integer-time output
    unit_exponents = MESH_UNIT_EXPONENTS.get(quantity_name)
    if unit_exponents is not None:
        mesh.unit_dimension = {
            getattr(io.Unit_Dimension, symbol): exponent
            for symbol, exponent in unit_exponents.items()
        }


def _set_mesh_position(record, name, component_index, active_dims, dynamic_parameters):
    """Record native Yee offsets for dark and dynamical GR fields."""
    locations = DARK_MESH_LOCATIONS.get(name)
    if getattr(dynamic_parameters, "bssn_parameters", None) is not None:
        locations = {"E": D_FIELD_LOCATIONS, "B": B_FIELD_LOCATIONS, "J": D_FIELD_LOCATIONS}.get(name)
    if locations is not None:
        location = locations[component_index]
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
    _set_mesh_position(record, name, 0, active_dims, dynamic_parameters)
    return [array]


def _write_openpmd_vector_mesh(iteration, name, components, dynamic_parameters, active_dims=(1,1,1)):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(
        mesh,
        dynamic_parameters,
        active_dims,
        quantity_name=name,
    )
    buffers = []
    for component_index, (component_name, component_data) in enumerate(zip(("x", "y", "z"), components)):
        array = _ensure_openpmd_array(component_data)
        buffers.append(array)
        record = mesh[component_name]
        record.reset_dataset(io.Dataset(array.dtype, array.shape))
        record.store_chunk(array, [0] * array.ndim, array.shape)
        record.unit_SI = 1.0
        _set_mesh_position(record, name, component_index, active_dims, dynamic_parameters)
    return buffers


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
    _set_mesh_position(record, name, 0, layout.active_dims, dynamic_parameters)
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
    _set_mesh_position(record, name, "xyz".index(component_name), layout.active_dims, dynamic_parameters)
    return record


def write_tiled_scalar_field_chunks_to_iteration(iteration, name, host_shards, *, dynamic_parameters, layout):
    record = _reset_scalar_mesh_record(iteration, name, dynamic_parameters=dynamic_parameters, layout=layout)
    buffers = []

    for shard_index, shard_data in host_shards:
        for offset, tile in _iter_tile_chunks_from_host_shard(
            shard_index,
            shard_data,
            layout=layout,
        ):
            tile = _ensure_openpmd_array(tile, dtype=layout.dtype)
            buffers.append(tile)
            record.store_chunk(tile, offset, list(tile.shape))
    return buffers


def write_tiled_vector_field_chunks_to_iteration(
    iteration,
    name,
    component_host_shards,
    *,
    dynamic_parameters,
    layout,
    component_names=("x", "y", "z"),
):
    buffers = []
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
                buffers.append(tile)
                record.store_chunk(tile, offset, list(tile.shape))
    return buffers


def write_tiled_field_snapshot_to_iteration(iteration, snapshot, *, dynamic_parameters, layout):
    """Store owned tile interiors in the existing common-size mesh records."""
    buffers = []
    # All ranks create record metadata in the same order, even with no chunks.
    for name in sorted(snapshot.fields):
        value = snapshot.fields[name]
        is_vector = isinstance(value, tuple) and len(value) == 3
        if is_vector:
            chunks = write_tiled_vector_field_chunks_to_iteration(
                iteration, name, value, dynamic_parameters=dynamic_parameters, layout=layout)
        else:
            chunks = write_tiled_scalar_field_chunks_to_iteration(
                iteration, name, value, dynamic_parameters=dynamic_parameters, layout=layout)
        buffers.extend(chunks)

    current_offset = getattr(snapshot, "current_time_offset", None)
    if current_offset is not None and "J" in snapshot.fields:
        iteration.meshes["J"].time_offset = current_offset
    return buffers


def write_tiled_field_snapshot_openpmd(
    snapshot,
    *,
    output_dir,
    filename,
    dynamic_parameters,
    layout,
    file_extension=".bp",
    comm=None,
):
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension, comm=comm)

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(dynamic_parameters.dt)
        iteration.time_unit_SI = 1.0

        buffers = write_tiled_field_snapshot_to_iteration(
            iteration, snapshot, dynamic_parameters=dynamic_parameters, layout=layout)
        series.flush()
    except BaseException:
        _abort_parallel_write(comm)
        raise
    finally:
        _close_openpmd_series(series, comm)


def write_openpmd_fields_to_iteration(iteration, field_map, dynamic_parameters, active_dims=(1,1,1)):
    buffers = []
    for name, data in field_map.items():
        is_vector = isinstance(data, (list, tuple)) and len(data) == 3
        if is_vector:
            buffers.extend(_write_openpmd_vector_mesh(iteration, name, data, dynamic_parameters, active_dims))
        else:
            buffers.extend(_write_openpmd_scalar_mesh(iteration, name, data, dynamic_parameters, active_dims))
    return buffers


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


def _particle_shards_by_index(shards):
    """Match owned tile views even when arrays group local tiles differently."""
    indexed = {}
    for index, data in shards:
        origin = tuple(_slice_start(part) for part in index[:3])
        for local in np.ndindex(data.shape[:3]):
            # Views retain tile axes and never assemble a whole particle array.
            tile_index = tuple(slice(start + offset, start + offset + 1)
                               for start, offset in zip(origin, local)) + tuple(index[3:])
            tile = data[tuple(slice(offset, offset + 1) for offset in local)]
            key = tuple(
                (part.start or 0,
                 part.stop if part.stop is not None else (part.start or 0) + tile.shape[axis],
                 part.step or 1) if isinstance(part, slice)
                else (int(part), int(part) + 1, 1)
                for axis, part in enumerate(tile_index[:5])
            )
            if key in indexed:
                raise ValueError("Particle snapshots must contain each owned tile exactly once.")
            indexed[key] = (tile_index, tile)
    return indexed


def _iter_snapshot_particle_chunks(
    snapshot, static_parameters, dynamic_parameters, *, include_tile_index=False, species_index=None,
):
    x_shards = _particle_shards_by_index(snapshot.x_shards)
    u_shards = _particle_shards_by_index(snapshot.u_shards)
    active_shards = _particle_shards_by_index(snapshot.active_shards)
    if x_shards.keys() != u_shards.keys() or x_shards.keys() != active_shards.keys():
        raise ValueError("Particle snapshot x, u, and active shards must own the same indices.")

    C = float(dynamic_parameters.C)
    species_charge = np.asarray(snapshot.species_charge, dtype=np.float64)
    species_mass = np.asarray(snapshot.species_mass, dtype=np.float64)
    species_weight = np.asarray(snapshot.species_weight, dtype=np.float64)

    metric_momentum = getattr(snapshot, "gamma_shards", None) is not None
    # GR snapshots carry Gamma computed from covariant u_i and the metric.
    gamma_shards = _particle_shards_by_index(snapshot.gamma_shards) if metric_momentum else {}
    if metric_momentum and x_shards.keys() != gamma_shards.keys():
        raise ValueError("Particle snapshot gamma shards must own the same indices as x.")

    for key, (x_index, x_chunk) in x_shards.items():
        u_chunk = u_shards[key][1]
        active_chunk = active_shards[key][1]
        gamma_chunk = gamma_shards[key][1] if metric_momentum else None
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

        tx0, ty0, tz0, s0 = tuple(_slice_start(part) for part in x_index[:4])
        ntx, nty, ntz, ns_local, _slots, _coords = x_chunk.shape

        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    for local_s in range(ns_local):
                        species = s0 + local_s
                        if species_index is not None and species != species_index:
                            continue
                        active = np.asarray(active_chunk[tx, ty, tz, local_s], dtype=bool)
                        n_active = int(np.count_nonzero(active))
                        if n_active == 0:
                            continue

                        x_live = np.asarray(x_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]
                        u_live = np.asarray(u_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]

                        charge = np.full(n_active, float(species_charge[species]), dtype=np.float64)
                        mass = np.full(n_active, float(species_mass[species]), dtype=np.float64)
                        weight = np.full(n_active, float(species_weight[species]), dtype=np.float64)

                        if metric_momentum:
                            gamma = np.asarray(gamma_chunk[tx, ty, tz, local_s], dtype=np.float64)[active]
                            momentum = u_live * mass[:, None]
                            x_diagnostic = x_live
                            # Covariant u_i already includes Gamma; use stored positions.
                        else:
                            x_diagnostic = (x_live if getattr(snapshot, "initial", False) else
                                            _diagnostic_position_array(x_live, u_live, static_parameters, dynamic_parameters))
                            gamma = 1.0 / np.sqrt(1.0 - np.sum(u_live * u_live, axis=1) / C**2)
                            momentum = u_live * (mass * gamma)[:, None]

                        chunk = species, x_diagnostic, momentum, charge, mass, weight, gamma
                        if include_tile_index:
                            yield (tx0 + tx, ty0 + ty, tz0 + tz, species), chunk
                        else:
                            yield chunk


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


def _store_particle_record_chunk(
    species_group, offset, x, momentum, charge, mass, weight, gamma, dtype=np.float64, buffers=None,
):
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
    if buffers is None:
        buffers = []
    buffers.extend((x, momentum, charge, mass, weight, gamma))

    for component, data in zip(("x", "y", "z"), (x[:, 0], x[:, 1], x[:, 2])):
        data = _ensure_openpmd_array(data, dtype=dtype)
        buffers.append(data)
        species_group["position"][component].store_chunk(data, start, extent)

    zeros = _ensure_openpmd_array(np.zeros(num_particles, dtype=np.dtype(dtype)), dtype=dtype)
    buffers.append(zeros)
    for component in ("x", "y", "z"):
        species_group["positionOffset"][component].store_chunk(zeros, start, extent)

    for component, data in zip(("x", "y", "z"), (momentum[:, 0], momentum[:, 1], momentum[:, 2])):
        data = _ensure_openpmd_array(data, dtype=dtype)
        buffers.append(data)
        species_group["momentum"][component].store_chunk(data, start, extent)

    species_group["weighting"].store_chunk(weight, start, extent)
    species_group["charge"].store_chunk(charge, start, extent)
    species_group["mass"].store_chunk(mass, start, extent)
    species_group["gamma"].store_chunk(gamma, start, extent)

    return int(offset) + num_particles


def write_tiled_particle_snapshot_to_iteration(
    iteration, snapshot, static_parameters, dynamic_parameters, dtype=np.float64, comm=None, species_index=None,
):
    """Compact local slots and place records in global tile order on every rank."""
    n_species = len(snapshot.species_names)
    tile_counts = tuple(int(getattr(dynamic_parameters, "N" + axis)) // int(width)
                        for axis, width in zip("xyz", static_parameters.tile_shape))
    counts = np.zeros(tile_counts + (n_species,), dtype=np.int64)
    for index, active in snapshot.active_shards:
        counts[index[:4]] += np.count_nonzero(active, axis=-1)
    local_counts = counts
    if comm is not None:
        from mpi4py import MPI
        counts = np.empty_like(local_counts)
        comm.Allreduce(local_counts, counts, op=MPI.SUM)

    flat_counts = counts.reshape((int(np.prod(tile_counts)), n_species))
    offsets = (np.cumsum(flat_counts, axis=0) - flat_counts).reshape(counts.shape)
    totals = np.sum(flat_counts, axis=0)
    selected_species = range(n_species) if species_index is None else (species_index,)
    for species in selected_species:
        species_group = iteration.particles[snapshot.species_names[species].replace(" ", "_")]
        _reset_particle_species_records(
            species_group,
            dtype=dtype,
            num_particles=int(totals[species]),
        )

    buffers = []
    stored = [0] * n_species
    for tile_species, chunk in _iter_snapshot_particle_chunks(
        snapshot, static_parameters, dynamic_parameters, include_tile_index=True, species_index=species_index,
    ):
        species, x, momentum, charge, mass, weight, gamma = chunk
        species_group = iteration.particles[snapshot.species_names[species].replace(" ", "_")]
        _store_particle_record_chunk(
            species_group,
            int(offsets[tile_species]),
            x,
            momentum,
            charge,
            mass,
            weight,
            gamma,
            dtype=dtype,
            buffers=buffers,
        )
        stored[species] += int(x.shape[0])

    expected = local_counts.sum(axis=(0, 1, 2))
    for species in selected_species:
        if stored[species] != int(expected[species]):
            raise RuntimeError(f"Stored {stored[species]} local particles for {snapshot.species_names[species]}, "
                               f"expected {expected[species]}.")
    return buffers


def write_tiled_particle_snapshot_openpmd(
    snapshot,
    *,
    output_dir,
    filename,
    static_parameters,
    dynamic_parameters,
    file_extension=".bp",
    dtype=np.float64,
    comm=None,
):
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension, comm=comm)

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(dynamic_parameters.dt)
        iteration.time_unit_SI = 1.0

        buffers = write_tiled_particle_snapshot_to_iteration(
            iteration,
            snapshot,
            static_parameters,
            dynamic_parameters,
            dtype=dtype,
            comm=comm,
        )

        series.flush()
    except BaseException:
        _abort_parallel_write(comm)
        raise
    finally:
        _close_openpmd_series(series, comm)


def write_openpmd_initial_particles(
    particles,
    static_parameters,
    dynamic_parameters,
    output_dir,
    filename="initial_particles.h5",
    species_config=None,
    species_names=None,
    metric=None,
    comm=None,
):
    """Write initial owned particle chunks to the existing per-species files."""
    from PyPIC3D.diagnostics.async_writer import make_tiled_particle_snapshot

    try:
        gamma = None if metric is None else particle_lorentz_factor(
            particles, metric, static_parameters, dynamic_parameters)
        bssn = getattr(dynamic_parameters, "bssn_parameters", None)
        time = float(bssn.time) if bssn is not None else 0.0
        snapshot = make_tiled_particle_snapshot(
            particles, step=0, time=time, species_names=species_names,
            species_config=species_config, gamma=gamma, initial=True)
        # Initial positions and velocities precede the leapfrog seed. The
        # snapshot's initial flag keeps x unchanged even for non-GR particles.
        output_path = os.path.join(output_dir, "data", "initial_particles")
        for species, name in enumerate(snapshot.species_names):
            series_filename = f"{name.replace(' ', '_')}_{filename}"
            series = _open_openpmd_series(
                output_path, series_filename, file_extension=None, comm=comm, access_mode=io.Access.create)
            try:
                iteration = series.iterations[0]
                iteration.time = time
                iteration.dt = float(dynamic_parameters.dt)
                iteration.time_unit_SI = 1.0
                buffers = write_tiled_particle_snapshot_to_iteration(
                    iteration, snapshot, static_parameters, dynamic_parameters,
                    comm=comm, species_index=species)
                series.flush()
            except BaseException:
                _abort_parallel_write(comm)
                raise
            finally:
                _close_openpmd_series(series, comm)
    except BaseException:
        _abort_parallel_write(comm)
        raise


def write_openpmd_initial_fields(
    field_map, static_parameters, dynamic_parameters, output_dir, filename="initial_fields.h5", comm=None,
):
    """Write initial tile interiors without globally assembling tiled fields."""
    from PyPIC3D.diagnostics.async_writer import make_tiled_field_snapshot

    try:
        tiled_fields = {}
        ordinary_fields = {}
        for name, value in field_map.items():
            vector = isinstance(value, (list, tuple)) and len(value) == 3
            component = value[0] if vector else value
            if component.ndim == 6:
                tiled_fields[name] = value
            else:
                ordinary_fields[name] = value
        if comm is not None and ordinary_fields:
            raise ValueError("MPI initial field output requires tile-major fields.")

        bssn = getattr(dynamic_parameters, "bssn_parameters", None)
        time = float(bssn.time) if bssn is not None else 0.0
        if tiled_fields:
            snapshot = make_tiled_field_snapshot(
                tiled_fields, step=0, time=time, current_time_offset=0.0 if bssn is not None else None)
            layout = TiledMeshLayout(
                global_shape=tuple(int(getattr(dynamic_parameters, "N" + axis)) for axis in "xyz"),
                tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
                guard_cells=int(static_parameters.guard_cells),
            )
        output_path = os.path.join(output_dir, "data", "initial_fields")
        series = _open_openpmd_series(
            output_path, filename, file_extension=None, comm=comm, access_mode=io.Access.create)
        try:
            iteration = series.iterations[0]
            iteration.time = time
            iteration.dt = float(dynamic_parameters.dt)
            iteration.time_unit_SI = 1.0
            buffers = []
            if tiled_fields:
                buffers.extend(write_tiled_field_snapshot_to_iteration(
                    iteration, snapshot, dynamic_parameters=dynamic_parameters, layout=layout))
            buffers.extend(write_openpmd_fields_to_iteration(
                iteration, _field_map_to_interior(ordinary_fields), dynamic_parameters))
            if bssn is not None and "J" in field_map:
                iteration.meshes["J"].time_offset = 0.0
            series.flush()
        except BaseException:
            _abort_parallel_write(comm)
            raise
        finally:
            _close_openpmd_series(series, comm)
    except BaseException:
        _abort_parallel_write(comm)
        raise
