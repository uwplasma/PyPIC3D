
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import openpmd_api as io
import jax.numpy as jnp
import os
import numpy as np
import importlib.metadata

from PyPIC3D.diagnostics.output_adapters import fields_for_output, particles_for_output


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
    arr = np.asarray(data, dtype=dtype)
    if squeeze:
        arr = np.squeeze(arr)
    if not arr.flags.c_contiguous or not arr.flags.writeable:
        arr = np.array(arr, dtype=dtype, copy=True, order="C")
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

def _configure_openpmd_mesh(mesh, world, active_dims=(1,1,1)):
    mesh.geometry = io.Geometry.cartesian
    # openpmd-api 0.16+ removed io.Data_Order; mesh.data_order accepts a string.
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"

    axes    = []
    ds      = []
    offsets = []
    # initialize lists for axes, spacings, and offsets
    if active_dims[0]:
        axes.append("x")
        ds.append(float(world["dx"]))
        offsets.append(-float(world["x_wind"]) / 2.0)
    if active_dims[1]:
        axes.append("y")
        ds.append(float(world["dy"]))
        offsets.append(-float(world["y_wind"]) / 2.0)
    if active_dims[2]:
        axes.append("z")
        ds.append(float(world["dz"]))
        offsets.append(-float(world["z_wind"]) / 2.0)
    # determine the active axes being used and set them

    mesh.axis_labels = axes
    mesh.grid_spacing = ds
    mesh.grid_global_offset = offsets
    
    mesh.unit_SI = 1.0


def _write_openpmd_scalar_mesh(iteration, name, data, world, active_dims=(1,1,1)):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world, active_dims)
    array = _ensure_openpmd_array(data)
    record = mesh[io.Mesh_Record_Component.SCALAR]
    record.reset_dataset(io.Dataset(array.dtype, array.shape))
    record.store_chunk(array, [0] * array.ndim, array.shape)
    record.unit_SI = 1.0


def _write_openpmd_vector_mesh(iteration, name, components, world, active_dims=(1,1,1)):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world, active_dims)
    for component_name, component_data in zip(("x", "y", "z"), components):
        array = _ensure_openpmd_array(component_data)
        record = mesh[component_name]
        record.reset_dataset(io.Dataset(array.dtype, array.shape))
        record.store_chunk(array, [0] * array.ndim, array.shape)
        record.unit_SI = 1.0


def _fields_to_interior_map(fields):
    """Extract physical interior (strip ghost cells) from a fields tuple and return a field_map dict."""
    E, B, J, rho, phi, external_fields, *rest = fields
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    external_E, external_B = external_fields
    field_map = {
        "E": tuple(comp[interior] for comp in E),
        "B": tuple(comp[interior] for comp in B),
        "J": tuple(comp[interior] for comp in J),
        "rho": rho[interior],
        "phi": phi[interior],
        "external_E": tuple(comp[interior] for comp in external_E),
        "external_B": tuple(comp[interior] for comp in external_B),
    }
    if rest:
        for idx, extra in enumerate(rest, start=1):
            if extra is None:
                continue
            if (
                isinstance(extra, (list, tuple))
                and len(extra) == 2
                and all(isinstance(memory, (list, tuple)) and len(memory) == 6 for memory in extra)
            ):
                # PML memory is solver state, not a physical diagnostic field.
                continue
            if isinstance(extra, (list, tuple)):
                field_map[f"field_{idx}"] = tuple(comp[interior] for comp in extra)
            else:
                field_map[f"field_{idx}"] = extra[interior]
    return field_map


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


def _reset_scalar_mesh_record(iteration, name, *, world, layout):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world, layout.active_dims)
    record = mesh[io.Mesh_Record_Component.SCALAR]
    record.reset_dataset(io.Dataset(np.dtype(layout.dtype), list(layout.global_shape)))
    record.unit_SI = 1.0
    return record


def _reset_vector_mesh_record(iteration, name, component_name, *, world, layout):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world, layout.active_dims)
    record = mesh[component_name]
    record.reset_dataset(io.Dataset(np.dtype(layout.dtype), list(layout.global_shape)))
    record.unit_SI = 1.0
    return record


def write_tiled_scalar_field_chunks_to_iteration(iteration, name, host_shards, *, world, layout):
    record = _reset_scalar_mesh_record(iteration, name, world=world, layout=layout)

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
    world,
    layout,
    component_names=("x", "y", "z"),
):
    for component_name, host_shards in zip(component_names, component_host_shards):
        record = _reset_vector_mesh_record(
            iteration,
            name,
            component_name,
            world=world,
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
    world,
    layout,
    file_extension=".bp",
):
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)

    try:
        iteration = series.iterations[int(snapshot.step)]
        iteration.time = float(snapshot.time)
        iteration.dt = float(world["dt"])
        iteration.time_unit_SI = 1.0

        for name, value in snapshot.fields.items():
            is_vector = isinstance(value, tuple) and len(value) == 3
            if is_vector:
                write_tiled_vector_field_chunks_to_iteration(
                    iteration,
                    name,
                    value,
                    world=world,
                    layout=layout,
                )
            else:
                write_tiled_scalar_field_chunks_to_iteration(
                    iteration,
                    name,
                    value,
                    world=world,
                    layout=layout,
                )

        series.flush()
    finally:
        series.close()


def write_openpmd_fields_to_iteration(iteration, field_map, world, active_dims=(1,1,1)):
    for name, data in field_map.items():
        is_vector = isinstance(data, (list, tuple)) and len(data) == 3
        if is_vector:
            _write_openpmd_vector_mesh(iteration, name, data, world, active_dims)
        else:
            _write_openpmd_scalar_mesh(iteration, name, data, world, active_dims)


def write_openpmd_particles_to_iteration(iteration, particles, constants, species_config=None, species_names=None, world=None):
    particles = particles_for_output(particles, species_config=species_config, species_names=species_names, world=world)
    # Tiled particles carry inactive capacity slots; openPMD should see the
    # active physical particles by species, matching the ordinary output path.

    if not particles:
        return

    C = float(constants["C"])

    for species in particles:
        species_name = species.name.replace(" ", "_")
        species_group = iteration.particles[species_name]

        x, y, z = species.x_diagnostic[:, 0], species.x_diagnostic[:, 1], species.x_diagnostic[:, 2]
        vx, vy, vz = species.u[:, 0], species.u[:, 1], species.u[:, 2]
        gamma = 1 / jnp.sqrt(1.0 - (vx**2 + vy**2 + vz**2) / C**2)

        x = _ensure_openpmd_array(x, squeeze=True)
        y = _ensure_openpmd_array(y, squeeze=True)
        z = _ensure_openpmd_array(z, squeeze=True)
        vx = _ensure_openpmd_array(vx, squeeze=True)
        vy = _ensure_openpmd_array(vy, squeeze=True)
        vz = _ensure_openpmd_array(vz, squeeze=True)
        gamma = _ensure_openpmd_array(gamma, squeeze=True)

        num_particles = x.shape[0]
        # number of particles in this species

        particle_mass = species.mass * species.weight
        particle_charge = species.charge * species.weight
        weights = species.weight
        # get the particle mass, charge, and weight for this species


        if jnp.ndim(weights) == 0:
            weights = np.full(num_particles, float(weights), dtype=np.float64)
        else:
            weights = _ensure_openpmd_array(weights, squeeze=True)

        if jnp.ndim(particle_mass) == 0:
            masses = np.full(num_particles, float(particle_mass), dtype=np.float64)
        else:
            masses = _ensure_openpmd_array(particle_mass, squeeze=True)
        
        if jnp.ndim(particle_charge) == 0:
            charges = np.full(num_particles, float(particle_charge), dtype=np.float64)
        else:
            charges = _ensure_openpmd_array(particle_charge, squeeze=True)
        # ensure weights, masses, and charges are 1D arrays of the correct length for openPMD output

        position = species_group["position"]
        for component, data in zip(("x", "y", "z"), (x, y, z)):
            record_component = position[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

        # positionOffset: required by openPMD consumers (WarpX expects it)
        pos_off = species_group["positionOffset"]
        zeros = np.zeros(num_particles, dtype=np.float64)
        for comp in ("x", "y", "z"):
            rc = pos_off[comp]
            rc.reset_dataset(io.Dataset(zeros.dtype, [num_particles]))
            rc.store_chunk(zeros, [0], [num_particles])
            rc.unit_SI = 1.0

        momentum = species_group["momentum"]
        for component, data in zip(("x", "y", "z"), (vx, vy, vz)):
            record_component = momentum[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            momenta = data * masses * gamma
            # compute the momentum for each particle
            record_component.store_chunk(momenta, [0], [num_particles])
            record_component.unit_SI = 1.0

        weighting = species_group["weighting"]
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charges / weights, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(masses / weights, [0], [num_particles])
        mass.unit_SI = 1.0


def write_openpmd_fields(fields, world, output_dir, plot_t, t, filename="fields", file_extension=".bp"):
    """
    Write all field data to an openPMD file for visualization in ParaView/VisIt.

    Args:
        fields (tuple): Field tuple from the solver (E, B, J, rho, ...).
        world (dict): Simulation world parameters.
        output_dir (str): Base output directory for the simulation.
        plot_t (int): openPMD iteration number/index used when writing this step.
        t (int): Simulation step index used to compute the physical time (t * world["dt"]).
        filename (str): Base name for the openPMD file.
        file_extension (str): File extension for the openPMD series (for example, ".bp").
    """
    fields = fields_for_output(fields, world)
    field_map = _fields_to_interior_map(fields)
    # extract physical interior (strip ghost cells)

    active_dims = (1, 1, 1)
    # keep singleton mesh axes so thin 2D runs stay in physical x-y-z coordinates


    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)
    # open or create the openPMD series
    iteration = series.iterations[int(plot_t)]
    # specify the iteration using the plot number
    iteration.time = float(t * world["dt"])
    # set the physical time
    iteration.dt = float(world["dt"])
    # set the time step
    iteration.time_unit_SI = 1.0
    # set the time unit
    write_openpmd_fields_to_iteration(iteration, field_map, world, active_dims)
    # write the field data to the iteration
    series.flush()
    series.close()
    # flush and close the series


def write_openpmd_particles(particles, world, constants, output_dir, plot_t, t, filename="particles", file_extension=".bp", species_config=None, species_names=None):
    """
    Write all particle data to an openPMD file for visualization in ParaView/VisIt.

    Args:
        particles (list): Particle species list.
        world (dict): Simulation world parameters.
        constants (dict): Physical constants (must include key 'C').
        output_dir (str): Base output directory for the simulation.
        t (int): Iteration index.
        filename (str): openPMD file name.
    """
    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)
    # open or create the openPMD series
    iteration = series.iterations[int(plot_t)]
    # specify the iteration using the plot number
    iteration.time = float(t * world["dt"])
    # set the physical time
    iteration.dt = float(world["dt"])
    # set the time step
    iteration.time_unit_SI = 1.0
    # set the time unit
    write_openpmd_particles_to_iteration(iteration, particles, constants, species_config=species_config, species_names=species_names, world=world)
    # write the particle data to the iteration
    series.flush()
    series.close()
    # flush and close the series



def write_openpmd_initial_particles(particles, world, constants, output_dir, filename="initial_particles.h5", species_config=None, species_names=None):
    """
    Write the initial particle states to separate openPMD files, one per species.

    Args:
        particles (list): List of particle species.
        world (dict): Dictionary containing the simulation world parameters.
        constants (dict): Dictionary of physical constants (must include key 'C' for the speed of light).
        output_dir (str): Base output directory for the simulation.
        filename (str): Base name of the openPMD output file (species name is prepended).
    """
    particles = particles_for_output(particles, species_config=species_config, species_names=species_names, world=world)
    # Initial particle dumps may receive tile-major runtime storage.  The
    # openPMD file still contains ordinary per-species particle records.

    if not particles:
        return
    
    C = constants['C']
    # speed of light

    output_path = os.path.join(output_dir, "data", "initial_particles")
    os.makedirs(output_path, exist_ok=True)

    def make_array_writable(arr):
        arr = np.array(arr, dtype=np.float64, copy=True, order="C")
        arr.setflags(write=True)
        return arr

    for species in particles:
        species_name = species.name.replace(" ", "_")
        series_filename = f"{species_name}_{filename}"
        series_path = os.path.join(output_path, series_filename)

        series = io.Series(series_path, io.Access.create)
        series.set_attribute("software", "PyPIC3D")
        series.set_attribute("softwareVersion", importlib.metadata.version("PyPIC3D"))

        iteration = series.iterations[0]
        iteration.time = 0.0
        iteration.dt = float(world["dt"])
        iteration.time_unit_SI = 1.0

        species_group = iteration.particles[species_name]

        x, y, z = species.x[:, 0], species.x[:, 1], species.x[:, 2]
        vx, vy, vz = species.u[:, 0], species.u[:, 1], species.u[:, 2]
        gamma = 1 / jnp.sqrt(1.0 - (vx**2 + vy**2 + vz**2) / C**2)
        # compute the Lorentz factor

        x = make_array_writable(x)
        y = make_array_writable(y)
        z = make_array_writable(z)
        vx = make_array_writable(vx)
        vy = make_array_writable(vy)
        vz = make_array_writable(vz)
        gamma = make_array_writable(gamma)

        num_particles = x.shape[0]
        particle_mass = species.mass * species.weight
        particle_charge = species.charge * species.weight
        particle_weight = species.weight

        if jnp.ndim(particle_weight) == 0:
            weights = np.full(num_particles, float(particle_weight), dtype=np.float64)
        else:
            weights = _ensure_openpmd_array(particle_weight, squeeze=True)

        if jnp.ndim(particle_mass) == 0:
            masses = np.full(num_particles, float(particle_mass), dtype=np.float64)
        else:
            masses = _ensure_openpmd_array(particle_mass, squeeze=True)

        if jnp.ndim(particle_charge) == 0:
            charges = np.full(num_particles, float(particle_charge), dtype=np.float64)
        else:
            charges = _ensure_openpmd_array(particle_charge, squeeze=True)

        mass_per_weight = masses / weights
        charge_per_weight = charges / weights

        position = species_group["position"]
        for component, data in zip(("x", "y", "z"), (x, y, z)):
            record_component = position[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

        # positionOffset: required by openPMD consumers (WarpX expects it)
        pos_off = species_group["positionOffset"]
        zeros = np.zeros(num_particles, dtype=np.float64)
        for comp in ("x", "y", "z"):
            rc = pos_off[comp]
            rc.reset_dataset(io.Dataset(zeros.dtype, [num_particles]))
            rc.store_chunk(zeros, [0], [num_particles])
            rc.unit_SI = 1.0

        momentum = species_group["momentum"]
        for component, data in zip(("x", "y", "z"), (vx, vy, vz)):
            record_component = momentum[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data * masses * gamma, [0], [num_particles])
            record_component.unit_SI = 1.0

        weighting = species_group["weighting"]
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charge_per_weight, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(mass_per_weight, [0], [num_particles])
        mass.unit_SI = 1.0

        series.flush()
        series.close()

def write_openpmd_initial_fields(fields, world, output_dir, filename="initial_fields.h5"):
    """
    Write the initial field states to an openPMD file.

    Args:
        fields (tuple): Field tuple from the solver (E, B, J, rho, ...).
        world (dict): Simulation world parameters.
        output_dir (str): Base output directory for the simulation.
        filename (str): openPMD file name.
    """
    fields = fields_for_output(fields, world)
    field_map = _fields_to_interior_map(fields)
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
    iteration.dt = float(world["dt"])
    iteration.time_unit_SI = 1.0
    write_openpmd_fields_to_iteration(iteration, field_map, world, active_dims)
    series.flush()
    series.close()
