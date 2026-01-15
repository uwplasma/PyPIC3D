
import openpmd_api as io
from jax import jit
import jax.numpy as jnp
import os
import numpy as np
import importlib.metadata

def _ensure_openpmd_array(data, dtype=np.float64):
    arr = np.asarray(data, dtype=dtype)
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


def _configure_openpmd_iteration(iteration, t, world):
    iteration.time = float(t * world["dt"])
    iteration.dt = float(world["dt"])
    iteration.time_unit_SI = 1.0


def _configure_openpmd_mesh(mesh, world):
    mesh.geometry = io.Geometry.cartesian
    # openpmd-api 0.16+ removed io.Data_Order; mesh.data_order accepts a string.
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
    mesh.grid_spacing = [float(world["dx"]), float(world["dy"]), float(world["dz"])]
    mesh.grid_global_offset = [
        -float(world["x_wind"]) / 2.0,
        -float(world["y_wind"]) / 2.0,
        -float(world["z_wind"]) / 2.0,
    ]
    mesh.axis_labels = ["x", "y", "z"]
    mesh.unit_SI = 1.0


def _write_openpmd_scalar_mesh(iteration, name, data, world):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world)
    array = _ensure_openpmd_array(data)
    record = mesh[io.Mesh_Record_Component.SCALAR]
    record.reset_dataset(io.Dataset(array.dtype, array.shape))
    record.store_chunk(array, [0] * array.ndim, array.shape)
    record.unit_SI = 1.0


def _write_openpmd_vector_mesh(iteration, name, components, world):
    mesh = iteration.meshes[name]
    _configure_openpmd_mesh(mesh, world)
    for component_name, component_data in zip(("x", "y", "z"), components):
        array = _ensure_openpmd_array(component_data)
        record = mesh[component_name]
        record.reset_dataset(io.Dataset(array.dtype, array.shape))
        record.store_chunk(array, [0] * array.ndim, array.shape)
        record.unit_SI = 1.0


def _write_openpmd_fields_to_iteration(iteration, field_map, world):
    for name, data in field_map.items():
        is_vector = isinstance(data, (list, tuple)) and len(data) == 3
        if is_vector:
            _write_openpmd_vector_mesh(iteration, name, data, world)
        else:
            _write_openpmd_scalar_mesh(iteration, name, data, world)


def _write_openpmd_particles_to_iteration(iteration, particles, constants):
    if not particles:
        return

    C = float(constants["C"])

    for species in particles:
        species_name = species.get_name().replace(" ", "_")
        species_group = iteration.particles[species_name]

        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        gamma = 1 / jnp.sqrt(1.0 - (vx**2 + vy**2 + vz**2) / C**2)

        x = _ensure_openpmd_array(x)
        y = _ensure_openpmd_array(y)
        z = _ensure_openpmd_array(z)
        vx = _ensure_openpmd_array(vx)
        vy = _ensure_openpmd_array(vy)
        vz = _ensure_openpmd_array(vz)
        gamma = _ensure_openpmd_array(gamma)

        num_particles = x.shape[0]
        particle_mass = float(species.mass)
        particle_charge = float(species.charge)

        position = species_group["position"]
        for component, data in zip(("x", "y", "z"), (x, y, z)):
            record_component = position[component]
            record_component.reset_dataset(io.Dataset(data.dtype, [num_particles]))
            record_component.store_chunk(data, [0], [num_particles])
            record_component.unit_SI = 1.0

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
            record_component.store_chunk(data * particle_mass * gamma, [0], [num_particles])
            record_component.unit_SI = 1.0

        weighting = species_group["weighting"]
        weights = np.full(num_particles, float(species.weight), dtype=np.float64)
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charges = np.full(num_particles, particle_charge, dtype=np.float64)
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charges, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        masses = np.full(num_particles, particle_mass, dtype=np.float64)
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(masses, [0], [num_particles])
        mass.unit_SI = 1.0


def write_openpmd_fields(fields, world, output_dir, t, filename="fields", file_extension=".bp"):
    """
    Write all field data to an openPMD file for visualization in ParaView/VisIt.

    Args:
        fields (tuple): Field tuple from the solver (E, B, J, rho, ...).
        world (dict): Simulation world parameters.
        output_dir (str): Base output directory for the simulation.
        t (int): Iteration index.
        filename (str): openPMD file name.
    """
    E, B, J, rho, *rest = fields
    field_map = {
        "E": E,
        "B": B,
        "J": J,
        "rho": rho,
    }

    if rest:
        field_map["phi"] = rest[0]
        for idx, extra in enumerate(rest[1:], start=1):
            field_map[f"field_{idx}"] = extra

    series = _open_openpmd_series(output_dir, filename, file_extension=file_extension)
    iteration = series.iterations[int(t)]
    _configure_openpmd_iteration(iteration, t, world)
    _write_openpmd_fields_to_iteration(iteration, field_map, world)
    series.flush()
    series.close()


def write_openpmd_particles(particles, world, constants, output_dir, t, filename="particles", file_extension=".bp"):
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
    iteration = series.iterations[int(t)]
    _configure_openpmd_iteration(iteration, t, world)
    _write_openpmd_particles_to_iteration(iteration, particles, constants)
    series.flush()
    series.close()


def write_openpmd_initial_particles(particles, world, constants, output_dir, filename="initial_particles.h5"):
    """
    Write the initial particle states to separate openPMD files, one per species.

    Args:
        particles (list): List of particle species.
        world (dict): Dictionary containing the simulation world parameters.
        constants (dict): Dictionary of physical constants (must include key 'C' for the speed of light).
        output_dir (str): Base output directory for the simulation.
        filename (str): Base name of the openPMD output file (species name is prepended).
    """
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
        species_name = species.get_name().replace(" ", "_")
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

        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
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
        particle_mass = float(species.mass)
        particle_charge = float(species.charge)

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
            record_component.store_chunk(data * particle_mass * gamma , [0], [num_particles])
            record_component.unit_SI = 1.0

        weighting = species_group["weighting"]
        weights = np.full(num_particles, float(species.weight), dtype=np.float64)
        weighting.reset_dataset(io.Dataset(weights.dtype, [num_particles]))
        weighting.store_chunk(weights, [0], [num_particles])
        weighting.unit_SI = 1.0

        charge = species_group["charge"]
        charges = np.full(num_particles, particle_charge, dtype=np.float64)
        charge.reset_dataset(io.Dataset(charges.dtype, [num_particles]))
        charge.store_chunk(charges, [0], [num_particles])
        charge.unit_SI = 1.0

        mass = species_group["mass"]
        masses = np.full(num_particles, particle_mass, dtype=np.float64)
        mass.reset_dataset(io.Dataset(masses.dtype, [num_particles]))
        mass.store_chunk(masses, [0], [num_particles])
        mass.unit_SI = 1.0

        series.flush()
        series.close()