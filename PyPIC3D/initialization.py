import os
from numbers import Integral
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_initialization import load_particles_from_toml
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.plasma_quantities import build_plasma_parameters_dict
from PyPIC3D.utilities.simulation_helpers import (
    convert_to_jax_compatible,
    courant_condition,
    make_dir,
    particle_sanity_check,
    print_stats,
)
from PyPIC3D.utilities.toml_helpers import (
    load_external_fields_from_toml,
    load_previous_fields_from_toml,
    update_parameters_from_toml,
)
from PyPIC3D.utilities.grids import (
    build_collocated_grid,
    build_tiled_yee_grids,
    build_yee_grid,
)
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, particles_for_output
from PyPIC3D.diagnostics.openPMD import (
    write_openpmd_initial_fields,
    write_openpmd_initial_particles,
)
from PyPIC3D.diagnostics.plotting import plot_initial_histograms
from PyPIC3D.boundary_conditions.ghost_cells import (
    make_field_mesh,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.solvers.gr_static.time_loop import time_loop_static_metric
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_ABSORBING,
    BC_CONDUCTING,
    BC_CONSTANT,
    BC_PERIODIC,
)
from PyPIC3D.boundary_conditions.PML import initialize_tiled_pml_state, load_pml_from_toml
from PyPIC3D.boundary_conditions.supergaussian import load_supergaussian_from_toml
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.relativity.flat import (
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
)
from PyPIC3D.relativity.kerr_schild import (
    initialize_kerr_schild_cartesian_metric,
    initialize_kerr_schild_spherical_metric,
)


def _encode_field_bc(bc_name):
    """
    Encode field boundary condition labels into integer codes for JAX-safe storage.
    """
    bc_codes = {
        "periodic": BC_PERIODIC,
        "conducting": BC_CONDUCTING,
        "constant": BC_CONSTANT,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported field boundary condition: {bc_name}")
    return bc_codes[bc_name]


def _encode_particle_bc(bc_name):
    """
    Encode global particle boundary condition labels into integer codes.
    """
    bc_codes = {
        "periodic": BC_PERIODIC,
        "reflecting": BC_CONDUCTING,
        "absorbing": BC_ABSORBING,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported particle boundary condition: {bc_name}")
    return bc_codes[bc_name]


def validate_field_solver(solver):
    """
    Keep the active field-solver names explicit so stale configs do not silently
    fall through to a different numerical update.
    """
    supported_solvers = ("electrodynamic_yee", "electrostatic", "static_metric")
    if solver not in supported_solvers:
        raise ValueError(
            f"Unsupported solver: {solver}. Use 'electrodynamic_yee', 'electrostatic', or 'static_metric'."
        )


def _tile_shape_from_static_config(static_config):
    return (
        int(static_config["particle_tile_nx"]),
        int(static_config["particle_tile_ny"]),
        int(static_config["particle_tile_nz"]),
    )


def _encode_current_calculation(current_calculation):
    if current_calculation not in ("j_from_rhov", "esirkepov", "GR_direct_deposition", "gr_direct_deposition", "GR_direct"):
        raise ValueError("Unsupported current_calculation. Use 'j_from_rhov', 'esirkepov', or 'GR_direct_deposition'.")
    if current_calculation == "esirkepov":
        return "esirkepov"
    if current_calculation in ("GR_direct_deposition", "gr_direct_deposition", "GR_direct"):
        return "GR_direct"
    return "direct"


def _validate_current_filter_contract(static_config):
    if (
        static_config["current_calculation"] == "esirkepov"
        and static_config["filter_j"] != "none"
    ):
        raise ValueError("Esirkepov current filtering is not supported; use filter_j='none'.")


def _validate_tiled_yee_configuration(static_config, dynamic_config):
    """
    Keep the first tile-native PIC path tied to the kernels that exist today.
    """

    if static_config["solver"] == "static_metric":
        if static_config["current_calculation"] not in ("GR_direct_deposition", "gr_direct_deposition", "GR_direct"):
            raise ValueError("static_metric requires current_calculation='GR_direct_deposition'")
        if static_config["particle_pusher"] != "hybrid_boris_geodesic":
            raise ValueError("static_metric requires particle_pusher='hybrid_boris_geodesic'")
    elif static_config["current_calculation"] not in ("j_from_rhov", "esirkepov"):
        raise ValueError("Yee runtime currently supports current_calculation='j_from_rhov' or 'esirkepov'")
    if static_config["solver"] != "static_metric" and static_config["particle_pusher"] not in ("boris", "higuera_cary"):
        raise ValueError("Yee runtime currently supports only particle_pusher='boris' or 'higuera_cary'")
    if static_config["filter_j"] not in ("none", "digital", "bilinear"):
        raise ValueError("Yee runtime currently supports only filter_j='none', filter_j='digital', or 'bilinear'")
    tile_shape = _tile_shape_from_static_config(static_config)
    grid_shape = (
        int(dynamic_config["Nx"]),
        int(dynamic_config["Ny"]),
        int(dynamic_config["Nz"]),
    )
    for cells, tile_width in zip(grid_shape, tile_shape):
        if cells % tile_width != 0:
            raise ValueError("Yee runtime requires the shared tile shape to divide Nx/Ny/Nz exactly")


def _configured_total_particles(config, dynamic_config):
    """Count the particles requested across the configured species blocks."""

    total_particles = 0
    grid_cells = int(dynamic_config["Nx"]) * int(dynamic_config["Ny"]) * int(dynamic_config["Nz"])

    for key, species in config.items():
        if not key.startswith("particle") or not isinstance(species, dict):
            continue
        if "N_particles" in species:
            total_particles += int(species["N_particles"])
        elif "N_per_cell" in species:
            total_particles += int(species["N_per_cell"] * grid_cells)

    return total_particles


def _resolve_particle_batch_size(static_config, dynamic_config, config):
    """Resolve the static pusher batch shape from the optional run setting."""

    particle_batch_size = static_config.get("particle_batch_size")
    if particle_batch_size is None:
        total_particles = _configured_total_particles(config, dynamic_config)
        particle_batch_size = max(100, total_particles // 4)

    if isinstance(particle_batch_size, bool) or not isinstance(particle_batch_size, Integral):
        raise ValueError("particle_batch_size must be a positive integer.")
    if particle_batch_size <= 0:
        raise ValueError("particle_batch_size must be a positive integer.")

    static_config["particle_batch_size"] = int(particle_batch_size)


def _effective_particle_batch_size(particles, configured_batch_size):
    """Limit a configured batch to the fixed particle-slot capacity of one tile."""

    tile_capacity = int(particles.active.shape[-2]) * int(particles.active.shape[-1])
    if tile_capacity == 0:
        return 1
    return min(int(configured_batch_size), tile_capacity)


def _apply_pml_field_boundaries(static_config, pml_config):
    """
    PML-active axes use nonwrapping field halos from initialization onward.
    """

    _, pml_x, pml_y, pml_z, _ = pml_config
    for axis, pml_axis_active in zip(("x", "y", "z"), (pml_x, pml_y, pml_z)):
        if pml_axis_active and static_config["boundary_conditions"][axis] == BC_PERIODIC:
            static_config["boundary_conditions"][axis] = BC_CONDUCTING


def _apply_supergaussian_field_boundaries(static_config, supergaussian_config):
    """
    Supergaussian-active axes use nonwrapping field halos from initialization onward.
    """

    _, sg_x, sg_y, sg_z, _ = supergaussian_config
    for axis, sg_axis_active in zip(("x", "y", "z"), (sg_x, sg_y, sg_z)):
        if sg_axis_active and static_config["boundary_conditions"][axis] == BC_PERIODIC:
            static_config["boundary_conditions"][axis] = BC_CONDUCTING


def _resolve_axis_bounds(dynamic_config, axis):
    """
    Resolve one coordinate domain before grid construction.
    """

    wind_key = f"{axis}_wind"
    min_key = f"{axis}_min"
    max_key = f"{axis}_max"

    lower = dynamic_config.get(min_key)
    upper = dynamic_config.get(max_key)
    if (lower is None) != (upper is None):
        raise ValueError(f"Both {min_key} and {max_key} must be provided together.")

    if lower is None:
        width = dynamic_config[wind_key]
        lower = -width / 2
        upper = width / 2
    else:
        width = upper - lower
        if width <= 0:
            raise ValueError(f"{max_key} must be greater than {min_key}.")

    dynamic_config[wind_key] = width
    dynamic_config[min_key] = lower
    dynamic_config[max_key] = upper


def default_parameters():
    """
    Return plotting, static, and dynamic parameter dictionaries.
    """
    plotting_parameters = {
        "plotvelocities": False,
        "plotchargedensity": False,
        "plot_openpmd_particles": False,
        "plot_openpmd_fields": False,
        "plotting_interval": 10,
        "openpmd_field_queue_size": 2,
        "openpmd_particle_queue_size": 2,
        "dump_particles": False,
        "dump_fields": False,
    }

    static_parameters = {
        "name": "Default Simulation",
        "output_dir": os.getcwd(),
        "solver": "electrodynamic_yee",
        "particle_x_bc": "periodic",
        "particle_y_bc": "periodic",
        "particle_z_bc": "periodic",
        "x_bc": "periodic",
        "y_bc": "periodic",
        "z_bc": "periodic",
        "Nt": None,
        "relativistic": True,
        "particle_pusher": "boris",
        "benchmark": False,
        "verbose": False,
        "GPUs": False,
        "metric": "flat_cartesian",
        "metric_mass": 1.0,
        "metric_spin": 0.0,
        "cfl": 1.0,
        "ds_per_debye": None,
        "shape_factor": 1,
        "guard_cells": 2,
        "electrostatic_schwarz_tol": 1.0e-6,
        "electrostatic_schwarz_max_iterations": 500,
        "electrostatic_local_cg_tol": 1.0e-6,
        "electrostatic_local_cg_max_iterations": 500,
        "particle_tile_nx": None,
        "particle_tile_ny": None,
        "particle_tile_nz": None,
        "particle_tile_capacity_factor": 1.0,
        "particle_batch_size": None,
        "current_calculation": "j_from_rhov",
        "filter_j": "bilinear",
        "supergaussian_active": False,
        "supergaussian_layers": (),
    }

    dynamic_parameters = {
        "Nx": 30,
        "Ny": 30,
        "Nz": 30,
        "x_wind": 1e-2,
        "y_wind": 1e-2,
        "z_wind": 1e-2,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "z_min": None,
        "z_max": None,
        "t_wind": 1e-12,
        "dt": None,
        "eps": 8.85418782e-12,
        "mu": 1.25663706e-6,
        "C": 2.99792458e8,
        "kb": 1.380649e-23,
        "alpha": 1.0,
    }

    return plotting_parameters, static_parameters, dynamic_parameters


def build_static_metric_state(static_parameters, dynamic_parameters):
    metric_name = static_parameters.metric
    if metric_name == "flat_cartesian":
        return initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
    if metric_name == "flat_cylindrical":
        return initialize_flat_cylindrical_metric(static_parameters, dynamic_parameters)
    if metric_name == "flat_spherical":
        return initialize_flat_spherical_metric(static_parameters, dynamic_parameters)
    if metric_name == "kerr_schild_cartesian":
        return initialize_kerr_schild_cartesian_metric(
            static_parameters,
            dynamic_parameters,
            mass=static_parameters.metric_mass,
            spin=static_parameters.metric_spin,
        )
    if metric_name == "kerr_schild_spherical":
        return initialize_kerr_schild_spherical_metric(
            static_parameters,
            dynamic_parameters,
            mass=static_parameters.metric_mass,
            spin=static_parameters.metric_spin,
        )
    raise ValueError(
        "Unsupported static metric. Use 'flat_cartesian', 'flat_cylindrical', "
        "'flat_spherical', 'kerr_schild_cartesian', or 'kerr_schild_spherical'."
    )


def setup_write_dir(static_config, plotting_parameters):
    output_dir = static_config["output_dir"]
    make_dir(f"{output_dir}/data")


def initialize_simulation(toml_file):
    """
    Initialize particles, fields, grids, and the timestep loop from split parameters.
    """

    config = {} if toml_file is None else toml_file
    plotting_parameters, static_config, dynamic_config = default_parameters()

    if toml_file is not None:
        static_config, dynamic_config, plotting_parameters = update_parameters_from_toml(
            toml_file,
            static_config,
            dynamic_config,
            plotting_parameters,
        )

    print(f"Initializing Simulation: { static_config['name'] }\n")
    print(f"Using boundary conditions: x: {static_config['x_bc']}, y: {static_config['y_bc']}, z: {static_config['z_bc']}\n")

    solver = static_config["solver"]
    validate_field_solver(solver)
    electrostatic = solver == "electrostatic"
    static_metric = solver == "static_metric"
    static_config["electrostatic"] = electrostatic

    _resolve_axis_bounds(dynamic_config, "x")
    _resolve_axis_bounds(dynamic_config, "y")
    _resolve_axis_bounds(dynamic_config, "z")

    Nx, Ny, Nz = dynamic_config["Nx"], dynamic_config["Ny"], dynamic_config["Nz"]
    x_wind, y_wind, z_wind = dynamic_config["x_wind"], dynamic_config["y_wind"], dynamic_config["z_wind"]
    t_wind = dynamic_config["t_wind"]

    if static_config["particle_tile_nx"] is None:
        static_config["particle_tile_nx"] = int(Nx)
    if static_config["particle_tile_ny"] is None:
        static_config["particle_tile_ny"] = int(Ny)
    if static_config["particle_tile_nz"] is None:
        static_config["particle_tile_nz"] = int(Nz)

    _resolve_particle_batch_size(static_config, dynamic_config, config)

    guard_cells = int(static_config["guard_cells"])
    if guard_cells < 1:
        raise ValueError("Tiled fields require at least one guard cell.")
    static_config["guard_cells"] = guard_cells
    _validate_current_filter_contract(static_config)

    setup_write_dir(static_config, plotting_parameters)

    dx, dy, dz = x_wind / Nx, y_wind / Ny, z_wind / Nz
    dynamic_config["dx"] = dx
    dynamic_config["dy"] = dy
    dynamic_config["dz"] = dz

    if dynamic_config["dt"] is not None:
        print(f"Using user defined dt: {dynamic_config['dt']}")
        dt = dynamic_config["dt"]
    else:
        dt = courant_condition(static_config["cfl"], dx, dy, dz, SimpleNamespace(**dynamic_config))
        dynamic_config["dt"] = dt

    if static_config["Nt"] is not None:
        Nt = int(static_config["Nt"])
    else:
        Nt = int(t_wind / dt)
    static_config["Nt"] = Nt

    if dynamic_config["dt"] is not None and config.get("static_parameters", {}).get("Nt") is not None:
        t_wind = dt * Nt
        print(f"Adjusting t_wind to {t_wind} based on provided dt and Nt")
        dynamic_config["t_wind"] = t_wind

    static_config["current_deposition"] = _encode_current_calculation(static_config["current_calculation"])
    static_config["current_filter"] = static_config["filter_j"]
    static_config["boundary_conditions"] = {
        "x": _encode_field_bc(static_config["x_bc"]),
        "y": _encode_field_bc(static_config["y_bc"]),
        "z": _encode_field_bc(static_config["z_bc"]),
    }
    static_config["particle_boundary_conditions"] = {
        "x": _encode_particle_bc(static_config["particle_x_bc"]),
        "y": _encode_particle_bc(static_config["particle_y_bc"]),
        "z": _encode_particle_bc(static_config["particle_z_bc"]),
    }

    raw_pml = config.get("pml", [])
    pml_active = bool(raw_pml)
    if pml_active and electrostatic:
        raise ValueError("PML is only supported for the electrodynamic_yee solver")
    if pml_active and static_metric:
        raise ValueError("PML is not yet supported for the static_metric solver")

    raw_supergaussian = config.get("supergaussian", [])
    supergaussian_active = bool(raw_supergaussian)
    if supergaussian_active and electrostatic:
        raise ValueError("supergaussian is not supported for the electrostatic solver")

    _validate_tiled_yee_configuration(static_config, dynamic_config)

    dynamic_setup = SimpleNamespace(**convert_to_jax_compatible(dynamic_config))
    pml_config = load_pml_from_toml(raw_pml, None, dynamic_setup)
    static_config["pml_active"] = pml_config[0]
    _apply_pml_field_boundaries(static_config, pml_config)
    supergaussian_config = load_supergaussian_from_toml(raw_supergaussian, dynamic_setup)
    static_config["supergaussian_active"] = supergaussian_config[0]
    static_config["supergaussian_layers"] = supergaussian_config[-1]
    _apply_supergaussian_field_boundaries(static_config, supergaussian_config)

    if electrostatic:
        center_grid, vertex_grid = build_collocated_grid(dynamic_setup)
    else:
        center_grid, vertex_grid = build_yee_grid(dynamic_setup)
    # The pusher and direct-current deposition paths use the legacy PyPIC3D
    # convention: "center" is the collocated/base Yee grid, while "vertex" is
    # the staggered grid used for the component offsets.

    dynamic_config["grids"] = {
        "vertex": vertex_grid,
        "center": center_grid,
    }

    tile_shape = _tile_shape_from_static_config(static_config)
    static_config["tile_shape"] = tile_shape
    static_config["Nx"] = int(Nx)
    static_config["Ny"] = int(Ny)
    static_config["Nz"] = int(Nz)
    tile_grid_shape = (
        int(Nx) // int(tile_shape[0]),
        int(Ny) // int(tile_shape[1]),
        int(Nz) // int(tile_shape[2]),
    )
    static_config["field_mesh"] = make_field_mesh(tile_grid_shape)

    grid_dynamic_config = convert_to_jax_compatible({
        key: value for key, value in dynamic_config.items() if key != "grids"
    })
    grid_setup = SimpleNamespace(
        **grid_dynamic_config,
        grids=SimpleNamespace(vertex=vertex_grid, center=center_grid),
    )
    static_setup = SimpleNamespace(tile_shape=tile_shape, guard_cells=guard_cells)
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_setup, grid_setup)
    dynamic_config["grids"]["tiled_center_grid"] = tiled_center_grid
    dynamic_config["grids"]["tiled_vertex_grid"] = tiled_vertex_grid

    static_parameters = build_static_parameters(static_config)
    dynamic_parameters = build_dynamic_parameters(dynamic_config)
    plotting_parameters = convert_to_jax_compatible(plotting_parameters)
    metric = (
        build_static_metric_state(static_parameters, dynamic_parameters)
        if static_metric
        else None
    )

    particles, species_config, particle_species_names, particle_metadata = load_particles_from_toml(
        config,
        static_parameters,
        dynamic_parameters,
    )
    effective_batch_size = _effective_particle_batch_size(
        particles,
        static_parameters.particle_batch_size,
    )
    if effective_batch_size != static_parameters.particle_batch_size:
        print(
            "Reducing particle_batch_size from "
            f"{static_parameters.particle_batch_size} to tile capacity {effective_batch_size}."
        )
        static_parameters = static_parameters._replace(
            particle_batch_size=effective_batch_size,
        )
    particles = shard_tiled_particles(particles, static_parameters)
    plotting_parameters = {
        **plotting_parameters,
        "particle_species_names": particle_species_names,
        "particle_species_metadata": particle_metadata,
    }

    initial_particle_records = particles_for_output(
        particles,
        species_config=species_config,
        species_names=particle_species_names,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
    )
    for particle_record in initial_particle_records:
        name = particle_record.name.replace(" ", "_")
        plot_initial_histograms(
            particle_record,
            dynamic_parameters,
            path=f"{static_parameters.output_dir}/data",
            name=name,
        )

    print_stats(static_parameters, dynamic_parameters)

    if particle_metadata:
        plasma_parameters = build_plasma_parameters_dict(static_parameters, dynamic_parameters, particle_metadata[0])
    else:
        plasma_parameters = {}

    particle_sanity_check(particles)

    if plotting_parameters["dump_particles"]:
        write_openpmd_initial_particles(
            particles,
            static_parameters,
            dynamic_parameters,
            static_parameters.output_dir,
            species_config=species_config,
            species_names=particle_species_names,
        )

    E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
    external_fields = (
        tuple(jax.numpy.zeros_like(comp) for comp in E),
        tuple(jax.numpy.zeros_like(comp) for comp in B),
    )

    field_components = [component for field in [E, B, J] for component in field]
    field_components, external_fields = load_external_fields_from_toml(
        field_components,
        external_fields,
        config,
        static_parameters,
        dynamic_parameters,
    )
    E, B, J = field_components[:3], field_components[3:6], field_components[6:9]

    E = update_tiled_vector_ghost_cells(E, static_parameters, num_guard_cells=guard_cells)
    B = update_tiled_vector_ghost_cells(B, static_parameters, num_guard_cells=guard_cells)
    external_E, external_B = external_fields
    external_E = update_tiled_vector_ghost_cells(external_E, static_parameters, num_guard_cells=guard_cells)
    external_B = update_tiled_vector_ghost_cells(external_B, static_parameters, num_guard_cells=guard_cells)
    external_fields = (external_E, external_B)

    static_metric_state = None
    if static_metric:
        static_metric_state = E, B
        static_metric_state = load_previous_fields_from_toml(
            static_metric_state,
            config,
            static_parameters,
            dynamic_parameters,
        )
        D_previous, B_previous = static_metric_state
        D_previous = update_tiled_vector_ghost_cells(D_previous, static_parameters, num_guard_cells=guard_cells)
        B_previous = update_tiled_vector_ghost_cells(B_previous, static_parameters, num_guard_cells=guard_cells)
        static_metric_state = D_previous, B_previous
        print("Skipping flat-space energy diagnostics for static_metric fields and covariant particle u_i\n")
    else:
        total_E, total_B = add_external_fields(E, B, external_fields)
        e_energy, b_energy, kinetic_energy = compute_energy(
            particles,
            total_E,
            total_B,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )
        print(f"Initial Electric Field Energy: {e_energy:.2e} J")
        print(f"Initial Magnetic Field Energy: {b_energy:.2e} J")
        print(f"Initial Kinetic Energy: {kinetic_energy:.2e} J")
        print(f"Total Initial Energy: {e_energy + b_energy + kinetic_energy:.2e} J\n")

    if static_parameters.relativistic:
        print("Relativistic simulation")
    else:
        print("Non-relativistic simulation")
    print(f"Using {static_parameters.particle_pusher} particle pusher")

    if static_metric:
        print(f"Using static_metric solver with {static_parameters.metric} metric")
        evolve_loop = time_loop_static_metric
    elif electrostatic:
        print("Using electrostatic solver")
        evolve_loop = time_loop_electrostatic
    else:
        print("Using electrodynamic Yee solver")
        evolve_loop = time_loop_electrodynamic

    if static_config["current_calculation"] == "esirkepov":
        print("Using Esirkepov current calculation method")
    elif static_config["current_deposition"] == "GR_direct":
        print(f"Using GR direct current calculation method with filter: {static_config['filter_j']}")
    elif static_config["current_calculation"] == "j_from_rhov":
        print(f"Using J from rhov current calculation method with filter: {static_config['filter_j']}")

    print(f"Using tiled Yee storage with tile shape: {tile_shape}")

    overflow = jnp.asarray(False)
    if static_metric:
        fields = (E, B, J, rho, phi, external_fields, metric, static_metric_state, overflow)
    elif electrostatic:
        fields = (E, B, J, rho, phi, external_fields, None, overflow)
    else:
        pml_state = None
        if pml_active:
            _, _, _, _, pml_profiles = pml_config
            pml_state = initialize_tiled_pml_state(
                static_parameters,
                dynamic_parameters,
                pml_profiles,
                tile_shape,
            )
        fields = (E, B, J, rho, phi, external_fields, pml_state, overflow)

    field_map = build_field_output_map(
        fields,
        particles,
        species_config,
        static_parameters,
        dynamic_parameters,
        include_fluid_velocity=bool(plotting_parameters["plotvelocities"]),
        include_charge_density=bool(plotting_parameters["plotchargedensity"]),
    )
    plotting_parameters = {
        **plotting_parameters,
        "field_map": field_map,
    }

    if plotting_parameters["dump_fields"]:
        write_openpmd_initial_fields(
            field_map,
            static_parameters,
            dynamic_parameters,
            static_parameters.output_dir,
            filename="initial_fields.h5",
        )

    return (
        evolve_loop,
        particles,
        fields,
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        species_config,
    )


def build_tiled_array(static_parameters, dynamic_parameters, dtype=jnp.float64):
    """
    Build one zero-filled tiled field component from the split geometry.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    Nx = int(dynamic_parameters.Nx)
    Ny = int(dynamic_parameters.Ny)
    Nz = int(dynamic_parameters.Nz)
    g = int(static_parameters.guard_cells)
    ntx = Nx // tile_nx
    nty = Ny // tile_ny
    ntz = Nz // tile_nz
    tiled_shape = (ntx, nty, ntz, tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g)
    return jnp.zeros(shape=tiled_shape, dtype=dtype)


def initialize_fields(static_parameters, dynamic_parameters):
    """
    Initialize tiled electric, magnetic, current, potential, and charge arrays.
    """

    Ex = build_tiled_array(static_parameters, dynamic_parameters)
    Ey = build_tiled_array(static_parameters, dynamic_parameters)
    Ez = build_tiled_array(static_parameters, dynamic_parameters)

    Bx = build_tiled_array(static_parameters, dynamic_parameters)
    By = build_tiled_array(static_parameters, dynamic_parameters)
    Bz = build_tiled_array(static_parameters, dynamic_parameters)

    Jx = build_tiled_array(static_parameters, dynamic_parameters)
    Jy = build_tiled_array(static_parameters, dynamic_parameters)
    Jz = build_tiled_array(static_parameters, dynamic_parameters)

    phi = build_tiled_array(static_parameters, dynamic_parameters)
    rho = build_tiled_array(static_parameters, dynamic_parameters)

    return (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), phi, rho
