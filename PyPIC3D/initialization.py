import os

import jax
import jax.numpy as jnp

from PyPIC3D.particles.particle_initialization import load_particles_from_toml
from PyPIC3D.utils import (
    add_external_fields,
    build_plasma_parameters_dict,
    compute_energy,
    convert_to_jax_compatible,
    courant_condition,
    load_external_fields_from_toml,
    make_dir,
    particle_sanity_check,
    print_stats,
    update_parameters_from_toml,
)
from PyPIC3D.utilities.grids import (
    build_collocated_grid,
    build_tiled_yee_grids,
    build_yee_grid,
)
from PyPIC3D.diagnostics.output_adapters import particles_for_output
from PyPIC3D.diagnostics.openPMD import (
    write_openpmd_initial_fields,
    write_openpmd_initial_particles,
)
from PyPIC3D.diagnostics.plotting import plot_initial_histograms
from PyPIC3D.boundary_conditions.ghost_cells import (
    make_field_mesh,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.evolve import time_loop_electrodynamic, time_loop_electrostatic
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.PML import initialize_tiled_pml_state, load_pml_from_toml
from PyPIC3D.parameters import build_dynamic_parameters, build_static_parameters


def _encode_field_bc(bc_name):
    """
    Encode field boundary condition labels into integer codes for JAX-safe storage.
    """
    bc_codes = {
        "periodic": BC_PERIODIC,
        "conducting": BC_CONDUCTING,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported field boundary condition: {bc_name}")
    return bc_codes[bc_name]


def _encode_particle_bc(bc_name):
    """
    Encode global particle boundary condition labels into integer codes.
    """
    bc_codes = {
        "periodic": 0,
        "reflecting": 1,
        "absorbing": 2,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported particle boundary condition: {bc_name}")
    return bc_codes[bc_name]


def validate_field_solver(solver):
    """
    Keep the active field-solver names explicit so stale configs do not silently
    fall through to a different numerical update.
    """
    supported_solvers = ("electrodynamic_yee", "electrostatic")
    if solver not in supported_solvers:
        raise ValueError(
            f"Unsupported solver: {solver}. Use 'electrodynamic_yee' or 'electrostatic'."
        )


def _tile_shape_from_static_config(static_config):
    return (
        int(static_config["particle_tile_nx"]),
        int(static_config["particle_tile_ny"]),
        int(static_config["particle_tile_nz"]),
    )


def _encode_current_calculation(current_calculation):
    if current_calculation not in ("j_from_rhov", "esirkepov"):
        raise ValueError("Unsupported current_calculation. Use 'j_from_rhov' or 'esirkepov'.")
    if current_calculation == "esirkepov":
        return "esirkepov"
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

    if static_config["current_calculation"] not in ("j_from_rhov", "esirkepov"):
        raise ValueError("Yee runtime currently supports current_calculation='j_from_rhov' or 'esirkepov'")
    if static_config["particle_pusher"] not in ("boris", "higuera_cary"):
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


def _apply_pml_field_boundaries(static_config, pml_config):
    """
    PML-active axes use nonwrapping field halos from initialization onward.
    """

    _, pml_x, pml_y, pml_z, _ = pml_config
    for axis, pml_axis_active in zip(("x", "y", "z"), (pml_x, pml_y, pml_z)):
        if pml_axis_active and static_config["boundary_conditions"][axis] == BC_PERIODIC:
            static_config["boundary_conditions"][axis] = BC_CONDUCTING


def default_parameters():
    """
    Return plotting, static, and dynamic parameter dictionaries.
    """
    plotting_parameters = {
        "plotting": True,
        "save_data": False,
        "plotfields": False,
        "plotpositions": False,
        "plotvelocities": False,
        "plotenergy": True,
        "plotcurrent": False,
        "plasmaFreq": False,
        "plot_phasespace": False,
        "plot_errors": False,
        "plot_dispersion": False,
        "plot_chargeconservation": False,
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
        "cfl": 1.0,
        "ds_per_debye": None,
        "shape_factor": 1,
        "guard_cells": 2,
        "particle_tile_nx": None,
        "particle_tile_ny": None,
        "particle_tile_nz": None,
        "particle_tile_capacity_factor": 1.0,
        "current_calculation": "j_from_rhov",
        "filter_j": "bilinear",
    }

    dynamic_parameters = {
        "Nx": 30,
        "Ny": 30,
        "Nz": 30,
        "x_wind": 1e-2,
        "y_wind": 1e-2,
        "z_wind": 1e-2,
        "t_wind": 1e-12,
        "dt": None,
        "eps": 8.85418782e-12,
        "mu": 1.25663706e-6,
        "C": 2.99792458e8,
        "kb": 1.380649e-23,
        "alpha": 1.0,
    }

    return plotting_parameters, static_parameters, dynamic_parameters


def setup_write_dir(static_parameters, plotting_parameters):
    output_dir = static_parameters["output_dir"]
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
    static_config["electrostatic"] = electrostatic

    Nx, Ny, Nz = dynamic_config["Nx"], dynamic_config["Ny"], dynamic_config["Nz"]
    x_wind, y_wind, z_wind = dynamic_config["x_wind"], dynamic_config["y_wind"], dynamic_config["z_wind"]
    t_wind = dynamic_config["t_wind"]

    if static_config["particle_tile_nx"] is None:
        static_config["particle_tile_nx"] = int(Nx)
    if static_config["particle_tile_ny"] is None:
        static_config["particle_tile_ny"] = int(Ny)
    if static_config["particle_tile_nz"] is None:
        static_config["particle_tile_nz"] = int(Nz)

    if electrostatic:
        static_config["particle_tile_nx"] = int(Nx)
        static_config["particle_tile_ny"] = int(Ny)
        static_config["particle_tile_nz"] = int(Nz)

    guard_cells = max(int(static_config["guard_cells"]), 2)
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
        dt = courant_condition(static_config["cfl"], dx, dy, dz, dynamic_config)
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

    _validate_tiled_yee_configuration(static_config, dynamic_config)
    pml_config = load_pml_from_toml(raw_pml, static_config, dynamic_config)
    static_config["pml_active"] = pml_config[0]
    _apply_pml_field_boundaries(static_config, pml_config)

    dynamic_setup = convert_to_jax_compatible(dynamic_config)
    if electrostatic:
        B_grid, E_grid = build_collocated_grid(dynamic_setup)
    else:
        B_grid, E_grid = build_yee_grid(dynamic_setup)

    dynamic_config["grids"] = {
        "vertex": E_grid,
        "center": B_grid,
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

    tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(
        static_config,
        dynamic_config,
        tile_shape,
        guard_cells,
    )
    dynamic_config["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
    dynamic_config["grids"]["tiled_center_grid"] = tiled_center_grid

    static_parameters = build_static_parameters(static_config)
    dynamic_parameters = build_dynamic_parameters(dynamic_config)
    plotting_parameters = convert_to_jax_compatible(plotting_parameters)

    particles, species_config, particle_species_names, particle_metadata = load_particles_from_toml(
        config,
        static_parameters,
        dynamic_parameters,
    )
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
            path=f"{static_parameters['output_dir']}/data",
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
            static_parameters["output_dir"],
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

    if static_parameters["relativistic"]:
        print("Relativistic simulation")
    else:
        print("Non-relativistic simulation")
    print(f"Using {static_parameters['particle_pusher']} particle pusher")

    if electrostatic:
        print("Using electrostatic solver")
        evolve_loop = time_loop_electrostatic
    else:
        print("Using electrodynamic Yee solver")
        evolve_loop = time_loop_electrodynamic

    if static_config["current_calculation"] == "esirkepov":
        print("Using Esirkepov current calculation method")
    elif static_config["current_calculation"] == "j_from_rhov":
        print(f"Using J from rhov current calculation method with filter: {static_config['filter_j']}")

    print(f"Using tiled Yee storage with tile shape: {tile_shape}")

    overflow = jnp.asarray(False)
    if electrostatic:
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

    if plotting_parameters["dump_fields"]:
        write_openpmd_initial_fields(
            fields,
            static_parameters,
            dynamic_parameters,
            static_parameters["output_dir"],
            filename="initial_fields.h5",
        )

    if static_parameters["GPUs"]:
        print("GPUs Detected! Using GPUs for simulation\n")
        particles = jax.device_put(particles, jax.devices("gpu")[0])
        if species_config is not None:
            species_config = jax.device_put(species_config, jax.devices("gpu")[0])

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


def build_tiled_array(static_parameters, dynamic_parameters=None, dtype=jnp.float64):
    """
    Build one zero-filled tiled field component from the split geometry.
    """

    if dynamic_parameters is None:
        dynamic_parameters = static_parameters

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters["tile_shape"]]
    Nx = int(dynamic_parameters["Nx"])
    Ny = int(dynamic_parameters["Ny"])
    Nz = int(dynamic_parameters["Nz"])
    g = int(static_parameters["guard_cells"])
    ntx = Nx // tile_nx
    nty = Ny // tile_ny
    ntz = Nz // tile_nz
    tiled_shape = (ntx, nty, ntz, tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g)
    return jnp.zeros(shape=tiled_shape, dtype=dtype)


def initialize_fields(static_parameters, dynamic_parameters=None):
    """
    Initialize tiled electric, magnetic, current, potential, and charge arrays.
    """

    if dynamic_parameters is None:
        dynamic_parameters = static_parameters

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
