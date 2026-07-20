import math

import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh, update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.parameters import DynamicParameters, GridParameters, StaticParameters
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.utilities.grids import build_collocated_grid, build_tiled_yee_grids, build_yee_grid


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def _axis_tuple(axis_values):
    if isinstance(axis_values, tuple):
        return tuple(int(value) for value in axis_values)
    return (
        int(axis_values["x"]),
        int(axis_values["y"]),
        int(axis_values["z"]),
    )


def kernel_parameters(
    *,
    Nx=8,
    Ny=6,
    Nz=4,
    x_wind=4.0,
    y_wind=3.0,
    z_wind=2.0,
    dx=None,
    dy=None,
    dz=None,
    dt=0.05,
    tile_shape=None,
    guard_cells=2,
    shape_factor=1,
    boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
    particle_boundary_conditions=(0, 0, 0),
    solver="electrodynamic_yee",
    electrostatic=False,
    relativistic=True,
    particle_pusher="boris",
    current_deposition="direct",
    current_filter="none",
    C=1.0,
    eps=1.0,
    mu=1.0,
    kb=1.0,
    alpha=1.0,
    name="test",
    output_dir=".",
    Nt=1,
    verbose=False,
    GPUs=False,
    benchmark=False,
    pml_active=False,
    particle_tile_capacity_factor=1.0,
):
    if dx is None:
        dx = x_wind / Nx
    if dy is None:
        dy = y_wind / Ny
    if dz is None:
        dz = z_wind / Nz
    if tile_shape is None:
        tile_shape = (Nx, Ny, Nz)

    tile_shape = tuple(int(width) for width in tile_shape)
    tile_grid_shape = (
        int(Nx) // tile_shape[0],
        int(Ny) // tile_shape[1],
        int(Nz) // tile_shape[2],
    )

    static_parameters = StaticParameters(
        name=name,
        output_dir=output_dir,
        Nt=int(Nt),
        verbose=bool(verbose),
        GPUs=bool(GPUs),
        benchmark=bool(benchmark),
        solver=solver,
        electrostatic=bool(electrostatic),
        relativistic=bool(relativistic),
        particle_pusher=particle_pusher,
        current_deposition=current_deposition,
        current_filter=current_filter,
        shape_factor=int(shape_factor),
        guard_cells=int(guard_cells),
        tile_shape=tile_shape,
        particle_tile_capacity_factor=float(particle_tile_capacity_factor),
        pml_active=bool(pml_active),
        boundary_conditions=tuple(int(value) for value in boundary_conditions),
        particle_boundary_conditions=tuple(int(value) for value in particle_boundary_conditions),
        field_mesh=make_field_mesh(tile_grid_shape),
    )

    dynamic_parameters = DynamicParameters(
        dt=jnp.asarray(dt),
        dx=jnp.asarray(dx),
        dy=jnp.asarray(dy),
        dz=jnp.asarray(dz),
        Nx=jnp.asarray(Nx),
        Ny=jnp.asarray(Ny),
        Nz=jnp.asarray(Nz),
        x_wind=jnp.asarray(x_wind),
        y_wind=jnp.asarray(y_wind),
        z_wind=jnp.asarray(z_wind),
        C=jnp.asarray(C),
        eps=jnp.asarray(eps),
        mu=jnp.asarray(mu),
        kb=jnp.asarray(kb),
        alpha=jnp.asarray(alpha),
        grids=GridParameters(
            vertex=(),
            center=(),
            tiled_vertex_grid=(),
            tiled_center_grid=(),
        ),
    )

    if electrostatic:
        center_grid, vertex_grid = build_collocated_grid(dynamic_parameters)
    else:
        center_grid, vertex_grid = build_yee_grid(dynamic_parameters)

    dynamic_parameters = dynamic_parameters._replace(
        grids=GridParameters(
            vertex=vertex_grid,
            center=center_grid,
            tiled_vertex_grid=(),
            tiled_center_grid=(),
        )
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)

    dynamic_parameters = dynamic_parameters._replace(
        grids=GridParameters(
            vertex=vertex_grid,
            center=center_grid,
            tiled_vertex_grid=tiled_vertex_grid,
            tiled_center_grid=tiled_center_grid,
        )
    )

    return static_parameters, dynamic_parameters


def kernel_parameters_from_values(parameter_set, dynamic_values=None):
    if dynamic_values is None:
        dynamic_values = {}

    tile_shape = parameter_set.get(
        "tile_shape",
        (
            parameter_set.get("particle_tile_nx", parameter_set["Nx"]),
            parameter_set.get("particle_tile_ny", parameter_set["Ny"]),
            parameter_set.get("particle_tile_nz", parameter_set["Nz"]),
        ),
    )
    dx = parameter_set.get("dx")
    dy = parameter_set.get("dy")
    dz = parameter_set.get("dz")
    x_wind = parameter_set.get("x_wind", float(parameter_set["Nx"]) * float(1.0 if dx is None else dx))
    y_wind = parameter_set.get("y_wind", float(parameter_set["Ny"]) * float(1.0 if dy is None else dy))
    z_wind = parameter_set.get("z_wind", float(parameter_set["Nz"]) * float(1.0 if dz is None else dz))

    return kernel_parameters(
        Nx=parameter_set["Nx"],
        Ny=parameter_set["Ny"],
        Nz=parameter_set["Nz"],
        x_wind=x_wind,
        y_wind=y_wind,
        z_wind=z_wind,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=parameter_set.get("dt", 0.0),
        tile_shape=tile_shape,
        guard_cells=parameter_set.get("guard_cells", 2),
        shape_factor=parameter_set.get("shape_factor", 1),
        boundary_conditions=_axis_tuple(parameter_set.get("boundary_conditions", (0, 0, 0))),
        particle_boundary_conditions=_axis_tuple(parameter_set.get("particle_boundary_conditions", (0, 0, 0))),
        solver=parameter_set.get("solver", "electrodynamic_yee"),
        electrostatic=parameter_set.get("electrostatic", False),
        relativistic=parameter_set.get("relativistic", True),
        particle_pusher=parameter_set.get("particle_pusher", "boris"),
        current_deposition=parameter_set.get("current_deposition", "direct"),
        current_filter=parameter_set.get("current_filter", "none"),
        C=parameter_set.get("C", dynamic_values.get("C", 1.0)),
        eps=parameter_set.get("eps", dynamic_values.get("eps", 1.0)),
        mu=parameter_set.get("mu", dynamic_values.get("mu", 1.0)),
        kb=parameter_set.get("kb", dynamic_values.get("kb", 1.0)),
        alpha=parameter_set.get("alpha", dynamic_values.get("alpha", 1.0)),
        name=parameter_set.get("name", "test"),
        output_dir=parameter_set.get("output_dir", "."),
        Nt=parameter_set.get("Nt", 1),
        verbose=parameter_set.get("verbose", False),
        GPUs=parameter_set.get("GPUs", False),
        benchmark=parameter_set.get("benchmark", False),
        pml_active=parameter_set.get("pml_active", False),
        particle_tile_capacity_factor=parameter_set.get("particle_tile_capacity_factor", 1.0),
    )


def particle_parameters_from_values(parameter_set, tile_shape=None, dynamic_values=None):
    if dynamic_values is None:
        dynamic_values = {}
    if tile_shape is None:
        tile_shape = parameter_set.get(
            "tile_shape",
            (
                parameter_set.get("particle_tile_nx", parameter_set["Nx"]),
                parameter_set.get("particle_tile_ny", parameter_set["Ny"]),
                parameter_set.get("particle_tile_nz", parameter_set["Nz"]),
            ),
        )
    tile_shape = tuple(int(width) for width in tile_shape)

    dx = parameter_set.get("dx", 1.0)
    dy = parameter_set.get("dy", 1.0)
    dz = parameter_set.get("dz", 1.0)
    x_wind = parameter_set.get("x_wind", float(parameter_set["Nx"]) * float(dx))
    y_wind = parameter_set.get("y_wind", float(parameter_set["Ny"]) * float(dy))
    z_wind = parameter_set.get("z_wind", float(parameter_set["Nz"]) * float(dz))

    static_parameters = StaticParameters(
        name=parameter_set.get("name", "test"),
        output_dir=parameter_set.get("output_dir", "."),
        Nt=int(parameter_set.get("Nt", 1)),
        verbose=bool(parameter_set.get("verbose", False)),
        GPUs=bool(parameter_set.get("GPUs", False)),
        benchmark=bool(parameter_set.get("benchmark", False)),
        solver=parameter_set.get("solver", "electrodynamic_yee"),
        electrostatic=bool(parameter_set.get("electrostatic", False)),
        relativistic=bool(parameter_set.get("relativistic", True)),
        particle_pusher=parameter_set.get("particle_pusher", "boris"),
        current_deposition=parameter_set.get("current_deposition", "direct"),
        current_filter=parameter_set.get("current_filter", "none"),
        shape_factor=int(parameter_set.get("shape_factor", 1)),
        guard_cells=int(parameter_set.get("guard_cells", 2)),
        tile_shape=tile_shape,
        particle_tile_capacity_factor=float(parameter_set.get("particle_tile_capacity_factor", 1.0)),
        pml_active=bool(parameter_set.get("pml_active", False)),
        boundary_conditions=_axis_tuple(parameter_set.get("boundary_conditions", (0, 0, 0))),
        particle_boundary_conditions=_axis_tuple(parameter_set.get("particle_boundary_conditions", (0, 0, 0))),
        field_mesh=None,
    )
    dynamic_parameters = DynamicParameters(
        dt=jnp.asarray(parameter_set.get("dt", 0.0)),
        dx=jnp.asarray(dx),
        dy=jnp.asarray(dy),
        dz=jnp.asarray(dz),
        Nx=jnp.asarray(parameter_set["Nx"]),
        Ny=jnp.asarray(parameter_set["Ny"]),
        Nz=jnp.asarray(parameter_set["Nz"]),
        x_wind=jnp.asarray(x_wind),
        y_wind=jnp.asarray(y_wind),
        z_wind=jnp.asarray(z_wind),
        C=jnp.asarray(parameter_set.get("C", dynamic_values.get("C", 1.0))),
        eps=jnp.asarray(parameter_set.get("eps", dynamic_values.get("eps", 1.0))),
        mu=jnp.asarray(parameter_set.get("mu", dynamic_values.get("mu", 1.0))),
        kb=jnp.asarray(parameter_set.get("kb", dynamic_values.get("kb", 1.0))),
        alpha=jnp.asarray(parameter_set.get("alpha", dynamic_values.get("alpha", 1.0))),
        grids=GridParameters(vertex=(), center=(), tiled_vertex_grid=(), tiled_center_grid=()),
    )
    return static_parameters, dynamic_parameters


def particle_parameters_from_tile_values(parameter_set, simulation_parameters=None, dynamic_values=None):
    parameter_set = dict(parameter_set)
    if simulation_parameters is not None:
        parameter_set["tile_shape"] = (
            simulation_parameters["particle_tile_nx"],
            simulation_parameters["particle_tile_ny"],
            simulation_parameters["particle_tile_nz"],
        )
        parameter_set["particle_tile_capacity_factor"] = simulation_parameters.get(
            "particle_tile_capacity_factor",
            parameter_set.get("particle_tile_capacity_factor", 1.0),
        )
    return particle_parameters_from_values(
        parameter_set,
        tile_shape=parameter_set.get("tile_shape"),
        dynamic_values=dynamic_values,
    )


def empty_tiled_scalar(static_parameters, dynamic_parameters, dtype=jnp.float64):
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    g = int(static_parameters.guard_cells)
    ntx = int(dynamic_parameters.Nx) // tile_nx
    nty = int(dynamic_parameters.Ny) // tile_ny
    ntz = int(dynamic_parameters.Nz) // tile_nz
    return jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=dtype,
    )


def empty_tiled_vector(static_parameters, dynamic_parameters, dtype=jnp.float64):
    return tuple(empty_tiled_scalar(static_parameters, dynamic_parameters, dtype=dtype) for _ in range(3))


def initialized_fields(static_parameters, dynamic_parameters, dtype=jnp.float64):
    E = empty_tiled_vector(static_parameters, dynamic_parameters, dtype=dtype)
    B = empty_tiled_vector(static_parameters, dynamic_parameters, dtype=dtype)
    J = empty_tiled_vector(static_parameters, dynamic_parameters, dtype=dtype)
    phi = empty_tiled_scalar(static_parameters, dynamic_parameters, dtype=dtype)
    rho = empty_tiled_scalar(static_parameters, dynamic_parameters, dtype=dtype)
    return E, B, J, phi, rho


def retiled_parameters(static_parameters, dynamic_parameters, tile_shape, guard_cells=None, **static_updates):
    tile_shape = tuple(int(width) for width in tile_shape)
    if guard_cells is None:
        guard_cells = int(static_parameters.guard_cells)

    tile_grid_shape = (
        int(dynamic_parameters.Nx) // tile_shape[0],
        int(dynamic_parameters.Ny) // tile_shape[1],
        int(dynamic_parameters.Nz) // tile_shape[2],
    )
    static_parameters = static_parameters._replace(
        tile_shape=tile_shape,
        guard_cells=int(guard_cells),
        field_mesh=make_field_mesh(tile_grid_shape),
        **static_updates,
    )
    dynamic_parameters = dynamic_parameters._replace(
        grids=GridParameters(
            vertex=dynamic_parameters.grids.vertex,
            center=dynamic_parameters.grids.center,
            tiled_vertex_grid=(),
            tiled_center_grid=(),
        )
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
    dynamic_parameters = dynamic_parameters._replace(
        grids=GridParameters(
            vertex=dynamic_parameters.grids.vertex,
            center=dynamic_parameters.grids.center,
            tiled_vertex_grid=tiled_vertex_grid,
            tiled_center_grid=tiled_center_grid,
        )
    )
    return static_parameters, dynamic_parameters


def active_interior(static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    return (
        0,
        0,
        0,
        slice(g, g + int(dynamic_parameters.Nx)),
        slice(g, g + int(dynamic_parameters.Ny)),
        slice(g, g + int(dynamic_parameters.Nz)),
    )


def field_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    g = int(static_parameters.guard_cells if num_guard_cells is None else num_guard_cells)
    Nx = int(dynamic_parameters.Nx)
    Ny = int(dynamic_parameters.Ny)
    Nz = int(dynamic_parameters.Nz)
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    source_gx = (int(field.shape[0]) - Nx) // 2
    source_gy = (int(field.shape[1]) - Ny) // 2
    source_gz = (int(field.shape[2]) - Nz) // 2
    interior = field[
        source_gx:source_gx + Nx,
        source_gy:source_gy + Ny,
        source_gz:source_gz + Nz,
    ]
    interior_tiles = interior.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
    interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)

    field_tiles = jnp.zeros(
        (
            ntx,
            nty,
            ntz,
            tile_nx + 2 * g,
            tile_ny + 2 * g,
            tile_nz + 2 * g,
        ),
        dtype=field.dtype,
    )
    field_tiles = field_tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)
    return update_tiled_ghost_cells(field_tiles, static_parameters, g)


def vector_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return tuple(
        field_tiles_from_global(component, static_parameters, dynamic_parameters, num_guard_cells)
        for component in field
    )


def particle_species(
    name,
    charge,
    mass,
    *,
    weight=1.0,
    x1,
    x2=None,
    x3=None,
    u1=None,
    u2=None,
    u3=None,
    v1=None,
    v2=None,
    v3=None,
    active_mask=None,
    update_pos=True,
    update_v=True,
    update_x=True,
    update_y=True,
    update_z=True,
    update_u=None,
    update_vx=True,
    update_vy=True,
    update_vz=True,
):
    x1 = jnp.asarray(x1, dtype=float)
    n_particles = int(x1.shape[0])

    if u1 is None and v1 is not None:
        u1 = v1
    if u2 is None and v2 is not None:
        u2 = v2
    if u3 is None and v3 is not None:
        u3 = v3

    if x2 is None:
        x2 = jnp.zeros(n_particles)
    if x3 is None:
        x3 = jnp.zeros(n_particles)
    if u1 is None:
        u1 = jnp.zeros(n_particles)
    if u2 is None:
        u2 = jnp.zeros(n_particles)
    if u3 is None:
        u3 = jnp.zeros(n_particles)
    if active_mask is None:
        active_mask = jnp.ones(n_particles, dtype=bool)

    if isinstance(update_x, (tuple, list)):
        update_x_components = tuple(update_x)
    else:
        update_x_components = (
            bool(update_pos and update_x),
            bool(update_pos and update_y),
            bool(update_pos and update_z),
        )

    if update_u is None:
        update_u_components = (
            bool(update_v and update_vx),
            bool(update_v and update_vy),
            bool(update_v and update_vz),
        )
    elif isinstance(update_u, (tuple, list)):
        update_u_components = tuple(update_u)
    else:
        update_u_components = (bool(update_u), bool(update_u), bool(update_u))

    return {
        "name": name,
        "charge": charge,
        "mass": mass,
        "weight": weight,
        "x": jnp.stack(
            (
                x1,
                jnp.asarray(x2, dtype=float),
                jnp.asarray(x3, dtype=float),
            ),
            axis=-1,
        ),
        "u": jnp.stack(
            (
                jnp.asarray(u1, dtype=float),
                jnp.asarray(u2, dtype=float),
                jnp.asarray(u3, dtype=float),
            ),
            axis=-1,
        ),
        "active": jnp.asarray(active_mask, dtype=bool),
        "update_x": update_x_components,
        "update_u": update_u_components,
    }


def _particle_tile_indices(x, dynamic_parameters, tile_shape):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]

    x_cell = np.floor((np.asarray(x[:, 0]) + float(dynamic_parameters.x_wind) / 2.0) / float(dynamic_parameters.dx)).astype(int)
    y_cell = np.floor((np.asarray(x[:, 1]) + float(dynamic_parameters.y_wind) / 2.0) / float(dynamic_parameters.dy)).astype(int)
    z_cell = np.floor((np.asarray(x[:, 2]) + float(dynamic_parameters.z_wind) / 2.0) / float(dynamic_parameters.dz)).astype(int)

    x_cell = np.clip(x_cell, 0, int(dynamic_parameters.Nx) - 1)
    y_cell = np.clip(y_cell, 0, int(dynamic_parameters.Ny) - 1)
    z_cell = np.clip(z_cell, 0, int(dynamic_parameters.Nz) - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def build_tiled_particles(
    species,
    static_parameters,
    dynamic_parameters=None,
    capacity_factor=None,
):
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    if capacity_factor is None:
        capacity_factor = static_parameters.particle_tile_capacity_factor

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx = int(math.ceil(int(dynamic_parameters.Nx) / tile_nx))
    nty = int(math.ceil(int(dynamic_parameters.Ny) / tile_ny))
    ntz = int(math.ceil(int(dynamic_parameters.Nz) / tile_nz))
    n_species = len(species)

    tile_counts = np.zeros((ntx, nty, ntz, n_species), dtype=int)
    particle_tile_data = []
    n_tiles = ntx * nty * ntz

    for species_index, species_data in enumerate(species):
        x = np.asarray(species_data["x"], dtype=float)
        u = np.asarray(species_data["u"], dtype=float)
        active = np.asarray(species_data["active"], dtype=bool)
        tx, ty, tz = _particle_tile_indices(x, dynamic_parameters, tile_shape)
        flat_tile = (tx * nty + ty) * ntz + tz
        particle_indices = np.arange(x.shape[0])

        flat_counts = np.bincount(flat_tile[particle_indices], minlength=n_tiles)
        tile_counts[:, :, :, species_index] = flat_counts.reshape((ntx, nty, ntz))
        particle_tile_data.append((x, u, active, tx, ty, tz, flat_tile, particle_indices))

    max_particles_per_tile = int(np.max(tile_counts)) if tile_counts.size else 0
    max_particles_per_tile = max(1, int(math.ceil(max_particles_per_tile * float(capacity_factor))))

    x_tiles = np.zeros((ntx, nty, ntz, n_species, max_particles_per_tile, 3), dtype=float)
    u_tiles = np.zeros_like(x_tiles)
    active_tiles = np.zeros((ntx, nty, ntz, n_species, max_particles_per_tile), dtype=bool)

    for species_index, (x, u, active, tx, ty, tz, flat_tile, all_particle_indices) in enumerate(particle_tile_data):
        order = np.argsort(flat_tile[all_particle_indices], kind="stable")
        particle_indices = all_particle_indices[order]
        sorted_flat_tile = flat_tile[particle_indices]

        flat_counts = tile_counts[:, :, :, species_index].reshape(-1)
        tile_starts = np.cumsum(flat_counts) - flat_counts
        slots = np.arange(particle_indices.size) - tile_starts[sorted_flat_tile]

        x_tiles[tx[particle_indices], ty[particle_indices], tz[particle_indices], species_index, slots, :] = x[particle_indices]
        u_tiles[tx[particle_indices], ty[particle_indices], tz[particle_indices], species_index, slots, :] = u[particle_indices]
        active_tiles[tx[particle_indices], ty[particle_indices], tz[particle_indices], species_index, slots] = active[particle_indices]

    particles = TiledParticles(
        x=jnp.asarray(x_tiles),
        u=jnp.asarray(u_tiles),
        active=jnp.asarray(active_tiles),
    )
    species_config = SpeciesConfig(
        charge=jnp.asarray([species_data["charge"] for species_data in species], dtype=float),
        mass=jnp.asarray([species_data["mass"] for species_data in species], dtype=float),
        weight=jnp.asarray([species_data["weight"] for species_data in species], dtype=float),
        update_x=jnp.asarray([species_data["update_x"] for species_data in species], dtype=bool),
        update_u=jnp.asarray([species_data["update_u"] for species_data in species], dtype=bool),
    )

    return particles, species_config


def species_names(species):
    return tuple(species_data["name"] for species_data in species)
