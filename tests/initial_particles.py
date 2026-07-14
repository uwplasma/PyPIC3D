import math

import jax.numpy as jnp
import numpy as np

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


def tiled_species(
    name,
    charge,
    mass,
    weight=1.0,
    N_particles=None,
    T=None,
    x1=None,
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
    **unused,
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


def _tile_axis_count(n_cells, cells_per_tile):
    return int(math.ceil(int(n_cells) / int(cells_per_tile)))


def _tile_indices(x, world, tile_shape):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]

    x_cell = np.floor((np.asarray(x[:, 0]) + float(world["x_wind"]) / 2.0) / float(world["dx"])).astype(int)
    y_cell = np.floor((np.asarray(x[:, 1]) + float(world["y_wind"]) / 2.0) / float(world["dy"])).astype(int)
    z_cell = np.floor((np.asarray(x[:, 2]) + float(world["z_wind"]) / 2.0) / float(world["dz"])).astype(int)

    x_cell = np.clip(x_cell, 0, int(world["Nx"]) - 1)
    y_cell = np.clip(y_cell, 0, int(world["Ny"]) - 1)
    z_cell = np.clip(z_cell, 0, int(world["Nz"]) - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def tile_shape_from_parameters(simulation_parameters):
    return (
        int(simulation_parameters["particle_tile_nx"]),
        int(simulation_parameters["particle_tile_ny"]),
        int(simulation_parameters["particle_tile_nz"]),
    )


def build_tiled_particles(species, world, tile_shape=None, simulation_parameters=None, capacity_factor=None):
    if tile_shape is not None and isinstance(tile_shape, dict):
        simulation_parameters = tile_shape
        tile_shape = None

    if tile_shape is None:
        tile_shape = tile_shape_from_parameters(simulation_parameters)
    if capacity_factor is None:
        capacity_factor = 1.0
        if simulation_parameters is not None:
            capacity_factor = float(simulation_parameters.get("particle_tile_capacity_factor", 1.0))

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)
    n_species = len(species)

    tile_counts = np.zeros((ntx, nty, ntz, n_species), dtype=int)
    particle_tile_data = []
    n_tiles = ntx * nty * ntz

    for species_index, species_data in enumerate(species):
        x = np.asarray(species_data["x"], dtype=float)
        u = np.asarray(species_data["u"], dtype=float)
        active = np.asarray(species_data["active"], dtype=bool)
        tx, ty, tz = _tile_indices(x, world, tile_shape)
        flat_tile = (tx * nty + ty) * ntz + tz
        particle_indices = np.arange(x.shape[0])

        flat_counts = np.bincount(flat_tile[particle_indices], minlength=n_tiles)
        tile_counts[:, :, :, species_index] = flat_counts.reshape((ntx, nty, ntz))
        particle_tile_data.append((x, u, active, tx, ty, tz, flat_tile, particle_indices))

    max_particles_per_tile = int(np.max(tile_counts)) if tile_counts.size else 0
    max_particles_per_tile = int(math.ceil(max_particles_per_tile * float(capacity_factor)))

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
