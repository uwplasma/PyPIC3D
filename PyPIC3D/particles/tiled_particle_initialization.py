import math

import jax.numpy as jnp
import numpy as np

from PyPIC3D.particles.particle_initialization import load_particles_from_toml
from PyPIC3D.particles.tiled_particles import TiledParticles


def _as_int(value):
    return int(jnp.asarray(value).item())


def _tile_axis_count(n_cells, cells_per_tile):
    return int(math.ceil(_as_int(n_cells) / _as_int(cells_per_tile)))


def _particle_tile_indices(x1, x2, x3, world, tile_nx, tile_ny, tile_nz):
    Nx = _as_int(world["Nx"])
    Ny = _as_int(world["Ny"])
    Nz = _as_int(world["Nz"])

    x_cell = np.floor((np.asarray(x1) + float(world["x_wind"]) / 2) / float(world["dx"])).astype(int)
    y_cell = np.floor((np.asarray(x2) + float(world["y_wind"]) / 2) / float(world["dy"])).astype(int)
    z_cell = np.floor((np.asarray(x3) + float(world["z_wind"]) / 2) / float(world["dz"])).astype(int)

    x_cell = np.clip(x_cell, 0, Nx - 1)
    y_cell = np.clip(y_cell, 0, Ny - 1)
    z_cell = np.clip(z_cell, 0, Nz - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def to_tiled_particles(particles, world, simulation_parameters):
    """
    Pack an existing particle species list into tile-major arrays.

    This routine only changes storage: particles are grouped by configurable
    spatial tiles with leading axes ``(ntx, nty, ntz, species, slot, ...)``.
    """

    tile_nx = _as_int(simulation_parameters.get("particle_tile_nx", 1))
    tile_ny = _as_int(simulation_parameters.get("particle_tile_ny", 1))
    tile_nz = _as_int(simulation_parameters.get("particle_tile_nz", 1))

    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)
    n_species = len(particles)

    tile_counts = np.zeros((ntx, nty, ntz, n_species), dtype=int)
    particle_tile_indices = []

    for species_index, species in enumerate(particles):
        x1, x2, x3 = species.get_forward_position()
        tx, ty, tz = _particle_tile_indices(x1, x2, x3, world, tile_nx, tile_ny, tile_nz)
        particle_tile_indices.append((tx, ty, tz))

        for p in range(_as_int(species.N_particles)):
            tile_counts[tx[p], ty[p], tz[p], species_index] += 1

    max_particles_per_tile = int(np.max(tile_counts)) if tile_counts.size else 0

    x = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile, 3))
    u = jnp.zeros_like(x)
    charge = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile))
    mass = jnp.zeros_like(charge)
    weight = jnp.zeros_like(charge)
    active = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile), dtype=bool)
    update_x1 = jnp.zeros_like(active)
    update_x2 = jnp.zeros_like(active)
    update_x3 = jnp.zeros_like(active)
    update_u1 = jnp.zeros_like(active)
    update_u2 = jnp.zeros_like(active)
    update_u3 = jnp.zeros_like(active)

    next_slot = np.zeros_like(tile_counts)

    for species_index, species in enumerate(particles):
        x1, x2, x3 = species.get_forward_position()
        u1, u2, u3 = species.get_velocity()
        tx, ty, tz = particle_tile_indices[species_index]

        for p in range(_as_int(species.N_particles)):
            tile_index = (tx[p], ty[p], tz[p], species_index)
            slot = next_slot[tile_index]
            next_slot[tile_index] += 1
            index = tile_index + (slot,)

            x = x.at[index + (0,)].set(x1[p])
            x = x.at[index + (1,)].set(x2[p])
            x = x.at[index + (2,)].set(x3[p])
            u = u.at[index + (0,)].set(u1[p])
            u = u.at[index + (1,)].set(u2[p])
            u = u.at[index + (2,)].set(u3[p])

            charge = charge.at[index].set(species.charge)
            mass = mass.at[index].set(species.mass)
            weight = weight.at[index].set(species.weight)
            active = active.at[index].set(species.active_mask[p])

            update_x1 = update_x1.at[index].set(species.update_pos and species.update_x)
            update_x2 = update_x2.at[index].set(species.update_pos and species.update_y)
            update_x3 = update_x3.at[index].set(species.update_pos and species.update_z)
            update_u1 = update_u1.at[index].set(species.update_v and species.update_vx)
            update_u2 = update_u2.at[index].set(species.update_v and species.update_vy)
            update_u3 = update_u3.at[index].set(species.update_v and species.update_vz)

    return TiledParticles(
        x=x,
        u=u,
        charge=charge,
        mass=mass,
        weight=weight,
        active=active,
        update_x1=update_x1,
        update_x2=update_x2,
        update_x3=update_x3,
        update_u1=update_u1,
        update_u2=update_u2,
        update_u3=update_u3,
    )


def load_tiled_particles_from_toml(config, simulation_parameters, world, constants):
    """
    Initialize particles from TOML input and pack them into tile-major arrays.

    The particle values are produced by the existing particle initializer.  This
    wrapper only changes storage by calling ``to_tiled_particles`` on the
    initialized species list.
    """

    particles = load_particles_from_toml(config, simulation_parameters, world, constants)
    return to_tiled_particles(particles, world, simulation_parameters)
