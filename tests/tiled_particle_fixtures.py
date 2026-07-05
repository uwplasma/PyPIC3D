import math

import jax.numpy as jnp
import numpy as np

from PyPIC3D.particles.tiled_particles import SpeciesConfig, TiledParticles


class particle_species:
    """
    Test-only particle constructor for building tiled particle fixtures.

    Production code no longer has a flat species class.  Several numerical tests
    still need a compact way to name positions, velocities, masks, and per-species
    constants before packing them into TiledParticles.
    """

    def __init__(
        self,
        name,
        N_particles,
        charge,
        mass,
        weight=1.0,
        T=0.0,
        x1=None,
        x2=None,
        x3=None,
        v1=None,
        v2=None,
        v3=None,
        xwind=1.0,
        ywind=1.0,
        zwind=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        dt=1.0,
        update_pos=True,
        update_x=True,
        update_y=True,
        update_z=True,
        update_v=True,
        update_vx=True,
        update_vy=True,
        update_vz=True,
        active_mask=None,
        x_bc="periodic",
        y_bc="periodic",
        z_bc="periodic",
    ):
        self.name = name
        self.N_particles = int(N_particles)
        self.charge = charge
        self.mass = mass
        self.weight = weight
        self.T = T
        self.xwind = xwind
        self.ywind = ywind
        self.zwind = zwind
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.x_bc = x_bc
        self.y_bc = y_bc
        self.z_bc = z_bc

        self.update_pos = update_pos
        self.update_x = update_pos and update_x
        self.update_y = update_pos and update_y
        self.update_z = update_pos and update_z
        self.update_v = update_v
        self.update_vx = update_v and update_vx
        self.update_vy = update_v and update_vy
        self.update_vz = update_v and update_vz

        self.x1 = jnp.asarray(x1 if x1 is not None else jnp.zeros(self.N_particles))
        self.x2 = jnp.asarray(x2 if x2 is not None else jnp.zeros(self.N_particles))
        self.x3 = jnp.asarray(x3 if x3 is not None else jnp.zeros(self.N_particles))
        self.v1 = jnp.asarray(v1 if v1 is not None else jnp.zeros(self.N_particles))
        self.v2 = jnp.asarray(v2 if v2 is not None else jnp.zeros(self.N_particles))
        self.v3 = jnp.asarray(v3 if v3 is not None else jnp.zeros(self.N_particles))

        if active_mask is None:
            self.active_mask = jnp.ones(self.N_particles, dtype=bool)
        else:
            self.active_mask = jnp.asarray(active_mask, dtype=bool)

    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge

    def get_mass(self):
        return self.mass

    def get_weight(self):
        return self.weight

    def get_temperature(self):
        return self.T

    def get_number_of_particles(self):
        return self.N_particles

    def get_velocity(self):
        return self.v1, self.v2, self.v3

    def get_active_mask(self):
        return self.active_mask

    def get_forward_position(self):
        return self.x1, self.x2, self.x3

    def _half_step_axis(self, x, v, wind, bc):
        x_diagnostic = x - v * self.dt / 2
        if bc == "periodic":
            half_wind = wind / 2
            x_diagnostic = jnp.where(
                x_diagnostic > half_wind,
                x_diagnostic - wind,
                jnp.where(x_diagnostic < -half_wind, x_diagnostic + wind, x_diagnostic),
            )
        return x_diagnostic

    def get_position(self):
        return (
            self._half_step_axis(self.x1, self.v1, self.xwind, self.x_bc),
            self._half_step_axis(self.x2, self.v2, self.ywind, self.y_bc),
            self._half_step_axis(self.x3, self.v3, self.zwind, self.z_bc),
        )

    def set_position(self, x1, x2, x3):
        self.x1 = jnp.asarray(x1)
        self.x2 = jnp.asarray(x2)
        self.x3 = jnp.asarray(x3)

    def set_velocity(self, v1, v2, v3):
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)
        self.v3 = jnp.asarray(v3)

    def set_mass(self, mass):
        self.mass = mass

    def set_weight(self, weight):
        self.weight = weight

    def update_position(self):
        self.x1 = jnp.where(self.update_x, self.x1 + self.v1 * self.dt, self.x1)
        self.x2 = jnp.where(self.update_y, self.x2 + self.v2 * self.dt, self.x2)
        self.x3 = jnp.where(self.update_z, self.x3 + self.v3 * self.dt, self.x3)

    def _apply_axis_bc(self, x, wind, bc):
        half_wind = wind / 2
        if int(bc) == 0:
            x = jnp.where(x > half_wind, x - wind, x)
            x = jnp.where(x < -half_wind, x + wind, x)
            return x, self.active_mask
        if int(bc) == 2:
            active = self.active_mask & (x >= -half_wind) & (x <= half_wind)
            return x, active
        return x, self.active_mask

    def boundary_conditions(self, world):
        particle_bc = world.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})
        self.x1, self.active_mask = self._apply_axis_bc(self.x1, self.xwind, particle_bc.get("x", 0))
        self.x2, self.active_mask = self._apply_axis_bc(self.x2, self.ywind, particle_bc.get("y", 0))
        self.x3, self.active_mask = self._apply_axis_bc(self.x3, self.zwind, particle_bc.get("z", 0))

    def kinetic_energy(self):
        v2 = self.v1**2 + self.v2**2 + self.v3**2
        return jnp.sum(self.active_mask.astype(v2.dtype) * 0.5 * self.mass * self.weight * v2)

    def momentum(self):
        vmag = jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2)
        return jnp.sum(self.active_mask.astype(vmag.dtype) * self.mass * self.weight * vmag)


def _tile_shape(world, simulation_parameters):
    if "tile_shape" in world:
        return tuple(int(width) for width in world["tile_shape"])
    return (
        int(simulation_parameters.get("particle_tile_nx", world["Nx"])),
        int(simulation_parameters.get("particle_tile_ny", world["Ny"])),
        int(simulation_parameters.get("particle_tile_nz", world["Nz"])),
    )


def _tile_axis_count(n_cells, cells_per_tile):
    return int(math.ceil(int(n_cells) / int(cells_per_tile)))


def _particle_tile_indices(x1, x2, x3, world, tile_shape):
    tile_nx, tile_ny, tile_nz = tile_shape
    Nx = int(world["Nx"])
    Ny = int(world["Ny"])
    Nz = int(world["Nz"])

    x_cell = np.floor((np.asarray(x1) + float(world["x_wind"]) / 2) / float(world["dx"])).astype(int)
    y_cell = np.floor((np.asarray(x2) + float(world["y_wind"]) / 2) / float(world["dy"])).astype(int)
    z_cell = np.floor((np.asarray(x3) + float(world["z_wind"]) / 2) / float(world["dz"])).astype(int)

    x_cell = np.clip(x_cell, 0, Nx - 1)
    y_cell = np.clip(y_cell, 0, Ny - 1)
    z_cell = np.clip(z_cell, 0, Nz - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def to_tiled_particles(particles, world, simulation_parameters):
    tile_shape = _tile_shape(world, simulation_parameters)
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)

    tile_counts = np.zeros((ntx, nty, ntz, len(particles)), dtype=int)
    species_data = []

    for species_index, species in enumerate(particles):
        x1, x2, x3 = species.get_forward_position()
        v1, v2, v3 = species.get_velocity()
        active = species.get_active_mask()
        tx, ty, tz = _particle_tile_indices(x1, x2, x3, world, tile_shape)

        for particle_index in range(species.get_number_of_particles()):
            tile_counts[tx[particle_index], ty[particle_index], tz[particle_index], species_index] += 1

        species_data.append((x1, x2, x3, v1, v2, v3, active, tx, ty, tz))

    capacity_factor = float(simulation_parameters.get("particle_tile_capacity_factor", 1.0))
    max_particles_per_tile = int(math.ceil(max(1, tile_counts.max()) * capacity_factor))

    x_tiles = np.zeros((ntx, nty, ntz, len(particles), max_particles_per_tile, 3), dtype=float)
    u_tiles = np.zeros_like(x_tiles)
    active_tiles = np.zeros((ntx, nty, ntz, len(particles), max_particles_per_tile), dtype=bool)
    write_counts = np.zeros_like(tile_counts)

    for species_index, (x1, x2, x3, v1, v2, v3, active, tx, ty, tz) in enumerate(species_data):
        for particle_index in range(particles[species_index].get_number_of_particles()):
            tile = (tx[particle_index], ty[particle_index], tz[particle_index], species_index)
            slot = write_counts[tile]
            write_counts[tile] += 1
            x_tiles[tile + (slot, slice(None))] = [
                float(x1[particle_index]),
                float(x2[particle_index]),
                float(x3[particle_index]),
            ]
            u_tiles[tile + (slot, slice(None))] = [
                float(v1[particle_index]),
                float(v2[particle_index]),
                float(v3[particle_index]),
            ]
            active_tiles[tile + (slot,)] = bool(active[particle_index])

    species_config = SpeciesConfig(
        charge=jnp.asarray([species.charge for species in particles]),
        mass=jnp.asarray([species.mass for species in particles]),
        weight=jnp.asarray([species.weight for species in particles]),
        update_x=jnp.asarray(
            [[species.update_x, species.update_y, species.update_z] for species in particles],
            dtype=bool,
        ),
        update_u=jnp.asarray(
            [[species.update_vx, species.update_vy, species.update_vz] for species in particles],
            dtype=bool,
        ),
    )

    return (
        TiledParticles(
            x=jnp.asarray(x_tiles),
            u=jnp.asarray(u_tiles),
            active=jnp.asarray(active_tiles),
        ),
        species_config,
    )
