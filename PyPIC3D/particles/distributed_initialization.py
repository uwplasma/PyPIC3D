"""Bounded native-particle input, routed directly to spatial tile owners."""

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import PARTICLE_STATE_TILE_SPEC, PARTICLE_ACTIVE_TILE_SPEC
from PyPIC3D.utilities.grids import grid_domain_bounds


def load_distributed_particles(config, static, dynamic, *, comm=None, chunk_size=262144):
    """Load original particle IDs in bounded blocks, preserving tile ordering.

    All six initial_x/y/z/vx/vy/vz entries must be native .npy files. Built-in
    random generation is deliberately unsupported here: changing random keys
    per process would silently change the physical initial condition.
    """
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    mesh = static.field_mesh
    counts = tuple(int(n) for n in mesh.devices.shape)
    widths = np.asarray(static.tile_shape, dtype=np.int64)
    shape = np.asarray([int(getattr(dynamic, "N" + axis)) for axis in "xyz"])
    spacing = np.asarray([float(getattr(dynamic, "d" + axis)) for axis in "xyz"])
    minimum = np.asarray([float(bounds[0]) for bounds in grid_domain_bounds(dynamic)])
    owners = np.asarray([device.process_index for device in mesh.devices.flat])
    blocks = [value for name, value in (config or {}).items() if name.startswith("particle")]
    occupancy = np.zeros((math.prod(counts), len(blocks)), dtype=np.int64)
    local_particles, metadata = [], []
    weight = 1.0

    for species, block in enumerate(blocks):
        if "N_particles" in block:
            n = int(block["N_particles"])
            per_cell = n / math.prod(shape)
        else:
            per_cell = float(block.get("N_per_cell", 0))
            n = int(per_cell*math.prod(shape))
        keys = ("initial_x", "initial_y", "initial_z", "initial_vx", "initial_vy", "initial_vz")
        if n and any(not isinstance(block.get(key), str) for key in keys):
            raise ValueError("Distributed particles require six native .npy initial_x/y/z/vx/vy/vz files; "
                             "built-in random particle generation is not supported")
        arrays = [np.load(block[key], mmap_mode="r", allow_pickle=False) for key in keys] if n else []
        if any(value.shape != (n,) for value in arrays):
            raise ValueError("Native particle arrays must each have shape (N_particles,)")
        received, velocity_square = [], 0.0
        # Every rank enters the same number of exchanges, including ranks
        # without particles in the final block or in an entire species.
        rounds = math.ceil(n / (size*chunk_size))
        for turn in range(rounds):
            start = (turn*size + rank)*chunk_size
            end = min(start + chunk_size, n)
            ids = np.arange(start, max(start, end), dtype=np.int64)
            values = [np.asarray(a[start:end], dtype=np.float64) for a in arrays]
            x = np.stack(values[:3], axis=-1)
            u = np.stack(values[3:], axis=-1)
            velocity_square += float(np.sum(u*u))
            cells = np.clip(np.floor((x-minimum)/spacing).astype(np.int64), 0, shape-1)
            tiles = cells // widths
            tile_ids = np.ravel_multi_index(tiles.T, counts)
            destinations = owners[tile_ids]
            packets = [(ids[destinations == dest], tile_ids[destinations == dest],
                        x[destinations == dest], u[destinations == dest]) for dest in range(size)]
            incoming = packets if comm is None else comm.alltoall(packets)
            received.extend(packet for packet in incoming if len(packet[0]))

        if received:
            ids, tiles, x, u = (np.concatenate([packet[k] for packet in received]) for k in range(4))
            order = np.argsort(ids, kind="stable")
            tiles, x, u = tiles[order], x[order], u[order]
            occupancy[:, species] = np.bincount(tiles, minlength=math.prod(counts))
        else:
            tiles, x, u = np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty((0, 3))
        local_particles.append((tiles, x, u))
        if comm is not None:
            velocity_square = comm.allreduce(velocity_square)
        mass, charge, kb = float(block["mass"]), float(block["charge"]), float(dynamic.kb)
        temperature = float(block.get("temperature", mass*velocity_square/(3*kb*n) if n else 0.0))
        # Preserve the serial metadata convention: vth describes the requested
        # distribution even when native files subsequently determine temperature.
        thermal_velocity = (math.sqrt(kb*float(block["temperature"])/mass) if "temperature" in block
                            else float(block.get("vth", math.sqrt(kb/mass))))
        if "weight" in block:
            weight = float(block["weight"])
        elif "number_density" in block:
            weight = float(block["number_density"])*float(np.prod(spacing))/per_cell
        metadata.append(dict(name=block["name"], N_particles=n, N_per_cell=per_cell,
                             charge=charge, mass=mass, temperature=temperature,
                             thermal_velocity=thermal_velocity, weight=weight,
                             x_bc=block.get("x_bc", "periodic"), y_bc=block.get("y_bc", "periodic"),
                             z_bc=block.get("z_bc", "periodic"),
                             update_x=tuple(bool(block.get("update_" + axis, True)) for axis in "xyz")))

    if comm is not None:
        from mpi4py import MPI
        global_occupancy = np.empty_like(occupancy)
        comm.Allreduce(occupancy, global_occupancy, op=MPI.SUM)
        occupancy = global_occupancy
    capacity = int(math.ceil(int(occupancy.max(initial=0))*float(static.particle_tile_capacity_factor)))
    local = {}
    for tile_id in np.flatnonzero(owners == rank):
        positions = np.zeros((len(blocks), capacity, 3))
        momenta = np.zeros_like(positions)
        active = np.zeros((len(blocks), capacity), dtype=bool)
        for species, (tiles, x, u) in enumerate(local_particles):
            selected = tiles == tile_id
            count = int(occupancy[tile_id, species])
            positions[species, :count], momenta[species, :count] = x[selected], u[selected]
            active[species, :count] = True
        local[np.unravel_index(tile_id, counts)] = (positions, momenta, active)

    def component(index, which):
        tile = tuple(part.start or 0 for part in index[:3])
        return local[tile][which][None, None, None]
    state_shape = counts + (len(blocks), capacity, 3)
    state_sharding = NamedSharding(mesh, PARTICLE_STATE_TILE_SPEC)
    active_sharding = NamedSharding(mesh, PARTICLE_ACTIVE_TILE_SPEC)
    particles = TiledParticles(
        jax.make_array_from_callback(state_shape, state_sharding, lambda i: component(i, 0), dtype=np.float64),
        jax.make_array_from_callback(state_shape, state_sharding, lambda i: component(i, 1), dtype=np.float64),
        jax.make_array_from_callback(state_shape[:-1], active_sharding, lambda i: component(i, 2), dtype=np.bool_))
    species_config = SpeciesConfig(
        charge=jnp.asarray([m["charge"] for m in metadata]), mass=jnp.asarray([m["mass"] for m in metadata]),
        weight=jnp.asarray([m["weight"] for m in metadata]),
        update_x=jnp.asarray([m["update_x"] for m in metadata], dtype=bool).reshape((-1, 3)))
    return particles, species_config, tuple(m["name"] for m in metadata), tuple(metadata)
