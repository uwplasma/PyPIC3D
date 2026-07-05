import numpy as np
import jax
import jax.numpy as jnp
import math

from PyPIC3D.utils import vth_to_T, T_to_vth
from PyPIC3D.particles.tiled_particles import SpeciesConfig, TiledParticles


def grab_particle_keys(config):
    """Return keys in a TOML config that start with 'particle'."""
    particle_keys = []
    for key in config.keys():
        if key[:8] == "particle":
            particle_keys.append(key)
    return particle_keys


def read_value(param, key, config, default_value):
    if param in config[key]:
        print(f"Reading user defined {param}")
        return config[key][param]
    return default_value


def load_initial_positions(param, config, key, default, N_particles, ds, ns, key1):
    if param in config[key]:
        if isinstance(config[key][param], str):
            print(f"Loading {param} from external source: {config[key][param]}")
            return jnp.load(config[key][param])
        val = config[key][param]
        if ns == 1:
            return val * jnp.ones(N_particles)
        return jax.random.uniform(
            key1, shape=(N_particles,), minval=val - (ds / 2), maxval=val + (ds / 2)
        )
    return default


def load_initial_velocities(param, config, key, default, N_particles):
    if param in config[key]:
        if isinstance(config[key][param], str):
            print(f"Loading {param} from external source: {config[key][param]}")
            return jnp.load(config[key][param])
        return jnp.full(N_particles, config[key][param]) + default
    return default


def compute_macroparticle_weight(config, particle_keys, simulation_parameters, world, constants):
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    kb = constants["kb"]
    eps = constants["eps"]

    if simulation_parameters["ds_per_debye"]:
        ds_per_debye = simulation_parameters["ds_per_debye"]
        inverse_total_debye = 0

        for toml_key in particle_keys:
            N_particles = config[toml_key]["N_particles"]
            charge = config[toml_key]["charge"]
            mass = config[toml_key]["mass"]
            if "temperature" in config[toml_key]:
                T = config[toml_key]["temperature"]
            elif "vth" in config[toml_key]:
                T = vth_to_T(config[toml_key]["vth"], mass, kb)

            inverse_total_debye += (
                jnp.sqrt(N_particles / (x_wind * y_wind * z_wind) / (eps * kb * T))
                * jnp.abs(charge)
            )

        ds2 = 0
        for d in [dx, dy, dz]:
            if d != 1:
                ds2 += d**2

        if ds2 == 0:
            raise ValueError(
                "Invalid configuration for 'ds_per_debye': at least one of dx, dy, dz must differ from 1."
            )
        weight = 1 / (ds2) / (ds_per_debye**2) / inverse_total_debye
    else:
        weight = 1.0

    return weight


def initial_particles(
    N_per_cell,
    N_particles,
    minx,
    maxx,
    miny,
    maxy,
    minz,
    maxz,
    mass,
    Tx,
    Ty,
    Tz,
    kb,
    key1,
    key2,
    key3,
):
    x = jax.random.uniform(key1, shape=(N_particles,), minval=minx, maxval=maxx)
    y = jax.random.uniform(key2, shape=(N_particles,), minval=miny, maxval=maxy)
    z = jax.random.uniform(key3, shape=(N_particles,), minval=minz, maxval=maxz)

    std_x = T_to_vth(Tx, mass, kb)
    std_y = T_to_vth(Ty, mass, kb)
    std_z = T_to_vth(Tz, mass, kb)
    vx = np.random.normal(0, std_x, N_particles)
    vy = np.random.normal(0, std_y, N_particles)
    vz = np.random.normal(0, std_z, N_particles)

    return x, y, z, vx, vy, vz


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


def _scaled_plasma_frequency(N_particles, charge, mass, weight, world, constants):
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]
    eps = constants["eps"]

    sqrt_dv = jnp.sqrt(x_wind * y_wind * z_wind)
    sqrt_ne = jnp.sqrt(N_particles * weight) / sqrt_dv
    sqrt_eps = jnp.sqrt(eps)
    sqrt_mass = jnp.sqrt(mass)
    return sqrt_ne * jnp.abs(charge) / (sqrt_eps * sqrt_mass)


def _scaled_debye_length(N_particles, charge, temperature, weight, world, constants):
    eps = constants["eps"]
    kb = constants["kb"]
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]

    density = weight * N_particles / (x_wind * y_wind * z_wind)
    return jnp.sqrt(eps * kb * temperature / (density * charge**2))


def load_particles_from_toml(config, simulation_parameters, world, constants):
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]
    Nx = world["Nx"]
    Ny = world["Ny"]
    Nz = world["Nz"]
    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    dt = world["dt"]
    kb = constants["kb"]
    eps = constants["eps"]

    i = 0
    particle_keys = grab_particle_keys(config)
    species_arrays = []
    species_metadata = []

    weight = compute_macroparticle_weight(
        config, particle_keys, simulation_parameters, world, constants
    )

    for toml_key in particle_keys:
        key1, key2, key3 = jax.random.key(i), jax.random.key(i + 1), jax.random.key(i + 2)
        i += 3

        particle_name = config[toml_key]["name"]
        print(f"\nInitializing particle species: {particle_name}")
        charge = config[toml_key]["charge"]
        mass = config[toml_key]["mass"]

        if "N_particles" in config[toml_key]:
            N_particles = config[toml_key]["N_particles"]
            N_per_cell = N_particles / (world["Nx"] * world["Ny"] * world["Nz"])
        elif "N_per_cell" in config[toml_key]:
            N_per_cell = config[toml_key]["N_per_cell"]
            N_particles = int(N_per_cell * world["Nx"] * world["Ny"] * world["Nz"])

        if "temperature" in config[toml_key]:
            T = config[toml_key]["temperature"]
            vth = T_to_vth(T, mass, kb)
        elif "vth" in config[toml_key]:
            vth = config[toml_key]["vth"]
            T = vth_to_T(vth, mass, kb)
        else:
            T = 1.0
            vth = T_to_vth(T, mass, kb)

        Tx = read_value("Tx", toml_key, config, T)
        Ty = read_value("Ty", toml_key, config, T)
        Tz = read_value("Tz", toml_key, config, T)

        xmin = read_value("xmin", toml_key, config, -x_wind / 2)
        xmax = read_value("xmax", toml_key, config, x_wind / 2)
        ymin = read_value("ymin", toml_key, config, -y_wind / 2)
        ymax = read_value("ymax", toml_key, config, y_wind / 2)
        zmin = read_value("zmin", toml_key, config, -z_wind / 2)
        zmax = read_value("zmax", toml_key, config, z_wind / 2)

        x, y, z, vx, vy, vz = initial_particles(
            N_per_cell,
            N_particles,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            mass,
            Tx,
            Ty,
            Tz,
            kb,
            key1,
            key2,
            key3,
        )

        x_bc = "periodic"
        if "x_bc" in config[toml_key]:
            assert config[toml_key]["x_bc"] in ["periodic", "reflecting", "absorbing"], (
                f"Invalid x boundary condition: {config[toml_key]['x_bc']}"
            )
            x_bc = config[toml_key]["x_bc"]

        y_bc = "periodic"
        if "y_bc" in config[toml_key]:
            assert config[toml_key]["y_bc"] in ["periodic", "reflecting", "absorbing"], (
                f"Invalid y boundary condition: {config[toml_key]['y_bc']}"
            )
            y_bc = config[toml_key]["y_bc"]

        z_bc = "periodic"
        if "z_bc" in config[toml_key]:
            assert config[toml_key]["z_bc"] in ["periodic", "reflecting", "absorbing"], (
                f"Invalid z boundary condition: {config[toml_key]['z_bc']}"
            )
            z_bc = config[toml_key]["z_bc"]

        x = load_initial_positions("initial_x", config, toml_key, x, N_particles, dx, Nx, key1)
        y = load_initial_positions("initial_y", config, toml_key, y, N_particles, dy, Ny, key2)
        z = load_initial_positions("initial_z", config, toml_key, z, N_particles, dz, Nz, key3)

        vx = load_initial_velocities("initial_vx", config, toml_key, vx, N_particles)
        vy = load_initial_velocities("initial_vy", config, toml_key, vy, N_particles)
        vz = load_initial_velocities("initial_vz", config, toml_key, vz, N_particles)

        if "temperature" not in config[toml_key]:
            T = (mass / (3 * kb * N_particles)) * (
                jnp.sum(vx**2) + jnp.sum(vy**2) + jnp.sum(vz**2)
            )

        if "weight" in config[toml_key]:
            weight = config[toml_key]["weight"]
        elif "ds_per_debye" in config[toml_key]:
            ds_per_debye = config[toml_key]["ds_per_debye"]

            ds2 = 0
            for d in [dx, dy, dz]:
                if d != 1:
                    ds2 += d**2

            if ds2 == 0:
                raise ValueError(
                    "Invalid configuration for 'ds_per_debye': at least one of dx, dy, dz must differ from 1."
                )
            weight = (
                x_wind * y_wind * z_wind * eps * kb * T
            ) / (N_particles * charge**2 * ds_per_debye**2 * ds2)

        update_pos = read_value("update_pos", toml_key, config, True)
        update_v = read_value("update_v", toml_key, config, True)
        update_vx = read_value("update_vx", toml_key, config, True)
        update_vy = read_value("update_vy", toml_key, config, True)
        update_vz = read_value("update_vz", toml_key, config, True)
        update_x = read_value("update_x", toml_key, config, True)
        update_y = read_value("update_y", toml_key, config, True)
        update_z = read_value("update_z", toml_key, config, True)

        x_array = jnp.stack((x, y, z), axis=-1)
        u_array = jnp.stack((vx, vy, vz), axis=-1)
        active_mask = jnp.ones((N_particles,), dtype=bool)

        metadata = {
            "name": particle_name,
            "N_particles": int(N_particles),
            "N_per_cell": N_per_cell,
            "charge": charge,
            "mass": mass,
            "temperature": T,
            "thermal_velocity": vth,
            "weight": weight,
            "x_bc": x_bc,
            "y_bc": y_bc,
            "z_bc": z_bc,
            "update_x": (update_pos and update_x, update_pos and update_y, update_pos and update_z),
            "update_u": (update_v and update_vx, update_v and update_vy, update_v and update_vz),
        }
        species_arrays.append((x_array, u_array, active_mask))
        species_metadata.append(metadata)

        kinetic_energy = 0.5 * weight * mass * jnp.sum(vx**2 + vy**2 + vz**2)
        pf = _scaled_plasma_frequency(N_particles, charge, mass, weight, world, constants)
        dl = _scaled_debye_length(N_particles, charge, T, weight, world, constants)

        print(f"Number of particles: {N_particles}")
        print(f"Number of particles per cell: {N_per_cell}")
        print(f"x, y, z boundary conditions: {x_bc}, {y_bc}, {z_bc}")
        print(f"Charge: {charge}")
        print(f"Mass: {mass}")
        print(f"Temperature: {T}")
        print(f"Thermal Velocity: {vth}")
        print(f"Particle Kinetic Energy: {kinetic_energy}")
        print(f"Particle Species Plasma Frequency: {pf}")
        print(f"Time Steps Per Plasma Period: {(1 / (dt * pf))}")
        print(f"Particle Species Debye Length: {dl}")
        print(f"Particle Weight: {weight}")
        print(f"Particle Species Scaled Charge: {charge * weight}")
        print(f"Particle Species Scaled Mass: {mass * weight}")

    tile_nx, tile_ny, tile_nz = [_as_int(width) for width in world["tile_shape"]]

    ntx = _tile_axis_count(world["Nx"], tile_nx)
    nty = _tile_axis_count(world["Ny"], tile_ny)
    ntz = _tile_axis_count(world["Nz"], tile_nz)
    n_species = len(species_arrays)

    tile_counts = np.zeros((ntx, nty, ntz, n_species), dtype=int)
    particle_tile_indices = []

    for species_index, (x_array, _u_array, active_mask) in enumerate(species_arrays):
        tx, ty, tz = _particle_tile_indices(x_array[:, 0], x_array[:, 1], x_array[:, 2], world, tile_nx, tile_ny, tile_nz)
        particle_tile_indices.append((tx, ty, tz))

        for p in range(int(active_mask.shape[0])):
            if bool(active_mask[p]):
                tile_counts[tx[p], ty[p], tz[p], species_index] += 1

    max_particles_per_tile = int(np.max(tile_counts)) if tile_counts.size else 0
    capacity_factor = float(simulation_parameters.get("particle_tile_capacity_factor", 1.0))
    max_particles_per_tile = int(math.ceil(max_particles_per_tile * capacity_factor))

    x_tiles = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile, 3))
    u_tiles = jnp.zeros_like(x_tiles)
    active_tiles = jnp.zeros((ntx, nty, ntz, n_species, max_particles_per_tile), dtype=bool)

    next_slot = np.zeros_like(tile_counts)

    for species_index, (x_array, u_array, active_mask) in enumerate(species_arrays):
        tx, ty, tz = particle_tile_indices[species_index]

        for p in range(int(active_mask.shape[0])):
            if not bool(active_mask[p]):
                continue
            tile_index = (tx[p], ty[p], tz[p], species_index)
            slot = next_slot[tile_index]
            next_slot[tile_index] += 1
            index = tile_index + (slot,)

            x_tiles = x_tiles.at[index].set(x_array[p])
            u_tiles = u_tiles.at[index].set(u_array[p])
            active_tiles = active_tiles.at[index].set(True)

    species_config = SpeciesConfig(
        charge=jnp.asarray([metadata["charge"] for metadata in species_metadata]),
        mass=jnp.asarray([metadata["mass"] for metadata in species_metadata]),
        weight=jnp.asarray([metadata["weight"] for metadata in species_metadata]),
        update_x=jnp.asarray([metadata["update_x"] for metadata in species_metadata], dtype=bool),
        update_u=jnp.asarray([metadata["update_u"] for metadata in species_metadata], dtype=bool),
    )
    particles = TiledParticles(
        x=x_tiles,
        u=u_tiles,
        active=active_tiles,
    )
    species_names = tuple(metadata["name"] for metadata in species_metadata)

    return particles, species_config, species_names, tuple(species_metadata)
