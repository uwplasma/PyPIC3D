import numpy as np
import jax
import jax.numpy as jnp
import math

from PyPIC3D.utils import vth_to_T, T_to_vth
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


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


def _particle_tile_indices(x1, x2, x3, dynamic_parameters, tile_nx, tile_ny, tile_nz):
    Nx = _as_int(dynamic_parameters.Nx)
    Ny = _as_int(dynamic_parameters.Ny)
    Nz = _as_int(dynamic_parameters.Nz)

    x_cell = np.floor((np.asarray(x1) + float(dynamic_parameters.x_wind) / 2) / float(dynamic_parameters.dx)).astype(int)
    y_cell = np.floor((np.asarray(x2) + float(dynamic_parameters.y_wind) / 2) / float(dynamic_parameters.dy)).astype(int)
    z_cell = np.floor((np.asarray(x3) + float(dynamic_parameters.z_wind) / 2) / float(dynamic_parameters.dz)).astype(int)

    x_cell = np.clip(x_cell, 0, Nx - 1)
    y_cell = np.clip(y_cell, 0, Ny - 1)
    z_cell = np.clip(z_cell, 0, Nz - 1)

    return x_cell // tile_nx, y_cell // tile_ny, z_cell // tile_nz


def _prepare_species_tile_data(species_arrays, dynamic_parameters, tile_shape):
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx = _tile_axis_count(dynamic_parameters.Nx, tile_nx)
    nty = _tile_axis_count(dynamic_parameters.Ny, tile_ny)
    ntz = _tile_axis_count(dynamic_parameters.Nz, tile_nz)
    n_tiles = ntx * nty * ntz

    tile_counts = np.zeros((ntx, nty, ntz, len(species_arrays)), dtype=int)
    species_tile_data = []

    for species_index, (x_array, u_array, active_mask) in enumerate(species_arrays):
        x_np = np.asarray(x_array)
        u_np = np.asarray(u_array)
        active_np = np.asarray(active_mask, dtype=bool)

        tx, ty, tz = _particle_tile_indices(
            x_np[:, 0],
            x_np[:, 1],
            x_np[:, 2],
            dynamic_parameters,
            tile_nx,
            tile_ny,
            tile_nz,
        )
        flat_tile = (tx * nty + ty) * ntz + tz
        active_indices = np.nonzero(active_np)[0]

        flat_counts = np.bincount(flat_tile[active_indices], minlength=n_tiles)
        tile_counts[:, :, :, species_index] = flat_counts.reshape((ntx, nty, ntz))

        species_tile_data.append((x_np, u_np, active_np, tx, ty, tz, flat_tile, active_indices))

    return tile_counts, species_tile_data


def _pack_species_arrays_into_tiles(species_arrays, dynamic_parameters, tile_shape, capacity_factor):
    tile_nx, tile_ny, tile_nz = tile_shape
    ntx = _tile_axis_count(dynamic_parameters.Nx, tile_nx)
    nty = _tile_axis_count(dynamic_parameters.Ny, tile_ny)
    ntz = _tile_axis_count(dynamic_parameters.Nz, tile_nz)
    n_species = len(species_arrays)

    tile_counts, species_tile_data = _prepare_species_tile_data(species_arrays, dynamic_parameters, tile_shape)
    max_particles_per_tile = int(np.max(tile_counts)) if tile_counts.size else 0
    max_particles_per_tile = int(math.ceil(max_particles_per_tile * capacity_factor))

    x_tiles = np.zeros((ntx, nty, ntz, n_species, max_particles_per_tile, 3), dtype=float)
    u_tiles = np.zeros_like(x_tiles)
    active_tiles = np.zeros((ntx, nty, ntz, n_species, max_particles_per_tile), dtype=bool)

    for species_index, (x_np, u_np, active_np, tx, ty, tz, flat_tile, active_indices) in enumerate(species_tile_data):
        if active_indices.size == 0:
            continue

        order = np.argsort(flat_tile[active_indices], kind="stable")
        particle_indices = active_indices[order]
        sorted_flat_tile = flat_tile[particle_indices]

        flat_counts = tile_counts[:, :, :, species_index].reshape(-1)
        tile_starts = np.cumsum(flat_counts) - flat_counts
        slots = np.arange(particle_indices.size) - tile_starts[sorted_flat_tile]

        x_tiles[
            tx[particle_indices],
            ty[particle_indices],
            tz[particle_indices],
            species_index,
            slots,
            :,
        ] = x_np[particle_indices]
        u_tiles[
            tx[particle_indices],
            ty[particle_indices],
            tz[particle_indices],
            species_index,
            slots,
            :,
        ] = u_np[particle_indices]
        active_tiles[
            tx[particle_indices],
            ty[particle_indices],
            tz[particle_indices],
            species_index,
            slots,
        ] = active_np[particle_indices]

    return x_tiles, u_tiles, active_tiles, tile_counts


def _plasma_frequency(N_per_cell, charge, mass, weight, dynamic_parameters):
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz
    # get spatial resolution of the simulation domain
    eps = dynamic_parameters.eps
    n = N_per_cell * weight / (dx * dy * dz)
    # define the number density
    sqrt_n = jnp.sqrt(n)
    # take the square root of the number density
    sqrt_eps = jnp.sqrt(eps)
    sqrt_mass = jnp.sqrt(mass)
    return sqrt_n * jnp.abs(charge) / (sqrt_eps * sqrt_mass) / (2 * jnp.pi)


def _debye_length(N_per_cell, charge, temperature, weight, dynamic_parameters):
    eps = dynamic_parameters.eps
    kb = dynamic_parameters.kb
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz
    n = N_per_cell * weight / (dx * dy * dz)
    # define the number density

    return jnp.sqrt(eps * kb * temperature / (n * charge**2))


def load_particles_from_toml(config, static_parameters, dynamic_parameters):
    if config is None:
        config = {}

    x_wind = dynamic_parameters.x_wind
    y_wind = dynamic_parameters.y_wind
    z_wind = dynamic_parameters.z_wind
    Nx = dynamic_parameters.Nx
    Ny = dynamic_parameters.Ny
    Nz = dynamic_parameters.Nz
    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    dt = dynamic_parameters.dt
    kb = dynamic_parameters.kb
    eps = dynamic_parameters.eps
    # get the simulation domain dimensions, grid sizes, spatial resolution, and thermal scalar values

    i = 0
    particle_keys = grab_particle_keys(config)
    species_arrays = []
    species_metadata = []

    weight = 1.0
    # start with a default weight of 1.0, which will be overwritten if specified in the config

    for toml_key in particle_keys:
        key1, key2, key3 = jax.random.key(i), jax.random.key(i + 1), jax.random.key(i + 2)
        i += 3

        particle_name = config[toml_key]["name"]
        print(f"\nInitializing particle species: {particle_name}")
        charge = config[toml_key]["charge"]
        mass = config[toml_key]["mass"]
        # get the particle species name, charge, and mass from the config

        if "N_particles" in config[toml_key]:
            N_particles = config[toml_key]["N_particles"]
            N_per_cell = N_particles / (Nx * Ny * Nz)
        elif "N_per_cell" in config[toml_key]:
            N_per_cell = config[toml_key]["N_per_cell"]
            N_particles = int(N_per_cell * Nx * Ny * Nz)
        # get the number of particles and number of particles per cell for the simulation

        if "temperature" in config[toml_key]:
            T = config[toml_key]["temperature"]
            vth = T_to_vth(T, mass, kb)
        elif "vth" in config[toml_key]:
            vth = config[toml_key]["vth"]
            T = vth_to_T(vth, mass, kb)
        else:
            T = 1.0
            vth = T_to_vth(T, mass, kb)
        # from the provided configuration, determine the temperature and thermal velocity of the particle species, using either the specified temperature or thermal velocity, or defaulting to a temperature of 1.0 if neither is provided

        Tx = read_value("Tx", toml_key, config, T)
        Ty = read_value("Ty", toml_key, config, T)
        Tz = read_value("Tz", toml_key, config, T)
        # overwrite the temperature in each direction if specified in the configuration, otherwise use the previously determined temperature

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
        # overwrite the initial positions and velocities of the particles if specified in the configuration, otherwise use the previously generated values

        if "temperature" not in config[toml_key]:
            T = (mass / (3 * kb * N_particles)) * (
                jnp.sum(vx**2) + jnp.sum(vy**2) + jnp.sum(vz**2)
            )

        if "weight" in config[toml_key]:
            weight = config[toml_key]["weight"]
        elif "number_density" in config[toml_key]:
            n = config[toml_key]["number_density"]
            weight = (n / N_per_cell) * (dx * dy * dz)
            # convert number density to weight based on the number of particles per cell and cell volume

        update_pos = read_value("update_pos", toml_key, config, True)
        update_v = read_value("update_v", toml_key, config, True)
        update_vx = read_value("update_vx", toml_key, config, True)
        update_vy = read_value("update_vy", toml_key, config, True)
        update_vz = read_value("update_vz", toml_key, config, True)
        update_x = read_value("update_x", toml_key, config, True)
        update_y = read_value("update_y", toml_key, config, True)
        update_z = read_value("update_z", toml_key, config, True)
        # determine whether to update the position and velocity components of the particle species based on the configuration, defaulting to True if not specified

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
        # calculate the total kinetic energy of the particle species
        pf = _plasma_frequency(N_per_cell, charge, mass, weight, dynamic_parameters)
        dl = _debye_length(N_per_cell, charge, T, weight, dynamic_parameters)
        # calculate the plasma frequency and Debye length for the particle species

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

    tile_nx, tile_ny, tile_nz = [_as_int(width) for width in static_parameters.tile_shape]
    # determine the number of tiles in each dimension based on the tile shape and simulation domain dimensions

    capacity_factor = float(static_parameters.particle_tile_capacity_factor)
    # determine the maximum number of particles per tile across all species and tiles, and apply a capacity factor to allow for some buffer

    x_tiles, u_tiles, active_tiles, _tile_counts = _pack_species_arrays_into_tiles(
        species_arrays,
        dynamic_parameters,
        (tile_nx, tile_ny, tile_nz),
        capacity_factor,
    )

    update_x = jnp.asarray([metadata["update_x"] for metadata in species_metadata], dtype=bool).reshape((-1, 3))
    update_u = jnp.asarray([metadata["update_u"] for metadata in species_metadata], dtype=bool).reshape((-1, 3))
    species_config = SpeciesConfig(
        charge=jnp.asarray([metadata["charge"] for metadata in species_metadata]),
        mass=jnp.asarray([metadata["mass"] for metadata in species_metadata]),
        weight=jnp.asarray([metadata["weight"] for metadata in species_metadata]),
        update_x=update_x,
        update_u=update_u,
    )
    particles = TiledParticles(
        x=jnp.asarray(x_tiles),
        u=jnp.asarray(u_tiles),
        active=jnp.asarray(active_tiles),
    )
    species_names = tuple(metadata["name"] for metadata in species_metadata)

    return particles, species_config, species_names, tuple(species_metadata)
