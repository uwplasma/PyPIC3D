import numpy as np
import jax
import jax.numpy as jnp

from PyPIC3D.utils import vth_to_T, plasma_frequency, debye_length, T_to_vth
from PyPIC3D.particles.species_class import particle_species


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
    particles = []
    particle_keys = grab_particle_keys(config)

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
            assert config[toml_key]["x_bc"] in ["periodic", "reflecting"], (
                f"Invalid x boundary condition: {config[toml_key]['x_bc']}"
            )
            x_bc = config[toml_key]["x_bc"]

        y_bc = "periodic"
        if "y_bc" in config[toml_key]:
            assert config[toml_key]["y_bc"] in ["periodic", "reflecting"], (
                f"Invalid y boundary condition: {config[toml_key]['y_bc']}"
            )
            y_bc = config[toml_key]["y_bc"]

        z_bc = "periodic"
        if "z_bc" in config[toml_key]:
            assert config[toml_key]["z_bc"] in ["periodic", "reflecting"], (
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

        particle = particle_species(
            name=particle_name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            T=T,
            x1=x,
            x2=y,
            x3=z,
            v1=vx,
            v2=vy,
            v3=vz,
            xwind=x_wind,
            ywind=y_wind,
            zwind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            weight=weight,
            x_bc=x_bc,
            y_bc=y_bc,
            z_bc=z_bc,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_pos=update_pos,
            update_v=update_v,
            shape=simulation_parameters["shape_factor"],
            dt=dt,
        )
        particles.append(particle)

        pf = plasma_frequency(particle, world, constants)
        dl = debye_length(particle, world, constants)
        print(f"Number of particles: {N_particles}")
        print(f"Number of particles per cell: {N_per_cell}")
        print(f"x, y, z boundary conditions: {x_bc}, {y_bc}, {z_bc}")
        print(f"Charge: {charge}")
        print(f"Mass: {mass}")
        print(f"Temperature: {T}")
        print(f"Thermal Velocity: {vth}")
        print(f"Particle Kinetic Energy: {particle.kinetic_energy()}")
        print(f"Particle Species Plasma Frequency: {pf}")
        print(f"Time Steps Per Plasma Period: {(1 / (dt * pf))}")
        print(f"Particle Species Debye Length: {dl}")
        print(f"Particle Weight: {weight}")
        print(f"Particle Species Scaled Charge: {particle.get_charge()}")
        print(f"Particle Species Scaled Mass: {particle.get_mass()}")

    return particles
