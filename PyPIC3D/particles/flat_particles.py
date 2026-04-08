import math

import jax
import jax.numpy as jnp


class _flat_particle_species_base:
    backend = "flat"

    def __init__(
        self,
        name,
        N_particles,
        charge,
        mass,
        weight,
        T,
        x1,
        x2,
        x3,
        v1,
        v2,
        v3,
        x_wind,
        y_wind,
        z_wind,
        dx,
        dy,
        dz,
        x_bc,
        y_bc,
        z_bc,
        update_pos,
        update_v,
        update_x,
        update_y,
        update_z,
        update_vx,
        update_vy,
        update_vz,
        shape,
        dt,
        species_meta,
        active_mask=None,
        unpadded_particle_count=None,
        padded_particle_count=None,
        n_devices=1,
        particles_per_shard=None,
        sharding_active=False,
    ):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
        self.weight = weight
        self.T = T
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.x_wind = x_wind
        self.y_wind = y_wind
        self.z_wind = z_wind
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x_bc = x_bc
        self.y_bc = y_bc
        self.z_bc = z_bc
        self.update_pos = update_pos
        self.update_v = update_v
        self.update_x = update_x
        self.update_y = update_y
        self.update_z = update_z
        self.update_vx = update_vx
        self.update_vy = update_vy
        self.update_vz = update_vz
        self.shape = shape
        self.dt = dt
        self.species_meta = species_meta
        self.active_mask = active_mask if active_mask is not None else jnp.ones_like(weight, dtype=bool)
        self.unpadded_particle_count = (
            N_particles if unpadded_particle_count is None else unpadded_particle_count
        )
        self.padded_particle_count = (
            N_particles if padded_particle_count is None else padded_particle_count
        )
        self.n_devices = n_devices
        self.particles_per_shard = (
            int(jnp.asarray(self.x1).shape[-1]) if particles_per_shard is None else particles_per_shard
        )
        self.sharding_active = sharding_active

    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge * self.weight

    def get_number_of_particles(self):
        return self.N_particles

    def get_temperature(self):
        return self.T

    def get_velocity(self):
        return self.v1, self.v2, self.v3

    def get_forward_position(self):
        return self.x1, self.x2, self.x3

    def get_position(self):
        x1_back = self.x1 - self.v1 * self.dt / 2
        x2_back = self.x2 - self.v2 * self.dt / 2
        x3_back = self.x3 - self.v3 * self.dt / 2

        half_x = self.x_wind / 2
        half_y = self.y_wind / 2
        half_z = self.z_wind / 2

        x1_back = jnp.where(
            x1_back > half_x,
            x1_back - self.x_wind,
            jnp.where(x1_back < -half_x, x1_back + self.x_wind, x1_back),
        )
        x2_back = jnp.where(
            x2_back > half_y,
            x2_back - self.y_wind,
            jnp.where(x2_back < -half_y, x2_back + self.y_wind, x2_back),
        )
        x3_back = jnp.where(
            x3_back > half_z,
            x3_back - self.z_wind,
            jnp.where(x3_back < -half_z, x3_back + self.z_wind, x3_back),
        )

        return x1_back, x2_back, x3_back

    def get_mass(self):
        # Padded particles use unit mass so q/m stays finite during the push.
        scaled_mass = self.mass * self.weight
        return jnp.where(self.active_mask, scaled_mass, jnp.ones_like(scaled_mass))

    def get_weight(self):
        return self.weight

    def get_shape(self):
        return self.shape

    def get_active_mask(self):
        return self.active_mask

    def momentum(self):
        vmag = jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2)
        return jnp.sum(vmag * self.mass * self.weight)

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            if self.update_vx:
                self.v1 = v1
            if self.update_vy:
                self.v2 = v2
            if self.update_vz:
                self.v3 = v3

    def update_position(self):
        if self.update_pos:
            if self.update_x:
                self.x1 = self.x1 + self.v1 * self.dt
            if self.update_y:
                self.x2 = self.x2 + self.v2 * self.dt
            if self.update_z:
                self.x3 = self.x3 + self.v3 * self.dt

    def boundary_conditions(self):
        half_x = self.x_wind / 2
        half_y = self.y_wind / 2
        half_z = self.z_wind / 2

        self.x1 = jnp.where(
            self.x1 > half_x,
            self.x1 - self.x_wind,
            jnp.where(self.x1 < -half_x, self.x1 + self.x_wind, self.x1),
        )
        self.x2 = jnp.where(
            self.x2 > half_y,
            self.x2 - self.y_wind,
            jnp.where(self.x2 < -half_y, self.x2 + self.y_wind, self.x2),
        )
        self.x3 = jnp.where(
            self.x3 > half_z,
            self.x3 - self.z_wind,
            jnp.where(self.x3 < -half_z, self.x3 + self.z_wind, self.x3),
        )


@jax.tree_util.register_pytree_node_class
class flat_particle_species(_flat_particle_species_base):
    backend = "flat"

    def tree_flatten(self):
        children = (
            self.x1,
            self.x2,
            self.x3,
            self.v1,
            self.v2,
            self.v3,
            self.charge,
            self.mass,
            self.weight,
            self.T,
            self.active_mask,
        )
        aux_data = (
            self.name,
            self.N_particles,
            self.x_wind,
            self.y_wind,
            self.z_wind,
            self.dx,
            self.dy,
            self.dz,
            self.x_bc,
            self.y_bc,
            self.z_bc,
            self.update_pos,
            self.update_v,
            self.update_x,
            self.update_y,
            self.update_z,
            self.update_vx,
            self.update_vy,
            self.update_vz,
            self.shape,
            self.dt,
            self.species_meta,
            self.unpadded_particle_count,
            self.padded_particle_count,
            self.n_devices,
            self.particles_per_shard,
            self.sharding_active,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            x1,
            x2,
            x3,
            v1,
            v2,
            v3,
            charge,
            mass,
            weight,
            T,
            active_mask,
        ) = children
        (
            name,
            N_particles,
            x_wind,
            y_wind,
            z_wind,
            dx,
            dy,
            dz,
            x_bc,
            y_bc,
            z_bc,
            update_pos,
            update_v,
            update_x,
            update_y,
            update_z,
            update_vx,
            update_vy,
            update_vz,
            shape,
            dt,
            species_meta,
            unpadded_particle_count,
            padded_particle_count,
            n_devices,
            particles_per_shard,
            sharding_active,
        ) = aux_data
        return cls(
            name=name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            weight=weight,
            T=T,
            x1=x1,
            x2=x2,
            x3=x3,
            v1=v1,
            v2=v2,
            v3=v3,
            x_wind=x_wind,
            y_wind=y_wind,
            z_wind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            x_bc=x_bc,
            y_bc=y_bc,
            z_bc=z_bc,
            update_pos=update_pos,
            update_v=update_v,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            shape=shape,
            dt=dt,
            species_meta=species_meta,
            active_mask=active_mask,
            unpadded_particle_count=unpadded_particle_count,
            padded_particle_count=padded_particle_count,
            n_devices=n_devices,
            particles_per_shard=particles_per_shard,
            sharding_active=sharding_active,
        )


@jax.tree_util.register_pytree_node_class
class flat_sharded_particle_species(_flat_particle_species_base):
    backend = "flat_sharded"

    def tree_flatten(self):
        children = (
            self.x1,
            self.x2,
            self.x3,
            self.v1,
            self.v2,
            self.v3,
            self.charge,
            self.mass,
            self.weight,
            self.T,
            self.active_mask,
        )
        aux_data = (
            self.name,
            self.N_particles,
            self.x_wind,
            self.y_wind,
            self.z_wind,
            self.dx,
            self.dy,
            self.dz,
            self.x_bc,
            self.y_bc,
            self.z_bc,
            self.update_pos,
            self.update_v,
            self.update_x,
            self.update_y,
            self.update_z,
            self.update_vx,
            self.update_vy,
            self.update_vz,
            self.shape,
            self.dt,
            self.species_meta,
            self.unpadded_particle_count,
            self.padded_particle_count,
            self.n_devices,
            self.particles_per_shard,
            self.sharding_active,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            x1,
            x2,
            x3,
            v1,
            v2,
            v3,
            charge,
            mass,
            weight,
            T,
            active_mask,
        ) = children
        (
            name,
            N_particles,
            x_wind,
            y_wind,
            z_wind,
            dx,
            dy,
            dz,
            x_bc,
            y_bc,
            z_bc,
            update_pos,
            update_v,
            update_x,
            update_y,
            update_z,
            update_vx,
            update_vy,
            update_vz,
            shape,
            dt,
            species_meta,
            unpadded_particle_count,
            padded_particle_count,
            n_devices,
            particles_per_shard,
            sharding_active,
        ) = aux_data
        return cls(
            name=name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            weight=weight,
            T=T,
            x1=x1,
            x2=x2,
            x3=x3,
            v1=v1,
            v2=v2,
            v3=v3,
            x_wind=x_wind,
            y_wind=y_wind,
            z_wind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            x_bc=x_bc,
            y_bc=y_bc,
            z_bc=z_bc,
            update_pos=update_pos,
            update_v=update_v,
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            shape=shape,
            dt=dt,
            species_meta=species_meta,
            active_mask=active_mask,
            unpadded_particle_count=unpadded_particle_count,
            padded_particle_count=padded_particle_count,
            n_devices=n_devices,
            particles_per_shard=particles_per_shard,
            sharding_active=sharding_active,
        )


def _normalize_attr(value):
    try:
        return jnp.asarray(value).item()
    except Exception:
        return value


def _same(attr_list):
    norm = [_normalize_attr(v) for v in attr_list]
    return len(set(norm)) == 1


def _species_metadata(species):
    return {
        "name": species.name,
        "N_particles": float(species.N_particles),
        "weight": float(species.weight),
        "charge": float(species.charge),
        "mass": float(species.mass),
        "temperature": float(species.T),
        "scaled mass": float(species.get_mass()),
        "scaled charge": float(species.get_charge()),
        "update_pos": species.update_pos,
        "update_v": species.update_v,
    }


def _collect_flat_particle_arrays(particles):
    species_meta = []
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []
    q_list, m_list, w_list, T_list = [], [], [], []

    for species in particles:
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        vx_list.append(vx)
        vy_list.append(vy)
        vz_list.append(vz)
        q_list.append(jnp.full_like(x, species.charge))
        m_list.append(jnp.full_like(x, species.mass))
        w_list.append(jnp.full_like(x, species.weight))
        T_list.append(jnp.full_like(x, species.T))
        species_meta.append(_species_metadata(species))

    return {
        "x": jnp.concatenate(x_list, axis=0),
        "y": jnp.concatenate(y_list, axis=0),
        "z": jnp.concatenate(z_list, axis=0),
        "vx": jnp.concatenate(vx_list, axis=0),
        "vy": jnp.concatenate(vy_list, axis=0),
        "vz": jnp.concatenate(vz_list, axis=0),
        "charge": jnp.concatenate(q_list, axis=0),
        "mass": jnp.concatenate(m_list, axis=0),
        "weight": jnp.concatenate(w_list, axis=0),
        "T": jnp.concatenate(T_list, axis=0),
        "species_meta": species_meta,
    }


def _build_flat_particle(first, arrays):
    return flat_particle_species(
        name="flat_all",
        N_particles=int(arrays["x"].shape[0]),
        charge=arrays["charge"],
        mass=arrays["mass"],
        weight=arrays["weight"],
        T=arrays["T"],
        x1=arrays["x"],
        x2=arrays["y"],
        x3=arrays["z"],
        v1=arrays["vx"],
        v2=arrays["vy"],
        v3=arrays["vz"],
        x_wind=first.x_wind,
        y_wind=first.y_wind,
        z_wind=first.z_wind,
        dx=first.dx,
        dy=first.dy,
        dz=first.dz,
        x_bc=first.x_bc,
        y_bc=first.y_bc,
        z_bc=first.z_bc,
        update_pos=first.update_pos,
        update_v=first.update_v,
        update_x=first.update_x,
        update_y=first.update_y,
        update_z=first.update_z,
        update_vx=first.update_vx,
        update_vy=first.update_vy,
        update_vz=first.update_vz,
        shape=first.shape,
        dt=first.dt,
        species_meta=arrays["species_meta"],
    )


def _pad_array(values, padded_particle_count, fill_value):
    pad_width = padded_particle_count - values.shape[0]
    if pad_width <= 0:
        return values
    fill = jnp.full((pad_width,), fill_value, dtype=values.dtype)
    return jnp.concatenate([values, fill], axis=0)


def _maybe_place_sharded(array, devices, enable_sharding):
    if not enable_sharding:
        return array
    return jax.device_put_sharded([array[i] for i in range(array.shape[0])], devices)


def unpad_sharded_array(array, count):
    return jnp.reshape(array, (-1,))[:count]


def check_flat_compat(particles):
    if not particles:
        return False
    if not _same([p.get_shape() for p in particles]):
        return False
    if (
        not _same([p.x_bc for p in particles])
        or not _same([p.y_bc for p in particles])
        or not _same([p.z_bc for p in particles])
    ):
        return False
    if particles[0].x_bc != "periodic" or particles[0].y_bc != "periodic" or particles[0].z_bc != "periodic":
        return False
    if not _same([p.update_pos for p in particles]) or not _same([p.update_v for p in particles]):
        return False
    return True


def to_flat_particles(particles):
    arrays = _collect_flat_particle_arrays(particles)
    return [_build_flat_particle(particles[0], arrays)]


def to_flat_sharded_particles(particles, n_devices=None, devices=None, place_on_devices=True):
    if not particles:
        return []

    if devices is None and n_devices is None:
        n_devices = 1
    elif devices is not None:
        n_devices = len(devices)

    if n_devices is None or n_devices < 1:
        raise ValueError("n_devices must be at least 1 for flat_sharded conversion")

    arrays = _collect_flat_particle_arrays(particles)
    first = particles[0]
    total_particles = int(arrays["x"].shape[0])
    padded_particle_count = int(math.ceil(total_particles / n_devices) * n_devices)
    particles_per_shard = padded_particle_count // n_devices

    padded_arrays = {
        "x": _pad_array(arrays["x"], padded_particle_count, 0.0),
        "y": _pad_array(arrays["y"], padded_particle_count, 0.0),
        "z": _pad_array(arrays["z"], padded_particle_count, 0.0),
        "vx": _pad_array(arrays["vx"], padded_particle_count, 0.0),
        "vy": _pad_array(arrays["vy"], padded_particle_count, 0.0),
        "vz": _pad_array(arrays["vz"], padded_particle_count, 0.0),
        "charge": _pad_array(arrays["charge"], padded_particle_count, 0.0),
        "mass": _pad_array(arrays["mass"], padded_particle_count, 1.0),
        "weight": _pad_array(arrays["weight"], padded_particle_count, 0.0),
        "T": _pad_array(arrays["T"], padded_particle_count, 0.0),
    }

    reshaped_arrays = {
        key: jnp.reshape(value, (n_devices, particles_per_shard))
        for key, value in padded_arrays.items()
    }
    active_mask = reshaped_arrays["weight"] > 0

    enable_sharding = bool(place_on_devices and devices is not None and len(devices) == n_devices)
    sharded_arrays = {
        key: _maybe_place_sharded(value, devices, enable_sharding)
        for key, value in reshaped_arrays.items()
    }
    active_mask = _maybe_place_sharded(active_mask, devices, enable_sharding)

    sharded = flat_sharded_particle_species(
        name="flat_sharded_all",
        N_particles=total_particles,
        charge=sharded_arrays["charge"],
        mass=sharded_arrays["mass"],
        weight=sharded_arrays["weight"],
        T=sharded_arrays["T"],
        x1=sharded_arrays["x"],
        x2=sharded_arrays["y"],
        x3=sharded_arrays["z"],
        v1=sharded_arrays["vx"],
        v2=sharded_arrays["vy"],
        v3=sharded_arrays["vz"],
        x_wind=first.x_wind,
        y_wind=first.y_wind,
        z_wind=first.z_wind,
        dx=first.dx,
        dy=first.dy,
        dz=first.dz,
        x_bc=first.x_bc,
        y_bc=first.y_bc,
        z_bc=first.z_bc,
        update_pos=first.update_pos,
        update_v=first.update_v,
        update_x=first.update_x,
        update_y=first.update_y,
        update_z=first.update_z,
        update_vx=first.update_vx,
        update_vy=first.update_vy,
        update_vz=first.update_vz,
        shape=first.shape,
        dt=first.dt,
        species_meta=arrays["species_meta"],
        active_mask=active_mask,
        unpadded_particle_count=total_particles,
        padded_particle_count=padded_particle_count,
        n_devices=n_devices,
        particles_per_shard=particles_per_shard,
        sharding_active=enable_sharding,
    )
    return [sharded]
