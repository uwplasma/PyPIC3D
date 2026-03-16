import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class flat_particle_species:
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
        return self.mass * self.weight

    def get_weight(self):
        return self.weight

    def get_shape(self):
        return self.shape

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

    def tree_flatten(self):
        children = (self.x1, self.x2, self.x3, self.v1, self.v2, self.v3)
        aux_data = (
            self.name,
            self.N_particles,
            self.charge,
            self.mass,
            self.weight,
            self.T,
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
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x1, x2, x3, v1, v2, v3 = children
        (
            name,
            N_particles,
            charge,
            mass,
            weight,
            T,
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
        )


def _normalize_attr(value):
    try:
        return jnp.asarray(value).item()
    except Exception:
        return value


def _same(attr_list):
    norm = [_normalize_attr(v) for v in attr_list]
    return len(set(norm)) == 1


def check_flat_compat(particles):
    if not particles:
        return False
    if not _same([p.get_shape() for p in particles]):
        return False
    if not _same([p.x_bc for p in particles]) or not _same([p.y_bc for p in particles]) or not _same([p.z_bc for p in particles]):
        return False
    if particles[0].x_bc != "periodic" or particles[0].y_bc != "periodic" or particles[0].z_bc != "periodic":
        return False
    if not _same([p.update_pos for p in particles]) or not _same([p.update_v for p in particles]):
        return False
    return True


def to_flat_particles(particles):
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
        species_meta.append(
            {
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
        )

    x = jnp.concatenate(x_list, axis=0)
    y = jnp.concatenate(y_list, axis=0)
    z = jnp.concatenate(z_list, axis=0)
    vx = jnp.concatenate(vx_list, axis=0)
    vy = jnp.concatenate(vy_list, axis=0)
    vz = jnp.concatenate(vz_list, axis=0)
    charge = jnp.concatenate(q_list, axis=0)
    mass = jnp.concatenate(m_list, axis=0)
    weight = jnp.concatenate(w_list, axis=0)
    T = jnp.concatenate(T_list, axis=0)

    first = particles[0]
    flat = flat_particle_species(
        name="flat_all",
        N_particles=int(x.shape[0]),
        charge=charge,
        mass=mass,
        weight=weight,
        T=T,
        x1=x,
        x2=y,
        x3=z,
        v1=vx,
        v2=vy,
        v3=vz,
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
        species_meta=species_meta,
    )
    return [flat]
