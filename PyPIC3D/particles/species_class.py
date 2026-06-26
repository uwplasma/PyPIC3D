import jax
from jax import jit
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@jit
def compute_index(x, dx, window):
    """Compute grid-cell index for a position."""
    scaled_x = x + window / 2
    return jnp.floor(scaled_x / dx).astype(int)


@jit
def apply_axis_boundary_condition(x, v, wind, half_wind, bc):
    """Apply boundary conditions to particle positions/velocities along one axis."""

    periodic = bc == 0
    reflecting = bc == 1

    periodic_x = x + wind * (x < -half_wind) - wind * (x > half_wind)
    reflected_x = jnp.where(
        x > half_wind,
        2.0 * half_wind - x,
        jnp.where(x < -half_wind, -2.0 * half_wind - x, x),
    )
    reflected_v = jnp.where((x >= half_wind) | (x <= -half_wind), -v, v)

    x_out = jnp.where(periodic, periodic_x, jnp.where(reflecting, reflected_x, x))
    v_out = jnp.where(reflecting, reflected_v, v)

    return x_out, v_out


@register_pytree_node_class
class particle_species:
    """A particle species (positions/velocities + metadata) stored as a JAX pytree."""

    def __init__(
        self,
        name,
        N_particles,
        charge,
        mass,
        T,
        v1,
        v2,
        v3,
        x1,
        x2,
        x3,
        xwind,
        ywind,
        zwind,
        dx,
        dy,
        dz,
        weight=1,
        x_bc="periodic",
        y_bc="periodic",
        z_bc="periodic",
        update_x=True,
        update_y=True,
        update_z=True,
        update_vx=True,
        update_vy=True,
        update_vz=True,
        update_pos=True,
        update_v=True,
        dt=0,
        active_mask=None,
    ):
        self.name = name
        self.N_particles = N_particles
        self.charge = charge
        self.mass = mass
        self.weight = weight
        self.T = T
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x_wind = xwind
        self.y_wind = ywind
        self.z_wind = zwind
        self.half_x_wind = 0.5 * xwind
        self.half_y_wind = 0.5 * ywind
        self.half_z_wind = 0.5 * zwind
        self.x_bc = x_bc
        self.y_bc = y_bc
        self.z_bc = z_bc
        self.x_periodic = x_bc == "periodic"
        self.x_reflecting = x_bc == "reflecting"
        self.y_periodic = y_bc == "periodic"
        self.y_reflecting = y_bc == "reflecting"
        self.z_periodic = z_bc == "periodic"
        self.z_reflecting = z_bc == "reflecting"
        self.x_absorbing = x_bc == "absorbing"
        self.y_absorbing = y_bc == "absorbing"
        self.z_absorbing = z_bc == "absorbing"
        self.update_x = update_x
        self.update_y = update_y
        self.update_z = update_z
        self.update_vx = update_vx
        self.update_vy = update_vy
        self.update_vz = update_vz
        self.update_pos = update_pos
        self.update_v = update_v
        self.dt = dt

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        if active_mask is None:
            active_mask = jnp.ones_like(x1, dtype=bool)
        self.active_mask = active_mask

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

    def get_active_velocity(self):
        active = self.active_mask.astype(self.v1.dtype)
        return active * self.v1, active * self.v2, active * self.v3

    def get_forward_position(self):
        return self.x1, self.x2, self.x3

    def get_active_mask(self):
        return self.active_mask

    def get_position(self):
        x1_back = self.x1 - self.v1 * self.dt / 2
        x2_back = self.x2 - self.v2 * self.dt / 2
        x3_back = self.x3 - self.v3 * self.dt / 2

        if self.x_bc == "periodic":
            x1_back = jnp.where(
                x1_back > self.x_wind / 2,
                x1_back - self.x_wind,
                jnp.where(x1_back < -self.x_wind / 2, x1_back + self.x_wind, x1_back),
            )

        if self.y_bc == "periodic":
            x2_back = jnp.where(
                x2_back > self.y_wind / 2,
                x2_back - self.y_wind,
                jnp.where(x2_back < -self.y_wind / 2, x2_back + self.y_wind, x2_back),
            )

        if self.z_bc == "periodic":
            x3_back = jnp.where(
                x3_back > self.z_wind / 2,
                x3_back - self.z_wind,
                jnp.where(x3_back < -self.z_wind / 2, x3_back + self.z_wind, x3_back),
            )

        return x1_back, x2_back, x3_back

    def get_mass(self):
        return self.mass * self.weight

    def get_weight(self):
        return self.weight

    def get_resolution(self):
        return self.dx, self.dy, self.dz

    def get_index(self):
        return (
            compute_index(self.x1, self.dx, self.x_wind),
            compute_index(self.x2, self.dy, self.y_wind),
            compute_index(self.x3, self.dz, self.z_wind),
        )

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            active = self.active_mask
            if self.update_vx:
                self.v1 = jnp.where(active, v1, self.v1)
            if self.update_vy:
                self.v2 = jnp.where(active, v2, self.v2)
            if self.update_vz:
                self.v3 = jnp.where(active, v3, self.v3)

    def set_position(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def set_mass(self, mass):
        self.mass = mass

    def set_weight(self, weight):
        self.weight = weight

    def kinetic_energy(self):
        v2 = jnp.square(self.v1) + jnp.square(self.v2) + jnp.square(self.v3)
        return 0.5 * self.weight * self.mass * jnp.sum(self.active_mask.astype(v2.dtype) * v2)

    def momentum(self):
        vmag = jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2)
        return self.mass * self.weight * jnp.sum(self.active_mask.astype(vmag.dtype) * vmag)

    def boundary_conditions(self, world=None):
        x1, x2, x3 = self.x1, self.x2, self.x3
        v1, v2, v3 = self.v1, self.v2, self.v3
        if world is None or "particle_boundary_conditions" not in world:
            particle_bc = {"x": 0, "y": 0, "z": 0}
        else:
            particle_bc = world["particle_boundary_conditions"]

        x1, v1 = apply_axis_boundary_condition(
            x1, v1, self.x_wind, self.half_x_wind, particle_bc["x"]
        )
        x2, v2 = apply_axis_boundary_condition(
            x2, v2, self.y_wind, self.half_y_wind, particle_bc["y"]
        )
        x3, v3 = apply_axis_boundary_condition(
            x3, v3, self.z_wind, self.half_z_wind, particle_bc["z"]
        )

        active_mask = self.active_mask
        x_inside = (x1 <= self.half_x_wind) & (x1 >= -self.half_x_wind)
        y_inside = (x2 <= self.half_y_wind) & (x2 >= -self.half_y_wind)
        z_inside = (x3 <= self.half_z_wind) & (x3 >= -self.half_z_wind)
        active_mask = jnp.where(particle_bc["x"] == 2, active_mask & x_inside, active_mask)
        active_mask = jnp.where(particle_bc["y"] == 2, active_mask & y_inside, active_mask)
        active_mask = jnp.where(particle_bc["z"] == 2, active_mask & z_inside, active_mask)

        self.x1, self.x2, self.x3 = x1, x2, x3
        self.v1, self.v2, self.v3 = v1, v2, v3
        self.active_mask = active_mask

    def update_position(self):
        if self.update_pos:
            active = self.active_mask.astype(self.x1.dtype)
            if self.update_x:
                self.x1 = self.x1 + active * self.v1 * self.dt
            if self.update_y:
                self.x2 = self.x2 + active * self.v2 * self.dt
            if self.update_z:
                self.x3 = self.x3 + active * self.v3 * self.dt

    def tree_flatten(self):
        children = (self.v1, self.v2, self.v3, self.x1, self.x2, self.x3, self.active_mask)

        aux_data = (
            self.name,
            self.N_particles,
            self.charge,
            self.mass,
            self.T,
            self.x_wind,
            self.y_wind,
            self.z_wind,
            self.dx,
            self.dy,
            self.dz,
            self.weight,
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
            self.dt,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        v1, v2, v3, x1, x2, x3, active_mask = children

        (
            name,
            N_particles,
            charge,
            mass,
            T,
            x_wind,
            y_wind,
            z_wind,
            dx,
            dy,
            dz,
            weight,
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
            dt,
        ) = aux_data

        obj = cls(
            name=name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            T=T,
            x1=x1,
            x2=x2,
            x3=x3,
            v1=v1,
            v2=v2,
            v3=v3,
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
            update_x=update_x,
            update_y=update_y,
            update_z=update_z,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
            update_pos=update_pos,
            update_v=update_v,
            dt=dt,
            active_mask=active_mask,
        )

        return obj
