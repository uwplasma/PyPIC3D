import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@jit
def compute_index(x, dx, window):
    """Compute grid-cell index for a position."""
    scaled_x = x + window / 2
    return jnp.floor(scaled_x / dx).astype(int)


@partial(jit, static_argnames=("periodic", "reflecting"))
def apply_axis_boundary_condition(x, v, wind, half_wind, periodic, reflecting):
    """Apply boundary conditions to particle positions/velocities along one axis."""

    def periodic_bc(state):
        x_in, v_in = state
        x_out = x_in + wind * (x_in < -half_wind) - wind * (x_in > half_wind)
        return x_out, v_in

    def reflecting_bc(state):
        x_in, v_in = state
        v_out = jnp.where((x_in >= half_wind) | (x_in <= -half_wind), -v_in, v_in)
        return x_in, v_out

    def identity_bc(state):
        return state

    return jax.lax.cond(
        periodic,
        periodic_bc,
        lambda state: jax.lax.cond(reflecting, reflecting_bc, identity_bc, state),
        (x, v),
    )


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
        shape=1,
        dt=0,
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
        self.update_x = update_x
        self.update_y = update_y
        self.update_z = update_z
        self.update_vx = update_vx
        self.update_vy = update_vy
        self.update_vz = update_vz
        self.update_pos = update_pos
        self.update_v = update_v
        self.shape = shape
        self.dt = dt

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

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

    def get_shape(self):
        return self.shape

    def get_index(self):
        return (
            compute_index(self.x1, self.dx, self.x_wind),
            compute_index(self.x2, self.dy, self.y_wind),
            compute_index(self.x3, self.dz, self.z_wind),
        )

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            if self.update_vx:
                self.v1 = v1
            if self.update_vy:
                self.v2 = v2
            if self.update_vz:
                self.v3 = v3

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
        return 0.5 * self.weight * self.mass * jnp.sum(v2)

    def momentum(self):
        return self.mass * self.weight * jnp.sum(jnp.sqrt(self.v1**2 + self.v2**2 + self.v3**2))

    def boundary_conditions(self):
        x1, x2, x3 = self.x1, self.x2, self.x3
        v1, v2, v3 = self.v1, self.v2, self.v3

        x1, v1 = apply_axis_boundary_condition(
            x1, v1, self.x_wind, self.half_x_wind, self.x_periodic, self.x_reflecting
        )
        x2, v2 = apply_axis_boundary_condition(
            x2, v2, self.y_wind, self.half_y_wind, self.y_periodic, self.y_reflecting
        )
        x3, v3 = apply_axis_boundary_condition(
            x3, v3, self.z_wind, self.half_z_wind, self.z_periodic, self.z_reflecting
        )

        self.x1, self.x2, self.x3 = x1, x2, x3
        self.v1, self.v2, self.v3 = v1, v2, v3

    def update_position(self):
        if self.update_pos:
            if self.update_x:
                self.x1 = self.x1 + self.v1 * self.dt
            if self.update_y:
                self.x2 = self.x2 + self.v2 * self.dt
            if self.update_z:
                self.x3 = self.x3 + self.v3 * self.dt

    def tree_flatten(self):
        children = (self.v1, self.v2, self.v3, self.x1, self.x2, self.x3)

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
            self.shape,
            self.dt,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        v1, v2, v3, x1, x2, x3 = children

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
            shape,
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
            shape=shape,
            dt=dt,
        )

        return obj
