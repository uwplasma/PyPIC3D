import jax
import jax.numpy as jnp
from functools import partial

from PyPIC3D.utils import wrap_around


def _init_index_and_frac(x, xmin, dx, shape_factor):
    s = (x - xmin) / dx
    if shape_factor == 1:
        i0 = jnp.floor(s).astype(jnp.int32)
    else:
        i0 = jnp.round(s).astype(jnp.int32)
    r = s - i0
    return i0, r


def _advance_index_and_frac(i0, r, dr, n, shape_factor):
    r_new = r + dr
    if shape_factor == 1:
        shift = jnp.floor(r_new).astype(jnp.int32)
    else:
        shift = jnp.floor(r_new + 0.5).astype(jnp.int32)
    r_new = r_new - shift
    i0_new = wrap_around(i0 + shift, n)
    return i0_new, r_new


@jax.tree_util.register_pytree_node_class
class indexed_particle_species:
    def __init__(
        self,
        name,
        N_particles,
        charge,
        mass,
        weight,
        T,
        v1,
        v2,
        v3,
        i1,
        i2,
        i3,
        r1,
        r2,
        r3,
        x_wind,
        y_wind,
        z_wind,
        dx,
        dy,
        dz,
        xmin,
        ymin,
        zmin,
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
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.x_wind = x_wind
        self.y_wind = y_wind
        self.z_wind = z_wind
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
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

    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge * self.weight

    def get_number_of_particles(self):
        return self.N_particles

    def get_velocity(self):
        return self.v1, self.v2, self.v3

    def get_mass(self):
        return self.mass * self.weight

    def get_shape(self):
        return self.shape

    def get_forward_position(self):
        x = (self.i1 + self.r1) * self.dx + self.xmin
        y = (self.i2 + self.r2) * self.dy + self.ymin
        z = (self.i3 + self.r3) * self.dz + self.zmin
        return x, y, z

    def get_indexed_position(self):
        return self.i1, self.i2, self.i3, self.r1, self.r2, self.r3

    def set_velocity(self, v1, v2, v3):
        if self.update_v:
            if self.update_vx:
                self.v1 = v1
            if self.update_vy:
                self.v2 = v2
            if self.update_vz:
                self.v3 = v3

    def update_position(self, world):
        if not self.update_pos:
            return

        dt = world["dt"]
        if self.update_x:
            self.i1, self.r1 = _advance_index_and_frac(
                self.i1,
                self.r1,
                self.v1 * dt / self.dx,
                world["Nx"],
                self.shape,
            )
        if self.update_y:
            self.i2, self.r2 = _advance_index_and_frac(
                self.i2,
                self.r2,
                self.v2 * dt / self.dy,
                world["Ny"],
                self.shape,
            )
        if self.update_z:
            self.i3, self.r3 = _advance_index_and_frac(
                self.i3,
                self.r3,
                self.v3 * dt / self.dz,
                world["Nz"],
                self.shape,
            )

    def tree_flatten(self):
        children = (
            self.v1,
            self.v2,
            self.v3,
            self.i1,
            self.i2,
            self.i3,
            self.r1,
            self.r2,
            self.r3,
        )
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
            self.xmin,
            self.ymin,
            self.zmin,
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
        v1, v2, v3, i1, i2, i3, r1, r2, r3 = children
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
            xmin,
            ymin,
            zmin,
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
        return cls(
            name=name,
            N_particles=N_particles,
            charge=charge,
            mass=mass,
            weight=weight,
            T=T,
            v1=v1,
            v2=v2,
            v3=v3,
            i1=i1,
            i2=i2,
            i3=i3,
            r1=r1,
            r2=r2,
            r3=r3,
            x_wind=x_wind,
            y_wind=y_wind,
            z_wind=z_wind,
            dx=dx,
            dy=dy,
            dz=dz,
            xmin=xmin,
            ymin=ymin,
            zmin=zmin,
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
        )


def to_indexed_particles(particles, world):
    indexed = []
    xmin, ymin, zmin = world["grids"]["center"][0][0], world["grids"]["center"][1][0], world["grids"]["center"][2][0]
    for species in particles:
        x, y, z = species.get_forward_position()
        shape = species.get_shape()
        i1, r1 = _init_index_and_frac(x, xmin, world["dx"], shape)
        i2, r2 = _init_index_and_frac(y, ymin, world["dy"], shape)
        i3, r3 = _init_index_and_frac(z, zmin, world["dz"], shape)
        i1 = wrap_around(i1, world["Nx"])
        i2 = wrap_around(i2, world["Ny"])
        i3 = wrap_around(i3, world["Nz"])
        v1, v2, v3 = species.get_velocity()

        indexed.append(
            indexed_particle_species(
                name=species.name,
                N_particles=species.N_particles,
                charge=species.charge,
                mass=species.mass,
                weight=species.weight,
                T=species.T,
                v1=v1,
                v2=v2,
                v3=v3,
                i1=i1,
                i2=i2,
                i3=i3,
                r1=r1,
                r2=r2,
                r3=r3,
                x_wind=species.x_wind,
                y_wind=species.y_wind,
                z_wind=species.z_wind,
                dx=species.dx,
                dy=species.dy,
                dz=species.dz,
                xmin=xmin,
                ymin=ymin,
                zmin=zmin,
                x_bc=species.x_bc,
                y_bc=species.y_bc,
                z_bc=species.z_bc,
                update_pos=species.update_pos,
                update_v=species.update_v,
                update_x=species.update_x,
                update_y=species.update_y,
                update_z=species.update_z,
                update_vx=species.update_vx,
                update_vy=species.update_vy,
                update_vz=species.update_vz,
                shape=shape,
                dt=species.dt,
            )
        )
    return indexed


def check_periodic_bc(particles):
    for species in particles:
        if species.x_bc != "periodic" or species.y_bc != "periodic" or species.z_bc != "periodic":
            return False
    return True
