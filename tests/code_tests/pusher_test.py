import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import ghost_cells
from tests.kernel_fixtures import build_tiled_particles, particle_parameters_from_tile_values, particle_species
from tests.kernel_fixtures import kernel_parameters_from_values
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, parameter_set, tile_shape, num_guard_cells=2):
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(num_guard_cells)
    Nx = int(field.shape[0]) - 2
    Ny = int(field.shape[1]) - 2
    Nz = int(field.shape[2]) - 2
    ntx = _tile_axis_count(Nx, tile_nx)
    nty = _tile_axis_count(Ny, tile_ny)
    ntz = _tile_axis_count(Nz, tile_nz)

    if g != 1:
        field_tiles = jnp.zeros(
            (
                ntx,
                nty,
                ntz,
                tile_nx + 2 * g,
                tile_ny + 2 * g,
                tile_nz + 2 * g,
            ),
            dtype=field.dtype,
        )
        for tx in range(ntx):
            for ty in range(nty):
                for tz in range(ntz):
                    ix = 1 + tx * tile_nx
                    iy = 1 + ty * tile_ny
                    iz = 1 + tz * tile_nz
                    interior = field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                    field_tiles = field_tiles.at[tx, ty, tz, g:-g, g:-g, g:-g].set(interior)
        parameter_set = dict(parameter_set)
        parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
        static_parameters, _ = kernel_parameters_from_values(parameter_set)
        return ghost_cells.update_tiled_ghost_cells(field_tiles, static_parameters, g)

    def tile_at(tx, ty, tz):
        start = (tx * tile_nx, ty * tile_ny, tz * tile_nz)
        size = (tile_nx + 2, tile_ny + 2, tile_nz + 2)
        return jax.lax.dynamic_slice(field, start, size)

    return jnp.stack(
        [
            jnp.stack(
                [
                    jnp.stack([tile_at(tx, ty, tz) for tz in range(ntz)], axis=0)
                    for ty in range(nty)
                ],
                axis=0,
            )
            for tx in range(ntx)
        ],
        axis=0,
    )


def tile_vector_field(field, parameter_set, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, parameter_set, tile_shape, num_guard_cells) for component in field)


class TestTiledParticlePusher(unittest.TestCase):
    def _build_parameter_values(self, Nx=8, Ny=6, Nz=4, shape_factor=1):
        parameter_set = {
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "dx": 4.0 / Nx,
            "dy": 3.0 / Ny,
            "dz": 2.0 / Nz,
            "dt": 0.05,
            "x_wind": 4.0,
            "y_wind": 3.0,
            "z_wind": 2.0,
            "shape_factor": shape_factor,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return parameter_set

    def _with_tiled_grids(self, parameter_set, tile_shape, g=1):
        parameter_set["tile_shape"] = tuple(int(width) for width in tile_shape)
        parameter_set["guard_cells"] = int(g)
        parameter_set["field_mesh"] = ghost_cells.make_field_mesh((
            int(parameter_set["Nx"]) // int(tile_shape[0]),
            int(parameter_set["Ny"]) // int(tile_shape[1]),
            int(parameter_set["Nz"]) // int(tile_shape[2]),
        ))
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set)
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
        parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
        parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
        return parameter_set

    def _copy_parameters_for_tile_shape(self, parameter_set, tile_shape, g):
        tiled_parameters = dict(parameter_set)
        tiled_parameters["grids"] = dict(parameter_set["grids"])
        return self._with_tiled_grids(tiled_parameters, tile_shape, g=g)

    def _simulation_parameters_for_tile_shape(self, tile_shape):
        return {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

    def _push_tiled_species(self, species, parameter_set, tile_shape, E, B, dynamic_values, relativistic=True, particle_pusher="boris"):
        particle_static, particle_dynamic = particle_parameters_from_tile_values(
            parameter_set,
            self._simulation_parameters_for_tile_shape(tile_shape),
            dynamic_values=dynamic_values,
        )
        tiled_particles, species_config = build_tiled_particles(
            [species],
            particle_static,
            particle_dynamic,
        )
        g = int(parameter_set["guard_cells"])
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(
            relativistic=bool(relativistic),
            particle_pusher=particle_pusher,
        )
        return particle_push(
            tiled_particles,
            species_config,
            tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=g),
            tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=g),
            static_parameters,
            dynamic_parameters,
        )

    def _deterministic_vector_field(self, parameter_set, scale):
        Nx, Ny, Nz = parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"]
        ii, jj, kk = jnp.meshgrid(
            jnp.arange(Nx, dtype=float),
            jnp.arange(Ny, dtype=float),
            jnp.arange(Nz, dtype=float),
            indexing="ij",
        )

        shape = (Nx + 2, Ny + 2, Nz + 2)
        Fx = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.2 + 0.03 * ii - 0.02 * jj + 0.04 * kk))
        Fy = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (-0.1 + 0.05 * ii + 0.01 * jj - 0.03 * kk))
        Fz = jnp.zeros(shape).at[1:-1, 1:-1, 1:-1].set(scale * (0.3 - 0.04 * ii + 0.02 * jj + 0.01 * kk))
        return Fx, Fy, Fz

    def _species(self, parameter_set, active_mask=None, update_vx=True, update_vy=True, update_vz=True):
        if active_mask is None:
            active_mask = jnp.array([True, True, True, True])

        return particle_species(
            name="test particles",
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            x1=jnp.array([-1.25, -0.25, 0.65, 1.45]),
            x2=jnp.array([-1.0, -0.25, 0.35, 1.05]),
            x3=jnp.array([-0.65, -0.15, 0.25, 0.75]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
            active_mask=active_mask,
            update_vx=update_vx,
            update_vy=update_vy,
            update_vz=update_vz,
        )

    def _flatten_active_by_position(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def test_particle_push_matches_one_tile_boris(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 3, 2)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)

        species = self._species(parameter_set)
        reference = self._species(parameter_set)
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=False,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=False,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)
        tiled_u = jax.device_get(tiled_u)
        reference_u = jax.device_get(reference_u)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))

    def test_particle_push_matches_one_tile_higuera_cary(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 3, 2)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=0.25)
        B = self._deterministic_vector_field(parameter_set, scale=0.05)

        def make_species():
            return particle_species(
                name="higuera cary particles",
                charge=-1.0,
                mass=2.0,
                weight=0.5,
                x1=jnp.array([-1.95, -1.01, 0.99, 1.95]),
                x2=jnp.array([-1.35, -0.01, 0.49, 1.35]),
                x3=jnp.array([-0.95, -0.01, 0.49, 0.95]),
                v1=jnp.array([0.02, -0.01, 0.03, -0.015]),
                v2=jnp.array([0.01, 0.02, -0.015, 0.005]),
                v3=jnp.array([-0.005, 0.015, 0.01, -0.02]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            particle_pusher="higuera_cary",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            particle_pusher="higuera_cary",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)
        tiled_u = jax.device_get(tiled_u)
        reference_u = jax.device_get(reference_u)

        self.assertTrue(jnp.allclose(tiled_u, reference_u, rtol=1.0e-12, atol=1.0e-12))

    def test_particle_push_respects_active_and_update_flags(self):
        parameter_set = self._build_parameter_values()
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 3, 2)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        E = self._deterministic_vector_field(parameter_set, scale=1.0)
        B = self._deterministic_vector_field(parameter_set, scale=0.2)
        species = self._species(
            parameter_set,
            active_mask=jnp.array([True, False, True, True]),
            update_vx=False,
            update_vy=True,
            update_vz=False,
        )
        particle_static, particle_dynamic = particle_parameters_from_tile_values(
            parameter_set,
            simulation_parameters,
            dynamic_values=dynamic_values,
        )
        tiled_particles, species_config = build_tiled_particles([species], particle_static, particle_dynamic)

        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(relativistic=False, particle_pusher="boris")
        pushed = particle_push(
            tiled_particles,
            species_config,
            tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=1),
            tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=1),
            static_parameters,
            dynamic_parameters,
        )

        self.assertTrue(jnp.allclose(pushed.u[..., 0], tiled_particles.u[..., 0]))
        self.assertTrue(jnp.allclose(pushed.u[..., 2], tiled_particles.u[..., 2]))
        self.assertTrue(jnp.allclose(pushed.u[..., 1][~tiled_particles.active], tiled_particles.u[..., 1][~tiled_particles.active]))

    def test_particle_push_matches_relativistic_one_tile_boris_on_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1)
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 1, 1)
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape)
        E = self._deterministic_vector_field(parameter_set, scale=0.1)
        B = self._deterministic_vector_field(parameter_set, scale=0.02)
        def make_species():
            return particle_species(
                name="one dimensional",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-1.25, 0.15, 1.25]),
                x2=jnp.zeros(3),
                x3=jnp.zeros(3),
                v1=jnp.array([0.02, -0.01, 0.03]),
                v2=jnp.array([0.0, 0.01, -0.02]),
                v3=jnp.array([-0.005, 0.025, 0.01]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, int(parameter_set["guard_cells"]))
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))

    def test_particle_push_matches_one_tile_boris_on_two_guard_reduced_axes(self):
        parameter_set = self._build_parameter_values(Nx=8, Ny=1, Nz=1, shape_factor=2)
        dynamic_values = {"C": 10.0}
        tile_shape = (2, 1, 1)
        g = 2
        parameter_set = self._with_tiled_grids(parameter_set, tile_shape, g=g)
        E = self._deterministic_vector_field(parameter_set, scale=0.1)
        B = self._deterministic_vector_field(parameter_set, scale=0.02)

        def make_species():
            return particle_species(
                name="two guard one dimensional",
                charge=1.0,
                mass=1.0,
                weight=1.0,
                x1=jnp.array([-1.25, 0.15, 1.25]),
                x2=jnp.zeros(3),
                x3=jnp.zeros(3),
                v1=jnp.array([0.02, -0.01, 0.03]),
                v2=jnp.array([0.0, 0.01, -0.02]),
                v3=jnp.array([-0.005, 0.025, 0.01]),
            )

        species = make_species()
        reference = make_species()
        reference_tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, reference_tile_shape, g)
        reference = self._push_tiled_species(
            reference,
            reference_parameters,
            reference_tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        pushed = self._push_tiled_species(
            species,
            parameter_set,
            tile_shape,
            E,
            B,
            dynamic_values,
            relativistic=True,
            particle_pusher="boris",
        )

        _, tiled_u = self._flatten_active_by_position(pushed)
        _, reference_u = self._flatten_active_by_position(reference)

        self.assertTrue(np.allclose(np.asarray(tiled_u), np.asarray(reference_u), rtol=1.0e-12, atol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
