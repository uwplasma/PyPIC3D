import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.evolve import time_loop_electrostatic
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.tests.tiled_particle_fixtures import particle_species
from PyPIC3D.tests.tiled_particle_fixtures import to_tiled_particles
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_scalar_field
from PyPIC3D.utilities.grids import build_collocated_grid, build_tiled_yee_grids, build_yee_grid


jax.config.update("jax_enable_x64", True)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def tile_scalar_field(field, world, tile_shape, num_guard_cells=2):
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
        return ghost_cells.update_tiled_ghost_cells(field_tiles, world, g, tile_shape)

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


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


class TestTiledElectrostatic(unittest.TestCase):
    def _build_world(self, shape_factor=1):
        world = {
            "Nx": 8,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.5,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.01,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": shape_factor,
            "guard_cells": 2,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        world["tile_shape"] = (2, 1, 1)
        world = self._world_with_tiled_grids(world, world["tile_shape"])
        return world

    def _world_with_tiled_grids(self, world, tile_shape):
        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, g)
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
        world["grids"] = grids
        return world

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

    def _simulation_parameters_for_tile_shape(self, tile_shape):
        return {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

    def _one_tile_shape(self, world):
        return (int(world["Nx"]), int(world["Ny"]), int(world["Nz"]))

    def _empty_fields(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        zero = jnp.zeros(shape)
        E = (zero, zero, zero)
        B = (zero, zero, zero)
        J = (zero, zero, zero)
        external_fields = (E, B)
        return E, B, J, zero, zero, external_fields

    def _species(self, world):
        return particle_species(
            name="electrons",
            N_particles=4,
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            x2=jnp.zeros(4),
            x3=jnp.zeros(4),
            v1=jnp.array([0.10, -0.05, 0.07, -0.02]),
            v2=jnp.zeros(4),
            v3=jnp.zeros(4),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def _neutral_species(self, world):
        electrons = self._species(world)
        ions = particle_species(
            name="ions",
            N_particles=4,
            charge=1.0,
            mass=4.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.25, -0.25, 0.75, 1.75]),
            x2=jnp.zeros(4),
            x3=jnp.zeros(4),
            v1=jnp.array([-0.03, 0.04, -0.02, 0.01]),
            v2=jnp.zeros(4),
            v3=jnp.zeros(4),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        return [electrons, ions]

    def _pad_tiled_particle_capacity(self, tiled_particles, min_slots):
        current_slots = tiled_particles.active.shape[-1]
        if current_slots >= min_slots:
            return tiled_particles

        pad_width = [(0, 0)] * tiled_particles.x.ndim
        pad_width[-2] = (0, min_slots - current_slots)
        x = jnp.pad(tiled_particles.x, pad_width)
        u = jnp.pad(tiled_particles.u, pad_width)

        scalar_pad_width = [(0, 0)] * tiled_particles.active.ndim
        scalar_pad_width[-1] = (0, min_slots - current_slots)

        return TiledParticles(
            x=x,
            u=u,
            active=jnp.pad(tiled_particles.active, scalar_pad_width),
        )

    def _build_tiled_state(self, particles, fields, world, tile_shape, min_slots=None):
        tiled_particles, species_config = to_tiled_particles(
            particles,
            world,
            self._simulation_parameters_for_tile_shape(tile_shape),
        )
        if min_slots is not None:
            tiled_particles = self._pad_tiled_particle_capacity(tiled_particles, min_slots)

        E, B, J, rho, phi, external_fields = fields
        g = int(world["guard_cells"])
        tiled_fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=g),
            tile_vector_field(B, world, tile_shape, num_guard_cells=g),
            tile_vector_field(J, world, tile_shape, num_guard_cells=g),
            tile_scalar_field(rho, world, tile_shape, num_guard_cells=g),
            tile_scalar_field(phi, world, tile_shape, num_guard_cells=g),
            (
                tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=g),
                tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=g),
            ),
            None,
            jnp.asarray(False),
        )

        return tiled_particles, species_config, tiled_fields

    def _long_two_stream_species(self, world):
        x0 = jnp.linspace(
            -0.5 * world["x_wind"],
            0.5 * world["x_wind"],
            world["Nx"],
            endpoint=False,
        ) + 0.5 * world["dx"]
        x0 = x0 + 0.02 * world["dx"] * jnp.sin(2.0 * jnp.pi * (x0 / world["x_wind"] + 0.5))
        zeros = jnp.zeros_like(x0)
        beam_speed = 0.08

        species_kwargs = {
            "N_particles": world["Nx"],
            "mass": 1.0,
            "weight": 1.0,
            "T": 0.0,
            "x1": x0,
            "x2": zeros,
            "x3": zeros,
            "v2": zeros,
            "v3": zeros,
            "xwind": world["x_wind"],
            "ywind": world["y_wind"],
            "zwind": world["z_wind"],
            "dx": world["dx"],
            "dy": world["dy"],
            "dz": world["dz"],
            "dt": world["dt"],
        }
        return [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]

    def _long_two_stream_state(self):
        world = {
            "Nx": 16,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.25,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.002,
            "x_wind": 4.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "guard_cells": 2,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_collocated_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}

        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = self._one_tile_shape(world)
        world = self._world_with_tiled_grids(world, tile_shape)
        Nt = 1500

        E, B, J, rho, phi, external_fields = self._empty_fields(world)
        fields = (E, B, J, rho, phi, external_fields)
        tiled_particles, species_config, tiled_fields = self._build_tiled_state(
            self._long_two_stream_species(world),
            fields,
            world,
            tile_shape,
        )

        return tiled_particles, species_config, tiled_fields, world, constants, tile_shape, Nt

    def test_scalar_tiles_round_trip_global_rho_and_phi(self):
        world = self._build_world()
        tile_shape = world["tile_shape"]
        scalar = jnp.arange((world["Nx"] + 2) * (world["Ny"] + 2) * (world["Nz"] + 2), dtype=float)
        scalar = scalar.reshape((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

        scalar_tiles = tile_scalar_field(scalar, world, tile_shape)
        scalar_from_tiles = assemble_tiled_scalar_field(scalar_tiles, world, tile_shape)

        self.assertTrue(jnp.allclose(scalar_from_tiles[1:-1, 1:-1, 1:-1], scalar[1:-1, 1:-1, 1:-1]))

    def test_tiled_rho_assembles_to_global_rho(self):
        world = self._build_world(shape_factor=1)
        constants = {"alpha": 1.0}
        species = self._species(world)
        tiled_particles, species_config = to_tiled_particles([species], world, self._simulation_parameters())
        _, _, _, rho, _, _ = self._empty_fields(world)
        rho_tiles = tile_scalar_field(rho, world, world["tile_shape"])

        tiled_rho = compute_rho(
            tiled_particles,
            species_config,
            rho_tiles,
            constants,
            world,
        )
        assembled_rho = assemble_tiled_scalar_field(tiled_rho, world, world["tile_shape"], num_guard_cells=int(world["guard_cells"]))

        reference_world = self._world_with_tiled_grids(world, self._one_tile_shape(world))
        reference_particles, reference_species_config = to_tiled_particles(
            [self._species(reference_world)],
            reference_world,
            self._simulation_parameters_for_tile_shape(reference_world["tile_shape"]),
        )
        reference_rho_tiles = tile_scalar_field(rho, reference_world, reference_world["tile_shape"], num_guard_cells=int(reference_world["guard_cells"]))
        reference_rho_tiles = compute_rho(
            reference_particles,
            reference_species_config,
            reference_rho_tiles,
            constants,
            reference_world,
        )
        reference_rho = assemble_tiled_scalar_field(
            reference_rho_tiles,
            reference_world,
            reference_world["tile_shape"],
            num_guard_cells=int(reference_world["guard_cells"]),
        )

        self.assertTrue(
            jnp.allclose(
                assembled_rho[1:-1, 1:-1, 1:-1],
                reference_rho[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tiled_electrostatic_step_uses_single_tile_state(self):
        world = self._build_world(shape_factor=1)
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        E, B, J, rho, phi, external_fields = self._empty_fields(world)
        fields = (E, B, J, rho, phi, external_fields)
        tile_shape = self._one_tile_shape(world)
        world = self._world_with_tiled_grids(world, tile_shape)
        tiled_particles, species_config, tiled_fields = self._build_tiled_state(
            self._neutral_species(world),
            fields,
            world,
            tile_shape,
        )
        tiled_particles, tiled_fields = time_loop_electrostatic(
            tiled_particles,
            species_config,
            tiled_fields,
            world,
            constants,
            solver="electrostatic",
            relativistic=False,
            particle_pusher="boris",
        )

        E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, *_ = tiled_fields
        self.assertEqual(E_tiles[0].shape[:3], (1, 1, 1))
        self.assertEqual(B_tiles[0].shape[:3], (1, 1, 1))
        self.assertEqual(J_tiles[0].shape[:3], (1, 1, 1))
        self.assertEqual(rho_tiles.shape[:3], (1, 1, 1))
        self.assertEqual(phi_tiles.shape[:3], (1, 1, 1))
        self.assertFalse(bool(tiled_fields[-1]))
        self.assertTrue(jnp.all(jnp.isfinite(rho_tiles)))
        self.assertTrue(jnp.all(jnp.isfinite(phi_tiles)))

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_ELECTROSTATIC") == "1",
        "Set RUN_SLOW_TILED_ELECTROSTATIC=1 to run the 1500-step single-tile electrostatic check.",
    )
    def test_long_two_stream_single_tile_electrostatic_remains_finite(self):
        tiled_particles, species_config, tiled_fields, world, constants, tile_shape, Nt = self._long_two_stream_state()

        tiled_loop = jax.jit(
            time_loop_electrostatic,
            static_argnames=("solver", "relativistic", "particle_pusher"),
        )

        for step in range(1, Nt + 1):
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                species_config,
                tiled_fields,
                world,
                constants,
                solver="electrostatic",
                relativistic=False,
                particle_pusher="boris",
            )

            E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, *_ = tiled_fields
            self.assertEqual(tuple(world["tile_shape"]), tuple(tile_shape))
            self.assertEqual(E_tiles[0].shape[:3], (1, 1, 1))
            self.assertFalse(bool(tiled_fields[-1]), f"step {step}: tiled particle capacity overflowed")
            for component in E_tiles + B_tiles + J_tiles:
                self.assertTrue(jnp.all(jnp.isfinite(component)), f"step {step}: nonfinite vector field")
            self.assertTrue(jnp.all(jnp.isfinite(rho_tiles)), f"step {step}: nonfinite rho")
            self.assertTrue(jnp.all(jnp.isfinite(phi_tiles)), f"step {step}: nonfinite phi")

    def test_initialize_simulation_accepts_tiled_electrostatic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled electrostatic init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrostatic",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "j_from_rhov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            loop, particles, fields, world, simulation_parameters, *_ = initialize_simulation(toml.loads(toml.dumps(config)))

            self.assertIs(loop, time_loop_electrostatic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(tuple(world["tile_shape"]), (8, 1, 1))
            self.assertEqual(tuple(simulation_parameters["tile_shape"]), (8, 1, 1))
            self.assertEqual(int(simulation_parameters["particle_tile_nx"]), 8)
            self.assertEqual(int(simulation_parameters["particle_tile_ny"]), 1)
            self.assertEqual(int(simulation_parameters["particle_tile_nz"]), 1)
            for vertex_axis, center_axis in zip(world["grids"]["vertex"], world["grids"]["center"]):
                self.assertTrue(jnp.allclose(vertex_axis, center_axis))
            self.assertIn("tiled_center_grid", world["grids"])
            self.assertIn("tiled_vertex_grid", world["grids"])
            self.assertEqual(int(world["guard_cells"]), 2)
            self.assertEqual(world["grids"]["tiled_center_grid"][0].shape, (1, 1, 1, 12))
            self.assertEqual(world["grids"]["tiled_center_grid"][1].shape, (1, 1, 1, 5))
            for tiled_vertex_axis, tiled_center_axis in zip(
                world["grids"]["tiled_vertex_grid"],
                world["grids"]["tiled_center_grid"],
            ):
                self.assertTrue(jnp.allclose(tiled_vertex_axis, tiled_center_axis))
            self.assertEqual(fields[0][0].shape[:3], (1, 1, 1))
            self.assertEqual(fields[3].shape[:3], (1, 1, 1))
            self.assertEqual(fields[4].shape[:3], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
