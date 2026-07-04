import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import toml

from PyPIC3D.evolve import time_loop_electrodynamic
from PyPIC3D.initialization import initialize_simulation, validate_field_solver
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.solvers.yee_tiled import (
    assemble_tiled_vector_field,
    tile_scalar_field,
    update_B,
    update_E,
)
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utils import compute_energy
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


jax.config.update("jax_enable_x64", True)

ROUND_OFF_RTOL = 1.0e-11
ROUND_OFF_ATOL = 1.0e-11


def tile_vector_field(field, world, tile_shape, num_guard_cells=2):
    return tuple(tile_scalar_field(component, world, tile_shape, num_guard_cells) for component in field)


def _update_ghost_cells(field, bc_x, bc_y, bc_z):
    field = jax.lax.cond(
        bc_x == BC_PERIODIC,
        lambda f: f.at[0, :, :].set(f[-2, :, :]).at[-1, :, :].set(f[1, :, :]),
        lambda f: f.at[0, :, :].set(0.0).at[-1, :, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_y == BC_PERIODIC,
        lambda f: f.at[:, 0, :].set(f[:, -2, :]).at[:, -1, :].set(f[:, 1, :]),
        lambda f: f.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        operand=field,
    )
    field = jax.lax.cond(
        bc_z == BC_PERIODIC,
        lambda f: f.at[:, :, 0].set(f[:, :, -2]).at[:, :, -1].set(f[:, :, 1]),
        lambda f: f.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0),
        operand=field,
    )
    return field


class TestTiledYeeIntegration(unittest.TestCase):
    def _build_world(self, boundary_conditions=None):
        if boundary_conditions is None:
            boundary_conditions = {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC}
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
            "shape_factor": 1,
            "guard_cells": 2,
            "current_deposition": "direct",
            "current_filter": "none",
            "boundary_conditions": boundary_conditions,
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
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

    def _empty_fields(self, world):
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        E = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        B = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        J = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
        phi = jnp.zeros(shape)
        rho = jnp.zeros(shape)
        external_fields = (E, B)
        return E, B, J, rho, phi, external_fields, None

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

    def _active_rows(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _long_two_stream_state(self, current_deposition="direct", current_filter="none"):
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
            "shape_factor": 2,
            "guard_cells": 2,
            "current_deposition": current_deposition,
            "current_filter": current_filter,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}

        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (4, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        Nt = 1500

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
        particles = [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]
        reference_species = [
            particle_species(name="electron_left", charge=-1.0, v1=-beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="electron_right", charge=-1.0, v1=beam_speed * jnp.ones_like(x0), **species_kwargs),
            particle_species(name="ion_background", charge=2.0, v1=zeros, **species_kwargs),
        ]

        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        tiled_particles, species_config = to_tiled_particles(particles, tiled_world, simulation_parameters)
        tiled_particles = self._pad_tiled_particle_capacity(tiled_particles, min_slots=12)
        g = int(tiled_world["guard_cells"])
        if current_deposition == "esirkepov":
            J_template = tile_vector_field(J, tiled_world, tile_shape, num_guard_cells=g)
            J_tiles = tuple(jnp.zeros_like(component) for component in J_template)
        else:
            J_tiles = tile_vector_field(J, tiled_world, tile_shape, num_guard_cells=g)

        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape, num_guard_cells=g),
            tile_vector_field(B, tiled_world, tile_shape, num_guard_cells=g),
            J_tiles,
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape, num_guard_cells=g),
                tile_vector_field(external_fields[1], tiled_world, tile_shape, num_guard_cells=g),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            reference_species,
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
            current_deposition,
        )
        reference_particles = self._pad_tiled_particle_capacity(reference_particles, min_slots=12)

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            tiled_world,
            reference_tile_shape,
            tile_shape,
            step=0,
        )

        return (
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_tile_shape,
            reference_world,
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            tile_shape,
            Nt,
        )

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
        active = jnp.pad(tiled_particles.active, scalar_pad_width)

        return TiledParticles(
            x=x,
            u=u,
            active=active,
        )

    def _tiled_species_rows(self, tiled_particles, species_index):
        active = tiled_particles.active[:, :, :, species_index, :]
        x = tiled_particles.x[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        u = tiled_particles.u[:, :, :, species_index, :, :].reshape(-1, 3)[active.reshape(-1)]
        order = jnp.lexsort((u[:, 0], x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]

    def _build_tiled_state(self, particles, fields, world, tile_shape, current_deposition="direct"):
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        tiled_particles, species_config = to_tiled_particles(particles, world, simulation_parameters)
        E, B, J, rho, phi, external_fields, pml_state = fields
        g = int(world["guard_cells"])
        if current_deposition == "esirkepov":
            J_template = tile_vector_field(J, world, tile_shape, num_guard_cells=g)
            J_tiles = tuple(jnp.zeros_like(component) for component in J_template)
        else:
            J_tiles = tile_vector_field(J, world, tile_shape, num_guard_cells=g)
        tiled_fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=g),
            tile_vector_field(B, world, tile_shape, num_guard_cells=g),
            J_tiles,
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=g),
                tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=g),
            ),
            pml_state,
            jnp.asarray(False),
        )
        return tiled_particles, species_config, tiled_fields

    def _jitted_electrodynamic_loop_with_static_world(self, world):
        def loop_with_static_world(
            particles,
            species_config,
            fields,
            constants,
            relativistic=True,
            particle_pusher="boris",
        ):
            return time_loop_electrodynamic(
                particles,
                species_config,
                fields,
                world,
                constants,
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )

        return jax.jit(
            loop_with_static_world,
            static_argnames=("relativistic", "particle_pusher"),
        )

    def _assert_vector_fields_close(self, reference_field, tiled_field, step, name):
        for component_name, reference_component, tiled_component in zip(("x", "y", "z"), reference_field, tiled_field):
            diff = jnp.max(jnp.abs(reference_component - tiled_component))
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: {name}{component_name} max abs diff {diff}",
            )

    def _assert_tiled_state_matches_standard(
        self,
        reference_particles,
        reference_fields,
        tiled_particles,
        tiled_fields,
        world,
        reference_tile_shape,
        tile_shape,
        step,
    ):
        if len(tiled_fields) > 7:
            self.assertFalse(bool(tiled_fields[-1]), f"step {step}: tiled particle capacity overflowed")

        species_count = reference_particles.active.shape[3]
        for species_index in range(species_count):
            reference_x, reference_u = self._tiled_species_rows(reference_particles, species_index)
            tiled_x, tiled_u = self._tiled_species_rows(tiled_particles, species_index)
            self.assertTrue(
                jnp.allclose(tiled_x, reference_x, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} position mismatch",
            )
            self.assertTrue(
                jnp.allclose(tiled_u, reference_u, rtol=ROUND_OFF_RTOL, atol=ROUND_OFF_ATOL),
                f"step {step}: species {species_index} velocity mismatch",
            )

        reference_E, reference_B, reference_J, *_ = reference_fields
        E_tiles, B_tiles, J_tiles, *_ = tiled_fields
        num_guard_cells = int(world.get("guard_cells", 1))
        self._assert_vector_fields_close(
            assemble_tiled_vector_field(reference_E, world, reference_tile_shape, num_guard_cells=num_guard_cells),
            assemble_tiled_vector_field(E_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "E",
        )
        self._assert_vector_fields_close(
            assemble_tiled_vector_field(reference_B, world, reference_tile_shape, num_guard_cells=num_guard_cells),
            assemble_tiled_vector_field(B_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "B",
        )
        self._assert_vector_fields_close(
            assemble_tiled_vector_field(reference_J, world, reference_tile_shape, num_guard_cells=num_guard_cells),
            assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            step,
            "J",
        )

    def _max_tiled_state_difference(
        self,
        reference_particles,
        reference_fields,
        reference_tile_shape,
        tiled_particles,
        tiled_fields,
        world,
        tile_shape,
    ):
        differences = {}
        species_count = reference_particles.active.shape[3]
        for species_index in range(species_count):
            reference_x, reference_u = self._tiled_species_rows(reference_particles, species_index)
            tiled_x, tiled_u = self._tiled_species_rows(tiled_particles, species_index)
            differences[f"x{species_index}"] = float(jnp.max(jnp.abs(tiled_x - reference_x)))
            differences[f"u{species_index}"] = float(jnp.max(jnp.abs(tiled_u - reference_u)))

        reference_E, reference_B, reference_J, *_ = reference_fields
        E_tiles, B_tiles, J_tiles, *_ = tiled_fields
        num_guard_cells = int(world.get("guard_cells", 1))
        for name, reference_field, tiled_field in (
            (
                "E",
                assemble_tiled_vector_field(reference_E, world, reference_tile_shape, num_guard_cells=num_guard_cells),
                assemble_tiled_vector_field(E_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            ),
            (
                "B",
                assemble_tiled_vector_field(reference_B, world, reference_tile_shape, num_guard_cells=num_guard_cells),
                assemble_tiled_vector_field(B_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            ),
            (
                "J",
                assemble_tiled_vector_field(reference_J, world, reference_tile_shape, num_guard_cells=num_guard_cells),
                assemble_tiled_vector_field(J_tiles, world, tile_shape, num_guard_cells=num_guard_cells),
            ),
        ):
            differences[name] = max(float(jnp.max(jnp.abs(tiled - reference))) for reference, tiled in zip(reference_field, tiled_field))

        return differences

    def test_validate_field_solver_accepts_tiled_yee(self):
        validate_field_solver("electrodynamic_yee")

    def test_tiled_yee_field_update_applies_standard_field_filter(self):
        world = self._build_world()
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.6}
        tile_shape = (2, 1, 1)
        shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))

        x = jnp.arange(world["Nx"], dtype=jnp.float64)[:, None, None]
        seed = jnp.zeros(shape)
        Ex = _update_ghost_cells(seed.at[interior].set(0.1 + 0.03 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Ey = _update_ghost_cells(seed.at[interior].set(0.2 - 0.02 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Ez = _update_ghost_cells(seed.at[interior].set(0.05 * jnp.sin(x + 1.0)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Bx = _update_ghost_cells(seed.at[interior].set(0.07 * jnp.cos(x + 0.5)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        By = _update_ghost_cells(seed.at[interior].set(0.04 + 0.01 * x), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        Bz = _update_ghost_cells(seed.at[interior].set(0.03 * jnp.sin(2.0 * x)), BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)
        E = (Ex, Ey, Ez)
        B = (Bx, By, Bz)
        J = (seed, seed, seed)
        rho = seed
        phi = seed
        external_fields = ((seed, seed, seed), (seed, seed, seed))
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        reference_fields = (
            tile_vector_field(E, reference_world, reference_tile_shape),
            tile_vector_field(B, reference_world, reference_tile_shape),
            tile_vector_field(J, reference_world, reference_tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], reference_world, reference_tile_shape),
                tile_vector_field(external_fields[1], reference_world, reference_tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        reference_E, reference_pml_state = update_E(
            reference_fields[0],
            reference_fields[1],
            reference_fields[2],
            reference_world,
            constants,
        )
        reference_B, reference_pml_state = update_B(
            reference_E,
            reference_fields[1],
            reference_world,
            constants,
            reference_pml_state,
        )
        reference_fields = (reference_E, reference_B, *reference_fields[2:])

        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape),
            tile_vector_field(B, tiled_world, tile_shape),
            tile_vector_field(J, tiled_world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape),
                tile_vector_field(external_fields[1], tiled_world, tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        tiled_E, tiled_pml_state = update_E(tiled_fields[0], tiled_fields[1], tiled_fields[2], tiled_world, constants)
        tiled_B, tiled_pml_state = update_B(tiled_E, tiled_fields[1], tiled_world, constants, tiled_pml_state)
        tiled_fields = (tiled_E, tiled_B, *tiled_fields[2:])

        reference_E, reference_B, *_ = reference_fields
        E_tiles, B_tiles, *_ = tiled_fields
        self._assert_vector_fields_close(
            assemble_tiled_vector_field(reference_E, reference_world, reference_tile_shape),
            assemble_tiled_vector_field(E_tiles, tiled_world, tile_shape),
            1,
            "E",
        )
        self._assert_vector_fields_close(
            assemble_tiled_vector_field(reference_B, reference_world, reference_tile_shape),
            assemble_tiled_vector_field(B_tiles, tiled_world, tile_shape),
            1,
            "B",
        )

    def test_initialize_simulation_uses_tiled_particles_and_fields_for_tiled_yee(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
            config_path = os.path.join(tmpdir, "tiled_yee.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(tuple(world["tile_shape"]), (2, 1, 1))

    def test_initialize_simulation_accepts_tiled_yee_higuera_cary_pusher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee higuera cary init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
                    "particle_pusher": "higuera_cary",
                    "relativistic": True,
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
            config_path = os.path.join(tmpdir, "tiled_yee_higuera_cary.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["particle_pusher"], "higuera_cary")

    def test_initialize_simulation_accepts_tiled_yee_digital_current_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee digital filter init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
                    "filter_j": "digital",
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
            config_path = os.path.join(tmpdir, "tiled_yee_digital.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["filter_j"], "digital")

    def test_initialize_simulation_accepts_tiled_yee_bilinear_current_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee bilinear filter init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
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
                    "filter_j": "bilinear",
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
            config_path = os.path.join(tmpdir, "tiled_yee_bilinear.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, simulation_parameters, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(simulation_parameters["filter_j"], "bilinear")

    def test_initialize_simulation_accepts_tiled_yee_conducting_field_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled yee conducting init smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "x_bc": "conducting",
                    "y_bc": "conducting",
                    "z_bc": "conducting",
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
            config_path = os.path.join(tmpdir, "tiled_yee_conducting.toml")
            with open(config_path, "w") as f:
                toml.dump(config, f)

            loop, particles, fields, world, *_rest = initialize_simulation(toml.load(config_path))

            self.assertIs(loop.func if hasattr(loop, "func") else loop, time_loop_electrodynamic)
            self.assertIsInstance(particles, TiledParticles)
            self.assertEqual(fields[0][0].ndim, 6)
            self.assertEqual(world["boundary_conditions"], {"x": 1, "y": 1, "z": 1})

    def test_tiled_yee_step_matches_standard_periodic_yee_step(self):
        world = self._build_world()
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
        )

        tiled_particles, species_config = to_tiled_particles([species], tiled_world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape),
            tile_vector_field(B, tiled_world, tile_shape),
            tile_vector_field(J, tiled_world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape),
                tile_vector_field(external_fields[1], tiled_world, tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_fields = time_loop_electrodynamic(
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            reference_tile_shape,
            tile_shape,
            1,
        )
        reference_E, reference_B, *_ = reference_fields
        E_tiles, B_tiles, *_ = tiled_fields
        reference_energy = compute_energy(
            reference_particles,
            reference_E,
            reference_B,
            world,
            constants,
            species_config=reference_species_config,
        )
        tiled_energy = compute_energy(tiled_particles, E_tiles, B_tiles, world, constants, species_config=species_config)
        for reference, tiled in zip(reference_energy, tiled_energy):
            self.assertTrue(jnp.allclose(tiled, reference, rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_yee_step_matches_standard_conducting_yee_step(self):
        world = self._build_world(
            boundary_conditions={"x": BC_CONDUCTING, "y": BC_CONDUCTING, "z": BC_CONDUCTING}
        )
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
        )

        tiled_particles, species_config = to_tiled_particles([species], tiled_world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape),
            tile_vector_field(B, tiled_world, tile_shape),
            tile_vector_field(J, tiled_world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape),
                tile_vector_field(external_fields[1], tiled_world, tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_fields = time_loop_electrodynamic(
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            reference_tile_shape,
            tile_shape,
            1,
        )

    def test_tiled_yee_step_uses_digital_filtered_current(self):
        world = self._build_world()
        world["current_filter"] = "digital"
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.6}
        tile_shape = (2, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
        )

        tiled_particles, species_config = to_tiled_particles([species], tiled_world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape),
            tile_vector_field(B, tiled_world, tile_shape),
            tile_vector_field(J, tiled_world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape),
                tile_vector_field(external_fields[1], tiled_world, tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_fields = time_loop_electrodynamic(
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            reference_tile_shape,
            tile_shape,
            1,
        )

    def test_tiled_yee_step_uses_bilinear_filtered_current(self):
        world = self._build_world()
        world["current_filter"] = "bilinear"
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = self._species(world)
        reference_species = self._species(world)
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)
        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
        )

        tiled_particles, species_config = to_tiled_particles([species], tiled_world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape),
            tile_vector_field(B, tiled_world, tile_shape),
            tile_vector_field(J, tiled_world, tile_shape),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape),
                tile_vector_field(external_fields[1], tiled_world, tile_shape),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_fields = time_loop_electrodynamic(
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            reference_tile_shape,
            tile_shape,
            1,
        )

    def test_tiled_yee_step_bilinear_quadratic_two_guards_reduced_axes_matches_standard(self):
        world = self._build_world()
        world["shape_factor"] = 2
        world["guard_cells"] = 2
        world["current_filter"] = "bilinear"
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 1, 1)
        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        reference_world = self._world_with_tiled_grids(world, reference_tile_shape)
        tiled_world = self._world_with_tiled_grids(world, tile_shape)
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

        species = particle_species(
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
            v2=jnp.array([0.03, -0.04, 0.02, -0.01]),
            v3=jnp.array([-0.02, 0.01, -0.03, 0.04]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        reference_species = particle_species(
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
            v2=jnp.array([0.03, -0.04, 0.02, -0.01]),
            v3=jnp.array([-0.02, 0.01, -0.03, 0.04]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        E, B, J, rho, phi, external_fields, pml_state = self._empty_fields(world)

        reference_particles, reference_species_config, reference_fields = self._build_tiled_state(
            [reference_species],
            (E, B, J, rho, phi, external_fields, pml_state),
            reference_world,
            reference_tile_shape,
        )

        tiled_particles, species_config = to_tiled_particles([species], tiled_world, simulation_parameters)
        tiled_fields = (
            tile_vector_field(E, tiled_world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(B, tiled_world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(J, tiled_world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], tiled_world, tile_shape, num_guard_cells=int(world["guard_cells"])),
                tile_vector_field(external_fields[1], tiled_world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            ),
            None,
            jnp.asarray(False),
        )
        reference_particles, reference_fields = time_loop_electrodynamic(
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )
        tiled_particles, tiled_fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            relativistic=False,
            particle_pusher="boris",
        )

        self._assert_tiled_state_matches_standard(
            reference_particles,
            reference_fields,
            tiled_particles,
            tiled_fields,
            world,
            reference_tile_shape,
            tile_shape,
            1,
        )

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_YEE") == "1",
        "Set RUN_SLOW_TILED_YEE=1 to run the 1500-step tiled Yee two-stream comparison.",
    )
    def test_long_two_stream_tiled_yee_direct_current_roundoff_origin_probe(self):
        (
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_tile_shape,
            reference_world,
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            tile_shape,
            Nt,
        ) = self._long_two_stream_state("direct", "none")

        reference_loop = self._jitted_electrodynamic_loop_with_static_world(reference_world)
        tiled_loop = self._jitted_electrodynamic_loop_with_static_world(tiled_world)

        for t in range(Nt):
            reference_particles, reference_fields = reference_loop(
                reference_particles,
                reference_species_config,
                reference_fields,
                constants,
                relativistic=False,
                particle_pusher="boris",
            )
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                species_config,
                tiled_fields,
                constants,
                relativistic=False,
                particle_pusher="boris",
            )

            differences = self._max_tiled_state_difference(
                reference_particles,
                reference_fields,
                reference_tile_shape,
                tiled_particles,
                tiled_fields,
                tiled_world,
                tile_shape,
            )
            if t == 0:
                self._assert_tiled_state_matches_standard(
                    reference_particles,
                    reference_fields,
                    tiled_particles,
                    tiled_fields,
                    tiled_world,
                    reference_tile_shape,
                    tile_shape,
                    t + 1,
                )
            if any(value > ROUND_OFF_ATOL for value in differences.values()):
                print(f"direct-current roundoff probe first exceeded {ROUND_OFF_ATOL} at step {t + 1}: {differences}")
                break

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TILED_YEE") == "1",
        "Set RUN_SLOW_TILED_YEE=1 to run the 1500-step tiled Yee two-stream comparison.",
    )
    def test_long_two_stream_tiled_yee_esirkepov_roundoff_origin_probe(self):
        (
            reference_particles,
            reference_species_config,
            reference_fields,
            reference_tile_shape,
            reference_world,
            tiled_particles,
            species_config,
            tiled_fields,
            tiled_world,
            constants,
            tile_shape,
            Nt,
        ) = self._long_two_stream_state("esirkepov", "none")

        reference_loop = self._jitted_electrodynamic_loop_with_static_world(reference_world)
        tiled_loop = self._jitted_electrodynamic_loop_with_static_world(tiled_world)

        for t in range(Nt):
            reference_particles, reference_fields = reference_loop(
                reference_particles,
                reference_species_config,
                reference_fields,
                constants,
                relativistic=False,
                particle_pusher="boris",
            )
            tiled_particles, tiled_fields = tiled_loop(
                tiled_particles,
                species_config,
                tiled_fields,
                constants,
                relativistic=False,
                particle_pusher="boris",
            )

            differences = self._max_tiled_state_difference(
                reference_particles,
                reference_fields,
                reference_tile_shape,
                tiled_particles,
                tiled_fields,
                tiled_world,
                tile_shape,
            )
            if t == 0:
                self._assert_tiled_state_matches_standard(
                    reference_particles,
                    reference_fields,
                    tiled_particles,
                    tiled_fields,
                    tiled_world,
                    reference_tile_shape,
                    tile_shape,
                    t + 1,
                )
            if any(value > ROUND_OFF_ATOL for value in differences.values()):
                print(f"Esirkepov roundoff probe first exceeded {ROUND_OFF_ATOL} at step {t + 1}: {differences}")
                break


if __name__ == "__main__":
    unittest.main()
