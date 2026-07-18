import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import (
    PML_WALLS,
    build_pml,
    initialize_pml_state,
    initialize_tiled_pml_state,
    load_pml_from_toml,
    tile_pml_profiles,
)
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_vector_field
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.solvers.first_order_yee import (
    update_B,
    update_E,
)
from PyPIC3D.utilities.grids import build_yee_grid
from PyPIC3D.utils import compute_energy
from tests.parameter_helpers import field_initialization_parameters

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
    # get the number of tiles along each axis

    interior_tiles = field[1:-1, 1:-1, 1:-1]
    interior_tiles = interior_tiles.reshape(ntx, tile_nx, nty, tile_ny, ntz, tile_nz)
    interior_tiles = interior_tiles.transpose(0, 2, 4, 1, 3, 5)
    # reshape the interior of the field into tiles, and then transpose to get the correct order of axes

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
    field_tiles = field_tiles.at[:, :, :, g:-g, g:-g, g:-g].set(interior_tiles)
    # populate the field tiles with the interior tiles, leaving the guard cells as zeros

    world = dict(world)
    world["tile_shape"] = tuple(int(width) for width in tile_shape)
    world["field_mesh"] = ghost_cells.make_field_mesh((ntx, nty, ntz))
    static_parameters, _ = field_initialization_parameters(world)
    return ghost_cells.update_tiled_ghost_cells(field_tiles, static_parameters, g)
    # update the guard cells of the tiled field using the ghost_cells function


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


def _update_ghost_cells_from_world(field, world):
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

    return _update_ghost_cells(field, bc_x, bc_y, bc_z)


def _empty_global_fields(world):
    shape = (world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2)
    E = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    B = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    J = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    return E, B, J


def _base_world(nx=24, ny=1, nz=1):
    world = {
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "dx": 1.0 / nx,
        "dy": 1.0,
        "dz": 1.0,
        "dt": 0.5 / nx,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }
    vertex_grid, center_grid = build_yee_grid(SimpleNamespace(**world))
    world["grids"] = {"center": center_grid, "vertex": vertex_grid}
    return world


def _dynamic_parameters(world, constants=None):
    if constants is None:
        constants = {}
    return SimpleNamespace(
        Nx=world["Nx"],
        Ny=world["Ny"],
        Nz=world["Nz"],
        dx=world["dx"],
        dy=world["dy"],
        dz=world["dz"],
        dt=world["dt"],
        x_wind=world["x_wind"],
        y_wind=world["y_wind"],
        z_wind=world["z_wind"],
        C=world.get("C", constants.get("C", 1.0)),
        eps=world.get("eps", constants.get("eps", 1.0)),
        mu=world.get("mu", constants.get("mu", 1.0)),
        alpha=world.get("alpha", constants.get("alpha", 1.0)),
        grids=SimpleNamespace(**world["grids"]),
    )


def _empty_config(tmpdir, solver="electrodynamic_yee", pml=None):
    sim = {
        "name": "pml init test",
        "output_dir": tmpdir,
        "solver": solver,
        "Nx": 8,
        "Ny": 1,
        "Nz": 1,
        "x_wind": 1.0,
        "y_wind": 1.0,
        "z_wind": 1.0,
        "Nt": 1,
        "dt": 1e-10,
    }
    config = {"simulation_parameters": sim, "plotting": {"plotting": False}}
    if pml is not None:
        config["pml"] = pml
    return config


def _load_pml(raw_pml, world, constants):
    dynamic_parameters = _dynamic_parameters(world, constants)
    return load_pml_from_toml(raw_pml, None, dynamic_parameters)


def _initialize_pml_state(world, constants=None):
    dynamic_parameters = _dynamic_parameters(world, constants)
    return initialize_pml_state(dynamic_parameters, world["pml"][-1])


def _initialize_tiled_pml_state(world, tile_shape, constants=None):
    world = {**world, "tile_shape": tile_shape}
    static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
    return initialize_tiled_pml_state(static_parameters, dynamic_parameters, world["pml"][-1], tile_shape)


def _tile_pml_profiles(world, tile_shape, constants=None):
    world = {**world, "tile_shape": tile_shape}
    static_parameters, _ = field_initialization_parameters(world, constants)
    return tile_pml_profiles(static_parameters, world["pml"][-1], tile_shape)


class TestPMLConfiguration(unittest.TestCase):
    def test_load_pml_from_toml_accepts_all_six_walls(self):
        raw = [
            {"wall": wall, "thickness": 2, "order": 3.0, "target_reflection": 1e-8}
            for wall in PML_WALLS
        ]

        active, pml_x, pml_y, pml_z, profiles = _load_pml(
            raw,
            _base_world(nx=8, ny=8, nz=8),
            {"C": 3.0},
        )
        sigma_x, sigma_y, sigma_z = profiles

        self.assertTrue(active)
        self.assertTrue(pml_x)
        self.assertTrue(pml_y)
        self.assertTrue(pml_z)
        self.assertEqual(sigma_x.shape, (10, 10, 10))
        self.assertTrue(jnp.any(sigma_x > 0.0))
        self.assertTrue(jnp.any(sigma_y > 0.0))
        self.assertTrue(jnp.any(sigma_z > 0.0))

    def test_load_pml_from_toml_rejects_invalid_duplicate_and_oversized_walls(self):
        world = _base_world(nx=8, ny=1, nz=1)

        with self.assertRaisesRegex(ValueError, "Invalid PML wall"):
            _load_pml([{"wall": "x+", "thickness": 2}], world, {"C": 3.0})

        with self.assertRaisesRegex(ValueError, "Duplicate PML wall"):
            _load_pml(
                [{"wall": "+x", "thickness": 2}, {"wall": "+x", "thickness": 2}],
                world,
                {"C": 3.0},
            )

        with self.assertRaisesRegex(ValueError, "exceeds active cells"):
            _load_pml([{"wall": "+x", "thickness": 9}], world, {"C": 3.0})

    def test_build_pml_ramp_only_on_requested_side(self):
        world = _base_world(nx=8, ny=1, nz=1)
        pml_layers = (("+x", "x", 3, 2.0, 9.0),)

        sigma_x, sigma_y, sigma_z = build_pml(_dynamic_parameters(world), pml_layers)

        self.assertEqual(sigma_x.shape, (10, 3, 3))
        self.assertTrue(jnp.allclose(sigma_x[1:5, :, :], 0.0))
        self.assertTrue(jnp.all(sigma_x[-4:-1, :, :] > 0.0))
        self.assertTrue(float(sigma_x[-2, 1, 1]) > float(sigma_x[-4, 1, 1]))
        self.assertTrue(jnp.allclose(sigma_y, 0.0))
        self.assertTrue(jnp.allclose(sigma_z, 0.0))

    def test_build_pml_ramps_from_interior_interface_to_outer_wall(self):
        world = _base_world(nx=8, ny=1, nz=1)
        pml_layers = (
            ("-x", "x", 3, 2.0, 9.0),
            ("+x", "x", 3, 2.0, 9.0),
        )

        sigma_x, _, _ = build_pml(_dynamic_parameters(world), pml_layers)
        sigma_x = sigma_x[:, 1, 1]

        self.assertGreater(float(sigma_x[1]), float(sigma_x[2]))
        self.assertGreater(float(sigma_x[2]), float(sigma_x[3]))
        self.assertGreater(float(sigma_x[-2]), float(sigma_x[-3]))
        self.assertGreater(float(sigma_x[-3]), float(sigma_x[-4]))


class TestPMLInitialization(unittest.TestCase):
    def test_initialize_pml_state_uses_physical_interior_shape(self):
        world = _base_world(nx=8, ny=4, nz=2)
        world["pml"] = (False, False, False, False, build_pml(_dynamic_parameters(world), ()))

        pml_state = _initialize_pml_state(world)
        e_memory, b_memory, _ = pml_state

        self.assertEqual(len(e_memory), 6)
        self.assertEqual(len(b_memory), 6)
        for memory in e_memory:
            self.assertEqual(memory.shape, (8, 4, 2))
        for memory in b_memory:
            self.assertEqual(memory.shape, (8, 4, 2))

    def test_initialize_tiled_pml_state_uses_tile_local_interior_shape(self):
        world = _base_world(nx=8, ny=4, nz=2)
        world["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "sigma_max": 3.0}],
            world,
            {"C": 1.0},
        )
        tile_shape = (2, 2, 1)

        pml_state = _initialize_tiled_pml_state(world, tile_shape, {"C": 1.0})
        e_memory, b_memory, tiled_profiles = pml_state

        self.assertEqual(len(e_memory), 6)
        self.assertEqual(len(b_memory), 6)
        for memory in e_memory:
            self.assertEqual(memory.shape, (4, 2, 2, 2, 2, 1))
        for memory in b_memory:
            self.assertEqual(memory.shape, (4, 2, 2, 2, 2, 1))
        for profile in tiled_profiles:
            self.assertEqual(profile.shape, (4, 2, 2, 6, 6, 5))

    def test_tile_pml_profiles_matches_global_profiles_on_tile_interiors(self):
        world = _base_world(nx=8, ny=4, nz=2)
        world["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 2, "sigma_max": 3.0},
                {"wall": "+x", "thickness": 2, "sigma_max": 3.0},
            ],
            world,
            {"C": 1.0},
        )
        tile_shape = (2, 2, 1)

        tiled_profiles = _tile_pml_profiles(world, tile_shape, {"C": 1.0})
        global_profiles = world["pml"][-1]
        assembled_profiles = assemble_tiled_vector_field(tiled_profiles, world, tile_shape, num_guard_cells=int(world["guard_cells"]))

        for assembled, reference in zip(assembled_profiles, global_profiles):
            self.assertTrue(
                jnp.allclose(
                    assembled[1:-1, 1:-1, 1:-1],
                    reference[1:-1, 1:-1, 1:-1],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

    def test_initialize_simulation_rejects_pml_for_electrostatic_solver(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "PML is only supported"):
                initialize_simulation(_empty_config(tmpdir, solver="electrostatic", pml=pml))

    def test_initialize_simulation_appends_pml_state_for_electrodynamic_yee(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir, pml=pml))

        fields = result[2]
        static_parameters = result[3]
        pml_state = fields[6]
        _, _, tiled_profiles = pml_state
        self.assertEqual(len(fields), 8)
        self.assertTrue(static_parameters.pml_active)
        self.assertIsNotNone(pml_state)
        self.assertTrue(jnp.any(tiled_profiles[0] > 0.0))
        self.assertTrue(jnp.allclose(tiled_profiles[1], 0.0))
        self.assertTrue(jnp.allclose(tiled_profiles[2], 0.0))
        self.assertEqual(static_parameters.boundary_conditions, (BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC))

    def test_initialize_simulation_uses_tiled_pml_state_for_electrodynamic_yee(self):
        pml = [{"wall": "+x", "thickness": 2, "sigma_max": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _empty_config(tmpdir, solver="electrodynamic_yee", pml=pml)
            config["simulation_parameters"].update(
                {
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "filter_j": "none",
                }
            )
            result = initialize_simulation(config)

        fields = result[2]
        static_parameters = result[3]
        pml_state = fields[6]
        overflow = fields[-1]
        e_memory, b_memory, tiled_profiles = pml_state

        self.assertEqual(len(fields), 8)
        self.assertFalse(bool(overflow))
        self.assertEqual(tuple(static_parameters.tile_shape), (2, 1, 1))
        self.assertEqual(e_memory[0].shape, (4, 1, 1, 2, 1, 1))
        self.assertEqual(b_memory[0].shape, (4, 1, 1, 2, 1, 1))
        self.assertEqual(int(static_parameters.guard_cells), 2)
        self.assertEqual(tiled_profiles[0].shape, (4, 1, 1, 6, 5, 5))

    def test_initialize_simulation_uses_none_pml_state_without_pml_for_electrodynamic_yee(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = initialize_simulation(_empty_config(tmpdir))

        fields = result[2]
        static_parameters = result[3]

        self.assertEqual(len(fields), 8)
        self.assertIsNone(fields[6])
        self.assertFalse(bool(fields[-1]))
        self.assertFalse(static_parameters.pml_active)


class TestPMLFDTDBehavior(unittest.TestCase):
    def test_no_pml_state_returns_field_and_none_state(self):
        world = _base_world(nx=4, ny=4, nz=4)
        constants = {"C": 2.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        tile_shape = (2, 2, 2)
        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((2, 2, 2))
        E, B, J = _empty_global_fields(world)
        B = (B[0], B[1], B[2].at[1:-1, 1:-1, 1:-1].set(1.0))

        E_tiles = tile_vector_field(E, world, tile_shape)
        B_tiles = tile_vector_field(B, world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
        E_after, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters)
        B_after, pml_state = update_B(E_after, B_tiles, static_parameters, dynamic_parameters, pml_state)

        self.assertEqual(len(E_after), 3)
        self.assertEqual(len(B_after), 3)
        self.assertIsNone(pml_state)

    def test_tiled_pml_matches_single_tile_pml_for_one_yee_step(self):
        world = _base_world(nx=8, ny=4, nz=2)
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        world["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 2, "sigma_max": 4.0},
                {"wall": "+x", "thickness": 2, "sigma_max": 4.0},
            ],
            world,
            constants,
        )
        world["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (2, 2, 1)
        E, B, J = _empty_global_fields(world)

        x = world["grids"]["vertex"][0][1:-1]
        y = world["grids"]["vertex"][1][1:-1]
        z = world["grids"]["vertex"][2][1:-1]
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Jx, Jy, Jz = J
        Ey = Ey.at[1:-1, 1:-1, 1:-1].set(jnp.sin(2.0 * jnp.pi * X) + 0.1 * Y)
        Ez = Ez.at[1:-1, 1:-1, 1:-1].set(0.2 * X - 0.3 * Z)
        By = By.at[1:-1, 1:-1, 1:-1].set(0.4 * X + 0.2 * Z)
        Bz = Bz.at[1:-1, 1:-1, 1:-1].set(jnp.cos(2.0 * jnp.pi * X) - 0.1 * Y)
        Jx = Jx.at[1:-1, 1:-1, 1:-1].set(0.05 * X)
        Jy = Jy.at[1:-1, 1:-1, 1:-1].set(-0.02 * Y)
        Jz = Jz.at[1:-1, 1:-1, 1:-1].set(0.03 * Z)
        E = (Ex, Ey, Ez)
        B = (Bx, By, Bz)
        J = (Jx, Jy, Jz)
        E = tuple(_update_ghost_cells_from_world(component, world) for component in E)
        B = tuple(_update_ghost_cells_from_world(component, world) for component in B)

        reference_tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        world["tile_shape"] = reference_tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((1, 1, 1))
        reference_static_parameters, reference_dynamic_parameters = field_initialization_parameters(world, constants)
        reference_pml_state = initialize_tiled_pml_state(
            reference_static_parameters,
            reference_dynamic_parameters,
            world["pml"][-1],
            reference_tile_shape,
        )
        reference_E, reference_pml_state = update_E(
            tile_vector_field(E, world, reference_tile_shape),
            tile_vector_field(B, world, reference_tile_shape),
            tile_vector_field(J, world, reference_tile_shape),
            reference_static_parameters,
            reference_dynamic_parameters,
            reference_pml_state,
        )
        reference_B, reference_pml_state = update_B(
            reference_E,
            tile_vector_field(B, world, reference_tile_shape),
            reference_static_parameters,
            reference_dynamic_parameters,
            reference_pml_state,
        )
        E_reference = assemble_tiled_vector_field(
            reference_E,
            world,
            reference_tile_shape,
            num_guard_cells=int(world["guard_cells"]),
        )
        B_reference = assemble_tiled_vector_field(
            reference_B,
            world,
            reference_tile_shape,
            num_guard_cells=int(world["guard_cells"]),
        )

        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((4, 2, 2))
        tiled_static_parameters, tiled_dynamic_parameters = field_initialization_parameters(world, constants)
        tiled_pml_state = initialize_tiled_pml_state(
            tiled_static_parameters,
            tiled_dynamic_parameters,
            world["pml"][-1],
            tile_shape,
        )
        E_tiles, tiled_pml_state = update_E(
            tile_vector_field(E, world, tile_shape),
            tile_vector_field(B, world, tile_shape),
            tile_vector_field(J, world, tile_shape),
            tiled_static_parameters,
            tiled_dynamic_parameters,
            tiled_pml_state,
        )
        B_tiles, tiled_pml_state = update_B(
            E_tiles,
            tile_vector_field(B, world, tile_shape),
            tiled_static_parameters,
            tiled_dynamic_parameters,
            tiled_pml_state,
        )

        E_tiled = assemble_tiled_vector_field(E_tiles, world, tile_shape, num_guard_cells=int(world["guard_cells"]))
        B_tiled = assemble_tiled_vector_field(B_tiles, world, tile_shape, num_guard_cells=int(world["guard_cells"]))

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        for reference, tiled in zip(E_reference, E_tiled):
            self.assertTrue(jnp.allclose(tiled[interior], reference[interior], rtol=1.0e-12, atol=1.0e-12))
        for reference, tiled in zip(B_reference, B_tiled):
            self.assertTrue(jnp.allclose(tiled[interior], reference[interior], rtol=1.0e-12, atol=1.0e-12))

    def test_tiled_pml_absorbs_field_energy_in_particle_free_1d_wave(self):
        world = _base_world(nx=40, ny=1, nz=1)
        constants = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        world["pml"] = _load_pml(
            [
                {"wall": "-x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
                {"wall": "+x", "thickness": 8, "order": 3.0, "sigma_max": 60.0},
            ],
            world,
            constants,
        )
        world["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (4, 1, 1)
        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((10, 1, 1))
        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
        tiled_pml_state = initialize_tiled_pml_state(
            static_parameters,
            dynamic_parameters,
            world["pml"][-1],
            tile_shape,
        )
        E, B, J = _empty_global_fields(world)

        x = world["grids"]["vertex"][0][1:-1]
        pulse = jnp.exp(-((x + 0.30) / 0.04) ** 2)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(pulse)
        Bz = Bz.at[1:-1, 1, 1].set(pulse)
        E_tiles = tile_vector_field((Ex, Ey, Ez), world, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), world, tile_shape)
        J_tiles = tile_vector_field(J, world, tile_shape)

        empty_particles = TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        )
        empty_species_config = SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
            update_u=jnp.zeros((0, 3), dtype=bool),
        )

        initial_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        def step(E_tiles, B_tiles, tiled_pml_state):
            E_tiles, tiled_pml_state = update_E(
                E_tiles,
                B_tiles,
                J_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            B_tiles, tiled_pml_state = update_B(
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                tiled_pml_state,
            )
            return E_tiles, B_tiles, tiled_pml_state

        step = jax.jit(step)
        for _ in range(60):
            E_tiles, B_tiles, tiled_pml_state = step(E_tiles, B_tiles, tiled_pml_state)

        final_energy = sum(
            compute_energy(
                empty_particles,
                E_tiles,
                B_tiles,
                static_parameters,
                dynamic_parameters,
                species_config=empty_species_config,
            )[:2]
        )
        self.assertTrue(jnp.isfinite(final_energy))
        self.assertLess(float(final_energy), 0.65 * float(initial_energy))

    def test_tiled_pml_step_uses_shared_guard_current_without_changing_timing(self):
        world = _base_world(nx=8, ny=1, nz=1)
        constants = {"C": 1.0, "eps": 2.0, "mu": 1.0, "alpha": 1.0}
        world["pml"] = _load_pml(
            [{"wall": "+x", "thickness": 2, "order": 3.0, "sigma_max": 4.0}],
            world,
            constants,
        )
        world["boundary_conditions"]["x"] = BC_CONDUCTING
        tile_shape = (2, 1, 1)
        world["tile_shape"] = tile_shape
        world["field_mesh"] = ghost_cells.make_field_mesh((4, 1, 1))
        E, B, J = _empty_global_fields(world)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        Ey = Ey.at[1:-1, 1, 1].set(jnp.linspace(0.0, 0.3, world["Nx"]))
        Bz = Bz.at[1:-1, 1, 1].set(jnp.linspace(0.1, -0.2, world["Nx"]))
        E_tiles = tile_vector_field((Ex, Ey, Ez), world, tile_shape)
        B_tiles = tile_vector_field((Bx, By, Bz), world, tile_shape)

        g = int(world["guard_cells"])
        J_tiles = tile_vector_field(J, world, tile_shape, num_guard_cells=g)
        Jx, Jy, Jz = J_tiles
        Jx = Jx.at[:, :, :, 1:-1, 1:-1, 1:-1].set(0.25)
        J_tiles = (Jx, Jy, Jz)

        static_parameters, dynamic_parameters = field_initialization_parameters(world, constants)
        pml_state = initialize_tiled_pml_state(static_parameters, dynamic_parameters, world["pml"][-1], tile_shape)
        E_after, pml_state = update_E(E_tiles, B_tiles, J_tiles, static_parameters, dynamic_parameters, pml_state)
        B_after, pml_state = update_B(E_after, B_tiles, static_parameters, dynamic_parameters, pml_state)

        for component in E_after + B_after:
            self.assertTrue(jnp.all(jnp.isfinite(component)))
        for memory in pml_state[0] + pml_state[1]:
            self.assertTrue(jnp.all(jnp.isfinite(memory)))


if __name__ == "__main__":
    unittest.main()
