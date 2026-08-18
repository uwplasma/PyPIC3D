import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.fluid_quantities import compute_velocity_field, fluid_velocity
from PyPIC3D.diagnostics.output_adapters import (
    assemble_tiled_scalar_field,
    build_field_output_map,
)
from tests.kernel_fixtures import build_tiled_particles, field_tiles_from_global, kernel_parameters, particle_species


jax.config.update("jax_enable_x64", True)


def tile_scalar_field(field, static_parameters, dynamic_parameters, num_guard_cells=None):
    return field_tiles_from_global(field, static_parameters, dynamic_parameters, num_guard_cells)


class TestTiledFluidQuantities(unittest.TestCase):
    def _build_parameters(
        self,
        shape_factor=2,
        tile_shape=None,
        solver="electrodynamic_yee",
    ):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        if tile_shape is None:
            tile_shape = (8, 6, 4)

        return kernel_parameters(
            Nx=8,
            Ny=6,
            Nz=4,
            x_wind=x_wind,
            y_wind=y_wind,
            z_wind=z_wind,
            dx=x_wind / 8,
            dy=y_wind / 6,
            dz=z_wind / 4,
            dt=0.08,
            shape_factor=shape_factor,
            guard_cells=2,
            tile_shape=tile_shape,
            solver=solver,
            electrostatic=solver == "electrostatic",
            boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

    def _empty_scalar(self, dynamic_parameters):
        return jnp.zeros(
            (
                int(dynamic_parameters.Nx) + 2,
                int(dynamic_parameters.Ny) + 2,
                int(dynamic_parameters.Nz) + 2,
            )
        )

    def _scalar_tiles(self, static_parameters, dynamic_parameters):
        return tile_scalar_field(
            self._empty_scalar(dynamic_parameters),
            static_parameters,
            dynamic_parameters,
            num_guard_cells=int(static_parameters.guard_cells),
        )

    def _assemble_scalar(self, field_tiles, static_parameters):
        return assemble_tiled_scalar_field(
            field_tiles,
            static_parameters,
            static_parameters.tile_shape,
            num_guard_cells=int(static_parameters.guard_cells),
        )

    def _weighted_average_particles(self):
        electrons = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=1.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([2.0]),
            v2=jnp.array([-4.0]),
            v3=jnp.array([0.5]),
        )
        ions = particle_species(
            name="ions",
            charge=1.0,
            mass=4.0,
            weight=3.0,
            x1=jnp.array([0.0]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            v1=jnp.array([10.0]),
            v2=jnp.array([4.0]),
            v3=jnp.array([-1.5]),
        )
        return [electrons, ions]

    def _spread_particles(self):
        electrons = particle_species(
            name="electrons",
            charge=-1.0,
            mass=1.0,
            weight=0.5,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15]),
        )
        ions = particle_species(
            name="ions",
            charge=2.0,
            mass=5.0,
            weight=0.25,
            x1=jnp.array([-1.25, -0.20, 0.75]),
            x2=jnp.array([1.15, -0.75, 0.45]),
            x3=jnp.array([0.35, -0.45, 0.85]),
            v1=jnp.array([-0.1, 0.2, -0.25]),
            v2=jnp.array([0.3, -0.05, 0.15]),
            v3=jnp.array([0.1, 0.05, -0.2]),
            active_mask=jnp.array([True, False, True]),
        )
        return [electrons, ions]

    def test_fluid_velocity_computes_weighted_local_average(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))

    def test_compute_velocity_field_uses_selected_direction(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = compute_velocity_field(
            tiled_particles,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            1,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 2.0, rtol=1.0e-12, atol=1.0e-12))

    def test_field_output_map_adds_requested_particle_diagnostics(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(
            particles,
            static_parameters,
            dynamic_parameters,
        )

        scalar_field = self._scalar_tiles(static_parameters, dynamic_parameters)
        vector_field = (scalar_field, scalar_field, scalar_field)
        fields = (
            vector_field,
            vector_field,
            vector_field,
            scalar_field,
            scalar_field,
            (vector_field, vector_field),
            None,
            jnp.asarray(False),
        )

        field_map = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
        )
        self.assertEqual(tuple(field_map), ("E", "B", "J"))

        expected_rho = compute_rho(
            tiled_particles,
            species_config,
            scalar_field,
            static_parameters,
            dynamic_parameters,
        )

        field_map = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
            include_fluid_velocity=True,
            include_charge_density=True,
        )
        self.assertEqual(tuple(field_map), ("E", "B", "J", "rho", "fluid_velocity"))
        self.assertTrue(jnp.allclose(field_map["rho"], expected_rho))
        self.assertTrue(jnp.any(field_map["rho"] != 0.0))

        for velocity_component, expected_velocity in zip(
            field_map["fluid_velocity"],
            (8.0, 2.0, -1.0),
        ):
            occupied = jnp.abs(velocity_component) > 0.0
            self.assertTrue(jnp.any(occupied))
            self.assertTrue(
                jnp.allclose(
                    velocity_component[occupied],
                    expected_velocity,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

    def test_electrostatic_field_output_map_defaults_to_rho_phi_E(self):
        static_parameters, dynamic_parameters = self._build_parameters(
            shape_factor=1,
            solver="electrostatic",
        )
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(
            particles,
            static_parameters,
            dynamic_parameters,
        )

        scalar_field = self._scalar_tiles(static_parameters, dynamic_parameters)
        phi = scalar_field + 3.0
        E = (scalar_field + 1.0, scalar_field + 2.0, scalar_field + 4.0)
        B = (scalar_field, scalar_field, scalar_field)
        J = (scalar_field, scalar_field, scalar_field)
        fields = (
            E,
            B,
            J,
            scalar_field,
            phi,
            (B, B),
            None,
            jnp.asarray(False),
        )

        field_map = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
        )
        expected_rho = compute_rho(
            tiled_particles,
            species_config,
            scalar_field,
            static_parameters,
            dynamic_parameters,
        )

        self.assertEqual(tuple(field_map), ("rho", "phi", "E"))
        self.assertTrue(jnp.allclose(field_map["rho"], expected_rho))
        self.assertTrue(jnp.any(field_map["rho"] != 0.0))
        self.assertIs(field_map["phi"], phi)
        self.assertIs(field_map["E"], E)

        field_map_with_charge_flag = build_field_output_map(
            fields,
            tiled_particles,
            species_config,
            static_parameters,
            dynamic_parameters,
            include_charge_density=True,
        )
        self.assertEqual(tuple(field_map_with_charge_flag), ("rho", "phi", "E"))
        self.assertTrue(jnp.allclose(field_map_with_charge_flag["rho"], expected_rho))

    def test_inactive_slots_do_not_contribute_to_fluid_velocity(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(
            particles,
            static_parameters,
            dynamic_parameters,
            capacity_factor=2.0,
        )

        inactive = ~tiled_particles.active
        x = tiled_particles.x.at[inactive, 0].set(0.0)
        x = x.at[inactive, 1].set(0.0)
        x = x.at[inactive, 2].set(0.0)
        u = tiled_particles.u.at[inactive, 0].set(1000.0)
        noisy_tiled_particles = tiled_particles._replace(x=x, u=u)

        velocity_tiles = fluid_velocity(
            noisy_tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        occupied = jnp.abs(velocity_tiles) > 0.0
        self.assertTrue(jnp.any(occupied))
        self.assertTrue(jnp.allclose(velocity_tiles[occupied], 8.0, rtol=1.0e-12, atol=1.0e-12))

    def test_empty_cells_are_zero(self):
        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=1)
        particles = self._weighted_average_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        self.assertTrue(jnp.any(velocity_tiles == 0.0))
        self.assertFalse(jnp.any(jnp.isnan(velocity_tiles)))

    def test_tile_edge_deposits_reach_adjacent_tile_for_cic_and_tsc(self):
        if len(jax.devices()) < 2:
            self.skipTest("two-tile fluid velocity test needs two logical devices")

        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                x1=jnp.array([-0.25, 0.25]),
                x2=jnp.array([0.0, 0.0]),
                x3=jnp.array([0.0, 0.0]),
                v1=jnp.array([2.0, 10.0]),
            )
        ]

        for shape_factor in (1, 2):
            with self.subTest(shape_factor=shape_factor):
                tiled_static, tiled_dynamic = self._build_parameters(
                    shape_factor=shape_factor,
                    tile_shape=(4, 6, 4),
                )

                tiled_particles, tiled_species = build_tiled_particles(
                    particles,
                    tiled_static,
                    tiled_dynamic,
                )

                tiled_velocity = fluid_velocity(
                    tiled_particles,
                    tiled_species,
                    self._scalar_tiles(tiled_static, tiled_dynamic),
                    0,
                    tiled_static,
                    tiled_dynamic,
                )

                tiled_velocity = self._assemble_scalar(tiled_velocity, tiled_static)

                x_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[0])))
                y_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[1])))
                z_index = int(jnp.argmin(jnp.abs(tiled_dynamic.grids.center[2])))
                self.assertAlmostEqual(float(tiled_velocity[x_index, y_index, z_index]), 6.0)

    def test_fluid_velocity_uses_particle_boundary_conditions(self):
        periodic_static, dynamic_parameters = self._build_parameters(shape_factor=1)
        field_conducting_static = periodic_static._replace(
            boundary_conditions=(BC_CONDUCTING, BC_PERIODIC, BC_PERIODIC),
            particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
        )

        particles = [
            particle_species(
                name="plasma",
                charge=1.0,
                mass=1.0,
                x1=jnp.array([-1.75, 1.75]),
                x2=jnp.array([0.0, 0.0]),
                x3=jnp.array([0.0, 0.0]),
                v1=jnp.array([2.0, 10.0]),
            )
        ]
        periodic_particles, periodic_species = build_tiled_particles(
            particles,
            periodic_static,
            dynamic_parameters,
        )
        conducting_field_particles, conducting_field_species = build_tiled_particles(
            particles,
            field_conducting_static,
            dynamic_parameters,
        )

        periodic_velocity = fluid_velocity(
            periodic_particles,
            periodic_species,
            self._scalar_tiles(periodic_static, dynamic_parameters),
            0,
            periodic_static,
            dynamic_parameters,
        )
        conducting_field_velocity = fluid_velocity(
            conducting_field_particles,
            conducting_field_species,
            self._scalar_tiles(field_conducting_static, dynamic_parameters),
            0,
            field_conducting_static,
            dynamic_parameters,
        )

        self.assertTrue(
            jnp.allclose(
                conducting_field_velocity,
                periodic_velocity,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

        velocity = self._assemble_scalar(conducting_field_velocity, field_conducting_static)
        x_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[0] + 2.0)))
        y_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[1])))
        z_index = int(jnp.argmin(jnp.abs(dynamic_parameters.grids.center[2])))
        self.assertAlmostEqual(float(velocity[x_index, y_index, z_index]), 6.0)

    def test_tile_major_velocity_runs_on_multi_tile_kernel_storage_when_devices_are_available(self):
        tile_shape = (4, 3, 2)
        n_tiles = (8 // tile_shape[0]) * (6 // tile_shape[1]) * (4 // tile_shape[2])
        if len(jax.devices()) < n_tiles:
            self.skipTest("multi-tile field mesh needs one logical device per tile")

        static_parameters, dynamic_parameters = self._build_parameters(shape_factor=2, tile_shape=tile_shape)
        particles = self._spread_particles()
        tiled_particles, species_config = build_tiled_particles(particles, static_parameters, dynamic_parameters)

        velocity_tiles = fluid_velocity(
            tiled_particles,
            species_config,
            self._scalar_tiles(static_parameters, dynamic_parameters),
            0,
            static_parameters,
            dynamic_parameters,
        )

        self.assertEqual(
            velocity_tiles.shape,
            (
                2,
                2,
                2,
                tile_shape[0] + 2 * int(static_parameters.guard_cells),
                tile_shape[1] + 2 * int(static_parameters.guard_cells),
                tile_shape[2] + 2 * int(static_parameters.guard_cells),
            ),
        )
        self.assertTrue(jnp.any(jnp.abs(velocity_tiles) > 0.0))
        self.assertFalse(jnp.any(jnp.isnan(velocity_tiles)))


if __name__ == "__main__":
    unittest.main()
