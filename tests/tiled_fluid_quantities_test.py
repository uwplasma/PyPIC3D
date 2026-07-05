import inspect
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D import __main__ as pypic_main
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.diagnostics.fluid_quantities import (
    compute_mass_density,
    compute_pressure_field,
    compute_velocity_field,
)
from PyPIC3D.deposition.rho import (
    compute_tiled_mass_density_from_tiled_particles,
    compute_tiled_pressure_field_from_tiled_particles,
    compute_tiled_velocity_field_from_tiled_particles,
)
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_scalar_field
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


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


class TestTiledFluidQuantities(unittest.TestCase):
    def _build_world(self, shape_factor=2):
        x_wind, y_wind, z_wind = 4.0, 3.0, 2.0
        world = {
            "dx": x_wind / 8,
            "dy": y_wind / 6,
            "dz": z_wind / 4,
            "Nx": 8,
            "Ny": 6,
            "Nz": 4,
            "x_wind": x_wind,
            "y_wind": y_wind,
            "z_wind": z_wind,
            "dt": 0.08,
            "shape_factor": shape_factor,
            "guard_cells": 2,
            "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        }
        vertex_grid, center_grid = build_yee_grid(world)
        world["grids"] = {"vertex": vertex_grid, "center": center_grid}
        return world

    def _world_with_tiled_grids(self, world, tile_shape=None):
        if tile_shape is None:
            tile_shape = (
                self._simulation_parameters()["particle_tile_nx"],
                self._simulation_parameters()["particle_tile_ny"],
                self._simulation_parameters()["particle_tile_nz"],
            )
        g = int(world["guard_cells"])
        world = dict(world)
        grids = dict(world["grids"])
        world["tile_shape"] = tile_shape
        tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, g)
        grids["tiled_vertex_grid"] = tiled_vertex_grid
        grids["tiled_center_grid"] = tiled_center_grid
        world["grids"] = grids
        return world

    def _scalar_tiles(self, world):
        return tile_scalar_field(
            self._empty_scalar(world),
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

    def _single_tile_world(self, world):
        return self._world_with_tiled_grids(
            self._build_world(shape_factor=world["shape_factor"]),
            tile_shape=(world["Nx"], world["Ny"], world["Nz"]),
        )

    def _assembled_mass_density(self, particles, world):
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        rho_tiles = compute_tiled_mass_density_from_tiled_particles(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            world,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        return assemble_tiled_scalar_field(
            rho_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

    def _assembled_velocity_field(self, particles, world, direction):
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        field_tiles = compute_tiled_velocity_field_from_tiled_particles(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            int(direction),
            world,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        return assemble_tiled_scalar_field(
            field_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

    def _assembled_pressure_field(self, particles, world, direction):
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        velocity_tiles = compute_tiled_velocity_field_from_tiled_particles(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            int(direction),
            world,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        pressure_tiles = compute_tiled_pressure_field_from_tiled_particles(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            velocity_tiles,
            int(direction),
            world,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        return assemble_tiled_scalar_field(
            pressure_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

    def _empty_scalar(self, world):
        return jnp.zeros((world["Nx"] + 2, world["Ny"] + 2, world["Nz"] + 2))

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 2,
            "particle_tile_nz": 2,
        }

    def _particles(self, world):
        electrons = particle_species(
            name="electrons",
            N_particles=5,
            charge=-1.0,
            mass=2.0,
            weight=0.5,
            T=1.0,
            x1=jnp.array([-1.75, -0.65, 0.15, 1.75, world["x_wind"] / 2 - 0.03]),
            x2=jnp.array([-1.15, -0.45, 0.35, 1.05, 0.0]),
            x3=jnp.array([-0.75, -0.20, 0.25, 0.80, 0.0]),
            v1=jnp.array([0.2, -0.1, 0.05, 0.3, -1.4]),
            v2=jnp.array([0.0, 0.15, -0.2, 0.1, 0.0]),
            v3=jnp.array([-0.05, 0.25, 0.1, -0.15, 0.0]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )
        ions = particle_species(
            name="ions",
            N_particles=3,
            charge=2.0,
            mass=5.0,
            weight=0.25,
            T=1.0,
            x1=jnp.array([-1.25, -0.20, 0.75]),
            x2=jnp.array([1.15, -0.75, 0.45]),
            x3=jnp.array([0.35, -0.45, 0.85]),
            v1=jnp.array([-0.1, 0.2, -0.25]),
            v2=jnp.array([0.3, -0.05, 0.15]),
            v3=jnp.array([0.1, 0.05, -0.2]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
            active_mask=jnp.array([True, False, True]),
        )
        return [electrons, ions]

    def test_tiled_particles_require_tile_major_mass_storage(self):
        world = self._build_world(shape_factor=2)
        particles = self._particles(world)
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())

        with self.assertRaisesRegex(ValueError, "tile-major scalar field storage"):
            compute_mass_density(tiled_particles, self._empty_scalar(world), world, species_config=species_config)

    def test_tile_major_mass_density_assembles_to_global_mass_density(self):
        world = self._build_world(shape_factor=2)
        world = self._world_with_tiled_grids(world)
        particles = self._particles(world)
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        reference_world = self._single_tile_world(world)
        mass_reference = self._assembled_mass_density(self._particles(reference_world), reference_world)
        mass_tiles = compute_tiled_mass_density_from_tiled_particles(
            tiled_particles,
            species_config,
            self._scalar_tiles(world),
            world,
            tile_shape=world["tile_shape"],
            g=int(world["guard_cells"]),
        )
        mass_from_tiles = assemble_tiled_scalar_field(
            mass_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

        self.assertTrue(
            jnp.allclose(
                mass_from_tiles[1:-1, 1:-1, 1:-1],
                mass_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_compute_mass_density_prefers_tile_major_output_for_tiled_scalar_field(self):
        world = self._build_world(shape_factor=2)
        world = self._world_with_tiled_grids(world)
        particles = self._particles(world)
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        reference_world = self._single_tile_world(world)
        mass_reference = self._assembled_mass_density(self._particles(reference_world), reference_world)
        rho_tiles = self._scalar_tiles(world)
        mass_tiles = compute_mass_density(tiled_particles, rho_tiles, world, species_config=species_config)
        mass_from_tiles = assemble_tiled_scalar_field(
            mass_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

        self.assertEqual(mass_tiles.ndim, 6)
        self.assertTrue(
            jnp.allclose(
                mass_from_tiles[1:-1, 1:-1, 1:-1],
                mass_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tile_major_velocity_field_assembles_to_global_velocity_field(self):
        world = self._build_world(shape_factor=2)
        world = self._world_with_tiled_grids(world)
        particles = self._particles(world)
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        reference_world = self._single_tile_world(world)
        velocity_reference = self._assembled_velocity_field(self._particles(reference_world), reference_world, 0)
        velocity_tiles = compute_velocity_field(
            tiled_particles,
            self._scalar_tiles(world),
            0,
            world,
            species_config=species_config,
        )
        velocity_from_tiles = assemble_tiled_scalar_field(
            velocity_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

        self.assertTrue(
            jnp.allclose(
                velocity_from_tiles[1:-1, 1:-1, 1:-1],
                velocity_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_tile_major_pressure_field_assembles_to_global_pressure_field(self):
        world = self._build_world(shape_factor=2)
        world = self._world_with_tiled_grids(world)
        particles = self._particles(world)
        tiled_particles, species_config = to_tiled_particles(particles, world, self._simulation_parameters())
        reference_world = self._single_tile_world(world)
        pressure_reference = self._assembled_pressure_field(self._particles(reference_world), reference_world, 0)
        field_tiles = self._scalar_tiles(world)
        velocity_tiles = compute_velocity_field(
            tiled_particles,
            field_tiles,
            0,
            world,
            species_config=species_config,
        )
        pressure_tiles = compute_pressure_field(
            tiled_particles,
            field_tiles,
            velocity_tiles,
            0,
            world,
            species_config=species_config,
        )
        pressure_from_tiles = assemble_tiled_scalar_field(
            pressure_tiles,
            world,
            world["tile_shape"],
            num_guard_cells=int(world["guard_cells"]),
        )

        self.assertTrue(
            jnp.allclose(
                pressure_from_tiles[1:-1, 1:-1, 1:-1],
                pressure_reference[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_public_fluid_quantities_reject_flat_particles(self):
        world = self._build_world(shape_factor=2)
        particles = self._particles(world)
        field = self._empty_scalar(world)

        with self.assertRaisesRegex(ValueError, "TiledParticles"):
            compute_mass_density(particles, field, world)
        with self.assertRaisesRegex(ValueError, "TiledParticles"):
            compute_velocity_field(particles, field, 0, world)
        with self.assertRaisesRegex(ValueError, "TiledParticles"):
            compute_pressure_field(particles, field, field, 0, world)

    def test_tiled_scalar_vtk_path_no_longer_rejects_electrodynamic_yee(self):
        run_source = inspect.getsource(pypic_main.run_PyPIC3D)

        self.assertNotIn("plot_vtk_scalars is not supported for electrodynamic_yee", run_source)


if __name__ == "__main__":
    unittest.main()
