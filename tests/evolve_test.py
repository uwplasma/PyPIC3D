import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.evolve import (
    _time_loop_electrostatic_global_reference,
    time_loop_electrodynamic,
    time_loop_electrostatic,
)
from PyPIC3D.deposition.rho import _compute_rho_flat
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.solvers.yee_tiled import tile_vector_field
from PyPIC3D.utils import build_collocated_grid, build_yee_grid

jax.config.update("jax_enable_x64", True)


def zero_current(particles, species_config, J, constants, world, tile_shape=None, g=None):
    return tuple(jnp.zeros_like(comp) for comp in J)


def unused_curl(Ex, Ey, Ez):
    return None


class TestEvolveExternalFields(unittest.TestCase):
    def test_public_loop_names_own_tiled_contracts(self):
        self.assertEqual(time_loop_electrodynamic.__module__, "PyPIC3D.evolve")
        self.assertEqual(time_loop_electrostatic.__module__, "PyPIC3D.evolve")

    def _electrostatic_grid_order_species(self, world, charge):
        return particle_species(
            name=f"charge {charge}",
            N_particles=1,
            charge=charge,
            mass=1.0,
            T=0.0,
            weight=1.0,
            v1=jnp.array([0.0]),
            v2=jnp.array([0.0]),
            v3=jnp.array([0.0]),
            x1=jnp.array([0.07]),
            x2=jnp.array([0.0]),
            x3=jnp.array([0.0]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            dt=world["dt"],
        )

    def test_electrostatic_push_uses_same_grid_order_as_electrodynamic(self):
        world = {
            "Nx": 4,
            "Ny": 1,
            "Nz": 1,
            "dx": 0.25,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        Ex = jnp.broadcast_to(vertex_grid[0][:, None, None], E[0].shape)
        E = (Ex, E[1], E[2])
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )

        pushed_particles = [
            self._electrostatic_grid_order_species(world, 1.0),
            self._electrostatic_grid_order_species(world, -1.0),
        ]
        reference_particles = [
            self._electrostatic_grid_order_species(world, 1.0),
            self._electrostatic_grid_order_species(world, -1.0),
        ]

        pushed_particles, _ = _time_loop_electrostatic_global_reference(
            pushed_particles,
            (E, B, J, rho, phi, external_fields),
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )

        for i in range(len(reference_particles)):
            reference_particles[i] = particle_push(
                reference_particles[i],
                E,
                B,
                center_grid,
                vertex_grid,
                world,
                constants,
                relativistic=False,
                particle_pusher="boris",
            )
            reference_particles[i].update_position()
            reference_particles[i].boundary_conditions(world)

        for pushed, reference in zip(pushed_particles, reference_particles):
            pushed_vx, pushed_vy, pushed_vz = pushed.get_velocity()
            reference_vx, reference_vy, reference_vz = reference.get_velocity()
            self.assertTrue(jnp.allclose(pushed_vx, reference_vx, rtol=1.0e-12, atol=1.0e-12))
            self.assertTrue(jnp.allclose(pushed_vy, reference_vy, rtol=1.0e-12, atol=1.0e-12))
            self.assertTrue(jnp.allclose(pushed_vz, reference_vz, rtol=1.0e-12, atol=1.0e-12))

    def test_electrostatic_rho_uses_boundary_applied_particle_positions(self):
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
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_collocated_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        particles = [
            particle_species(
                name="right_crossing_charge",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=0.0,
                weight=1.0,
                v1=jnp.array([0.10]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([1.99995]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            ),
            particle_species(
                name="neutralizing_charge",
                N_particles=1,
                charge=-1.0,
                mass=1.0,
                T=0.0,
                weight=1.0,
                v1=jnp.array([0.0]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([0.0]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            ),
        ]

        particles, fields = _time_loop_electrostatic_global_reference(
            particles,
            (E, B, J, rho, phi, external_fields),
            world,
            constants,
            unused_curl,
            J_func=None,
            solver="fdtd",
            relativistic=False,
            particle_pusher="boris",
        )
        _, _, _, rho_after, _, _ = fields
        recomputed_rho = _compute_rho_flat(particles, jnp.zeros_like(rho_after), world, constants)

        self.assertTrue(
            jnp.allclose(
                rho_after[1:-1, 1:-1, 1:-1],
                recomputed_rho[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_external_electric_field_pushes_particles_without_evolving_maxwell_fields(self):
        world = {
            "Nx": 3,
            "Ny": 3,
            "Nz": 3,
            "dx": 1.0 / 3.0,
            "dy": 1.0 / 3.0,
            "dz": 1.0 / 3.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        external_E = (jnp.ones_like(external_E[0]), external_E[1], external_E[2])
        external_fields = (external_E, external_B)

        particles = [
            particle_species(
                name="test",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=0.0,
                v1=jnp.array([0.0]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([0.0]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                dt=world["dt"],
            )
        ]
        tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        world["guard_cells"] = 2
        tiled_particles, species_config = to_tiled_particles(particles, world, simulation_parameters)
        external_E_tiles = tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=int(world["guard_cells"]))
        external_B_tiles = tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=int(world["guard_cells"]))
        fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(B, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(J, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            rho,
            phi,
            (external_E_tiles, external_B_tiles),
            None,
        )

        tiled_particles, fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            world,
            constants,
            unused_curl,
            zero_current,
            solver="electrodynamic_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        E_after, B_after, J_after, rho_after, phi_after, external_after, pml_state, overflow = fields
        vx = tiled_particles.u[:, :, :, 0, :, 0][tiled_particles.active[:, :, :, 0, :]]
        vy = tiled_particles.u[:, :, :, 0, :, 1][tiled_particles.active[:, :, :, 0, :]]
        vz = tiled_particles.u[:, :, :, 0, :, 2][tiled_particles.active[:, :, :, 0, :]]

        self.assertIsNone(pml_state)
        self.assertFalse(bool(overflow))
        self.assertGreater(float(vx[0]), 0.0)
        self.assertTrue(jnp.allclose(vy, 0.0))
        self.assertTrue(jnp.allclose(vz, 0.0))
        self.assertTrue(jnp.allclose(E_after[0], 0.0))
        self.assertTrue(jnp.allclose(B_after[0], 0.0))
        self.assertTrue(jnp.allclose(external_after[0][0], external_E_tiles[0]))

    def test_absorbing_particle_mask_survives_jitted_electrodynamic_step(self):
        world = {
            "Nx": 3,
            "Ny": 1,
            "Nz": 1,
            "dx": 1.0 / 3.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.1,
            "x_wind": 1.0,
            "y_wind": 1.0,
            "z_wind": 1.0,
            "shape_factor": 1,
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 2, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(world)
        world["grids"] = {"center": center_grid, "vertex": vertex_grid}
        constants = {"C": 10.0, "eps": 1.0, "mu": 1.0, "alpha": 0.0}

        E, B, J, phi, rho = initialize_fields(world["Nx"], world["Ny"], world["Nz"])
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        particles = [
            particle_species(
                name="absorbing",
                N_particles=1,
                charge=1.0,
                mass=1.0,
                T=0.0,
                weight=1.0,
                v1=jnp.array([0.2]),
                v2=jnp.array([0.0]),
                v3=jnp.array([0.0]),
                x1=jnp.array([0.49]),
                x2=jnp.array([0.0]),
                x3=jnp.array([0.0]),
                xwind=world["x_wind"],
                ywind=world["y_wind"],
                zwind=world["z_wind"],
                dx=world["dx"],
                dy=world["dy"],
                dz=world["dz"],
                x_bc="periodic",
                dt=world["dt"],
            )
        ]
        tile_shape = (world["Nx"], world["Ny"], world["Nz"])
        simulation_parameters = {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }
        world["guard_cells"] = 2
        tiled_particles, species_config = to_tiled_particles(particles, world, simulation_parameters)
        fields = (
            tile_vector_field(E, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(B, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            tile_vector_field(J, world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            rho,
            phi,
            (
                tile_vector_field(external_fields[0], world, tile_shape, num_guard_cells=int(world["guard_cells"])),
                tile_vector_field(external_fields[1], world, tile_shape, num_guard_cells=int(world["guard_cells"])),
            ),
            None,
        )

        tiled_particles, _ = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            world,
            constants,
            unused_curl,
            zero_current,
            solver="electrodynamic_yee",
            tile_shape=tile_shape,
            g=int(world["guard_cells"]),
            relativistic=False,
            particle_pusher="boris",
        )

        active = tiled_particles.active[:, :, :, 0, :].reshape(-1)
        x = tiled_particles.x[:, :, :, 0, :, 0].reshape(-1)

        self.assertEqual(x.shape[0], 1)
        self.assertTrue(jnp.array_equal(active, jnp.array([False])))


if __name__ == "__main__":
    unittest.main()
