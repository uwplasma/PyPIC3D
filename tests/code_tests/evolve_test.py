import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh
from PyPIC3D.evolve import (
    time_loop_electrodynamic,
    time_loop_electrostatic,
)
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.parameters import build_dynamic_parameters, build_static_parameters
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from tests.kernel_fixtures import kernel_parameters_from_values

jax.config.update("jax_enable_x64", True)


def add_tiled_grids_to_parameters(parameter_set, tile_shape):
    g = int(parameter_set["guard_cells"])
    parameter_set["tile_shape"] = tile_shape
    tile_grid_shape = (
        int(parameter_set["Nx"]) // int(tile_shape[0]),
        int(parameter_set["Ny"]) // int(tile_shape[1]),
        int(parameter_set["Nz"]) // int(tile_shape[2]),
    )
    parameter_set["field_mesh"] = make_field_mesh(tile_grid_shape)
    static_parameters = SimpleNamespace(tile_shape=tile_shape, guard_cells=g)
    dynamic_parameters = SimpleNamespace(
        dx=parameter_set["dx"],
        dy=parameter_set["dy"],
        dz=parameter_set["dz"],
        grids=SimpleNamespace(vertex=parameter_set["grids"]["vertex"], center=parameter_set["grids"]["center"]),
    )
    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(static_parameters, dynamic_parameters)
    parameter_set["grids"]["tiled_center_grid"] = tiled_center_grid
    parameter_set["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
    # make tiled versions of the grids for the parallelized particle pushers to use


def one_slot_tiled_particles(
    x,
    u,
    charge=1.0,
    mass=1.0,
    weight=1.0,
    active=True,
    update_x=(True, True, True),
    update_u=(True, True, True),
):
    particles = TiledParticles(
        x=jnp.asarray(x, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        u=jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.asarray([[[[[active]]]]], dtype=bool),
    )
    # create a single tile of particles with one particle in it, and a single species config
    species_config = SpeciesConfig(
        charge=jnp.asarray([charge], dtype=float),
        mass=jnp.asarray([mass], dtype=float),
        weight=jnp.asarray([weight], dtype=float),
        update_x=jnp.asarray([update_x], dtype=bool),
        update_u=jnp.asarray([update_u], dtype=bool),
    )
    return particles, species_config


class TestEvolveExternalFields(unittest.TestCase):
    def test_public_loop_names_own_tiled_contracts(self):
        self.assertEqual(time_loop_electrodynamic.__module__, "PyPIC3D.evolve")
        self.assertEqual(time_loop_electrostatic.__module__, "PyPIC3D.evolve")
        # test that both these methods exist in the public API and are not private methods

    def test_external_electric_field_pushes_particles_without_evolving_maxwell_fields(self):
        parameter_set = {
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
            "current_deposition": "direct",
            "current_filter": "none",
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        # build parameter_set and grids, and set dynamic_values for the simulation

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_tiled_grids_to_parameters(parameter_set, tile_shape)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_E = tuple(jnp.zeros_like(comp) for comp in E)
        external_B = tuple(jnp.zeros_like(comp) for comp in B)
        external_E = (jnp.ones_like(external_E[0]), external_E[1], external_E[2])
        external_fields = (external_E, external_B)
        # build empty fields and set an external electric field in the x-direction

        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.0, 0.0, 0.0]),
            u=jnp.array([0.0, 0.0, 0.0]),
        )
        # create a single particle at the origin with zero velocity
        fields = (
            E,
            B,
            J,
            rho,
            phi,
            external_fields,
            None,
            jnp.asarray(False),
        )

        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        tiled_particles, fields = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        ) # advance the simulation for one time step with the external electric field

        E_after, B_after, J_after, rho_after, phi_after, external_after, pml_state, overflow = fields
        vx = tiled_particles.u[:, :, :, 0, :, 0][tiled_particles.active[:, :, :, 0, :]]
        vy = tiled_particles.u[:, :, :, 0, :, 1][tiled_particles.active[:, :, :, 0, :]]
        vz = tiled_particles.u[:, :, :, 0, :, 2][tiled_particles.active[:, :, :, 0, :]]
        # unpack the fields and particle velocities after the time step

        self.assertIsNone(pml_state)
        # ensure the PML state is None, as we are not using PML in this test
        self.assertFalse(bool(overflow))
        # ensure the particle did not overflow the simulation domain
        self.assertGreater(float(vx[0]), 0.0)
        self.assertTrue(jnp.allclose(vy, 0.0))
        self.assertTrue(jnp.allclose(vz, 0.0))
        # ensure the particle has gained velocity in the x-direction due to the external electric field, and has not gained velocity in the y or z directions
        self.assertTrue(jnp.allclose(external_after[0][0], external_fields[0][0]))
        # ensure the external electric field has not changed after the time step

    def test_electrodynamic_step_accepts_split_kernel_parameters(self):
        parameter_set = {
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
            "current_deposition": "direct",
            "current_filter": "none",
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_tiled_grids_to_parameters(parameter_set, tile_shape)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        fields = (E, B, J, rho, phi, external_fields, None, jnp.asarray(False))
        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.0, 0.0, 0.0]),
            u=jnp.array([0.05, 0.0, 0.0]),
        )
        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        step = jax.jit(lambda particles, fields, dynamic: time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic,
        ))
        tiled_particles, fields = step(tiled_particles, fields, dynamic_parameters)

        self.assertFalse(bool(fields[-1]))
        self.assertTrue(jnp.all(tiled_particles.active))
        self.assertGreater(float(tiled_particles.x[0, 0, 0, 0, 0, 0]), 0.0)

    def test_absorbing_particle_mask_survives_jitted_electrodynamic_step(self):
        parameter_set = {
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
            "current_deposition": "direct",
            "current_filter": "none",
            "boundary_conditions": {"x": 0, "y": 0, "z": 0},
            "particle_boundary_conditions": {"x": 2, "y": 0, "z": 0},
        }
        center_grid, vertex_grid = build_yee_grid(SimpleNamespace(**parameter_set))
        parameter_set["grids"] = {"center": center_grid, "vertex": vertex_grid}
        dynamic_values = {"C": 1.0, "eps": 1.0, "mu": 1.0, "alpha": 1.0}
        # build parameter_set and grids, and set dynamic_values for the simulation

        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        parameter_set["guard_cells"] = 2
        add_tiled_grids_to_parameters(parameter_set, tile_shape)
        field_static, field_dynamic = kernel_parameters_from_values(parameter_set, dynamic_values)
        E, B, J, phi, rho = initialize_fields(field_static, field_dynamic)
        external_fields = (
            tuple(jnp.zeros_like(comp) for comp in E),
            tuple(jnp.zeros_like(comp) for comp in B),
        )
        # build empty fields and set external fields to zero

        tiled_particles, species_config = one_slot_tiled_particles(
            x=jnp.array([0.49, 0.0, 0.0]),
            u=jnp.array([0.2, 0.0, 0.0]),
        )
        # create a single particle at x=0.49 with a velocity in the x-direction, which will cause it to cross the absorbing boundary at x=0.5

        fields = (
            E,
            B,
            J,
            rho,
            phi,
            external_fields,
            None,
            jnp.asarray(False),
        )
        # build the fields tuple with the initialized fields and external fields

        static_parameters = build_static_parameters({
            **parameter_set,
            "solver": "electrodynamic_yee",
            "electrostatic": False,
            "relativistic": False,
            "particle_pusher": "boris",
        })
        dynamic_parameters = build_dynamic_parameters(parameter_set, dynamic_values)

        tiled_particles, _ = time_loop_electrodynamic(
            tiled_particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        # advance the simulation for one time step, which should cause the particle to cross the absorbing boundary and become inactive

        active = tiled_particles.active[:, :, :, 0, :].reshape(-1)
        x = tiled_particles.x[:, :, :, 0, :, 0].reshape(-1)
        # reshape the active mask and x positions of the particles to check their status after the time step

        self.assertEqual(x.shape[0], 1)
        self.assertTrue(jnp.array_equal(active, jnp.array([False])))
        # ensure that the particle has become inactive after crossing the absorbing boundary


if __name__ == "__main__":
    unittest.main()
