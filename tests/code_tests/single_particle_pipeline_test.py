import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.ghost_cells import (
    fold_tiled_ghost_cells,
    fold_tiled_vector_ghost_cells,
    update_tiled_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.J_from_rhov import J_from_rhov
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.diagnostics.output_adapters import assemble_tiled_scalar_field, assemble_tiled_vector_field
from PyPIC3D.evolve import time_loop_electrodynamic
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles, update_tiled_particle_positions
from PyPIC3D.utilities.filters import digital_filter, digital_filter_vector
from tests.kernel_fixtures import build_tiled_particles, empty_tiled_scalar, empty_tiled_vector, kernel_parameters, particle_species


jax.config.update("jax_enable_x64", True)


def _runtime_parameters(
    *,
    shape_factor=1,
    tile_shape=(4, 1, 1),
    guard_cells=2,
    current_deposition="direct",
    current_filter="none",
    boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
    particle_boundary_conditions=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
    alpha=1.0,
    dt=0.05,
):
    return kernel_parameters(
        Nx=8,
        Ny=1,
        Nz=1,
        x_wind=4.0,
        y_wind=1.0,
        z_wind=1.0,
        tile_shape=tile_shape,
        guard_cells=guard_cells,
        shape_factor=shape_factor,
        current_deposition=current_deposition,
        current_filter=current_filter,
        electrostatic=False,
        solver="electrodynamic_yee",
        boundary_conditions=boundary_conditions,
        particle_boundary_conditions=particle_boundary_conditions,
        relativistic=False,
        C=1.0,
        eps=1.0,
        mu=1.0,
        alpha=alpha,
        dt=dt,
    )


def _one_particle(static_parameters, dynamic_parameters, x, u, charge=-1.0, mass=1.0, weight=0.5):
    species = [
        particle_species(
            name="single",
            charge=charge,
            mass=mass,
            weight=weight,
            x1=jnp.asarray([x[0]], dtype=float),
            x2=jnp.asarray([x[1]], dtype=float),
            x3=jnp.asarray([x[2]], dtype=float),
            u1=jnp.asarray([u[0]], dtype=float),
            u2=jnp.asarray([u[1]], dtype=float),
            u3=jnp.asarray([u[2]], dtype=float),
        )
    ]
    return build_tiled_particles(species, static_parameters, dynamic_parameters)


def _active_particle_slot(particles):
    active_indices = np.argwhere(np.asarray(jax.device_get(particles.active)))
    if active_indices.shape[0] != 1:
        raise AssertionError(f"expected one active particle, found {active_indices.shape[0]}")
    return tuple(int(value) for value in active_indices[0])


def _particle_state(particles):
    tx, ty, tz, species, slot = _active_particle_slot(particles)
    x = particles.x[tx, ty, tz, species, slot]
    u = particles.u[tx, ty, tz, species, slot]
    return (tx, ty, tz), x, u


def _weights(delta_x, delta_y, delta_z, dynamic_parameters, shape_factor):
    if shape_factor == 1:
        weights = get_first_order_weights(
            delta_x,
            delta_y,
            delta_z,
            dynamic_parameters.dx,
            dynamic_parameters.dy,
            dynamic_parameters.dz,
        )
    else:
        weights = get_second_order_weights(
            delta_x,
            delta_y,
            delta_z,
            dynamic_parameters.dx,
            dynamic_parameters.dy,
            dynamic_parameters.dz,
        )
    return tuple(jnp.asarray(axis_weights) for axis_weights in weights)


def _collapse(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


def _node_stencils(tile, x, static_parameters, dynamic_parameters):
    tx, ty, tz = tile
    g = int(static_parameters.guard_cells)
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    ntx, nty, ntz = static_parameters.field_mesh.devices.shape
    local_shape = (tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g)
    reduced = (
        tile_nx == 1 and int(ntx) == 1,
        tile_ny == 1 and int(nty) == 1,
        tile_nz == 1 and int(ntz) == 1,
    )
    x_grid = dynamic_parameters.grids.tiled_center_grid[0][tx, ty, tz]
    y_grid = dynamic_parameters.grids.tiled_center_grid[1][tx, ty, tz]
    z_grid = dynamic_parameters.grids.tiled_center_grid[2][tx, ty, tz]

    x_pos = jnp.asarray([x[0]])
    y_pos = jnp.asarray([x[1]])
    z_pos = jnp.asarray([x[2]])
    _, x0, deltax_node, xpts = prepare_particle_axis_stencil(
        x_pos,
        x_grid,
        local_shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_nx * dynamic_parameters.dx,
        ghost_cells=True,
    )
    _, _y0, deltay_node, ypts = prepare_particle_axis_stencil(
        y_pos,
        y_grid,
        local_shape[1],
        static_parameters.shape_factor,
        2,
        wind=tile_ny * dynamic_parameters.dy,
        ghost_cells=True,
    )
    _, _z0, deltaz_node, zpts = prepare_particle_axis_stencil(
        z_pos,
        z_grid,
        local_shape[2],
        static_parameters.shape_factor,
        2,
        wind=tile_nz * dynamic_parameters.dz,
        ghost_cells=True,
    )

    node_weights = _weights(deltax_node, deltay_node, deltaz_node, dynamic_parameters, static_parameters.shape_factor)
    xpts, wx_node = _collapse(jnp.asarray(xpts), node_weights[0], local_shape[0], reduced[0], g)
    ypts, wy_node = _collapse(jnp.asarray(ypts), node_weights[1], local_shape[1], reduced[1], g)
    zpts, wz_node = _collapse(jnp.asarray(zpts), node_weights[2], local_shape[2], reduced[2], g)

    deltax_face = (x_pos - x_grid[0]) - (x0 + 0.5) * dynamic_parameters.dx
    deltay_face = (y_pos - y_grid[0]) - (_y0 + 0.5) * dynamic_parameters.dy
    deltaz_face = (z_pos - z_grid[0]) - (_z0 + 0.5) * dynamic_parameters.dz
    x_face_weights, _, _ = _weights(
        deltax_face,
        deltay_node,
        deltaz_node,
        dynamic_parameters,
        static_parameters.shape_factor,
    )
    _, y_face_weights, _ = _weights(
        deltax_node,
        deltay_face,
        deltaz_node,
        dynamic_parameters,
        static_parameters.shape_factor,
    )
    _, _, z_face_weights = _weights(
        deltax_node,
        deltay_node,
        deltaz_face,
        dynamic_parameters,
        static_parameters.shape_factor,
    )
    _, wx_face = _collapse(xpts, x_face_weights, local_shape[0], reduced[0], g)
    _, wy_face = _collapse(ypts, y_face_weights, local_shape[1], reduced[1], g)
    _, wz_face = _collapse(zpts, z_face_weights, local_shape[2], reduced[2], g)

    return (xpts, ypts, zpts), (wx_node, wy_node, wz_node), (wx_face, wy_face, wz_face)


def _add_stencil(field, tile, points, weights, scale):
    tx, ty, tz = tile
    xpts, ypts, zpts = points
    wx, wy, wz = weights
    for i in range(xpts.shape[0]):
        for j in range(ypts.shape[0]):
            for k in range(zpts.shape[0]):
                ix = int(xpts[i, 0])
                iy = int(ypts[j, 0])
                iz = int(zpts[k, 0])
                value = scale * wx[i, 0] * wy[j, 0] * wz[k, 0]
                field = field.at[tx, ty, tz, ix, iy, iz].add(value, mode="drop")
    return field


def _manual_rho_tiles(particles, species_config, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    tile, x, _u = _particle_state(particles)
    points, node_weights, _face_weights = _node_stencils(tile, x, static_parameters, dynamic_parameters)
    rho = empty_tiled_scalar(static_parameters, dynamic_parameters)
    charge_density = species_config.charge[0] * species_config.weight[0] / (
        dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    )

    rho = _add_stencil(rho, tile, points, node_weights, charge_density)
    rho = fold_tiled_ghost_cells(rho, static_parameters, g, bc_type=1)
    rho = update_tiled_ghost_cells(rho, static_parameters, g, bc_type=1)

    if static_parameters.current_filter == "digital":
        rho = digital_filter(rho, dynamic_parameters.alpha, num_guard_cells=g)
        rho = update_tiled_ghost_cells(rho, static_parameters, g, bc_type=1)

    return rho


def _manual_direct_current_tiles(particles, species_config, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    tile, x, u = _particle_state(particles)
    points, node_weights, face_weights = _node_stencils(tile, x, static_parameters, dynamic_parameters)
    Jx, Jy, Jz = empty_tiled_vector(static_parameters, dynamic_parameters)
    charge_density = species_config.charge[0] * species_config.weight[0] / (
        dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    )

    Jx = _add_stencil(Jx, tile, points, (face_weights[0], node_weights[1], node_weights[2]), charge_density * u[0])
    Jy = _add_stencil(Jy, tile, points, (node_weights[0], face_weights[1], node_weights[2]), charge_density * u[1])
    Jz = _add_stencil(Jz, tile, points, (node_weights[0], node_weights[1], face_weights[2]), charge_density * u[2])

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)
    J = update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)

    if static_parameters.current_filter == "digital":
        J = digital_filter_vector(J, dynamic_parameters.alpha, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)

    return J


def _shift_old_stencil(weights, shift):
    old_weights = jnp.stack(weights, axis=0)
    return [jnp.roll(old_weights[:, 0], -int(shift))[i, jnp.newaxis] for i in range(5)]


def _manual_esirkepov_current_tiles_1d(particles, species_config, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    tile, old_x, u = _particle_state(particles)
    tx, ty, tz = tile
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    local_Nx = tile_nx + 2 * g
    x_grid = dynamic_parameters.grids.tiled_center_grid[0][tx, ty, tz]

    old_position = jnp.asarray([old_x[0]])
    new_position = jnp.asarray([old_x[0] + u[0] * dynamic_parameters.dt])
    x0 = jnp.round((new_position - x_grid[0]) / dynamic_parameters.dx).astype(int) if static_parameters.shape_factor == 2 else jnp.floor((new_position - x_grid[0]) / dynamic_parameters.dx).astype(int)
    old_x0 = jnp.round((old_position - x_grid[0]) / dynamic_parameters.dx).astype(int) if static_parameters.shape_factor == 2 else jnp.floor((old_position - x_grid[0]) / dynamic_parameters.dx).astype(int)
    deltax = new_position - (x0 * dynamic_parameters.dx + x_grid[0])
    old_deltax = old_position - (old_x0 * dynamic_parameters.dx + x_grid[0])
    zero_delta = jnp.asarray([0.0])
    xw, _, _ = _weights(deltax, zero_delta, zero_delta, dynamic_parameters, static_parameters.shape_factor)
    oxw, _, _ = _weights(old_deltax, zero_delta, zero_delta, dynamic_parameters, static_parameters.shape_factor)
    zero = jnp.zeros_like(xw[0])
    xw = [zero, xw[0], xw[1], xw[2], zero]
    oxw = [zero, oxw[0], oxw[1], oxw[2], zero]
    oxw = _shift_old_stencil(oxw, int(x0[0] - old_x0[0]))

    offsets = jnp.asarray([-2, -1, 0, 1, 2], dtype=x0.dtype)
    xpts = x0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
    ypt = g
    zpt = g

    Jx, Jy, Jz = empty_tiled_vector(static_parameters, dynamic_parameters)
    q = species_config.charge[0] * species_config.weight[0]
    dJx = -(q / (dynamic_parameters.dy * dynamic_parameters.dz)) / dynamic_parameters.dt
    dJy = q * u[1] / (dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz)
    dJz = q * u[2] / (dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz)
    Fx = jnp.asarray([dJx * (xw[i][0] - oxw[i][0]) for i in range(5)])
    Jy_weights = jnp.asarray([0.5 * (xw[i][0] + oxw[i][0]) for i in range(5)])
    Jx_loc = jnp.cumsum(Fx)

    for i in range(5):
        ix = int(xpts[i, 0])
        Jx = Jx.at[tx, ty, tz, ix, ypt, zpt].add(Jx_loc[i], mode="drop")
        Jy = Jy.at[tx, ty, tz, ix, ypt, zpt].add(dJy * Jy_weights[i], mode="drop")
        Jz = Jz.at[tx, ty, tz, ix, ypt, zpt].add(dJz * Jy_weights[i], mode="drop")

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)
    return update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)


def _assemble_scalar(field, static_parameters):
    return assemble_tiled_scalar_field(
        field,
        static_parameters,
        static_parameters.tile_shape,
        num_guard_cells=int(static_parameters.guard_cells),
    )


def _assemble_vector(field, static_parameters):
    return assemble_tiled_vector_field(
        field,
        static_parameters,
        static_parameters.tile_shape,
        num_guard_cells=int(static_parameters.guard_cells),
    )


def _assert_vector_close(test_case, actual, expected, rtol=1.0e-12, atol=1.0e-12):
    for actual_component, expected_component in zip(actual, expected):
        error = float(jnp.max(jnp.abs(actual_component - expected_component)))
        test_case.assertTrue(
            jnp.allclose(actual_component, expected_component, rtol=rtol, atol=atol),
            f"max component error {error}",
        )


def _empty_electrodynamic_fields(static_parameters, dynamic_parameters):
    E, B, J, phi, rho = initialize_fields(static_parameters, dynamic_parameters)
    external_fields = (
        tuple(jnp.zeros_like(component) for component in E),
        tuple(jnp.zeros_like(component) for component in B),
    )
    return E, B, J, rho, phi, external_fields, None, jnp.asarray(False)


def _expected_B_from_Ey(Ey, dynamic_parameters):
    expected = jnp.zeros_like(Ey)
    active = (slice(1, -1), slice(1, -1), slice(1, -1))
    backward_x = (slice(0, -2), slice(1, -1), slice(1, -1))
    dEy_dx = (Ey[active] - Ey[backward_x]) / dynamic_parameters.dx
    return expected.at[active].set(-dynamic_parameters.dt * dEy_dx)


class TestSingleParticleStencils(unittest.TestCase):
    def test_compute_rho_filter_selector_uses_static_current_filter(self):
        x = (1.97, 0.0, 0.0)
        u = (0.0, 0.2, 0.0)
        static_raw_06, dynamic_raw_06 = _runtime_parameters(shape_factor=2, current_filter="none", alpha=0.6)
        static_raw_10, dynamic_raw_10 = _runtime_parameters(shape_factor=2, current_filter="none", alpha=1.0)
        static_digital, dynamic_digital = _runtime_parameters(shape_factor=2, current_filter="digital", alpha=0.6)

        particles_raw_06, species_config = _one_particle(static_raw_06, dynamic_raw_06, x, u)
        particles_raw_10, _ = _one_particle(static_raw_10, dynamic_raw_10, x, u)
        particles_digital, _ = _one_particle(static_digital, dynamic_digital, x, u)

        rho_raw_06 = compute_rho(
            particles_raw_06,
            species_config,
            empty_tiled_scalar(static_raw_06, dynamic_raw_06),
            static_raw_06,
            dynamic_raw_06,
        )
        rho_raw_10 = compute_rho(
            particles_raw_10,
            species_config,
            empty_tiled_scalar(static_raw_10, dynamic_raw_10),
            static_raw_10,
            dynamic_raw_10,
        )
        rho_digital = compute_rho(
            particles_digital,
            species_config,
            empty_tiled_scalar(static_digital, dynamic_digital),
            static_digital,
            dynamic_digital,
        )
        expected_digital = _manual_rho_tiles(particles_digital, species_config, static_digital, dynamic_digital)

        self.assertTrue(jnp.allclose(rho_raw_06, rho_raw_10, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(rho_digital, expected_digital, rtol=1.0e-12, atol=1.0e-12))
        self.assertGreater(float(jnp.max(jnp.abs(rho_digital - rho_raw_06))), 1.0e-12)

    def test_single_particle_rho_matches_exact_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(shape_factor=shape_factor)
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, (0.0, 0.2, 0.0))
                    rho = compute_rho(
                        particles,
                        species_config,
                        empty_tiled_scalar(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )
                    expected = _manual_rho_tiles(particles, species_config, static_parameters, dynamic_parameters)

                    self.assertTrue(jnp.allclose(rho, expected, rtol=1.0e-12, atol=1.0e-12))
                    total_charge = jnp.sum(_assemble_scalar(rho, static_parameters)[1:-1, 1:-1, 1:-1])
                    total_charge *= dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
                    self.assertAlmostEqual(float(total_charge), -0.5, places=12)

    def test_single_particle_direct_current_matches_exact_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        u = (0.11, -0.17, 0.07)
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(shape_factor=shape_factor)
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, u)
                    J = J_from_rhov(
                        particles,
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )
                    expected = _manual_direct_current_tiles(particles, species_config, static_parameters, dynamic_parameters)

                    _assert_vector_close(self, J, expected)

    def test_single_particle_esirkepov_current_matches_exact_1d_shape_stencils(self):
        positions = {
            "interior": (-1.32, 0.0, 0.0),
            "tile_face": (-0.03, 0.0, 0.0),
            "global_boundary": (1.97, 0.0, 0.0),
        }
        u = (0.08, -0.17, 0.07)
        for shape_factor in (1, 2):
            for location, x in positions.items():
                with self.subTest(shape_factor=shape_factor, location=location):
                    static_parameters, dynamic_parameters = _runtime_parameters(
                        shape_factor=shape_factor,
                        current_deposition="esirkepov",
                    )
                    particles, species_config = _one_particle(static_parameters, dynamic_parameters, x, u)
                    J = Esirkepov_current(
                        particles,
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )
                    expected = _manual_esirkepov_current_tiles_1d(
                        particles,
                        species_config,
                        static_parameters,
                        dynamic_parameters,
                    )

                    _assert_vector_close(self, J, expected)


class TestSingleParticleElectrodynamicPipeline(unittest.TestCase):
    def test_direct_current_source_propagates_into_E_and_B(self):
        static_parameters, dynamic_parameters = _runtime_parameters(
            shape_factor=2,
            current_deposition="direct",
        )
        particles, species_config = _one_particle(
            static_parameters,
            dynamic_parameters,
            (-1.32, 0.0, 0.0),
            (0.0, 0.2, 0.0),
        )
        fields = _empty_electrodynamic_fields(static_parameters, dynamic_parameters)

        particles_after, fields_after = time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        E_after, B_after, J_after, *_rest = fields_after

        centered_particles = update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt / 2)
        centered_particles, overflow = refresh_tiled_particle_tiles(centered_particles, static_parameters, dynamic_parameters)
        self.assertFalse(bool(overflow))
        expected_J = _manual_direct_current_tiles(centered_particles, species_config, static_parameters, dynamic_parameters)
        _assert_vector_close(self, J_after, expected_J)

        E_global = _assemble_vector(E_after, static_parameters)
        J_global = _assemble_vector(expected_J, static_parameters)
        for E_component, J_component in zip(E_global, J_global):
            self.assertTrue(
                jnp.allclose(
                    E_component[1:-1, 1:-1, 1:-1],
                    -dynamic_parameters.dt * J_component[1:-1, 1:-1, 1:-1] / dynamic_parameters.eps,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

        B_global = _assemble_vector(B_after, static_parameters)
        expected_Bz = _expected_B_from_Ey(E_global[1], dynamic_parameters)
        self.assertTrue(jnp.allclose(B_global[0][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(B_global[1][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(
            jnp.allclose(
                B_global[2][1:-1, 1:-1, 1:-1],
                expected_Bz[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        self.assertFalse(bool(fields_after[-1]))
        self.assertEqual(int(jnp.sum(particles_after.active)), 1)

    def test_esirkepov_current_source_propagates_into_E_and_B(self):
        static_parameters, dynamic_parameters = _runtime_parameters(
            shape_factor=1,
            current_deposition="esirkepov",
        )
        initial_x = (-1.32, 0.0, 0.0)
        initial_u = (0.08, 0.2, 0.0)
        particles, species_config = _one_particle(static_parameters, dynamic_parameters, initial_x, initial_u)
        fields = _empty_electrodynamic_fields(static_parameters, dynamic_parameters)

        particles_after, fields_after = time_loop_electrodynamic(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
        )
        E_after, B_after, J_after, *_rest = fields_after

        expected_J = _manual_esirkepov_current_tiles_1d(particles, species_config, static_parameters, dynamic_parameters)
        _assert_vector_close(self, J_after, expected_J)

        active_x = particles_after.x[..., 0][particles_after.active]
        self.assertTrue(
            jnp.allclose(
                active_x,
                jnp.asarray([initial_x[0] + initial_u[0] * float(dynamic_parameters.dt)]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

        E_global = _assemble_vector(E_after, static_parameters)
        J_global = _assemble_vector(expected_J, static_parameters)
        for E_component, J_component in zip(E_global, J_global):
            self.assertTrue(
                jnp.allclose(
                    E_component[1:-1, 1:-1, 1:-1],
                    -dynamic_parameters.dt * J_component[1:-1, 1:-1, 1:-1] / dynamic_parameters.eps,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            )

        B_global = _assemble_vector(B_after, static_parameters)
        expected_Bz = _expected_B_from_Ey(E_global[1], dynamic_parameters)
        self.assertTrue(jnp.allclose(B_global[0][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(jnp.allclose(B_global[1][1:-1, 1:-1, 1:-1], 0.0, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(
            jnp.allclose(
                B_global[2][1:-1, 1:-1, 1:-1],
                expected_Bz[1:-1, 1:-1, 1:-1],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        self.assertFalse(bool(fields_after[-1]))


if __name__ == "__main__":
    unittest.main()
