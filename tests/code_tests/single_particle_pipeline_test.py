import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_PARTICLE,
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
from PyPIC3D.solvers.yee.time_loop import time_loop_electrodynamic
from PyPIC3D.utilities.field_helpers import filter_electric_field_for_particles, yee_push_fields
from PyPIC3D.initialization import initialize_fields
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles, update_tiled_particle_positions
from PyPIC3D.pusher.boris import interpolate_field_to_particles
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.filters import digital_filter, digital_filter_vector
from tests.kernel_fixtures import build_tiled_particles, empty_tiled_scalar, empty_tiled_vector, kernel_parameters, particle_species


def _runtime_parameters(
    *,
    shape_factor=1,
    tile_shape=None,
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
    active_indices = jnp.argwhere(jnp.asarray(jax.device_get(particles.active)))
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


def _direct_deposition_stencils(tile, x, static_parameters, dynamic_parameters):
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
    x_grid, y_grid, z_grid = dynamic_parameters.grids.center

    x_pos = jnp.asarray([x[0]])
    y_pos = jnp.asarray([x[1]])
    z_pos = jnp.asarray([x[2]])
    x_pos, _, deltax_node, xpts_node = prepare_particle_axis_stencil(
        x_pos,
        x_grid,
        x_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_nx * dynamic_parameters.dx,
        ghost_cells=True,
    )
    _, _, deltax_face, xpts_face = prepare_particle_axis_stencil(
        x_pos,
        x_grid + 0.5 * dynamic_parameters.dx,
        x_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_nx * dynamic_parameters.dx,
        ghost_cells=True,
    )
    y_pos, _, deltay_node, ypts_node = prepare_particle_axis_stencil(
        y_pos,
        y_grid,
        y_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_ny * dynamic_parameters.dy,
        ghost_cells=True,
    )
    _, _, deltay_face, ypts_face = prepare_particle_axis_stencil(
        y_pos,
        y_grid + 0.5 * dynamic_parameters.dy,
        y_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_ny * dynamic_parameters.dy,
        ghost_cells=True,
    )
    z_pos, _, deltaz_node, zpts_node = prepare_particle_axis_stencil(
        z_pos,
        z_grid,
        z_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_nz * dynamic_parameters.dz,
        ghost_cells=True,
    )
    _, _, deltaz_face, zpts_face = prepare_particle_axis_stencil(
        z_pos,
        z_grid + 0.5 * dynamic_parameters.dz,
        z_grid.shape[0],
        static_parameters.shape_factor,
        2,
        wind=tile_nz * dynamic_parameters.dz,
        ghost_cells=True,
    )

    node_weights = _weights(deltax_node, deltay_node, deltaz_node, dynamic_parameters, static_parameters.shape_factor)
    face_weights = _weights(deltax_face, deltay_face, deltaz_face, dynamic_parameters, static_parameters.shape_factor)

    offsets = (
        tx * tile_nx - (g - 1),
        ty * tile_ny - (g - 1),
        tz * tile_nz - (g - 1),
    )
    node_points = (
        jnp.asarray(xpts_node) - offsets[0],
        jnp.asarray(ypts_node) - offsets[1],
        jnp.asarray(zpts_node) - offsets[2],
    )
    face_points = (
        jnp.asarray(xpts_face) - offsets[0],
        jnp.asarray(ypts_face) - offsets[1],
        jnp.asarray(zpts_face) - offsets[2],
    )

    collapsed_node_points = []
    collapsed_face_points = []
    collapsed_node_weights = []
    collapsed_face_weights = []
    for axis in range(3):
        points, weights = _collapse(
            node_points[axis],
            node_weights[axis],
            local_shape[axis],
            reduced[axis],
            g,
        )
        collapsed_node_points.append(points)
        collapsed_node_weights.append(weights)

        points, weights = _collapse(
            face_points[axis],
            face_weights[axis],
            local_shape[axis],
            reduced[axis],
            g,
        )
        collapsed_face_points.append(points)
        collapsed_face_weights.append(weights)

    return (
        tuple(collapsed_node_points),
        tuple(collapsed_face_points),
        tuple(collapsed_node_weights),
        tuple(collapsed_face_weights),
    )


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
    node_points, _face_points, node_weights, _face_weights = _direct_deposition_stencils(
        tile, x, static_parameters, dynamic_parameters
    )
    rho = empty_tiled_scalar(static_parameters, dynamic_parameters)
    charge_density = species_config.charge[0] * species_config.weight[0] / (
        dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    )

    rho = _add_stencil(rho, tile, node_points, node_weights, charge_density)
    rho = fold_tiled_ghost_cells(rho, static_parameters, g, bc_type=BC_TYPE_PARTICLE)
    rho = update_tiled_ghost_cells(rho, static_parameters, g, bc_type=BC_TYPE_PARTICLE)

    if static_parameters.current_filter == "digital":
        rho = digital_filter(rho, dynamic_parameters.alpha, num_guard_cells=g)
        rho = update_tiled_ghost_cells(
            rho,
            static_parameters,
            g,
            bc_type=BC_TYPE_PARTICLE,
        )

    return rho


def _manual_direct_current_tiles(particles, species_config, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    tile, x, u = _particle_state(particles)
    node_points, face_points, node_weights, face_weights = _direct_deposition_stencils(
        tile, x, static_parameters, dynamic_parameters
    )
    Jx, Jy, Jz = empty_tiled_vector(static_parameters, dynamic_parameters)
    charge_density = species_config.charge[0] * species_config.weight[0] / (
        dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    )

    Jx_points = (face_points[0], node_points[1], node_points[2])
    Jy_points = (node_points[0], face_points[1], node_points[2])
    Jz_points = (node_points[0], node_points[1], face_points[2])

    Jx = _add_stencil(
        Jx,
        tile,
        Jx_points,
        (face_weights[0], node_weights[1], node_weights[2]),
        charge_density * u[0],
    )
    Jy = _add_stencil(
        Jy,
        tile,
        Jy_points,
        (node_weights[0], face_weights[1], node_weights[2]),
        charge_density * u[1],
    )
    Jz = _add_stencil(
        Jz,
        tile,
        Jz_points,
        (node_weights[0], node_weights[1], face_weights[2]),
        charge_density * u[2],
    )

    J = fold_tiled_vector_ghost_cells(
        (Jx, Jy, Jz),
        static_parameters,
        g,
        bc_type=BC_TYPE_PARTICLE,
    )
    J = update_tiled_vector_ghost_cells(
        J,
        static_parameters,
        g,
        bc_type=BC_TYPE_PARTICLE,
    )

    if static_parameters.current_filter == "digital":
        J = digital_filter_vector(J, dynamic_parameters.alpha, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(
            J,
            static_parameters,
            g,
            bc_type=BC_TYPE_PARTICLE,
        )

    return J


def _periodic_test_electric_field(static_parameters, dynamic_parameters):
    Nx = int(dynamic_parameters.Nx)
    Ny = int(dynamic_parameters.Ny)
    Nz = int(dynamic_parameters.Nz)
    g = int(static_parameters.guard_cells)

    ix, iy, iz = jnp.meshgrid(
        jnp.arange(Nx, dtype=float),
        jnp.arange(Ny, dtype=float),
        jnp.arange(Nz, dtype=float),
        indexing="ij",
    )

    Ex_interior = (
        0.83 * jnp.sin(2.0 * jnp.pi * (ix + 0.17) / Nx)
        + 0.37 * jnp.cos(4.0 * jnp.pi * (iy + 0.31) / Ny)
        + 0.19 * jnp.sin(6.0 * jnp.pi * (iz + 0.11) / Nz)
    )
    Ey_interior = (
        -0.41 * jnp.cos(4.0 * jnp.pi * (ix + 0.23) / Nx)
        + 0.71 * jnp.sin(2.0 * jnp.pi * (iy + 0.37) / Ny)
        + 0.29 * jnp.cos(6.0 * jnp.pi * (iz + 0.07) / Nz)
    )
    Ez_interior = (
        0.53 * jnp.sin(6.0 * jnp.pi * (ix + 0.43) / Nx)
        - 0.47 * jnp.cos(2.0 * jnp.pi * (iy + 0.13) / Ny)
        + 0.61 * jnp.sin(4.0 * jnp.pi * (iz + 0.29) / Nz)
    )

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    ntx, nty, ntz = static_parameters.field_mesh.devices.shape

    E = list(empty_tiled_vector(static_parameters, dynamic_parameters))
    for component_index, interior in enumerate((Ex_interior, Ey_interior, Ez_interior)):
        for tx in range(int(ntx)):
            for ty in range(int(nty)):
                for tz in range(int(ntz)):
                    ix = tx * tile_nx
                    iy = ty * tile_ny
                    iz = tz * tile_nz
                    tile_interior = interior[
                        ix:ix + tile_nx,
                        iy:iy + tile_ny,
                        iz:iz + tile_nz,
                    ]
                    E[component_index] = E[component_index].at[
                        tx, ty, tz, g:-g, g:-g, g:-g
                    ].set(tile_interior)

    return update_tiled_vector_ghost_cells(
        tuple(E),
        static_parameters,
        num_guard_cells=g,
        bc_type=0,
    )


def _gather_electric_field(E, tile, position, static_parameters, dynamic_parameters):
    x = jnp.asarray([position[0]], dtype=float)
    y = jnp.asarray([position[1]], dtype=float)
    z = jnp.asarray([position[2]], dtype=float)
    g = int(static_parameters.guard_cells)

    tx, ty, tz = tile
    center_x = dynamic_parameters.grids.tiled_center_grid[0][tx, ty, tz]
    center_y = dynamic_parameters.grids.tiled_center_grid[1][tx, ty, tz]
    center_z = dynamic_parameters.grids.tiled_center_grid[2][tx, ty, tz]
    vertex_x = dynamic_parameters.grids.tiled_vertex_grid[0][tx, ty, tz]
    vertex_y = dynamic_parameters.grids.tiled_vertex_grid[1][tx, ty, tz]
    vertex_z = dynamic_parameters.grids.tiled_vertex_grid[2][tx, ty, tz]

    ntx, nty, ntz = static_parameters.field_mesh.devices.shape
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    active_axes = (
        int(ntx) * tile_nx > 1,
        int(nty) * tile_ny > 1,
        int(ntz) * tile_nz > 1,
    )

    component_grids = (
        (vertex_x, center_y, center_z),
        (center_x, vertex_y, center_z),
        (center_x, center_y, vertex_z),
    )

    gathered = []
    for component, component_grid in zip(E, component_grids):
        value = interpolate_field_to_particles(
            component[tx, ty, tz],
            x,
            y,
            z,
            component_grid,
            static_parameters.shape_factor,
            ghost_cells=True,
            active_axes=active_axes,
            inactive_axis_indices=(g, g, g),
        )
        gathered.append(value[0])

    return jnp.asarray(gathered)


def _grid_current_work(E, J, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    cell_volume = dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz

    work = 0.0
    for electric_component, current_component in zip(E, J):
        work += jnp.sum(
            electric_component[:, :, :, g:-g, g:-g, g:-g]
            * current_component[:, :, :, g:-g, g:-g, g:-g]
        ) * cell_volume

    return work


def _shift_old_stencil(weights, shift):
    old_weights = jnp.stack(weights, axis=0)
    return [jnp.roll(old_weights[:, 0], -int(shift))[i, jnp.newaxis] for i in range(5)]


def _manual_esirkepov_current_tiles_1d(particles, species_config, static_parameters, dynamic_parameters):
    g = int(static_parameters.guard_cells)
    tile, old_x, u = _particle_state(particles)
    tx, ty, tz = tile
    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
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

    J = fold_tiled_vector_ghost_cells(
        (Jx, Jy, Jz),
        static_parameters,
        g,
        bc_type=BC_TYPE_PARTICLE,
    )
    return update_tiled_vector_ghost_cells(
        J,
        static_parameters,
        g,
        bc_type=BC_TYPE_PARTICLE,
    )


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


def _expected_Bz_after_split_update(Ey_before, Ey_after, dynamic_parameters):
    expected = jnp.zeros_like(Ey_after)
    active = (slice(1, -1), slice(1, -1), slice(1, -1))
    forward_x = (slice(2, None), slice(1, -1), slice(1, -1))
    # Ey is centered in x, while Bz is staggered in x under the legacy
    # center=collocated, vertex=staggered contract.
    dEy_before_dx = (Ey_before[forward_x] - Ey_before[active]) / dynamic_parameters.dx
    dEy_after_dx = (Ey_after[forward_x] - Ey_after[active]) / dynamic_parameters.dx

    half_dt = dynamic_parameters.dt / 2
    # The field loop advances B by half a timestep on each side of the E update.
    return expected.at[active].set(-half_dt * (dEy_before_dx + dEy_after_dx))


class TestSingleParticleStencils(unittest.TestCase):
    def test_coupling_filter_leaves_none_external_and_magnetic_fields_unchanged(self):
        static_none_06, dynamic_none_06 = _runtime_parameters(current_filter="none", alpha=0.6)
        static_none_10, dynamic_none_10 = _runtime_parameters(current_filter="none", alpha=1.0)
        static_digital, dynamic_digital = _runtime_parameters(current_filter="digital", alpha=0.6)

        E = _periodic_test_electric_field(static_none_06, dynamic_none_06)
        B = tuple(0.3 * component for component in E)
        external_E = tuple(0.2 * component for component in E)
        external_B = tuple(-0.4 * component for component in B)

        coupling_none_06 = filter_electric_field_for_particles(
            E,
            static_none_06,
            dynamic_none_06,
        )
        coupling_none_10 = filter_electric_field_for_particles(
            E,
            static_none_10,
            dynamic_none_10,
        )
        coupling_digital = filter_electric_field_for_particles(
            E,
            static_digital,
            dynamic_digital,
        )
        push_E, push_B = yee_push_fields(
            E,
            B,
            (external_E, external_B),
            static_digital,
            dynamic_digital,
        )

        _assert_vector_close(self, coupling_none_06, E)
        _assert_vector_close(self, coupling_none_10, E)
        self.assertTrue(any(
            not jnp.allclose(filtered, unfiltered, rtol=1.0e-12, atol=1.0e-12)
            for filtered, unfiltered in zip(coupling_digital, E)
        ))
        _assert_vector_close(
            self,
            tuple(total - filtered for total, filtered in zip(push_E, coupling_digital)),
            external_E,
        )
        _assert_vector_close(
            self,
            push_B,
            tuple(field + external for field, external in zip(B, external_B)),
        )

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

    def test_direct_current_scatter_is_adjoint_to_production_yee_gather(self):
        velocity = jnp.asarray((0.71, -0.43, 0.29))
        charge = -1.3
        weight = 0.6

        configurations = {
            "one_tile_3d": {
                "grid_shape": (8, 8, 8),
                "tile_shape": (8, 8, 8),
                "wind": (8.0, 8.0, 8.0),
                "positions": {
                    "lower_cell_half": (-3.75, -2.75, -1.75),
                    "upper_periodic_seam": (3.99, 3.91, 3.83),
                },
            },
            "one_tile_reduced": {
                "grid_shape": (8, 1, 1),
                "tile_shape": (8, 1, 1),
                "wind": (8.0, 1.0, 1.0),
                "positions": {
                    "lower_tile_interface": (-0.01, 0.0, 0.0),
                    "upper_periodic_seam": (3.99, 0.0, 0.0),
                },
            },
        }

        parameter_cases = (
            ("one_tile_3d", 1, "none"),
            ("one_tile_3d", 2, "digital"),
            ("one_tile_3d", 1, "bilinear"),
            ("one_tile_reduced", 2, "none"),
            ("one_tile_reduced", 1, "digital"),
            ("one_tile_reduced", 2, "bilinear"),
        )

        for configuration, shape_factor, current_filter in parameter_cases:
            values = configurations[configuration]
            Nx, Ny, Nz = values["grid_shape"]
            x_wind, y_wind, z_wind = values["wind"]
            static_parameters, dynamic_parameters = kernel_parameters(
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                x_wind=x_wind,
                y_wind=y_wind,
                z_wind=z_wind,
                tile_shape=values["tile_shape"],
                guard_cells=2,
                shape_factor=shape_factor,
                current_deposition="direct",
                current_filter=current_filter,
                relativistic=False,
                alpha=0.6,
                dt=0.1,
            )
            E = _periodic_test_electric_field(static_parameters, dynamic_parameters)
            coupling_E = filter_electric_field_for_particles(
                E,
                static_parameters,
                dynamic_parameters,
            )

            for phase, position in values["positions"].items():
                with self.subTest(
                    configuration=configuration,
                    shape_factor=shape_factor,
                    current_filter=current_filter,
                    phase=phase,
                ):
                    particles, species_config = _one_particle(
                        static_parameters,
                        dynamic_parameters,
                        position,
                        velocity,
                        charge=charge,
                        weight=weight,
                    )
                    tile, _, _ = _particle_state(particles)
                    J = J_from_rhov(
                        particles,
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                    )

                    grid_work = _grid_current_work(
                        E,
                        J,
                        static_parameters,
                        dynamic_parameters,
                    )
                    particle_field = _gather_electric_field(
                        coupling_E,
                        tile,
                        position,
                        static_parameters,
                        dynamic_parameters,
                    )
                    particle_work = charge * weight * jnp.dot(velocity, particle_field)

                    work_scale = max(abs(float(grid_work)), abs(float(particle_work)), 1.0e-30)
                    relative_residual = abs(float(grid_work - particle_work)) / work_scale

                    self.assertLessEqual(relative_residual, 1.0e-12)

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
                        update_tiled_particle_positions(particles, species_config, dynamic_parameters.dt),
                        species_config,
                        empty_tiled_vector(static_parameters, dynamic_parameters),
                        static_parameters,
                        dynamic_parameters,
                        coordinate_velocity=particles.u,
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
        E_before_global = _assemble_vector(fields[0], static_parameters)

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
        expected_Bz = _expected_Bz_after_split_update(
            E_before_global[1],
            E_global[1],
            dynamic_parameters,
        )
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
        E_before_global = _assemble_vector(fields[0], static_parameters)

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
        expected_Bz = _expected_Bz_after_split_update(
            E_before_global[1],
            E_global[1],
            dynamic_parameters,
        )
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
