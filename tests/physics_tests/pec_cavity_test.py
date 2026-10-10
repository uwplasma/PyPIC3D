import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells, apply_tiled_pec_boundary
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING
from PyPIC3D.solvers.yee.first_order_yee import (
    update_B,
    update_E,
    yee_curl_e_to_b,
)
from tests.kernel_fixtures import initialized_fields, kernel_parameters


def _magnitude_squared(vector):
    return sum(jnp.sum(component**2) for component in vector)


def _dot_product(first, second):
    return sum(jnp.sum(a * b) for a, b in zip(first, second))


def _active_vector(vector, guard_cells):
    g = int(guard_cells)
    active = (0, 0, 0, slice(g, -g), slice(g, -g), slice(g, -g))
    return tuple(component[active] for component in vector)


def _tm111_electric_mode(num_points, spacing):
    """Construct the discrete TM111 electric mode on the active Yee arrays."""


    # TM111 has wavenumbers: kx = pi/a, ky = pi/b, kz = pi/z
    # Ex = -kx kz / k^2 E0 Sin(kx x ) Cos( ky y) Sin(kz z)
    # Ey = -ky kz / k^2 E0 Cos(kx x)  Sin( ky y) Sin(kz z)
    # Ez =              E0 Cos(kx x)  Cos( ky y) Cos(kz z)
    

    phase = math.pi * jnp.arange(num_points+1, dtype=jnp.float64) / num_points
    sine = jnp.sin(phase)
    sine = sine.at[0].set(0.0)
    sine = sine.at[-1].set(0.0)
    # define the phase of the sine wave

    gradient = jnp.diff(sine) / spacing
    sine = sine[:-1]  # the upper C endpoint lives in the first halo slot

    modified_wavenumber = 2.0 * math.sin(math.pi / (2.0 * num_points)) / spacing
    transverse_wavenumber_squared = 2.0 * modified_wavenumber**2

    gradient_x, sine_y, sine_z = jnp.meshgrid(gradient, sine, sine, indexing="ij")
    sine_x, gradient_y, sine_z_for_ey = jnp.meshgrid(sine, gradient, sine, indexing="ij")
    sine_x_for_ez, sine_y_for_ez, gradient_z = jnp.meshgrid(sine, sine, gradient, indexing="ij")

    Ex = (
        -modified_wavenumber
        / transverse_wavenumber_squared
        * gradient_x
        * sine_y
        * sine_z
    )
    Ey = (
        -modified_wavenumber
        / transverse_wavenumber_squared
        * sine_x
        * gradient_y
        * sine_z_for_ey
    )
    Ez = sine_x_for_ez * sine_y_for_ez * gradient_z / modified_wavenumber

    return (Ex, Ey, Ez), modified_wavenumber


def _tangential_electric_residual(E, g):
    walls = []
    for axis in range(3):
        for i, value in enumerate(E):
            if i == axis:
                continue
            for endpoint in (g, value.shape[axis+3]-g):
                index = [0, 0, 0, slice(g, -g), slice(g, -g), slice(g, -g)]
                index[axis+3] = endpoint
                walls.append(jnp.max(jnp.abs(value[tuple(index)])))
    return jnp.max(jnp.stack(walls))


def _observed_order(coarse_error, fine_error, coarse_spacing, fine_spacing):
    return math.log(coarse_error / fine_error) / math.log(coarse_spacing / fine_spacing)


class TestPECStandingWaveCavity(unittest.TestCase):
    def _initial_state(self, num_points, dt):
        spacing = 1.0 / num_points
        nominal_width = num_points * spacing

        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=num_points,
            Ny=num_points,
            Nz=num_points,
            x_wind=nominal_width,
            y_wind=nominal_width,
            z_wind=nominal_width,
            x_min=-0.5,
            y_min=-0.5,
            z_min=-0.5,
            dx=spacing,
            dy=spacing,
            dz=spacing,
            dt=dt,
            tile_shape=(num_points, num_points, num_points),
            boundary_conditions=(BC_CONDUCTING, BC_CONDUCTING, BC_CONDUCTING),
            current_filter="none",
            C=1.0,
            eps=1.0,
            mu=1.0,
            pml_active=False,
            supergaussian_active=False,
        )

        E_values, modified_wavenumber = _tm111_electric_mode(num_points, spacing)
        E, B, J, _phi, _rho = initialized_fields(static_parameters, dynamic_parameters)

        g = int(static_parameters.guard_cells)
        active = (0, 0, 0, slice(g, -g), slice(g, -g), slice(g, -g))
        E = tuple(
            component.at[active].set(values)
            for component, values in zip(E, E_values)
        )
        E = apply_tiled_pec_boundary(E, static_parameters)
        E = update_tiled_vector_ghost_cells(E, static_parameters, g, locations=D_FIELD_LOCATIONS)
        B = update_tiled_vector_ghost_cells(B, static_parameters, g, locations=B_FIELD_LOCATIONS)

        return (
            E,
            B,
            J,
            E_values,
            modified_wavenumber,
            spacing,
            static_parameters,
            dynamic_parameters,
        )

    def _run_cavity(self, num_points, num_steps, total_time):
        dt = total_time / num_steps
        (
            E,
            B,
            J,
            E_values,
            modified_wavenumber,
            spacing,
            static_parameters,
            dynamic_parameters,
        ) = self._initial_state(num_points, dt)

        g = int(static_parameters.guard_cells)
        E_template = _active_vector(E, g)
        curl_E_template = tuple(
            component[0, 0, 0]
            for component in yee_curl_e_to_b(E, static_parameters, dynamic_parameters)
        )
        omega_discrete_space = math.sqrt(3.0) * modified_wavenumber
        B_template = tuple(-component / omega_discrete_space for component in curl_E_template)

        E_template_norm_squared = _magnitude_squared(E_template)
        B_template_norm_squared = _magnitude_squared(B_template)
        cell_volume = spacing**3

        def diagnostics(E_now, B_now):
            E_active = _active_vector(E_now, g)
            B_active = _active_vector(B_now, g)
            curl_E = yee_curl_e_to_b(E_now, static_parameters, dynamic_parameters)

            physical_energy = 0.5 * cell_volume * (
                _magnitude_squared(E_active) + _magnitude_squared(B_active)
            )
            modified_energy = (
                physical_energy
                - cell_volume * dt**2 * _magnitude_squared(curl_E) / 8.0
            )

            electric_amplitude = _dot_product(E_active, E_template) / E_template_norm_squared
            magnetic_amplitude = _dot_product(B_active, B_template) / B_template_norm_squared
            electric_residual = tuple(
                component - electric_amplitude * template
                for component, template in zip(E_active, E_template)
            )
            magnetic_residual = tuple(
                component - magnetic_amplitude * template
                for component, template in zip(B_active, B_template)
            )
            modal_residual = jnp.sqrt(
                (
                    _magnitude_squared(electric_residual)
                    + _magnitude_squared(magnetic_residual)
                )
                / E_template_norm_squared
            )

            return (
                physical_energy,
                modified_energy,
                jnp.max(jnp.abs(B_active[2])),
                jnp.max(jnp.abs(B_active[0])),
                jnp.max(jnp.abs(B_active[1])),
                _tangential_electric_residual(E_now, g),
                modal_residual,
            )

        def step(fields, _unused):
            E_now, B_now = fields

            B_now, _ = update_B(
                E_now,
                B_now,
                static_parameters,
                dynamic_parameters,
            )
            E_now, _ = update_E(
                E_now,
                B_now,
                J,
                static_parameters,
                dynamic_parameters,
            )
            B_now, _ = update_B(
                E_now,
                B_now,
                static_parameters,
                dynamic_parameters,
            )

            return (E_now, B_now), diagnostics(E_now, B_now)

        @jax.jit
        def evolve(E_start, B_start):
            return jax.lax.scan(
                step,
                (E_start, B_start),
                xs=None,
                length=num_steps,
            )

        initial_diagnostics = diagnostics(E, B)
        (E_final, B_final), history = evolve(E, B)
        (
            physical_energy,
            modified_energy,
            Bz_maximum,
            Bx_maximum,
            By_maximum,
            wall_residual,
            modal_residual,
        ) = history

        E_final_active = _active_vector(E_final, g)
        B_final_active = _active_vector(B_final, g)
        endpoint_error_squared = _magnitude_squared(
            tuple(
                component - reference
                for component, reference in zip(E_final_active, E_values)
            )
        ) + _magnitude_squared(B_final_active)
        endpoint_error = jnp.sqrt(endpoint_error_squared / E_template_norm_squared)

        initial_physical_energy = initial_diagnostics[0]
        initial_modified_energy = initial_diagnostics[1]
        physical_energy_excursion = jnp.max(
            jnp.abs(physical_energy / initial_physical_energy - 1.0)
        )
        modified_energy_excursion = jnp.max(
            jnp.abs(modified_energy / initial_modified_energy - 1.0)
        )

        return {
            "spacing": spacing,
            "endpoint_error": float(endpoint_error),
            "physical_energy_excursion": float(physical_energy_excursion),
            "modified_energy_excursion": float(modified_energy_excursion),
            "Bz_maximum": float(jnp.max(Bz_maximum)),
            "Bx_maximum": float(jnp.max(Bx_maximum)),
            "By_maximum": float(jnp.max(By_maximum)),
            "wall_residual": float(jnp.max(wall_residual)),
            "modal_residual": float(jnp.max(modal_residual)),
            "finite": bool(
                jnp.all(jnp.isfinite(physical_energy))
                & jnp.all(jnp.isfinite(modified_energy))
                & jnp.isfinite(endpoint_error)
            ),
        }

    def _assert_cavity_invariants(self, result):
        self.assertTrue(result["finite"])
        self.assertLess(result["Bz_maximum"], 1.0e-12)
        self.assertGreater(result["Bx_maximum"], 1.0e-1)
        self.assertGreater(result["By_maximum"], 1.0e-1)
        self.assertLess(result["wall_residual"], 1.0e-12)
        self.assertLess(result["modal_residual"], 1.0e-10)
        self.assertLess(result["modified_energy_excursion"], 1.0e-11)

    def test_discrete_tm111_initial_mode_respects_pec_boundaries(self):
        (
            E,
            B,
            J,
            _E_values,
            _modified_wavenumber,
            _spacing,
            static_parameters,
            dynamic_parameters,
        ) = self._initial_state(num_points=12, dt=1.0e-2)

        g = int(static_parameters.guard_cells)
        E_active = _active_vector(E, g)
        B_active = _active_vector(B, g)
        J_active = _active_vector(J, g)
        curl_E = yee_curl_e_to_b(E, static_parameters, dynamic_parameters)

        for component in (*E_active, *B_active, *J_active):
            self.assertTrue(bool(jnp.all(jnp.isfinite(component))))
        for component in (*B_active, *J_active):
            self.assertTrue(bool(jnp.all(component == 0.0)))

        self.assertLess(float(_tangential_electric_residual(E, g)), 1.0e-12)
        self.assertLess(float(jnp.max(jnp.abs(curl_E[2]))), 1.0e-12)

    def test_tm111_converges_with_space_and_time_refinement(self):
        continuum_period = 2.0 / math.sqrt(3.0)
        results = []

        for num_points in (8, 12, 16):
            spacing = 1.0 / num_points
            dt_limit = 0.5 / (3.0 / spacing)
            num_steps = math.ceil(continuum_period / dt_limit)
            results.append(self._run_cavity(num_points, num_steps, continuum_period))

        for result in results:
            self._assert_cavity_invariants(result)

        field_errors = [result["endpoint_error"] for result in results]
        energy_errors = [result["physical_energy_excursion"] for result in results]
        spacings = [result["spacing"] for result in results]

        self.assertGreater(field_errors[0], field_errors[1])
        self.assertGreater(field_errors[1], field_errors[2])
        self.assertGreater(energy_errors[0], energy_errors[1])
        self.assertGreater(energy_errors[1], energy_errors[2])

        for coarse, fine, coarse_spacing, fine_spacing in zip(
            field_errors[:-1],
            field_errors[1:],
            spacings[:-1],
            spacings[1:],
        ):
            self.assertGreater(
                _observed_order(coarse, fine, coarse_spacing, fine_spacing),
                1.8,
            )
        for coarse, fine, coarse_spacing, fine_spacing in zip(
            energy_errors[:-1],
            energy_errors[1:],
            spacings[:-1],
            spacings[1:],
        ):
            self.assertGreater(
                _observed_order(coarse, fine, coarse_spacing, fine_spacing),
                1.8,
            )

        self.assertLess(field_errors[-1], 2.0e-2)
        self.assertLess(energy_errors[-1], 2.0e-3)

    def test_tm111_converges_with_timestep_refinement(self):
        num_points = 12
        spacing = 1.0 / num_points
        modified_wavenumber = (
            2.0
            * math.sin(math.pi / (2.0 * num_points))
            / spacing
        )
        discrete_period = 2.0 * math.pi / (math.sqrt(3.0) * modified_wavenumber)
        results = [
            self._run_cavity(num_points, num_steps, discrete_period)
            for num_steps in (24, 48, 96)
        ]

        for result in results:
            self._assert_cavity_invariants(result)

        field_errors = [result["endpoint_error"] for result in results]
        energy_errors = [result["physical_energy_excursion"] for result in results]

        self.assertGreater(field_errors[0], field_errors[1])
        self.assertGreater(field_errors[1], field_errors[2])
        self.assertGreater(energy_errors[0], energy_errors[1])
        self.assertGreater(energy_errors[1], energy_errors[2])

        for coarse, fine in zip(field_errors[:-1], field_errors[1:]):
            self.assertGreater(math.log(coarse / fine, 2.0), 1.8)
        for coarse, fine in zip(energy_errors[:-1], energy_errors[1:]):
            self.assertGreater(math.log(coarse / fine, 2.0), 1.8)

        self.assertLess(field_errors[-1], 2.0e-3)
        self.assertLess(energy_errors[-1], 2.0e-3)


if __name__ == "__main__":
    unittest.main()
