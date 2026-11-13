"""Shared-device BSSN/PIC transfers and coupled time-centering regressions.

Run with JAX_NUM_CPU_DEVICES=2 on a CPU host.
"""

import unittest
from unittest.mock import patch
import importlib

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNVariables
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.particle_tile_communication import shard_tiled_particles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_tile_layout, bssn_to_tiles, tiles_to_bssn, build_bssn_metric,
    initialize_dynamical_relativity,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    electromagnetic_moments, compute_energy_momentum_tensor,
)
from PyPIC3D.diagnostics.gr_work_energy_theorem import sample_energy, volume_integral, surface_integral
from PyPIC3D.diagnostics.static_metric import bssn_constraint_norms
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from tests.kernel_fixtures import kernel_parameters

jax.config.update("jax_enable_x64", True)


def coupled_example(tiles, *, particles=False, current_filter="none", crossing=False):
    static, dynamic = kernel_parameters(
        Nx=16, Ny=1, Nz=1, x_wind=16., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., tile_shape=(16 // tiles, 1, 1),
        dt=.002, guard_cells=3, solver="dynamic_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        particle_batch_size=4, current_filter=current_filter)
    static = static._replace(metric_backreaction=True)
    phase = 2 * jnp.pi * jnp.arange(16)[:, None, None] / 16
    zero = jnp.zeros_like(phase)
    identity = jnp.eye(3)[:, :, None, None, None] * jnp.ones_like(phase)
    vector = jnp.zeros((3,) + phase.shape)
    state = BSSNVariables(identity, 1. + .003 * jnp.cos(phase), identity * 0,
                          .001 * jnp.sin(phase), vector, 1. + .002 * jnp.sin(phase),
                          vector.at[0].set(.01 * jnp.cos(phase)))
    bssn = build_bssn_parameters(static, dynamic, state=state,
                                 evolution_parameters={"nu": .002, "kappa": .03})
    dynamic = dynamic._replace(bssn_parameters=bssn)
    D = tuple(bssn_to_tiles(value, static) for value in
              (.03 + .002 * jnp.sin(phase), .01 * jnp.cos(phase), zero))
    B = tuple(bssn_to_tiles(value, static) for value in
              (zero, .02 + .001 * jnp.sin(phase), .01 * jnp.cos(phase)))
    zeros = tuple(jnp.zeros_like(value) for value in D)
    if particles:
        position_values = [.2, 7.99999, 8.00001, 15.8] if crossing else [.2, 7.9, 8.1, 15.8]
        positions = jnp.array(position_values).reshape(tiles, 1, 1, 1, 4 // tiles)
        x = jnp.zeros(positions.shape + (3,)).at[..., 0].set(positions)
        u = jnp.broadcast_to(jnp.array([.1, -.03, .02]), x.shape)
        if crossing:
            u = u.at[..., 0].set(jnp.array([.1, .2, -.2, -.1]).reshape(positions.shape))
        active = jnp.ones(positions.shape, dtype=bool)
        species = SpeciesConfig(jnp.array([.3]), jnp.array([1.]), jnp.array([.01]),
                                jnp.ones((1, 3), dtype=bool))
    else:
        x = jnp.zeros((tiles, 1, 1, 0, 0, 3))
        u, active = x, jnp.zeros(x.shape[:-1], dtype=bool)
        empty = jnp.zeros((0,))
        species = SpeciesConfig(empty, empty, empty, jnp.zeros((0, 3), dtype=bool))
    particle_state = shard_tiled_particles(TiledParticles(x, u, active), static)
    fields = (D, B, zeros, zeros[0], zeros[0], (zeros, zeros), None, (D, B), jnp.asarray(False))
    particle_state, fields, dynamic = initialize_dynamical_relativity(
        particle_state, species, fields, static, dynamic)
    return static, dynamic, particle_state, species, fields


def gather_bssn(value, static):
    return tiles_to_bssn(value, static._replace(guard_cells=bssn_tile_layout(static)["guard_cells"]))


@unittest.skipUnless(jax.device_count() >= 2, "requires two JAX devices")
class TiledBSSNCouplingTest(unittest.TestCase):
    def assertClose(self, actual, expected, atol=3e-11):
        # Gather only in the test assertion: reference and tiled results have
        # different committed device sets and cannot share one jitted compare.
        actual = jnp.asarray(jax.device_get(actual))
        expected = jnp.asarray(jax.device_get(expected))
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=3e-11, atol=atol)),
                        f"maximum error {float(jnp.max(jnp.abs(actual - expected)))}")

    def test_metric_and_particle_sources_match_across_tile_boundary(self):
        for filtering in ("none", "bilinear", "digital"):
            with self.subTest(filtering=filtering):
                single, tiled = (coupled_example(n, particles=True, current_filter=filtering) for n in (1, 2))
                sources = []
                for static, dynamic, particles, species, fields in (single, tiled):
                    moments = electromagnetic_moments(fields[0], fields[7][2], static,
                                                        tiled=static.field_mesh.size > 1)
                    sources.append(compute_energy_momentum_tensor(
                        particles, species, moments, dynamic.bssn_parameters.state, static, dynamic))
                for actual, expected in zip(sources[1], sources[0]):
                    self.assertClose(gather_bssn(actual, tiled[0]), expected)
                reference_metric = build_bssn_metric(single[1].bssn_parameters.state, tiled[0], single[1])
                for actual, expected in zip(jax.tree.leaves(tiled[4][6]), jax.tree.leaves(reference_metric)):
                    self.assertClose(actual, expected)

    def test_owned_work_and_constraints_match_serial_reductions(self):
        single, tiled = (coupled_example(n) for n in (1, 2))
        samples, constraints = [], []
        for static, dynamic, particles, species, fields in (single, tiled):
            samples.append(sample_energy(particles, species, fields, static, dynamic))
            constraints.append(bssn_constraint_norms(particles, species, fields, static, dynamic))
        for actual, expected in zip(samples[1], samples[0]):
            self.assertClose(actual, expected)
        for name in constraints[0]:
            self.assertClose(constraints[1][name], constraints[0][name])

    def test_tiled_quadrature_preserves_open_and_endpoint_weights(self):
        for boundary in (0, 1, 2, 3):
            with self.subTest(boundary=boundary):
                static, dynamic = kernel_parameters(
                    Nx=16, Ny=1, Nz=1, x_wind=16., y_wind=1., z_wind=1.,
                    tile_shape=(8, 1, 1), guard_cells=4,
                    boundary_conditions=(boundary, 0, 0))
                x = jnp.arange(16 + (boundary == 1))[:, None, None]
                density = 1. + .03 * x + .001 * x**2
                tiled = bssn_to_tiles(density, static)
                self.assertClose(volume_integral(tiled, static, dynamic),
                                 volume_integral(density, static, dynamic))
                flux = jnp.stack((density, density * 0, density * 0))
                self.assertClose(surface_integral(bssn_to_tiles(flux, static), static, dynamic),
                                 surface_integral(flux, static, dynamic))

    def test_source_only_filter_preserves_serial_moments(self):
        single, tiled = (coupled_example(n) for n in (1, 2))
        results = []
        for static, dynamic, particles, species, fields in (single, tiled):
            moments = electromagnetic_moments(fields[0], fields[7][2], static,
                                                tiled=static.field_mesh.size > 1)
            results.append(compute_energy_momentum_tensor(
                particles, species, moments, dynamic.bssn_parameters.state, static, dynamic,
                source_filter="binomial"))
        for actual, expected in zip(results[1], results[0]):
            self.assertClose(gather_bssn(actual, tiled[0]), expected)

    def test_source_filter_uses_em_exteriors_when_gravity_is_periodic(self):
        from PyPIC3D.relativity.filtering import binomial_source_filter
        for boundary in (1, 2, 3):
            with self.subTest(boundary=boundary):
                static, dynamic = kernel_parameters(
                    Nx=16, Ny=1, Nz=1, x_wind=16., y_wind=1., z_wind=1.,
                    tile_shape=(8, 1, 1), guard_cells=3,
                    solver="dynamic_metric", metric="numerical",
                    boundary_conditions=(boundary, 0, 0))
                bssn = build_bssn_parameters(static, dynamic)
                dynamic = dynamic._replace(bssn_parameters=bssn)
                x = jnp.arange(16 + (boundary == 1))[:, None, None]
                values = 1 + .02 * x**2
                parameters = static._replace(guard_cells=bssn_tile_layout(static)["guard_cells"])
                tiled = bssn_to_tiles(values, parameters)
                actual = binomial_source_filter(tiled, static.boundary_conditions,
                                                static_parameters=static, dynamic_parameters=dynamic)
                expected = binomial_source_filter(values, static.boundary_conditions)
                self.assertClose(gather_bssn(actual, static), expected)

    @unittest.skipUnless(jax.device_count() >= 8, "requires eight JAX devices")
    def test_particle_cloud_straddles_eight_tile_corners(self):
        sources = []
        phase = jnp.asarray([[x, y, z] for x in (-.01, .01) for y in (-.01, .01)
                             for z in (-.01, .01)])
        for count in (1, 2):
            static, dynamic = kernel_parameters(
                Nx=16, Ny=16, Nz=16, x_wind=16., y_wind=16., z_wind=16.,
                x_min=0., y_min=0., z_min=0., tile_shape=(16 // count,) * 3,
                guard_cells=3, solver="dynamic_metric", metric="numerical",
                particle_pusher="hybrid_boris_geodesic", particle_batch_size=8)
            static = static._replace(metric_backreaction=True)
            bssn = build_bssn_parameters(static, dynamic)
            dynamic = dynamic._replace(bssn_parameters=bssn)
            metric = build_bssn_metric(bssn.state, static, dynamic)
            positions = (8 + phase).reshape((count,) * 3 + (1, 8 // count**3, 3))
            momenta = jnp.broadcast_to(jnp.array([.2, -.1, .05]), positions.shape)
            particles = TiledParticles(positions, momenta, jnp.ones(positions.shape[:-1], dtype=bool))
            species = SpeciesConfig(jnp.array([.3]), jnp.array([1.]), jnp.array([.01]),
                                    jnp.ones((1, 3), dtype=bool))
            zero = jnp.zeros_like(metric.center.lapse)
            fields = (zero, zero, zero)
            moments = electromagnetic_moments(fields, fields, static, tiled=count > 1)
            result = compute_energy_momentum_tensor(particles, species, moments, bssn.state, static, dynamic)
            sources.append(result if count == 1 else tuple(gather_bssn(value, static) for value in result))
        for actual, expected in zip(sources[1], sources[0]):
            self.assertClose(actual, expected)

    def test_tiled_callbacks_receive_local_coordinate_metadata(self):
        static, dynamic, particles, species, fields = coupled_example(2)
        from JAX_BSSN.parallel import map_tiled
        state = dynamic.bssn_parameters.state

        def prescribe(local_state, time, *, params):
            self.assertIsNotNone(params.owned_shape)
            x = params.x_min + params.dx * jnp.arange(local_state.lapse.shape[0])[:, None, None]
            lapse = jnp.broadcast_to(1 + .002 * jnp.cos(2 * jnp.pi * x / 16), local_state.lapse.shape)
            return lapse, local_state.shift * 0, lapse * 0, local_state.shift * 0

        def boundary(local_state, time, *, params):
            self.assertEqual(len(params.owned_faces), 6)
            return local_state

        actual = map_tiled(lambda q, p: prescribe(q, dynamic.dt, params=p)[0], state,
                           dynamic.bssn_parameters.evolution_parameters, **bssn_tile_layout(static),
                           output_template=state.lapse)
        expected = 1 + .002 * jnp.cos(2 * jnp.pi * jnp.arange(16)[:, None, None] / 16)
        self.assertClose(gather_bssn(actual, static), expected)
        result = jax.eval_shape(lambda p, f, d: time_loop_dynamical_relativity(
            p, species, f, static, d, prescribed_gauge=prescribe, gravity_boundary=boundary),
            particles, fields, dynamic)
        self.assertEqual(result[2].bssn_parameters.state.lapse.shape, state.lapse.shape)

    def test_direct_current_keeps_metric_and_coordinates_on_owned_tiles(self):
        from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
        single, tiled = (coupled_example(n, particles=True) for n in (1, 2))
        currents = []
        for static, dynamic, particles, species, fields in (single, tiled):
            compiled = GR_direct_deposition.lower(
                particles, species, fields[2], fields[6], static, dynamic).compile()
            currents.append(compiled(particles, species, fields[2], fields[6], dynamic))
            if static.field_mesh.size > 1:
                gathers = [line.strip() for line in compiled.as_text().lower().splitlines()
                           if "all-gather(" in line]
                self.assertEqual(len(gathers), 0, "\n".join(gathers[:6]))
        for actual, expected in zip(currents[1], currents[0]):
            self.assertClose(tiles_to_bssn(actual, tiled[0]), tiles_to_bssn(expected, single[0]))

    def test_eager_diagnostics_accept_replicated_species_metadata(self):
        from jax.sharding import NamedSharding, PartitionSpec as P
        from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
        from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push

        static, dynamic, particles, species, fields = coupled_example(2, particles=True)
        replicated = jax.tree.map(lambda value: jax.device_put(
            value, NamedSharding(static.field_mesh, P())), species)
        dynamic = jax.jit(lambda parameters: parameters)(dynamic)
        reference = sample_energy(particles, species, fields, static, dynamic)
        actual = sample_energy(particles, replicated, fields, static, dynamic)
        for value, expected in zip(actual, reference):
            self.assertClose(value, expected)
        norms = bssn_constraint_norms(particles, replicated, fields, static, dynamic)
        self.assertTrue(all(bool(jnp.isfinite(value)) for value in norms.values()))
        current = GR_direct_deposition(particles, replicated, fields[2], fields[6], static, dynamic)
        pushed = hybrid_boris_geodesic_push(
            particles, replicated, fields[0], fields[7][2], fields[6], static, dynamic)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(value)))
                            for value in jax.tree.leaves((current, pushed))))

    def test_full_split_step_preserves_fields_geometry_and_history(self):
        single, tiled = (coupled_example(n, particles=True, crossing=True) for n in (1, 2))
        module = importlib.import_module("PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity")
        parallel = importlib.import_module("JAX_BSSN.parallel")
        serial_rhs = module.compute_bssn_rhs_with_matter
        tiled_rhs = parallel.compute_tiled_bssn_rhs_with_matter
        results, stages = [], []
        gathers = []
        for static, dynamic, particles, species, fields in (single, tiled):
            records = []

            def record_serial(state, parameters, *sources):
                rhs = serial_rhs(state, parameters, *sources)
                records.append((state, rhs))
                return rhs

            def record_tiled(state, parameters, *sources, **layout):
                rhs = tiled_rhs(state, parameters, *sources, **layout)
                records.append((state, rhs))
                return rhs

            def recorded_step(p, f, d):
                result = time_loop_dynamical_relativity(p, species, f, static, d)
                return result, tuple(records)

            step = jax.jit(recorded_step)
            with patch.object(module, "compute_bssn_rhs_with_matter", record_serial), \
                 patch.object(parallel, "compute_tiled_bssn_rhs_with_matter", record_tiled):
                lowered = step.lower(particles, fields, dynamic)
            compiled = lowered.compile()
            result, record = compiled(particles, fields, dynamic)
            jax.block_until_ready(result)
            results.append(result)
            stages.append(record)
            if static.field_mesh.size > 1:
                executable = compiled.as_text().lower()
                gathers = [line.strip() for line in executable.splitlines() if "all-gather(" in line]
                self.assertTrue("collective-permute(" in executable)
        self.assertEqual(len(stages[1]), 4)
        for actual_stage, expected_stage in zip(stages[1], stages[0]):
            for actual, expected in zip(jax.tree.leaves(actual_stage), jax.tree.leaves(expected_stage)):
                self.assertClose(gather_bssn(actual, tiled[0]), expected)
        particle_records = []
        for result in results:
            x = jnp.asarray(jax.device_get(result[0].x)).reshape(-1, 3)
            u = jnp.asarray(jax.device_get(result[0].u)).reshape(-1, 3)
            order = jnp.argsort(x[:, 0])
            particle_records.append((x[order], u[order]))
            self.assertFalse(bool(result[1][-1]))
        for actual, expected in zip(particle_records[1], particle_records[0]):
            self.assertClose(actual, expected)
        self.assertTrue(bool(jnp.any(results[1][0].u[0, 0, 0, 0, :, 0] < 0)))
        self.assertTrue(bool(jnp.any(results[1][0].u[1, 0, 0, 0, :, 0] > 0)))
        for actual, expected in zip(results[1][2].bssn_parameters.state,
                                    results[0][2].bssn_parameters.state):
            self.assertClose(gather_bssn(actual, tiled[0]), expected)
        # D, staggered B, current, and every retained Maxwell time level.
        for slot in (0, 1, 2):
            for actual, expected in zip(results[1][1][slot], results[0][1][slot]):
                self.assertClose(tiles_to_bssn(actual, tiled[0]), tiles_to_bssn(expected, single[0]))
        for slot in (0, 1, 2):
            for actual, expected in zip(results[1][1][7][slot], results[0][1][7][slot]):
                self.assertClose(tiles_to_bssn(actual, tiled[0]), tiles_to_bssn(expected, single[0]))

        self.assertEqual(len(gathers), 0, "\n".join(gathers[:6]))


if __name__ == "__main__":
    unittest.main()
