"""Regression contracts for persistent Maxwell densities and physical adapters."""
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.ownership import owned_nodes
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.diagnostics.output_adapters import build_field_output_map, field_map_for_output
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, build_yee_metric
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from PyPIC3D.relativity.field_state import (
    densitize_vector, physical_vector, densitize_fields, physical_fields,
)
from PyPIC3D.solvers.GR_yee.static_metric import compute_covariant_E, compute_covariant_H, update_D, update_B
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.support.pec_projector_fixtures import coupled_setup


def random_vectors(metric):
    rng = np.random.default_rng(2026)
    return [tuple(jnp.asarray(rng.normal(size=metric.center.lapse.shape))
                  for _ in range(3)) for _ in range(3)]


def curl(vector, dynamic, forward):
    """Independent cyclic finite-difference reference."""
    spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
    def derivative(value, axis):
        shift = -1 if forward else 1
        return (jnp.roll(value, shift, axis+3)-value)*(1 if forward else -1)/spacing[axis]
    return tuple(derivative(vector[(i+2)%3], (i+1)%3)
                 - derivative(vector[(i+1)%3], (i+2)%3) for i in range(3))


class TestDensitizedMaxwell(unittest.TestCase):
    def assert_vectors_close(self, actual, expected, tolerance=3e-13):
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)

    def test_state_and_output_use_each_native_volume(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, J = random_vectors(m)
        scalar = jnp.zeros_like(D[0])
        original = (D, B, J, scalar, scalar, (D, B), m, (B, D), False)
        state = densitize_fields(original)
        for slot, samples in ((0, m.D), (1, m.B), (2, m.D)):
            self.assert_vectors_close(state[slot], tuple(
                value*sample.sqrt_gamma for value, sample in zip(original[slot], samples)))
        # External D/B are densities too, each on its own field's volumes.
        for density, value, samples in zip(state[5], original[5], (m.D, m.B)):
            self.assert_vectors_close(density, tuple(v*sample.sqrt_gamma for v, sample in zip(value, samples)))
        restored = physical_fields(state)
        for slot in (0, 1, 2):
            self.assert_vectors_close(restored[slot], original[slot])
        for slot in (5, 7):
            for a, b in zip(restored[slot], original[slot]):
                self.assert_vectors_close(a, b)
        output = build_field_output_map(state, None, None, s, d)
        for label, expected in zip(('E', 'B', 'J'), (D, B, J)):
            self.assert_vectors_close(output[label], expected)
        # The assembled output also converts before gathering the tiled mesh.
        assembled = field_map_for_output(output, s)
        for output_vector, expected in zip((assembled[name] for name in ("E", "B", "J")), (D, B, J)):
            for a, b in zip(output_vector, expected):
                np.testing.assert_allclose(a, b[0, 0, 0, 2:-2, 2:-2, 2:-2], atol=3e-13)

    def test_constitutive_fields_match_physical_reconstruction(self):
        _, _, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        density_D, density_B = densitize_vector(D, m.D), densitize_vector(B, m.B)
        for magnetic, locations, samples, compute in (
                (False, D_FIELD_LOCATIONS, m.D, compute_covariant_E),
                (True, B_FIELD_LOCATIONS, m.B, compute_covariant_H)):
            expected = []
            for i, (location, sample) in enumerate(zip(locations, samples)):
                # metric-weighted transfer of the physical vector to this location
                dd = jnp.stack(reconstruct_vector(density_D, D_FIELD_LOCATIONS, location), axis=-1)/sample.sqrt_gamma[..., None]
                bb = jnp.stack(reconstruct_vector(density_B, B_FIELD_LOCATIONS, location), axis=-1)/sample.sqrt_gamma[..., None]
                lower = jnp.sum(sample.gamma[..., i, :]*(bb if magnetic else dd), axis=-1)
                cross = jnp.cross(sample.shift, dd if magnetic else bb)[..., i]
                expected.append(sample.lapse*lower + (-1 if magnetic else 1)*sample.sqrt_gamma*cross)
            self.assert_vectors_close(compute(density_D, density_B, m), expected)

    def test_updates_match_physical_equations_and_boundaries(self):
        # Non-orthogonal conducting edges on the complete physical domain.
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (1, 1, 0))
        D, B, J = random_vectors(m)
        dt = .007
        for magnetic, locations, samples, initial, auxiliary in (
                (False, D_FIELD_LOCATIONS, m.D, D, B),
                (True, B_FIELD_LOCATIONS, m.B, B, D)):
            derivatives = curl(auxiliary, d, magnetic)
            expected = tuple(jnp.where(owned_nodes(value.shape, locations[i], s),
                value + dt*((-1 if magnetic else 1)*derivatives[i]/samples[i].sqrt_gamma
                            - (0 if magnetic else 4*jnp.pi*J[i])), value)
                for i, value in enumerate(initial))
            expected = refresh_fields(densitize_vector(expected, samples), s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B(auxiliary, density, m, s, d, dt) if magnetic else
                      update_D(density, auxiliary, densitize_vector(J, m.D), m, s, d, dt))
            self.assert_vectors_close(physical_vector(actual, samples), physical_vector(expected, samples), 2e-12)

    def test_vacuum_curls_preserve_density_divergences(self):
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        D = refresh_fields(densitize_vector(D, m.D), s, D_FIELD_LOCATIONS, 'D', m)
        B = refresh_fields(densitize_vector(B, m.B), s, B_FIELD_LOCATIONS, 'B', m)
        # Auxiliary fields must also have periodic, communicated halos.
        auxiliary = refresh_fields(random_vectors(m)[2], s, D_FIELD_LOCATIONS)
        zero = tuple(jnp.zeros_like(a) for a in D)
        next_D = update_D(D, auxiliary, zero, m, s, d, .001)
        auxiliary_B = refresh_fields(random_vectors(m)[2], s, B_FIELD_LOCATIONS)
        next_B = update_B(auxiliary_B, B, m, s, d, .001)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        for before, after, forward in ((D, next_D, False), (B, next_B, True)):
            error = divergence(after, d, forward=forward)-divergence(before, d, forward=forward)
            np.testing.assert_allclose(error[interior], 0., rtol=0, atol=1e-11)

    def test_horizon_and_absorber_keep_physical_boundary_policy(self):
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (3, 1, 0))
        s = s._replace(horizon_field_cells=2, supergaussian_active=True,
                       supergaussian_layers=((0, 1, 3, 4., 10.),))
        D, B, _ = random_vectors(m)
        zero = tuple(jnp.zeros_like(a) for a in D)
        for magnetic, locations, samples, initial in (
                (False, D_FIELD_LOCATIONS, m.D, D),
                (True, B_FIELD_LOCATIONS, m.B, B)):
            # the absorber damps the physical field, then the boundaries refresh its density
            expected = apply_tiled_supergaussian_absorber(initial, s, d, .03, locations=locations)
            expected = refresh_fields(densitize_vector(expected, samples), s, locations, 'B' if magnetic else 'D', m)
            density = densitize_vector(initial, samples)
            actual = (update_B(zero, density, m, s, d, .03) if magnetic else
                      update_D(density, zero, zero, m, s, d, .03))
            self.assert_vectors_close(physical_vector(actual, samples), physical_vector(expected, samples))

    def test_new_volume_does_not_rescale_stored_density_in_zero_rhs(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        density_D, density_B = densitize_vector(D, m.D), densitize_vector(B, m.B)
        # A spatial metric scaled by four has an eightfold volume factor.
        def scale(sample):
            return sample._replace(gamma=sample.gamma*4, gamma_inv=sample.gamma_inv/4,
                                   sqrt_gamma=sample.sqrt_gamma*8)
        changed = m._replace(D=tuple(map(scale, m.D)), B=tuple(map(scale, m.B)),
                             center=scale(m.center), vertex=scale(m.vertex))
        zero = tuple(jnp.zeros_like(a) for a in D)
        next_D = update_D(density_D, zero, zero, changed, s, d, .01)
        next_B = update_B(zero, density_B, changed, s, d, .01)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        for actual, original in ((next_D, density_D), (next_B, density_B)):
            self.assert_vectors_close(tuple(a[interior] for a in actual),
                                      tuple(a[interior] for a in original))
        self.assert_vectors_close(tuple(a[interior] for a in physical_vector(next_D, changed.D)),
                                  tuple(a[interior]/8 for a in D))

    def test_esirkepov_source_preserves_gauss_in_curved_metric(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        s = s._replace(current_deposition='esirkepov', current_filter='none')
        d = d._replace(dt=.02)
        x = jnp.array([.43, .47, .5]).reshape(1, 1, 1, 1, 1, 3)
        old = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], dtype=bool))
        new = old._replace(x=x+jnp.array([.01, -.02, 0.]))
        species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1),
                                jnp.array([[True, True, False]]))
        scalar = jnp.zeros_like(m.center.sqrt_gamma)
        zero = (scalar,)*3
        current = Esirkepov_current(old, new, species, zero, s, d)
        updated = update_D(zero, zero, current, m, s, d, d.dt)
        rho_old = compute_rho(old, species, scalar, s, d)
        rho_new = compute_rho(new, species, scalar, s, d)
        drift = divergence(updated, d)-4*jnp.pi*(rho_new-rho_old)
        interior = (slice(None),)*3+(slice(3, -3),)*3
        np.testing.assert_allclose(drift[interior], 0., rtol=0., atol=1e-11)

    def test_time_loop_centers_densitized_current_and_pushes_densities(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, J = random_vectors(m)
        scalar = jnp.zeros_like(D[0])
        state = densitize_fields((D, B, J, scalar, scalar, (D, B), m, (D, B), False))
        new_J = tuple(a*.25 for a in state[2])
        captured = []
        def push(particles, species, DD, BB, *args):
            # evolved plus external densities; the push divides by the node volumes
            self.assert_vectors_close(DD, tuple(2*a for a in state[0]))
            self.assert_vectors_close(BB, tuple(2*a for a in state[1]))
            return particles, particles
        def fake_update_D(DD, H, JJ, *args):
            captured.append(JJ)
            return DD
        def transform(JJ):
            self.assert_vectors_close(JJ, new_J)
            return tuple(2*a for a in JJ)
        module = 'PyPIC3D.solvers.GR_yee.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push', side_effect=push), \
             patch(module+'GR_direct_deposition', return_value=new_J), \
             patch(module+'refresh_tiled_particle_tiles', side_effect=lambda p,*a:(p,False)), \
             patch(module+'update_B', side_effect=lambda E,B,*a:B), \
             patch(module+'update_D', side_effect=fake_update_D):
            _, result = time_loop_static_metric(None, None, state, s, d, current_transform=transform)
        filtered = tuple(2*a for a in new_J)
        self.assert_vectors_close(captured[0], tuple((a+b)/2 for a,b in zip(state[2], filtered)))
        self.assert_vectors_close(captured[1], filtered)
        self.assert_vectors_close(result[2], filtered)
        self.assertIs(result[7][0], state[0])
        self.assertIs(result[7][1], state[1])



def signed_volume_metric(position):
    """Identity gamma with a volume that changes sign along periodic y but never vanishes on a node."""
    volume = (1.+.5*position[0])*jnp.sin(2*jnp.pi*(position[1]+.1))
    return 1., jnp.zeros(3), jnp.eye(3), jnp.eye(3), volume


class TestNativeBoundaries(unittest.TestCase):
    """Density refreshes obey the physical boundary policy of each copied node."""

    def test_constant_extrapolation_copies_the_physical_field(self):
        """BC_CONSTANT x ghosts hold sqrt_gamma(ghost) times the copied owner's physical value."""
        s, d, coupled = coupled_setup((16, 8, 1), (16, 8, 1), (3, 0, 0))
        g, n = s.guard_cells, s.tile_shape[0]
        # sqrt_gamma varies along x in both metrics; the second also flips sign along y.
        for name, m in (('coupled', coupled), ('signed', build_yee_metric(d, signed_volume_metric))):
            for kind, locations, samples in (('D', D_FIELD_LOCATIONS, m.D), ('B', B_FIELD_LOCATIONS, m.B)):
                with self.subTest(metric=name, field=kind):
                    density = densitize_vector(random_vectors(m)[0], samples)
                    refreshed = refresh_fields(density, s, locations, kind, m)
                    physical, original = physical_vector(refreshed, samples), physical_vector(density, samples)
                    transverse = (slice(g, -g), slice(g, -g))
                    for component, location in enumerate(locations):
                        owned = owned_nodes(density[component].shape, location, s)
                        np.testing.assert_array_equal(jnp.where(owned, refreshed[component], 0.),
                                                      jnp.where(owned, density[component], 0.))
                        # Copy the first and last owned planes into the global ghosts.
                        for tile, ghosts, source in ((0, slice(0, g), g), (0, slice(g+n, None), g+n-1)):
                            ghost = physical[component][(tile, 0, 0, ghosts)+transverse]
                            owner = original[component][(tile, 0, 0, slice(source, source+1))+transverse]
                            np.testing.assert_allclose(ghost, jnp.broadcast_to(owner, ghost.shape), rtol=1e-13, atol=0)

    def test_horizon_freeze_holds_the_reference_plane_physical_value(self):
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (3, 0, 0))
        s = s._replace(horizon_field_cells=2)
        g = s.guard_cells
        reference = g+2
        for kind, locations, samples in (('D', D_FIELD_LOCATIONS, m.D), ('B', B_FIELD_LOCATIONS, m.B)):
            density = densitize_vector(random_vectors(m)[0], samples)
            physical = physical_vector(refresh_fields(density, s, locations, kind, m), samples)
            original = physical_vector(density, samples)
            for component in range(3):
                # the low ghosts copy the first frozen layer, so they hold the plane too
                frozen = physical[component][0, 0, 0, :reference, g:-g, g:-g]
                plane = original[component][0, 0, 0, reference:reference+1, g:-g, g:-g]
                np.testing.assert_allclose(frozen, jnp.broadcast_to(plane, frozen.shape),
                                           rtol=1e-13, atol=0, err_msg=f'{kind}{component}')

    def test_coupled_walls_remove_physical_tangential_D(self):
        """Each wall row holds D^i gamma^aa = gamma^ia D^a (i != a) for the physical D.

        Both rows of a wall vanish exactly when the tangential covariant D does.
        The D^y rows beside the x/y edge are the coupled edge solve.
        """
        s, d, m = coupled_setup((16, 8, 1), (16, 8, 1), (1, 1, 0))
        g = s.guard_cells
        density = densitize_vector(random_vectors(m)[0], m.D)
        refreshed = refresh_fields(density, s, D_FIELD_LOCATIONS, 'D', m)
        # (tile, wall node): the single tile owns both ends of each axis.
        walls = {0: ((0, g), (0, g+16)), 1: ((0, g), (0, g+8))}
        for component, location in enumerate(D_FIELD_LOCATIONS):
            sample = m.D[component]
            physical = [value/sample.sqrt_gamma for value in reconstruct_vector(refreshed, D_FIELD_LOCATIONS, location)]
            for axis in (0, 1):
                if component == axis:
                    continue
                inverse = sample.gamma_inv
                row = physical[component]*inverse[..., axis, axis] - inverse[..., component, axis]*physical[axis]
                other = 1-axis
                # D^z nodes on the other wall are edge nodes with a two-normal row
                span = slice(g, g+s.tile_shape[other]) if location[other] == 'V' else slice(g+1, g+s.tile_shape[other])
                for tile, wall in walls[axis]:
                    index = [tile, 0, 0, None, None, g]
                    index[3+axis], index[3+other] = wall, span
                    np.testing.assert_allclose(row[tuple(index)], 0., rtol=0, atol=1e-12,
                                               err_msg=f'D^{"xyz"[component]} on {"xy"[axis]} wall {tile, wall}')

    def test_push_gathers_the_physical_field_from_densities(self):
        s, d, m = coupled_setup((8, 6, 1), (8, 6, 1), (0, 0, 0))
        D, B, _ = random_vectors(m)
        x = jnp.array([.43, .47, .5]).reshape(1, 1, 1, 1, 1, 3)
        particles = TiledParticles(x, jnp.full_like(x, .1), jnp.ones(x.shape[:-1], dtype=bool))
        species = SpeciesConfig(jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.array([[True, True, False]]))
        # Unit volumes at the D/B samples make the push see the supplied physical field directly.
        def unit(sample):
            return sample._replace(sqrt_gamma=jnp.ones_like(sample.sqrt_gamma))
        unit_volume = m._replace(D=tuple(map(unit, m.D)), B=tuple(map(unit, m.B)))
        expected, _ = hybrid_boris_geodesic_push(particles, species, D, B, unit_volume, s, d)
        actual, _ = hybrid_boris_geodesic_push(particles, species, densitize_vector(D, m.D),
                                               densitize_vector(B, m.B), m, s, d)
        self.assertGreater(float(jnp.max(jnp.abs(actual.u-particles.u))), 1e-3)
        np.testing.assert_allclose(actual.u, expected.u, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(actual.x, expected.x, rtol=1e-13, atol=1e-15)

if __name__ == '__main__':
    unittest.main()
