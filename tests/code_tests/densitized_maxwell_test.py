from tests.support.densitized_maxwell_runtime_fixtures import (
    signed_volume_metric,

    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    SpeciesConfig,
    TiledParticles,
    apply_tiled_supergaussian_absorber,
    build_field_output_map,
    build_yee_metric,
    coupled_setup,
    densitize_fields,
    densitize_vector,
    field_map_for_output,
    hybrid_boris_geodesic_push,
    jnp,
    np,
    owned_nodes,
    patch,
    physical_fields,
    physical_vector,
    random_vectors,
    reconstruct_vector,
    refresh_fields,
    time_loop_static_metric,
    unittest,
    update_B,
    update_D,
)


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
        def fake_update_D(DD, H, JJ, *args, absorber=None):
            self.assertIsNone(absorber)
            captured.append(JJ)
            return DD
        def transform(JJ):
            self.assert_vectors_close(JJ, new_J)
            return tuple(2*a for a in JJ)
        # Nonempty storage selects the mocked particle push/deposition branch.
        position = jnp.zeros((1, 1, 1, 1, 1, 3))
        particles = TiledParticles(position, position, jnp.ones(position.shape[:-1], dtype=bool))
        module = 'PyPIC3D.solvers.GR_yee.time_loop.'
        with patch(module+'hybrid_boris_geodesic_push', side_effect=push), \
             patch(module+'GR_direct_deposition', return_value=new_J), \
             patch(module+'refresh_tiled_particle_tiles', side_effect=lambda p,*a:(p,False)), \
             patch(module+'update_B', side_effect=lambda E,B,*a,absorber=None:B), \
             patch(module+'update_D', side_effect=fake_update_D):
            _, result = time_loop_static_metric(particles, None, state, s, d, current_transform=transform)
        filtered = tuple(2*a for a in new_J)
        self.assert_vectors_close(captured[0], tuple((a+b)/2 for a,b in zip(state[2], filtered)))
        self.assert_vectors_close(captured[1], filtered)
        self.assert_vectors_close(result[2], filtered)
        self.assertIs(result[7][0], state[0])
        self.assertIs(result[7][1], state[1])


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


if __name__ == "__main__":
    unittest.main()
