"""Short, source-driven PEC regressions in constant oblique coordinates.

D/B/J are native contravariant densities; E/H are covectors on their
respective Yee grids. No particles, metric evolution, damping, or file output.
"""
from itertools import combinations, product
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions import pec
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS, D_FIELD_LOCATIONS, build_yee_metric, location_grid,
)
from PyPIC3D.relativity.field_state import densitize_vector, physical_vector
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.solvers.GR_yee.static_metric import compute_covariant_E, compute_covariant_H, update_B, update_D
from tests.kernel_fixtures import kernel_parameters


N, GUARD, DT, STEPS, DRIVEN_STEPS = 8, 2, 0.01, 24, 12
J_MAX = 1.e-3
RAMP = np.concatenate((np.sin(np.pi*(np.arange(DRIVEN_STEPS)+0.5)/DRIVEN_STEPS)**2,
                       np.zeros(STEPS-DRIVEN_STEPS)))
FIELD_SCALE = 4*np.pi*J_MAX*DT*RAMP.sum()
PEC_TOL = 1.e-10*FIELD_SCALE
CONSTRAINT_TOL = 1.e-10*FIELD_SCALE*N
SYSTEM_TOL = 1.e-12
CELL = (0, 0, 0)+(slice(GUARD, GUARD+N),)*3
# Gauss nodes on the conductor carry surface charge, not the prescribed
# volume charge. The backward D divergence is checked at every interior C node.
CHARGE = (0, 0, 0)+(slice(GUARD+1, GUARD+N),)*3


def _parameters():
    return kernel_parameters(
        Nx=N, Ny=N, Nz=N, tile_shape=(N,)*3, guard_cells=GUARD,
        solver='static_metric', metric='numerical', boundary_conditions=(1, 1, 1),
        x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1., dt=DT,
    )


def _metric(dynamic, angle):
    cosine = 0. if angle == 90 else 0.5
    gamma = jnp.array([[1., cosine, 0.], [cosine, 1., 0.], [0., 0., 1.]])
    inverse, root = jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))
    return build_yee_metric(dynamic, lambda position: (1., jnp.zeros(3), gamma, inverse, root))


def _current(dynamic, metric):
    def potential(position):
        u = (position-0.125)/0.75
        taper = jnp.prod(jnp.where((u > 0)&(u < 1), jnp.sin(jnp.pi*u)**2, 0.))
        return taper*jnp.exp(-jnp.sum((position-0.5)**2)/(2*0.22**2))

    current = []
    for component, location in enumerate(D_FIELD_LOCATIONS):
        grid = location_grid(dynamic.grids.tiled_center_grid,
                             dynamic.grids.tiled_vertex_grid, location)
        positions = jnp.stack(jnp.meshgrid(*(a[0, 0, 0] for a in grid), indexing='ij'), -1)
        gradient = jax.vmap(jax.grad(potential))(positions.reshape(-1, 3)).reshape(positions.shape)
        current.append(jnp.sum(metric.D[component].gamma_inv[..., component, :]*gradient, -1))
    scale = J_MAX/max(float(jnp.max(jnp.abs(value))) for value in current)
    return densitize_vector(tuple(scale*value for value in current), metric.D)


def _make_step(static, dynamic):
    # Metric and current are dynamic arguments: both angles share one compilation.
    @jax.jit
    def advance(D, B, charge, amplitude, current, metric):
        E = compute_covariant_E(D, B, metric)
        B = update_B(E, B, metric, static, dynamic, DT/2)
        J = tuple(amplitude*value for value in current)
        H = compute_covariant_H(D, B, metric)
        D = update_D(D, H, J, metric, static, dynamic, DT)
        E = compute_covariant_E(D, B, metric)
        B = update_B(E, B, metric, static, dynamic, DT/2)
        charge = charge-DT*divergence(J, dynamic)
        return D, B, charge
    return advance


def _make_projection_probe(static, kind, locations):
    @jax.jit
    def project(fields, metric):
        captures = []
        original = getattr(pec, f'solve_{kind}_edges')

        def capture(first, second, *args):
            result = original(first, second, *args)
            captures.append((first, second, result))
            return result

        # Capture during tracing and return the arrays as explicit JIT outputs;
        # converting tracers to NumPy or retaining them outside this call is invalid.
        with patch.object(pec, f'solve_{kind}_edges', side_effect=capture):
            refreshed = getattr(pec, f'enforce_pec_{kind}')(fields, static, locations, metric)
        return refreshed, tuple(captures)
    return project


def _electric_trace(E):
    """Co-locate actual covariant E on physical C nodes, without ghost mixing.

    Each component has just one V axis. Use adjacent owned half nodes to
    extrapolate its physical endpoints; all other coordinates are already C.
    This is interpolation of computed E, not a second PEC projection.
    """
    components = []
    for axis, value in enumerate(E):
        owned = [slice(GUARD, GUARD+N+1)]*3
        owned[axis] = slice(GUARD, GUARD+N)
        samples = np.moveaxis(np.asarray(value)[(0, 0, 0)+tuple(owned)], axis, 0)
        trace = np.concatenate(((1.5*samples[0]-0.5*samples[1])[None],
                                0.5*(samples[:-1]+samples[1:]),
                                (1.5*samples[-1]-0.5*samples[-2])[None]), axis=0)
        components.append(np.moveaxis(trace, 0, axis))
    return np.stack(components, -1)


def _energy(density, covector, locations, spacing):
    total = 0.
    for value, lower, location in zip(density, covector, locations):
        owned = tuple(slice(GUARD, GUARD+N+(loc == 'C')) for loc in location)
        integrand = np.asarray(value)[(0, 0, 0)+owned]*np.asarray(lower)[(0, 0, 0)+owned]
        weights = []
        for loc in location:
            w = np.ones(N+(loc == 'C'))
            if loc == 'C':
                w[[0, -1]] = 0.5
            weights.append(w)
        total += np.einsum('ijk,i,j,k->', integrand, *weights)
    return total*spacing**3/(8*np.pi)


class TestPECObliqueBox(unittest.TestCase):
    def assert_metric(self, metric, dynamic, angle):
        gamma = np.asarray(metric.center.gamma)[0, 0, 0, GUARD, GUARD, GUARD]
        inverse = np.linalg.inv(gamma)
        expected = np.eye(3)
        expected[0, 1] = expected[1, 0] = 0. if angle == 90 else 0.5
        np.testing.assert_allclose(gamma, expected, rtol=0, atol=1.e-13,
                                   err_msg=f'{angle=}: incorrect oblique metric')
        self.assertGreater(np.linalg.eigvalsh(gamma).min(), 0., f'{angle=}: gamma={gamma}')
        self.assertGreater(np.linalg.det(gamma), 0., f'{angle=}: gamma={gamma}')
        self.assertGreater(abs(np.linalg.det(gamma[:2, :2])), 0., f'{angle=}: singular xy block')
        if angle == 60:
            self.assertNotEqual(gamma[0, 1], 0., '60-degree test must couple x/y')
            np.testing.assert_allclose(inverse, [[4/3, -2/3, 0], [-2/3, 4/3, 0], [0, 0, 1]],
                                       rtol=0, atol=1.e-13)
        for sample in (*metric.D, *metric.B, metric.center, metric.vertex):
            for name, expected_value in (('gamma', gamma), ('gamma_inv', inverse),
                                         ('sqrt_gamma', np.sqrt(np.linalg.det(gamma))),
                                         ('lapse', 1.), ('shift', np.zeros(3))):
                value = np.asarray(getattr(sample, name))
                np.testing.assert_allclose(value, np.broadcast_to(expected_value, value.shape),
                                           rtol=0, atol=1.e-13, err_msg=f'{angle=}: nonconstant {name}')
            identity = np.asarray(sample.gamma)@np.asarray(sample.gamma_inv)
            np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), identity.shape),
                                       rtol=0, atol=1.e-13, err_msg=f'{angle=}: metric inverse')
        tile = jax.tree.map(lambda x: x[0, 0, 0], metric.center)
        grid = tuple(axis[0, 0, 0] for axis in dynamic.grids.tiled_center_grid)
        sampled = interpolate_metric(tile, jnp.array([[.31, .43, .57], [.71, .26, .62]]),
                                     grid, 'numerical', (True,)*3, (GUARD,)*3)
        for name in ('grad_lapse', 'grad_shift', 'grad_gamma_inv'):
            np.testing.assert_allclose(getattr(sampled, name), 0., rtol=0, atol=1.e-13,
                                       err_msg=f'{angle=}: static flat metric {name}')

        zero = jnp.zeros_like(metric.center.lapse)
        D = densitize_vector((jnp.ones_like(zero), zero, zero), metric.D)
        E = compute_covariant_E(D, (zero,)*3, metric)
        for actual, expected_value in zip(E, (1., expected[1, 0], 0.)):
            np.testing.assert_allclose(actual, expected_value, rtol=0, atol=1.e-13,
                                       err_msg=f'{angle=}: constitutive x/y coupling')

    def assert_boundaries(self, E, metric, tolerance, context):
        trace = _electric_trace(E)
        inverse = np.asarray(metric.center.gamma_inv)[0, 0, 0, GUARD, GUARD, GUARD]
        indices = np.indices((N+1,)*3)
        incident = (indices == 0)|(indices == N)
        count = incident.sum(axis=0)
        for axis, side in product(range(3), (0, N)):
            # n_i = +/- delta_i^axis / sqrt(gamma^{axis,axis}).
            normal = np.eye(3)[axis]*(1 if side else -1)/np.sqrt(inverse[axis, axis])
            normal_up = inverse@normal
            tangent = trace-np.einsum('...i,i->...', trace, normal_up)[..., None]*normal
            squared = np.einsum('...i,ij,...j->...', tangent, inverse, tangent)
            residual = np.sqrt(np.maximum(squared, 0.))
            for number, kind in ((1, 'face interior'), (2, 'edge'), (3, 'corner')):
                mask = (indices[axis] == side)&(count == number)
                values = np.where(mask, residual, -np.inf)
                index = np.unravel_index(np.argmax(values), values.shape)
                error = values[index]
                self.assertTrue(np.isfinite(error) and error < tolerance,
                                f'{context}: {kind}, surface={"xyz"[axis]}={side/N}, '
                                f'index={index}, residual={error:.6e}, tol={tolerance:.6e}, '
                                f'E={trace[index]}, tangential_E={tangent[index]}')
        # The constitutive wall rows are checked at native half nodes too,
        # including the first/last half nodes coupled across each xy edge.
        for component, location in enumerate(D_FIELD_LOCATIONS):
            for axis in range(3):
                if axis == component:
                    continue
                for side in (0, N):
                    selection = [slice(GUARD, GUARD+N+(loc == 'C')) for loc in location]
                    selection[axis] = slice(GUARD+side, GUARD+side+1)
                    values = np.asarray(E[component])[(0, 0, 0)+tuple(selection)]
                    local = np.unravel_index(np.argmax(np.abs(values)), values.shape)
                    index = tuple(sl.start+i for sl, i in zip(selection, local))
                    self.assertLess(abs(values[local]), tolerance,
                                    f'{context}: native wall/edge row E_{"xyz"[component]}, '
                                    f'surface={"xyz"[axis]}={side/N}, index={index}, '
                                    f'value={values[local]:.6e}, tol={tolerance:.6e}')

    def assert_constraint(self, value, context):
        value = np.asarray(value)
        rms, maximum = np.sqrt(np.mean(value**2)), np.max(np.abs(value))
        self.assertTrue(np.isfinite(rms) and max(rms, maximum) < CONSTRAINT_TOL,
                        f'{context}: L2(RMS)={rms:.6e}, Linf={maximum:.6e}, '
                        f'tol={CONSTRAINT_TOL:.6e}')

    def test_pec_oblique_box_charging(self):
        static, dynamic = _parameters()
        advance = _make_step(static, dynamic)
        for angle in (60, 90):
            with self.subTest(angle=angle):
                metric = _metric(dynamic, angle)
                self.assert_metric(metric, dynamic, angle)
                current = _current(dynamic, metric)
                scalar = jnp.zeros_like(metric.center.lapse)
                D = B = (scalar,)*3
                charge = scalar
                energies, variations, electric_rms = [], [], []
                for step, amplitude in enumerate(RAMP, start=1):
                    context = f'{angle=}, {step=}'
                    previous = np.stack([np.asarray(v)[CELL] for v in D])
                    D, B, charge = advance(D, B, charge, jnp.asarray(amplitude), current, metric)
                    E = compute_covariant_E(D, B, metric)
                    H = compute_covariant_H(D, B, metric)
                    physical_D, physical_B = physical_vector(D, metric.D), physical_vector(B, metric.B)
                    for name, vector in (('D', physical_D), ('B', physical_B), ('E', E), ('H', H)):
                        for component, value in enumerate(vector):
                            values = np.asarray(value)
                            self.assertTrue(np.isfinite(values).all(), f'{context}: nonfinite {name}[{component}]')
                            maximum = np.max(np.abs(values))
                            self.assertLess(maximum, 10*FIELD_SCALE,
                                            f'{context}: explosive {name}[{component}], '
                                            f'max={maximum:.6e}, bound={10*FIELD_SCALE:.6e}')
                    self.assertTrue(np.isfinite(np.asarray(charge)).all(), f'{context}: nonfinite charge')
                    net = float(jnp.sum(charge[CHARGE]))/N**3
                    self.assertLess(abs(net), CONSTRAINT_TOL,
                                    f'{context}: current injected net charge={net:.6e}, tol={CONSTRAINT_TOL:.6e}')
                    self.assert_boundaries(E, metric, PEC_TOL, context)
                    self.assert_constraint((divergence(D, dynamic)-4*np.pi*charge)[CHARGE],
                                           context+' Gauss')
                    self.assert_constraint(divergence(B, dynamic, forward=True)[CELL],
                                           context+' div B')
                    ue = _energy(D, E, D_FIELD_LOCATIONS, 1/N)
                    ub = _energy(B, H, B_FIELD_LOCATIONS, 1/N)
                    self.assertGreater(ue, 0., f'{context}: nonpositive electric energy={ue}')
                    self.assertGreaterEqual(ub, 0., f'{context}: negative magnetic energy={ub}')
                    energies.append((ue, ub))
                    now = np.stack([np.asarray(v)[CELL] for v in D])
                    variations.append(np.linalg.norm(now-previous)/max(np.linalg.norm(previous), 1.e-15))
                    electric_rms.append(np.sqrt(np.mean([np.mean(np.asarray(v)[CELL]**2) for v in physical_D])))
                self.assertGreater(min(electric_rms[3:]), 1.e-4*FIELD_SCALE,
                                   f'{angle=}: source failed to excite D, RMS history={electric_rms}')
                charge_values = np.asarray(charge)[CHARGE]
                self.assertGreater(np.max(charge_values), 1.e-4*FIELD_SCALE,
                                   f'{angle=}: no positive charge separation')
                self.assertLess(np.min(charge_values), -1.e-4*FIELD_SCALE,
                                f'{angle=}: no negative charge separation')
                self.assertLess(ub, ue, f'{angle=}: final UB={ub:.6e} must be below UE={ue:.6e}')
                early, late = np.mean(variations[3:9]), np.mean(variations[-6:])
                self.assertLess(late, early, f'{angle=}: growing relative D variation, early={early}, late={late}')
                free_energy = np.sum(energies[DRIVEN_STEPS:], axis=1)
                excursion = free_energy.max()/free_energy.min()
                self.assertLess(excursion, 1.25,
                                f'{angle=}: source-free energy max/min={excursion}, limit=1.25, history={free_energy}')

    def assert_captured_systems(self, captures, final, metric, angle, kind):
        self.assertEqual(len(captures), 1, f'{angle=}, {kind=}: edge solver not called once')
        first, second, solved = captures[0]
        inverse = np.asarray(metric.center.gamma_inv)[0, 0, 0, GUARD, GUARD, GUARD]
        correction = 0.
        for p, q in combinations(range(3), 2):
            # For constant volume the D transfer has weight 2/4; the B
            # reflection has coefficient -2 and transfer weight 1/4.
            if kind == 'D':
                alpha, beta = inverse[q, p]/(2*inverse[p, p]), inverse[p, q]/(2*inverse[q, q])
            else:
                alpha, beta = -inverse[p, q]/(2*inverse[q, q]), -inverse[q, p]/(2*inverse[p, p])
            matrix = np.array([[1., -alpha], [-beta, 1.]])
            determinant, rank, condition = np.linalg.det(matrix), np.linalg.matrix_rank(matrix), np.linalg.cond(matrix)
            r = 3-p-q
            for low_p, low_q in product((True, False), repeat=2):
                pw, ph, pg = (GUARD, GUARD, GUARD-1) if low_p else (GUARD+N, GUARD+N-1, GUARD+N)
                qw, qh, qg = (GUARD, GUARD, GUARD-1) if low_q else (GUARD+N, GUARD+N-1, GUARD+N)
                def at(p_node, q_node):
                    index = [0, 0, 0, slice(None), slice(None), slice(None)]
                    index[p+3], index[q+3] = p_node, q_node
                    return tuple(index)

                if kind == 'D':
                    a, b = at(pw, qh), at(ph, qw)
                    a1, b1, a0, b0 = first[q][a], first[p][b], second[q][a], second[p][b]
                    rhs = np.stack((a1-alpha*b0, b1-beta*a0), -1)
                    solution = np.stack((solved[q][a], solved[p][b]), -1)
                    original = np.stack((a1, b1), -1)
                    # Third-wall intersections are zero constraints, not 2x2 solves.
                    for wall in (GUARD, GUARD+N):
                        np.testing.assert_allclose(solution[wall], 0., atol=SYSTEM_TOL, rtol=0,
                                                   err_msg=f'{angle=}: D corner intersection {p,q,r}, {wall=}')
                    mask = np.ones(solution.shape[0], dtype=bool)
                    mask[[GUARD, GUARD+N]] = False
                    rhs, solution, original = rhs[mask], solution[mask], original[mask]
                else:
                    c, d = at(pw, qg), at(pg, qw)
                    c1, c2, d2 = first[p][c], second[p][c], second[q][d]
                    rhs = np.stack((c2-alpha*d2, d2-beta*c1), -1)
                    solution = np.stack((solved[p][c], solved[q][d]), -1)
                    original = np.stack((c2, d2), -1)
                    # Later edge pairs overwrite intersecting exterior slabs.
                    # The final corner sweep, not this intermediate return,
                    # is responsible for the triple-exterior ghost equations.
                    active = slice(GUARD, GUARD+N)  # the third B coordinate is V
                    rhs, solution, original = rhs[active], solution[active], original[active]
                context = (f'{angle=}, {kind=} edge={"xyz"[p]}{"xyz"[q]}, low={low_p,low_q}, '
                           f'A={matrix}, det={determinant}, rank={rank}, cond={condition}, '
                           f'solution={solution}, rhs={rhs}')
                self.assertTrue(np.isfinite(matrix).all() and np.isfinite(solution).all()
                                and np.isfinite(rhs).all(), context)
                self.assertTrue(abs(determinant) > SYSTEM_TOL and rank == 2 and condition < 1.e4, context)
                reference = np.linalg.solve(matrix, rhs.T).T
                np.testing.assert_allclose(solution, reference, rtol=SYSTEM_TOL, atol=SYSTEM_TOL, err_msg=context)
                np.testing.assert_allclose(solution@matrix.T, rhs, rtol=SYSTEM_TOL, atol=SYSTEM_TOL, err_msg=context)
                if kind == 'B':
                    # Independently reconstruct each completed reflection's RHS
                    # from the three other interpolation samples. The fourth
                    # sample is the other unknown in this 2x2 system. Include
                    # the entire third axis, especially both corner ghost slabs.
                    pi, qi = pw+(1 if low_p else -1), qw+(1 if low_q else -1)
                    rhs_final = np.stack((
                        final[p][at(pw, qh)]+alpha*(final[q][at(ph, qw)]
                            +final[q][at(pg, qi)]+final[q][at(ph, qi)]),
                        final[q][at(ph, qw)]+beta*(final[p][at(pw, qh)]
                            +final[p][at(pi, qg)]+final[p][at(pi, qh)])), -1)
                    solution_final = np.stack((final[p][c], final[q][d]), -1)
                    final_context = (context+'; completed edge/corner reflections, '
                                     f'final solution={solution_final}, final rhs={rhs_final}')
                    self.assertTrue(np.isfinite(solution_final).all() and np.isfinite(rhs_final).all(), final_context)
                    np.testing.assert_allclose(solution_final, np.linalg.solve(matrix, rhs_final.T).T,
                                               rtol=SYSTEM_TOL, atol=SYSTEM_TOL, err_msg=final_context)
                    np.testing.assert_allclose(solution_final@matrix.T, rhs_final,
                                               rtol=SYSTEM_TOL, atol=SYSTEM_TOL, err_msg=final_context)
                if (p, q) == (0, 1):
                    correction = max(correction, np.max(np.abs(solution-original)))
        if angle == 60:
            self.assertGreater(correction, 1.e-6, f'{kind=}: xy coupled solve had no measurable effect')
        else:
            self.assertLess(correction, SYSTEM_TOL, f'{kind=}: Cartesian edge correction={correction}')

    def test_pec_oblique_coupled_systems(self):
        static, dynamic = _parameters()
        projectors = {kind: _make_projection_probe(static, kind, locations)
                      for kind, locations in (('D', D_FIELD_LOCATIONS), ('B', B_FIELD_LOCATIONS))}
        for angle in (60, 90):
            with self.subTest(angle=angle):
                metric = _metric(dynamic, angle)
                shape = metric.center.lapse.shape
                zero = (jnp.zeros(shape),)*3
                for kind, locations, amplitudes in (
                        ('D', D_FIELD_LOCATIONS, (1., 2., 3.)),
                        ('B', B_FIELD_LOCATIONS, (.4, -.2, .3))):
                    fields = tuple(jnp.full(shape, value) for value in amplitudes)
                    refreshed, captures = projectors[kind](fields, metric)
                    captures = jax.tree.map(np.asarray, captures)
                    self.assert_captured_systems(captures, tuple(np.asarray(v) for v in refreshed),
                                                 metric, angle, kind)
                    twice = refresh_fields(refreshed, static, locations, kind, metric)
                    for component, (value, again) in enumerate(zip(refreshed, twice)):
                        np.testing.assert_allclose(again, value, rtol=SYSTEM_TOL, atol=SYSTEM_TOL,
                                                   err_msg=f'{angle=}, {kind=}, {component=}: non-idempotent PEC')
                    if kind == 'D':
                        E = compute_covariant_E(refreshed, zero, metric)
                        self.assert_boundaries(E, metric, SYSTEM_TOL, f'{angle=}, manufactured D')
                    # z is orthogonal to x/y: its last reflection must have
                    # exact parity even in triple-exterior corner ghosts.
                    corner_magnitude = 0.
                    for sides in product((False, True), repeat=3):
                        for component, location in enumerate(locations):
                            index = [GUARD-1 if not high else GUARD+N+(loc == 'C')
                                     for high, loc in zip(sides, location)]
                            owner = index.copy()
                            owner[2] = (GUARD+(location[2] == 'C') if not sides[2] else GUARD+N-1)
                            sign = (1 if component == 2 else -1)*(1 if kind == 'D' else -1)
                            value = float(refreshed[component][(0, 0, 0)+tuple(index)])
                            expected = sign*float(refreshed[component][(0, 0, 0)+tuple(owner)])
                            corner_magnitude = max(corner_magnitude, abs(value))
                            np.testing.assert_allclose(value, expected, rtol=SYSTEM_TOL, atol=SYSTEM_TOL,
                                                       err_msg=f'{angle=}, {kind=}: corner={sides}, z surface, '
                                                       f'{component=}, index={index}, value={value}, expected={expected}')
                    self.assertGreater(corner_magnitude, 1.e-6, f'{angle=}, {kind=}: vacuous corner reflection probe')


if __name__ == '__main__':
    unittest.main()
