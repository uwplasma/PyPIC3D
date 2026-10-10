"""Second-order standard interpolation for a smooth full 3-D metric tensor."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.relativity.core import build_yee_metric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, location_grid
from PyPIC3D.relativity.field_state import densitize_vector
from PyPIC3D.solvers.GR_yee.static_metric import compute_covariant_E, compute_covariant_H


def geometry(q):
    x, y, z = q
    v = jnp.array([.3+.1*jnp.sin(x), .2+.1*jnp.cos(y), .1+.1*jnp.sin(z)])
    outer = v[:, None]*v[None, :]
    norm = jnp.dot(v, v)
    return (.9+.04*jnp.sin(x+y),
            jnp.array([.03*jnp.cos(y), .04*jnp.sin(z), .02*jnp.sin(x)]),
            jnp.eye(3)+outer, jnp.eye(3)-outer/(1+norm), jnp.sqrt(1+norm))


def vectors(q):
    x, y, z = q
    D = jnp.array([jnp.sin(x)+.2*jnp.cos(z), jnp.cos(y)+.1*jnp.sin(x), .3*jnp.cos(x+z)])
    B = jnp.array([.4*jnp.cos(y+z), .7*jnp.sin(x+z), .2*jnp.cos(x+y)])
    return D, B


def on_grid(grid, function):
    q = jnp.stack(jnp.meshgrid(*(a[0, 0, 0] for a in grid), indexing='ij'), axis=-1)
    values = jax.vmap(function)(q.reshape(-1, 3))
    return values.reshape((1, 1, 1)+q.shape[:-1]+values.shape[1:])


class TestRegularConstitutiveInterpolation(unittest.TestCase):
    def test_smooth_full_tensor_second_order_in_three_dimensions(self):
        errors = []
        for n in (8, 16, 32):
            g = 3
            c = tuple((jnp.arange(-g, n+g, dtype=float)/n)[None, None, None, :] for _ in range(3))
            v = tuple(a+.5/n for a in c)
            d = SimpleNamespace(grids=SimpleNamespace(tiled_center_grid=c, tiled_vertex_grid=v))
            m = build_yee_metric(d, geometry)
            D = tuple(on_grid(location_grid(c, v, loc), lambda q, i=i: vectors(q)[0][i])
                      for i, loc in enumerate(D_FIELD_LOCATIONS))
            B = tuple(on_grid(location_grid(c, v, loc), lambda q, i=i: vectors(q)[1][i])
                      for i, loc in enumerate(B_FIELD_LOCATIONS))
            # The constitutive kernels consume native densities sqrt(gamma) V^i.
            D, B = densitize_vector(D, m.D), densitize_vector(B, m.B)
            total = 0.
            for fields, locs, magnetic in ((compute_covariant_E(D, B, m), D_FIELD_LOCATIONS, False),
                                            (compute_covariant_H(D, B, m), B_FIELD_LOCATIONS, True)):
                for i, loc in enumerate(locs):
                    def exact(q):
                        alpha, beta, gamma, _, root = geometry(q)
                        dd, bb = vectors(q)
                        return (alpha*(gamma@(bb if magnetic else dd))
                                +(-1 if magnetic else 1)*root*jnp.cross(beta, dd if magnetic else bb))[i]
                    analytic = on_grid(location_grid(c, v, loc), exact)
                    delta = np.asarray(fields[i]-analytic)[0, 0, 0, g:-g, g:-g, g:-g]
                    total += np.mean(delta**2)
            errors.append(np.sqrt(total))
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse/fine, 3.7, errors)


if __name__ == '__main__':
    unittest.main()
