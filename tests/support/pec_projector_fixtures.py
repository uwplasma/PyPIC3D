"""Surface placement and distributed staggered PEC contracts."""
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.pec import _pec_setup, _prepare_normal_D, _project_native_nodes
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.ghost_cells import apply_tiled_pec_boundary
from PyPIC3D.relativity.core import Metric, YeeMetric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS, build_yee_metric
from PyPIC3D.relativity.field_interpolation import copy_densities, reconstruct_vector
from PyPIC3D.relativity.field_state import densitize_fields, physical_fields
from PyPIC3D.solvers.GR_yee.static_metric import compute_covariant_E, compute_covariant_H, update_B, update_D
from PyPIC3D.diagnostics.static_metric import divergence
from tests.kernel_fixtures import kernel_parameters


def constant_metric(shape, gamma=None, shift=(0., 0., 0.)):
    gamma = jnp.eye(3) if gamma is None else jnp.asarray(gamma)
    m = Metric(jnp.ones(shape), jnp.broadcast_to(jnp.asarray(shift), shape+(3,)),
               jnp.broadcast_to(gamma, shape+(3,3)),
               jnp.broadcast_to(jnp.linalg.inv(gamma), shape+(3,3)),
               jnp.full(shape, jnp.sqrt(jnp.linalg.det(gamma))))
    return YeeMetric((m,)*3, (m,)*3, m, m)


def make_setup(n=(6, 5, 1), g=2, bc=(1, 0, 0), tiles=None, gamma=None, shift=(0.,0.,0.)):
    s,d = kernel_parameters(Nx=n[0], Ny=n[1], Nz=n[2], tile_shape=tiles or n,
                            guard_cells=g, boundary_conditions=bc, solver='static_metric',
                            x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1.)
    shape = tuple(a//b for a,b in zip(n,s.tile_shape))+tuple(a+2*g for a in s.tile_shape)
    return s,d,constant_metric(shape,gamma,shift)


def coupled_metric(position):
    """Smooth periodic metric whose off-diagonal terms couple every wall pair."""
    X, Y, Z = (2*jnp.pi*value for value in position)
    gamma = jnp.array([[2.+.2*jnp.sin(X), .3+.05*jnp.cos(Y), .15+.03*jnp.sin(Z)],
                       [.3+.05*jnp.cos(Y), 3.+.1*jnp.cos(Y), -.2+.02*jnp.cos(X)],
                       [.15+.03*jnp.sin(Z), -.2+.02*jnp.cos(X), 1.5+.1*jnp.sin(Z)]])
    lapse = 1.+.03*jnp.sin(X)
    shift = jnp.array([.04*jnp.sin(X), .02*jnp.sin(Y), .01*jnp.cos(Z)])
    return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))


def coupled_setup(n, tiles, bc):
    s, d = kernel_parameters(Nx=n[0], Ny=n[1], Nz=n[2], tile_shape=tiles, boundary_conditions=bc,
                             solver='static_metric', metric='numerical',
                             x_min=0., y_min=0., z_min=0., x_wind=1., y_wind=1., z_wind=1.)
    return s, d, build_yee_metric(d, coupled_metric)


class EdgeCoupledProjectionFixtures:
    """Where two walls meet in a non-orthogonal metric, the wall rows couple."""


class StaggeredProjectorsFixtures:
    pass
