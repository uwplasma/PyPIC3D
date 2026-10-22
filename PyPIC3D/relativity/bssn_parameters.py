"""Numerical-relativity state carried by the dynamical PIC parameters."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters as BSSNEvolutionParameters
from JAX_BSSN.bssn.variables import BSSNVariables


class BSSNParameters(NamedTuple):
    """Evolved geometry, NR controls, source strength, and coordinate time."""

    state: object
    evolution_parameters: object
    gravity_G: object
    time: object


def build_bssn_parameters(static_parameters, dynamic_parameters, *, state=None,
                          evolution_parameters=None, gravity_G=1.0, time=0.0):
    """Build flat initial geometry unless the caller supplies BSSN variables.

    The first coupling uses Cartesian geometrized units and one device. BSSN
    has one spatial spacing; unresolved periodic axes need not share it.
    The default flat state is not a matter constraint solve.
    """
    tile_counts = dynamic_parameters.grids.tiled_center_grid[0].shape[:3]
    shape = tuple(count * width + (bc == 1) for count, width, bc in zip(
        tile_counts, static_parameters.tile_shape, static_parameters.boundary_conditions))
    spacings = (dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)
    active_spacings = [float(spacing) for spacing, size in zip(spacings, shape) if size > 1]
    dx = active_spacings[0] if active_spacings else float(dynamic_parameters.dx)
    if any(abs(spacing - dx) > 1.e-12 * abs(dx) for spacing in active_spacings):
        raise ValueError("BSSN requires equal spacing on all resolved Cartesian axes")
    if any(float(value) != 1.0 for value in (dynamic_parameters.C, dynamic_parameters.eps, dynamic_parameters.mu)):
        raise ValueError("Dynamical relativity requires normalized C=1, eps=1, mu=1")
    if static_parameters.field_mesh is not None and static_parameters.field_mesh.size != 1:
        raise ValueError("Dynamical relativity currently requires one device")
    if not jax.config.x64_enabled:
        raise ValueError("Dynamical relativity requires jax_enable_x64=True")

    if state is None:
        zero = jnp.zeros(shape, dtype=jnp.float64)
        identity = jnp.broadcast_to(jnp.eye(3)[:, :, None, None, None], (3, 3) + shape)
        state = BSSNVariables(identity, jnp.ones_like(zero), jnp.zeros_like(identity),
                              zero, jnp.zeros((3,) + shape), jnp.ones_like(zero),
                              jnp.zeros((3,) + shape))
    elif state.conformal_factor.shape != shape:
        raise ValueError("BSSN state must match the physical PIC center-node allocation")

    if evolution_parameters is None or isinstance(evolution_parameters, dict):
        overrides = {} if evolution_parameters is None else evolution_parameters
        origins = tuple(axis[1] for axis in dynamic_parameters.grids.center)
        evolution_parameters = BSSNEvolutionParameters(
            dx=dx, dt=dynamic_parameters.dt, zero_shift=1, gauge=0,
            x_min=origins[0], y_min=origins[1], z_min=origins[2])
        evolution_parameters = evolution_parameters._replace(**overrides)
    if abs(float(evolution_parameters.dx) - dx) > 1.e-12 * abs(dx):
        raise ValueError("BSSN and PIC spatial spacings must agree")
    if float(evolution_parameters.dt) != float(dynamic_parameters.dt):
        raise ValueError("BSSN and PIC macro timesteps must agree")

    evolution_parameters = jax.tree.map(jnp.asarray, evolution_parameters)
    return BSSNParameters(state, evolution_parameters, jnp.asarray(gravity_G), jnp.asarray(time))
