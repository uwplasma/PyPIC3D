"""Conservative binomial filtering of conformal finite-difference sources."""
import jax.numpy as jnp
from PyPIC3D.boundary_conditions.ghost_cells import source_boundaries
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS


def smooth_conformal(value, static, passes, *, component=None):
    location = ('C',)*3 if component is None else D_FIELD_LOCATIONS[component]
    parity = tuple(-1 if axis == component else 1 for axis in range(3))
    # Radial fields have constant halos; particle absorption is a separate
    # source sink. The checked region lies beyond its filter support.
    for _ in range(passes):
        value = source_boundaries(value, static, location=location,
                                  reflecting_parity=parity, particle=False)
        for axis in (3, 4):
            value = .5*value+.25*(jnp.roll(value, 1, axis)+jnp.roll(value, -1, axis))
    return source_boundaries(value, static, location=location,
                             reflecting_parity=parity, particle=False)


def filter_current(current, static, passes):
    """Filter the densitized current with the same conformal operator as charge."""
    if passes == 0:
        return current
    return tuple(smooth_conformal(value, static, passes, component=i)
                 for i, value in enumerate(current))
