"""Transfers of native densities between the eight C/V metric locations."""
import jax.numpy as jnp

from .core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS

def location_interpolate(field, source_location, target_location):
    """Average a field from one C/V location to another, one axis at a time.

    Axis ``a`` of the location is array axis ``a + 3`` of a tiled field.  A
    C-to-V move averages with the upper neighbour, a V-to-C move with the
    lower one.
    """
    interpolated = field
    for axis in range(3):
        if source_location[axis] == target_location[axis]:
            continue
        shift = -1 if source_location[axis] == "C" else 1
        interpolated = 0.5 * (interpolated + jnp.roll(interpolated, shift, axis=axis + 3))
    return interpolated


def reconstruct_vector(vector, locations, target):
    """Transfer native density components ``sqrt(gamma) V^i`` to one C/V location.

    Averaging densities is the metric-weighted transfer of the physical
    vector, so the result is the density at ``target``. Dividing by the
    target ``sqrt_gamma`` recovers the physical vector there.
    """
    return tuple(
        value if source == target else location_interpolate(value, source, target)
        for value, source in zip(vector, locations)
    )


def copy_densities(copy, vector, volumes):
    """Apply a node-copying operation to densities as if to the physical vector.

    Halo exchange, constant extrapolation and horizon freezing copy a value
    from a source node to a target node. A copied density still carries the
    source volume, so each copy is rescaled by ``sqrt_gamma(target) /
    sqrt_gamma(source)``; applying the same copy to the volumes yields the
    source volume at every node. Owned nodes keep their exact values. Volumes
    may be signed (sin(theta) < 0 past a pole); nodes the copy zeroes
    (absorbing halos) have no source volume and stay zero.
    """
    copied = copy(vector)
    sources = copy(volumes)
    return tuple(
        value * jnp.where(source != 0, volume / jnp.where(source != 0, source, 1.0), 1.0)
        for value, volume, source in zip(copied, volumes, sources)
    )


def metric_at_location(metric, location):
    if location == ('C', 'C', 'C'):
        return metric.center
    if location == ('V', 'V', 'V'):
        return metric.vertex
    if location in D_FIELD_LOCATIONS:
        return metric.D[D_FIELD_LOCATIONS.index(location)]
    return metric.B[B_FIELD_LOCATIONS.index(location)]
