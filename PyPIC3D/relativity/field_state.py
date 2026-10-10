"""Conversions at the file boundary of the densitized static-GR solver state.

Every D/B/J the solver evolves, refreshes, or hands to the particle push is a
native density sqrt(gamma) V^i at its own Yee location: runtime slots 0/1/2,
the external D/B in slot 5, and the previous D/B in slot 7. rho, phi and the
metric keep their own conventions. Only file inputs and outputs use physical
contravariant vectors, so these converters belong in initialization and
output adapters, never inside a step.
"""


def densitize_vector(vector, samples):
    """Multiply each component by its own native metric volume factor."""
    return tuple(value * sample.sqrt_gamma for value, sample in zip(vector, samples))


def physical_vector(vector, samples):
    """Recover a physical contravariant vector from its native density."""
    return tuple(value / sample.sqrt_gamma for value, sample in zip(vector, samples))


def _convert_fields(fields, convert):
    D, B, J, rho, phi, external, metric, previous, overflow = fields
    return (convert(D, metric.D), convert(B, metric.B), convert(J, metric.D),
            rho, phi,
            (convert(external[0], metric.D), convert(external[1], metric.B)),
            metric,
            (convert(previous[0], metric.D), convert(previous[1], metric.B)),
            overflow)


def densitize_fields(fields):
    """Convert a physical file input state into runtime densities."""
    return _convert_fields(fields, densitize_vector)


def physical_fields(fields):
    """Return a physical view of runtime state for file output."""
    return _convert_fields(fields, physical_vector)
