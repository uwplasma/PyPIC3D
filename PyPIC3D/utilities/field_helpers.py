import jax

from PyPIC3D.utilities.filters import tiled_bilinear_filter_vector, tiled_digital_filter_vector


def add_external_fields(E, B, external_fields):
    """
    Add prescribed external fields to the self-consistent fields.

    Maxwell updates should use E and B by themselves. Particle pushes and total
    energy diagnostics should use the returned totals, because those are the
    fields particles actually see.
    """
    external_E, external_B = external_fields
    total_E = tuple(e + ext_e for e, ext_e in zip(E, external_E))
    total_B = tuple(b + ext_b for b, ext_b in zip(B, external_B))
    return total_E, total_B


def filter_electric_field_for_particles(E, static_parameters, dynamic_parameters):
    """Apply the adjoint current filter to the evolved electric gather.

    The digital and bilinear filters are symmetric, so F.T = F.
    """

    current_filter = static_parameters.current_filter

    def bilinear_filtered_field(E):
        return tiled_bilinear_filter_vector(E, static_parameters)

    def digital_filtered_field(E):
        return tiled_digital_filter_vector(E, dynamic_parameters.alpha, static_parameters)

    return jax.lax.cond(
        current_filter == "bilinear",
        bilinear_filtered_field,
        lambda E: jax.lax.cond(
            current_filter == "digital",
            digital_filtered_field,
            lambda E: E,
            E,
        ),
        E,
    )


def yee_push_fields(E, B, external_fields, static_parameters, dynamic_parameters):
    """Filter the evolved electric gather, then add unfiltered external fields."""
    coupling_E = filter_electric_field_for_particles(E, static_parameters, dynamic_parameters)
    return add_external_fields(coupling_E, B, external_fields)
