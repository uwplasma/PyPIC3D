"""Analytic time-symmetric Oppenheimer-Snyder geometry on global C nodes.

Writes the three native ``[metric]`` arrays that os_collapse.toml loads. The
prescribed ADM mass fixes the continuum geometry; no discrete constraint solve
adjusts it to the deposited particle density. The stored conformal factor is
**W = psi**-2**, not chi, and the conformal metric is exactly the identity, so
the slice is momentarily static (A_ij = K = Gamma^i = 0, alpha = 1, beta^i = 0).

Those four zero variables stay literal ``false`` in the TOML, which creates
zeros. ``conformal_metric`` and ``lapse`` cannot be omitted the same way:
omission would create zeros and fail the positive-definite and W > 0 checks in
PyPIC3D.relativity.bssn_parameters.

Edit the globals below and run ``python initial_metric.py``. Keep them
consistent with os_collapse.toml; nothing synchronizes the two automatically.
"""

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp


# Geometry and grid. These must match os_collapse.toml.
TOTAL_MASS = 1.0
SURFACE_AREAL_RADIUS = 5.0
NX = 64
HALF_WIDTH = 8.0
OUTPUT_DIR = Path(__file__).with_name("initial_data")


def surface_isotropic_radius(total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Match the stellar surface to the exterior Schwarzschild isotropic chart."""
    return 0.5 * (surface_areal_radius - total_mass
                  + math.sqrt(surface_areal_radius * (surface_areal_radius - 2 * total_mass)))


def conformal_factor_psi(radius, total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """OS conformal factor, continuous with its first derivative at the surface."""
    surface = surface_isotropic_radius(total_mass, surface_areal_radius)
    B = total_mass / (2 * surface**3)
    A = (1 + total_mass / (2 * surface))**1.5
    interior = A / jnp.sqrt(1 + B * radius**2)
    # Keep the unused exterior branch finite at the regular stellar center.
    exterior = 1 + total_mass / (2 * jnp.maximum(radius, surface))
    return jnp.where(radius <= surface, interior, exterior)


def collocated_axis(minimum, spacing, count):
    """Physical C nodes of the collocated grid: minimum + i * spacing.

    This reproduces the per-tile coordinates JAX_BSSN hands the geometry
    callback bitwise. build_yee_grid's equivalent linspace form is
    mathematically identical but differs in the last bit, which would leave a
    few-ulp signature in the conformal factor; keep the multiply-add form.
    """
    return minimum + jnp.arange(count, dtype=jnp.float64) * spacing


def build_initial_metric(*, nx=NX, half_width=HALF_WIDTH, total_mass=TOTAL_MASS,
                         surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Return the native global C-node arrays, keyed by their [metric] names.

    Shapes follow the serial contract ``count * tile_width + (bc == conducting)``.
    The demo uses constant field boundaries, so no axis gains an endpoint plane
    and every array is exactly (nx, nx, nx).
    """
    dx = 2 * half_width / nx
    axis = collocated_axis(-half_width, dx, nx)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    radius = jnp.sqrt(x**2 + y**2 + z**2)

    one = jnp.ones((nx, nx, nx), dtype=jnp.float64)
    conformal_factor = conformal_factor_psi(radius, total_mass, surface_areal_radius)**-2
    conformal_metric = jnp.eye(3, dtype=jnp.float64)[:, :, None, None, None] * one
    return dict(conformal_metric=conformal_metric,
                conformal_factor=jnp.broadcast_to(conformal_factor, one.shape),
                lapse=one)


def save_initial_metric(output_dir, arrays):
    """Write the native .npy inputs, replacing files from an earlier run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        jnp.save(directory / f"{name}.npy", value)


def main():
    jax.config.update("jax_enable_x64", True)
    arrays = build_initial_metric()
    save_initial_metric(OUTPUT_DIR, arrays)
    factor = arrays["conformal_factor"]
    print(f"Saved initial metric in {Path(OUTPUT_DIR).resolve()}")
    print(f"  surface isotropic radius {surface_isotropic_radius():.16g}")
    print(f"  W in [{float(jnp.min(factor)):.16g}, {float(jnp.max(factor)):.16g}]")


if __name__ == "__main__":
    main()
