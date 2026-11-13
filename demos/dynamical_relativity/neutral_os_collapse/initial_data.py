"""Quiet-start neutral dust for the Oppenheimer-Snyder collapse demo.

Equal particle weights sample the proper rest-mass measure of the analytic
geometry written by initial_metric.py; no discrete constraint solve changes the
paper's homogeneous density profile. The star starts at rest, so every momentum
file is exactly zero. Native ``initial_v*`` files hold covariant u_i, which for
a momentarily static slice is the same zero.

Edit the globals below and run ``python initial_data.py`` to write the six .npy
inputs. Keep the particle count and the printed weight consistent with
os_collapse.toml; nothing synchronizes the two automatically.
"""

import math
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
from scipy.stats import qmc

if __package__ in (None, ""):
    from initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS, surface_isotropic_radius
else:
    from .initial_metric import SURFACE_AREAL_RADIUS, TOTAL_MASS, surface_isotropic_radius


PARTICLE_COUNT = 262144
SPECIES_NAME = "dust"
OUTPUT_DIR = Path(__file__).with_name("initial_data")


def total_rest_mass(total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Integrate uniform proper density; binding energy makes this exceed ADM M."""
    s = math.sqrt(2 * total_mass / surface_areal_radius)
    return 3 * total_mass * (math.asin(s) - s * math.sqrt(1 - s*s)) / (2 * s**3)


def enclosed_rest_mass_fraction(radius, total_mass=TOTAL_MASS,
                                surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Enclosed proper mass of the initial sphere, divided by total rest mass."""
    surface = surface_isotropic_radius(total_mass, surface_areal_radius)
    root_B = math.sqrt(total_mass / (2 * surface**3))
    theta_surface = math.atan(root_B * surface)
    theta = jnp.arctan(root_B * jnp.minimum(radius, surface))
    return (4 * theta - jnp.sin(4 * theta)) / (
        4 * theta_surface - math.sin(4 * theta_surface))


def macroparticle_weight(particle_count=PARTICLE_COUNT):
    """Equal weight whose sum is the total proper rest mass. Goes in the TOML."""
    return total_rest_mass() / particle_count


def load_dust(particle_count=PARTICLE_COUNT):
    """Return (particle_count, 3) Sobol positions with eightfold reflection symmetry.

    An unscrambled Sobol sequence on one octant is inverted for the radius by
    bisecting the enclosed-mass relation, then reflected through all eight sign
    combinations. The reflection makes the loaded cloud exactly centrally
    symmetric, so it carries no net momentum or dipole.
    """
    octant_count = particle_count // 8
    if particle_count != 8 * octant_count or octant_count < 1 or octant_count & (octant_count - 1):
        raise ValueError("OS quiet loading needs eight times a power-of-two particle count")
    sample = jnp.asarray(qmc.Sobol(3, scramble=False).random_base2(octant_count.bit_length() - 1))
    sample = sample + 0.5 / octant_count
    # Midpoints avoid duplicate particles at the origin and on octant planes.
    surface = surface_isotropic_radius()
    root_B = math.sqrt(TOTAL_MASS / (2 * surface**3))
    theta_surface = math.atan(root_B * surface)
    target = sample[:, 0] * (4 * theta_surface - math.sin(4 * theta_surface))

    def bisect(_, bracket):
        lower, upper = bracket
        middle = 0.5 * (lower + upper)
        below = 4 * middle - jnp.sin(4 * middle) < target
        return jnp.where(below, middle, lower), jnp.where(below, upper, middle)

    lower, upper = jax.lax.fori_loop(
        0, 48, bisect, (jnp.zeros_like(target), jnp.full_like(target, theta_surface)),
    )
    radius = jnp.tan(0.5 * (lower + upper)) / root_B
    cosine = sample[:, 1]
    sine = jnp.sqrt(1 - cosine**2)
    phi = 0.5 * jnp.pi * sample[:, 2]
    octant = radius[:, None] * jnp.stack((sine * jnp.cos(phi), sine * jnp.sin(phi), cosine), axis=-1)
    signs = jnp.asarray(tuple(product((-1.0, 1.0), repeat=3)))
    return (signs[:, None, :] * octant[None, :, :]).reshape((particle_count, 3))


def save_initial_data(output_dir, positions, *, name=SPECIES_NAME):
    """Write the six native 1-D .npy inputs the TOML names.

    All six are mandatory: the distributed particle loader refuses to fall back
    to built-in random generation, because changing random keys per process
    would silently change the physical initial condition.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    momentum = jnp.zeros(positions.shape[0], dtype=positions.dtype)
    for component, axis in enumerate("xyz"):
        jnp.save(directory / f"{name}_initial_{axis}.npy", positions[:, component])
        jnp.save(directory / f"{name}_initial_v{axis}.npy", momentum)


def main():
    jax.config.update("jax_enable_x64", True)
    positions = load_dust(PARTICLE_COUNT)
    save_initial_data(OUTPUT_DIR, positions)
    weight = macroparticle_weight(PARTICLE_COUNT)
    print(f"Saved {PARTICLE_COUNT} dust particles in {Path(OUTPUT_DIR).resolve()}")
    print(f"  total rest mass   {total_rest_mass():.16g}")
    print(f"  weight (for TOML) {weight:.17g}")
    print(f"  max radius        {float(jnp.max(jnp.linalg.norm(positions, axis=-1))):.16g}")


if __name__ == "__main__":
    main()
