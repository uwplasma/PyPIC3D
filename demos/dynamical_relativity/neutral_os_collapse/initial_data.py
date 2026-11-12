"""Analytic, time-symmetric Oppenheimer-Snyder dust in isotropic coordinates.

The prescribed ADM mass fixes the continuum geometry. Equal particle weights
sample its proper rest-mass measure; no discrete constraint solve changes the
paper's homogeneous density profile. Returned fields and momenta share t=0.
The runner calls initialize_dynamical_relativity once to seed the leapfrog.
"""

import math
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
from scipy.stats import qmc

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_ABSORBING, BC_CONSTANT
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.parameters import build_dynamic_parameters, build_static_parameters


TOTAL_MASS = 1.0
SURFACE_AREAL_RADIUS = 5.0
LAPSE_THRESHOLD = 0.1
GUARD_CELLS = 3
PARTICLE_BATCH_SIZE = 8192


def surface_isotropic_radius(total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Match the stellar surface to the exterior Schwarzschild isotropic chart."""
    return 0.5 * (surface_areal_radius - total_mass
                  + math.sqrt(surface_areal_radius * (surface_areal_radius - 2 * total_mass)))


def total_rest_mass(total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Integrate uniform proper density; binding energy makes this exceed ADM M."""
    s = math.sqrt(2 * total_mass / surface_areal_radius)
    return 3 * total_mass * (math.asin(s) - s * math.sqrt(1 - s*s)) / (2 * s**3)


def conformal_factor_psi(radius, total_mass=TOTAL_MASS, surface_areal_radius=SURFACE_AREAL_RADIUS):
    """OS conformal factor, continuous with its first derivative at the surface."""
    surface = surface_isotropic_radius(total_mass, surface_areal_radius)
    B = total_mass / (2 * surface**3)
    A = (1 + total_mass / (2 * surface))**1.5
    interior = A / jnp.sqrt(1 + B * radius**2)
    # Keep the unused exterior branch finite at the regular stellar center.
    exterior = 1 + total_mass / (2 * jnp.maximum(radius, surface))
    return jnp.where(radius <= surface, interior, exterior)


def enclosed_rest_mass_fraction(radius, total_mass=TOTAL_MASS,
                                surface_areal_radius=SURFACE_AREAL_RADIUS):
    """Enclosed proper mass of the initial sphere, divided by total rest mass."""
    surface = surface_isotropic_radius(total_mass, surface_areal_radius)
    root_B = math.sqrt(total_mass / (2 * surface**3))
    theta_surface = math.atan(root_B * surface)
    theta = jnp.arctan(root_B * jnp.minimum(radius, surface))
    return (4 * theta - jnp.sin(4 * theta)) / (
        4 * theta_surface - math.sin(4 * theta_surface))


def load_dust(particle_count=262144):
    """Load equal-rest-mass Sobol points with exact eightfold reflection symmetry."""
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
    position = (signs[:, None, :] * octant[None, :, :]).reshape((1, 1, 1, 1, particle_count, 3))
    particles = TiledParticles(position, jnp.zeros_like(position),
                               jnp.ones(position.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.zeros(1), jnp.ones(1),
                            jnp.asarray([total_rest_mass() / particle_count]),
                            jnp.ones((1, 3), dtype=bool))
    return particles, species


def build_initial_data(*, nx=64, particle_count=262144, half_width=8.0,
                       dt=0.025, end_time=60.0, output_dir="."):
    """Return common-time particles, species, physical fields, static, dynamic."""
    jax.config.update("jax_enable_x64", True)
    length = 2 * half_width
    dx = length / nx
    config = dict(Nx=nx, Ny=nx, Nz=nx, x_wind=length, y_wind=length, z_wind=length,
                  x_min=-half_width, y_min=-half_width, z_min=-half_width,
                  dx=dx, dy=dx, dz=dx, dt=dt, C=1.0, eps=1.0, mu=1.0, kb=1.0, alpha=1.0)
    # Neutral dust has identically zero direct electric current; its physical
    # mass and stresses still enter the separate native matter deposition.
    static = build_static_parameters(dict(
        **config, name="neutral_os_collapse", output_dir=str(output_dir),
        Nt=math.ceil(end_time / dt), solver="dynamic_metric", metric="numerical",
        relativistic=True, particle_pusher="hybrid_boris_geodesic",
        current_deposition="direct", current_filter="none", shape_factor=2,
        guard_cells=GUARD_CELLS, tile_shape=(nx, nx, nx),
        particle_batch_size=PARTICLE_BATCH_SIZE, metric_backreaction=True,
        boundary_conditions=(BC_CONSTANT,) * 3, particle_boundary_conditions=(BC_ABSORBING,) * 3,
        lapse_deleting=True, lapse_threshold=LAPSE_THRESHOLD,
    ))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tiled_center, tiled_vertex = build_tiled_yee_grids(
        static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)),
    )
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tiled_center,
                           tiled_vertex_grid=tiled_vertex)
    dynamic = build_dynamic_parameters(config)
    bssn = build_bssn_parameters(static, dynamic, evolution_parameters=dict(
        gauge=1, zero_shift=0, g=0.75, eta=2.0, kappa=0.0, nu=0.002,
        xl_bc=1, xr_bc=1, yl_bc=1, yr_bc=1, zl_bc=1, zr_bc=1,
    ))

    # Nonconducting C-node allocations have nx entries: xmin through xmax-dx.
    # Use the actual native shape rather than fitting both endpoints into nx.
    axes = tuple(-half_width + jnp.arange(size) * dx for size in bssn.state.conformal_factor.shape)
    radius = jnp.sqrt(axes[0][:, None, None]**2 + axes[1][None, :, None]**2
                      + axes[2][None, None, :]**2)
    state = bssn.state._replace(conformal_factor=conformal_factor_psi(radius)**-2)
    dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))

    particles, species = load_dust(particle_count)
    zero_scalar = jnp.zeros((1, 1, 1) + (nx + 2 * GUARD_CELLS,) * 3, dtype=jnp.float64)
    zero = (zero_scalar, zero_scalar, zero_scalar)
    fields = (zero, zero, zero, zero_scalar, zero_scalar, (zero, zero), None, None, jnp.asarray(False))
    return particles, species, fields, static, dynamic
