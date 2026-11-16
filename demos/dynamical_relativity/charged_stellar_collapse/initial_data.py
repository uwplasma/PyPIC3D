"""Uniform lattice of charged dust with a Cartesian Hamiltonian solve.

Edit the globals and run python initial_data.py. Keep the grid and the printed
particle count/mass/charge/weight synchronized with charged_star.toml.
Density is uniform per coordinate volume; proper density and ADM mass follow
from the solved geometry. Gaussian units have G=c=1. All inputs are at t=0;
native initialization seeds the leapfrog histories once. Initial_v* store u_i.
"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

if __package__ in (None, ""):
    from initial_metric import BOUNDARY_TOLERANCE, metric_residual, solve_initial_metric
else:
    from .initial_metric import BOUNDARY_TOLERANCE, metric_residual, solve_initial_metric

from JAX_BSSN.evolution.spatial_derivatives import diff1_physical
from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import bssn_to_tiles, tiles_to_bssn, build_bssn_metric
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_particle_energy_momentum, compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor, electromagnetic_moments,
)
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters


TOTAL_REST_MASS = 1.0
SPHERE_RADIUS = 10.0                       # Coordinate radius, not areal radius.
MASS = 1.0
CHARGE_TO_MASS = 0.2
PARTICLE_SPACING = 0.25                    # Gives 268096 particles at R=10.
NX = 64
HALF_WIDTH = 16.0
DT = 0.05
OUTPUT_DIR = Path(__file__).with_name("initial_data")


def build_parameters(nx, half_width, output_dir):
    """One Cartesian tile with the demo's native boundary and gauge settings."""
    dx = 2 * half_width / nx
    config = dict(Nx=nx, Ny=nx, Nz=nx, dx=dx, dy=dx, dz=dx, dt=DT,
                  x_min=-half_width, y_min=-half_width, z_min=-half_width,
                  x_wind=2*half_width, y_wind=2*half_width, z_wind=2*half_width,
                  C=1.0, eps=1.0, mu=1.0, kb=1.0)
    static = build_static_parameters(dict(
        **config, name="charged_stellar_collapse", output_dir=str(output_dir), Nt=3000,
        solver="dynamic_metric", metric="numerical", relativistic=True,
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        current_filter="none", shape_factor=2, guard_cells=3, tile_shape=(nx, nx, nx),
        particle_batch_size=8192, boundary_conditions=(3, 3, 3),
        particle_boundary_conditions=(2, 2, 2), metric_backreaction=True,
        lapse_deleting=True, lapse_threshold=0.1))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tiled_center, tiled_vertex = build_tiled_yee_grids(
        static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)))
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tiled_center,
                           tiled_vertex_grid=tiled_vertex)
    dynamic = build_dynamic_parameters(config)
    bssn = build_bssn_parameters(static, dynamic, evolution_parameters=dict(
        gauge=1, zero_shift=0, eta=2.0, g=0.75, kappa=0.0, nu=0.002,
        xl_bc=1, xr_bc=1, yl_bc=1, yr_bc=1, zl_bc=1, zr_bc=1))
    return static, dynamic._replace(bssn_parameters=bssn)


def load_dust(radius=SPHERE_RADIUS, spacing=PARTICLE_SPACING,
              total_rest_mass=TOTAL_REST_MASS, mass=MASS, charge_to_mass=CHARGE_TO_MASS):
    """Equal-weight, half-cell-centered Cartesian lattice clipped to a sphere."""
    half_count = math.ceil(radius / spacing)
    axis = (jnp.arange(2 * half_count, dtype=jnp.float64) - half_count + 0.5) * spacing
    positions = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    positions = positions[jnp.sum(positions**2, axis=-1) < radius**2]
    count = positions.shape[0]
    x = positions.reshape(1, 1, 1, 1, count, 3)
    particles = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.asarray([mass * charge_to_mass]), jnp.asarray([mass]),
                            jnp.asarray([total_rest_mass / (mass * count)]),
                            jnp.ones((1, 3), dtype=bool))
    return particles, species


def sphere_displacement(coordinates, radius, charge):
    """Analytic densitized D^i for uniform coordinate charge density."""
    r = jnp.sqrt(sum(x*x for x in coordinates))
    scale = charge / jnp.maximum(r, radius)**3
    return tuple(scale * x for x in coordinates)


def build_initial_data(*, nx=NX, half_width=HALF_WIDTH, radius=SPHERE_RADIUS,
                       particle_spacing=PARTICLE_SPACING, total_rest_mass=TOTAL_REST_MASS,
                       mass=MASS, charge_to_mass=CHARGE_TO_MASS, output_dir=".",
                       constraint_tolerance=None):
    """Return particles, species, physical fields, static, dynamic, diagnostics."""
    jax.config.update("jax_enable_x64", True)
    static, dynamic = build_parameters(nx, half_width, output_dir)
    # Leave the deposited sphere clear of the outer constraint boundary.
    if radius + 2 * float(dynamic.dx) >= half_width:
        raise ValueError("The sphere and its deposition stencil must fit inside the metric boundary")
    particles, species = load_dust(radius, particle_spacing, total_rest_mass, mass, charge_to_mass)
    flat = dynamic.bssn_parameters.state
    flat_metric = build_bssn_metric(flat, static, dynamic)
    zero = tuple(jnp.zeros_like(component.lapse) for component in flat_metric.D)
    charge = compute_rho(particles, species, zero[0], static, dynamic)

    # Evaluate each component at its own Yee face, then use the runtime halos.
    axis = -half_width + jnp.arange(nx) * dynamic.dx
    total_charge = total_rest_mass * charge_to_mass
    displacement = []
    for component in range(3):
        coordinates = jnp.meshgrid(*(axis + (0.5 * dynamic.dx if d == component else 0.0)
                                    for d in range(3)), indexing="ij")
        value = sphere_displacement(coordinates, radius, total_charge)[component]
        displacement.append(bssn_to_tiles(value, static))
    displacement = update_tiled_ghost_cells(tuple(displacement), static, static.guard_cells)
    moments = electromagnetic_moments(displacement, zero, static)
    particle_energy = compute_particle_energy_momentum(particles, species, flat_metric, static, dynamic)[0]
    electromagnetic_energy = compute_electromagnetic_energy_momentum(moments, flat)[0]

    state, history, boundary_history, tolerance = solve_initial_metric(
        dynamic.bssn_parameters, particle_energy, electromagnetic_energy,
        constraint_tolerance=constraint_tolerance)
    dynamic = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(state=state))
    metric = build_bssn_metric(state, static, dynamic)
    rho, momentum, _ = compute_energy_momentum_tensor(particles, species, moments, state, static, dynamic)
    predicted = state.conformal_factor**3 * particle_energy + state.conformal_factor**4 * electromagnetic_energy
    source_error = float(jnp.max(jnp.abs(rho - predicted)))
    if not bool(jnp.all(jnp.isfinite(rho))) or source_error > 1.0e-12 * max(float(jnp.max(jnp.abs(rho))), 1.0e-30):
        raise RuntimeError("Production energy density disagrees with the Newton source")
    H, robin, boundary = metric_residual(jnp.log(state.conformal_factor), dynamic.bssn_parameters,
                                        particle_energy, electromagnetic_energy)
    h_norm = float(jnp.max(jnp.where(boundary, 0.0, jnp.abs(H))))
    boundary_error = float(jnp.max(jnp.where(boundary, jnp.abs(robin), 0.0)))
    if h_norm > tolerance or boundary_error > BOUNDARY_TOLERANCE:
        raise RuntimeError("Initial data failed the final Hamiltonian/Robin check")
    gauss = tiles_to_bssn(divergence(displacement, dynamic) - 4*jnp.pi*charge, static)

    # ADM surface integral on the finite outer cube. Trapezoidal face weights
    # count shared edges/corners correctly. Exterior EM energy is not included.
    psi = state.conformal_factor**-0.5
    mass_flux = 0.0
    weights = jnp.ones(nx).at[0].set(0.5).at[-1].set(0.5)
    for d in range(3):
        derivative = diff1_physical(psi, d, dynamic.bssn_parameters.evolution_parameters)
        flux = jnp.take(derivative, -1, axis=d) - jnp.take(derivative, 0, axis=d)
        mass_flux = mass_flux + jnp.sum(flux * weights[:, None] * weights[None, :]) * dynamic.dx**2
    adm_mass = float(-mass_flux / (2*jnp.pi))
    # Six coordinate-axis surface samples give a simple spherical radius estimate.
    middle = nx // 2
    surface_psi = []
    for d in range(3):
        ray = jnp.moveaxis(psi, d, 0)[:, middle, middle]
        surface_psi.extend((jnp.interp(radius, axis, ray), jnp.interp(-radius, axis, ray)))
    diagnostics = dict(total_mass=adm_mass, total_charge=total_charge, total_rest_mass=total_rest_mass,
                       coordinate_radius=radius, surface_areal_radius=float(radius*jnp.mean(jnp.asarray(surface_psi)**2)),
                       hamiltonian_linf=h_norm, hamiltonian_boundary_linf=float(jnp.max(jnp.where(boundary, jnp.abs(H), 0.0))),
                       robin_linf=boundary_error, source_linf=source_error,
                       momentum_source_linf=float(jnp.max(jnp.abs(momentum))),
                       gauss_linf=float(jnp.max(jnp.abs(gauss))),
                       gauss_interior_linf=float(jnp.max(jnp.abs(gauss[1:-1, 1:-1, 1:-1]))),
                       newton_history=jnp.asarray(history), robin_history=jnp.asarray(boundary_history),
                       constraint_tolerance=tolerance)
    physical_D = tuple(value / component.sqrt_gamma for value, component in zip(displacement, metric.D))
    fields = (physical_D, zero, zero, charge, zero[0], (zero, zero), metric, None, jnp.asarray(False))
    return particles, species, fields, static, dynamic, diagnostics


def save_initial_data(output_dir, particles, fields, static, dynamic, diagnostics):
    """Write native inputs and the plotter's mass/charge metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for component, axis in enumerate("xyz"):
        jnp.save(directory / f"charged_dust_initial_{axis}.npy", particles.x[..., component].reshape(-1))
        jnp.save(directory / f"charged_dust_initial_v{axis}.npy", particles.u[..., component].reshape(-1))
        jnp.save(directory / f"physical_D_{axis}.npy", tiles_to_bssn(fields[0][component], static))
    for name in ("conformal_factor", "conformal_metric", "lapse"):
        jnp.save(directory / f"{name}.npy", getattr(dynamic.bssn_parameters.state, name))
    # Keep the filename consumed by the existing RN plotter; no radial solve
    # or radial profiles are needed. total_mass is a finite-cube ADM estimate.
    jnp.savez(directory / "radial_profile.npz", **diagnostics)


def main():
    particles, species, fields, static, dynamic, diagnostics = build_initial_data()
    save_initial_data(OUTPUT_DIR, particles, fields, static, dynamic, diagnostics)
    print(f"Saved initial arrays in {OUTPUT_DIR}")
    print(f"Particles={particles.active.size}, mass={float(species.mass[0]):.17g}, "
          f"charge={float(species.charge[0]):.17g}, weight={float(species.weight[0]):.17g}")
    for name, value in diagnostics.items():
        if "history" not in name:
            print(f"  {name}: {value:.8g}")
    print("ADM mass is estimated on the finite outer cube; Gauss error includes sphere sampling and field boundaries.")


if __name__ == "__main__":
    main()
