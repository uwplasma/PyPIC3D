"""Constraint-solved, periodic warm electron-ion two-stream initial data.

Gaussian units have c = k_B = 1; the default electron mass and |charge| are 1.
Temperatures are k_B T, not T/m, and NUMBER_DENSITY is the total electron
(and ion) number density
per initial coordinate volume. The proper box volume is normalized to the
coordinate volume by the metric solve. The conformally flat periodic matter
constraints require nonzero mean curvature; we choose its expanding branch.
No homogeneous source is subtracted.

Particles and physical fields returned here are at a common time. Native
PyPIC3D initialization seeds the leapfrog histories once after this solve.
Edit the globals below and run python initial_data.py to write the .npy inputs.
Keep the grid, species masses/charges/counts/weights, gravity, and filtering
settings in your existing two_stream.toml consistent with these globals.
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
    from initial_metric import MAX_NEWTON_ITERATIONS, VOLUME_TOLERANCE, solve_initial_metric
else:
    from .initial_metric import MAX_NEWTON_ITERATIONS, VOLUME_TOLERANCE, solve_initial_metric

from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.diagnostics.constraints import compute_constraint_norms
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.particles.velocity_distribution import maxwell_juttner_momentum
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_to_tiles,
    build_bssn_metric,
    tiles_to_bssn,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_electromagnetic_energy_momentum,
    compute_energy_momentum_tensor,
    compute_particle_energy_momentum,
    electromagnetic_moments,
)
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.relativity.filtering import filter_gr_grid, particle_metric
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid
from PyPIC3D.utilities.parameters import (
    build_dynamic_parameters,
    build_static_parameters,
)


# Plasma, perturbation, and numerical controls. Change these to define a run.
NX = 128
PARTICLES_PER_CELL = 64                    # Per electron beam; ions have twice as many.
NUMBER_DENSITY = 100.0 / (4.0 * math.pi)    # omega_pe = 10 for the default mass and charge.
MASS = 1.0                               # Electron/beam particle mass m_e.
CHARGE_TO_MASS_RATIO = 1.0                 # Nonnegative |q_e|/m_e; q_i = -q_e.
BEAM_SPEED = 0.5                          # Local orthonormal bulk speed, in units of c.
ELECTRON_TEMPERATURE = 0.0025
ION_TEMPERATURE = 0.0025
MASS_RATIO = 1.0                         # m_i/m_e; ion mass = MASS_RATIO * MASS.
WAVE_NUMBER = 1.0
MODE_NUMBER = 1
PERTURBATION = 0.01                       # x_e = q - epsilon sin(k q)/k.
GRAVITY_G = 8.16e-8
SEED = 42
CFL = 0.2
END_TIME = 25.0                          # Electron plasma periods, T_pe = 2*pi/omega_pe.
GUARD_CELLS = 3
PARTICLE_BATCH_SIZE = 8192
KO_DISSIPATION = 0.002
CURRENT_FILTER = "bilinear"               # Matched source and particle-gather filter.
CURRENT_FILTER_ALPHA = 1.0                 # Central weight for the digital stencil.
OUTPUT_DIR = Path(__file__).with_name("initial_data")


def build_parameters(nx, length, end_time, cfl, gravity_G, output_dir,
                     *, current_filter=CURRENT_FILTER, filter_alpha=CURRENT_FILTER_ALPHA):
    """One periodic tile with one resolved spatial axis and 3D momenta."""
    dx = length / nx
    nt = max(1, math.ceil(end_time / (cfl * dx)))
    dt = end_time / nt
    config = dict(
        Nx=nx, Ny=1, Nz=1, x_wind=length, y_wind=1.0, z_wind=1.0,
        x_min=0.0, y_min=0.0, z_min=0.0, dx=dx, dy=1.0, dz=1.0, dt=dt,
        C=1.0, eps=1.0, mu=1.0, kb=1.0, alpha=filter_alpha,
    )
    static = build_static_parameters(dict(
        **config, name="full_relativistic_two_stream", output_dir=str(output_dir),
        Nt=nt, solver="dynamic_metric", metric="numerical", relativistic=True,
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter=current_filter, shape_factor=2, guard_cells=GUARD_CELLS,
        tile_shape=(nx, 1, 1), particle_batch_size=PARTICLE_BATCH_SIZE,
        boundary_conditions=(0, 0, 0), particle_boundary_conditions=(0, 0, 0),
    ))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tiled_center, tiled_vertex = build_tiled_yee_grids(
        static, SimpleNamespace(**config, grids=SimpleNamespace(center=center, vertex=vertex)),
    )
    config["grids"] = dict(center=center, vertex=vertex, tiled_center_grid=tiled_center,
                           tiled_vertex_grid=tiled_vertex)
    dynamic = build_dynamic_parameters(config)
    bssn = build_bssn_parameters(
        static, dynamic, gravity_G=gravity_G,
        evolution_parameters=dict(gauge=0, zero_shift=1, nu=KO_DISSIPATION,
                                  xl_bc=0, xr_bc=0, yl_bc=0, yr_bc=0, zl_bc=0, zr_bc=0),
    )
    return static, dynamic._replace(bssn_parameters=bssn)


def load_plasma(nx, particles_per_cell, length, wave_number, perturbation,
                number_density, beam_speed, electron_temperature, ion_temperature,
                mass_ratio, seed, *, mass=MASS, charge_to_mass_ratio=CHARGE_TO_MASS_RATIO):
    """Load quiet positions and mirrored local orthonormal warm momenta.

    Matching electron positions with opposite momenta cancel gravitational
    momentum exactly. Ions have thermal +/- pairs at the same unperturbed
    positions. Every particle moves during the subsequent evolution.
    """
    ion_mass = mass_ratio * mass
    charge_magnitude = charge_to_mass_ratio * mass
    count = nx * particles_per_cell
    q = (jnp.arange(count, dtype=jnp.float64) + 0.5) * length / count
    electron_x = jnp.mod(q - perturbation * jnp.sin(wave_number * q) / wave_number, length)
    electron_key, sign_key, ion_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    thermal = jnp.stack(maxwell_juttner_momentum(
        count, mass, electron_temperature, electron_temperature, electron_temperature,
        1.0, electron_key,
    ), axis=-1)
    gamma_thermal = jnp.sqrt(1.0 + jnp.sum(thermal**2, axis=-1))

    # Equal-time drifting Juttner sampling needs the momentum Jacobian
    # 1 + beta*u_x/Gamma before the Lorentz boost. Bias the sign of |u_x|
    # in an isotropic draw; this implements that factor without rejection.
    abs_ux = jnp.abs(thermal[:, 0])
    positive_probability = 0.5 * (1.0 + beam_speed * abs_ux / gamma_thermal)
    sign = jnp.where(jax.random.uniform(sign_key, (count,), dtype=jnp.float64)
                     < positive_probability, 1.0, -1.0)
    thermal = thermal.at[:, 0].set(sign * abs_ux)
    beam = thermal.at[:, 0].set(
        (thermal[:, 0] + beam_speed * gamma_thermal) / math.sqrt(1.0 - beam_speed**2),
    )
    ion = jnp.stack(maxwell_juttner_momentum(
        count, ion_mass, ion_temperature, ion_temperature, ion_temperature,
        1.0, ion_key,
    ), axis=-1)
    ion_pairs = jnp.stack((ion, -ion), axis=1).reshape(2 * count, 3)

    # A common species capacity keeps the existing tiled particle layout.
    # Electron slots above count are inactive, not extra physical particles.
    x = jnp.zeros((1, 1, 1, 3, 2 * count, 3), dtype=jnp.float64)
    u = jnp.zeros_like(x)
    active = jnp.zeros(x.shape[:-1], dtype=bool)
    for species_index, direction in ((0, 1.0), (1, -1.0)):
        x = x.at[0, 0, 0, species_index, :count, 0].set(electron_x)
        u = u.at[0, 0, 0, species_index, :count].set(direction * beam)
        active = active.at[0, 0, 0, species_index, :count].set(True)
    x = x.at[0, 0, 0, 2, :, 0].set(jnp.repeat(q, 2))
    u = u.at[0, 0, 0, 2].set(ion_pairs)
    active = active.at[0, 0, 0, 2].set(True)
    weight = number_density * length / (2 * count)
    species = SpeciesConfig(
        charge_magnitude * jnp.array([-1.0, -1.0, 1.0]), jnp.array([mass, mass, ion_mass]),
        jnp.full((3,), weight), jnp.ones((3, 3), dtype=bool),
    )
    return TiledParticles(x, u, active), species


def covariant_particle_momenta(particles, metric, static, dynamic):
    """Map fixed orthonormal u_hat to covariant u using the pusher's metric."""
    sampled = interpolate_metric(
        jax.tree.map(lambda value: value[0, 0, 0], particle_metric(metric, static, dynamic)),
        particles.x.reshape(-1, 3),
        tuple(axis[0, 0, 0] for axis in dynamic.grids.tiled_center_grid),
        static.metric, (True, False, False), (static.guard_cells,) * 3,
        derivatives=False,
    )
    scale = jnp.sqrt(sampled.gamma[:, 0, 0]).reshape(particles.active.shape)
    u = jnp.where(particles.active[..., None], particles.u * scale[..., None], 0.0)
    return particles._replace(u=u)


def build_initial_data(*, nx=NX, particles_per_cell=PARTICLES_PER_CELL,
                       end_time=None, cfl=CFL, wave_number=WAVE_NUMBER,
                       mode_number=MODE_NUMBER, perturbation=PERTURBATION,
                       number_density=NUMBER_DENSITY, beam_speed=BEAM_SPEED,
                       electron_temperature=ELECTRON_TEMPERATURE,
                       ion_temperature=ION_TEMPERATURE, mass_ratio=MASS_RATIO,
                       mass=MASS, charge_to_mass_ratio=CHARGE_TO_MASS_RATIO,
                       gravity_G=GRAVITY_G, seed=SEED, output_dir=".",
                       current_filter=CURRENT_FILTER,
                       filter_alpha=CURRENT_FILTER_ALPHA,
                       constraint_tolerance=None,
                       max_newton_iterations=MAX_NEWTON_ITERATIONS):
    """Return particles, species, physical fields, static, dynamic.

    With end_time=None, convert the global END_TIME from electron plasma
    periods to code time using the supplied density, mass, and charge. An
    explicit end_time is in code units, including for a neutral plasma.

    Everything is at t=0. Do not evolve this state until the production
    initialize_dynamical_relativity routine has seeded its leapfrog histories.
    """
    jax.config.update("jax_enable_x64", True)
    if nx < 8 or particles_per_cell < 1 or mode_number < 1 or int(mode_number) != mode_number:
        raise ValueError("Use nx>=8, particles_per_cell>=1, and a positive integer mode_number")
    if 2 * mode_number >= nx:
        raise ValueError("The seeded wave must be below the grid Nyquist mode")
    if wave_number <= 0 or number_density <= 0 or gravity_G < 0 or cfl <= 0:
        raise ValueError("The wave number, density, and CFL must be positive; gravity_G>=0")
    if abs(perturbation) >= 1 or abs(beam_speed) >= 1 or mass_ratio <= 0:
        raise ValueError("The displacement map must be monotone, |beam_speed|<1, and mass_ratio>0")
    if mass <= 0 or charge_to_mass_ratio < 0:
        raise ValueError("Particle mass must be positive and |charge|/mass nonnegative")
    if electron_temperature < 0 or ion_temperature < 0:
        raise ValueError("Physical electron and ion temperatures must be nonnegative")

    length = 2.0 * math.pi * mode_number / wave_number
    charge_magnitude = charge_to_mass_ratio * mass
    omega_pe = math.sqrt(4.0 * math.pi * number_density * charge_magnitude**2 / mass)
    if end_time is None:
        if omega_pe == 0.0:
            raise ValueError("Plasma periods require nonzero charge; supply end_time in code units for neutral plasma")
        end_time = END_TIME * 2.0 * math.pi / omega_pe
    if end_time <= 0:
        raise ValueError("The duration must be positive")

    static, dynamic = build_parameters(nx, length, end_time, cfl, gravity_G, output_dir,
                                       current_filter=current_filter, filter_alpha=filter_alpha)
    particles, species = load_plasma(
        nx, particles_per_cell, length, wave_number, perturbation, number_density,
        beam_speed, electron_temperature, ion_temperature, mass_ratio, seed,
        mass=mass, charge_to_mass_ratio=charge_to_mass_ratio,
    )
    flat = dynamic.bssn_parameters.state
    flat_metric = build_bssn_metric(flat, static, dynamic)
    zero = tuple(jnp.zeros_like(component.lapse) for component in flat_metric.D)

    # Solve the native backward-difference Gauss law for densitized D_x.
    charge = compute_rho(particles, species, zero[0], static, dynamic)
    owned_charge = tiles_to_bssn(charge, static)
    if abs(float(jnp.mean(owned_charge))) > 1.0e-12 * number_density * charge_magnitude:
        raise RuntimeError("Periodic Gauss solve requires globally neutral particle loading")
    electric_x = 4.0 * jnp.pi * dynamic.dx * jnp.cumsum(owned_charge[:, 0, 0])
    electric_x = electric_x - jnp.mean(electric_x)
    displacement = (bssn_to_tiles(electric_x[:, None, None], static), zero[1], zero[2])
    magnetic = zero
    moments = electromagnetic_moments(displacement, magnetic, static)
    particle_energy = compute_particle_energy_momentum(
        particles, species, flat_metric, static._replace(current_filter="none"), dynamic,
    )[0]
    electromagnetic_energy = compute_electromagnetic_energy_momentum(moments, flat)[0]
    state, _, _, tolerance = solve_initial_metric(
        dynamic.bssn_parameters, particle_energy, electromagnetic_energy, omega_pe,
        constraint_tolerance=constraint_tolerance, max_newton_iterations=max_newton_iterations,
        static=static, dynamic=dynamic,
    )
    dynamic = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(state=state))
    metric = build_bssn_metric(state, static, dynamic)
    particles = covariant_particle_momenta(particles, metric, static, dynamic)

    # Re-evaluate the actual production sources after coordinate conversion.
    rho, momentum, _ = compute_energy_momentum_tensor(
        particles, species, moments, state, static, dynamic,
    )
    W = state.conformal_factor
    predicted_rho = W**3 * filter_gr_grid(particle_energy + W * electromagnetic_energy,
                                         static, dynamic)
    source_error = float(jnp.max(jnp.abs(rho - predicted_rho)))
    if not bool(jnp.all(jnp.isfinite(rho))) or source_error > 1.0e-12 * float(jnp.max(rho)):
        raise RuntimeError("Final production energy density disagrees with the initial metric source")
    violations = compute_all_constraints_with_matter(
        state, dynamic.bssn_parameters.evolution_parameters, gravity_G * rho, gravity_G * momentum,
    )
    constraints = {name: float(value) for name, value in compute_constraint_norms(violations).items()}
    constraints["gauss_linf"] = float(jnp.max(jnp.abs(tiles_to_bssn(
        divergence(displacement, dynamic) - 4.0 * jnp.pi * charge, static,
    ))))
    constraints["div_b_linf"] = float(jnp.max(jnp.abs(tiles_to_bssn(
        divergence(magnetic, dynamic, forward=True), static,
    ))))
    volume_error = float(jnp.abs(jnp.mean(state.conformal_factor**-3) - 1.0))
    if (not all(math.isfinite(value) for value in constraints.values())
            or constraints["hamiltonian_linf"] > tolerance
            or constraints["momentum_linf"] > tolerance
            or constraints["gauss_linf"] > 1.0e-11 * omega_pe**2
            or volume_error > VOLUME_TOLERANCE):
        raise RuntimeError(f"Initial data failed production constraint verification: {constraints}")

    physical_D = tuple(value / component.sqrt_gamma for value, component in zip(displacement, metric.D))
    physical_B = tuple(value / component.sqrt_gamma for value, component in zip(magnetic, metric.B))
    fields = (physical_D, physical_B, zero, charge, zero[0], (zero, zero), metric, None, jnp.asarray(False))
    return particles, species, fields, static, dynamic


def save_initial_data(output_dir, particles, fields, static, dynamic):
    """Write the 28 native .npy inputs, replacing files from an earlier run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # Strip inactive padding, retaining each species' exact quiet-start order.
    # Native initial_v* files hold covariant u_i, not three-velocities.
    for index, name in enumerate(("electron_plus", "electron_minus", "ion")):
        active = particles.active[:, :, :, index, :]
        position = particles.x[:, :, :, index, :, :][active]
        momentum = particles.u[:, :, :, index, :, :][active]
        for component, axis in enumerate("xyz"):
            jnp.save(directory / f"{name}_initial_{axis}.npy", position[:, component])
            jnp.save(directory / f"{name}_initial_v{axis}.npy", momentum[:, component])

    # Save owned physical Yee samples without guards.
    for vector, name in zip(fields[:2], ("D", "B")):
        for component, axis in zip(vector, "xyz"):
            jnp.save(directory / f"physical_{name}_{axis}.npy", tiles_to_bssn(component, static))

    state = dynamic.bssn_parameters.state
    for name in ("conformal_metric", "conformal_factor", "trace_K", "lapse"):
        jnp.save(directory / f"{name}.npy", getattr(state, name))


def main():
    particles, _, fields, static, dynamic = build_initial_data(
        nx=NX, particles_per_cell=PARTICLES_PER_CELL, cfl=CFL,
        wave_number=WAVE_NUMBER, mode_number=MODE_NUMBER, perturbation=PERTURBATION,
        number_density=NUMBER_DENSITY, beam_speed=BEAM_SPEED,
        electron_temperature=ELECTRON_TEMPERATURE, ion_temperature=ION_TEMPERATURE,
        mass_ratio=MASS_RATIO, mass=MASS, charge_to_mass_ratio=CHARGE_TO_MASS_RATIO,
        gravity_G=GRAVITY_G, seed=SEED,
        current_filter=CURRENT_FILTER, filter_alpha=CURRENT_FILTER_ALPHA,
    )
    save_initial_data(OUTPUT_DIR, particles, fields, static, dynamic)
    print(f"Saved initial arrays in {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
