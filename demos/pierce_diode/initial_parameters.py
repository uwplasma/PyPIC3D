#!/usr/bin/env python3
"""
Convert the Pierce beam-plasma parameters used in Matsumoto, Yokoyama,
and Summers (Phys. Plasmas 3, 177-191, 1996) into SI units for a chosen
electron number density n0 and diode length L.

This script is set up for the blocking-oscillation case:
    alpha = 2.95*pi
    Nx = 256
    v_th(code) = 2
    omega_p * dt = 0.004
    particles_per_cell = 16

Paper definitions:
    alpha = L * omega_p / V0
    omega_p = sqrt(n0 * e^2 / (epsilon_0 * m_e))
    v_th^2 = k_B * T_e / m_e

For the particle simulation with Nx=256, the code velocity unit is
    v_unit = omega_p * dx = omega_p * L / Nx

Usage:
    python initial_parameters.py --n 1e15 --L 0.01

The calculated values are printed and written to the adjacent
``pierce_diode.toml`` file by default.
"""

import argparse
import math
from pathlib import Path

import toml


# SI constants
E_CHARGE = 1.602176634e-19       # C
M_E = 9.1093837139e-31           # kg
EPS0 = 8.8541878188e-12          # F/m
K_B = 1.380649e-23               # J/K
C = 299_792_458.0                # m/s


# Paper / simulation parameters
ALPHA = 2.95 * math.pi
NX = 256
VTH_CODE = 2.0
OMEGA_P_DT = 0.004
PARTICLES_PER_CELL = 16
TRANSIT_TIMES = 10.0
RANDOM_SEED = 0


def calculate_parameters(n0: float, L: float) -> dict:
    """Return the blocking-regime Pierce parameters in SI units."""

    if n0 <= 0:
        raise ValueError("Electron number density n0 must be > 0.")
    if L <= 0:
        raise ValueError("Diode length L must be > 0.")

    omega_p = math.sqrt(n0 * E_CHARGE**2 / (EPS0 * M_E))
    dx = L / NX

    # Velocity normalization corresponding to the paper's Nx=256 particle runs
    v_unit = omega_p * dx

    # Beam injection velocity from alpha = L*omega_p/V0
    V0 = L * omega_p / ALPHA
    V0_code = V0 / v_unit  # exactly NX / ALPHA

    # Paper thermal velocity, v_th(code)=2
    vth = VTH_CODE * v_unit

    # Warm-fluid drift velocity used in the paper
    vd_code = math.sqrt(V0_code**2 + VTH_CODE**2)
    vd = vd_code * v_unit

    # Temperature from v_th^2 = k_B*T_e/m_e
    Te_K = M_E * vth**2 / K_B
    Te_eV = M_E * vth**2 / E_CHARGE

    # Debye length using the paper's convention
    lambda_D = vth / omega_p

    # Timestep from omega_p * dt = 0.004
    dt = OMEGA_P_DT / omega_p

    # Beam flux/current density
    electron_flux = n0 * V0
    conventional_current_density = -E_CHARGE * electron_flux

    # Dimensionless numerical checks
    cfl_beam = V0 * dt / dx
    cfl_drift = vd * dt / dx
    debye_cells = lambda_D / dx

    # 1-D nominal macroparticle counts/rates corresponding to the paper
    initial_macroparticles = NX * PARTICLES_PER_CELL
    macro_injection_per_timestep = PARTICLES_PER_CELL * V0 * dt / dx
    transit_time = L / V0
    t_wind = TRANSIT_TIMES * transit_time
    Nt = int(t_wind / dt)
    injected_macroparticles = math.floor(Nt * macro_injection_per_timestep)
    required_particle_slots = initial_macroparticles + injected_macroparticles
    particle_tile_capacity_factor = required_particle_slots / initial_macroparticles

    return {
        "alpha": ALPHA,
        "n0": n0,
        "L": L,
        "Nx": NX,
        "dx": dx,
        "omega_p": omega_p,
        "plasma_period": 2.0 * math.pi / omega_p,
        "v_unit": v_unit,
        "vth_code": VTH_CODE,
        "vth": vth,
        "V0_code": V0_code,
        "V0": V0,
        "vd_code": vd_code,
        "vd": vd,
        "Te_K": Te_K,
        "Te_eV": Te_eV,
        "lambda_D": lambda_D,
        "debye_cells": debye_cells,
        "omega_p_dt": OMEGA_P_DT,
        "dt": dt,
        "electron_flux": electron_flux,
        "current_density": conventional_current_density,
        "particles_per_cell": PARTICLES_PER_CELL,
        "initial_macroparticles": initial_macroparticles,
        "macro_injection_per_timestep": macro_injection_per_timestep,
        "transit_time": transit_time,
        "transit_times": TRANSIT_TIMES,
        "t_wind": t_wind,
        "Nt": Nt,
        "injected_macroparticles": injected_macroparticles,
        "required_particle_slots": required_particle_slots,
        "particle_tile_capacity_factor": particle_tile_capacity_factor,
        "V0_over_c": V0 / C,
        "vth_over_c": vth / C,
        "vd_over_c": vd / C,
        "cfl_beam": cfl_beam,
        "cfl_drift": cfl_drift,
    }


def print_parameters(p: dict) -> None:
    """Pretty-print the calculated SI parameters."""

    print()
    print("=" * 72)
    print("Pierce blocking-oscillation parameters: alpha = 2.95*pi")
    print("=" * 72)

    print("\nINPUT / FIXED PARAMETERS")
    print(f"  n0                         = {p['n0']:.8e} m^-3")
    print(f"  L                          = {p['L']:.8e} m")
    print(f"  alpha                      = {p['alpha']:.10f}")
    print(f"  Nx                         = {p['Nx']}")
    print(f"  v_th (paper/code)          = {p['vth_code']:.6g}")
    print(f"  omega_p * dt               = {p['omega_p_dt']:.6g}")
    print(f"  particles/cell             = {p['particles_per_cell']}")

    print("\nGRID / PLASMA")
    print(f"  dx                         = {p['dx']:.8e} m")
    print(f"  omega_p                    = {p['omega_p']:.8e} rad/s")
    print(f"  plasma period 2*pi/omega_p = {p['plasma_period']:.8e} s")
    print(f"  velocity unit omega_p*dx   = {p['v_unit']:.8e} m/s")

    print("\nVELOCITIES")
    print(f"  V0 (code units)            = {p['V0_code']:.8f}")
    print(f"  V0                         = {p['V0']:.8e} m/s")
    print(f"  v_th                       = {p['vth']:.8e} m/s")
    print(f"  v_d (code units)           = {p['vd_code']:.8f}")
    print(f"  v_d                        = {p['vd']:.8e} m/s")

    print("\nTEMPERATURE / DEBYE LENGTH")
    print(f"  T_e                        = {p['Te_K']:.8e} K")
    print(f"  T_e                        = {p['Te_eV']:.8e} eV")
    print(f"  lambda_D                   = {p['lambda_D']:.8e} m")
    print(f"  lambda_D / dx              = {p['debye_cells']:.6f} cells")

    print("\nTIME STEP")
    print(f"  dt                         = {p['dt']:.8e} s")

    print("\nBEAM SOURCE")
    print(f"  electron number flux       = {p['electron_flux']:.8e} m^-2 s^-1")
    print(f"  conventional Jx            = {p['current_density']:.8e} A/m^2")
    print(f"  nominal initial macros     = {p['initial_macroparticles']}")
    print(f"  mean injected macros/step  = {p['macro_injection_per_timestep']:.8f}")
    print(f"  transit time L/V0          = {p['transit_time']:.8e} s")
    print(f"  requested transit times    = {p['transit_times']:.6g}")
    print(f"  requested time steps       = {p['Nt']}")
    print(f"  fixed slots per species    = {p['required_particle_slots']}")

    print("\nNUMERICAL / RELATIVITY CHECKS")
    print(f"  V0/c                       = {p['V0_over_c']:.8e}")
    print(f"  v_th/c                     = {p['vth_over_c']:.8e}")
    print(f"  v_d/c                      = {p['vd_over_c']:.8e}")
    print(f"  beam CFL V0*dt/dx          = {p['cfl_beam']:.8f}")
    print(f"  drift CFL vd*dt/dx         = {p['cfl_drift']:.8f}")

    if p["vd_over_c"] >= 0.1:
        print(
            "\nWARNING: vd >= 0.1 c. The 1996 model is nonrelativistic; "
            "this n0/L scaling may not be appropriate."
        )

    print("=" * 72)


def build_configuration(p: dict) -> dict:
    """Build the complete PyPIC3D input for the one-dimensional diode."""

    return {
        "simulation_parameters": {
            "name": "Warm Pierce diode blocking instability",
            "t_wind": p["t_wind"],
            "dt": p["dt"],
            "solver": "electrostatic",
            "Nx": p["Nx"],
            "Ny": 1,
            "Nz": 1,
            "particle_tile_nx": p["Nx"],
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
            "particle_tile_capacity_factor": p["particle_tile_capacity_factor"],
            "x_min": 0.0,
            "x_max": p["L"],
            "y_wind": 1.0,
            "z_wind": 1.0,
            "verbose": False,
            "shape_factor": 2,
            "alpha": 1.0,
            "relativistic": False,
            "particle_x_bc": "absorbing",
            "particle_y_bc": "periodic",
            "particle_z_bc": "periodic",
            "x_bc": "conducting",
            "y_bc": "periodic",
            "z_bc": "periodic",
        },
        "plotting": {
            "plotting_interval": 10,
            "plot_openpmd_particles": True,
            "plot_openpmd_fields": True,
        },
        "pierce_diode": {
            "electron_species": "electron1",
            "ion_species": "ion1",
            "alpha": p["alpha"],
            "omega_p": p["omega_p"],
            "beam_velocity": p["V0"],
            "thermal_velocity": p["vth"],
            "macro_injection_per_timestep": p["macro_injection_per_timestep"],
            "transit_times": p["transit_times"],
            "random_seed": RANDOM_SEED,
        },
        "particle1": {
            "name": "electron1",
            "N_per_cell": PARTICLES_PER_CELL,
            "number_density": p["n0"],
            "charge": -E_CHARGE,
            "mass": M_E,
            "vth": p["vth"],
            "initial_vx": p["V0"],
        },
        "particle2": {
            "name": "ion1",
            "N_per_cell": PARTICLES_PER_CELL,
            "number_density": p["n0"],
            "charge": E_CHARGE,
            "mass": 1.67262192595e-27,
            "vth": 0.0,
            "update_x": False,
            "update_y": False,
            "update_z": False,
        },
    }


def write_configuration(p: dict, config_path: str | Path) -> Path:
    """Write the generated Pierce parameters to a PyPIC3D TOML file."""

    config_path = Path(config_path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as config_file:
        toml.dump(build_configuration(p), config_file)

    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate SI parameters for the Pierce blocking-oscillation "
            "case alpha=2.95*pi."
        )
    )
    parser.add_argument(
        "--n",
        "--density",
        dest="n0",
        type=float,
        default=1e15,
        help="Electron number density n0 in m^-3, e.g. 1e15",
    )
    parser.add_argument(
        "--L",
        "--length",
        dest="L",
        type=float,
        default=0.01,
        help="Diode length L in meters, e.g. 0.01",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("pierce_diode.toml"),
        help="Generated PyPIC3D TOML path",
    )
    args = parser.parse_args()

    n0 = args.n0
    L = args.L
    params = calculate_parameters(n0, L)
    print_parameters(params)
    config_path = write_configuration(params, args.config)
    print(f"\nWrote PyPIC3D configuration: {config_path}")


if __name__ == "__main__":
    main()
