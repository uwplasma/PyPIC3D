"""Generate neutral warm two-stream particle files for dark_two_stream.toml."""

import argparse
from pathlib import Path

import numpy as np
import toml


def warm_particles(n, length, beam_speed, electron_vth, ion_vth,
                   seed=42, seed_mode=4, seed_fraction=0.01, C=2.99792458e8):
    """Co-located charges, mirrored thermal velocities, and a common current seed."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    x[:, 0] = length * ((np.arange(n) + 0.5) / n - 0.5)
    thermal = rng.normal(0.0, electron_vth, (n, 3))
    thermal -= thermal.mean(axis=0)
    velocity = thermal.copy()
    velocity[:, 0] += beam_speed
    perturbation = seed_fraction * beam_speed * np.sin(2 * np.pi * seed_mode * x[:, 0] / length)
    forward, backward = velocity.copy(), -velocity.copy()
    forward[:, 0] += perturbation
    backward[:, 0] += perturbation

    # Two ions at each electron position cancel charge locally. Opposite
    # ion velocities cancel their initial current without making ions cold.
    ion_thermal = rng.normal(0.0, ion_vth, (n, 3))
    populations = {
        "electron1": (x.copy(), forward),
        "electron2": (x.copy(), backward),
        "ion1": (np.concatenate((x, x)), np.concatenate((ion_thermal, -ion_thermal))),
    }
    for name, (_, v) in populations.items():
        if np.any(np.sum(v*v, axis=1) >= C**2):
            raise ValueError(f"{name}: the specified Gaussian loading produced a superluminal particle")
    return populations


def generate_initial_data(config):
    """Write the six arrays per species to the paths in the native solver TOML.

    Paths are relative to the working directory, matching PyPIC3D's loader.
    Velocities are unshifted: the solver supplies its usual half-step startup.
    """
    simulation, loading = config["simulation_parameters"], config["initial_data"]
    electrons = [config["particle1"], config["particle2"]]
    ions = config["particle3"]
    n = electrons[0]["N_particles"]
    if electrons[1]["N_particles"] != n or ions["N_particles"] != 2*n:
        raise ValueError("Neutral paired loading requires N electrons in each beam and 2*N ions")
    charges = np.array([p["charge"] * p["weight"] for p in [*electrons, ions]])
    if not np.isclose(charges[0], charges[1], rtol=1e-14, atol=0) or not np.isclose(
            charges[0] + charges[1] + 2*charges[2], 0, atol=1e-14*np.max(np.abs(charges))):
        raise ValueError("Macroparticle charges must match between beams and cancel the two co-located ions")
    if electrons[0]["vth"] != electrons[1]["vth"]:
        raise ValueError("Mirrored electron beams require the same thermal speed")
    if not 0 < loading["seed_mode"] < simulation["Nx"]/2:
        raise ValueError("The seeded mode must be positive and below the grid's Nyquist mode")

    populations = warm_particles(
        n, simulation["x_wind"], loading["beam_speed"], electrons[0]["vth"], ions["vth"],
        loading["seed"], loading["seed_mode"], loading["seed_fraction"], simulation["C"])
    for block, (x, v) in zip([*electrons, ions], populations.values()):
        for axis, coordinate in enumerate("xyz"):
            for label, values in (("initial_" + coordinate, x[:, axis]),
                                  ("initial_v" + coordinate, v[:, axis])):
                path = Path(block[label])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
        print(f"{block['name']}: wrote {len(x)} particles, weight = {block['weight']:.12g}")
    return populations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dark_two_stream.toml"))
    args = parser.parse_args()
    generate_initial_data(toml.load(args.config))


if __name__ == "__main__":
    main()
