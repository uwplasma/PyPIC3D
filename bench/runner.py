#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import toml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal PyPIC3D benchmark in a target repo checkout.")
    parser.add_argument("--repo", required=True, help="Path to a PyPIC3D git checkout to import")
    parser.add_argument("--config", required=True, help="Repo-relative path to a TOML config")
    parser.add_argument("--steps", type=int, default=2048, help="Number of steps to time (steady-state)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (not timed)")
    parser.add_argument("--sample-every", type=int, default=0, help="If >0, sample energies every N steps")
    parser.add_argument("--sample-steps", type=int, default=0, help="Total steps to run for sampling (0 => Nt)")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config_arg = Path(args.config)
    config_path = config_arg if config_arg.is_absolute() else (repo / args.config)
    cfg = toml.load(str(config_path))

    sys.path.insert(0, str(repo))

    import jax
    import jax.numpy as jnp
    from jax import lax, block_until_ready

    from PyPIC3D.initialization import initialize_simulation
    from PyPIC3D.utils import compute_energy

    enable_x64 = bool(cfg.get("simulation_parameters", {}).get("enable_x64", True))
    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_platform_name", "cpu")

    (
        loop,
        particles,
        fields,
        world,
        simulation_parameters,
        constants,
        plotting_parameters,
        plasma_parameters,
        solver,
        electrostatic,
        verbose,
        GPUs,
        Nt,
        curl_func,
        J_func,
        relativistic,
    ) = initialize_simulation(cfg)

    def make_advance(loop, world, constants, curl_func, J_func, solver, relativistic, simulation_parameters):
        use_scan = bool(simulation_parameters.get("use_scan", False))
        scan_chunk = int(simulation_parameters.get("scan_chunk", 256) or 256)
        step_impl = getattr(loop, "__wrapped__", None)

        def step(particles, fields):
            return loop(
                particles,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                relativistic=relativistic,
            )

        if use_scan:
            def _scan_chunk(particles, fields, *, n_steps: int):
                def body(carry, _):
                    p, f = carry
                    p, f = loop(
                        p,
                        f,
                        world,
                        constants,
                        curl_func,
                        J_func,
                        solver,
                        relativistic=relativistic,
                    )
                    return (p, f), None

                (p, f), _ = lax.scan(body, (particles, fields), xs=None, length=n_steps)
                return p, f

            scan_chunk_jit = jax.jit(_scan_chunk, donate_argnums=(0, 1), static_argnames=("n_steps",))

            def advance(particles, fields, n_steps: int):
                while n_steps > 0:
                    k = scan_chunk if n_steps >= scan_chunk else n_steps
                    particles, fields = scan_chunk_jit(particles, fields, n_steps=k)
                    n_steps -= k
                return particles, fields

            return advance, use_scan, scan_chunk

        def advance(particles, fields, n_steps: int):
            for _ in range(n_steps):
                particles, fields = step(particles, fields)
            return particles, fields

        return advance, use_scan, scan_chunk

    advance, use_scan, scan_chunk = make_advance(
        loop, world, constants, curl_func, J_func, solver, relativistic, simulation_parameters
    )

    result = {
        "repo": str(repo),
        "config": args.config,
        "enable_x64": enable_x64,
        "use_scan": use_scan,
        "scan_chunk": scan_chunk,
        "steps": int(args.steps),
    }

    if int(args.steps) > 0:
        # Warmup (compile) outside the timed region.
        particles_w, fields_w = particles, fields
        particles_w, fields_w = advance(
            particles_w, fields_w, max(0, args.warmup) * (scan_chunk if use_scan else 1)
        )
        block_until_ready(fields_w[0][0])

        # Steady-state timing.
        t0 = time.time()
        particles_t, fields_t = advance(particles_w, fields_w, int(args.steps))
        block_until_ready(fields_t[0][0])
        t1 = time.time()

        result["wall_s"] = float(t1 - t0)
        result["s_per_step"] = float((t1 - t0) / max(1, int(args.steps)))
    else:
        result["wall_s"] = 0.0
        result["s_per_step"] = 0.0

    if args.sample_every and args.sample_every > 0:

        sample_every = int(args.sample_every)
        total_steps = int(args.sample_steps) if args.sample_steps else int(Nt)

        times = []
        e_energy = []
        err = []

        # reference energy at sample start
        E, B, J, rho, *rest = fields
        e0, b0, k0 = compute_energy(particles, E, B, world, constants)
        initial_energy = e0 + b0 + k0

        p_s, f_s = particles, fields
        t = 0
        while t <= total_steps:
            E, B, J, rho, *rest = f_s
            ee, be, ke = compute_energy(p_s, E, B, world, constants)
            tot = ee + be + ke
            times.append(float(t * float(world["dt"])))
            e_energy.append(float(ee))
            err.append(float(abs(initial_energy - tot) / max(float(initial_energy), 1e-10)))

            if t == total_steps:
                break
            k = min(sample_every, total_steps - t)
            p_s, f_s = advance(p_s, f_s, k)
            t += k
        result["samples"] = {
            "dt": float(world["dt"]),
            "t": times,
            "electric_energy": e_energy,
            "energy_error": err,
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
