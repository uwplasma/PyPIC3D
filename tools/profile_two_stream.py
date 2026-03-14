#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import toml


def _maybe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a PyPIC3D config with JAX/Perfetto traces.")
    parser.add_argument("--config", required=True, help="Path to a PyPIC3D TOML config")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps to execute under tracing")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps to compile before tracing")
    parser.add_argument("--out", default=None, help="Output directory (default: <output_dir>/profile)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    toml_file = toml.load(str(config_path))

    import jax
    from jax import block_until_ready
    from jax import lax
    import jax.profiler

    from PyPIC3D.initialization import initialize_simulation

    sim_params = toml_file.get("simulation_parameters", {}) or {}
    enable_x64 = bool(sim_params.get("enable_x64", True))
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
    ) = initialize_simulation(toml_file)
    dt = float(world["dt"])
    nt = int(world["Nt"])
    use_scan = bool(simulation_parameters.get("use_scan", False))
    scan_chunk = int(simulation_parameters.get("scan_chunk", 256) or 256)

    out_dir = Path(args.out) if args.out else Path(simulation_parameters["output_dir"]) / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

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

        for _ in range(max(0, args.warmup)):
            particles, fields = scan_chunk_jit(particles, fields, n_steps=scan_chunk)
            block_until_ready(fields[0][0])
    else:
        # Warmup to pay compilation cost outside the trace.
        for _ in range(max(0, args.warmup)):
            particles, fields = loop(
                particles,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                relativistic=relativistic,
            )
            block_until_ready(fields[0][0])

    # Best-effort HLO dump (step or scan chunk).
    try:
        if use_scan:
            lowered = scan_chunk_jit.lower(particles, fields, n_steps=scan_chunk)
        else:
            lowered = loop.lower(
                particles,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                relativistic=relativistic,
            )
        hlo_comp = lowered.compiler_ir(dialect="hlo")
        hlo = hlo_comp.as_hlo_text() if hasattr(hlo_comp, "as_hlo_text") else str(hlo_comp)
        _maybe_write_text(out_dir / ("scan_chunk.hlo.txt" if use_scan else "step.hlo.txt"), hlo)
    except Exception as e:
        _maybe_write_text(out_dir / "hlo_dump_error.txt", repr(e))

    steps_total = min(int(args.steps), nt)
    if use_scan:
        steps = max(scan_chunk, (steps_total // scan_chunk) * scan_chunk)
    else:
        steps = steps_total

    trace_dir = out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True)

    start = time.time()
    if use_scan:
        remaining = steps
        while remaining:
            k = scan_chunk if remaining >= scan_chunk else remaining
            particles, fields = scan_chunk_jit(particles, fields, n_steps=k)
            remaining -= k
    else:
        for _ in range(steps):
            particles, fields = loop(
                particles,
                fields,
                world,
                constants,
                curl_func,
                J_func,
                solver,
                relativistic=relativistic,
            )
    block_until_ready(fields[0][0])
    end = time.time()

    jax.profiler.stop_trace()

    _maybe_write_text(
        out_dir / "run.txt",
        "\n".join(
            [
                f"config={config_path}",
                f"enable_x64={enable_x64}",
                f"use_scan={use_scan}",
                f"scan_chunk={scan_chunk}",
                f"dt={dt}",
                f"steps={steps}",
                f"wall_s={end - start}",
                f"s_per_step={(end - start) / max(1, steps)}",
            ]
        )
        + "\n",
    )

    print(f"Wrote profile to: {out_dir}")


if __name__ == "__main__":
    main()
