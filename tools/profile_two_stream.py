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

    out_dir = Path(args.out) if args.out else Path(simulation_parameters["output_dir"]) / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Best-effort HLO dump for the step function.
    try:
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
        _maybe_write_text(out_dir / "step.hlo.txt", hlo)
    except Exception as e:
        _maybe_write_text(out_dir / "hlo_dump_error.txt", repr(e))

    steps = min(int(args.steps), nt)

    trace_dir = out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True)

    start = time.time()
    for i in range(steps):
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
