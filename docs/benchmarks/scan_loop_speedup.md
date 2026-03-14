# Scan-based timestep batching (performance)

This PR adds an optional `simulation_parameters.use_scan=true` mode that batches PIC timesteps with `jax.lax.scan`.
The goal is to reduce Python↔XLA dispatch overhead while preserving the existing config-driven CLI workflow.

## How to reproduce

From the repository root:

```bash
./.venv/bin/PyPIC3D --config demos/two_stream/two_stream_fast.toml
```

To regenerate the benchmark plots (compares `origin/main` to the current checkout):

```bash
./.venv/bin/python bench/compare.py
```

## Results (steady-state)

Generated figures live in `docs/benchmarks/images/`:

- `docs/benchmarks/images/runtime_s_per_step.png`
- `docs/benchmarks/images/speedup.png`

## Accuracy (two-stream)

Energy trajectory comparisons:

- `docs/benchmarks/images/two_stream_electric_energy.png`
- `docs/benchmarks/images/two_stream_energy_error.png`

