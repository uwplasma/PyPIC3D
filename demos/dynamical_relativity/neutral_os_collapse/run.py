"""Evolve the neutral OS star with the native dynamical PIC/BSSN timestep.

Run this script without arguments for the approved 64-cubed, t=60M case.
The checkpoint contains native staggered particles and all field histories;
restart reconstructs only the derived Yee metric, never the leapfrog history.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.diagnostics.constraints import compute_constraint_norms
from PyPIC3D.relativity.dynamical_relativity import (
    build_bssn_metric, initialize_dynamical_relativity, synchronize_particles,
)
from PyPIC3D.relativity.energy_momentum_tensor import compute_particle_energy_momentum
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity

if __package__ in (None, ""):
    from initial_data import build_initial_data, surface_isotropic_radius
    from diagnostics import extract_profiles
else:
    from .initial_data import build_initial_data, surface_isotropic_radius
    from .diagnostics import extract_profiles


DEFAULT_OUTPUT = Path(__file__).with_name("outputs")
HISTORY_COLUMNS = (
    "step", "time", "central_lapse", "minimum_lapse", "central_proper_time",
    "active_particles", "retained_rest_mass", "deleted_lapse", "escaped_outer",
    "minimum_W", "maximum_EM", "central_shift_over_lapse", "step_seconds",
)


def write_json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_tree(particles, fields, dynamic):
    """Retain native state; the Yee metric is derived from saved BSSN data."""
    return particles, fields[:6], fields[7:], dynamic.bssn_parameters


def save_checkpoint(path, particles, fields, dynamic, config, row):
    leaves = jax.tree.leaves(checkpoint_tree(particles, fields, dynamic))
    metadata = dict(version=1, config=config, history=row,
                    native_dt=float(fields[7][3]), time=float(dynamic.bssn_parameters.time))
    temporary = path.with_suffix(".tmp.npz")
    jnp.savez(temporary, metadata=json.dumps(metadata),
              **{f"q{i}": value for i, value in enumerate(leaves)})
    temporary.replace(path)


def load_checkpoint(path, particles, fields, static, dynamic):
    """Restore saved histories directly, without a common-time reseeding."""
    # The initializer returns common-time fields. Supply only a shape template
    # for the five runtime history slots; none of these zeros are evolved.
    runtime = fields[:7] + ((fields[0], fields[1], fields[1], dynamic.dt / 2, None), fields[8])
    template, tree = jax.tree.flatten(checkpoint_tree(particles, runtime, dynamic))
    with jnp.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        if metadata["version"] != 1 or metadata["config"]["dt"] != float(dynamic.dt):
            raise ValueError("OS restart requires the saved schema and unchanged timestep")
        if set(saved.files) != {"metadata", *(f"q{i}" for i in range(len(template)))}:
            raise ValueError("OS restart native state layout differs from this code")
        values = [jnp.asarray(saved[f"q{i}"]) for i in range(len(template))]
        if any(a.shape != jnp.shape(b) or a.dtype != jnp.asarray(b).dtype
               for a, b in zip(values, template)):
            raise ValueError("OS restart arrays differ from the configured particle/grid layout")
    particles, initial, tail, bssn = jax.tree.unflatten(tree, values)
    dynamic = dynamic._replace(bssn_parameters=bssn)
    metric = build_bssn_metric(bssn.state, static, dynamic)
    fields = initial + (metric,) + tail
    if float(fields[7][3]) != float(dynamic.dt) / 2 or float(bssn.time) != metadata["time"]:
        raise ValueError("OS restart has inconsistent saved time levels")
    return particles, fields, dynamic, metadata["history"]


def scalar_diagnostics(particles, fields, dynamic, species, center):
    state = dynamic.bssn_parameters.state
    gamma = state.conformal_metric
    determinant = (gamma[0, 0] * (gamma[1, 1]*gamma[2, 2] - gamma[1, 2]**2)
                   - gamma[0, 1] * (gamma[0, 1]*gamma[2, 2] - gamma[1, 2]*gamma[0, 2])
                   + gamma[0, 2] * (gamma[0, 1]*gamma[1, 2] - gamma[1, 1]*gamma[0, 2]))
    positive_metric = (jnp.all(gamma[0, 0] > 0)
                       & jnp.all(gamma[0, 0]*gamma[1, 1] - gamma[0, 1]**2 > 0)
                       & jnp.all(determinant > 0))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in
                               jax.tree.leaves((particles, fields, state))]))
    maximum_em = jnp.max(jnp.stack([jnp.max(jnp.abs(value))
                                  for value in (*fields[0], *fields[1], *fields[2])]))
    count = jnp.sum(particles.active)
    minimum_W, minimum_lapse = jnp.min(state.conformal_factor), jnp.min(state.lapse)
    center_shift = state.shift[(slice(None),) + center]
    center_gamma = gamma[(slice(None), slice(None)) + center] / state.conformal_factor[center]**2
    center_shift_norm = jnp.sqrt(jnp.maximum(center_shift @ center_gamma @ center_shift, 0.))
    return dict(
        valid=finite & positive_metric & (minimum_W > 0) & (minimum_lapse > 0)
              & ~fields[8] & (maximum_em == 0),
        central_lapse=state.lapse[center], minimum_lapse=minimum_lapse,
        minimum_W=minimum_W, maximum_EM=maximum_em, active_particles=count,
        retained_rest_mass=count * species.mass[0] * species.weight[0],
        central_shift_over_lapse=center_shift_norm / state.lapse[center],
    )


def matter_constraints(particles, fields, static, dynamic, species, *, synchronized=True):
    """Measure physical dust sources, including the sharp stellar surface."""
    if synchronized:
        particles = synchronize_particles(particles, species, fields, static, dynamic)
    rho, momentum, stress = compute_particle_energy_momentum(
        particles, species, fields[6], static, dynamic)
    state, params = dynamic.bssn_parameters.state, dynamic.bssn_parameters.evolution_parameters
    violations = compute_all_constraints_with_matter(state, params, rho, momentum)
    norms = compute_constraint_norms(violations)
    axis = jnp.arange(state.lapse.shape[0]) * params.dx + params.x_min
    radius = jnp.sqrt(axis[:, None, None]**2 + axis[None, :, None]**2 + axis[None, None, :]**2)
    star_radius = surface_isotropic_radius()
    index = jnp.arange(state.lapse.shape[0])
    face = (index < 2) | (index >= state.lapse.shape[0]-2)
    masks = dict(initial_bulk=radius < star_radius - 2*params.dx,
                 initial_surface=jnp.abs(radius-star_radius) <= 2*params.dx,
                 exterior=(radius > star_radius + 2*params.dx) & (radius < 6.),
                 outer_faces=face[:, None, None] | face[None, :, None] | face[None, None, :])
    for name, mask in masks.items():
        norms[f"hamiltonian_{name}_l2"] = jnp.sqrt(
            jnp.sum(jnp.where(mask, violations.hamiltonian**2, 0.)) / jnp.maximum(jnp.sum(mask), 1))
    volume = state.conformal_factor**-3 * params.dx**3
    norms.update(matter_energy_integral=jnp.sum(rho * volume),
                 maximum_matter_density=jnp.max(rho), maximum_momentum=jnp.max(jnp.abs(momentum)),
                 maximum_stress=jnp.max(jnp.abs(stress)))
    return norms


def resources(output):
    gpu = "unavailable"
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
        gpu = probe.stdout.strip()
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                gpu_memory_MiB_and_utilization_percent=gpu,
                host_load=list(os.getloadavg()), affinity=sorted(os.sched_getaffinity(0)),
                disk_free_bytes=shutil.disk_usage(output).free)


def projected_storage_bytes(output, array_bytes, config):
    """Budget scheduled states once, including states already saved on restart."""
    states = {output / f"checkpoint_{slot}.npz" for slot in (0, 1)}
    states.update(output / name for name in ("last_valid.npz", "failed_state.npz"))
    states.update(output / "snapshots" / f"state_{step:06d}.npz"
                  for step in snapshot_output_steps(config["dt"], config["end_time"]))
    # Scheduled snapshots, two checkpoints, two failure states, one atomic-write
    # temporary. Backups and other existing files consume additional space.
    additional = sum(path.stat().st_size for path in output.rglob("*")
                     if path.is_file() and path not in states)
    return (len(states)+1)*array_bytes + 100*1024**2 + additional


def snapshot_output_steps(dt, end_time):
    """Retain a native snapshot every 20M and at the requested endpoint."""
    return {round(20.*index/dt) for index in range(int(end_time/20.)+1)} | {round(end_time/dt)}


def extend_configuration(config, end_time):
    """Change only the stopping time of a saved physical configuration."""
    if end_time <= config["end_time"]:
        raise ValueError("--extend-to must be later than the saved target time")
    return dict(config, end_time=end_time)


def profile_output_steps(dt, end_time):
    """Use the nearest accepted step to each quarter-M diagnostic time."""
    return {round(.25*index/dt) for index in range(1, round(end_time/.25)+1)} | {round(end_time/dt)}


def preserve_restart_tail(output, step):
    """Keep abandoned diagnostic records before continuing an older checkpoint."""
    backup = output / ("before_resume_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    for name in ("history.csv", "constraints.jsonl"):
        path = output / name
        if not path.exists():
            continue
        if name.endswith(".csv"):
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        retained = [row for row in rows if int(row["step"]) <= step]
        if len(retained) != len(rows):
            backup.mkdir(exist_ok=True)
            shutil.copy2(path, backup / name)
            with path.open("w", newline="") as stream:
                if name.endswith(".csv"):
                    writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
                    writer.writeheader()
                    writer.writerows(retained)
                else:
                    stream.writelines(json.dumps(row) + "\n" for row in retained)
    for folder, pattern in (("profiles", "profile_*.npz"), ("snapshots", "state_*.npz")):
        for path in sorted((output / folder).glob(pattern)):
            if int(path.stem.split("_")[-1]) > step:
                (backup / folder).mkdir(parents=True, exist_ok=True)
                path.replace(backup / folder / path.name)


def run(args):
    jax.config.update("jax_enable_x64", True)
    output = args.output_dir.resolve()
    config = dict(nx=args.nx, particle_count=args.particles, half_width=args.half_width,
                  dt=args.dt, end_time=args.end_time)
    if args.resume:
        with jnp.load(args.resume, allow_pickle=False) as saved:
            config = json.loads(saved["metadata"].item())["config"]
    if args.extend_to is not None:
        if not args.resume:
            raise ValueError("--extend-to requires a native restart checkpoint")
        config = extend_configuration(config, args.extend_to)
    if (output / "metadata.json").exists() and not args.resume:
        raise FileExistsError(f"Preserving existing OS output at {output}; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "profiles").mkdir(exist_ok=True)
    (output / "snapshots").mkdir(exist_ok=True)
    started, step = time.monotonic(), 0
    row = None
    status = dict(state="running", stage="initializing", step=0, target_time=config["end_time"])
    write_json(output / "status.json", status)

    try:
        particles, species, fields, static, dynamic = build_initial_data(**config, output_dir=str(output))
        center = tuple(int(round(-float(origin)/float(dynamic.dx))) for origin in
                       (dynamic.bssn_parameters.evolution_parameters.x_min,
                        dynamic.bssn_parameters.evolution_parameters.y_min,
                        dynamic.bssn_parameters.evolution_parameters.z_min))
        if args.resume:
            particles, fields, dynamic, row = load_checkpoint(args.resume, particles, fields, static, dynamic)
            step = int(row["step"])
            preserve_restart_tail(output, step)
        else:
            particles, fields, dynamic = initialize_dynamical_relativity(
                particles, species, fields, static, dynamic)
        jax.block_until_ready((particles, fields, dynamic))

        scalar = jax.jit(lambda p, f, d: scalar_diagnostics(p, f, d, species, center))
        values = jax.device_get(scalar(particles, fields, dynamic))
        if not bool(values["valid"]):
            raise RuntimeError("Initial/restarted OS state is invalid or has nonzero EM fields")
        if row is None:
            row = dict(step=0, time=0., central_proper_time=0., deleted_lapse=0, escaped_outer=0,
                       step_seconds=0., **{key: float(value) for key, value in values.items() if key != "valid"})
            row["active_particles"] = int(row["active_particles"])
        array_bytes = sum(value.size*value.dtype.itemsize for value in
                          jax.tree.leaves(checkpoint_tree(particles, fields, dynamic)))
        estimated_bytes = projected_storage_bytes(output, array_bytes, config)
        if estimated_bytes > 5*1024**3 or shutil.disk_usage(output).free < estimated_bytes + 2*1024**3:
            raise RuntimeError("OS output would exceed its 5 GiB budget or available disk reserve")
        metadata = dict(config, total_mass=1., mass=1., surface_areal_radius=5.,
                        total_rest_mass=float(species.mass[0]*species.weight[0])*config["particle_count"],
                        gauge="advective 1+log; native Gamma driver", lapse_threshold=.1,
                        source_filter="none", particle_loading="Sobol proper mass with eight reflections",
                        field_boundary="constant", particle_boundary="absorbing", metric_boundary="Sommerfeld",
                        conformal_factor="W=psi^-2; chi=W^2", coordinates="native evolution Cartesian nodes",
                        paper="arXiv:2611.12831v1, Eqs.95-103", source_method="native PIC-BSSN split",
                        jax_version=jax.__version__, devices=[str(device) for device in jax.devices()],
                        bssn_controls={name: float(getattr(dynamic.bssn_parameters.evolution_parameters, name))
                                       for name in ("eta", "g", "kappa", "nu", "gauge", "zero_shift")},
                        checkpoint_bytes=array_bytes, estimated_output_bytes=estimated_bytes,
                        initial_resources=resources(output),
                        interpretation="One coarse run; convergence and boundary independence unestablished")
        if args.resume and (output / "metadata.json").exists():
            metadata = json.loads((output / "metadata.json").read_text())
            metadata.update(end_time=config["end_time"], estimated_output_bytes=estimated_bytes)
            metadata.setdefault("restarts", []).append(dict(path=str(args.resume), step=step, resources=resources(output)))
        write_json(output / "metadata.json", metadata)
        if not args.resume:
            # Preserve the initial native state even if compilation is interrupted.
            save_checkpoint(output / "snapshots" / "state_000000.npz", particles, fields, dynamic, config, row)
            save_checkpoint(output / "checkpoint_0.npz", particles, fields, dynamic, config, row)

        def advance(particles, fields, dynamic):
            before = particles.active
            particles, fields, dynamic = time_loop_dynamical_relativity(
                particles, species, fields, static, dynamic)
            removed = before & ~particles.active
            outside = jnp.any((particles.x < -config["half_width"])
                              | (particles.x > config["half_width"]), axis=-1)
            values = scalar_diagnostics(particles, fields, dynamic, species, center)
            values.update(new_escaped_outer=jnp.sum(removed & outside),
                          new_deleted_lapse=jnp.sum(removed & ~outside))
            return particles, fields, dynamic, values

        write_json(output / "status.json", dict(status, stage="compiling", step=step, time=row["time"]))
        compile_start = time.monotonic()
        compiled = jax.jit(advance).lower(particles, fields, dynamic).compile()
        metadata.setdefault("compilations", []).append(dict(seconds=time.monotonic()-compile_start,
                                                           memory=str(compiled.memory_analysis())))
        write_json(output / "metadata.json", metadata)
        constraints = jax.jit(lambda p, f, d: matter_constraints(p, f, static, d, species))
        profiles = jax.jit(lambda state: extract_profiles(state, dynamic.bssn_parameters.evolution_parameters))
        profile_steps = profile_output_steps(config["dt"], config["end_time"])
        constraint_stride = max(1, round(1./config["dt"]))
        checkpoint_stride = max(1, round(5./config["dt"]))
        final_step = round(config["end_time"]/config["dt"])
        stop_step = final_step if args.stop_after is None else min(final_step, step+args.stop_after)
        snapshots = snapshot_output_steps(config["dt"], config["end_time"])

        with (output / "history.csv").open("a" if args.resume else "w", newline="") as history, \
             (output / "constraints.jsonl").open("a" if args.resume else "w") as constraint_file, \
             (output / "resources.jsonl").open("a" if args.resume else "w") as resource_file:
            writer = csv.DictWriter(history, fieldnames=HISTORY_COLUMNS)
            if not args.resume or history.tell() == 0:
                writer.writeheader()
                writer.writerow(row)
                history.flush()

            def record_profiles():
                arrays = profiles(dynamic.bssn_parameters.state)
                jnp.savez(output / "profiles" / f"profile_{step:06d}.npz",
                          **arrays, step=step, time=row["time"])

            def record_constraints():
                norms = jax.device_get(constraints(particles, fields, dynamic))
                constraint_file.write(json.dumps(dict(step=step, time=row["time"],
                    **{key: float(value) for key, value in norms.items()}), allow_nan=False) + "\n")
                constraint_file.flush()
                resource_file.write(json.dumps(dict(step=step, **resources(output))) + "\n")
                resource_file.flush()

            if not args.resume:
                record_profiles()
                record_constraints()
            elif step == 0:
                # Compilation can stop after the initial checkpoint is saved
                # but before these baseline diagnostics have been written.
                if not (output / "profiles" / "profile_000000.npz").exists():
                    record_profiles()
                if constraint_file.tell() == 0:
                    record_constraints()

            timings = []
            for next_step in range(step+1, stop_step+1):
                before = time.monotonic()
                candidate_p, candidate_f, candidate_d, values = compiled(particles, fields, dynamic)
                values = jax.device_get(values)
                elapsed = time.monotonic()-before
                if not bool(values["valid"]):
                    save_checkpoint(output / "last_valid.npz", particles, fields, dynamic, config, row)
                    failed_row = dict(row, step=next_step, time=next_step*config["dt"])
                    save_checkpoint(output / "failed_state.npz", candidate_p, candidate_f, candidate_d, config, failed_row)
                    raise RuntimeError(f"Invalid state at step {next_step}: finite/positive geometry, zero EM or capacity check failed")
                particles, fields, dynamic = candidate_p, candidate_f, candidate_d
                step = next_step
                current_time = float(dynamic.bssn_parameters.time)
                central = float(values["central_lapse"])
                row = dict(step=step, time=current_time,
                           central_proper_time=row["central_proper_time"]
                             + .5*(row["central_lapse"]+central)*(current_time-row["time"]),
                           deleted_lapse=row["deleted_lapse"]+int(values["new_deleted_lapse"]),
                           escaped_outer=row["escaped_outer"]+int(values["new_escaped_outer"]),
                           step_seconds=elapsed,
                           **{key: float(values[key]) for key in ("central_lapse", "minimum_lapse",
                               "minimum_W", "maximum_EM", "retained_rest_mass", "central_shift_over_lapse")},
                           active_particles=int(values["active_particles"]))
                writer.writerow(row)
                timings.append(elapsed)
                if step in profile_steps or step == stop_step:
                    record_profiles()
                    history.flush()
                    status.update(stage="evolving", step=step, time=current_time,
                                  elapsed_seconds=time.monotonic()-started,
                                  mean_step_seconds=sum(timings[-10:])/len(timings[-10:]),
                                  active_particles=row["active_particles"], minimum_lapse=row["minimum_lapse"])
                    write_json(output / "status.json", status)
                    print(json.dumps(status), flush=True)
                if step % constraint_stride == 0 or step == stop_step:
                    record_constraints()
                if step in snapshots:
                    save_checkpoint(output / "snapshots" / f"state_{step:06d}.npz", particles, fields, dynamic, config, row)
                if step % checkpoint_stride == 0 or step == stop_step:
                    save_checkpoint(output / f"checkpoint_{(step//checkpoint_stride)%2}.npz",
                                    particles, fields, dynamic, config, row)

        state = "complete" if step == final_step else "smoke_complete"
        status.update(state=state, stage="finished", step=step, time=row["time"],
                      final_history=row, elapsed_seconds=time.monotonic()-started,
                      all_particles_removed=row["active_particles"] == 0)
        write_json(output / "status.json", status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        status.update(state="failed", step=step, error=str(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started)
        if row is not None:
            status["last_valid_history"] = row
        write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--particles", type=int, default=262144)
    parser.add_argument("--half-width", type=float, default=8.)
    parser.add_argument("--dt", type=float, default=.025)
    parser.add_argument("--end-time", type=float, default=60.)
    parser.add_argument("--resume", type=Path, help="restore this checkpoint's saved configuration and native histories")
    parser.add_argument("--extend-to", type=float, help="extend a restart's stopping time without changing its numerical setup")
    parser.add_argument("--stop-after", type=int, help="stop after this many additional steps for a resumable smoke test")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
