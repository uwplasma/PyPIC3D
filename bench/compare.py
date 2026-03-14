#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Case:
    name: str
    config: str


def _git(cmd, cwd: Path) -> str:
    out = subprocess.check_output(["git", *cmd], cwd=str(cwd), text=True).strip()
    return out


def _ensure_worktree(repo: Path, ref: str, worktree_path: Path) -> None:
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    if worktree_path.exists():
        return
    subprocess.check_call(["git", "worktree", "add", "--detach", str(worktree_path), ref], cwd=str(repo))


def _run_runner(python: Path, runner: Path, repo: Path, config: str, steps: int, warmup: int, sample_every: int, sample_steps: int) -> dict:
    cmd = [
        str(python),
        str(runner),
        "--repo",
        str(repo),
        "--config",
        config,
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
    ]
    if sample_every:
        cmd += ["--sample-every", str(sample_every)]
    if sample_steps:
        cmd += ["--sample-steps", str(sample_steps)]
    out = subprocess.check_output(cmd, text=True)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def _summarize(values):
    if len(values) == 1:
        return {"mean": values[0], "median": values[0], "stdev": 0.0, "n": 1}
    return {"mean": mean(values), "median": median(values), "stdev": stdev(values), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare origin/main vs current branch with plots.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="docs/benchmarks/images")
    parser.add_argument("--worktrees", default="_bench/worktrees")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    python = repo / ".venv" / "bin" / "python"
    runner = repo / "bench" / "runner.py"

    main_ref = "origin/main"
    head_ref = "HEAD"

    main_sha = _git(["rev-parse", main_ref], repo)
    head_sha = _git(["rev-parse", head_ref], repo)

    wt_root = repo / args.worktrees
    main_repo = wt_root / f"main-{main_sha[:8]}"
    head_repo = repo
    _ensure_worktree(repo, main_ref, main_repo)

    cases = [
        Case("two_stream_fast", str(repo / "demos/two_stream/two_stream_fast.toml")),
        Case("weibel_fast", str(repo / "demos/weibel/weibel_fast.toml")),
    ]

    results = {
        "main": {"ref": main_ref, "sha": main_sha, "repo": str(main_repo), "cases": {}},
        "branch": {"ref": head_ref, "sha": head_sha, "repo": str(head_repo), "cases": {}},
    }

    for side, side_repo in [("main", main_repo), ("branch", head_repo)]:
        for case in cases:
            s_per_step = []
            for _ in range(int(args.repeats)):
                r = _run_runner(
                    python=python,
                    runner=runner,
                    repo=side_repo,
                    config=case.config,
                    steps=int(args.steps),
                    warmup=int(args.warmup),
                    sample_every=0,
                    sample_steps=0,
                )
                s_per_step.append(r["s_per_step"])
            results[side]["cases"][case.name] = {
                "config": case.config,
                "s_per_step": s_per_step,
                "summary": _summarize(s_per_step),
            }

    out_dir = repo / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Runtime bar chart (median).
    labels = [c.name for c in cases]
    main_vals = [results["main"]["cases"][c.name]["summary"]["median"] for c in cases]
    branch_vals = [results["branch"]["cases"][c.name]["summary"]["median"] for c in cases]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width / 2 for i in x], main_vals, width, label="origin/main")
    ax.bar([i + width / 2 for i in x], branch_vals, width, label="this PR")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("seconds / step (median)")
    ax.set_title("PyPIC3D steady-state runtime (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_s_per_step.png", dpi=200)
    plt.close(fig)

    # Speedup chart.
    speedups = [m / b for m, b in zip(main_vals, branch_vals)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, speedups)
    ax.axhline(1.0, color="k", linewidth=1)
    ax.set_ylabel("speedup (main / this PR)")
    ax.set_title("Speedup per case")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png", dpi=200)
    plt.close(fig)

    # Accuracy/trajectory plots (energy + error) for two_stream.
    sample_case = Case("two_stream", str(repo / "demos/two_stream/two_stream_fast.toml"))
    samples = {}
    for side, side_repo in [("main", main_repo), ("branch", head_repo)]:
        r = _run_runner(
            python=python,
            runner=runner,
            repo=side_repo,
            config=sample_case.config,
            steps=0,
            warmup=0,
            sample_every=10,
            sample_steps=1639,
        )
        samples[side] = r["samples"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(samples["main"]["t"], samples["main"]["electric_energy"], label="origin/main")
    ax.plot(samples["branch"]["t"], samples["branch"]["electric_energy"], label="this PR", linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Electric field energy (J)")
    ax.set_title("Two-stream: electric field energy vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "two_stream_electric_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(samples["main"]["t"], samples["main"]["energy_error"], label="origin/main")
    ax.semilogy(samples["branch"]["t"], samples["branch"]["energy_error"], label="this PR", linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative energy error")
    ax.set_title("Two-stream: energy error residual vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "two_stream_energy_error.png", dpi=200)
    plt.close(fig)

    # Accuracy/trajectory plots (energy + error) for weibel.
    sample_case = Case("weibel", str(repo / "demos/weibel/weibel_fast.toml"))
    samples = {}
    for side, side_repo in [("main", main_repo), ("branch", head_repo)]:
        r = _run_runner(
            python=python,
            runner=runner,
            repo=side_repo,
            config=sample_case.config,
            steps=0,
            warmup=0,
            sample_every=10,
            sample_steps=2000,
        )
        samples[side] = r["samples"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(samples["main"]["t"], samples["main"]["electric_energy"], label="origin/main")
    ax.plot(samples["branch"]["t"], samples["branch"]["electric_energy"], label="this PR", linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Electric field energy (J)")
    ax.set_title("Weibel: electric field energy vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "weibel_electric_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(samples["main"]["t"], samples["main"]["energy_error"], label="origin/main")
    ax.semilogy(samples["branch"]["t"], samples["branch"]["energy_error"], label="this PR", linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative energy error")
    ax.set_title("Weibel: energy error residual vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "weibel_energy_error.png", dpi=200)
    plt.close(fig)

    # Write a JSON summary alongside plots.
    summary_path = out_dir.parent / "benchmark_summary.json"
    (summary_path).write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote plots to: {out_dir}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
