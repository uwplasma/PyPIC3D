"""Select CI devices and a bounded, runner-local JAX compilation cache."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


CACHE_SETTINGS = {
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "10000000000",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
}

CUDA_PROBE = """
import json
import jax
import jax.numpy as jnp

devices = jax.devices("gpu")
for device in devices:
    x = jax.device_put(jnp.asarray([1., 2.], dtype=jnp.float64), device)
    result = jax.jit(lambda x: x * x + 1)(x).block_until_ready()
    assert bool(jnp.all(result == jnp.asarray([2., 5.])))
print(json.dumps({"devices": [str(device) for device in devices]}))
"""


def cuda13_available():
    """Check CUDA 13's Linux, driver, and compute-capability requirements."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        print("CUDA devices are hidden by CUDA_VISIBLE_DEVICES.", flush=True)
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"NVIDIA detection unavailable: {error}", flush=True)
        return False
    if result.returncode:
        print(f"NVIDIA detection failed: {result.stderr.strip()}", flush=True)
        return False
    for row in result.stdout.splitlines():
        try:
            driver, capability = row.split(",")
            if int(driver.strip().split(".")[0]) >= 580 and float(capability) >= 7.5:
                return True
        except ValueError:
            continue
    print("No CUDA 13 compatible NVIDIA device found.", flush=True)
    return False


def probe_cuda():
    # A separate interpreter prevents backend initialization from fixing the
    # device count or platform for the later one- and eight-device test stages.
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cuda", JAX_ENABLE_X64="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false",
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", CUDA_PROBE], env=environment,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            devices = json.loads(result.stdout.strip().splitlines()[-1])["devices"]
            print(f"Usable CUDA devices ({len(devices)}): {devices}", flush=True)
            return len(devices)
        print(f"CUDA initialization failed; selecting CPU.\n{result.stdout}\n{result.stderr}", flush=True)
    except (subprocess.TimeoutExpired, ValueError, IndexError, KeyError) as error:
        print(f"CUDA probe failed; selecting CPU: {error}", flush=True)
    return 0


def report_cache():
    if os.environ.get("JAX_ENABLE_COMPILATION_CACHE") != "true":
        print("Persistent compilation cache disabled.", flush=True)
        return
    cache_dir = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    count = size = 0
    for path in cache_dir.iterdir():
        try:
            if path.is_file():
                size += path.stat().st_size
                count += 1
        except FileNotFoundError:
            pass  # Another job can evict an entry while this snapshot is taken.
    print(f"JAX cache: {cache_dir}; {count} files; {size} bytes (including metadata)", flush=True)


def configure():
    gpu_count = 0
    if cuda13_available():
        version = importlib.metadata.version("jax")
        # Keep jax, jaxlib, and the CUDA plugin on the same release. Installation
        # errors are dependency failures, not successful CPU test runs.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"jax[cuda13]=={version}"], check=True,
        )
        gpu_count = probe_cuda()

    test_platform = "cuda" if gpu_count else "cpu"
    distributed_platform = "cuda" if gpu_count >= 8 else "cpu"
    settings = dict(CACHE_SETTINGS)
    settings.update(
        JAX_PLATFORMS=test_platform, JAX_ENABLE_X64="1", JAX_NUM_CPU_DEVICES="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )

    repository = os.environ["GITHUB_REPOSITORY"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pull_request = event.get("pull_request")
    head_repo = (pull_request.get("head", {}).get("repo") or {}) if pull_request else {}
    fork = pull_request is not None and head_repo.get("full_name") != repository
    settings["JAX_ENABLE_COMPILATION_CACHE"] = "false" if fork else "true"
    settings["JAX_COMPILATION_CACHE_DIR"] = ""
    if not fork:
        cache_dir = Path.home() / ".cache" / "pypic3d-ci" / repository / "jax-v1"
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cache_dir.chmod(0o700)
        settings["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    with open(os.environ["GITHUB_ENV"], "a") as environment_file:
        for name, value in settings.items():
            environment_file.write(f"{name}={value}\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output_file:
        output_file.write(f"distributed_platform={distributed_platform}\n")
    os.environ.update(settings)

    print(f"Code/physics backend: {test_platform}; devices: {gpu_count or 1}", flush=True)
    print(f"Distributed backend: {distributed_platform}; devices: {gpu_count if gpu_count >= 8 else 8}", flush=True)
    for package in ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt", "filelock"):
        try:
            print(f"{package}={importlib.metadata.version(package)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}=not installed", flush=True)
    for name, value in settings.items():
        print(f"{name}={value}", flush=True)
    report_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Report the current cache size only.")
    args = parser.parse_args()
    if args.report:
        report_cache()
    else:
        configure()
