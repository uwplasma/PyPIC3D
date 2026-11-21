import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import call, patch

from PyPIC3D.__main__ import _configure_jax_runtime


class TestRuntimeSelection(unittest.TestCase):
    def test_runtime_keeps_jax_platform_selection_and_reports_devices(self):
        devices = ["CudaDevice(id=0)", "CudaDevice(id=1)"]

        with patch("PyPIC3D.__main__.jax.config.update") as update:
            with patch("PyPIC3D.__main__.jax.devices", return_value=devices):
                with patch("PyPIC3D.__main__.jax.default_backend", return_value="gpu"):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        backend, selected_devices = _configure_jax_runtime()

        self.assertEqual(update.call_args_list, [call("jax_enable_x64", True)])
        self.assertEqual(backend, "gpu")
        self.assertIs(selected_devices, devices)
        self.assertIn("Using JAX backend: gpu", output.getvalue())
        self.assertIn("JAX devices (2)", output.getvalue())

    def test_fresh_process_honors_explicit_cpu_platform_selection(self):
        environment = os.environ.copy()
        environment["JAX_PLATFORMS"] = "cpu"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        command = (
            "from PyPIC3D.__main__ import _configure_jax_runtime; "
            "backend, devices = _configure_jax_runtime(); "
            "print(f'EXPLICIT_SELECTION={backend}:{len(devices)}')"
        )

        completed = subprocess.run(
            [sys.executable, "-c", command],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Using JAX backend: cpu", completed.stdout)
        self.assertIn("EXPLICIT_SELECTION=cpu:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
