"""Shared unittest package for PyPIC3D tests."""

import os

# Set defaults before importing JAX. Distributed tests need a separate unittest
# invocation with JAX_NUM_CPU_DEVICES=8; explicit environment overrides win.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
