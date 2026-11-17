Development Guide
=================

Local Setup
-----------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

The ``dynamic_metric`` solver needs the optional ``JAX_BSSN`` dependency. Its
tests skip without it, and the ``demos/dynamical_relativity/`` examples cannot
run:

.. code-block:: bash

   pip install -e '.[relativity]'

Run Tests
---------

Run the complete suite, including numerical and convergence tests:

.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 \
     python -m unittest discover -s tests/code_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 \
     python -m unittest discover -s tests/physics_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=8 \
     python -m unittest discover -s tests/distributed_tests -t . -p '*test*.py'

Run these commands sequentially in separate interpreters so JAX initializes with
the appropriate device count. The discovery pattern includes both filename
conventions. Individual tests default to one CPU device and float64; explicit
environment overrides still work:

.. code-block:: bash

   JAX_PLATFORMS=cuda python -m unittest tests.code_tests.yee_test
   JAX_NUM_CPU_DEVICES=8 python -m unittest discover -s tests/distributed_tests -t . -p '*test*.py'

For coverage, install ``coverage``, run ``python -m coverage erase``, then prefix
each discovery command with ``python -m coverage run --append --context <group>``
in place of ``python``. Use contexts ``code``, ``physics``, and ``distributed``;
then run ``python -m coverage report`` or ``python -m coverage xml``.

Time fresh-process runs on the same machine with
``JAX_ENABLE_COMPILATION_CACHE=false``. Use unittest's elapsed time, CI step
durations, or GNU ``/usr/bin/time -v`` for wall time and peak resident memory.
Do not compare a cold run with a warmed compilation cache or concurrent workloads.
The ten-minute target never takes precedence over physics cases, refinement
levels, or numerical tolerances.

Build Docs
----------

.. code-block:: bash

   pip install -r docs/requirements.in
   sphinx-autobuild docs docs/_build/html --port 8008

This watches the source files, rebuilds on every save, and live-reloads the
open page. Open http://localhost:8008 to view the docs.

For a one-shot build that fails on any warning --- the same check the
``Documentation`` CI workflow runs --- use:

.. code-block:: bash

   python -m sphinx -b html -W --keep-going docs docs/_build/html

Debugging Tips
--------------

- Start with one tile covering the complete domain to separate numerical
  behavior from cross-device communication.
- For a multi-tile failure, verify tile divisibility, exposed device count, 
  number of ghost cells, and particle capacity.
