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

Tests have two cost tiers, ``light_tests`` and ``heavy_tests``, each with
``code_tests``, ``physics_tests``, and ``distributed_tests``. Code tests exercise
one-tile API contracts; physics tests check one-tile identities, physical
references, conservation, and convergence. Decomposition, communication,
ownership, and sharded-output tests belong in the distributed category,
including synthetic multi-tile arrays without communication.

Run the three light suites, followed by the three heavy suites:

.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 python -m unittest discover -s tests/light_tests/code_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 python -m unittest discover -s tests/light_tests/physics_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=8 python -m unittest discover -s tests/light_tests/distributed_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 python -m unittest discover -s tests/heavy_tests/code_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=1 python -m unittest discover -s tests/heavy_tests/physics_tests -t . -p '*test*.py'
   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_NUM_CPU_DEVICES=8 python -m unittest discover -s tests/heavy_tests/distributed_tests -t . -p '*test*.py'

Run these commands sequentially in separate interpreters so JAX initializes with
the appropriate device count. The discovery pattern includes both filename
conventions. Individual tests default to one CPU device and float64; explicit
environment overrides still work:

.. code-block:: bash

   JAX_PLATFORMS=cuda python -m unittest tests.light_tests.code_tests.boris_test

PRs run the ``ci`` workflow and only the light tier. Weekly ``Test Coverage``
runs both tiers on Sunday at 06:17 UTC. Both workflows support manual dispatch,
install the optional relativity dependency, and use isolated virtual environments
with the existing bounded runner-local compilation cache. Ordinary tests use CUDA
when available; distributed tests use eight CPU devices unless at least eight
CUDA devices are available. Fork PRs use separate compilation caches.

For coverage, install ``coverage`` and run ``python -m coverage erase``. Replace
``python -m unittest`` in each command above with
``python -m coverage run --append --context <tier>-<category> -m unittest``.
The six contexts are ``light-code``, ``light-physics``, ``light-distributed``,
``heavy-code``, ``heavy-physics``, and ``heavy-distributed``. Then run
``python -m coverage report`` and ``python -m coverage xml``. Weekly coverage
attempts every group even when another group fails and reports the overall run
as failed while retaining available coverage data.

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
