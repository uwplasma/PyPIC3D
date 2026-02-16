Development Guide
=================

Local Setup
-----------

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

Run Tests
---------

.. code-block:: bash

    pytest

Build Docs
----------

.. code-block:: bash

    pip install -r docs/requirements.in
    sphinx-build -b html docs docs/_build/html

Configuration and Runtime Notes
-------------------------------

- CLI entrypoint: ``PyPIC3D.__main__:main``.
- ``__main__.py`` currently forces CPU backend via ``jax_platform_name = cpu``.
- Defaults are defined in ``initialization.default_parameters`` and merged with
  TOML using ``utils.update_parameters_from_toml``.
- Unknown keys in ``simulation_parameters``/``plotting``/``constants`` are
  ignored by the merge helper.

Debugging Tips
--------------

- Start with a small grid (for example ``Nx=Ny=Nz=16`` where applicable).
- Increase ``plotting_interval`` for faster benchmarking runs.
- Verify boundary-condition choices explicitly for both fields and particles.
- Compare ``energy_error.txt`` across branches when validating changes.
