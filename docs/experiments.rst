Computational Experiments
=========================

PyPIC3D does not ship with a dedicated experiment framework. Instead, batch
studies and parameter scans are typically managed with small Python scripts or
shell loops that generate configuration files and invoke the CLI. This approach
keeps experiment logic lightweight and easy to customize for a given project.

Suggested Workflow
------------------

1. Start from a baseline TOML configuration.
2. Programmatically edit a parameter (e.g., ``N_particles`` or ``Nx``).
3. Launch a run with ``PyPIC3D --config``.
4. Collect the resulting ``data/output.toml`` and diagnostic files.

For examples of lightweight benchmarking and convergence studies, see the
scripts under ``demos/convergence_testing``.
