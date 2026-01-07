Usage
=====

This section describes how to configure and run PyPIC3D from the command line,
along with the expected outputs.

Running a Simulation
--------------------

PyPIC3D is executed with a TOML configuration file passed via ``--config``:

.. code-block:: bash

    PyPIC3D --config path/to/config.toml

The CLI reads the TOML file, initializes the simulation, and writes outputs into
``simulation_parameters.output_dir`` (defaults to the current working directory).
By default the runtime emits data under ``<output_dir>/data`` including energy
diagnostics, field slices, and optional phase-space data.

Configuration File Layout
-------------------------

PyPIC3D expects a TOML file with top-level sections for simulation parameters,
plotting settings, and one or more particle species. Optional sections can
define constants and external fields.

.. code-block:: toml

    [simulation_parameters]
    name = "Two-stream example"
    solver = "fdtd"                  # fdtd, spectral, vector_potential, curl_curl
    electrostatic = false
    relativistic = true
    current_calculation = "j_from_rhov"  # j_from_rhov, esirkepov
    Nx = 100
    Ny = 1
    Nz = 1
    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0
    t_wind = 1e-8
    cfl = 0.9
    shape_factor = 1
    x_bc = "periodic"
    y_bc = "periodic"
    z_bc = "periodic"
    output_dir = "./outputs"

    [plotting]
    plotting_interval = 10
    plot_phasespace = false
    plot_vtk_particles = true

    [constants]
    eps = 8.85418782e-12
    mu = 1.25663706e-6
    C = 2.99792458e8
    kb = 1.380649e-23
    alpha = 1.0

    [particle1]
    name = "electrons"
    N_particles = 5000
    charge = -1.602e-19
    mass = 9.1093837e-31
    temperature = 1.0
    xmin = -0.5
    xmax = 0.5
    x_bc = "periodic"

External Fields
---------------

External field arrays can be added by declaring ``field`` sections in the TOML
file. Each entry must provide a name, a field type ( 0-2 for ``E``, 3-5 for ``B``, 6-8 for ``J``), and
a NumPy ``.npy`` file path matching the grid shape.

.. code-block:: toml

    [field1]
    name = "initial_bias"
    type = "E"
    path = "inputs/initial_E.npy"

Outputs
-------

PyPIC3D writes energy diagnostics to ``<output_dir>/data/*.txt`` at the plotting
interval, and can optionally output VTK slices and particle data for downstream
visualization tools. Use the plotting configuration flags to control which data
products are generated.
