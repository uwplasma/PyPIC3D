Particle Species
================

Each ``[particleX]`` TOML section defines one particle species. PyPIC3D supports
multi-species plasmas and per-species boundary behavior.

Required Fields
---------------

Each species must define:

- ``name``
- ``charge``
- ``mass``
- ``N_particles`` or ``N_per_cell``

Example
-------

.. code-block:: toml

    [particle1]
    name = "electrons"
    N_particles = 30000
    charge = -1.602e-19
    mass = 9.1093837e-31
    temperature = 293000
    shape_factor = 1

Common Optional Fields
----------------------

- Thermal setup: ``temperature`` or ``vth`` (optionally ``Tx``, ``Ty``, ``Tz``)
- Weighting: ``weight`` or ``ds_per_debye``
- Spatial bounds: ``xmin/xmax``, ``ymin/ymax``, ``zmin/zmax``
- Boundary conditions: ``x_bc``, ``y_bc``, ``z_bc``
- External initial state: ``initial_x/y/z``, ``initial_vx/vy/vz`` (``.npy``)
- Update controls: ``update_pos``, ``update_v``, component-level flags

Initialization Behavior
-----------------------

If external arrays are not provided:

- positions are sampled uniformly inside species bounds
- velocities are sampled from thermal distributions derived from
  ``temperature``/``vth``

If scalar ``initial_x/y/z`` values are used, PyPIC3D places particles near that
location with sub-cell variation when applicable.

Particle Boundary Conditions
----------------------------

Per-species boundary options:

- ``periodic``
- ``reflecting``

These are independent from field boundary conditions.

Shape Factors
-------------

``simulation_parameters.shape_factor`` controls interpolation/deposition order:

- ``1``: first-order
- ``2``: second-order
