Particle Species
================

Each ``[particleX]`` TOML section defines one species. Multiple species share
the same tiled mesh and global particle boundary conditions.

Required Fields
---------------

Each species defines:

- ``name``
- ``charge``
- ``mass``
- either ``N_particles`` or ``N_per_cell``

Example
-------

.. code-block:: toml

   [particle1]
   name = "electrons"
   N_particles = 30000
   charge = -1.602e-19
   mass = 9.1093837e-31
   temperature = 293000
   number_density = 1.0e15

Initialization Options
----------------------

- Thermal state: ``temperature`` or ``vth``, with optional directional
  temperatures ``Tx``, ``Ty``, and ``Tz``.
- Statistical weight: ``weight`` directly, or ``number_density`` to derive the
  weight from particles per cell and cell volume.
- Sampling bounds: ``xmin/xmax``, ``ymin/ymax``, and ``zmin/zmax``. Defaults
  are the complete centered simulation domain.
- Initial arrays or scalar offsets: ``initial_x``, ``initial_y``,
  ``initial_z``, ``initial_vx``, ``initial_vy``, and ``initial_vz``. String
  values are loaded from ``.npy`` files.
- Update controls: ``update_x``, ``update_y``, and ``update_z``. A false flag
  freezes both the position and velocity component and suppresses that
  species' current contribution in the same direction.

Without external arrays, positions are sampled uniformly within the species
bounds and velocities are sampled from the configured thermal distributions.
A scalar initial position places particles within a sub-cell interval around
that coordinate when the axis contains more than one cell. A scalar initial
velocity is added to the sampled thermal velocity.

Particle Boundary Conditions
----------------------------

The runtime particle boundaries are global simulation parameters:

- ``particle_x_bc``
- ``particle_y_bc``
- ``particle_z_bc``

Supported values are:

- ``periodic``: wrap positions across the domain.
- ``reflecting``: reflect position and the normal velocity.
- ``absorbing``: mark particles inactive after they leave the domain.

Inactive particles retain allocated slots for static JAX shapes but no longer
move, push, deposit, or contribute to particle diagnostics.

Tiled Particle State
--------------------

``TiledParticles`` contains position ``x``, velocity ``u``, and active-mask
arrays with leading tile and species axes. ``SpeciesConfig`` separately stores
charge, mass, weight, and update masks once per species. This avoids repeating
constant metadata in every particle slot.

Particle slots have fixed capacity after initialization. Retiling moves active
particles between adjacent tile owners; insufficient destination capacity is a
hard runtime error. See :doc:`tiling` for exact shapes and capacity selection.

Leapfrog Time Staggering
------------------------

Positions and velocities are staggered in time. Every solver advances
``u^{n-1/2}`` to ``u^{n+1/2}`` using the force gathered at ``x^n``, then moves
the position with the new velocity. The scheme is second-order accurate in
``dt`` only if the run begins from ``u^{-1/2}``, so ``initialize_simulation``
pushes the configured ``initial_vx/vy/vz`` back half a step with the same
pusher the run will use. This happens once, unconditionally, after the initial
energy report, and is announced on startup. Code that overwrites
``particles.u`` after ``initialize_simulation`` returns must reapply
``PyPIC3D.pusher.particle_push.seed_leapfrog_velocity`` itself.

One consequence is that diagnostics reading ``particles.x`` and ``particles.u``
from the same state combine two time levels, which leaves an ``O(dt)`` offset in
reported kinetic energies and momenta. That offset is a property of the
diagnostic, not of the trajectory.

For the ``static_metric`` solver ``u`` is the covariant spatial four-velocity
``u_i`` in the active chart, so ``initial_vx/vy/vz`` are read as ``u_1``,
``u_2`` and ``u_3`` and are not three-velocity components.

Particle Output
---------------

openPMD particle series contain ``position``, ``momentum``, ``weighting``,
``charge``, ``mass``, and a scalar ``gamma`` record for every solver.

For flat runs, ``gamma`` is the Lorentz factor of the stored velocity,
``momentum`` is ``m*gamma*v``, and runtime snapshots shift positions back by
``u*dt/2`` so that position and velocity refer to the same time level.

For the ``static_metric`` solver, ``momentum`` is ``m*u_i``, the covariant
spatial components in units with ``c=1``; ``u_i`` already includes the
Lorentz factor. ``gamma`` is ``sqrt(1 + gamma^ij u_i u_j)``, computed by
``relativity.interpolate_metric.particle_lorentz_factor`` with the same
Hermite metric sample the pusher uses. Positions are written as stored, with
no half-step shift.

Shape Factors
-------------

``simulation_parameters.shape_factor`` controls interpolation and deposition:

- ``1``: first-order particle shapes.
- ``2``: second-order particle shapes.

Use the same shape factor when comparing runs because it changes the particle
stencil, numerical smoothing, and guard-cell reach.
