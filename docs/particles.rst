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

Callable Velocity Distributions
-------------------------------

``PyPIC3D.particles.velocity_distribution`` provides two JAX samplers. Both
return a tuple ``(vx, vy, vz)`` of arrays with shape ``(N_particles,)`` and
are JIT-compiled with a static particle count. Supply a fresh JAX random key
for each independent sample; repeating a key repeats the sample.

- ``maxwellian(N_particles, mass, Tx, Ty, Tz, kb, key)`` draws independent
  Gaussian components with variance ``kb*T_i/mass``. Directional temperatures
  may differ or be zero.
- ``maxwell_juttner(N_particles, mass, Tx, Ty, Tz, kb, key, c=1.0)`` draws
  rest-frame three-velocities using the modified Canfield four-Gamma rejection
  method. Its first seven arguments match ``maxwellian``, including physical
  temperature units. ``mass``, ``kb``, and ``c`` must be positive finite
  scalars. The returned velocities have the units of ``c``. This replaces
  the earlier dimensionless ``theta`` interface.

Maxwell-Juttner sampling supports isotropic, axisymmetric, and triaxial
temperatures through one momentum-space transformation. Define

.. math::

   \Theta_i = \frac{k_b T_i}{mc^2}, \qquad
   \Theta_0 = \frac{\Theta_x+\Theta_y+\Theta_z}{3}.

The sampler draws one isotropic dimensionless momentum vector at
``Theta0``, then rescales its components:

.. math::

   u_i = q_i\sqrt{\frac{\Theta_i}{\Theta_0}}, \qquad
   f(\mathbf{u}) \propto \exp\left[-\frac{1}{\Theta_0}
   \sqrt{1+\sum_i\frac{\Theta_0}{\Theta_i}u_i^2}\right].

Here ``u = p/(mass*c)`` is dimensionless momentum. The returned physical
velocity is ``v = c*u/sqrt(1 + sum(u**2))``, using the Lorentz factor after
rescaling. Equal temperatures recover the isotropic distribution. Directional
temperatures parameterize the distribution shape; they are not measured
relativistic pressure temperatures. See `Livadiotis (2016), Annales
Geophysicae 34, 1145-1158 <https://doi.org/10.5194/angeo-34-1145-2016>`_.

Each directional temperature is scalar per call; use ``jax.vmap`` over keys
and temperatures for batches. Temperatures must be positive and finite,
except that ``Tx = Ty = Tz = 0`` returns zero velocities. Negative, nonfinite,
or partly zero temperatures return NaNs, including under JIT and ``vmap``.
Unsupported argument shapes raise ``ValueError``. Extreme temperatures may
exceed the floating-point range or make speeds round to ``c``. No velocity
clipping, momentum cutoff, or pressure-matching solve is applied.

For example, in normalized units:

.. code-block:: python

   import jax
   from PyPIC3D.particles.velocity_distribution import maxwellian, maxwell_juttner

   maxwell_key, isotropic_key, triaxial_key = jax.random.split(jax.random.key(42), 3)
   vx, vy, vz = maxwellian(10000, mass=1.0, Tx=0.01, Ty=0.02,
                           Tz=0.01, kb=1.0, key=maxwell_key)
   vx, vy, vz = maxwell_juttner(10000, mass=1.0, Tx=1.0, Ty=1.0,
                               Tz=1.0, kb=1.0, key=isotropic_key, c=1.0)
   vx, vy, vz = maxwell_juttner(10000, mass=1.0, Tx=0.2, Ty=0.5,
                               Tz=1.0, kb=1.0, key=triaxial_key, c=1.0)

TOML initialization continues to use the Maxwellian. It now uses JAX random
keys instead of NumPy's global random state, so previous random realizations
change, including x positions from the split position/sampling key. External
arrays and scalar offsets retain their existing behavior. Maxwell-Juttner
sampling is available through the callable API only. Its output is
three-velocity; static-metric callers must convert it to the appropriate
covariant four-velocity before assigning particle state. Neither sampler
applies a bulk boost or performs leapfrog time centering.

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
