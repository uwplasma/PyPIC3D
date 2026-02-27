PyPIC3D Documentation
=====================

.. container:: hero

   PyPIC3D is a JAX-based particle-in-cell code for electrodynamic and
   electrostatic plasma simulation. Runs are composed with TOML, launched
   from the CLI, and instrumented for diagnostics in VTK and OpenPMD.

   .. container:: hero-actions

      `Launch a demo <demos.html>`__
      `See usage requirements <usage.html>`__

.. container:: hero-callout

   **Focused for researchers**

   PyPIC3D bundles easy to read algorithms, autodifferentiation, and easily modifiable code 
   to enable researchers to rapidly prototype new numerical algorithms and experiments for 
   3D3V plasma simulations.

   `Browse feature demos <demos.html>`__

Quick navigation
----------------

.. cards::
   :columns: 3
   :gutter: 1.25rem
   :class: features

   * **Minimal CLI onboarding**  
     Install the package, configure :doc:`usage`, then execute a demo while
     capturing diagnostics.

   * **Solver pipeline**  
     Understand Yee, PSTD, and conservation steps in :doc:`solvers`.

   * **Simulation grid design**  
     Built-in helpers for species, grids, particles, and boundaries live in :doc:`grid`.

   * **Particle diagnostics**  
     Visualize scaling, charges, and diagnostics in :doc:`particles`.

   * **Development workflow**  
     Tests, docs builds, and local helpers are described in :doc:`development`.

   * **Join in**  
     Contribution instructions, dev workflow, and community notes are in :doc:`contributing`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Dive deeper

   usage
   solvers
   chargeconservation
   grid
   particles
   demos
   architecture
   development
   contributing

Indices and tables
==================

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`

