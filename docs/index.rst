PyPIC3D
=====================

.. container:: hero

   PyPIC3D is a JAX-based particle-in-cell code for
   electrodynamic and electrostatic plasma simulations. PyPIC3D 
   features domain decomposition and is parallelized using the
   JAX XLA compiler across tiles and devices. Simulations can be
   configured using either TOML configuration files or within 
   Python scripts. The code is designed to be lightweight and 
   easy to modify for rapid development of new numerical methods.

   .. container:: hero-actions

      :doc:`Launch a demo <demos>`
      :doc:`Configure a run <usage>`

.. container:: hero-callout

   **Focused for researchers**

   PyPIC3D keeps the particle push, deposition, Yee staggering, ghost cells,
   and boundary operations visible so new numerical methods can be evaluated
   and prototyped without a large framework layer.

Quick navigation
----------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Configure a simulation
      :link: usage
      :link-type: doc

      Start from the active TOML schema and CLI workflow.

   .. grid-item-card:: Domain decomposition
      :link: tiling
      :link-type: doc

      Choose tile sizes, guard depth, and understand 
      the parallelization strategy.
      
   .. grid-item-card:: Field solvers
      :link: solvers
      :link-type: doc

      Overview of the field solver algorithms.

   .. grid-item-card:: Inspect the grids
      :link: grid
      :link-type: doc

      Review coordinate arrays, staggering, and physical boundaries.

   .. grid-item-card:: Initialize particles
      :link: particles
      :link-type: doc

      Define species and understand fixed-capacity tiled storage.

   .. grid-item-card:: Develop PyPIC3D
      :link: development
      :link-type: doc

      Set up the repository, run tests, and build these docs.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Dive deeper

   usage
   tiling
   solvers
   einstein_maxwell
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
