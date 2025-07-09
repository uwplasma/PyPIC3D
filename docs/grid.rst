Discrete Grids
--------------

PYPIC3D contains both staggered and collocated grids. PyPIC3D defaults to the Yee grid when using the first order solvers and the collocated grid when using the 2nd order vector potential solver.


Yee Grid
--------

The Yee grid tracks the electric field and current on the cell faces and the magnetic field at the cell centers.

.. image:: images/yeegrid.png
    :alt: Yee Grid
    :align: center



Collocated Grid
---------------

The collocated grid tracks both the fields and the current at the cell centers.