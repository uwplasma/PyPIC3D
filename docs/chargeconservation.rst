Charge Conservation Method
===========================

There are 3 charge conservation methods in PyPIC3D including a charge density and velocity based method and two continuity equation based methods.

Villasenor-Buneman
------------------
The Villasenor-Buneman method is charge conserving method for 1st order shape factors that assumes particles are moving in a straight line and deposits current onto each of the cell faces.

Esirkepov
---------
The Esirkepov method is a charge conserving method that makes use of a generalized shape factor to deposit current onto the cell faces. The Esirkepov method is typically faster than the Villasenor-Buneman method due to a lack of loops and can be used for particles with arbitrary shape factors.

Charge Density Method
---------------------
The charge density method uses the particle velocities and their charge to deposit current using the analytical definition of J

.. math::
    \mathbf{J} = \rho v













References
----------

For more details on the Villasenor-Buneman and the Esirkepov methods, refer to the original papers:

Villasenor, J., & Buneman, O. (1992). Rigorous charge conservation for local electromagnetic field solvers. *Journal of Computational Physics*, 58(2), 189-196.

Esirkepov, T. Z. (2001). Exact charge conservation scheme for particle-in-cell simulation with an arbitrary form-factor. Computer Physics Communications, 135(2), 144-153.