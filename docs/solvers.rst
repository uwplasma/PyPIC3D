Solvers Overview
================

.. math::

    \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}

    \nabla \cdot \mathbf{B} = 0

    \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}

    \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}

PyPIC3D provides multiple time integrators for Maxwell's equations and related
formulations. Select a solver by setting
``simulation_parameters.solver`` in the TOML configuration.

Initialization
--------------

The electric field is initialized as 0 by default, but can be initialized using the following equation:

.. math::

    \nabla^2 \mathbf{\phi} = \frac{-\rho}{\epsilon_0}

PyPIC3D can solve for the initial electric field using both spectral and finite
difference methods. The magnetic field is initialized to zero.

.. Successive Over-Relaxation (SOR)
.. ++++++++++++++++++++++++++++++++

.. The Successive Over-Relaxation (SOR) method is an iterative method used to solve linear systems of equations. The SOR method is used to solve the Poisson equation for the electric field in PyPIC3D.
.. The algorithm is as follows:

.. 1. Initialize the electric potential to an array of random values.
.. 2. Iterate until convergence:
..     a. For each grid point, update the electric potential using the formula:

..         .. math::

..             \phi_{i,j,k}^{n+1} = (1 - \omega) \phi_{i,j,k}^n
..             + \frac{\omega}{6} (\phi_{i+1,j,k}^{n+1} + \phi_{i-1,j,k}^{n} + \phi_{i,j+1,k}^{n+1} + \phi_{i,j-1,k}^{n} + \phi_{i,j,k+1}^{n+1} + \phi_{i,j,k-1}^{n}) - \frac{\rho_{i,j,k}}{6 \epsilon_0}


..     b. Check for convergence by calculating the residual:
    
..         .. math::

..             \text{residual} = \max(| \nabla^2 \phi_{i,j,k}^{n} - \frac{\rho_{i,j,k}}{\epsilon_0} |) 
        

.. Conjugate Gradient Method
.. ++++++++++++++++++++++++++

.. The Conjugate Gradient Method is an iterative method used to solve linear systems of equations. The Conjugate Gradient Method is used to solve the Poisson equation for the electric field in PyPIC3D.


First-Order Yee Solvers
-----------------------
PyPIC3D evolves the electric and magnetic fields on a staggered Yee grid using
either a pseudo-spectral time-domain (PSTD) solver or a centered finite
difference time-domain (FDTD) solver.

Spectral (PSTD)
***************
.. math::

        \tilde{\mathbf{E}}^{n+1} = \tilde{\mathbf{E}}^n + C^2 \Delta t ( ik \times \tilde{\mathbf{B}}^n - \mu_0 \tilde{J^n} )

        \tilde{\mathbf{B}}^{n+1} = \tilde{\mathbf{B}}^n - \Delta t ( ik \times \tilde{\mathbf{E}}^{n+1} )

Finite Difference (FDTD)
************************

.. math::

    \mathbf{E}^{n+1} = \mathbf{E}^n + \Delta t \left( \nabla \times \mathbf{B}^n - \frac{\mathbf{J}^n}{\epsilon_0} \right)

    \mathbf{B}^{n+1} = \mathbf{B}^n - \Delta t \left( \nabla \times \mathbf{E}^{n+1} \right)




Vector Potential Solver
-----------------------

PyPIC3D also includes a vector potential formulation that evolves :math:`\mathbf{A}`
under the temporal gauge:

.. math::
    \mathbf{A}_0 = 0

    \frac{\partial^2_t \mathbf{A}}{c^2} - \nabla^2 \mathbf{A} = \mu_0 \mathbf{J} - \nabla (\nabla \cdot \mathbf{A})


Under this formulation, the electric and magnetic fields are calculated from the vector potential as the following:

.. math::
    \mathbf{E} = -\partial_t \mathbf{A}

    \mathbf{B} = \nabla \times \mathbf{A}

The vector potential is advanced in time using a backward Euler update, and the
electric and magnetic fields are reconstructed with centered finite differences.

Curl-Curl Solver
----------------

The curl-curl solver advances the fields using a second-order finite difference
formulation that evolves both :math:`\mathbf{E}` and :math:`\mathbf{B}` from
their previous two time levels. Enable it with
``simulation_parameters.solver = "curl_curl"``.

Electrostatic Mode
------------------

For electrostatic simulations, set ``simulation_parameters.electrostatic = true``.
In this mode, the electric field is computed from the Poisson equation each
time step and the magnetic field remains static.
