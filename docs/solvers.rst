Solvers Overview
================

.. math::

    \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}

    \nabla \cdot \mathbf{B} = 0

    \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}

    \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}

PyPIC3D provides several solvers for solving the above equations, including: spectral and finite difference solvers.

Initialization
--------------

The initial electric field is calculated using the following equation:

.. math::

    \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}

PyPIC3D can solve for the initial electric field using both spectral and finite difference solvers.
The magnetic field is initialized to zero.

Successive Over-Relaxation (SOR)
++++++++++++++++++++++++++++++++

The Successive Over-Relaxation (SOR) method is an iterative method used to solve linear systems of equations. The SOR method is used to solve the Poisson equation for the electric field in PyPIC3D.
The algorithm is as follows:

1. Initialize the electric potential to an array of random values.
2. Iterate until convergence:
    a. For each grid point, update the electric potential using the formula:

        .. math::

            \phi_{i,j,k}^{n+1} = (1 - \omega) \phi_{i,j,k}^n
            + \frac{\omega}{6} (\phi_{i+1,j,k}^{n+1} + \phi_{i-1,j,k}^{n} + \phi_{i,j+1,k}^{n+1} + \phi_{i,j-1,k}^{n} + \phi_{i,j,k+1}^{n+1} + \phi_{i,j,k-1}^{n}) - \frac{\rho_{i,j,k}}{6 \epsilon_0}


    b. Check for convergence by calculating the residual:
    
        .. math::

            \text{residual} = \max(| \nabla^2 \phi_{i,j,k}^{n} - \frac{\rho_{i,j,k}}{\epsilon_0} |) 
        

Conjugate Gradient Method
++++++++++++++++++++++++++

The Conjugate Gradient Method is an iterative method used to solve linear systems of equations. The Conjugate Gradient Method is used to solve the Poisson equation for the electric field in PyPIC3D.


Spectral Solver
----------------
PyPIC3D has a explicit spectral solver that evolves the electric and magnetic fields.

.. math::

        \tilde{\mathbf{E}}^{n+1} = \tilde{\mathbf{E}}^n + C^2 \Delta t ( ik \times \tilde{\mathbf{B}}^n - \mu_0 \tilde{J^n} )

        \tilde{\mathbf{B}}^{n+1} = \tilde{\mathbf{B}}^n - \Delta t ( ik \times \tilde{\mathbf{E}}^{n+1} )


Finite Difference Solvers
--------------------------
Finite difference solvers are numerical methods that approximate derivatives by using differences between function values at discrete points.

.. math::

    \mathbf{E}^{n+1} = \mathbf{E}^n + \Delta t \left( \nabla \times \mathbf{B}^n - \frac{\mathbf{J}^n}{\epsilon_0} \right)

    \mathbf{B}^{n+1} = \mathbf{B}^n - \Delta t \left( \nabla \times \mathbf{E}^{n+1} \right)
