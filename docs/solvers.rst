Solvers Overview
================

.. math::

    \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}

    \nabla \cdot \mathbf{B} = 0

    \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}

    \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}

PyPIC3D provides several solvers for solving the above equations, including: spectral and finite difference solvers.

Spectral Solver
----------------
PyPIC3D has a explicit spectral solver that evolves the electric and magnetic fields.

.. math::

        \mathbf{E}^{n+1} = \mathbf{E}^n + \Delta t ( C^2 * ik \times \mathbf{B}^n - \frac{J^n}{\epsilon_0} )

        \mathbf{B}^{n+1} = \mathbf{B}^n - \Delta t ( ik \times \mathbf{E}^{n+1} )


Finite Difference Solvers
--------------------------
Finite difference solvers are numerical methods that approximate derivatives by using differences between function values at discrete points.

.. math::

    \left( \nabla \times \mathbf{E} \right)_x = \frac{E_z(y+1) - E_z(y-1)}{2 \Delta y} - \frac{E_y(z+1) - E_y(z-1)}{2 \Delta z}

    \left( \nabla \times \mathbf{E} \right)_y = \frac{E_x(z+1) - E_x(z-1)}{2 \Delta z} - \frac{E_z(x+1) - E_z(x-1)}{2 \Delta x}

    \left( \nabla \times \mathbf{E} \right)_z = \frac{E_y(x+1) - E_y(x-1)}{2 \Delta x} - \frac{E_x(y+1) - E_x(y-1)}{2 \Delta y}

    \left( \nabla \times \mathbf{B} \right)_x = \frac{B_z(y+1) - B_z(y-1)}{2 \Delta y} - \frac{B_y(z+1) - B_y(z-1)}{2 \Delta z}

    \left( \nabla \times \mathbf{B} \right)_y = \frac{B_x(z+1) - B_x(z-1)}{2 \Delta z} - \frac{B_z(x+1) - B_z(x-1)}{2 \Delta x}

    \left( \nabla \times \mathbf{B} \right)_z = \frac{B_y(x+1) - B_y(x-1)}{2 \Delta x} - \frac{B_x(y+1) - B_x(y-1)}{2 \Delta y}

    \mathbf{E}^{n+1} = \mathbf{E}^n + \Delta t \left( \nabla \times \mathbf{B}^n - \frac{\mathbf{J}^n}{\epsilon_0} \right)

    \mathbf{B}^{n+1} = \mathbf{B}^n - \Delta t \left( \nabla \times \mathbf{E}^{n+1} \right)
