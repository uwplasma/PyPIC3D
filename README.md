## PyPIC3D

PyPIC3D is a 3D particle-in-cell (PIC) simulation code developed in Python, leveraging the Jax library for high-performance numerical computing. The primary goal of this project is to explore the feasibility of using Python and Jax for plasma simulations, which are traditionally performed using languages like C++ and Julia.

### Key Features

- **Autodifferentiation**: Jax's built-in autodifferentiation capabilities allow for efficient gradient computations, which can be highly beneficial for optimization and surrogate modeling in plasma simulations.
- **Benchmarking**: The code is benchmarked against equivalent implementations in Julia and C++ to evaluate performance and accuracy.
- **Plasma Simulation**: Designed to simulate plasma behavior in three dimensions, providing insights into complex plasma dynamics.

### Applications

PyPIC3D can be used for various plasma simulation applications, including but not limited to:

- **Stellarator Simulations**: Investigating the behavior of plasma in stellarator devices, which are used in fusion research.
- **Optimization**: Utilizing autodifferentiation for optimizing plasma configurations and parameters.
- **Surrogate Modeling**: Creating surrogate models to approximate plasma behavior, reducing computational costs for large-scale simulations.

### Getting Started

To get started with PyPIC3D, ensure you have Python and Jax installed. Follow the installation instructions and examples provided in the repository to set up and run your simulations.

### Code Structure

The PyPIC3D codebase is organized into several key sections:

- **Core**: Contains the core functionalities of the PIC simulation, including particle push, field solve, and boundary conditions.
- **Utils**: Utility functions and helper scripts that support the main simulation code, such as data processing and visualization tools.
- **Examples**: Example scripts demonstrating how to set up and run different types of plasma simulations using PyPIC3D.
- **Tests**: Unit tests and integration tests to ensure the correctness and stability of the codebase.

Each section is designed to be modular and easily extensible, allowing users to customize and expand the functionality of PyPIC3D according to their needs.

### Contributing

Contributions to PyPIC3D are welcome. Please refer to the contributing guidelines in the repository for more information on how to contribute.

### License

This project is licensed under the MIT License. See the LICENSE file for details.