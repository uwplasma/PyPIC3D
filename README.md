August 1st, 2024

This project is the start of my PhD work with Dr. Rogerio Jorge. I am writing a 3D particle-in-cell code in Python using the library called "Jax" and benchmarking the code against similar variants I will make in Julia and C++. I am doing this to evaluate
whether or not Python/Jax can potentially be used for plasma simulations. Jax has built-in autodifferentiation capabilities and if it can be used for plasma simulations, autodifferentiation could be a very powerful tool for optimization and surrogate
modelling in various plasma sims such as Stellerator simulations.

For the purposes of this study, the magnetic field update will be neglected due to time limitations.