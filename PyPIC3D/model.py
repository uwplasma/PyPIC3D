# Christopher Woolford, Sept 9 2024
# This python script contains the definition of the model used to calculate
# the preconditioner for the Poisson equation.

import equinox as eqx
import jax
import jax.numpy as jnp
# import external libraries

class PoissonPrecondition(eqx.Module):
    Nx: int
    Ny: int
    Nz: int
    mlp1: eqx.Module
    conv1: eqx.Module
    # define type of attributes
    def __init__(self, Nx, Ny, Nz, hidden_dim, key):
        """
        Initializes the model object.

        Args:
            num_layers (int): Number of hidden layers in the model.
            Nx (int): Number of elements in the x-direction.
            Ny (int): Number of elements in the y-direction.
            Nz (int): Number of elements in the z-direction.
            hidden_dim (int): Dimension of the hidden layers.
            key: Key for random number generation.
        """

        super().__init__()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        # define the dimensions of the input
        assert self.Nx == self.Ny == self.Nz
        self.mlp1 = eqx.nn.MLP( in_size=int(self.Nx*self.Ny*self.Nz), out_size=int(self.Nx*self.Ny), width_size=hidden_dim, activation=jax.nn.relu,  depth=5, key=key )
        kernel_size = 3
        padding     = (kernel_size-1) // 2
        self.conv1 = eqx.nn.Conv1d(2,1, kernel_size=kernel_size, padding=padding, key=key)
        # define the multilayer perceptron layers and convolutional layers
    def __call__(self, phi, rho):
        """
        Applies the neural network model to the input data.

        Args:
            phi: Input data of shape (Nx, Ny, Nz).
            rho: Input data of shape (Nx, Ny, Nz).


        Returns:
            Output data of shape (Nx, Ny).
        """

        phi = jnp.reshape(phi, (self.Nx*self.Ny*self.Nz))
        rho = jnp.reshape(rho, (self.Nx*self.Ny*self.Nz))
        # reshape the input
        x = jnp.stack( (phi, rho), axis=0 )
        x = self.conv1(x)
        x = x.squeeze()
        x = self.mlp1(x)
        x = jnp.reshape(x, (self.Nx, self.Ny))
        return x
    # define the forward pass of the model
