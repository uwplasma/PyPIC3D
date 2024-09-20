# Christopher Woolford, Sept 9 2024
# This python script contains the definition of the model used to calculate
# the preconditioner for the Poisson equation.

import equinox as eqx
import jax
import jax.numpy as jnp
# import external libraries

class GraphAttentionLayer(eqx.Module):
    n_heads: int
    is_concat: bool
    n_hidden: int
    linearinput: eqx.nn.Linear
    attn: eqx.nn.Linear
    activation: jax.nn.leaky_relu
    softmax: jax.nn.softmax
    dropout: eqx.nn.Dropout
    # define the type of attributes
    def __init__(self, in_features, out_features, n_heads, key, is_concat=True, dropout=0.6, lrelu_slope=0.2):
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linearinput = eqx.nn.Linear(in_features, self.n_hidden * n_heads, use_bias=False, key=key)
        # linear input layer

        self.attn = eqx.nn.Linear(self.n_hidden * 2, 1, use_bias=False, key=key)
        # calculate attention scores

        self.activation = jax.nn.leaky_relu(lrelu_slope)
        # activation for attention score

        self.softmax = jax.nn.softmax
        # softmax to calculate attention weights

        self.dropout = eqx.nn.Dropout(dropout)
        # dropout layer

    def __call__(self, x, adj):
        """
        Applies the graph attention layer to the input data.

        Args:
            x: Input data of shape (N, F).
            adj: Adjacency matrix of shape (N, N).

        Returns:
            Output data of shape (N, F').
        """
        N = x.shape[0]
        # get the number of nodes

        h = self.linearinput(x)
        # apply the linear input layer

        h = jnp.reshape(h, (N, self.n_heads, self.n_hidden))
        # reshape the hidden layer

        h_repeat = jnp.tile(h, (1, 1))
        # repeat the hidden layer

        h_repeat_inverleave = jnp.repeat(h_repeat, N, dim=0)

        h_concat = jnp.concatenate([h_repeat_inverleave, h_repeat], axis=-1)

        h_concat = jnp.reshape(h_concat, (N, N, self.n_heads, 2*self.n_hidden))
        # concatenate the hidden layer

        e = self.activation(self.attn(h_concat))
        
        assert adj.shape[0] == 1 or adj.shape[0] == N
        assert adj.shape[1] == 1 or adj.shape[1] == N
        # check the shape of the adjacency matrix

        a = self.softmax(e)

        a = self.dropout(a)
        # apply dropout

        attn_res = jnp.einsum('ijh, jhf -> ihf', a, h)

        if self.is_concat:
            return jnp.reshape(attn_res, (N, self.n_heads * self.n_hidden))
        else:
            return jnp.mean(attn_res, axis=1)



class PoissonPrecondition(eqx.Module):
    Nx: int
    Ny: int
    Nz: int
    mlp1: eqx.Module
    #mlp2: eqx.Module
    #gat : eqx.Module
    #adj : jnp.ndarray
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
        self.mlp1 = eqx.nn.MLP(int(2*self.Nx*self.Ny*self.Nz), int(self.Nx*self.Ny), hidden_dim, activation=jax.nn.hard_tanh,  depth=3, key=key )
        #self.mlp2 = eqx.nn.MLP( int(2*self.Nx*self.Ny), self.Nx*self.Ny, hidden_dim, depth=2, key=key )
        # define the multilayer perceptrons

        #self.gat = GraphAttentionLayer( int(2*self.Nx*self.Ny), int(self.Nx*self.Ny), 2, key=key, is_concat=True, dropout=0.6, lrelu_slope=0.2)
        #self.adj = jnp.ones((int(2*self.Nx*self.Ny), int(2*self.Nx*self.Ny)))

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
        x = jnp.concatenate( (phi, rho) )
        # add the input
        x = jnp.reshape(x, (2*self.Nx*self.Ny*self.Nz))
        x = self.mlp1(x)
        #x = self.gat(x, self.adj)
        #x = self.mlp2(x)
        x = jnp.reshape(x, (self.Nx, self.Ny))
        return x
    # define the forward pass of the model