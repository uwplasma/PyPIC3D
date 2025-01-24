import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.fft import fft_slab_decomposition, fft_pencil_decomposition

jax.config.update("jax_enable_x64", True)

class TestFFTDecompositions(unittest.TestCase):

    def setUp(self):
        # Create a sample 3D field for testing
        self.field = jnp.ones((25, 25, 25))

    def test_fft_slab_decomposition(self):
        # Perform FFT using slab decomposition
        result = fft_slab_decomposition(self.field)

        # Check if the result has the same shape as the input
        self.assertEqual(result.shape, self.field.shape)

        # Check if the result is a complex array
        self.assertTrue(jnp.iscomplexobj(result))

    # def test_fft_slab_decomposition_known_freq(self):
    #     # Create a sample 3D field with a known frequency
    #     freq = 2.0
    #     x = jnp.linspace(0, 2 * jnp.pi, 50)
    #     y = jnp.linspace(0, 2 * jnp.pi, 50)
    #     z = jnp.linspace(0, 2 * jnp.pi, 50)
    #     X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    #     self.field = jnp.sin(freq * X)

    #     # Perform FFT using slab decomposition
    #     result = fft_slab_decomposition(self.field, axis=0)
    #     # along x-axis

    #     import matplotlib.pyplot as plt

    #     # Plot a 1D slice along the x-axis
    #     plt.plot(jnp.abs(result[:, 0, 0]))
    #     plt.title('1D Slice along x-axis of FFT result')
    #     plt.xlabel('Index')
    #     plt.ylabel('Magnitude')
    #     plt.grid(True)
    #     plt.show()
    #     # Check if the result has the expected frequency component
    #     expected_freq_component = 3 * (8 / 2) * (8 / 2) * (8 / 2)
    #     self.assertAlmostEqual(jnp.abs(result[1, 0, 0]), expected_freq_component, places=5)
    #     self.assertAlmostEqual(jnp.abs(result[0, 1, 0]), expected_freq_component, places=5)
    #     self.assertAlmostEqual(jnp.abs(result[0, 0, 1]), expected_freq_component, places=5)


    def test_fft_pencil_decomposition(self):
        # Perform FFT using pencil decomposition
        result = fft_pencil_decomposition(self.field)

        # Check if the result has the same shape as the input
        self.assertEqual(result.shape, self.field.shape)

        # Check if the result is a complex array
        self.assertTrue(jnp.iscomplexobj(result))


    # def test_fft_pencil_decomposition_known_freq(self):
    #     # Create a sample 3D field with a known frequency
    #     freq = 2.0
    #     x = jnp.linspace(0, 2 * jnp.pi, 8)
    #     y = jnp.linspace(0, 2 * jnp.pi, 8)
    #     z = jnp.linspace(0, 2 * jnp.pi, 8)
    #     X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    #     self.field = jnp.sin(freq * X) + jnp.sin(freq * Y) + jnp.sin(freq * Z)
        
    #     # Perform FFT using pencil decomposition
    #     result = fft_pencil_decomposition(self.field)
        
    #     # Check if the result has the expected frequency component
    #     expected_freq_component = 3 * (8 / 2) * (8 / 2) * (8 / 2)
    #     self.assertAlmostEqual(jnp.abs(result[1, 0, 0]), expected_freq_component, places=5)
    #     self.assertAlmostEqual(jnp.abs(result[0, 1, 0]), expected_freq_component, places=5)
    #     self.assertAlmostEqual(jnp.abs(result[0, 0, 1]), expected_freq_component, places=5)

if __name__ == '__main__':
    unittest.main()