import numpy as np

# Define the grid size
grid_size = (50, 50, 50)

# Create a 30x30x30 grid of zeros for the E field components
Ex = np.zeros(grid_size)
Ey = np.zeros(grid_size)
Ez = np.zeros(grid_size)

# Define the wave parameters
amplitude = 1000.0
wavelength = 1e-2/7
phase = 0.0

# Create the wave in the E field components
x = np.linspace(0, 1e-2, grid_size[0])
y = np.linspace(0, 1e-2, grid_size[1])
z = np.linspace(0, 1e-2, grid_size[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

#Ex[:59,:,:] = amplitude * np.sin(2 * np.pi * X[:59,:,:] / wavelength + phase)

Ex[:,:,:] = 1*Ex[:,:,:]

#Ey[:59,:,:] = amplitude * np.sin(2 * np.pi * X[:59,:,:] / wavelength + phase)
#Ez[:59,:,:] = amplitude * np.sin(2 * np.pi * X[:59,:,:] / wavelength + phase)
#Ey = amplitude * np.sin(2 * np.pi * Y / wavelength + phase)
#Ez = amplitude * np.sin(2 * np.pi * Z / wavelength + phase)
import matplotlib.pyplot as plt

# # Plot slices of the E field components
# slice_index = 15

# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot Ex slice
# cax1 = axs[0].imshow(Ex[:, :, slice_index], origin='lower', extent=(0, grid_size[0], 0, grid_size[1]))
# axs[0].set_title('Ex slice at z={}'.format(slice_index))
# fig.colorbar(cax1, ax=axs[0])

# # Plot Ey slice
# cax2 = axs[1].imshow(Ey[:, :, slice_index], origin='lower', extent=(0, grid_size[0], 0, grid_size[1]))
# axs[1].set_title('Ey slice at z={}'.format(slice_index))
# fig.colorbar(cax2, ax=axs[1])

# # Plot Ez slice
# cax3 = axs[2].imshow(Ez[:, :, slice_index], origin='lower', extent=(0, grid_size[0], 0, grid_size[1]))
# axs[2].set_title('Ez slice at z={}'.format(slice_index))
# fig.colorbar(cax3, ax=axs[2])

# Plot 1D slice of the E field components along x-axis at y=15, z=15
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, Ex[:, 15, 15], label='Ex')
ax.plot(x, Ey[:, 15, 15], label='Ey')
ax.plot(x, Ez[:, 15, 15], label='Ez')

ax.set_title('1D slice of E field components at y=15, z=15')
ax.set_xlabel('x')
ax.set_ylabel('E field')
ax.legend()

plt.tight_layout()
plt.tight_layout()
plt.show()

# Save the E field components to .npy files
np.save('/home/christopherwoolford/Documents/PyPIC3D/longitudinal_Ex.npy', Ex)
np.save('/home/christopherwoolford/Documents/PyPIC3D/longitudinal_Ey.npy', Ey)
np.save('/home/christopherwoolford/Documents/PyPIC3D/longitudinal_Ez.npy', Ez)