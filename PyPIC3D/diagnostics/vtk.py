import vtk
from vtk.util import numpy_support
import numpy as np
import os
from pyevtk.hl import gridToVTK, pointsToVTK
import jax.numpy as jnp

def plot_fields(fieldx, fieldy, fieldz, t, name, dx, dy, dz):
    """
    Plot the fields in a 3D grid.

    Args:
        fieldx (ndarray): Array representing the x-component of the field.
        fieldy (ndarray): Array representing the y-component of the field.
        fieldz (ndarray): Array representing the z-component of the field.
        t (float): Time value.
        name (str): Name of the field.
        dx (float): Spacing between grid points in the x-direction.
        dy (float): Spacing between grid points in the y-direction.
        dz (float): Spacing between grid points in the z-direction.

    Returns:
        None
    """
    Nx = fieldx.shape[0]
    Ny = fieldx.shape[1]
    Nz = fieldx.shape[2]
    x = np.linspace(0, Nx, Nx) * dx
    y = np.linspace(0, Ny, Ny) * dy
    z = np.linspace(0, Nz, Nz) * dz

    # Create directory if it doesn't exist
    directory = "./plots/fields"
    if not os.path.exists(directory):
        os.makedirs(directory)

    gridToVTK(f"./plots/fields/{name}_{t:09}", x, y, z,   \
            cellData = {f"{name}_x" : np.asarray(fieldx), \
             f"{name}_y" : np.asarray(fieldy), f"{name}_z" : np.asarray(fieldz)}) 
# plot the electric fields in the vtk file format


def plot_vtk_particles(particles, t, path):
    """
    Plot the particles in VTK format.

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        path (str): The path to save the plot.

    Returns:
        None
    """
    if not os.path.exists(f"{path}/data/particles"):
        os.makedirs(f"{path}/data/particles")

    particle_names = [species.get_name().replace(" ", "") for species in particles]
    for species in particles:
        name = species.get_name().replace(" ", "")
        x, y, z = map(np.asarray, species.get_position())
        vx, vy, vz = map(np.asarray, species.get_velocity())
        # Get the position and velocity of the particles
        q = np.asarray( species.get_charge() * np.ones_like(vx) )
        # Get the charge of the particles

        pointsToVTK(f"{path}/data/particles/{name}.{t:09}", x, y, z, \
                {"v": (vx, vy, vz), "q": q})
        # save the particles in the vtk file format


def plot_field_slice_vtk(field_slices, field_names, slice, grid, t, name, path, world):
    """
    Plot a slice of a field in VTK format using Python VTK library.

    Args:
        field_slices (list): List of 2D field slices to be plotted.
        field_names (list): List of field names corresponding to the slices.
        slice (int): Slice direction (0=x-slice, 1=y-slice, 2=z-slice, 3=full 3D).
        grid (tuple): The grid dimensions (x, y, z).
        t (int): The time step.
        name (str): The name of the field.
        path (str): The path to save the plot.
        world (dict): World parameters containing grid information.

    Returns:
        None
    """

    x, y, z = grid
    nx, ny, nz = world['Nx'], world['Ny'], world['Nz']
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    if not os.path.exists(f"{path}/data/{name}_slice"):
        os.makedirs(f"{path}/data/{name}_slice")
    # Create directory if it doesn't exist

    # Create VTK structured grid
    structured_grid = vtk.vtkStructuredGrid()

    if slice == 0:
        x = np.asarray([x[nx//2]])
    elif slice == 1:
        y = np.asarray([y[ny//2]])
    elif slice == 2:
        z = np.asarray([z[nz//2]])

    structured_grid.SetDimensions(x.shape[0], y.shape[0], z.shape[0])
    # Set the dimensions of the structured grid based on the slice type

    # Efficiently create all grid points using numpy meshgrid and bulk insert
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    structured_grid.SetPoints(points)
    # Create points for the structured grid based on the slice type

    for idx, (field_slice, field_name) in enumerate(zip(field_slices, field_names)):
        # Ensure field_slice is 2D and handle VTK ordering
        field_data = np.asarray(field_slice)
        if field_data.ndim == 2:
            # VTK expects data in (k,j,i) order, but our slice is typically (j,k)
            field_data = field_data.T.flatten()
        else:
            field_data = field_data.flatten()

        vtk_array = numpy_support.numpy_to_vtk(field_data)
        vtk_array.SetName(field_name)
        structured_grid.GetPointData().AddArray(vtk_array)

    # Write the VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(f"{path}/data/{name}_slice/{name}_slice_{t:09}.vtk")
    writer.SetInputData(structured_grid)
    writer.Write()


def plot_vectorfield_slice_vtk(field_slices, field_names, slice, grid, t, name, path, world):
    """
    Plot a slice of a field in VTK format as vector data using Python VTK library.

    Args:
        field_slices (list): List of 2D field slices to be plotted. Should be [Fx, Fy, Fz] for vector fields.
        field_names (list): List of field names corresponding to the slices (e.g., ['Ex', 'Ey', 'Ez']).
        slice (int): Slice direction (0=x-slice, 1=y-slice, 2=z-slice, 3=full 3D).
        grid (tuple): The grid dimensions (x, y, z).
        t (int): The time step.
        name (str): The name of the field.
        path (str): The path to save the plot.
        world (dict): World parameters containing grid information.

    Returns:
        None
    """

    x, y, z = grid
    nx, ny, nz = world['Nx'], world['Ny'], world['Nz']
    dx, dy, dz = world['dx'], world['dy'], world['dz']

    if not os.path.exists(f"{path}/data/{name}_slice"):
        os.makedirs(f"{path}/data/{name}_slice")

    # Handle slicing
    if slice == 0:
        x = np.asarray([x[nx//2]])
    elif slice == 1:
        y = np.asarray([y[ny//2]])
    elif slice == 2:
        z = np.asarray([z[nz//2]])

    # Create VTK structured grid
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(x.shape[0], y.shape[0], z.shape[0])

    # Create grid points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((Z.ravel(), Y.ravel(), X.ravel()))
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    structured_grid.SetPoints(points)

    # Stack field slices as vector data
    # Each field_slice should be 2D, shape (len(x), len(y)) or similar
    # We flatten and stack them as (N, 3) for VTK vector
    for field_slice, field_name in zip(field_slices, field_names):
        field_arrays = []
        for comp in field_slice:
            field_data = np.asarray(comp)
            # convert to np array
            if field_data.ndim == 2:
                field_data = field_data.T.flatten()  # VTK expects Fortran order
            else:
                field_data = field_data.flatten()
            field_arrays.append(field_data)
        # Stack as (N, 3)
        vector_data = np.stack(field_arrays, axis=-1)
        # If shape is (N, 3), convert to VTK
        vtk_vector_array = numpy_support.numpy_to_vtk(vector_data, deep=True)
        vtk_vector_array.SetName(f"{field_name}_vector")
        structured_grid.GetPointData().AddArray(vtk_vector_array)

    # Write the VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(f"{path}/data/{name}_slice/{name}_slice_{t:09}.vtk")
    writer.SetInputData(structured_grid)
    writer.Write()


def write_slice(field_slice, x1, x2, t, name, path, dt):
    """
    Plots a slice of a field and saves it in VTK format.
    Parameters:
    field_slice (numpy.ndarray): The 2D slice of the field to be plotted.
    x1 (numpy.ndarray): The x-coordinates of the field slice.
    x2 (numpy.ndarray): The y-coordinates of the field slice.
    t (int): The time step or index for the slice.
    name (str): The name of the field or slice.
    path (str): The directory path where the VTK file will be saved.
    dt (float): The time step size (not used in the function but included in parameters).
    Returns:
    None
    """

    x3 = np.zeros(1)

    field_slice = jnp.expand_dims(field_slice, axis=-1)

    gridToVTK(f"{path}/data/{name}_slice/{name}_slice_{t:09}", x1, x2, x3,  \
            cellData = {f"{name}" : field_slice})
    # plot the slice of the field in the vtk file format