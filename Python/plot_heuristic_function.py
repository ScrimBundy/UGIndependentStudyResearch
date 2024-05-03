import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def G(x, t, l=0):
    if l == 0:
        return np.where(x <= t, (x / t), 1 + (x - t))
    else:
        v = (t - l) / 2.0
        a = 1 / (v * v)
        return np.where(x <= t, np.where(x >= l, a*(x - v - l)**2, 1 + (l - x)), 1 + (x - t))


def F(x, y, z, n):
    xy_target = (2.0 * n) / 3.0
    xy_lower = (1.0 * n) / 12.0
    z_target = (2.0 * math.sqrt(2.0)) * math.sqrt(n)

    score = G(x, xy_target, xy_lower)
    score += G(y, xy_target, xy_lower)
    score += G(z, z_target)

    return score


# Function to generate custom colormap
def generate_custom_cmap(max_f):
    colors = [(0, 'cyan'), (0.5 / max_f, 'cyan'),      # Turquoise
              (1.5 / max_f, 'green'),                # Green
              (3 / max_f, 'yellow'),                # Yellow
              (1, 'red')]                       # Red
    return LinearSegmentedColormap.from_list('custom_colormap', colors)


def func():
    # Define the bounds
    n = 20
    resolution = 100
    x = np.linspace(0, n, resolution)
    y = np.linspace(0, n, resolution)

    # Create a meshgrid for the boundary where x + y + z = n
    boundary_xy = np.linspace(0, n, resolution)
    boundary_x, boundary_y = np.meshgrid(boundary_xy, boundary_xy)
    boundary_z = n - boundary_x - boundary_y

    # Mask the boundary points where any coordinate is non-positive.
    # Note floating point error on z coordinate
    mask = (boundary_x >= 0) & (boundary_y >= 0) & (boundary_z >= -0.00001)
    boundary_x = np.ma.masked_where(~mask, boundary_x)
    boundary_y = np.ma.masked_where(~mask, boundary_y)
    boundary_z = np.ma.masked_where(~mask, boundary_z)

    # Flatten the arrays
    boundary_x_flat = boundary_x.flatten()
    boundary_y_flat = boundary_y.flatten()
    boundary_z_flat = boundary_z.flatten()

    # Calculate F values for points on the boundary
    F_values_boundary = F(boundary_x_flat, boundary_y_flat, boundary_z_flat, n)

    # Define custom colormap
    custom_cmap = generate_custom_cmap(np.max(F_values_boundary))

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the function values on the boundary
    ax.scatter(boundary_x_flat, boundary_y_flat, boundary_z_flat, c=F_values_boundary, cmap=custom_cmap)

    # Set labels and title
    ax.set_xlabel('Front Half')
    ax.set_ylabel('Back Half')
    ax.set_zlabel('Separator')
    ax.set_title('Heuristic Graph')

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(ax.collections[0], shrink=0.5, aspect=5)
    cbar.set_label('Heuristic Value')

    plt.show()


if __name__ == '__main__':
    func()