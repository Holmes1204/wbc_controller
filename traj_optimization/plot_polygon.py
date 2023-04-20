import matplotlib.pyplot as plt
import numpy as np

def plot_convex_shape(vertices, color='k'):
    """Plot a convex shape given its vertices using matplotlib."""
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    # plt.fill(x, y, color=color)
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    plt.plot(x, y, color=color)




# Example usage:
if __name__ =="__main__":
    # Define convex shape vertices
    shape1 = np.array([[0, 0], [1, 0], [1, 0.5], [0, 1]])
    shape2 = np.array([[0.5, 0.5], [1.2, 0.5], [1.5, 1.5], [0.5, 1.5]])

    # Plot shape1 in blue color
    plot_convex_shape(shape1, color='b')
    # Plot shape2 in red color
    plot_convex_shape(shape2, color='r')

    # Set plot properties
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convex Shapes')
    plt.grid(True)

    # Show the plot
    plt.show()