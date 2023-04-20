import numpy as np

def get_normals(vertices):
    """Compute the normals of a convex shape given its vertices."""
    normals = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        edge = vertices[(i + 1) % num_vertices] - vertices[i]
        normal = np.array([-edge[1], edge[0]])
        normal = normal.astype(np.float64)
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return normals

def project(shape, axis):
    """Project a convex shape onto an axis."""
    min_proj = max_proj = np.dot(axis, shape[0])
    for vertex in shape[1:]:
        proj = np.dot(axis, vertex)
        if proj < min_proj:
            min_proj = proj
        elif proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

def check_collision(shape1, shape2):
    """Check collision between two convex shapes using SAT."""
    axes = get_normals(shape1) + get_normals(shape2)
    for axis in axes:
        min_proj1, max_proj1 = project(shape1, axis)
        min_proj2, max_proj2 = project(shape2, axis)
        if max_proj1 < min_proj2 or max_proj2 < min_proj1:
            return False
    return True

# Example usage:

# Define convex shapes as lists of 2D vertices in clockwise order
shape1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
shape2 = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])

# Check collision
if check_collision(shape1, shape2):
    print("Collision detected!")
else:
    print("No collision.")