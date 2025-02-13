import math
from collections import namedtuple
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
Point = namedtuple("Point", ["x", "y", "z"])
#loads data, expecting it to be in cartesian form


def main():
    points = load_data()
    max_dist, pair = diameter(points)
    if pair:
        plotdiameter(points, pair)
    print("The diameter given by the points is " + str(max_dist))

    coefficients = conic_trajectory(points)
    plot_conic(coefficients, points)


def load_data():
    return np.random.rand(30, 3)
def get_convexhull(points) -> ConvexHull:
    hull = ConvexHull(points)
    return hull
#computes diameter
def signed_area(a, b, c) -> float:
    #a, b, c are the points
    #compute v1 and v2:
    v1 = b - a
    v2 = c - a

    #cross product:
    cross_product = np.cross(v1, v2)

    #compute signed area
    return np.linalg.norm(cross_product)

def diameter(points):
    convexhull = get_convexhull(points)
    antipodal = set()
    vertices = convexhull.vertices
    m = len(convexhull.vertices)
    k = 2

    #find initial k
    while signed_area(points[vertices[m - 1]], points[vertices[0]], points[vertices[k + 1]]) > signed_area(points[vertices[m - 1]], points[vertices[0]], points[vertices[k]]):
        k += 1
    
    #find antipodal pairs
    i = 1
    j = k
    while (i <= k and j <= m):
        antipodal.add((tuple(points[vertices[i]]), tuple(points[vertices[j]])))
        while (signed_area(points[vertices[i]], points[vertices[i + 1]], points[vertices[j + 1]]) > signed_area(points[vertices[i]], points[vertices[i + 1]], points[vertices[j]]) and j < m):
            antipodal.add((tuple(points[vertices[i]]), tuple(points[vertices[j]])))
            j += 1
        i += 1
    #find max squared distance
    max_dist = 0
    pair = None
    for p1, p2 in antipodal:
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        if distance > max_dist:
            max_dist = distance
            pair = (p1, p2)

    return max_dist, pair
def plotdiameter(points, pair):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color = 'blue', label = 'Points')
    x_vals = [pair[0][0], pair[1][0]]
    y_vals = [pair[0][1], pair[1][1]]
    z_vals = [pair[0][2], pair[1][2]]
    ax.plot(x_vals, y_vals, z_vals, color='red', linewidth = 2, label="Diameter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def conic_trajectory(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    X = np.column_stack((x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, np.ones(x.shape)))
    _, _, V = np.linalg.svd(X)
    coeffs = V[-1, :]
    return coeffs

def plot_conic(coeffs, points):
    A, B, C, D, E, F, G, H, I, J = coeffs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Points')

    x_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
    y_vals = np.linspace(points[:, 1].min(), points[:, 1].max(), 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.sqrt(np.abs(-(A * X**2 + B * Y**2 + D *X * Y + G * X + H * Y + J) / C))
    
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='none')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()







    