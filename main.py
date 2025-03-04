import math
from collections import namedtuple
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from ellipse import Ellipse
from convexhull import ConvexHullCalc
Point = namedtuple("Point", ["x", "y", "z"])
#loads data, expecting it to be in cartesian form


def main():
    theta = np.linspace(0, 2 * np.pi, 30)
    a = 5
    b = 3
    x = a * np.cos(theta) + np.random.normal(0, 0.2, size=theta.shape)
    y = b * np.cos(theta) + np.random.normal(0, 0.2, size=theta.shape)

    z = np.random.normal(0, 0.2, size = theta.shape) #random height variations
    points = np.vstack([x, y, z]).T

    hullanalyzer = ConvexHullCalc(points)
    max_dist, pair = hullanalyzer.diameter(points)

    print("Diameter of convex hull: ", max_dist)
    if pair:
        hullanalyzer.plotdiameter(points, pair)
    
    ellipsefit = Ellipse(points)
    ellipse_coefficients = ellipsefit.fit2d()
    ellipse_3d = ellipsefit.transformback(ellipse_coefficients)
    ellipsefit.plot_ellipse(ellipse_3d)

if __name__ == "__main__":
    main()
    
"""
def ellipse(points):
    theta = np.linspace(0, 2*np.pi, 100)
    major = 5
    minor = 3

    x = major * np.cos(theta)
    y = minor * np.sin(theta)

    noise_level = 0.2

    x_noisy = x + np.random.normal(0, noise_level, size = x.shape)
    y_noisy = x + np.random.normal(0, noise_level, size = y.shape)

    plt.scatter(x_noisy, y_noisy)




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

    t_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    x_vals = t_vals
    y_vals = (-G * x_vals - J) / H
    z_vals = np.sqrt(np.abs(-(A*x_vals**2 + B * y_vals**2 + D * x_vals * y_vals + G * x_vals + H * y_vals + J) / C))
    
    
    ax.plot(x_vals, y_vals, z_vals, color='red', linewidth=2, label = "Conic Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

"""









    