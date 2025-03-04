import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

class Ellipse: 
    def __init__(self, points):
        self.points = points
        self.centroid, self.normal = self.fit_plane()
        self.points2d, self.basis = self.plane()

    def fit_plane(self):
        #Fits plane to 3D points using least squares.
        centroid = np.mean(self.points, axis=0)
        _, _, Vt = np.linalg.svd(self.points - centroid)
        normal = Vt[2]
        return centroid, normal

    def plane(self):
        u, v = np.linalg.svd(np.cov(self.points.T))[0][:, :2].T
        basis = np.vstack([u, v])
        return np.dot(self.points - self.centroid, basis.T), basis
    def fit2d(self):
        x, y = self.points2d[:, 0], self.points2d[:, 1]
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
        _, _, Vt = np.linalg.svd(D)
        return Vt[-1]

    def transformback(self, coeffs):
        A, B, C, D, E, F = coeffs
        theta = np.linspace(0, 2 * np.pi, 100)

        a, b = np.sqrt(1 / np.abs([A, C]))
        x, y = a * np.cos(theta), b * np.sin(theta)

        ellipse2d = np.vstack([x, y])
        return ellipse2d.T @ self.basis + self.centroid

    def plot_ellipse(self, ellipse3D):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], color='blue', label="Original Points")
        ax.plot(ellipse3D[:, 0], ellipse3D[:, 1], ellipse3D[:, 2], color='red', label="Fitted Ellipse")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()
