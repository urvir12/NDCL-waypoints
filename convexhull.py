import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

class ConvexHullCalc:
    def __init__(self, points):
        self.points = points
        self.hull = self.computeCH()
    def computeCH(self):
        return ConvexHull(self.points)
    
    def signed_area(self, a, b, c):
        v1 = b - a
        v2 = c - a

        #cross product:
        cross_product = np.cross(v1, v2)

        #compute signed area
        return np.linalg.norm(cross_product)
    
    
    def diameter(self, points):
        convexhull = self.hull
        antipodal = set()
        vertices = convexhull.vertices
        m = len(convexhull.vertices)
        k = 2

        #find initial k
        while self.signed_area(points[vertices[m - 1]], points[vertices[0]], points[vertices[k + 1]]) > self.signed_area(points[vertices[m - 1]], points[vertices[0]], points[vertices[k]]):
            k += 1
    
        #find antipodal pairs
        i = 1
        j = k
        while (i <= k and j <= m):
            antipodal.add((tuple(points[vertices[i]]), tuple(points[vertices[j]])))
            while (j + 1 < m and self.signed_area(points[vertices[i]], points[vertices[i + 1]], points[vertices[j + 1]]) > self.signed_area(points[vertices[i]], points[vertices[i + 1]], points[vertices[j]]) and j < m):
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
    @staticmethod
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