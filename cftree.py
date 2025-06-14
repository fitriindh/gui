
import numpy as np
from sklearn.cluster import KMeans

class CFNode:
    def __init__(self, threshold, branching_factor):
        self.points = []
        self.centroid = None
        self.threshold = threshold
        self.branching_factor = branching_factor

    def add_point(self, point):
        if len(self.points) == 0:
            self.centroid = point
            self.points.append(point)
            return True

        new_centroid = np.mean(np.vstack((self.points, point)), axis=0)
        distance = np.linalg.norm(self.centroid - new_centroid)

        if distance <= self.threshold:
            self.points.append(point)
            self.centroid = new_centroid
            return True
        return False

    def is_full(self):
        return len(self.points) >= self.branching_factor

    def split(self):
        if len(self.points) < 2:
            return [self]

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(self.points)
        labels = kmeans.labels_

        new_nodes = []
        for label in np.unique(labels):
            new_node = CFNode(self.threshold, self.branching_factor)
            new_node.points = [self.points[i] for i in range(len(labels)) if labels[i] == label]
            new_node.centroid = np.mean(new_node.points, axis=0)
            new_nodes.append(new_node)

        return new_nodes

class CFTree:
    def __init__(self, threshold, branching_factor, n_clusters):
        self.nodes = []
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def insert_point(self, point):
        for node in self.nodes:
            if node.add_point(point):
                return

        new_node = CFNode(self.threshold, self.branching_factor)
        new_node.add_point(point)
        self.nodes.append(new_node)

        if len(self.nodes) > self.branching_factor:
            self.nodes = self.split_nodes()

    def split_nodes(self):
        new_nodes = []
        for node in self.nodes:
            if node.is_full():
                new_nodes.extend(node.split())
            else:
                new_nodes.append(node)
        return new_nodes

    def fit_clusters(self):
        cluster_centroids = [node.centroid for node in self.nodes if len(node.points) > 0]
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(cluster_centroids)

        final_labels = kmeans.labels_
        cluster_map = dict(zip(range(len(cluster_centroids)), final_labels))

        return cluster_map

    def predict(self, data):
        cluster_map = self.fit_clusters()
        labels = []
        for point in data:
            distances = [np.linalg.norm(point - node.centroid) for node in self.nodes]
            closest_node_idx = np.argmin(distances)
            labels.append(cluster_map[closest_node_idx])
        return labels
