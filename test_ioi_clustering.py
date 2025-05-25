import numpy as np
from collections import defaultdict

class IOICluster:
    def __init__(self, cluster_width):
        self.cluster_width = cluster_width
        self.clusters = []
    
    def compute_iois(self, events):
        iois = []
        for i, ei in enumerate(events):
            for j, ej in enumerate(events):
                if i != j:
                    iois.append((i, j, abs(ej - ei)))
        return iois
    
    def cluster_iois(self, iois):
        for i, j, ioi in iois:
            best_cluster = None
            best_distance = float('inf')
            for cluster in self.clusters:
                interval = np.mean(cluster)
                distance = abs(interval - ioi)
                if distance < self.cluster_width and distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is not None:
                best_cluster.append(ioi)
            else:
                self.clusters.append([ioi])

    def merge_clusters(self):
        merged = True
        while merged:
            merged = False
            new_clusters = []
            skip_indices = set()
            for i, ci in enumerate(self.clusters):
                if i in skip_indices:
                    continue
                for j in range(i+1, len(self.clusters)):
                    if j in skip_indices:
                        continue
                    interval_i = np.mean(ci)
                    interval_j = np.mean(self.clusters[j])
                    if abs(interval_i - interval_j) < self.cluster_width:
                        ci.extend(self.clusters[j])
                        skip_indices.add(j)
                        merged = True
                new_clusters.append(ci)
            self.clusters = [c for i, c in enumerate(new_clusters) if i not in skip_indices]

    def merge_clusters2(self):
        to_be_deleted = []
        for i, c_i in enumerate(self.clusters):
            c_i_mean = np.mean(c_i)
            for j, c_j in zip(range(i+1, len(self.clusters)), self.clusters[i+1:]):
                c_j_mean = np.mean(c_j)
                if abs(c_i_mean-c_j_mean) < self.cluster_width:
                    c_i+= c_j
                    to_be_deleted.append(j)
        self.clusters = [cluster for i, cluster in enumerate(self.clusters) if i not in to_be_deleted]


    def score_clusters(self):
        scores = [0] * len(self.clusters)
        for i, ci in enumerate(self.clusters):
            interval_i = np.mean(ci)
            for cj in self.clusters:
                for n in range(1, 5):  # try harmonic multiples up to 4
                    interval_j = np.mean(cj)
                    if abs(interval_i - n * interval_j) < self.cluster_width:
                        scores[i] += self.f(n) * len(cj)
        return scores

    def f(self, n):
        # Example scoring function: inverse harmonic penalty
        return 1.0 / n

    def run(self, events):
        iois = self.compute_iois(events)
        self.cluster_iois(iois)
        self.merge_clusters2()
        return self.score_clusters()

# Example usage
events = [0, 1.1, 2.0, 3.2, 5.0, 6.1]  # Example event onset times
cluster_width = 0.3
ioi_clustering = IOICluster(cluster_width)
scores = ioi_clustering.run(events)

for i, cluster in enumerate(ioi_clustering.clusters):
    print(f"Cluster {i}: IOIs = {cluster}, Score = {scores[i]}")
