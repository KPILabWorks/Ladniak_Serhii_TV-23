import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Генеруємо дані (імітація профілів енергоспоживання)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Масштабуємо дані
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Функція для візуалізації (без plt.show)
def plot_clusters(X, labels, title, fig_num):
    plt.figure(fig_num, figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
plot_clusters(X_scaled, kmeans_labels, "K-Means Clustering", 1)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
plot_clusters(X_scaled, dbscan_labels, "DBSCAN Clustering", 2)

# Ієрархічна кластеризація
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X_scaled)
plot_clusters(X_scaled, agglo_labels, "Agglomerative Clustering", 3)

# Dendrogram
linked = linkage(X_scaled, 'ward')
plt.figure(4, figsize=(8, 4))
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")

# Показати всі вікна одночасно
plt.show()
