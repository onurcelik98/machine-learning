import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches

# # Parsing the csv into a dataframe
# df = pd.read_csv('merc.csv')
#
# # Removing duplicates rows and rows with meaningless values
# df["engineSize"] = df["engineSize"].replace(0, np.nan)
# df["tax"] = df["tax"].replace(0, np.nan)
# df["fuelType"] = df["fuelType"].replace("Other", np.nan)
# df["transmission"] = df["transmission"].replace("Other", np.nan)
# df.dropna(inplace=True)
# df.drop_duplicates(inplace=True)
#
# # Modifying column dtypes to appropriate dtypes
# df["model"] = df["model"].astype("category")
# df["year"] = df["year"].astype("category")
# df["price"] = df["price"].astype("int64")
# df["transmission"] = df["transmission"].astype("category")
# df["mileage"] = df["mileage"].astype("int64")
# df["fuelType"] = df["fuelType"].astype("category")
# df["tax"] = df["tax"].astype("int64")
# df["mpg"] = df["mpg"].astype("float64")
# df["engineSize"] = df["engineSize"].astype("float64")
#
# # Removing outliers
# df = df[((df["fuelType"] != "Hybrid") | (df["mpg"] > 10))]
#
# # Removing uninteresting columns
# df2 = df.drop(columns=["price", "mileage", "tax"])
# number_of_inputs = 5
#
# # Splitting data
# df_dummies = pd.get_dummies(df2["transmission"])
# df_dummies["mpg"] = df2["mpg"]
# df_dummies["engineSize"] = df2["engineSize"]
# X = np.array(df_dummies[["mpg", "engineSize"]])
# y = np.array(df2["fuelType"])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# length = len(y)
# y_enum = [None] * length
# for i in range(length):
#     if y[i] == "Diesel":
#         y_enum[i] = 0
#     elif y[i] == "Petrol":
#         y_enum[i] = 1
#     else:
#         y_enum[i] = 2
# y_enum = np.array(y_enum)
#
# plt.figure(figsize=(10.24, 7.68))
# plt.scatter(x=X[:, -2], y=X[:, -1], c=y_enum, s=40)
# plt.xlabel("Miles per gallon")
# plt.ylabel("Engine size")
# plt.title("Scatter plot")
# plt.show()

num_clusters = 3
X, y = make_blobs(n_samples=1000, n_features=2, centers=num_clusters, cluster_std=2, random_state=41)

plt.figure(figsize=(10.24, 7.68))
plt.scatter(x=X[:, -2], y=X[:, -1], c=y, s=40)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Scatter plot of actual labels")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
p_patch = mpatches.Patch(color='tab:purple', label='Cluster-3')
plt.legend(handles=[y_patch, c_patch, p_patch])
# plt.savefig("plots/clustering/actual_labels")
plt.show()

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 1000
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)
        return clusters

    def calculate_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    def predict(self, clusters):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def fit(self, X):
        # Randomly assigning clusters to each data point
        centroids = np.zeros((self.K, self.num_features))
        for k in range(self.K):
            centroids[k] = X[np.random.choice(range(self.num_examples))]

        # Iteratively calculating centroids & clusters
        clusters = []
        for i in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_centroids(clusters, X)
            diff = centroids - previous_centroids
            if not diff.any():
                break

        # Making predictions
        y_pred = self.predict(clusters)
        return y_pred


#%% PCA

X_scaled = preprocessing.scale(X)
model_pca = PCA()
model_pca.fit(X_scaled)
pca_data = model_pca.transform(X_scaled)

# Plotting
per_var = np.round(model_pca.explained_variance_ratio_ * 100, decimals=1)
labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Component")
plt.title("Scree Plot")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
p_patch = mpatches.Patch(color='tab:purple', label='Cluster-3')
plt.legend(handles=[y_patch, c_patch, p_patch])
# plt.savefig("plots/clustering/scree_plot")
plt.show()

pca_df = pd.DataFrame(pca_data, columns=labels)
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=y, s=40)
plt.title("PCA Graph")
plt.xlabel(f"PC1 - {per_var[0]}")
plt.ylabel(f"PC2 - {per_var[1]}")
# plt.savefig("plots/clustering/pca_graph")
plt.show()

# Measuring loading scores
loading_scores = pd.Series(model_pca.components_[0])
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
print(loading_scores)



#%%

np.random.seed(10)
num_clusters = 3

model_kmsk = KMeansClustering(X, num_clusters)
y_pred_mixed = model_kmsk.fit(X)
y_pred = y_pred_mixed
for i in range(len(y_pred_mixed)):
    if y_pred_mixed[i] == 0:
        y_pred[i] = 2
    elif y_pred_mixed[i] == 1:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# length = len(y_pred_enum)
# y_pred = [None] * length
# for i in range(length):
#     if y_pred_enum[i] == 0:
#         y_pred[i] = "Diesel"
#     elif y_pred_enum[i] == 2:
#         y_pred[i] = "Petrol"
#     else:
#         y_pred[i] = "Hybrid"
# y_pred = np.array(y_pred)


# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y, y_pred)}\n")
print(f"{classification_report(y, y_pred)}\n")

plt.figure(figsize=(10.24, 7.68))
plt.scatter(X[:, -2], X[:, -1], c=y_pred, s=40)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("My k-Means clusters")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
p_patch = mpatches.Patch(color='tab:purple', label='Cluster-3')
plt.legend(handles=[y_patch, c_patch, p_patch])
# plt.savefig("plots/clustering/my_kmeans_cluster_plot")
plt.show()

cm = confusion_matrix(y, y_pred)

#%% K-means clustering

model_km = KMeans(n_clusters=3, random_state=5)
model_km.fit(X)

y_pred_mixed = model_km.predict(X)
y_pred = y_pred_mixed
for i in range(len(y_pred_mixed)):
    if y_pred_mixed[i] == 0:
        y_pred[i] = 2
    elif y_pred_mixed[i] == 1:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# length = len(y_pred_enum)
# y_pred = [None] * length
# for i in range(length):
#     if y_pred_enum[i] == 0:
#         y_pred[i] = "Diesel"
#     elif y_pred_enum[i] == 1:
#         y_pred[i] = "Petrol"
#     else:
#         y_pred[i] = "Hybrid"
# y_pred = np.array(y_pred)

# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y, y_pred)}\n")
print(f"{classification_report(y, y_pred)}\n")

plt.figure(figsize=(10.24, 7.68))
plt.scatter(X[:, -2], X[:, -1], c=y_pred, s=40)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("k-Means Clusters")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
p_patch = mpatches.Patch(color='tab:purple', label='Cluster-3')
plt.legend(handles=[y_patch, c_patch, p_patch])
# plt.savefig("plots/clustering/kmeans_cluster_plot")
plt.show()



#%% Hierarchical clustering

# Plotting dendrogram
linked = linkage(X, 'single')

labelList = np.ndarray(range(1, 11))

plt.figure(figsize=(19.20, 10.80))
dendrogram(linked,
            orientation='top',
            # labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")
# plt.savefig("plots/clustering/dendrogram")
plt.show()

# Training the model
model_hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model_hc.fit_predict(X)
y_pred_mixed = model_hc.labels_
y_pred = y_pred_mixed
for i in range(len(y_pred_mixed)):
    if y_pred_mixed[i] == 0:
        y_pred[i] = 0
    elif y_pred_mixed[i] == 1:
        y_pred[i] = 2
    else:
        y_pred[i] = 1

# length = len(y_pred_enum)
# y_pred = [None] * length
# for i in range(length):
#     if y_pred_enum[i] == 0:
#         y_pred[i] = "Diesel"
#     elif y_pred_enum[i] == 1:
#         y_pred[i] = "Hybrid"
#     else:
#         y_pred[i] = "Petrol"
# y_pred = np.array(y_pred)

# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y, y_pred)}\n")
print(f"{classification_report(y, y_pred)}\n")


plt.figure(figsize=(10.24, 7.68))
plt.scatter(X[:, -2], X[:, -1], c=y_pred, s=40)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Hierarchical clusters")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
p_patch = mpatches.Patch(color='tab:purple', label='Cluster-3')
plt.legend(handles=[y_patch, c_patch, p_patch])
# plt.savefig("plots/clustering/hierarchical_cluster_plot")
plt.show()


#%% DBSCAN (?)

model_db = DBSCAN(eps=1.8, min_samples=50)
model_db.fit(X)
y_pred_mixed = model_db.labels_
y_pred = y_pred_mixed
for i in range(len(y_pred_mixed)):
    if y_pred_mixed[i] == -1:
        y_pred[i] = -1
    elif y_pred_mixed[i] == 0:
        y_pred[i] = 1
    elif y_pred_mixed[i] == 1:
        y_pred[i] = 0
    else:
        y_pred[i] = 2

# length = len(y_pred_enum)
# y_pred = [None] * length
# for i in range(length):
#     if y_pred_enum[i] == 0:
#         y_pred[i] = "Diesel"
#     elif y_pred_enum[i] == 1:
#         y_pred[i] = "Petrol"
#     elif y_pred_enum[i] == 2:
#         y_pred[i] = "Hybrid"
#     else:
#         y_pred[i] = "-1"
# y_pred = np.array(y_pred)

# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y, y_pred)}\n")
print(f"{classification_report(y, y_pred)}\n")
print(len(set(y_pred)))
print(np.unique(y_pred_mixed))


plt.figure(figsize=(10.24, 7.68))
plt.scatter(X[:, -2], X[:, -1], c=y_pred, s=40)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("DBSCAN clusters")
y_patch = mpatches.Patch(color='y', label='Cluster-1')
c_patch = mpatches.Patch(color='c', label='Cluster-2')
g_patch= mpatches.Patch(color='g', label='Cluster-3')
p_patch = mpatches.Patch(color='tab:purple', label='Noise')
plt.legend(handles=[y_patch, c_patch, g_patch, p_patch])
# plt.savefig("plots/clustering/dbscan_cluster_plot")
plt.show()
