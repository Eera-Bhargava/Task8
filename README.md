# Task8
# Mall Customers Segmentation using K-Means Clustering

This project applies **unsupervised machine learning** to segment mall customers using their **Annual Income** and **Spending Score**. The goal is to identify distinct customer groups to enable better business targeting.

---

##  Dataset

- **Filename:** `Mall_Customers.csv`
- **Features Used:**
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## Project Objectives

1. Load and preprocess the dataset.
2. Standardize the data.
3. Optionally reduce dimensions using PCA for 2D visualization.
4. Fit the K-Means algorithm and assign cluster labels.
5. Use the Elbow Method to determine the optimal number of clusters.
6. Visualize clusters using PCA.
7. Evaluate clustering using Silhouette Score.

---

## Technologies Used

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Implementation Overview

```python
# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select relevant features
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler().fit_transform(features)

# PCA for visualization
from sklearn.decomposition import PCA
reduced = PCA(n_components=2).fit_transform(scaled)

# KMeans Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(scaled)

# Elbow Method to find optimal k
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

# Silhouette Score
from sklearn.metrics import silhouette_score
score = silhouette_score(scaled, labels)
print("Silhouette Score:", score)
