from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Datos aleatorios dos dimensiones
x2,y2 = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means con 4 clusters
KMEANS = KMeans(n_clusters=4)

# Ajustamos el modelo a los datos
KMEANS.fit(x2)

# Obtener los centros de los clústeres y las etiquetas
# de clúster para cada punto

centroids = KMEANS.cluster_centers_
labels = KMEANS.labels_

# graficar los datos y los centros de clústeres
plt.scatter(x2[:,0], x2[:,1], c=labels, cmap="viridis")
plt.scatter(centroids[:,0], centroids[:,1], s=200, marker='^', c='red')
plt.title('Resultados de K-Means')
plt.show()



