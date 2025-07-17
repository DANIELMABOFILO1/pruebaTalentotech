from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('penguins.csv')

df.dropna(inplace=True)

X = df[['body_mass_g', 'flipper_length_mm']]

kmeans = KMeans(n_clusters=3, random_state=42)

df['cluster'] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

labels = kmeans.labels_

print(df.head(50))

plt.scatter(X['body_mass_g'], X['flipper_length_mm'], c=kmeans.labels_, cmap='viridis')

plt.scatter(centroids[:,0], centroids[:,1], s=200, marker='^', c='red')

plt.xlabel('Body Mass (g)')

plt.ylabel('Flipper Length (mm)')

plt.title('Clustering de pingüinos')

plt.show()
