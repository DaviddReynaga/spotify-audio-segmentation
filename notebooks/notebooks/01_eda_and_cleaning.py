import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Definir rutas relativas a la raíz del proyecto de forma automática
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, 'dataset.csv')
print(f"Buscando dataset en: {csv_path}")

# 2. Cargar datos y limpieza básica
df = pd.read_csv(csv_path)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.dropna(subset=['track_name', 'artists'])
if 'track_id' in df.columns:
    df = df.drop_duplicates(subset=['track_id'])

print(f"Registros limpios listos para segmentación: {len(df)}")

# 3. Features acústicas para el modelo
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# 4. Escalado estándar (Media = 0, Varianza = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 5. Método del Codo (Elbow Method)
inertia = []
K_range = range(2, 9)

for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertia.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Elbow Method - Selección óptima de k')
plt.grid(True)
plt.tight_layout()
elbow_img_path = os.path.join(REPORTS_DIR, 'elbow_method.png')
plt.savefig(elbow_img_path)
print(f"Gráfico guardado en: {elbow_img_path}")

# 6. Modelo final con k = 4
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 7. Resumen de perfiles y exportación
cluster_profile = df.groupby('cluster')[features].mean().reset_index()
print("\n--- Perfil Promedio de Cada Cluster ---")
print(cluster_profile)

cluster_profile.to_csv(os.path.join(DATA_DIR, 'cluster_profiles.csv'), index=False)
df.to_csv(os.path.join(DATA_DIR, 'spotify_tracks_clustered.csv'), index=False)
print("\nArchivos exportados exitosamente a la carpeta data/.")