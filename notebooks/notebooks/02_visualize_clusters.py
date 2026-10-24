import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Rutas del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# 2. Cargar perfiles calculados
df_profiles = pd.read_csv(os.path.join(DATA_DIR, 'cluster_profiles.csv'))

# 3. Transformar datos para graficar con Seaborn (Melt)
# Excluimos tempo y loudness para graficar primero las variables de escala 0 a 1
features_ratio = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
df_melted = df_profiles.melt(id_vars='cluster', value_vars=features_ratio, var_name='Audio Feature', value_name='Average Value')

# 4. Crear gráfico comparativo
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Audio Feature', y='Average Value', hue='cluster', palette='Set2')
plt.title('Arquetipos de Audio: Comparativa de Variables por Cluster', fontsize=14, pad=15)
plt.xlabel('Variable Acústica', fontsize=11)
plt.ylabel('Valor Promedio (0.0 a 1.0)', fontsize=11)
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 5. Guardar gráfico
output_path = os.path.join(REPORTS_DIR, 'cluster_archetypes_comparison.png')
plt.savefig(output_path, dpi=300)
print(f"Gráfica guardada exitosamente en: {output_path}")

# 6. Mostrar métricas de tempo y volumen por cluster
print("\n--- Comparativa de Tempo (BPM) y Loudness (dB) ---")
print(df_profiles[['cluster', 'tempo', 'loudness']])