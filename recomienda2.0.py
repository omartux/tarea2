'''
MCC 2023
RdI
Omar Castillo Alarcon
Edwin Fredy Chambi Mamani
Gludher Quispe Cotacallapa
Erwin Cruz Mamani
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

peliculas = pd.read_csv("data/peliculas.csv")
calificaciones = pd.read_csv("data/calificaciones.csv")

# peliculas.head()
#
# calificaciones.head()

dataset_final = calificaciones.pivot(index='idPelicula', columns='idUsuario', values='calificacion')

print (dataset_final.head()) # para mostrar la carga correcta del df

dataset_final.fillna(0, inplace=True) #llena fillna NaN con ceros

print (dataset_final.head()) # sin NaN, cambiado por ceros

usuarios_sin_votar = calificaciones.groupby('idPelicula')['calificacion'].agg('count')
peliculas_sin_votos = calificaciones.groupby('idUsuario')['calificacion'].agg('count')

dataset_final = dataset_final.loc[usuarios_sin_votar[usuarios_sin_votar > 10].index, :] # minimo 10 votos para ser votante
dataset_final = dataset_final.loc[:, peliculas_sin_votos[peliculas_sin_votos > 50].index] #minimo 50 votos para considerarse recomendable
dataset_final


from scipy.sparse import csr_matrix #libreria para mejorar la eficiencia en matrices con alta sparcidad
#print(muestra_csr)

csr_datos = csr_matrix(dataset_final.values)
dataset_final.reset_index(inplace=True)

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_datos)

def obtener_recomendacion_pelicula(nombre_pelicula):
    n_peliculas_a_recomendar = 10
    lista_peliculas = peliculas[peliculas['titulo'].str.contains(nombre_pelicula)]
    if len(lista_peliculas):
        id_pelicula = lista_peliculas.iloc[0]['idPelicula']
        id_pelicula = dataset_final[dataset_final['idPelicula'] == id_pelicula].index[0]
        distancias, indices = knn.kneighbors(csr_datos[id_pelicula], n_neighbors=n_peliculas_a_recomendar+1)
        indices_recomendados = sorted(list(zip(indices.squeeze().tolist(), distancias.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        marco_recomendaciones = []
        for val in indices_recomendados:
            id_pelicula = dataset_final.iloc[val[0]]['idPelicula']
            idx = peliculas[peliculas['idPelicula'] == id_pelicula].index
            marco_recomendaciones.append({'Título': peliculas.iloc[idx]['titulo'].values[0], 'Distancia':val[1]})
        df = pd.DataFrame(marco_recomendaciones,index=range(1,n_peliculas_a_recomendar+1))
        print (df)
        return df
    else:
        return "No se encontraron peliculas, revise su entrada"

obtener_recomendacion_pelicula('Iron Man')

obtener_recomendacion_pelicula('Toy Story')

obtener_recomendacion_pelicula('Interstellar')

obtener_recomendacion_pelicula('Nixon')