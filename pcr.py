# -*- coding: utf-8 -*-
"""
Created on Mon May 10 01:07:39 2021

@author: hecto
"""

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Datos
# ==============================================================================
datos1 = pd.read_csv('boruta/datasetX.csv')
datos = datos1.drop(columns = datos1.columns[0])

# Correlación entre columnas numéricas
# ==============================================================================

def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = datos.select_dtypes(include=['float64', 'int']) \
              .corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(5)

# División de los datos en train y test
# ==============================================================================
X = datos.drop(columns=['Propágulos','Hojas','Flores','Hojarasca'])
X = X.select_dtypes(include=['float64'])
biologico = datos.drop(columns=['ORP','S','P','pH','T'])
biologico = biologico.select_dtypes(include=['float64'])


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:,0:,])
X.iloc[:,0:,] = imputer.transform(X.iloc[:,0:,])
X = X.astype('float64')

imputer = imputer.fit(biologico.iloc[:,0:,])
biologico.iloc[:,0:,] = imputer.transform(biologico.iloc[:,0:,])
biologico = biologico.astype('float64')

PCR(X,biologico['Propágulos'])
PCR(X,biologico['Hojas'])
PCR(X,biologico['Flores'])
PCR(X,biologico['Hojarasca'])
def PCR(X, y):
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y.values.reshape(-1,1),
                                            train_size   = 0.7,
                                            random_state = 1234,
                                            shuffle      = True
                                        )
    
# Creación y entrenamiento del modelo
# ==============================================================================
modelo = LinearRegression(normalize=True)
modelo.fit(X = X_train, y = y_train)

# Predicciones test
# ==============================================================================
predicciones = modelo.predict(X=X_test)
predicciones = predicciones.flatten()

# Error de test del modelo 
# ==============================================================================
rmse_ols_test = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de test es: {rmse_ols_test}")

from sklearn.metrics import mean_squared_error, r2_score
r2_test = r2_score(y_test,predicciones)

# Predicciones train
# ==============================================================================
predicciones = modelo.predict(X=X_train)
predicciones = predicciones.flatten()

# Error de test del modelo 
# ==============================================================================
rmse_ols_train = mean_squared_error(
            y_true  = y_train,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de train es: {rmse_ols_train}")

from sklearn.metrics import mean_squared_error, r2_score
r2_train = r2_score(y_train,predicciones)


# Entrenamiento modelo de regresión precedido por PCA con escalado
# ==============================================================================
pipe_modelado = make_pipeline(StandardScaler(), PCA(), LinearRegression())
pipe_modelado.fit(X=X_train, y=y_train)

pipe_modelado.set_params


# Predicciones test
# ==============================================================================
predicciones = pipe_modelado.predict(X=X_test)
predicciones = predicciones.flatten()

# Error de test del modelo 
# ==============================================================================
rmse_pcr = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de test es: {rmse_pcr}")

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = {'pca__n_components': [1, 2, 3, 4, 5]}

# Búsqueda por grid search con validación cruzada
# ==============================================================================
grid = GridSearchCV(
        estimator  = pipe_modelado,
        param_grid = param_grid,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = KFold(n_splits=5, random_state=123), 
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)

# Resultados
# ==============================================================================
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)') \
    .drop(columns = 'params') \
    .sort_values('mean_test_score', ascending = False) \
    .head(3)
    
    # Gráfico resultados validación cruzada para cada hiperparámetro
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.84), sharey=True)

resultados.plot('param_pca__n_components', 'mean_train_score', ax=ax)
resultados.plot('param_pca__n_components', 'mean_test_score', ax=ax)
ax.fill_between(resultados.param_pca__n_components.astype(np.float),
                resultados['mean_train_score'] + resultados['std_train_score'],
                resultados['mean_train_score'] - resultados['std_train_score'],
                alpha=0.2)
ax.fill_between(resultados.param_pca__n_components.astype(np.float),
                resultados['mean_test_score'] + resultados['std_test_score'],
                resultados['mean_test_score'] - resultados['std_test_score'],
                alpha=0.2)
ax.legend()
ax.set_title('Evolución del error CV')
ax.set_ylabel('neg_root_mean_squared_error');

# Mejores hiperparámetros por validación cruzada
# ==============================================================================
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

# Entrenamiento modelo de regresión precedido por PCA con escalado
# ==============================================================================
pipe_modelado = make_pipeline(StandardScaler(), PCA(n_components=3), LinearRegression())
pipe_modelado.fit(X=X_train, y=y_train)

# Predicciones test
# ==============================================================================
predicciones = pipe_modelado.predict(X=X_train)
predicciones = predicciones.flatten()

# Error de train del modelo 
# ==============================================================================
rmse_pcr_train = mean_squared_error(
            y_true  = y_train,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de train es: {rmse_pcr_train}")
from sklearn.metrics import mean_squared_error, r2_score
r2_train = r2_score(y_train,predicciones)


# Predicciones test
# ==============================================================================
predicciones = pipe_modelado.predict(X=X_test)
predicciones = predicciones.flatten()

# Error de test del modelo 
# ==============================================================================
rmse_pcr_test = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de test es: {rmse_pcr_test}")
from sklearn.metrics import mean_squared_error, r2_score
r2_test = r2_score(y_test,predicciones)

from sinfo import sinfo
sinfo()