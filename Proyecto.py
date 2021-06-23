# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 19:28:37 2020

@author: hecto
"""
# Red Neuronal Artificial

# Importando Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from boruta import BorutaPy
from sklearn.impute import SimpleImputer
import seaborn as sns
import os
import errno
import pylab as pl
from math import fabs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical 
from sympy import pretty_print as pp, latex

### make X and y
def cargarDatos(ruta):
    global lista_franja, lista_epoca, lista_año, dataset, listaBiologico, lista_region, lista_sitio, listaSitioR, listaParametro
    dataset = pd.read_csv(ruta, sep=",", header=0)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    _dataAux = dataset.select_dtypes(include=['float64'])
    imputer = imputer.fit(_dataAux.iloc[:,0:,])
    _dataAux.iloc[:,0:,] = imputer.transform(_dataAux.iloc[:,0:,])
    _dataAux = _dataAux.astype('float64')    
    dataset = pd.concat([_dataAux, dataset.select_dtypes(include=['int64','object'])], axis=1)
    
    lista_franja = dataset['Franja'].drop_duplicates() 
    lista_epoca = dataset['Epoca del año'].drop_duplicates()
    lista_año = dataset['Año'].drop_duplicates()
    lista_sitio = dataset['Sitio'].drop_duplicates()
    lista_region = dataset['Region'].drop_duplicates()
    
    listaBiologico = list()
    listaBiologico.append('Propágulos')
    listaBiologico.append('Hojas')
    listaBiologico.append('Flores')
    listaBiologico.append('Hojarasca')
    
    listaParametro = list()
    listaParametro.append('S')
    listaParametro.append('ORP')
    listaParametro.append('P')
    
    _lista_epocaAux = list()
    for _epoca in list(lista_epoca):
        _lista_epocaAux.append(_epoca)
    lista_epoca = _lista_epocaAux 
    
    listaSitioR = {}
    for _sitio in list(lista_sitio):
        listaSitioR[_sitio] = ''
        
    dataset = dataset.sort_values(['Año','Region','Sitio'],ascending=True)   
    dataset = dataset.reset_index()
    dataset = dataset.drop(['index'], axis=1)

def transformar(x,año):
    y = list()
    z1 = 0
    año = np.array(año)
    for i in list(x):
        z = año[z1]
        if i == 1:
            y.append('Enero '+str(z))
        elif i == 2:
            y.append('Febrero '+str(z))
        elif i == 3:
            y.append('Marzo '+str(z))
        elif i == 4:
            y.append('Abril '+str(z))
        elif i == 5:
            y.append('Mayo '+str(z))
        elif i == 6:
            y.append('Junio '+str(z))
        elif i == 7:
            y.append('Julio '+str(z))
        elif i == 8:
            y.append('Agosto '+str(z))
        elif i == 9:
            y.append('Septiembre '+str(z))
        elif i == 10:
            y.append('Octubre '+str(z))
        elif i == 11:
            y.append('Noviembre '+str(z))
        elif i == 12:
            y.append('Diciembre '+str(z))
        z1 += 1    
    return y

# Importando Set de Datos
# Red Neuronal Artificial

#X = datos.drop(columns=['Propágulos','Hojas','Flores','Hojarasca'])
#X = X.select_dtypes(include=['float64'])
#biologico = datos.drop(columns=['ORP','S','P','pH','T'])
#biologico = biologico.select_dtypes(include=['float64'])
#
    
    #read_test = pd.read_excel (r'estatal_test.xlsx ')
    #read_test.to_csv (r'estatal_test.csv ', index = None, header = True)
def validate(predicciones,y_pred_train,y_train,y_test,X,y,y_pred_all):
    lista = {}
    
    r2_train = r2_score(y_train,y_pred_train)
    
    rmse_train = mean_squared_error(
                y_true  = y_train,
                y_pred  = y_pred_train,
                squared = False
               )
    
    
    
    rmse_test = mean_squared_error(
                y_true  = y_test,
                y_pred  = predicciones,
                squared = False
               )
    
    rmse_all = mean_squared_error(
                y_true  = y,
                y_pred  = y_pred_all,
                squared = False
               )
    
    corr_test = pearsonr(x = np.array(y_train).flatten(), y = np.array(y_pred_train).flatten())
    lista['pearson_train_r'] = corr_test[0]
    lista['pearson_train_p'] = corr_test[1]
    corr_test = pearsonr(x = np.array(y_test).flatten(), y = np.array(predicciones).flatten())
    lista['pearson_test_r'] = corr_test[0]
    lista['pearson_test_p'] = corr_test[1]
    corr_test = pearsonr(x = np.array(y).flatten(), y = np.array(y_pred_all).flatten())
    lista['pearson_all_r'] = corr_test[0]
    lista['pearson_all_p'] = corr_test[1]
    
    
    r2_test = r2_score(y_test,predicciones)
    lista['rmse_train'] = rmse_train
    lista['r2_train'] = r2_train
    
    r2_all = r2_score(y,y_pred_all)
    lista['rmse_all'] = rmse_all
    lista['r2_all'] = r2_all
    
    lista['rmse_test'] = rmse_test    
    lista['r2_test'] = r2_test
    print(lista)
    return lista

def constructionRNR(X_train, X_test, y_train, y_test,sc,funcion1,perdida, X,y):
    # Parte 2 - Creando Red Neuronal Artificial
    # Inicializando Red Neuronal
    _resultado = {}    
    # Inicializando Modelo
    clasificador = Sequential()
    unit = 50
    # Agregando Primera Capa LSTM y Regularizacion de Desercion
    clasificador.add(LSTM(units=unit, return_sequences=True, 
                      input_dim=X_train.shape[1]))
    clasificador.add(Dropout(0.2))
    
    # Agregando 2da Capa LSTM
    clasificador.add(LSTM(units=unit, return_sequences=True))
    clasificador.add(Dropout(0.2))
    
    # Agregando 3ra Capa LSTM
    clasificador.add(LSTM(units=unit, return_sequences=True))
    clasificador.add(Dropout(0.2))
    
    # Agregando 4ta Capa LSTM
    clasificador.add(LSTM(units=unit))
    clasificador.add(Dropout(0.2))
    
    # Agregando Capa de Salida
    clasificador.add(Dense(units=1, activation=funcion1))
    
    # Optimizador y Funcion de Perdida (Compilacion)
    clasificador.compile(optimizer='adam',loss=perdida,  
                     metrics=['accuracy','mse'])
    
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
    
    clasificador.fit(X_train, y_train, batch_size=5, epochs=130, verbose=1)    
    # Prediciendo Set de Prueba
    y_pred = clasificador.predict(X_test, verbose=0)
    #y_pred = (y_pred>0.5)
   
    y_pred_train = clasificador.predict(X_train, verbose=0)
    #y_pred = (y_pred>0.5)
    #y_pred_train = sc.inverse_transform(y_pred_train)
    
    _resultado['validate'] = validate(y_pred,y_pred_train,y_train,y_test)
    y_pred = sc.inverse_transform(y_pred)
    _resultado['y_pred'] = y_pred
    scores = clasificador.evaluate(X_train, y_train, verbose=0)
    _resultado['scores'] = scores
    return _resultado

def constructionRNA(X_train, X_test, y_train, y_test,sc,funcion1,funcion2,funcion3,perdida,epoca,neuronas,X,y):
    # Parte 2 - Creando Red Neuronal Artificial
    # Inicializando Red Neuronal
    _resultado = {}
    clasificador = Sequential()
    
    # Agregando Capa Input y Primera Capa Oculta
    clasificador.add(Dense(units=50, kernel_initializer='uniform', 
                           activation=funcion1, input_dim=X_train.shape[1]))
    
    # Agregando Segunda Capa Oculta
    clasificador.add(Dense(units=neuronas, kernel_initializer='uniform', 
                           activation=funcion2))
    
    # Agregando Capa de Salida
    clasificador.add(Dense(units=1, kernel_initializer='uniform', 
                           activation=funcion3))
    
    # Compilando Red Neuronal / Descenso Gradiente Estocastica mean_squared_error
    clasificador.compile(optimizer='adam', loss=perdida, 
                         metrics=['accuracy','mse'])
    
    #binary_crossentropy Ajustando Red Neuronal en el Set de Entrenamiento
    clasificador.fit(X_train, y_train, batch_size=5, epochs=epoca, verbose=1,validation_data=(X_test,y_test))
    
    
    # Prediciendo Set de Prueba
    y_pred = clasificador.predict(X_test, verbose=0)
    #y_pred = (y_pred>0.5)
   
    y_pred_train = clasificador.predict(X_train, verbose=0)
    y_pred_all = clasificador.predict(X, verbose=0)
    #y_pred = (y_pred>0.5)
    #y_pred_train = sc.inverse_transform(y_pred_train)
    
    _resultado['validate'] = validate(y_pred,y_pred_train,y_train,y_test,X,y,y_pred_all)
    y_pred = sc.inverse_transform(y_pred)
    y_test = sc.inverse_transform(y_test)
    _resultado['y_pred'] = y_pred
    _resultado['y_test'] = y_test
    scores = clasificador.evaluate(X_train, y_train, verbose=0)
    _resultado['scores'] = scores
    
    return _resultado



#y_train = _dataAux.loc[0:int(dataset.shape[0]*.7),_aux:_aux]
#        y_test = _dataAux.loc[int(dataset.shape[0]*.7)+1:,_aux:_aux]
#        y_train = y_train.values.reshape(-1,1)
#        y_test = y_test.values.reshape(-1,1)
#        # Escalado de Caracteristicas (Normalizacion)
#        sc = MinMaxScaler(feature_range=(_min, _max))
#        X_train = sc.fit_transform(X_train)
#        X_test = sc.fit_transform(X_test)
#        y_train = sc.fit_transform(y_train)
#        y_test = sc.fit_transform(y_test)

def procesamientoRNA(_listaX,_listaY,_dataset,_min,_max,funcion1,funcion2,funcion3,perdida,epoca,neuronas):
    _lista = {}
    _listaRNA = {}
    _listaRNR = {}
    _dataset_train, _dataset_test = train_test_split(_dataset,train_size = 0.8)
    X_train = _dataset_train.drop(columns=np.array(_listaY))
    X_train = X_train.select_dtypes(include=['float64'])
    
    X_test = _dataset_test.drop(columns=np.array(_listaY))
    X_test = X_test.select_dtypes(include=['float64'])
    
    X = _dataset.drop(columns=np.array(_listaY))
    X = X.select_dtypes(include=['float64'])
    
    for _aux in list(_listaY):
        y_test = _dataset_test[_aux]
        y_test = y_test.values.reshape(-1,1)
        
        y_train = _dataset_train[_aux]
        y_train = y_train.values.reshape(-1,1)
        
        y = _dataset[_aux]
        y = y.values.reshape(-1,1)
        
        # Escalado de Caracteristicas (Normalizacion)
        sc = MinMaxScaler(feature_range=(_min, _max))
        X_test = sc.fit_transform(X_test)
        X_train = sc.fit_transform(X_train)
        y_train = sc.fit_transform(y_train)
        y_test = sc.fit_transform(y_test)
        X = sc.fit_transform(X)
        y = sc.fit_transform(y)
        
        _listaRNA[_aux] = constructionRNA(X_train, X_test, y_train, y_test,sc,funcion1,funcion2,funcion3,perdida,epoca,neuronas,X,y)
        #_listaRNR[_aux] = constructionRNR(X_train, X_test, y_train, y_test,sc,funcion3,perdida,X,y)
    _lista['RNA'] = _listaRNA
    _lista['datos'] = _dataset_test
    #_lista['RNR'] = _listaRNR
    return _lista  


#dataR = dataR.drop(['Sitio'], axis=1)
def rnaUniversal(_listaX,_listaY,_dataset,_min,_max,funcion1,funcion2,funcion3,perdida,epoca,neuronas):  
    global X,y
    _lista = {}
    _resultadosFranja = {}
    _resultadosRegion = {} 
    for franja in lista_franja:
       data = _dataset.loc[_dataset.loc[:, 'Franja'] == franja] 
       data = data.drop(['Franja'], axis=1)
       _resultadosFranja[franja] = procesamientoRNA(_listaX,_listaY,data,_min,_max,funcion1,funcion2,funcion3,perdida,epoca,neuronas)
       for region in lista_region:
            dataR = data.loc[data.loc[:, 'Region'] == region]
            _resultadosRegion[region+str(franja)] = procesamientoRNA(_listaX,_listaY,dataR,_min,_max,funcion1,funcion2,funcion3,perdida,epoca,neuronas)
    _lista['resultadosFranja'] = _resultadosFranja
    _lista['resultadosRegion'] = _resultadosRegion
    return _lista        

from sklearn.metrics import mean_squared_error
dataset['Mes'] = transformar(dataset['Mes'])

plt.figure (figsize = (15,5))
plt.rcParams['figure.figsize']=(15,10)

def crearRuta(_ruta):
    try:
        os.makedirs(str(_ruta))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def graficar(y_true,y_pred,mes,biologico,_ruta,_title,tipo):
    fig, ax = plt.subplots()  
    if (tipo == 'biologico'):
        ax.plot(range(len(mes)),y_true,c='g', label = biologico+' conocido')
        ax.plot(range(len(mes)),y_pred,c='r', label = biologico + ' predecido')
    else:    
        ax.plot(range(len(mes)),y_true,c='g', label = lista_abreviaturas[biologico]+' conocido')
        ax.plot(range(len(mes)),y_pred,c='r', label = lista_abreviaturas[biologico] + ' predecido')
    
    plt.rcParams.update({'font.size': 16})
    ax.set_xticks(np.arange(len(mes)))
    ax.set_xticklabels(list(mes),rotation=70, ha='right')
    if (tipo == 'biologico'):
        ax.yaxis.set_ticks(np.arange(0, 10, 0.5))
        ax.set_ylabel('g*$\mathregular{m^{-2}}$*$\mathregular{mes^{-1}}$')
    else:
        if min(dataset[biologico]) < 0:
            ax.yaxis.set_ticks(np.arange(min(dataset[biologico]), max(dataset[biologico])+10, 40))
        elif max(dataset[biologico]) > 100:
            ax.yaxis.set_ticks(np.arange(0, max(dataset[biologico])+10, 25))
        elif max(dataset[biologico]) < 10: 
            ax.yaxis.set_ticks(np.arange(0, 10, 0.5))
        else: 
            ax.yaxis.set_ticks(np.arange(0, max(dataset[biologico])+10, 10))
        ax.set_ylabel(lista_unidades[biologico])
    ax.xaxis.set_ticks(np.arange(1,len(mes),1),tuple(mes))
    ax.set_xlabel('Mes')
    ax.legend()
        
    fig.tight_layout()
    plt.savefig(str(_ruta+_title+'.png'), bbox_inches='tight')
    plt.show()
        
def generarGrafica(listaResultados,funcion,tipo,lista):
    for franja in list(lista_franja):
        mes = transformar(listaResultados['resultadosFranja'][franja]['datos']['Mes'],listaResultados['resultadosFranja'][franja]['datos']['Año'])
        for biologico in list(lista):
           y_pred = listaResultados['resultadosFranja'][franja]['RNA'][biologico]['y_pred']
           y_true = listaResultados['resultadosFranja'][franja]['RNA'][biologico]['y_test']
           #y_true = listaResultados['resultadosFranja'][franja]['datos'][biologico]
           ruta = 'RNA/Gráficas/General/'+tipo+'/'+funcion+'/'+str(franja)+'/'
           _title = biologico+' real vs predecido'
           crearRuta(ruta)
           graficar(y_true,y_pred,mes,biologico,ruta,_title,tipo)
           regresion(biologico, biologico+' predecido', _title+' regresión',ruta, y_true, y_pred)  
           
        for region in list(lista_region):
           mes = transformar(listaResultados['resultadosRegion'][region+str(franja)]['datos']['Mes'],listaResultados['resultadosFranja'][franja]['datos']['Año'])
           for biologico in list(lista):
               y_pred = listaResultados['resultadosRegion'][region+str(franja)]['RNA'][biologico]['y_pred']
               y_true = listaResultados['resultadosRegion'][region+str(franja)]['RNA'][biologico]['y_test']
               #y_true = listaResultados['resultadosRegion'][region+str(franja)]['datos'][biologico]
               ruta = 'RNA/Gráficas/Regional/'+tipo+'/'+funcion+'/'+str(franja)+'/'+region+'/'
               _title = biologico+' real vs predecido'
               crearRuta(ruta)
               graficar(y_true,y_pred,mes,biologico,ruta,_title,tipo)
       
cargarDatos('boruta/datasetXRed.csv')
lista_resultado_relu = rnaUniversal(listaParametro,listaBiologico,dataset,0,1,'sigmoid','tanh','relu','mean_squared_error',100,50)        
lista_resultado_tanh = rnaUniversal(listaParametro,listaBiologico,dataset,0,1,'sigmoid','tanh','tanh','mean_squared_error',100,3)        
lista_resultado_sigmoid = rnaUniversal(listaParametro,listaBiologico,dataset,0,1,'sigmoid','tanh','sigmoid','mean_squared_error',100,3)      
        
generarGrafica(lista_resultado_relu,'relu','biologico',listaBiologico)
generarGrafica(lista_resultado_tanh,'tanh','biologico',listaBiologico)
generarGrafica(lista_resultado_sigmoid,'sigmoid','biologico',listaBiologico)
#PARAMETROS
lista_resultado_relu = rnaUniversal(listaBiologico,listaParametro,dataset,0,1,'sigmoid','tanh','relu','mean_squared_error',100,50)        
lista_resultado_tanh = rnaUniversal(listaBiologico,listaParametro,dataset,0,1,'sigmoid','tanh','tanh','mean_squared_error',100,3)        
lista_resultado_sigmoid = rnaUniversal(listaBiologico,listaParametro,dataset,0,1,'sigmoid','tanh','sigmoid','mean_squared_error',100,3)      
lista_abreviaturas = {}
lista_abreviaturas['ORP'] = 'Potencial redox'
lista_abreviaturas['P'] = 'Precipitación'
lista_abreviaturas['S'] = 'Salinidad'

lista_unidades = {}
lista_unidades['ORP'] = 'mV'
lista_unidades['P'] = 'mm'
lista_unidades['S'] = 'UPS'
        
generarGrafica(lista_resultado_relu,'relu','parametros',listaParametro)
generarGrafica(lista_resultado_tanh,'tanh','parametros',listaParametro)
generarGrafica(lista_resultado_sigmoid,'sigmoid','parametros',listaParametro)








import scipy as sp
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt   
from pandas.plotting import table
import seaborn as sns
import os
import errno
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

def regresion(textX, textY, title, ruta, X_train, y_train):
    ########## IMPLEMENTACIÓN DE REGRESIÓN LINEAL SIMPLE ##########
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #Defino el algoritmo a utilizar
    corr_test = pearsonr(x = np.array(X_train).flatten(), y = np.array(y_train).flatten())
    print("Coeficiente de correlación de Pearson: ", corr_test[0])
    print("P-value: ", corr_test[1])
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1,1)
    
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1,1)
    X_train = X_train.astype('float64')
    y_train = y_train.astype('float64')
    
    lr = linear_model.LinearRegression()
    #Entreno el modelo
    lr.fit(X_train, y_train)
    #Realizo una predicción
    Y_pred = lr.predict(X_train)
    #Graficamos los datos junto con el modelo
    
    r2 = r2_score(X_train,Y_pred)
    mse =  mean_squared_error(y_true = X_train, y_pred = y_train)
    rmse = np.sqrt(mse)
    
    
    
    information = {}
    information['cofA'] = lr.coef_[0][0]
    information['cofB'] = lr.intercept_[0]
    information['ecuation'] = 'Y = ', str("%.3f" % lr.coef_[0][0]),'x +', str("%.3f" % lr.intercept_[0])
    information['MSE'] = mse
    information['RMSE'] = rmse
    information['R2'] = str(r2)
    information['Precision'] = lr.score(X_train, Y_pred)
    information['p-value'] = corr_test[1]
    p = 0
    if corr_test[1] < 0.0001:
        p = 0.00001
    else:
        p = corr_test[1]
    ecuacion = 'Y = '+str("%.3f" % lr.coef_[0][0])+'x + '+str("%.3f" % lr.intercept_[0])
    valores = [['Pendiente',"%.3f" % lr.coef_[0][0]], 
            ["Intersección","%.3f" % lr.intercept_[0]],
            ["Ecuación",ecuacion],
            ["MSE","%.3f" %mse],
            ["RMSE","%.3f" %rmse],
            ["R2","%.3f" % r2], 
            ["Precisión",str("%.3f" % (lr.score(X_train, y_train)*100))+"%"],
            ["p-value", str(p)],
            ["n",len(X_train)]
            ]
    
    plt.figure()
    plt.subplot(1,3,(1,2))
    plt.plot(X_train, Y_pred, color='red', linewidth=3)
    plt.scatter(X_train, y_train)
    plt.title('Regresión Lineal Simple '+str(title))
    plt.xlabel(str(textX))
    plt.ylabel(str(textY))
    
    
    plt.subplot(4, 4, 4)
    plt.table(cellText=valores, colWidths = [0.6]*len(X_train), loc='center')
    plt.grid(b = None)
    plt.axis("off")
    plt.savefig(ruta+title, bbox_inches='tight',dpi=300)
    
    return information
    