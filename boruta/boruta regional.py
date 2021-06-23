# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:40:13 2021

@author: hecto
"""

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
import seaborn as sns
import os
import errno
import pylab as pl
from math import fabs

### make X and y
def cargarDatos(ruta):
    global lista_franja, lista_epoca, lista_año, dataset, listaBiologico, lista_region, lista_sitio, listaSitioR
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
    
    _lista_epocaAux = list()
    for _epoca in list(lista_epoca):
        _lista_epocaAux.append(_epoca)
    lista_epoca = _lista_epocaAux 
    
    listaSitioR = {}
    for _sitio in list(lista_sitio):
        listaSitioR[_sitio] = ''

def forestResults(_X_boruta,_y, _X,data,var):
    _data = pd.DataFrame()
    _data = data.copy()
    _data = _data.drop([var], axis=1)
    _resultado = pd.DataFrame()
    ### fit a random forest (suggested max_depth between 3 and 7)
    forest = RandomForestRegressor(max_depth = 5, random_state = 42)
    forest.fit(_X_boruta,_y)
    ### store feature importances
    importance = forest.feature_importances_[:len(_X.columns)]
    feat_imp_shadow = forest.feature_importances_[len(_X.columns):]
    ### compute hits
    hits = importance > feat_imp_shadow.max()
    stats = hits
    
    ### initialize hits counter
    hits = np.zeros((len(_X.columns)))
    ### repeat 20 times
    for iter_ in range(37):
       ### make X_shadow by randomly permuting each column of X
       np.random.seed(iter_)
       _X_shadow = _X.apply(np.random.permutation)
       _X_boruta = pd.concat([_X, _X_shadow], axis = 1)
       ### fit a random forest (suggested max_depth between 3 and 7)
       forest = RandomForestRegressor(max_depth = 5, random_state = 42)
       forest.fit(_X_boruta, _y)
       ### store feature importance
       feat_imp_X = forest.feature_importances_[:len(_X.columns)]
       feat_imp_shadow = forest.feature_importances_[len(_X.columns):]
       ### compute hits for this trial and add to counter
       hits += (feat_imp_X > feat_imp_shadow.max())
        
    y2 = pd.DataFrame(_y)
    test = SelectKBest(score_func=f_regression, k = 'all')
    fited = test.fit(_X, y2.values.ravel())
    f = fited.scores_
    f = pd.DataFrame(f)
    f = f.replace(np.nan,0)
    f = f.copy().values
    
    p = fited.pvalues_
    p = pd.DataFrame(p)
    p = p.replace(np.nan,0)
    p = p.copy().values
    
    _resultado['Hits'] = hits
    _resultado['Var'] = _data.columns.tolist()
    _resultado['F'] = f
    _resultado['Importance'] = importance
    _resultado['p'] = p
    _resultado['Status'] = stats
    _resultado['f_regression'] = fited.scores_
    return _resultado   

def getRanking(_X,_y,data,var):
    area = {}
    _data = pd.DataFrame()
    _data = data.copy()
    _data = _data.drop([var], axis=1)
   
    ###initialize Boruta
    forest = RandomForestRegressor(
       n_jobs = -1, 
       max_depth = 5
    )
    boruta = BorutaPy(
       estimator = forest, 
       n_estimators = 'auto',
       max_iter = 100 # number of trials to perform
    )
    ### fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(np.array(_X), np.array(_y))
    ### print results
    green_area = _data.columns[boruta.support_].to_list()
    blue_area = _data.columns[boruta.support_weak_].to_list()

    area['Rank'] = boruta.ranking_
    area['Green area'] = green_area
    area['Blue area'] = blue_area
    
    return area

def getData(_data1, var):
    data = pd.DataFrame()
    data = _data1.copy()

    datosProcesados = {}
    _y = data.pop(var)
    _y = np.array(_y)
    _y = _y.reshape(-1,1)
    
    _X = data.copy().values
    _X = _X.astype('float64')
    _y = _y.astype('float64')
    _X = pd.DataFrame(_X)
    
    ### make X_shadow by randomly permuting each column of X
    np.random.seed(42)
    _X_shadow = _X.apply(np.random.permutation)
    _X_shadow.columns = ['shadow_' + feat for feat in data.columns]
    ### make X_boruta by appending X_shadow to X
    _X_boruta = pd.concat([_X, _X_shadow], axis = 1)
    
    datosProcesados['X'] = _X
    datosProcesados['y'] = _y
    datosProcesados['X_boruta'] = _X_boruta 
    
    return datosProcesados

def imp_df(column_names, importances):
    df = pd.DataFrame({'Parámetro fisicoquímico': column_names,
                       'Importancia (Z)': importances}) \
           .sort_values('Importancia (Z)', ascending = False) \
           .reset_index(drop = True)
    return df

def getResultado(var,title,_dataset,listaSitioR):
    _listaFranja = {}
    _listaBestResult = {}
    _listaBestVar = {}
    _listaFResult = {}
    _listaPResult = {}
    _listaFVar = {}
    
    for franja in lista_franja:
       _resultadoFranja = {}
       _bestResult = {}
       _bestVar = {}
       
       _regionBestResult = list()
       _regionBestVar = list()
       _sitioResult = {}
       _sitioVar = {}
       _particularBestResult = {}
       _particularBestVar = {}
       
       #F y p value
       _bestResultP = {}
       _bestResultF = {}
       _bestVarF = {}
       
       _regionBestResultF = list()
       _regionBestVarF = list()
       _sitioResultF = {}
       _sitioVarF = {}
       _particularBestResultF = {}
       _particularBestVarF = {}
       
       _regionBestResultP = list()
       _sitioResultP = {}
       _particularBestResultP = {}       
       
       data = _dataset.loc[_dataset.loc[:, 'Franja'] == franja] 
       data = data.drop(['Franja'], axis=1)
       _listaSitioR = listaSitioR.copy()
       for region in lista_region:
           _resultadosRegion = {}   
           _listaSitios = {}
           _listaEpocas = {}
           _listaAño = {}
           dataR = data.loc[data.loc[:, 'Region'] == region]
           dataR= dataR.drop(['Region'], axis=1)
           dataR = dataR.drop(['Mes'], axis=1)
           dataR = dataR.drop(['Año'], axis=1)
           dataR = dataR.drop(['Epoca del año'], axis=1)
           dataR = dataR.drop(['Sitio'], axis=1)
           _datosProcesados = getData(dataR, var)

           if(dataR.empty != True):           
               X_boruta = _datosProcesados['X_boruta']
               y = _datosProcesados['y']
               X = _datosProcesados['X']
                
               _resultado = forestResults(X_boruta,y,X,dataR,var)
               _area = getRanking(X,y,dataR,var)
               _resultado['Rank'] = _area['Rank'] 
               _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
               _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
               
               _resultadosRegion['area'+str(region)] = _area
               _resultadosRegion['region'+str(region)] = _resultado
                                            
               _resultadoFranja[str(region)+str(franja)] = _resultadosRegion
               
               dataR = data.loc[data.loc[:, 'Region'] == region]
               dataR= dataR.drop(['Region'], axis=1)
               
               sps = str(title)+'/Franja'+str(franja)+'/region'+str(region)
               
               _aux = _resultado.loc[_resultado["Importance"].idxmax()]
               _regionBestResult.append(_aux['Importance'])
               _regionBestVar.append(_aux['Var'])
               
               _aux = _resultado.loc[_resultado["F"].idxmax()]
               _regionBestResultF.append(_aux['F'])
               _regionBestVarF.append(_aux['Var'])
               
               _aux = _resultado.loc[_resultado["F"].idxmax()]
               _regionBestResultP.append(_aux['p'])               
               
               try:
                    os.makedirs(str(sps))
               except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
               fig = plt.figure()
               ax = fig.add_subplot(111)
               sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['Importance']), orient = 'h', color = 'royalblue') \
                   .set_title('Análisis de la región '+str(region)+' F'+str(franja), fontsize = 20)
               ax.xaxis.set_ticks(np.arange(0, 1, 0.1)) 
               plt.savefig(str(sps)+'/'+str(region)+' F'+str(franja), bbox_inches='tight')
               
           #importancia
           _regionAñoResult = list()
           _regionAñoVar = list()
           # F y p value
           _regionAñoResultP = list()
           _regionAñoResultF = list()
           _regionAñoVarF = list()
           for año in list(lista_año):
               _resultadosAño = {}
               _dataAño = dataR.loc[dataR.loc[:, 'Año'] == año]
               _dataAño = _dataAño.drop(['Sitio'], axis=1)
               _dataAño = _dataAño.drop(['Mes'], axis=1)
               _dataAño = _dataAño.drop(['Año'], axis=1)
               _dataAño = _dataAño.drop(['Epoca del año'], axis=1)
               if(_dataAño.empty != True):
                   try:
                       _datosProcesados = getData(_dataAño, var)
                       X_boruta = _datosProcesados['X_boruta']
                       y = _datosProcesados['y']
                       X = _datosProcesados['X']
                       
                       _resultado = forestResults(X_boruta,y,X,_dataAño,var)
                       _area = getRanking(X,y,_dataAño,var)
                       _resultado['Rank'] = _area['Rank']
                       _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
                       _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
                       
                       _resultadosAño['area'+str(año)] = _area
                       _resultadosAño['año'+str(año)] = _resultado
                       
                       _aux = _resultado.loc[_resultado["Importance"].idxmax()]
                       _regionAñoResult.append(_aux['Importance'])
                       _regionAñoVar.append(_aux['Var'])

                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _regionAñoResultF.append(_aux['F'])
                       _regionAñoVarF.append(_aux['Var']) 
                       
                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _regionAñoResultP.append(_aux['p'])
                   except OSError as e:
                       if e.errno != errno.EEXIST:
                           raise                   
                 
                   _listaAño['año'+str(año)] = _resultadosAño 
                   
                   sps = str(title)+'/Franja'+str(franja)+'/region'+str(region)+'/año'
                   try:
                        os.makedirs(str(sps))
                   except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                   fig = plt.figure()
                   ax = fig.add_subplot(111)
                   sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['Importance']), orient = 'h', color = 'royalblue') \
                       .set_title('Análisis de la región '+str(region)+' F'+str(franja)+' '+str(año), fontsize = 20)
                   ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
                   plt.savefig(str(sps)+'/ Región '+str(region)+' F'+str(franja)+' '+str(año), bbox_inches='tight')
           
               else:
                    _regionAñoResult.append(float(0))
                    _regionAñoVar.append('')
                    _regionAñoResultF.append(float(0))
                    _regionAñoResultP.append(float(0))
                    _regionAñoVarF.append('')
#           #
           _particularBestResult['año'+str(region)] = _regionAñoResult
           _particularBestVar['año'+str(region)] = _regionAñoVar
           _particularBestResultF['año'+str(region)] = _regionAñoResultF
           _particularBestVarF['año'+str(region)] = _regionAñoVarF
           _particularBestResultP['año'+str(region)] = _regionAñoResultP
           for sitio in list(lista_sitio):
               _resultadosSitio = {}
               _dataSitio = dataR.loc[dataR.loc[:, 'Sitio'] == sitio]
               _dataSitio = _dataSitio.drop(['Sitio'], axis=1)
               _dataSitio = _dataSitio.drop(['Mes'], axis=1)
               _dataSitio = _dataSitio.drop(['Año'], axis=1)
               _dataSitio = _dataSitio.drop(['Epoca del año'], axis=1)
               if(_dataSitio.empty != True):
                   try:
                       _datosProcesados = getData(_dataSitio, var)
                       X_boruta = _datosProcesados['X_boruta']
                       y = _datosProcesados['y']
                       X = _datosProcesados['X']
                        
                       _resultado = forestResults(X_boruta,y,X,_dataSitio,var)
                       _area = getRanking(X,y,_dataSitio,var)
                       _resultado['Rank'] = _area['Rank'] 
                       _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
                       _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
                       
                       _resultadosSitio['area'+str(sitio)] = _area
                       _resultadosSitio['sitio'+str(sitio)] = _resultado
                       
                       _aux = _resultado.loc[_resultado["Importance"].idxmax()]
                       _sitioResult[sitio] = _aux['Importance']
                       _sitioVar[sitio] = _aux['Var']
                       
                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _sitioResultF[sitio] = _aux['F']
                       _sitioVarF[sitio] = _aux['Var']
                       
                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _sitioResultP[sitio] = _aux['p']
                       
                       _listaSitioR[sitio] = _aux['Var']
                   except OSError as e:
                       if e.errno != errno.EEXIST:
                           raise
                                    
                   _listaSitios['sitio'+str(sitio)] = _resultadosSitio
                   
                   sps = str(title)+'/Franja'+str(franja)+'/region'+str(region)+'/sitio'
                   try:
                        os.makedirs(str(sps))
                   except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                   fig = plt.figure()
                   ax = fig.add_subplot(111)
                   sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['Importance']), orient = 'h', color = 'royalblue') \
                       .set_title('Análisis del sitio '+str(sitio)+' F'+str(franja), fontsize = 20)
                   ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
                   plt.savefig(str(sps)+'/Sitio '+str(sitio)+' F'+str(franja), bbox_inches='tight')
                   
               else:
                   if _listaSitioR[sitio] == '':
                       _sitioResult[sitio] = float(0)
                       _sitioVar[sitio] = ''
                       _sitioResultF[sitio] = float(0)
                       _sitioVarF[sitio] = ''
                       _sitioResultP[sitio] = float(0)
           
            #importancia          
           _listaEpocaAñoResult = {}
           _listaEpocaAñoVar = {}
           _generalEpocaResult = list()
           _generalEpocaVar = list()
           _especificoEpocaVar = {}
           _especificoEpocaResult = {}
           
           #F y p Value
           _listaEpocaAñoResultF = {}
           _listaEpocaAñoVarF = {}
           _generalEpocaResultF = list()
           _generalEpocaVarF = list()
           _especificoEpocaVarF = {}
           _especificoEpocaResultF = {}
           
           _listaEpocaAñoResultP = {}
           _generalEpocaResultP = list()
           _especificoEpocaResultP = {}
           
           for epoca in list(lista_epoca):
               _resultadosEpoca = {}
               _dataEpoca = dataR.loc[dataR.loc[:, 'Epoca del año'] == epoca]
               _dataEpoca = _dataEpoca.drop(['Epoca del año'], axis=1)
               _dataEpoca = _dataEpoca.drop(['Mes'], axis=1)
               _dataEpoca = _dataEpoca.drop(['Año'], axis=1)
               _dataEpoca = _dataEpoca.drop(['Sitio'], axis=1)
               if(_dataEpoca.empty != True):
                   try: 
                       _datosProcesados = getData(_dataEpoca, var)
                       X_boruta = _datosProcesados['X_boruta']
                       y = _datosProcesados['y']
                       X = _datosProcesados['X']
                        
                       _resultado = forestResults(X_boruta,y,X,_dataEpoca,var)
                       _area = getRanking(X,y,_dataEpoca,var)
                       _resultado['Rank'] = _area['Rank'] 
                       _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
                       _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
                       
                       _resultadosEpoca['area'+str(epoca)] = _area
                       _resultadosEpoca['epoca'+str(epoca)] = _resultado
                       
                       _aux = _resultado.loc[_resultado["Importance"].idxmax()]
                       _generalEpocaResult.append(_aux['Importance'])
                       _generalEpocaVar.append(_aux['Var'])
                       
                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _generalEpocaResultF.append(_aux['F'])
                       _generalEpocaVarF.append(_aux['Var'])
                       
                       _aux = _resultado.loc[_resultado["F"].idxmax()]
                       _generalEpocaResultP.append(_aux['p'])
                       
                   except OSError as e:
                       if e.errno != errno.EEXIST:
                           raise
                   
                   _listaEpocas[str(epoca)+'General'+str(region)] = _resultadosEpoca            
                   
                   sps = str(title)+'/Franja'+str(franja)+'/region'+str(region)+'/epoca'
                   try:
                        os.makedirs(str(sps))
                   except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                   fig = plt.figure()
                   ax = fig.add_subplot(111)
                   sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['Importance']), orient = 'h', color = 'royalblue') \
                       .set_title('Análisis de la región '+str(region)+' F'+str(franja)+' '+str(epoca), fontsize = 20)
                   ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
                   plt.savefig(str(sps)+'/ Región '+str(region)+' F'+str(franja)+' '+str(epoca), bbox_inches='tight')
               else:
                   _generalEpocaResult.append(float(0))
                   _generalEpocaVarF.append('')
                   _generalEpocaResultF.append(float(0))
                   _generalEpocaVarF.append('')
                   _generalEpocaResultP.append(float(0))
#                
                #importancia   
               _listaEpocaAño = {} 
               _regionalEpocaResult = list()
               _regionalEpocaVar = list()
               
               # F y p value
               _regionalEpocaResultF = list()
               _regionalEpocaVarF = list()
               _regionalEpocaResultP = list()
               for año in list(lista_año):
                   _resultadosEpocaAño = {}
                   _dataAño = dataR.loc[dataR.loc[:, 'Epoca del año'] == epoca]
                   _dataAño = _dataAño.loc[_dataAño.loc[:, 'Año'] == año]
                   _dataAño = _dataAño.drop(['Epoca del año'], axis=1)
                   _dataAño = _dataAño.drop(['Sitio'], axis=1)
                   _dataAño = _dataAño.drop(['Mes'], axis=1)
                   _dataAño = _dataAño.drop(['Año'], axis=1)
                   if(_dataAño.empty != True):
                       try:
                           _datosProcesados = getData(_dataAño, var)
                           X_boruta = _datosProcesados['X_boruta']
                           y = _datosProcesados['y']
                           X = _datosProcesados['X']
                           
                           _resultado = forestResults(X_boruta,y,X,_dataAño,var)
                           _area = getRanking(X,y,_dataAño,var)
                           _resultado['Rank'] = _area['Rank']
                           _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
                           _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
                           
                           _resultadosEpocaAño['area'+str(año)] = _area
                           _resultadosEpocaAño['año'+str(año)] = _resultado
                           
                           _aux = _resultado.loc[_resultado["Importance"].idxmax()]
                           _regionalEpocaResult.append(_aux['Importance'])
                           _regionalEpocaVar.append(_aux['Var'])   
                           
                           _aux = _resultado.loc[_resultado["F"].idxmax()]
                           _regionalEpocaResultF.append(_aux['F'])
                           _regionalEpocaVarF.append(_aux['Var'])   
                           
                           _aux = _resultado.loc[_resultado["F"].idxmax()]
                           _regionalEpocaResultP.append(_aux['p'])
                       except OSError as e:
                           if e.errno != errno.EEXIST:
                               raise
                                            
                       _listaEpocaAño['año'+str(año)] = _resultadosEpocaAño 
                       
                       sps = str(title)+'/Franja'+str(franja)+'/region'+str(region)+'/epoca/año'
                       try:
                            os.makedirs(str(sps))
                       except OSError as e:
                            if e.errno != errno.EEXIST:
                                raise
                       fig = plt.figure()
                       ax = fig.add_subplot(111)
                       sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['Importance']), orient = 'h', color = 'royalblue') \
                           .set_title('Análisis de la región '+str(region)+' F'+str(franja)+' '+str(epoca)+' '+str(año), fontsize = 20)
                       ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
                       plt.savefig(str(sps)+'/ Región '+str(region)+' F'+str(franja)+' '+str(epoca)+' '+str(año), bbox_inches='tight')
                   
                   else:
                       _regionalEpocaResult.append(float(0))
                       _regionalEpocaVar.append('')
                       _regionalEpocaResultF.append(float(0))
                       _regionalEpocaVarF.append('')
                       _regionalEpocaResultP.append(float(0))
                       
               _listaEpocas[str(epoca)+'Años'+str(region)] = _listaEpocaAño
               _listaEpocaAñoResult[epoca] = _regionalEpocaResult
               _listaEpocaAñoVar[epoca] =  _regionalEpocaVar
               
               _listaEpocaAñoResultF[epoca] = _regionalEpocaResultF
               _listaEpocaAñoVarF[epoca] =  _regionalEpocaVarF
               _listaEpocaAñoResultP[epoca] = _regionalEpocaResultP
               
           _especificoEpocaResult['general'] = _generalEpocaResult
           _especificoEpocaVar['general'] = _generalEpocaVar
           _especificoEpocaVar['año'] = _listaEpocaAñoVar
           _especificoEpocaResult['año'] = _listaEpocaAñoResult
           
           _especificoEpocaResultF['general'] = _generalEpocaResultF
           _especificoEpocaVarF['general'] = _generalEpocaVarF
           _especificoEpocaVarF['año'] = _listaEpocaAñoVarF
           _especificoEpocaResultF['año'] = _listaEpocaAñoResultF
           
           _especificoEpocaResultP['general'] = _generalEpocaResultP
           _especificoEpocaResultP['año'] = _listaEpocaAñoResultP
        #  nivel región 
           _resultadoFranja['listaAño'+str(region)] = _listaAño 
           _resultadoFranja['listaSitiosRegion'+str(region)] = _listaSitios
           _resultadoFranja['listaEpocaRegion'+str(region)] = _listaEpocas
           _particularBestResult['epoca'+str(region)] = _especificoEpocaResult
           _particularBestVar['epoca'+str(region)] = _especificoEpocaVar
           
           _particularBestResultF['epoca'+str(region)] = _especificoEpocaResultF
           _particularBestVarF['epoca'+str(region)] = _especificoEpocaVarF
           _particularBestResultP['epoca'+str(region)] = _especificoEpocaResultP
      
        #nivel franja
       _listaFranja['resultados'+str(franja)] = _resultadoFranja
       _bestResult['general'] = _regionBestResult
       _bestVar['general'] = _regionBestVar
       _bestResult['sitio'] = _sitioResult
       _bestVar['sitio'] = _sitioVar
       _bestResult['particular'+str(franja)] = _particularBestResult
       _bestVar['particular'+str(franja)] = _particularBestVar
       _listaBestResult[franja] = _bestResult
       _listaBestVar[franja] = _bestVar
       
       _bestResultF['general'] = _regionBestResultF
       _bestVarF['general'] = _regionBestVarF
       _bestResultF['sitio'] = _sitioResultF
       _bestVarF['sitio'] = _sitioVarF
       _bestResultF['particular'+str(franja)] = _particularBestResultF
       _bestVarF['particular'+str(franja)] = _particularBestVarF
       _listaFResult[franja] = _bestResultF
       _listaFVar[franja] = _bestVarF
       
       _bestResultP['general'] = _regionBestResultP
       _bestResultP['sitio'] = _sitioResultP
       _bestResultP['particular'+str(franja)] = _particularBestResultP
       _listaPResult[franja] = _bestResultP
       
    _listaFranja['GeneralBestResult'] = _listaBestResult
    _listaFranja['GeneralBestVar'] = _listaBestVar
    _listaFranja['GeneralPResult'] = _listaPResult
    _listaFranja['GeneralFResult'] = _listaFResult
    _listaFranja['GeneralFVar'] = _listaFVar
    return _listaFranja   

def amountComplete(n):
     if fabs(n) > 0.00009:
          return str(n)
     elif '.' in str(n):
          ent = str(n).split('.')
          ent2 = ent[1].split('e')
          part2 = str(ent[0]+ent2[0])
          
          ccero = ent[1].split('-')
          part1 = ''
          
          if int(ccero[1]) >= 10:
               part1 = str('0.'+'0'*(int(ccero[1])-1))
          else:
               x = []
               for i in ccero[1]:
                    x.append(i)
               part1 = str('0.'+'0'*(int(ccero[1])-1))
     
          return part1+part2
     else:
          ent = str(n).split('e')
          part2 = ent[0]
          ccero = ent[1].replace('-','')
          part1 = ''
     
          if int(ccero) >= 10:
               part1 = str('0.'+'0'*(int(ccero)-1))
          else:
               x = []
               for i in ccero:
                    x.append(i)
               part1 = str('0.'+'0'*(int(x[1])-1))
     
          return part1+part2

def autolabel(rects, _franja,_opc,ax,generalVarResult,generalBestResultP):
    i = 0
    for rect in rects:
        height = rect.get_height()
        label = ''
        if _opc == 1:
            if generalBestResultP[_franja][i] != 0 or generalBestResultP[_franja][i] != -0:
                label = generalVarResult[_franja][i]
            else:
                label = ''
        else:
            if generalBestResultP[_franja][i] != 0 or generalBestResultP[_franja][i] != -0:
                _round = round(generalBestResultP[_franja][i],5)
                if generalBestResultP[_franja][i] < 0.0001:
                    _round = 0.00001 
                label = generalVarResult[_franja][i] +'\n' +str(amountComplete(_round))
            else:
                label = '' +'\n' +str('')
        ax.annotate('{}'.format(label),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", fontsize=16, weight='bold',
                    ha='center', va='bottom')
        i = i+1

def crearRuta(_ruta):
    try:
        os.makedirs(str(_ruta))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def graficar(listResultadoGeneral,listaBiologico,lista_franja,lista_region, _ruta,_title, _opc,_colors): 
    generalVarResult = {}
    generalBestResult = {}
    generalBestResultP = {}
    
    _auxList = list()
    for _reg in list(lista_region):
        _auxList.append(_reg)
    
    lista_region = _auxList
    
    for _fran in list(lista_franja):     
        _i = 0
        for _x in list(lista_region):
            _auxResult = list()
            _auxVar = list()
            if _opc == 1:
                for _y in list(listaBiologico):
                    _auxResult.append(listResultadoGeneral[_y]['GeneralBestResult'][_fran]['general'][_i])
                    _auxVar.append(listResultadoGeneral[_y]['GeneralBestVar'][_fran]['general'][_i])
            else: 
                _auxResultP = list()
                for _y in list(listaBiologico):
                    _auxResult.append(listResultadoGeneral[_y]['GeneralFResult'][_fran]['general'][_i])
                    _auxVar.append(listResultadoGeneral[_y]['GeneralFVar'][_fran]['general'][_i])
                    _auxResultP.append(listResultadoGeneral[_y]['GeneralPResult'][_fran]['general'][_i])
                generalBestResultP[_i] = _auxResultP
            generalVarResult[_i] = _auxVar
            generalBestResult[_i] = _auxResult
            _i += 1
                 
        _n = len(lista_region)
        _X = np.arange(1,_n,1)
    
        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
        x = np.arange(len(listaBiologico))
        width = 1.0/(_n+0.5)
        
        fig, ax = plt.subplots()  
        
        x_color = 0
        _listRect = list()
        for result in range(len(generalVarResult)):
            _listRect.append(ax.bar(np.arange(len(listaBiologico))+result*width, np.array(generalBestResult[result]), width, label=str(lista_region[result]),color=_colors[x_color]))
            x_color += 1
            
        #ax.set_title('Análisis general regional franja '+str(_fran))
        ax.set_xticks(x)
        ax.set_xticklabels(list(listaBiologico),ha='left')
        if _opc == 1:
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
            ax.set_ylabel('Puntuación Z')
            _title2 = _title+'(Boruta)(F'+str(_fran)+')'
        else:
            ax.yaxis.set_ticks(np.arange(0, 220, 20))
            ax.set_ylabel('Valor (F)')
            _title2 = _title+'(Anova)(F'+str(_fran)+')'
        ax.xaxis.set_ticks(_X,tuple(listaBiologico))
        ax.set_xlabel('Parámetro biológico')
        
        ax.legend()
    
        f = 0
        for franja in list(_listRect):  
            if _opc == 2:
                autolabel(franja,f,_opc,ax,generalVarResult,generalBestResultP)
            else:
                autolabel(franja,f,_opc,ax,generalVarResult,generalBestResult)
            f = f+1
        
        _ruta2 = _ruta+'//'
        if _opc == 1:
            _ruta2 = _ruta+'Boruta/'
        else:
            _ruta2 = _ruta+'Anova/'
        crearRuta(_ruta2)
        
        fig.tight_layout()
        plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
        plt.show()
  
def graficarGeneralAño(listResultadoGeneral,listaBiologico,lista_franja,lista_region,lista_año, _ruta,_title,_opc,_colors): 
    generalVarResult = {}
    generalBestResult = {}
    generalBestResultP = {}
    
    for _fran in list(lista_franja):    
        for _region in list(lista_region):
            _i = 0
            if _opc == 1:
                for _y in list(listaBiologico):        
                    generalBestResult[_i] = listResultadoGeneral[_y]['GeneralBestResult'][_fran]['particular'+str(_fran)]['año'+str(_region)]
                    generalVarResult[_i] = listResultadoGeneral[_y]['GeneralBestVar'][_fran]['particular'+str(_fran)]['año'+str(_region)]
                    _i += 1
            else:
                for _y in list(listaBiologico):        
                    generalBestResult[_i] = listResultadoGeneral[_y]['GeneralFResult'][_fran]['particular'+str(_fran)]['año'+str(_region)]
                    generalVarResult[_i] = listResultadoGeneral[_y]['GeneralFVar'][_fran]['particular'+str(_fran)]['año'+str(_region)]
                    generalBestResultP[_i] = listResultadoGeneral[_y]['GeneralPResult'][_fran]['particular'+str(_fran)]['año'+str(_region)]
                    _i += 1
                    
            _n = len(listaBiologico)
            _X = np.arange(1,_n,1)
        
            if _opc == 1:
                plt.rcParams['figure.figsize']=(15,10)
                plt.rcParams.update({'font.size': 20})
            else:
                plt.rcParams['figure.figsize']=(20,10)
                plt.rcParams.update({'font.size': 20})
            x = np.arange(len(lista_año))
            width = 1.0/(_n+1)
            
            fig, ax = plt.subplots()  
            
            x_color = 0
            _listRect = list()
            for result in range(len(generalVarResult)):
                _listRect.append(ax.bar(np.arange(len(lista_año))+result*width, np.array(generalBestResult[result]), width, label=str(listaBiologico[result]),color=_colors[x_color]))
                x_color += 1
                
            _title2 = ''
            #ax.set_title('Análisis anual de la región '+str(_region)+' franja '+str(_fran))
            ax.set_xticks(x)
            ax.set_xticklabels(list(lista_año),ha='left')
            if _opc == 1:
                ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
                ax.set_ylabel('Puntuación Z')
                _title2 = _title+' '+str(_region)+' (Boruta) (F'+str(_fran)+')'
            else:
                ax.yaxis.set_ticks(np.arange(0, 220, 20))
                ax.set_ylabel('Valor (F)')
                _title2 = _title+' '+str(_region)+' (Anova) (F'+str(_fran)+')'
            ax.xaxis.set_ticks(_X,tuple(lista_año))
            ax.set_xlabel('Año')
            
            ax.legend()
        
            f = 0
            for franja in list(_listRect):  
                if _opc == 2:
                    autolabel(franja,f,_opc,ax,generalVarResult,generalBestResultP)
                else:
                    autolabel(franja,f,_opc,ax,generalVarResult,generalBestResult)
                f = f+1
                
            _ruta2 = _ruta+str(_region)+'/'    
            if _opc == 1:
                _ruta2 = _ruta+'Boruta/'+str(_region)+'/'   
            else:
                _ruta2 = _ruta+'Anova/'+str(_region)+'/'   
                
            crearRuta(_ruta2)
            
            fig.tight_layout()
            plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
            plt.show()

def graficarEpocaAño(listResultadoGeneral,listaBiologico,lista_franja,lista_region,lista_año,lista_epoca, _ruta,_title,_opc,_colors): 
    generalVarResult = {}
    generalBestResult = {}
    generalBestResultP = {}
    
    for _fran in list(lista_franja):    
        for _region in list(lista_region):
            for _epoca in list(lista_epoca):
                _i = 0
                if _opc == 1:
                    for _y in list(listaBiologico):        
                        generalBestResult[_i] = listResultadoGeneral[_y]['GeneralBestResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca]
                        generalVarResult[_i] = listResultadoGeneral[_y]['GeneralBestVar'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca]
                        _i += 1
                else:
                    for _y in list(listaBiologico):        
                        generalBestResult[_i] = listResultadoGeneral[_y]['GeneralFResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca]
                        generalVarResult[_i] = listResultadoGeneral[_y]['GeneralFVar'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca]
                        generalBestResultP[_i] = listResultadoGeneral[_y]['GeneralPResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca]
                        _i += 1
                
                _n = len(listaBiologico)
                _X = np.arange(1,_n,1)
            
                if _opc == 1:
                    plt.rcParams['figure.figsize']=(15,10)
                    plt.rcParams.update({'font.size': 20})
                else:
                    plt.rcParams['figure.figsize']=(20,10)
                    plt.rcParams.update({'font.size': 20})
                x = np.arange(len(lista_año))
                width = 1.0/(_n+1)
                
                fig, ax = plt.subplots()  
                
                x_color = 0
                _listRect = list()
                for result in range(len(generalVarResult)):
                    _listRect.append(ax.bar(np.arange(len(lista_año))+result*width, np.array(generalBestResult[result]), width, label=str(listaBiologico[result]),color=_colors[x_color]))
                    x_color += 1
                    
                _title2 = ''
                #ax.set_title('Análisis anual de la región '+str(_region)+' '+str(_epoca)+' franja '+str(_fran))
                ax.set_xticks(x)
                ax.set_xticklabels(list(lista_año),ha='left')
                if _opc == 1:
                    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
                    ax.set_ylabel('Puntuación Z')
                    _title2 = _title+' '+str(_region)+' '+str(_epoca)+' (Boruta) (F'+str(_fran)+')'
                else:
                    ax.yaxis.set_ticks(np.arange(0, 220, 20))
                    ax.set_ylabel('Valor (F)')
                    _title2 = _title+' '+str(_region)+' '+str(_epoca)+' (ANova) (F'+str(_fran)+')'
                ax.xaxis.set_ticks(_X,tuple(lista_año))
                ax.set_xlabel('Año')
                
                ax.legend()
            
                f = 0
                for franja in list(_listRect):  
                    if _opc == 2:
                        autolabel(franja,f,_opc,ax,generalVarResult,generalBestResultP)
                    else:
                        autolabel(franja,f,_opc,ax,generalVarResult,generalBestResult)
                    f = f+1
                    
                _ruta2 = _ruta+str(_region)+'/'    
                if _opc == 1:
                    _ruta2 = _ruta+'Boruta/'+str(_region)+'/'   
                else:
                    _ruta2 = _ruta+'Anova/'+str(_region)+'/'   
                    
                crearRuta(_ruta2)
                
                fig.tight_layout()
                plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
                plt.show()

def graficarGeneralEstacionalidad(listResultadoGeneral,listaBiologico,lista_franja,lista_region,lista_epoca, _ruta,_title,_opc,_colors): 
    generalVarResult = {}
    generalBestResult = {}
    generalBestResultP = {}
    
    for _fran in list(lista_franja):    
        for _region in list(lista_region):
            _j = 0
            for _epoca in list(lista_epoca):
                _i = 0
                _auxResult = list()
                _auxVar = list()
                if _opc == 1:
                    for _y in list(listaBiologico):        
                         _auxResult.append(listResultadoGeneral[_y]['GeneralBestResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca][_i])
                         _auxVar.append(listResultadoGeneral[_y]['GeneralBestVar'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca][_i])
                    _i += 1                    
                else:
                    _auxResultP = list()
                    for _y in list(listaBiologico):        
                         _auxResult.append(listResultadoGeneral[_y]['GeneralFResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca][_i])
                         _auxVar.append(listResultadoGeneral[_y]['GeneralFVar'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca][_i])
                         _auxResultP.append(listResultadoGeneral[_y]['GeneralPResult'][_fran]['particular'+str(_fran)]['epoca'+str(_region)]['año'][_epoca][_i])
                    _i += 1
                    generalBestResultP[_j] =  _auxResultP
                generalBestResult[_j] =  _auxResult
                generalVarResult[_j] = _auxVar
                _j += 1
                
            _n = len(lista_epoca)
            _X = np.arange(1,_n,1)
        
            plt.rcParams['figure.figsize']=(15,10)
            plt.rcParams.update({'font.size': 20})
            x = np.arange(len(listaBiologico))
            width = 1.0/(_n+1)
            
            fig, ax = plt.subplots()  
            
            x_color = 0
            _listRect = list()
            for result in range(len(generalVarResult)):
                _listRect.append(ax.bar(np.arange(len(listaBiologico))+result*width, np.array(generalBestResult[result]), width, label=str(lista_epoca[result]),color=_colors[x_color]))
                x_color += 1
                
            _title2 = ''    
            #ax.set_title('Análisis general por estacionalidad de la región '+str(_region)+' franja '+str(_fran))
            ax.set_xticks(x)
            ax.set_xticklabels(list(listaBiologico),ha='left')
            if _opc == 1:
                ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
                ax.set_ylabel('Puntuación Z')
                _title2 = _title+' '+str(_region)+'(Boruta) (F'+str(_fran)+')'
            else:
                ax.yaxis.set_ticks(np.arange(0, 220, 20))
                ax.set_ylabel('Valor (F)')
                _title2 = _title+' '+str(_region)+'(Anova) (F'+str(_fran)+')'
            ax.xaxis.set_ticks(_X,tuple(listaBiologico))
            ax.set_xlabel('Parámetro biológico')
            ax.set_ylabel('Importancia (p)')
            
            ax.legend()
        
            f = 0
            for franja in list(_listRect):  
                if _opc == 2:
                    autolabel(franja,f,_opc,ax,generalVarResult,generalBestResultP)
                else:
                    autolabel(franja,f,_opc,ax,generalVarResult,generalBestResult)
                f = f+1
                
            _ruta2 = _ruta+str(_region)+'/'    
            if _opc == 1:
                _ruta2 = _ruta+'Boruta/'+str(_region)+'/'   
            else:
                _ruta2 = _ruta+'Anova/'+str(_region)+'/'   
                
            crearRuta(_ruta2)    
            fig.tight_layout()
            plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
            plt.show()

def graficarGeneralPorSitio(listResultadoGeneral,listaBiologico,lista_franja,lista_sitio, _ruta,_title,_opc,_colors): 
    generalVarResult = {}
    generalBestResult = {}
    generalBestResultP = {}
    
    for _fran in list(lista_franja):    
        _i = 0
        for _y in list(listaBiologico):     
             _auxResult = list()
             _auxVar = list()
             if _opc == 1:
                 for _sitio in list(lista_sitio):
                     _auxResult.append(listResultadoGeneral[_y]['GeneralBestResult'][_fran]['sitio'][_sitio])
                     _auxVar.append(listResultadoGeneral[_y]['GeneralBestVar'][_fran]['sitio'][_sitio])
             else:
                 _auxResultP = list()
                 for _sitio in list(lista_sitio):
                     _auxResult.append(listResultadoGeneral[_y]['GeneralFResult'][_fran]['sitio'][_sitio])
                     _auxVar.append(listResultadoGeneral[_y]['GeneralFVar'][_fran]['sitio'][_sitio])
                     _auxResultP.append(listResultadoGeneral[_y]['GeneralPResult'][_fran]['sitio'][_sitio])
                 generalBestResultP[_i] = _auxResultP
             generalBestResult[_i] = _auxResult
             generalVarResult[_i] = _auxVar
             
             _i += 1
        
        _n = len(listaBiologico)
        _X = np.arange(1,_n,1)
    
        if _opc == 2:
            plt.rcParams['figure.figsize']=(25,10)
            plt.rcParams.update({'font.size': 20})
        else:
            plt.rcParams['figure.figsize']=(15,10)
            plt.rcParams.update({'font.size': 20})
        x = np.arange(len(lista_sitio))
        width = 1.0/(_n+1)
        
        fig, ax = plt.subplots()  
        
        x_color = 0
        _listRect = list()
        for result in range(len(generalVarResult)):
            _listRect.append(ax.bar(np.arange(len(lista_sitio))+result*width, np.array(generalBestResult[result]), width, label=str(listaBiologico[result]),color=_colors[x_color]))
            x_color += 1
            
        _title2 = ''
        #ax.set_title('Análisis general por sitio franja '+str(_fran))
        ax.set_xticks(x)
        ax.set_xticklabels(list(lista_sitio),ha='left')
        if _opc == 1:
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
            ax.set_ylabel('Puntuación Z')
            _title2 = _title+'(Boruta)(F'+str(_fran)+')'
        else:
            ax.yaxis.set_ticks(np.arange(0, 220, 20))
            ax.set_ylabel('Valor (F)')
            _title2 = _title+'(Anova)(F'+str(_fran)+')' 
        ax.xaxis.set_ticks(_X,tuple(lista_sitio))
        ax.set_xlabel('Sitio')
        
        ax.legend()
    
        f = 0
        for franja in list(_listRect):  
            if _opc == 2:
                autolabel(franja,f,_opc,ax,generalVarResult,generalBestResultP)
            else:
                autolabel(franja,f,_opc,ax,generalVarResult,generalBestResult)
            f = f+1
            
        _ruta2 = _ruta
        if _opc == 1:
            _ruta2 = _ruta+'Boruta/'
        else:
            _ruta2 = _ruta+'Anova/'
            
        crearRuta(_ruta2)
        fig.tight_layout()
        plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
        plt.show()
    
def generarGrafica(_dataset,_componente,_ruta,_listaFranjaResult,_listaBiologico, _opc,_lista_franja,_lista_epoca,_opc2,_colors):
    lista_año = _dataset['Año'].drop_duplicates()
    _ruta2 = _ruta +'Resumen regional/'
    try:
        os.makedirs(str(_ruta2))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if _opc == 1:
        _title = 'Resumen general regional '
        graficar(_listaFranjaResult,_listaBiologico,_lista_franja,_componente,_ruta2,_title,_opc2,_colors)
    elif _opc == 2:
        _title = 'Resumen general por sitio'
        graficarGeneralPorSitio(_listaFranjaResult,_listaBiologico,_lista_franja,_componente,_ruta2,_title,_opc2,_colors)
    elif _opc == 3:
        _title = 'Resumen regional anual-temporalidad '
        graficarEpocaAño(_listaFranjaResult,_listaBiologico,_lista_franja,_componente,lista_año,_lista_epoca,_ruta2,_title,_opc2,_colors)        
    elif _opc == 4:
        _title = 'Resumen anual por componenete '
        graficarGeneralAño(_listaFranjaResult,_listaBiologico,_lista_franja,_componente,lista_año,_ruta2,_title,_opc2,_colors)            
    elif _opc == 5:
        _title = 'Resumen regional por componente y estacionalidad'
        graficarGeneralEstacionalidad(_listaFranjaResult,_listaBiologico,_lista_franja,_componente,_lista_epoca,_ruta2,_title,_opc2,_colors)
                 
def obtenerResultadoPorEspecie(_specie,listaBiologico,listaSitioR,_dataset):
    ListaResultado = {}
    for _aux in list(listaBiologico):
        _listaAux = listaBiologico.copy()
        _listaAux.remove(_aux)
        
        _dataAux = _dataset.copy()
        for _delete in list(_listaAux):
            _dataAux = _dataAux.drop([_delete], axis=1)
            
        _title = 'Boruta resultados '+_specie+'/Graficas/Gráfica '+_aux.replace('.','')
        ListaResultado[_aux] = getResultado(_aux,_title,_dataAux,listaSitioR)
    return ListaResultado

def graficarResultados(_specie,listaBiologico,listaSitioR,_dataset,_colors,ListaResultado):   
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,1,lista_franja,'',1,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,4,lista_franja,'',1,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,3,lista_franja,lista_epoca,1,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,5,lista_franja,lista_epoca,1,_colors)
    generarGrafica(_dataset,lista_sitio,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,2,lista_franja,'',1,_colors)
    
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,1,lista_franja,'',2,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,4,lista_franja,'',2,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,3,lista_franja,lista_epoca,2,_colors)
    generarGrafica(_dataset,lista_region,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,5,lista_franja,lista_epoca,2,_colors)
    generarGrafica(_dataset,lista_sitio,'Boruta resultados '+_specie+'/Graficas/Resultado General/',ListaResultado,listaBiologico,2,lista_franja,'',2,_colors)
    
    
def exportarResultadosExcel(rutaResultados,listaBiologico,listaResultado,lista_region,lista_franja,lista_epoca,_dataset,_specie):
    for biologico in list(listaBiologico):
        _ruta = rutaResultados+'Export excel '+_specie+'/Boruta/Archivos/regional/'+biologico+'/'
        crearRuta(_ruta)
        writer = pd.ExcelWriter(_ruta+'resultados por región.xlsx')
        for franja in list(lista_franja):
            _dataAux = _dataset.loc[_dataset.loc[:, 'Franja'] == franja]
            for region in list(lista_region):
                _dataAux2 = _dataAux.loc[_dataAux.loc[:, 'Region'] == region]
                _lista_sitio = _dataAux2['Sitio'].drop_duplicates()
                _aux = listaResultado[biologico]['resultados'+str(franja)][region+str(franja)]['region'+str(region)]
                _aux.to_excel(writer, sheet_name=region+str(franja), index=False,header=True)
                
                _lista_año = _dataAux2['Año'].drop_duplicates()
                _ruta2 = _ruta+str(franja)+'/'+region+'/'
                crearRuta(_ruta2)
                writerAño = pd.ExcelWriter(_ruta2+'resultados por año '+region+'.xlsx')
                for año in list(_lista_año):
                    _añoRegion = listaResultado[biologico]['resultados'+str(franja)]['listaAño'+region]['año'+str(año)]['año'+str(año)]
                    if _añoRegion.empty != True:
                        _añoRegion.to_excel(writerAño, sheet_name=str(año)+region+str(franja), index=False,header=True)
                writerAño.save()
                writerAño.close()
                
                writerSitio = pd.ExcelWriter(_ruta2+'resultados por sitio '+region+'.xlsx')
                for sitio in list(_lista_sitio):
                    _resultadoSitio = listaResultado[biologico]['resultados'+str(franja)]['listaSitiosRegion'+region]['sitio'+str(sitio)]['sitio'+str(sitio)]
                    if _resultadoSitio.empty != True:
                        _resultadoSitio.to_excel(writerSitio, sheet_name=region+str(franja)+sitio, index=False,header=True)
                writerSitio.save()
                writerSitio.close()
                
                writerEpoca = pd.ExcelWriter(_ruta2+'resultados por epoca '+region+'.xlsx')
                writerAñoEpoca = pd.ExcelWriter(_ruta2+'resultados anuales por epoca '+region+'.xlsx')
                for epoca in list(lista_epoca):
                    _epocaR = listaResultado[biologico]['resultados'+str(franja)]['listaEpocaRegion'+region][epoca+'General'+str(region)]['epoca'+str(epoca)]
                    if _epocaR.empty != True:
                        _epocaR.to_excel(writerEpoca, sheet_name=str(epoca)+region+str(franja), index=False,header=True)
                    
                    _dataAux3 = _dataAux2.loc[_dataAux2.loc[:, 'Epoca del año'] == epoca]
                    _lista_año = _dataAux3['Año'].drop_duplicates()
                    
                    for año in list(_lista_año):
                        _añoEpoca = listaResultado[biologico]['resultados'+str(franja)]['listaEpocaRegion'+region][epoca+'Años'+str(region)]['año'+str(año)]['año'+str(año)]
                        if _añoEpoca.empty != True:
                            _añoEpoca.to_excel(writerAñoEpoca, sheet_name=str(epoca)+str(año)+region+str(franja), index=False,header=True)
                writerAñoEpoca.save()
                writerAñoEpoca.close()
                writerEpoca.save()
                writerEpoca.close()
                
        writer.save()
        writer.close()

def exportarExcelResumen(rutaResultados,listaBiologico,listaResultado,lista_region,lista_franja,lista_epoca,lista_sitios,_dataset,_specie):
    for biologico in list(listaBiologico):
        _ruta = rutaResultados+'Export excel '+_specie+'/Boruta/Resumen/regional/'+biologico+'/'
        crearRuta(_ruta)
        writer = pd.ExcelWriter(_ruta+'resultados por sitio.xlsx')
        writerRegion = pd.ExcelWriter(_ruta+'resultados por región.xlsx')
        writerAño = pd.ExcelWriter(_ruta+'resultados por año.xlsx')
        writerEpoca = pd.ExcelWriter(_ruta+'resultados por epoca.xlsx')
        for franja in list(lista_franja):
            _resultadoSitio = pd.DataFrame()
            bestResult = list()
            bestVar = list()
            fresult = list()
            fvar = list()
            presult = list()
            for sitio in list(lista_sitio):
                bestResult.append(listaResultado[biologico]['GeneralBestResult'][franja]['sitio'][sitio])
                bestVar.append(listaResultado[biologico]['GeneralBestVar'][franja]['sitio'][sitio])
                fresult.append(listaResultado[biologico]['GeneralFResult'][franja]['sitio'][sitio])
                fvar.append(listaResultado[biologico]['GeneralFVar'][franja]['sitio'][sitio])
                presult.append(listaResultado[biologico]['GeneralPResult'][franja]['sitio'][sitio])    
            _resultadoSitio['Sitio'] = lista_sitio
            _resultadoSitio['Var_Boruta'] = bestVar
            _resultadoSitio['Importancia'] = bestResult
            _resultadoSitio['Var_F'] = fvar
            _resultadoSitio['F'] = fresult
            _resultadoSitio['P'] = presult
            _resultadoSitio.to_excel(writer, sheet_name=str(franja), index=False,header=True)
            
            _resultadoRegion = pd.DataFrame()
            crearRuta(_ruta+'Epoca región/')
            
            for region in list(lista_region): 
                writerEpocaAño = pd.ExcelWriter(_ruta+'Epoca región/resultados por epoca '+region+' F'+str(franja)+'.xlsx')
                _resultadosAño = pd.DataFrame()
                _resultadosAño['Año'] = lista_año
                _resultadosAño['Var_Boruta'] = listaResultado[biologico]['GeneralBestResult'][franja]['particular'+str(franja)]['año'+region]
                _resultadosAño['Importancia'] = listaResultado[biologico]['GeneralBestVar'][franja]['particular'+str(franja)]['año'+region]
                _resultadosAño['Var_F'] = listaResultado[biologico]['GeneralFResult'][franja]['particular'+str(franja)]['año'+region]
                _resultadosAño['F'] = listaResultado[biologico]['GeneralFVar'][franja]['particular'+str(franja)]['año'+region]
                _resultadosAño['P'] = listaResultado[biologico]['GeneralPResult'][franja]['particular'+str(franja)]['año'+region]    
                _resultadosAño.to_excel(writerAño, sheet_name=region+'AñoF'+str(franja), index=False,header=True)
                
                _resultadosEpoca = pd.DataFrame()
                _resultadosEpoca['Epoca'] = lista_epoca
                _resultadosEpoca['Var_Boruta'] = listaResultado[biologico]['GeneralBestResult'][franja]['particular'+str(franja)]['epoca'+region]['general']
                _resultadosEpoca['Importancia'] = listaResultado[biologico]['GeneralBestVar'][franja]['particular'+str(franja)]['epoca'+region]['general']
                _resultadosEpoca['Var_F'] = listaResultado[biologico]['GeneralFResult'][franja]['particular'+str(franja)]['epoca'+region]['general']
                _resultadosEpoca['F'] = listaResultado[biologico]['GeneralFVar'][franja]['particular'+str(franja)]['epoca'+region]['general']
                _resultadosEpoca['P'] = listaResultado[biologico]['GeneralPResult'][franja]['particular'+str(franja)]['epoca'+region]['general']    
                _resultadosEpoca.to_excel(writerEpoca, sheet_name=region+'AñoF'+str(franja), index=False,header=True)
                
                for epoca in list(lista_epoca):
                    _resultadosEpocaAño = pd.DataFrame()
                    _resultadosEpocaAño['Año'] = lista_año
                    _resultadosEpocaAño['Var_Boruta'] = listaResultado[biologico]['GeneralBestResult'][franja]['particular'+str(franja)]['epoca'+region]['año'][epoca]
                    _resultadosEpocaAño['Importancia'] = listaResultado[biologico]['GeneralBestVar'][franja]['particular'+str(franja)]['epoca'+region]['año'][epoca]
                    _resultadosEpocaAño['Var_F'] = listaResultado[biologico]['GeneralFResult'][franja]['particular'+str(franja)]['epoca'+region]['año'][epoca]
                    _resultadosEpocaAño['F'] = listaResultado[biologico]['GeneralFVar'][franja]['particular'+str(franja)]['epoca'+region]['año'][epoca]
                    _resultadosEpocaAño['P'] = listaResultado[biologico]['GeneralPResult'][franja]['particular'+str(franja)]['epoca'+region]['año'][epoca]    
                    _resultadosEpocaAño.to_excel(writerEpocaAño, sheet_name=epoca+'AñoF'+str(franja), index=False,header=True)
                writerEpocaAño.save()
                writerEpocaAño.close()    
            _resultadoRegion['Región'] = lista_region
            _resultadoRegion['Var_Boruta'] = listaResultado[biologico]['GeneralBestVar'][franja]['general']
            _resultadoRegion['Importancia'] = listaResultado[biologico]['GeneralBestResult'][franja]['general']
            _resultadoRegion['Var_F'] = listaResultado[biologico]['GeneralFVar'][franja]['general']
            _resultadoRegion['F'] = listaResultado[biologico]['GeneralFResult'][franja]['general']
            _resultadoRegion['P'] = listaResultado[biologico]['GeneralPResult'][franja]['general']
            _resultadoRegion.to_excel(writerRegion, sheet_name='F'+str(franja), index=False,header=True)
            
        writer.save()
        writer.close()
        writerRegion.save()
        writerRegion.close()
        writerAño.save()
        writerAño.close()
        writerEpoca.save()
        writerEpoca.close()

cargarDatos('boruta/datasetX.csv')
plt.figure (figsize = (15,5))
plt.rcParams['figure.figsize']=(15,10)
    
_specie = 'Rm'
colors =  ['blue', 'darkorange', 'green', 'red','yellow', 'cyan', 'gray'] 

listResultadoGeneral = obtenerResultadoPorEspecie(_specie,listaBiologico,listaSitioR,dataset)
graficarResultados(_specie,listaBiologico,listaSitioR,dataset,colors,listResultadoGeneral)
exportarResultadosExcel('Boruta resultados '+_specie+'/',listaBiologico,listResultadoGeneral,lista_region,lista_franja,lista_epoca,dataset,_specie)       
exportarExcelResumen('Boruta resultados '+_specie+'/',listaBiologico,listResultadoGeneral,lista_region,lista_franja,lista_epoca,lista_sitio,dataset,_specie)       