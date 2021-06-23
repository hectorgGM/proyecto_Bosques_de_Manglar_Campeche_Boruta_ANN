# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:59:01 2020

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
from sklearn.impute import SimpleImputer
### make X and y
def cargarDatos(ruta):
    global lista_franja, lista_epoca, lista_año, dataset, listaBiologico
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
    
    listaBiologico = list()
    listaBiologico.append('Propágulos')
    listaBiologico.append('Hojas')
    listaBiologico.append('Flores')
    listaBiologico.append('Hojarasca')
    
    _lista_epocaAux = list()
    for _epoca in list(lista_epoca):
        _lista_epocaAux.append(_epoca)
    lista_epoca = _lista_epocaAux 

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
    for iter_ in range(20):
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
    _resultado['f_regression'] = feat_imp_X
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
    _data = pd.DataFrame()
    _data = _data1.copy()
    datosProcesados = {}
    _y = _data.pop(var)
    _y = np.array(_y)
    _y = _y.reshape(-1,1)
    
    _X = _data.copy().values
    
    _X = _X.astype('float64')
    _y = _y.astype('float64')
    _X = pd.DataFrame(_X)
    
    ### make X_shadow by randomly permuting each column of X
    np.random.seed(42)
    _X_shadow = _X.apply(np.random.permutation)
    _X_shadow.columns = ['shadow_' + feat for feat in _data.columns]
    ### make X_boruta by appending X_shadow to X
    _X_boruta = pd.concat([_X, _X_shadow], axis = 1)
    
    datosProcesados['X'] = _X
    datosProcesados['y'] = _y
    datosProcesados['X_boruta'] = _X_boruta 
    
    return datosProcesados

def getDataMatriz(data, var):
    _data = pd.DataFrame()
    _data = data.copy()
    datosProcesados = {}
    _X = _data.copy().values
    
    _y = _data.pop(var)
    _y = np.array(_y)
    _y = _y.reshape(-1,1)
    
    labelencoder_X_2 = LabelEncoder()
    _X[:, 1] = labelencoder_X_2.fit_transform(_X[:, 1])
    _X[:, -1] = labelencoder_X_2.fit_transform(_X[:, -1])
    _X = _X.astype('float64')
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(_X[:,0:,])
    _X[:,0:,] = imputer.transform(_X[:,0:,])
    imputer = imputer.fit(_y[:,0:1,])
    _y[:,0:1,] = imputer.transform(_y[:,0:1,])
    _X = _X.astype('float64')
    _y = _y.astype('float64')
    _X = pd.DataFrame(_X)
    
    datosProcesados['X'] = _X
    datosProcesados['y'] = _y
    
    return datosProcesados

def imp_df(column_names, importances):
    df = pd.DataFrame({'Parámetro fisicoquímico': column_names,
                       'Importancia (Z)': importances}) \
           .sort_values('Importancia (Z)', ascending = False) \
           .reset_index(drop = True)
    return df

def getResultadoGeneral(var,title,_dataset, i):
    _listaFranja = {}
    _listEpocaResult = {}
    _listEpocaVar = {}
    _listaFranjaResult = {}
    _listaFranjaVar = {}
    _listaGeneralResult = list()
    _listaGeneralVar = list()
    _listGeneralEResult = list()
    _listGeneralEVar = list()
    
    #F y p value
    _listEpocaResultF = {}
    _listEpocaVarF = {}
    _listaFranjaResultF = {}
    _listaFranjaVarF = {}
    _listaGeneralResultF = list()
    _listaGeneralVarF = list()
    _listGeneralEResultF = list()
    _listGeneralEVarF = list()
    
    _listEpocaResultP = {}
    _listaFranjaResultP = {}
    _listaGeneralResultP = list()
    _listGeneralEResultP = list()
    
    for franja in lista_franja:
       data = _dataset.loc[_dataset.loc[:, 'Franja'] == franja] 
       data = data.drop(['Franja'], axis=1)
       data = data.drop(['Region'], axis=1)
       data = data.drop(['Mes'], axis=1)
       data = data.drop(['Sitio'], axis=1)
       _resultadosFranja = {}
       
       dataR = data
       dataR = dataR.drop(['Año'], axis=1)
       dataR = dataR.drop(['Epoca del año'], axis=1)
       _listaEpocas = {}
       _listaAño = {}
       if(dataR.empty != True):
           _datosProcesados = getData(dataR, var)
           X_boruta = _datosProcesados['X_boruta']
           y = _datosProcesados['y']
           X = _datosProcesados['X']
            
           _resultado = forestResults(X_boruta,y,X,dataR,var)
           _area = getRanking(X,y,dataR,var)
           _resultado['Rank'] = _area['Rank'] 
           _resultado = _resultado[['Hits','Rank','Var','F','Importance','p','f_regression']]
           _resultado = _resultado.sort_values(['Rank', 'F', 'Importance'])
           
           _resultadosFranja['area'+str(franja)] = _area
           _resultadosFranja['franja'+str(franja)] = _resultado
           
           _aux = _resultado.loc[_resultado["Importance"].idxmax()]
           _listaGeneralResult.append(_aux['Importance'])
           _listaGeneralVar.append(_aux['Var'])
           
           _aux = _resultado.loc[_resultado["F"].idxmax()]
           _listaGeneralResultF.append(_aux['F'])
           _listaGeneralVarF.append(_aux['Var'])
           
           _aux = _resultado.loc[_resultado["F"].idxmax()]
           _listaGeneralResultP.append(_aux['p'])
           
           _listaFranja['general'+str(franja)] = _resultadosFranja
             
           sps = str(title)+'/Franja'+str(franja)+'/estatal'
           try:
                os.makedirs(str(sps))
           except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
           fig = plt.figure()
           ax = fig.add_subplot(111)
           sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['f_regression']), orient = 'h', color = 'royalblue') \
               .set_title('Análisis estatal F'+str(franja), fontsize = 20)
           ax.xaxis.set_ticks(np.arange(0, 1, 0.1)) 
           plt.savefig(str(sps)+'/ F'+str(franja), bbox_inches='tight')
       
       _generalBestResult = list()
       _generalVarResult = list()
       
       #F y p value
       _generalBestResultF = list()
       _generalVarResultF = list()
       _generalBestResultP = list()
       for año in list(lista_año):
           _resultadosAño = {}
           _dataAño = data.loc[data.loc[:, 'Año'] == año]
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
                   _generalBestResult.append(_aux['Importance'])
                   _generalVarResult.append(_aux['Var'])
                   
                   _aux = _resultado.loc[_resultado["F"].idxmax()]
                   _generalBestResultF.append(_aux['F'])
                   _generalVarResultF.append(_aux['Var'])
                   
                   _aux = _resultado.loc[_resultado["F"].idxmax()]
                   _generalBestResultP.append(_aux['p'])
                   
               except OSError as e:
                   if e.errno != errno.EEXIST:
                       raise
               
               
               _listaAño['año'+str(año)] = _resultadosAño 
           
               sps = str(title)+'/Franja'+str(franja)+'/estatal/año'
               try:
                    os.makedirs(str(sps))
               except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
               fig = plt.figure()
               ax = fig.add_subplot(111)
               sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['f_regression']), orient = 'h', color = 'royalblue') \
                   .set_title('Análisis estatal F'+str(franja)+' '+str(año), fontsize = 20)
               ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
               plt.savefig(str(sps)+'/F'+str(franja)+' '+str(año), bbox_inches='tight')
           
           else:
               _generalBestResult.append(float(0))
               _generalVarResult.append('')
               
               _generalBestResultF.append(float(0))
               _generalVarResultF.append('')
               _generalBestResultP.append(float(0))
            
       _listaFranjaResult[franja] = _generalBestResult
       _listaFranjaVar[franja] = _generalVarResult
       
       _listaFranjaResultF[franja] = _generalBestResultF
       _listaFranjaVarF[franja] = _generalVarResultF
       _listaFranjaResultP[franja] = _generalBestResultP

       
       for epoca in list(lista_epoca):
           _resultadosEpoca = {}
           _dataEpoca = data.loc[data.loc[:, 'Epoca del año'] == epoca]
           _dataEpoca = _dataEpoca.drop(['Epoca del año'], axis=1)
           _dataEpoca = _dataEpoca.drop(['Año'], axis=1)
           
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
                   _listGeneralEResult.append(_aux['Importance'])
                   _listGeneralEVar.append(_aux['Var'])
                   
                   _aux = _resultado.loc[_resultado["F"].idxmax()]
                   _listGeneralEResultF.append(_aux['F'])
                   _listGeneralEVarF.append(_aux['Var'])
                   
                   _aux = _resultado.loc[_resultado["F"].idxmax()]
                   _listGeneralEResultP.append(_aux['p'])
               except OSError as e:
                   if e.errno != errno.EEXIST:
                       raise
               
             
               _listaEpocas[str(epoca)] = _resultadosEpoca            
           
               sps = str(title)+'/Franja'+str(franja)+'/estatal/epoca'
               try:
                    os.makedirs(str(sps))
               except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
               fig = plt.figure()
               ax = fig.add_subplot(111)
               sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['f_regression']), orient = 'h', color = 'royalblue') \
                   .set_title('Análisis estatal F'+str(franja)+' '+str(epoca), fontsize = 20)
               ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
               plt.savefig(str(sps)+'/ F'+str(franja)+' '+str(epoca), bbox_inches='tight')
           
           else:
               _listGeneralEResult.append(float(0))
               _listGeneralEVar.append('')
               
               _listGeneralEResultF.append(float(0))
               _listGeneralEVarF.append('')               
               _listGeneralEResultP.append(float(0))

           _listaEpocaAño = {}
           _generalEpocaResult = list()
           _generalEpocaVar = list()
           
           _generalEpocaResultF = list()
           _generalEpocaVarF = list()           
           _generalEpocaResultP = list()
           
           for año in list(lista_año):
               _resultadosEpocaAño = {}
               _dataAño = data.loc[data.loc[:, 'Epoca del año'] == epoca]
               _dataAño = _dataAño.loc[_dataAño.loc[:, 'Año'] == año]
               _dataAño = _dataAño.drop(['Epoca del año'], axis=1)
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
                                    
                   _listaEpocaAño['año'+str(año)] = _resultadosEpocaAño 
                   
                   sps = str(title)+'/Franja'+str(franja)+'/estatal/epoca/año'
                   try:
                        os.makedirs(str(sps))
                   except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                   fig = plt.figure()
                   ax = fig.add_subplot(111)
                   sns.barplot(x = 'Importancia (Z)', y = 'Parámetro fisicoquímico', data = imp_df(_resultado['Var'], _resultado['f_regression']), orient = 'h', color = 'royalblue') \
                       .set_title('Análisis estatal F'+str(franja)+' '+str(epoca)+' '+str(año), fontsize = 20)
                   ax.xaxis.set_ticks(np.arange(0, 1, 0.1))     
                   plt.savefig(str(sps)+'/ F'+str(franja)+' '+str(epoca)+' '+str(año), bbox_inches='tight')
           
               else:
                    _generalEpocaResult.append(float(0))
                    _generalEpocaVar.append('')
                   
                    _generalEpocaResultF.append(float(0))
                    _generalEpocaVarF.append('')
                    _generalEpocaResultP.append(float(0))
                    
           _listEpocaResult[i] = _generalEpocaResult
           _listEpocaVar[i] =  _generalEpocaVar
           _listEpocaResultF[i] = _generalEpocaResultF
           _listEpocaVarF[i] =  _generalEpocaVarF
           _listEpocaResultP[i] = _generalEpocaResultP
           i = i+1
           _listaEpocas[str(epoca)+':año'] = _listaEpocaAño                    
           _listaFranja['listaAñoF'+str(franja)] = _listaAño 
           _listaFranja['listaEpocaF'+str(franja)] = _listaEpocas
           _listaFranja['listaEpocaResult'] = _listEpocaResult
           _listaFranja['listaEpocaVar'] = _listEpocaVar
           _listaFranja['listaFranjaResult'] = _listaFranjaResult
           _listaFranja['listaFranjaVar'] = _listaFranjaVar
           
           _listaFranja['listaGeneralResult'] = _listaGeneralResult
           _listaFranja['listaGeneralVar'] = _listaGeneralVar
           _listaFranja['listaGeneralEResult'] = _listGeneralEResult
           _listaFranja['listaGeneralEVar'] = _listGeneralEVar
           
           #F y p values
           _listaFranja['listaEpocaResultF'] = _listEpocaResultF
           _listaFranja['listaEpocaVarF'] = _listEpocaVarF
           _listaFranja['listaFranjaResultF'] = _listaFranjaResultF
           _listaFranja['listaFranjaVarF'] = _listaFranjaVarF
           
           _listaFranja['listaGeneralResultF'] = _listaGeneralResultF
           _listaFranja['listaGeneralVarF'] = _listaGeneralVarF
           _listaFranja['listaGeneralEResultF'] = _listGeneralEResultF
           _listaFranja['listaGeneralEVarF'] = _listGeneralEVarF
           
           _listaFranja['listaEpocaResultP'] = _listEpocaResultP
           _listaFranja['listaFranjaResultP'] = _listaFranjaResultP
           
           _listaFranja['listaGeneralResultP'] = _listaGeneralResultP
           _listaFranja['listaGeneralEResultP'] = _listGeneralEResultP
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
            if generalBestResultP[_franja][i] != 0:
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
        
def graficar(_generalBestResult,_generalVarResult,_lista, _componente, _ruta,_title,_opc,generalBestResultP,_colors):     
    _n = len(_generalBestResult)
    _X = np.arange(1,_n,1)
 
    plt.rcParams['figure.figsize']=(15,10)
    plt.rcParams.update({'font.size': 20})
    x = np.arange(len(_lista))
    width = 1.0/(_n+0.5)
    
    fig, ax = plt.subplots()  
    
    _listRect = list()
    franja = 1
    i = 1
    x_color = 0
    for result in range(_n):
        _listRect.append(ax.bar(np.arange(len(_lista))+result*width, np.array(_generalBestResult[i]), width, label='Franja '+str(franja), color = _colors[x_color]))

        i += 1
        x_color += 1
        franja += 1
        
    #ax.set_title('Análisis general estatal ('+_componente+')')
    ax.set_xticks(x)
    ax.set_xticklabels(list(_lista),ha='left')
    _title2 =''
    if _opc == 1:
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
        ax.set_ylabel('Puntuación Z')
        _title2 = _title+'(Boruta)'
    else:
        ax.yaxis.set_ticks(np.arange(0, 220, 20))
        ax.set_ylabel('Valor (F)')
        _title2 = _title+'(Anova)' 
    ax.xaxis.set_ticks(_X,tuple(_lista))
    ax.set_xlabel('Año')
    if (len(lista_franja) > 1):
        ax.legend()

    f = 1
    for franja in list(_listRect):  
        if _opc == 2:
            autolabel(franja,f,_opc,ax,_generalVarResult,generalBestResultP)
        else:
            autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
        f = f+1
        
    _ruta2 = _ruta+'//'
    if _opc == 1:
        _ruta2 = _ruta+'Boruta/'+_componente.replace('.','')+'/'
    else:
        _ruta2 = _ruta+'Anova/'+_componente.replace('.','')+'/'
    crearRuta(_ruta2)    
        
    fig.tight_layout()
    plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
    plt.show()
           
def graficarResumen(_listResultadoGeneral,_listBiologico,_lista_año, _componente, _ruta,_title,_opc,generalBestResultP,_colors):         
    _generalVarResult = {}
    _generalBestResult = {}
    _generalBestResultP = {}
    _n = len(_listBiologico)
    _X = np.arange(1,_n,1)    
    
    def separarByFranja(_franja,_lista):
        _listAux = list()
        _n = len(_lista)
        _intervalo = int(_n/len(lista_franja))
        if(_franja == 1):
            _n = _intervalo
            _intervalo = 0
        
        for _i in range (_intervalo,_n):
            _listAux.append(_lista[_i])
            
        return _listAux
    
    for _franja in list(lista_franja):
        _z = 0
        for _biologico in list(_listBiologico):
            if _opc == 1:
                _generalVarResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaFranjaVar'][_franja])
                _generalBestResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaFranjaResult'][_franja])    
            else:
                _generalVarResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaFranjaVarF'][_franja])
                _generalBestResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaFranjaResultF'][_franja])    
                _generalBestResultP[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaFranjaResultP'][_franja])  
            _z += 1    

        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
        x = np.arange(len(_listBiologico))
        width = 1.0/(_n+1)
        
        fig, ax = plt.subplots()
        _listRect = list()
        
        x_color = 0
        for result in range(len(_generalBestResult)):
            _listRect.append(ax.bar(np.arange(len(_lista_año))+result*width, np.array(_generalBestResult[result]), width, label=_listBiologico[result],color=_colors[x_color]))
            x_color += 1
            
        #ax.set_title('Análisis estatal: Resumen general por componente y estacionalidad Franja '+str(_franja))
        ax.set_xticks(x)
        ax.set_xticklabels(list(_lista_año),ha='left')
        _title2 =''
        if _opc == 1:
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
            ax.set_ylabel('Puntuación Z')
            _title2 = _title+' (F'+str(_franja)+') (Boruta)'
        else:
            ax.yaxis.set_ticks(np.arange(0, 220, 20))
            ax.set_ylabel('Valor (F)')
            _title2 = _title+' (F'+str(_franja)+') (Anova)'
        ax.xaxis.set_ticks(_X,tuple(_lista_año))
        ax.set_xlabel('Año')
        ax.legend()
    
        f = 0
        for franja in list(_listRect):  
            if _opc == 2:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResultP)
            else:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
            f = f+1
        
        _ruta2 = _ruta+'/'
        if _opc == 1:
            _ruta2 = _ruta+'Boruta/'
        else:
            _ruta2 = _ruta+'Anova/'
        crearRuta(_ruta2) 
        
        fig.tight_layout()
        plt.savefig(str(_ruta2+_title2+' (F'+str(_franja)+')'), bbox_inches='tight')
        plt.show()
    

def graficarEpocaAño(generalBestResult,generalVarResult,_lista, _componente, _ruta,_title,_opc,generalBestResultP,_colors): 
    _generalVarResult = {}
    _generalBestResult = {}
    _generalBestResultP = {}
    
    _n = len(generalBestResult)
    _X = np.arange(1,_n,1)
    _intervalo = int(_n/len(lista_franja))

    if _opc == 1:
        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
    else:
        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
    x = np.arange(len(_lista))
    
    def separarByFranja(_intervalo,_n,_franja):
        _listAux = {}
        _generalVarResult = {}
        _generalBestResult = {}
        _generalBestResultP = {}
        
        if(_franja == 1):
            _n = _intervalo
            _intervalo = 0
        
        _z = 0
        for _i in range (_intervalo,_n):
            _generalBestResult[_z] = generalBestResult[_i]
            _generalVarResult[_z] = generalVarResult[_i]
            
            if _opc == 2:
                _generalBestResultP[_z] = generalBestResultP[_i]
                
            _z += 1  
        
        _listAux['p'] = _generalBestResultP
        _listAux['resultado'] = _generalBestResult
        _listAux['var'] = _generalVarResult
        
        return _listAux
    
    for _franja in list(lista_franja):
        fig, ax = plt.subplots()
        _listAux = separarByFranja(_intervalo,_n,_franja)
        _generalBestResult = _listAux['resultado']
        _generalVarResult = _listAux['var']
        _generalBestResultP = _listAux['p']
        _listRect = list()
        _n2 = len(_generalBestResult)
        width = 1.0/(_n2+0.5)
        
        x_color = 0
        for result in range(len(_generalBestResult)):
            _listRect.append(ax.bar(np.arange(len(_lista))+result*width, np.array(_generalBestResult[result]), width, label=lista_epoca[result],color=_colors[x_color]))
            x_color += 1
            
        #ax.set_title('Análisis estatal: Resumen general por estacionalidad Franja '+str(_franja)+' ('+_componente+')')
        ax.set_xticks(x)
        ax.set_xticklabels(list(_lista),ha='left')
        _title2 =''
        if _opc == 1:
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
            ax.set_ylabel('Puntuación Z')
            _title2 = _title+'(F'+str(_franja)+') (Boruta)'
        else:
            ax.yaxis.set_ticks(np.arange(0, 220, 20))
            ax.set_ylabel('Valor (F)')
            _title2 = _title+'(F'+str(_franja)+') (Anova)'
        ax.xaxis.set_ticks(_X,tuple(_lista))
        ax.set_xlabel('Año')
        
        ax.legend()
    
        f = 0
        for franja in list(_listRect):  
            if _opc == 2:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResultP)
            else:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
            f = f+1
            
        _ruta2 = _ruta+'/'
        if _opc == 1:
            _ruta2 = _ruta+'Boruta/'+_componente.replace('.','')+'/'
        else:
            _ruta2 = _ruta+'Anova/'+_componente.replace('.','')+'/'
        crearRuta(_ruta2)     
            
        fig.tight_layout()
        plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
        plt.show()
   
def graficarGeneralAño(_listResultadoGeneral,_listBiologico,_lista, _ruta,_title,_franja,_opc,_colors): 
    _generalVarResult = {}
    _generalBestResult = {}
    _generalBestResultP = {}
    _z = 0    
    
    for _biologico in list(_listBiologico):
        if _opc == 1:
            _generalBestResult[_z] = _listResultadoGeneral[_biologico]['listaFranjaResult'][_franja]
            _generalVarResult[_z] = _listResultadoGeneral[_biologico]['listaFranjaVar'][_franja]
        else:
            _generalVarResult[_z] = _listResultadoGeneral[_biologico]['listaFranjaVarF'][_franja]
            _generalBestResult[_z] = _listResultadoGeneral[_biologico]['listaFranjaResultF'][_franja]
            _generalBestResultP[_z] = _listResultadoGeneral[_biologico]['listaFranjaResultP'][_franja]
        _z += 1    
    
    _n = len(_generalBestResult)
    _X = np.arange(1,_n,1)

    if _opc == 1:
        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
    else:
        plt.rcParams['figure.figsize']=(18,10)
        plt.rcParams.update({'font.size': 20})
    x = np.arange(len(_lista))
    width = 1.0/(_n+1)
    
    fig, ax = plt.subplots()
    _listRect = list()
    
    x_color = 0
    for result in range(len(_generalBestResult)):
        _listRect.append(ax.bar(np.arange(len(_lista))+result*width, np.array(_generalBestResult[result]), width, label=_listBiologico[result],color=_colors[x_color]))
        x_color += 1
        
    #ax.set_title('Análisis estatal: Resumen general por componente Franja '+str(_franja))
    ax.set_xticks(x)
    ax.set_xticklabels(list(_lista),ha='left')
    _title2 =''
    if _opc == 1:
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
        ax.set_ylabel('Puntuación Z')
        _title2 = _title+'(F'+str(_franja)+') (Boruta)'
    else:
        ax.yaxis.set_ticks(np.arange(0, 220, 20))
        ax.set_ylabel('Valor (F)')
        _title2 = _title+'(F'+str(_franja)+') (Anova)'
    ax.xaxis.set_ticks(_X,tuple(_lista))
    ax.set_xlabel('Año')
    
    ax.legend()

    f = 0
    for franja in list(_listRect):  
        if _opc == 2:
            autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResultP)
        else:
            autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
        f = f+1
    
    _ruta2 = _ruta+'/'
    if _opc == 1:
        _ruta2 = _ruta+'Boruta/'
    else:
        _ruta2 = _ruta+'Anova/'
    crearRuta(_ruta2) 
    
    fig.tight_layout()
    plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
    plt.show()

def graficarGeneral(_listResultadoGeneral,_listBiologico, _ruta,_title,_opc,_colors): 
    _generalVarResult = {}
    _generalBestResult = {}
    _generalBestResultP = {}
    _z = 0
    
    for _franja in list(lista_franja):
        _auxResult = list()
        _auxResultP = list()
        _auxVar = list()
        for _biologico in list(_listBiologico):
            if _opc ==1:
                _auxResult.append(_listResultadoGeneral[_biologico]['listaGeneralResult'][_franja-1])
                _auxVar.append(_listResultadoGeneral[_biologico]['listaGeneralVar'][_franja-1])
            else: 
                _auxResult.append(_listResultadoGeneral[_biologico]['listaGeneralResultF'][_franja-1])
                _auxVar.append(_listResultadoGeneral[_biologico]['listaGeneralVarF'][_franja-1])
                _auxResultP.append(_listResultadoGeneral[_biologico]['listaGeneralResultP'][_franja-1])
            
        _generalVarResult[_z] = _auxVar
        _generalBestResult[_z] = _auxResult
        _generalBestResultP[_z] = _auxResultP
        _z += 1    
    
    _n = len(_listBiologico)
    _X = np.arange(1,_n,1)

    plt.rcParams['figure.figsize']=(15,10)
    plt.rcParams.update({'font.size': 20})
    x = np.arange(len(_listBiologico))
    width = 1.0/(len(_generalBestResult)+0.5)
    fig, ax = plt.subplots()
    _listRect = list()
    
    x_color = 0
    for result in range(len(_generalBestResult)):
        _listRect.append(ax.bar(np.arange(len(_listBiologico))+result*width, np.array(_generalBestResult[result]), width, label='Franja '+str(result+1),color=_colors[x_color]))
        x_color += 1
        
    #ax.set_title('Análisis estatal: Resumen general por Franja')
    ax.set_xticks(x)
    ax.set_xticklabels(list(_listBiologico),ha='center')
    _title2 =''
    if _opc == 1:
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
        ax.set_ylabel('Puntuación Z')
        _title2 = _title+'(Boruta)'
    else:
        ax.yaxis.set_ticks(np.arange(0, 220, 20))
        ax.set_ylabel('Valor (F)')
        _title2 = _title+'(Anova)'
    ax.xaxis.set_ticks(_X,tuple(_listBiologico))
    ax.set_xlabel('Parámetro biológico')
    if (len(lista_franja)>1):
        ax.legend()

    f = 0
    for franja in list(_listRect):  
        if _opc == 2:
            autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResultP)
        else:
            autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
        f = f+1
    
    _ruta2 = _ruta+'/'
    if _opc == 1:
        _ruta2 = _ruta+'Boruta/'
    else:
        _ruta2 = _ruta+'Anova/'
    crearRuta(_ruta2) 
    
    fig.tight_layout()
    plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
    plt.show()
    
def graficarGeneralEstacionalidad(_listResultadoGeneral,_listBiologico,_listaEpoca, _ruta,_title,_opc,_colors): 
    _generalVarResult = {}
    _generalBestResult = {}
    _generalBestResultP = {}
    _n = len(_listBiologico)
    _X = np.arange(1,_n,1)    
    
    def separarByFranja(_franja,_lista):
        _listAux = list()
        _n = len(_lista)
        _intervalo = int(_n/len(lista_franja))
        if(_franja == 1):
            _n = _intervalo
            _intervalo = 0
        
        for _i in range (_intervalo,_n):
            _listAux.append(_lista[_i])
            
        return _listAux
        
    def separarByEstacion(_listaAux,_listaEpoca):
        _listResult = {}
        for _i in range(len(_listaEpoca)):
            _aux = list()
            for _j in range(len(_listaAux)):
                _aux.append(_listaAux[_j][_i])
            _listResult[_i] = _aux
        return _listResult    
    
    for _franja in list(lista_franja):
        _z = 0
        for _biologico in list(_listBiologico):        
            if _opc == 1:
                _generalVarResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaGeneralEVar'])
                _generalBestResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaGeneralEResult'])    
            else:
                _generalVarResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaGeneralEVarF'])
                _generalBestResult[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaGeneralEResultF'])    
                _generalBestResultP[_z] = separarByFranja(_franja,_listResultadoGeneral[_biologico]['listaGeneralEResultP'])  
            _z += 1    
        
        _generalVarResult = separarByEstacion(_generalVarResult,_listaEpoca)
        _generalBestResult = separarByEstacion(_generalBestResult,_listaEpoca)
        
        if _opc == 2:
            _generalBestResultP = separarByEstacion(_generalBestResultP,_listaEpoca)
        
        plt.rcParams['figure.figsize']=(15,10)
        plt.rcParams.update({'font.size': 20})
        x = np.arange(len(_listBiologico))
        width = 1.0/(len(_generalBestResult)+0.5)
        
        fig, ax = plt.subplots()
        _listRect = list()
        
        x_color = 0
        for result in range(len(_generalBestResult)):
            _listRect.append(ax.bar(np.arange(len(_listBiologico))+result*width, np.array(_generalBestResult[result]), width, label=_listaEpoca[result],color=_colors[x_color]))
            x_color += 1
            
        #ax.set_title('Análisis estatal: Resumen general por componente y estacionalidad Franja '+str(_franja))
        ax.set_xticks(x)
        ax.set_xticklabels(list(_listBiologico),ha='left')
        _title2 =''
        if _opc == 1:
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1)) 
            ax.set_ylabel('Puntuación Z')
            _title2 = _title+' (F'+str(_franja)+') (Boruta)'
        else:
            ax.yaxis.set_ticks(np.arange(0, 220, 20))
            ax.set_ylabel('Valor (F)')
            _title2 = _title+' (F'+str(_franja)+') (Anova)'
        ax.xaxis.set_ticks(_X,tuple(_listBiologico))
        ax.set_xlabel('Parámetro biológico')
        ax.legend()
    
        f = 0
        for franja in list(_listRect):  
            if _opc == 2:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResultP)
            else:
                autolabel(franja,f,_opc,ax,_generalVarResult,_generalBestResult)
            f = f+1
        
        _ruta2 = _ruta+'/'
        if _opc == 1:
            _ruta2 = _ruta+'Boruta/'
        else:
            _ruta2 = _ruta+'Anova/'
        crearRuta(_ruta2) 
        
        fig.tight_layout()
        plt.savefig(str(_ruta2+_title2), bbox_inches='tight')
        plt.show()
    
def generarGrafica(_dataset,_componente,_ruta,_listaFranjaResult,_listaFranjaVar, _opc,_opc2,_generalBestResultP,_colors):
    lista_año = _dataset['Año'].drop_duplicates()
    _ruta2 = _ruta +'Resumen estatal/'
    try:
        os.makedirs(str(_ruta2))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if _opc == 1:
        _title = 'Resumen estatal por año ('+ _componente.replace('.','') +')'
        graficar(_listaFranjaResult,_listaFranjaVar,lista_año,_componente,_ruta2,_title,_opc2,_generalBestResultP,_colors)
    elif _opc == 2:
        _title = 'Resumen estatal general'
        graficarGeneral(_listaFranjaResult,_listaFranjaVar,_ruta2,_title,_opc2,_colors)
    elif _opc == 3:
        _title = 'Resumen estatal anual-temporalidad ('+ _componente.replace('.','') +')'
        graficarEpocaAño(_listaFranjaResult,_listaFranjaVar,lista_año,_componente,_ruta2,_title,_opc2,_generalBestResultP,_colors)        
    elif _opc == 4:
        for _franja in list(lista_franja):
            _title = 'Resumen estatal por componenete franja '+str(_franja)
            graficarGeneralAño(_listaFranjaResult,_listaFranjaVar,lista_año,_ruta2,_title,_franja,_opc2,_colors)
    elif _opc == 5:
        _title = 'Resumen estatal por componente y estacionalidad'
        graficarGeneralEstacionalidad(_listaFranjaResult,_listaFranjaVar,_componente,_ruta2,_title,_opc2,_colors)
    elif _opc == 6:
        _title = 'Resumen estatal por componente y año'
        graficarResumen(_listaFranjaResult,_listaFranjaVar,lista_año,_componente,_ruta2,_title,_opc2,_generalBestResultP,_colors)
            

def obtenerResultadoPorEspecie(_specie,listaBiologico,_dataset):
    listResultadoGeneral = {}
    for _aux in list(listaBiologico):
        _listaAux = listaBiologico.copy()
        _listaAux.remove(_aux)
        
        _dataAux = _dataset.copy()
        for _delete in list(_listaAux):
            _dataAux = _dataAux.drop([_delete], axis=1)
            
        _title = 'Boruta resultados '+_specie+'/Graficas/Gráfica '+_aux.replace('.','')
        listResultadoGeneral[_aux] = getResultadoGeneral(_aux,_title,_dataAux, 0)
    return listResultadoGeneral


def graficarResultados(_specie,listaBiologico,_dataset,_colors,listResultadoGeneral):
    for _aux in list(listaBiologico):
        _listaAux = listaBiologico.copy()
        _listaAux.remove(_aux)
        
        _dataAux = _dataset.copy()
        for _delete in list(_listaAux):
            _dataAux = _dataAux.drop([_delete], axis=1)
            
        _ruta = 'Boruta resultados '+_specie+'/Graficas/Resultado General/'
        generarGrafica(_dataAux,_aux,_ruta,listResultadoGeneral[_aux]['listaFranjaResult'],listResultadoGeneral[_aux]['listaFranjaVar'],1,1,'',_colors)
        generarGrafica(_dataAux,_aux,_ruta,listResultadoGeneral[_aux]['listaEpocaResult'],listResultadoGeneral[_aux]['listaEpocaVar'],3,1,'',_colors)
        
        generarGrafica(_dataAux,_aux,_ruta,listResultadoGeneral[_aux]['listaFranjaResultF'],listResultadoGeneral[_aux]['listaFranjaVarF'],1,2,listResultadoGeneral[_aux]['listaFranjaResultP'],_colors)
        generarGrafica(_dataAux,_aux,_ruta,listResultadoGeneral[_aux]['listaEpocaResultF'],listResultadoGeneral[_aux]['listaEpocaVarF'],3,2,listResultadoGeneral[_aux]['listaEpocaResultP'],_colors)
        
    generarGrafica(_dataset,_aux,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,4,1,'',_colors)
    generarGrafica(_dataset,'','Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,2,1,'',_colors)
    generarGrafica(_dataset,lista_epoca,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,5,1,'',_colors)
    generarGrafica(_dataset,lista_epoca,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,6,1,'',_colors)
    
    generarGrafica(_dataset,_aux,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,4,2,'',_colors)
    generarGrafica(_dataset,_aux,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,2,2,'',_colors)
    generarGrafica(_dataset,lista_epoca,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,5,2,'',_colors)
    generarGrafica(_dataset,lista_epoca,'Boruta resultados '+_specie+'/Graficas/Resultado General/',listResultadoGeneral,listaBiologico,6,2,'',_colors)
    

def exportarResultadosExcel(rutaResultados,listaBiologico,listaResultado,lista_franja,lista_epoca,_dataset,_specie):
    for biologico in list(listaBiologico):
        _ruta = rutaResultados+'Export excel '+_specie+'/Boruta/Archivos/estatal/'+biologico+'/'
        crearRuta(_ruta)
        writer = pd.ExcelWriter(_ruta+'resultados nivel estatal.xlsx')
        for franja in list(lista_franja):
            _dataAux = _dataset.loc[_dataset.loc[:, 'Franja'] == franja]
            _aux = listaResultado[biologico]['general'+str(franja)]['franja'+str(franja)]
            _aux.to_excel(writer, sheet_name='franja '+str(franja), index=False,header=True)
                
            _lista_año = _dataAux['Año'].drop_duplicates()

            writerAño = pd.ExcelWriter(_ruta+'resultados por año franja '+str(franja)+'.xlsx')
            for año in list(_lista_año):
                _añoRegion = listaResultado[biologico]['listaAñoF'+str(franja)]['año'+str(año)]['año'+str(año)]
                if _añoRegion.empty != True:
                    _añoRegion.to_excel(writerAño, sheet_name=str(año)+'F'+str(franja), index=False,header=True)
            writerAño.save()
            writerAño.close()
            
            writerAñoEpoca = pd.ExcelWriter(_ruta+'resultados anuales por epoca F'+str(franja)+'.xlsx')
            writerEpoca = pd.ExcelWriter(_ruta+'resultados por epoca F'+str(franja)+'.xlsx')
            for epoca in list(lista_epoca):
                _epocaR = listaResultado[biologico]['listaEpocaF'+str(franja)][epoca]['epoca'+epoca]
                if _epocaR.empty != True:
                    _epocaR.to_excel(writerEpoca, sheet_name=str(epoca)+'F'+str(franja), index=False,header=True)
                
                _dataAux2 = _dataAux.loc[_dataAux.loc[:, 'Epoca del año'] == epoca]
                _lista_año = _dataAux2['Año'].drop_duplicates()
                
                for año in list(_lista_año):
                    _añoEpoca = listaResultado[biologico]['listaEpocaF'+str(franja)][epoca+':año']['año'+str(año)]['año'+str(año)]
                    if _añoEpoca.empty != True:
                        _añoEpoca.to_excel(writerAñoEpoca, sheet_name=str(epoca)+str(año)+'F'+str(franja), index=False,header=True)
                    
                writerEpoca.save()
                writerEpoca.close()
            writerAñoEpoca.save()
            writerAñoEpoca.close()   
        writer.save()
        writer.close()

def exportarExcelResumen(rutaResultados,listaBiologico,listaResultado,lista_franja,lista_epoca,_dataset,_specie):
    for biologico in list(listaBiologico):
        _ruta = rutaResultados+'Export excel '+_specie+'/Boruta/Resumen/estatal/'+biologico+'/'
        crearRuta(_ruta)
        writer = pd.ExcelWriter(_ruta+'resultado general.xlsx')        
        _resultadoGeneral = pd.DataFrame()
        _resultadoGeneral['Franja'] = lista_franja
        _resultadoGeneral['Var_Boruta'] = listaResultado[biologico]['listaGeneralVar']
        _resultadoGeneral['Importancia'] = listaResultado[biologico]['listaGeneralResult']
        _resultadoGeneral['Var_F'] = listaResultado[biologico]['listaGeneralVarF']
        _resultadoGeneral['F'] = listaResultado[biologico]['listaGeneralResultF']
        _resultadoGeneral['P'] = listaResultado[biologico]['listaGeneralResultP']
        _resultadoGeneral.to_excel(writer, sheet_name='Resumen general', index=False,header=True)
        writer.save()
        writer.close()
        
        writerAño = pd.ExcelWriter(_ruta+'resultados por año.xlsx')        
        writerEpoca = pd.ExcelWriter(_ruta+'resultados por epoca general.xlsx')
        crearRuta(_ruta+'Epoca/')
        for franja in list(lista_franja):            
            bestResult = list()
            bestVar = list()
            fresult = list()
            fvar = list()
            presult = list()
            
            _resultadoEpoca = pd.DataFrame()
            _len = len(listaResultado[biologico]['listaGeneralEResult'])
            
            if franja == 1:
                _interval = 0
                _len = int(_len/len(lista_franja))
            else:
                _interval = int(_len/len(lista_franja))
            
            
            for _i in range(_interval,_len):
                bestResult.append(listaResultado[biologico]['listaGeneralEResult'][_i])
                bestVar.append(listaResultado[biologico]['listaGeneralEVar'][_i])
                fresult.append(listaResultado[biologico]['listaGeneralEResultF'][_i])
                fvar.append(listaResultado[biologico]['listaGeneralEVarF'][_i])
                presult.append(listaResultado[biologico]['listaGeneralEResultP'][_i])    
            _resultadoEpoca['epoca'] = lista_epoca
            _resultadoEpoca['Var_Boruta'] = bestVar
            _resultadoEpoca['Importancia'] = bestResult
            _resultadoEpoca['Var_F'] = fvar
            _resultadoEpoca['F'] = fresult
            _resultadoEpoca['P'] = presult
            _resultadoEpoca.to_excel(writerEpoca, sheet_name='Franja '+str(franja), index=False,header=True)
            
            _resultadosAño = pd.DataFrame()
            _resultadosAño['Año'] = lista_año
            _resultadosAño['Var_Boruta'] = listaResultado[biologico]['listaFranjaVar'][franja]
            _resultadosAño['Importancia'] = listaResultado[biologico]['listaFranjaResult'][franja]
            _resultadosAño['Var_F'] = listaResultado[biologico]['listaFranjaVarF'][franja]
            _resultadosAño['F'] = listaResultado[biologico]['listaFranjaResultF'][franja]
            _resultadosAño['P'] = listaResultado[biologico]['listaFranjaVar'][franja]
            _resultadosAño.to_excel(writerAño, sheet_name='AñoF'+str(franja), index=False,header=True)
                 
            _x = 0
            for _i in range(_interval,_len):
                _resultadosEpocaAño = pd.DataFrame()
                writerEpocaAño = pd.ExcelWriter(_ruta+'Epoca/resultados por epoca '+lista_epoca[_x]+' F'+str(franja)+'.xlsx')
                _resultadosEpocaAño['Año'] = lista_año
                _resultadosEpocaAño['Var_Boruta'] = listaResultado[biologico]['listaEpocaVar'][_i]
                _resultadosEpocaAño['Importancia'] = listaResultado[biologico]['listaEpocaResult'][_i]
                _resultadosEpocaAño['Var_F'] = listaResultado[biologico]['listaEpocaVarF'][_i]
                _resultadosEpocaAño['F'] = listaResultado[biologico]['listaEpocaResultF'][_i]
                _resultadosEpocaAño['P'] = listaResultado[biologico]['listaEpocaResultP'][_i] 
                _resultadosEpocaAño.to_excel(writerEpocaAño, sheet_name=lista_epoca[_x]+'AñoF'+str(franja), index=False,header=True)
                writerEpocaAño.save()
                writerEpocaAño.close()    
                _x += 1
            
        writerAño.save()
        writerAño.close()
        writerEpoca.save()
        writerEpoca.close()

cargarDatos('boruta/datasetX.csv')
plt.figure (figsize = (15,5))
plt.rcParams['figure.figsize']=(15,10)
sns.set_style("white")      
_specie = 'Rm'
colors =  ['blue', 'darkorange', 'green', 'red','yellow', 'cyan', 'gray']

listResultadoGeneral = obtenerResultadoPorEspecie(_specie,listaBiologico,dataset)
graficarResultados(_specie,listaBiologico,dataset,colors,listResultadoGeneral)
exportarResultadosExcel('Boruta resultados '+_specie+'/',listaBiologico,listResultadoGeneral,lista_franja,lista_epoca,dataset,_specie)       
exportarExcelResumen('Boruta resultados '+_specie+'/',listaBiologico,listResultadoGeneral,lista_franja,lista_epoca,dataset,_specie)      