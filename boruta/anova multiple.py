# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:04:09 2021

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
import dataframe_image as dfi
from bioinfokit.analys import stat
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats


### make X and y
dataset = pd.read_csv('boruta/datasetX.csv', sep=",", header=0)
lista_region = dataset['Region'].drop_duplicates()
lista_franja = dataset['Franja'].drop_duplicates()
lista_epoca = dataset['Epoca del año'].drop_duplicates()
lista_año = dataset['Año'].drop_duplicates()
lista_sitio = dataset['Sitio'].drop_duplicates()

listaParametro = list()
listaParametro.append('S')
listaParametro.append('ORP')
listaParametro.append('P')
listaParametro.append('pH')
listaParametro.append('T')

listaBiologico = list()
listaBiologico.append('Propágulos')
listaBiologico.append('Hojas')
listaBiologico.append('Flores')
listaBiologico.append('Hojarasca')

listaTiempo = list()
listaTiempo.append('Mes')
listaTiempo.append('Año')

listaLugar = list()
listaLugar.append('Region')
listaLugar.append('Sitio')

def getData(_data1):
    data = pd.DataFrame()
    data = _data1.copy()
    data = data.fillna(0)    
    return data

def crearRuta(_ruta):
    try:
        os.makedirs(str(_ruta))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def getLinReg(biologico,listaParametro,_dataset,_franja,ruta,rutaResultados,_title):
    _anovaSitioF = pd.DataFrame()
#    crearRuta(rutaResultados)
    #writer = pd.ExcelWriter(rutaResultados+_title+'.xlsx')
    listaR = {}
    for parametro in list(listaParametro):       
       reg = sm.OLS(_dataset[biologico], _dataset[parametro]).fit()
       reg.summary()
       listaR[parametro] =  reg.summary()
    return listaR   
       _anova = res.anova_summary
       _anova.to_excel(writer, sheet_name=parametro, index=True,header=True)
       _anovaSitioF['df'] = _anova.loc[:,'df']      
       _anovaSitioF[parametro+'_F'] = _anova.loc[:,'F']
       _anovaSitioF[parametro+'_P'] = _anova.loc[:,'PR(>F)']
       _anovaSitioF = _anovaSitioF.round(5)       
    _anovaSitioF['Factor'] = _anova.index
#    for parametro in list(listaParametro):  
#        _anovaSitioF.loc[_anovaSitioF[parametro+'_P'] < 0.0001,parametro+'_P'] = 0.00001
#    _anovaSitioF.set_index('Factor',inplace=True)
#    _anovaSitioF.style.set_caption('Variables (Valor F)')
#    ruta += '/ANOVA/'+lugar+'/'
#    crearRuta(ruta)
#    dfi.export(_anovaSitioF, ruta+'Anova '+lugar+' - '+tiempo+' (F'+str(_franja)+').png') 
#    
#    writer.save()
#    writer.close()
    
#    return _anovaSitioF

def getAnova(tiempo,lugar,listaParametro,_dataset,_franja,ruta,rutaResultados,_title):
    _anovaSitioF = pd.DataFrame()
    crearRuta(rutaResultados)
    writer = pd.ExcelWriter(rutaResultados+_title+'.xlsx')
    for parametro in list(listaParametro):       
       res = stat()
       res.anova_stat(df=_dataset, res_var=parametro, anova_model=parametro+'~C('+lugar+')+C('+tiempo+')+C(Epoca)+C('+lugar+'):C('+tiempo+')+C('+lugar+'):C(Epoca)+C('+tiempo+'):C(Epoca)+C('+lugar+'):C('+tiempo+'):C(Epoca)')
       _anova = res.anova_summary
       _anova.to_excel(writer, sheet_name=parametro, index=True,header=True)
       _anovaSitioF['df'] = _anova.loc[:,'df']      
       _anovaSitioF[parametro+'_F'] = _anova.loc[:,'F']
       _anovaSitioF[parametro+'_P'] = _anova.loc[:,'PR(>F)']
       _anovaSitioF = _anovaSitioF.round(5)       
    _anovaSitioF['Factor'] = _anova.index
    for parametro in list(listaParametro):  
        _anovaSitioF.loc[_anovaSitioF[parametro+'_P'] < 0.0001,parametro+'_P'] = 0.00001
    _anovaSitioF.set_index('Factor',inplace=True)
    _anovaSitioF.style.set_caption('Variables (Valor F)')
    ruta += '/ANOVA/'+lugar+'/'
    crearRuta(ruta)
    dfi.export(_anovaSitioF, ruta+'Anova '+lugar+' - '+tiempo+' (F'+str(_franja)+').png') 
    
    writer.save()
    writer.close()
    
    return _anovaSitioF

def getTukey(tiempo,lugar,listaParametro,d_melt,_franja,ruta,rutaResultados,_title):
    _tukeyF = pd.DataFrame()    
    _tukeyFTiempo = pd.DataFrame()    
    _tukeyCombinadoF = pd.DataFrame()
    
    crearRuta(rutaResultados)
    writer = pd.ExcelWriter(rutaResultados+_title+' lugar.xlsx')
    writer2 = pd.ExcelWriter(rutaResultados+_title+' tiempo.xlsx')
    writer3 = pd.ExcelWriter(rutaResultados+_title+' combinado.xlsx')
    for parametro in list(listaParametro):  
       res = stat()
       res.tukey_hsd(df=d_melt, res_var=parametro, xfac_var=lugar, anova_model=parametro+'~C('+lugar+')+C('+tiempo+')+C('+lugar+'):C('+tiempo+')')
       _anova = res.tukey_summary       
       _tukeyF['Grupo'] = _anova.loc[:,'group1'].astype(str)+'-'+_anova.loc[:,'group2'].astype(str) 
       _tukeyF[parametro+'_diff'] = _anova.loc[:,'Diff']       
       _tukeyF[parametro+'_P'] = _anova.loc[:,'p-value']
       _tukeyF = _tukeyF.round(4)
       _anova.to_excel(writer, sheet_name=parametro, index=True,header=True)
       
       res.tukey_hsd(df=d_melt, res_var=parametro, xfac_var=tiempo, anova_model=parametro+'~C('+lugar+')+C('+tiempo+')+C('+lugar+'):C('+tiempo+')')
       _anova = res.tukey_summary
       _tukeyFTiempo['Grupo'] = _anova.loc[:,'group1'].astype(str)+'-'+_anova.loc[:,'group2'].astype(str)
       _tukeyFTiempo[parametro+'_diff'] = _anova.loc[:,'Diff']       
       _tukeyFTiempo[parametro+'_P'] = _anova.loc[:,'p-value']
       _tukeyFTiempo = _tukeyFTiempo.round(4)
       _anova.to_excel(writer2, sheet_name=parametro, index=True,header=True)
       
       res.tukey_hsd(df=d_melt, res_var=parametro, xfac_var=[lugar,tiempo], anova_model=parametro+'~C('+lugar+')+C('+tiempo+')+C('+lugar+'):C('+tiempo+')')
       _anova = res.tukey_summary
       _tukeyCombinadoF['Grupo'] = _anova.loc[:,'group1'].astype(str)+'-'+_anova.loc[:,'group2'].astype(str) 
       _tukeyCombinadoF[parametro+'_diff'] = _anova.loc[:,'Diff']       
       _tukeyCombinadoF[parametro+'_P'] = _anova.loc[:,'p-value']
       _tukeyCombinadoF = _tukeyCombinadoF.round(4)
       _anova.to_excel(writer3, sheet_name=parametro, index=True,header=True)
       
    _tukeyF = pd.concat([_tukeyF,_tukeyFTiempo])    
    _tukeyF.set_index('Grupo',inplace=True)
    for parametro in list(listaParametro):  
        _tukeyF.loc[_tukeyF[parametro+'_P'] < 0.0001,parametro+'_P'] = 0.00001    
    _tukeyCombinadoF.set_index('Grupo',inplace=True)
    
    ruta += '/Tukey/'+lugar+'/'
    crearRuta(ruta)
    dfi.export(_tukeyF, ruta+'Tukey-HDSF '+lugar+' - '+tiempo+' (F'+str(_franja)+').png')    
            
    writer.save()
    writer.close()
    writer3.save()
    writer3.close()
    writer2.save()
    writer2.close()
    
    _lista = {}
    _lista['tukeyF'] = _tukeyF    
    _lista['tukeyCombinadoF'] = _tukeyCombinadoF
    return _lista

def getLevene(tiempo,lugar,listaParametro,_dataset,_franja,ruta):
    _anovaSitioF = pd.DataFrame()    
    for parametro in list(listaParametro):       
       res = stat()
       res.levene(df=_dataset, res_var=parametro, xfac_var=[lugar, tiempo ,'Epoca'])
       _anova = res.levene_summary  
       _anovaSitioF['Parámetro'] = _anova['Parameter']
       _anovaSitioF[parametro] = _anova['Value']
       _anovaSitioF = _anovaSitioF.round(4)       
    ruta += '/Levene/'+lugar+'/'
    crearRuta(ruta)
    dfi.export(_anovaSitioF, ruta+'Levene '+lugar+' - '+tiempo+' (F'+str(_franja)+').png')    
    return _anovaSitioF

def getBartlett(tiempo,lugar,listaParametro,_dataset,_franja,ruta):
    _anovaSitioF = pd.DataFrame()    
    for parametro in list(listaParametro):       
       res = stat()
       res.bartlett(df=_dataset, res_var=parametro, xfac_var=[lugar, tiempo ,'Epoca'])
       _anova = res.bartlett_summary  
       _anovaSitioF['Parámetro'] = _anova['Parameter']
       _anovaSitioF[parametro] = _anova['Value']
       _anovaSitioF = _anovaSitioF.round(4)       
    ruta += '/Bartlett/'+lugar+'/'
    crearRuta(ruta)
    dfi.export(_anovaSitioF, ruta+'Bartlett '+lugar+' - '+tiempo+' (F'+str(_franja)+').png')     
    return _anovaSitioF

def getShapiro(tiempo,lugar,listaParametro,_dataset,_franja,ruta):
    _anovaSitioF = pd.DataFrame()
    a = list()
    a.append('w')
    a.append('p')
    _anovaSitioF['Parámetro'] = a 
    for parametro in list(listaParametro):       
       res = stat()
       res.anova_stat(df=_dataset, res_var=parametro, anova_model=parametro+'~C('+lugar+')+C('+tiempo+')+C(Epoca)+C('+lugar+'):C('+tiempo+')+C('+lugar+'):C(Epoca)+C('+tiempo+'):C(Epoca)+C('+lugar+'):C('+tiempo+'):C(Epoca)')
       w, pvalue = stats.shapiro(res.anova_model_out.resid)
       a = list()
       a.append(w)
       a.append(pvalue)
       _anovaSitioF[parametro] = a 
       _anovaSitioF = _anovaSitioF.round(2)       
    ruta += '/Shapiro/'+lugar+'/'
    crearRuta(ruta)
    dfi.export(_anovaSitioF, ruta+'Shapiro '+lugar+' - '+tiempo+' (F'+str(_franja)+').png')     
    return _anovaSitioF

def getResultado(_dataset, lista_franja,listaTiempo,listaLugar,listaParametro,ruta,ruta2,_sp):
    _listaFranja = {}
    _listaAnova = {}
    _listaTukey = {}
    _listaTukeyCom = {}
    _listaShapiro = {}
    _listaLevene = {}
    _listaBartlett = {}
    for franja in lista_franja:   
       data = dataset.loc[dataset.loc[:, 'Franja'] == franja] 
       data = data.drop(['Franja'], axis=1)
       data = data.rename(columns={'Epoca del año':'Epoca'})   
       data = data.drop(data[data['Epoca']=='Norte'].index)
       data = getData(data)
       for lugar in list(listaLugar):
           for tiempo in list(listaTiempo):
               _ruta = _sp+'/Graficas/Resultado General/Prueba estadistica/'+ruta
               _listaAnova[lugar+tiempo+str(franja)] = getAnova(tiempo,lugar,listaParametro,data,franja,_ruta,ruta2,tiempo+lugar+str(franja)+'(Anova)')
               
               _listaAux = getTukey(tiempo,lugar,listaParametro,data,franja,_ruta,ruta2,tiempo+lugar+str(franja)+'(Tukey)')
               _listaTukeyCom[lugar+tiempo+str(franja)] = _listaAux['tukeyCombinadoF']
               _listaTukey[lugar+tiempo+str(franja)] = _listaAux['tukeyF']
               
               _listaShapiro[lugar+tiempo+str(franja)] = getShapiro(tiempo,lugar,listaParametro,data,franja,_ruta)
               _listaLevene[lugar+tiempo+str(franja)] = getLevene(tiempo,lugar,listaParametro,data,franja,_ruta)
               _listaBartlett[lugar+tiempo+str(franja)] = getBartlett(tiempo,lugar,listaParametro,data,franja,_ruta)
               
    _listaFranja['listaTukey'] = _listaTukey
    _listaFranja['listaTukeyCom'] = _listaTukeyCom
    _listaFranja['listaAnova'] = _listaAnova
    _listaFranja['listaShapiro'] = _listaShapiro
    _listaFranja['listaLevene'] = _listaLevene
    _listaFranja['listaBartlett'] = _listaBartlett
    return  _listaFranja

def exportToExcel(listaResultado,ruta,title,lista_franja,listaLugar,listaTiempo,llave,prueba,_index):
    crearRuta(ruta)
    writer = pd.ExcelWriter(ruta+title+'.xlsx')
    for franja in list(lista_franja):
        for lugar in list(listaLugar):
            for tiempo in list(listaTiempo): 
                _llave = lugar+tiempo+str(franja)
                listaResultado[llave][_llave].to_excel(writer, sheet_name=prueba+lugar+tiempo+"F"+str(franja), index=_index,header=True)
    writer.save()
    writer.close()

def exportarResultadoGeneral(_listaResultado,ruta,tipo):
    exportToExcel(_listaResultado,ruta,'Resultados Tukey por '+tipo,lista_franja,listaLugar,listaTiempo,'listaTukey','Tukey',True)
    exportToExcel(_listaResultado,ruta,'Resultados Tukey Combinado por '+tipo,lista_franja,listaLugar,listaTiempo,'listaTukey','Tukey',True)
    exportToExcel(_listaResultado,ruta,'Resultados Shapiro por '+tipo,lista_franja,listaLugar,listaTiempo,'listaShapiro','Shapiro',False)
    exportToExcel(_listaResultado,ruta,'Resultados Anova por '+tipo,lista_franja,listaLugar,listaTiempo,'listaAnova','Anova',True)
    exportToExcel(_listaResultado,ruta,'Resultados Levene por '+tipo,lista_franja,listaLugar,listaTiempo,'listaLevene','Levene',False)
    exportToExcel(_listaResultado,ruta,'Resultados Bartlett por '+tipo,lista_franja,listaLugar,listaTiempo,'listaBartlett','Bartlett',False)

specie = 'Boruta resultados Rm'
ruta = 'Boruta resultados Rm/Export excel Rm/Pruebas estadistica/Archivos/Parametros/'
listaResultadoParametro = getResultado(dataset,lista_franja,listaTiempo,listaLugar,listaParametro,'Parametros',ruta,specie)

ruta = 'Boruta resultados Rm/Export excel Rm/Pruebas estadistica/Archivos/Biologico/'
listaResultadoBiologico = getResultado(dataset,lista_franja,listaTiempo,listaLugar,listaBiologico,'Biologico',ruta,specie)

ruta = 'Boruta resultados Rm/Export excel Rm/Pruebas estadistica/Resumen/Parametros/'    
tipo = 'parámetros'
exportarResultadoGeneral(listaResultadoParametro,ruta,tipo)

ruta = 'Boruta resultados Rm/Export excel Rm/Pruebas estadistica/Resumen/Biologico/'    
tipo = 'biológico'
exportarResultadoGeneral(listaResultadoBiologico,ruta,tipo)

def ajustarPValue (_dataux, listaParametro):
    for parametro in list(listaParametro):    
        _dataux.loc[_dataux[parametro] < 0.0001,parametro] = 0.0001
    return _dataux    

import pingouin as pg
sps = 'Rm'     
sns.set_style("whitegrid")
sns.set(rc = {'figure.figsize':(10, 5)})
listAux = list()
listAux =  listaBiologico + listaParametro 
for franja in lista_franja:
   data = dataset.loc[dataset.loc[:, 'Franja'] == franja] 
   data = data.drop(['Franja'], axis=1)
   
   for region in lista_region:
       dataR = data.loc[data.loc[:, 'Region'] == region]
       dataR= dataR.drop(['Region'], axis=1)
       
       lista_sitio = dataR['Sitio'].drop_duplicates()               
       dataR = dataR.drop(['Mes'], axis=1)
       dataR = dataR.drop(['Año'], axis=1)
       dataR = dataR.drop(['Epoca del año'], axis=1)
       dataR = dataR.drop(['Sitio'], axis=1)
       _datosProcesados = getData(dataR)

       if(dataR.empty != True):         
           try:
                os.makedirs(str(sps)+'/F'+str(franja)+'/Región '+str(region))
           except OSError as e:
               if e.errno != errno.EEXIST:
                   raise
           aux = _datosProcesados[np.array(listAux)].rcorr(stars=False,method='pearson')
           aux = aux.replace('-', int(1))           
           aux = aux.astype('Float64')
           aux = ajustarPValue(aux,listAux)
           aux.round(4)
           plt.figure()
           sns.heatmap(_datosProcesados[np.array(listAux)].corr(method='pearson').round(2), cmap = 'Blues', annot = True).set_title('Matrix de Correlación Pearson R2 '+str(sps)+' Región '+str(region)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/Rm '+str(region)+' Franja '+str(franja)+'(Correlación Pearson R2)', bbox_inches='tight')
           
           plt.figure()
           sns.heatmap(aux, cmap = 'Blues', annot = True).set_title('Matrix de Correlación Pearson p-Value '+str(sps)+' Región '+str(region)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/Rm '+str(region)+' Franja '+str(franja)+'(Correlación Pearson p-Value)', bbox_inches='tight')
           
           aux = _datosProcesados[np.array(listAux)].rcorr(stars=False, method='spearman')
           aux = aux.replace('-', int(1))
           aux = aux.astype('Float64')
           aux = ajustarPValue(aux,listAux)
           aux.round(4)
           plt.figure()
           sns.heatmap(_datosProcesados[np.array(listAux)].corr(method='spearman').round(2), cmap = 'Blues', annot = True).set_title('Matrix de Correlación Spearman R2 '+str(sps)+' Región '+str(region)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/Rm '+str(region)+' Franja '+str(franja)+'(Correlación Spearman R2)', bbox_inches='tight')
           
           plt.figure()
           sns.heatmap(aux, cmap = 'Blues', annot = True).set_title('Matrix de Correlación Spearman p-Value '+str(sps)+' Región '+str(region)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/Rm '+str(region)+' Franja '+str(franja)+'(Correlación Spearman p-Value)', bbox_inches='tight')
           
           aux = _datosProcesados[np.array(listAux)].rcorr(stars=False)
           aux = aux.replace('-', int(1))
           aux = aux.astype('Float64')
           aux = ajustarPValue(aux,listAux)
           aux.round(4)
           plt.figure()
           sns.heatmap(_datosProcesados[np.array(listAux)].corr(method='kendall').round(2), cmap = 'Blues', annot = True).set_title('Matrix de Correlación Kendall R2 '+str(sps)+' Región '+str(region)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/Rm '+str(region)+' Franja '+str(franja)+'(Coeficiente Tau de Kendall T)', bbox_inches='tight')
           
                  
       for sitio in list(lista_sitio):
           _dataSitio = dataR.loc[dataR.loc[:, 'Sitio'] == sitio]
           _dataSitio = _dataSitio.drop(['Sitio'], axis=1)
           
           _datosProcesados = getDataMatriz(_dataSitio, var)
           y = _datosProcesados['y']
           X = _datosProcesados['X'] 
           
           try:
               os.makedirs(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/sitios')
           except OSError as e:
               if e.errno != errno.EEXIST:
                   raise
           y1 = np.array(_dataSitio.columns.tolist())
           X.columns=y1
           plt.figure()
           sns.heatmap(X.assign().corr().round(2), cmap = 'Blues', annot = True).set_title('Matrix de Correlación '+str(sps)+' Sitio '+str(sitio)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/sitios/Rm '+str(sitio)+' Franja '+str(franja), bbox_inches='tight')

       for epoca in list(lista_epoca):
           _dataEpoca = dataR.loc[dataR.loc[:, 'Epoca del año'] == epoca]
           _dataEpoca = _dataEpoca.drop(['Epoca del año'], axis=1)
           _datosProcesados = getDataMatriz(_dataEpoca, var)
           y = _datosProcesados['y']
           X = _datosProcesados['X']
           
           try:
               os.makedirs(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/epocas')
           except OSError as e:
               if e.errno != errno.EEXIST:
                   raise
           y1 = np.array(_dataEpoca.columns.tolist())
           X.columns=y1
           plt.figure()
           sns.heatmap(X.assign().corr().round(2), cmap = 'Blues', annot = True).set_title('Matrix de Correlación '+str(sps)+' Epoca de '+str(epoca)+' Franja '+str(franja), fontsize = 16)
           plt.savefig(str(sps)+'/F'+str(franja)+'/Región '+str(region)+'/epocas/Rm '+str(epoca)+' Franja '+str(franja), bbox_inches='tight')
