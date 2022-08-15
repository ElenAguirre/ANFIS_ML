#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import *
from lineartree import LinearTreeRegressor
import pprint
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")


#Data of Cross-Validation of ANFIS
CV_tr_1 = pd.read_excel("CrossV_tr1.xlsx", sheet_name='Sheet1')
CV_tr_2 = pd.read_excel("CrossV_tr2.xlsx", sheet_name='Sheet1')
CV_tr_3 = pd.read_excel("CrossV_tr3.xlsx", sheet_name='Sheet1')
CV_tr_4 = pd.read_excel("CrossV_tr4.xlsx", sheet_name='Sheet1')
CV_tr_5 = pd.read_excel("CrossV_tr5.xlsx", sheet_name='Sheet1')
#Test CV
CV_ts_1 = pd.read_excel("CrossV_ts1.xlsx", sheet_name='Sheet1')
CV_ts_2 = pd.read_excel("CrossV_ts2.xlsx", sheet_name='Sheet1')
CV_ts_3 = pd.read_excel("CrossV_ts3.xlsx", sheet_name='Sheet1')
CV_ts_4 = pd.read_excel("CrossV_ts4.xlsx", sheet_name='Sheet1')
CV_ts_5 = pd.read_excel("CrossV_ts5.xlsx", sheet_name='Sheet1')

def Error(y_pred, y_test):
    n = len(y_test);
    # EMA
    EMA = abs(y_pred - y_test).sum() / n
    # REQM
    REQM = np.sqrt(((y_test - y_pred)**2).sum() / n)
    # ERA
    ERA = abs(y_test - y_pred).sum() / abs(y_test - y_test.mean()).sum()
    # EQR
    EQR = np.sqrt(((y_test - y_pred)**2).sum()/ ((y_test - y_test.mean())**2).sum())
    # r
    Sup_ = (n * y_test*y_pred).sum() - (y_test.sum() * y_pred.sum())
    Inf_ = np.sqrt(((n * (y_test**2).sum()) - (y_test.sum())**2) * ((n * (y_pred**2).sum()) - (y_pred.sum())**2))
    r = Sup_ / Inf_
    # R2
    R2 = 1 - (((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum())

    err = [EMA, REQM, ERA,EQR,r,R2]
    return err 

def Treinamento(model):
    error = pd.DataFrame(np.zeros((5, 6)))
    error.columns = ['EMA', 'REQM', 'ERA', 'EQR', 'r', 'R2']
    count = 0
    kfold = range(1,6)
    for i in list(kfold):
        cv_train = ("CV_tr_"+str(i))
        cv_train = pd.DataFrame(globals()[cv_train])
        cv_test = ("CV_ts_"+str(i))
        cv_test = pd.DataFrame(globals()[cv_test])
        
        if model == "R_Linear":
            mod = LinearRegression()
            mod.fit(cv_train[['AT','V','AP','RH']], cv_train[['PE']])          
            coef = pd.DataFrame({"Variaveis":cv_train[['AT','V','AP','RH']].columns.tolist(),
                                 "Coeficiente":mod.coef_[0], "intercepto":mod.intercept_[0]})
            # Guardar COEF
            y_pred = mod.predict(cv_test[['AT','V','AP','RH']])
            error.loc[count] = Error(y_pred, cv_test[['PE']])
            count += 1
            print("Modelo ", i, "\n")
            print(error)
            print(coef)
            FileName1 = ("y_pred_"+str(model)+str(i)+".csv")
            pd.DataFrame(y_pred).to_csv(FileName1)
        elif model == "M5P":
            param_ = {'min_samples_split':[6, 10, 12, 15], 'max_depth':[3, 5, 10]}
            md = LinearTreeRegressor(base_estimator=LinearRegression())
            mod = GridSearchCV(md, param_, n_jobs=-1)
            mod.fit(cv_train[['AT','V','AP','RH']], cv_train[['PE']])
            BsParm = mod.best_params_
            y_pred = mod.predict(cv_test[['AT','V','AP','RH']])
            error.loc[count] = Error(y_pred.reshape(len(y_pred),1), cv_test[['PE']])
            count += 1
            print("Modelo ", i, "\n")
            print(error)
            print(BsParm)
            FileName1 = ("y_pred_"+str(model)+str(i)+".csv")
            pd.DataFrame(y_pred).to_csv(FileName1)
        elif model == "MVS":
            param_ = {'kernel':["rbf","linear","poly"], 'C':[10,30,50,100],'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
                      'epsilon':[0.01, 0.1, 0.15],'degree':[3,4,5]} #'':[]
            md = svm.SVR()
            mod = GridSearchCV(md, param_, n_jobs=-1)
            mod.fit(cv_train[['AT','V','AP','RH']], np.ravel(cv_train[['PE']]))
            BsParm = mod.best_params_
            y_pred = mod.predict(cv_test[['AT','V','AP','RH']])
            error.loc[count] = Error(y_pred.reshape(len(y_pred),1), cv_test[['PE']]) #
            count += 1
            print("Modelo ", i, "\n")
            print(error)
            print(BsParm)
            FileName1 = ("y_pred_"+str(model)+str(i)+".csv")
            pd.DataFrame(y_pred).to_csv(FileName1)
        else:
            param_ = {'hidden_layer_sizes': [(50),(30,50),(50,100)],
                      'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
                      'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive']} #'':[]
            md = MLPRegressor(max_iter=100000, random_state=0,verbose=False)
            mod = GridSearchCV(md, param_, n_jobs=-1, cv=2)
            mod.fit(cv_train[['AT','V','AP','RH']], np.ravel(cv_train[['PE']]))
            BsParm = mod.best_params_
            y_pred = mod.predict(cv_test[['AT','V','AP','RH']])
            error.loc[count] = Error(y_pred.reshape(len(y_pred),1), cv_test[['PE']]) #
            count += 1
            print("Modelo ", i, "\n")
            print(error)
            print(BsParm)
            FileName1 = ("y_pred_"+str(model)+str(i)+".csv")
            pd.DataFrame(y_pred).to_csv(FileName1)
            
    FileName2 = ("Error_"+str(model)+".csv")
    error.to_csv(FileName2)


Treinamento(model = "R_Linear")
#        EMA      REQM       ERA       EQR         r        R2
#0  3.662271  4.520312  0.250046  0.267749  0.963654  0.928310
#1  3.627835  4.490550  0.240419  0.259535  0.965853  0.932642
#2  3.593696  4.434069  0.241929  0.259050  0.965871  0.932893
#3  3.626182  4.646501  0.246533  0.275050  0.961624  0.924347
#4  3.635963  4.712650  0.244363  0.275130  0.961425  0.924303
Treinamento(model = "M5P")
#        EMA      REQM       ERA       EQR         r        R2
#0  3.113647  3.999151  0.212588  0.236879  0.971585  0.943888
#1  3.180240  4.040262  0.210757  0.233510  0.972402  0.945473
#2  3.118637  3.978200  0.209947  0.232417  0.972629  0.945982
#3  3.155458  4.216956  0.214530  0.249624  0.968490  0.937688
#4  3.183308  4.284720  0.213942  0.250147  0.968208  0.937426
Treinamento(model = "MVS")
#        EMA      REQM       ERA       EQR         r        R2
#0  3.177640  4.071538  0.216958  0.241167  0.970808  0.941838
#1  3.195162  4.087183  0.211746  0.236222  0.971750  0.944199
#2  3.137841  4.023946  0.211240  0.235090  0.972018  0.944733
#3  3.148229  4.244425  0.214038  0.251250  0.968390  0.936874
#4  3.202078  4.318192  0.215203  0.252101  0.967787  0.936445
Treinamento(model = "MLP")
#        EMA      REQM       ERA       EQR         r        R2
#0  3.439708  4.306157  0.234851  0.255064  0.966941  0.934942
#1  3.399663  4.267158  0.225298  0.246624  0.969220  0.939177
#2  3.361325  4.202801  0.226285  0.245539  0.969764  0.939711
#3  3.477135  4.506100  0.236399  0.266739  0.964608  0.928850
#4  3.471999  4.562935  0.233344  0.266390  0.964158  0.929037

# Erro
anfis = pd.read_excel("Anfis_er.xlsx")
anfis_ = anfis.mean()

RL = pd.read_csv('Error_R_Linear.csv')
RL = RL.drop(["Unnamed: 0"], axis=1)
RL_ = RL.mean()

M5P = pd.read_csv('Error_M5P.csv')
M5P = M5P.drop(["Unnamed: 0"], axis=1)
M5P_ = M5P.mean()

MVS = pd.read_csv('Error_MVS.csv')
MVS = MVS.drop(["Unnamed: 0"], axis=1)
MVS_ = MVS.mean()

MLP = pd.read_csv('Error_MLP.csv')
MLP = MLP.drop(["Unnamed: 0"], axis=1)
MLP_ = MLP.mean()

index = ['ANFIS', 'RL', 'M5P', 'MVS', 'MLP']
dt = pd.DataFrame([anfis_,RL_,M5P_,MVS_,MLP_],index=['ANFIS', 'RL', 'M5P', 'MVS', 'MLP'])
dt

from matplotlib.ticker import FormatStrFormatter

col = ['#f28482','#ffac81','#adc178','#f4d35e','#669bbc']

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6))  = plt.subplots(2, 3,figsize=(9,4))#, sharey='row'

ax1.bar(index, dt['EMA'], color =col, width = 0.7)
ax1.set_title("EMA", fontsize=12) 
ax2.bar(index, dt['REQM'], color =col, width = 0.7)
ax2.set_title("REQM", fontsize=12) 
ax3.bar(index, dt['ERA'], color =col, width = 0.7)
ax3.set_title("ERA", fontsize=12) 
ax4.bar(index, dt['EQR'], color =col, width = 0.7)
ax4.set_title("EQR", fontsize=12) 
ax5.plot(dt[['r']], color ='#006d77', marker = 'h', linewidth=1, markeredgewidth=1)
ax5.set_title("r", fontsize=12) 
ax6.plot(dt[['R2']], color ='#735d78',marker = 'o', linewidth=1, markeredgewidth=1)
ax6.set_title("$R^{2}$", fontsize=12) 
fig.subplots_adjust(wspace=0.25, hspace=0.44)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax5.set_ylim([0.95, 0.99])
ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax6.set_ylim([0.92, 0.96])
plt.show()

#fig.savefig('Metrics.pdf', bbox_inches='tight')

real = CV_ts_4[['PE']]
pred_Anfis = pd.read_excel("Pred_ANFIS_CV4.xlsx", sheet_name='Sheet1')
pred_Anfis  = pred_Anfis.drop(["AT","V","AP","RH","PE"], axis=1)
pred_RL = pd.read_csv('y_pred_R_Linear4.csv')
pred_RL = pred_RL.drop(["Unnamed: 0"], axis=1)
pred_M5P = pd.read_csv('y_pred_M5P4.csv')
pred_M5P = pred_M5P.drop(["Unnamed: 0"], axis=1)
pred_MLP = pd.read_csv('y_pred_MLP4.csv')
pred_MLP = pred_MLP.drop(["Unnamed: 0"], axis=1)
pred_MVS = pd.read_csv('y_pred_MVS4.csv')
pred_MVS = pred_MVS.drop(["Unnamed: 0"], axis=1)

fig = plt.figure(constrained_layout=False,figsize=(6,8))
gs = gridspec.GridSpec(3, 2, figure=fig)
line = mlines.Line2D([0, 1], [0, 1], color='#bc4749', linewidth=1, linestyle='--')
line2 = mlines.Line2D([0, 1], [0, 1], color='#bc4749', linewidth=1, linestyle='--')
line3 = mlines.Line2D([0, 1], [0, 1], color='#bc4749', linewidth=1, linestyle='--')
line4 = mlines.Line2D([0, 1], [0, 1], color='#bc4749', linewidth=1, linestyle='--')
line5 = mlines.Line2D([0, 1], [0, 1], color='#bc4749', linewidth=1, linestyle='--')
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
R_1 = ("$R^{2}:$ "+str(round(dt.iloc[1]['R2'],4)))
R_2 = ("$R^{2}:$ "+str(round(dt.iloc[2]['R2'],4)))
R_3 = ("$R^{2}:$ "+str(round(dt.iloc[4]['R2'],4)))
R_4 = ("$R^{2}:$ "+str(round(dt.iloc[0]['R2'],4)))
R_5 = ("$R^{2}:$ "+str(round(dt.iloc[3]['R2'],4)))
## pred_RL
ax1.scatter(real, pred_RL, color=col[1], s=20, alpha=0.3)
transform = ax1.transAxes
line.set_transform(transform)
ax1.add_line(line)
ax1.set_xlabel("Previsões ~ Regressão Linear")
ax1.set_ylabel("Dados reais")
ax1.annotate(R_1, (425, 485), size=9)
## pred_M5P
ax2.scatter(real, pred_M5P, color=col[2], s=20, alpha=0.3)
transform2 = ax2.transAxes
line2.set_transform(transform2)
ax2.add_line(line2)
ax2.set_xlabel("Previsões ~ Modelo M5P")
ax2.set_ylabel("Dados reais")
ax2.annotate(R_2, (425, 485), size=9)
## pred_MLP
ax3.scatter(real, pred_MLP, color=col[4], s=20, alpha=0.3)
transform3 = ax3.transAxes
line3.set_transform(transform3)
ax3.add_line(line3)
ax3.set_xlabel("Previsões ~ Multilayer Perceptron")
ax3.set_ylabel("Dados reais")
ax3.annotate(R_3, (425, 485), size=9)
## pred_Anfis
ax4.scatter(real, pred_Anfis, color=col[0], s=20, alpha=0.3)
transform4 = ax4.transAxes
line4.set_transform(transform4)
ax4.add_line(line4)
ax4.set_xlabel("Previsões ~ ANFIS")
ax4.set_ylabel("Dados reais")
ax4.annotate(R_4, (425, 485), size=9)
## pred_MVS
ax5.scatter(real, pred_MVS, color=col[3], s=20, alpha=0.3)
transform5 = ax5.transAxes
line5.set_transform(transform5)
ax5.add_line(line5)
ax5.set_xlabel("Previsões ~ Máquina de Vector")
ax5.set_ylabel("Dados reais")
ax5.annotate(R_5, (425, 485), size=9)
#fig.suptitle("GridSpec")
fig.subplots_adjust(wspace=0.33, hspace=0.33)
plt.show()
#, color="#", s=20, alpha=0.5

fig.savefig('Distr.pdf', bbox_inches='tight')

















