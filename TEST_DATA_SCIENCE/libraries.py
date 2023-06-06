#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:37:12 2020

@author: nacho
"""
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from pipelinehelper import PipelineHelper
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import sys, os
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RepeatedKFold
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import scikitplot as skplt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from warnings import filterwarnings
filterwarnings('ignore')
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from sklearn.exceptions import ConvergenceWarning
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold



#Visualizar pca em 2D y 3D
class pca():
    def __init__(self,  df=None, titulo="Unspecified", label_y=None):
        self.df = df
        self.label_y = str(label_y)
        self.titulo = str(titulo)
        print(list(df))
        print(f"Numero de elementos de {label_y}\n", df[label_y].value_counts())
    def pca_2D(self):
        df_PCA = self.df.drop([self.label_y], axis=1)
        #instanciamos el metodo pca con 2 componentes
        pca = PCA(n_components=2)
        #encontramos los componentes principales usando 
        #el método de ajuste con 2 componentes
        #transformamos los datos scaled_data en 2 componentes con pca
        pca.fit(df_PCA)
        x_pca = pca.transform(df_PCA)
        ######
        #instanciamos un objeto para hacer PCA
        scaler = StandardScaler()
        #escalar los datos, estandarizarlos, para que cada
        #caracteristica tenga una varianza unitaria 
        scaler.fit(df_PCA)
        #aplicamos la reducción de rotación y dimensionalidad
        scaled_data = scaler.transform(df_PCA)
        pca = PCA().fit(scaled_data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('How many components are needed to describe the data.')
        ######
        print("Dimension de los features orginales: ", df_PCA.shape)
        print("Dimension de los features con 2 componentes", x_pca.shape)
        
        #visualizar los datos en 2 dimensiones
        #plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()
        scatter = plt.scatter(x_pca[:,0],
                    x_pca[:,1],
                    c=self.df[self.label_y],
                    cmap='rainbow',
                    marker='o',
                    s=2,
                    linewidths=0)
        #genera legend del target
        labels = np.unique(self.df[self.label_y])
        handles = [plt.Line2D([],[],marker=".", ls="", 
                              color=scatter.cmap(scatter.norm(yi))) for yi in labels]
        plt.legend(handles, labels)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.title(self.titulo)
        #plt.show()
        y = self.df[self.label_y]
        return x_pca, y
    def pca_3D(self):
        sns.set_style("white")  
        self.df[self.label_y] = pd.Categorical(self.df[self.label_y])
        my_color = self.df[self.label_y].cat.codes
        df_PCA = self.df.drop([self.label_y], axis=1)
        pca = PCA(n_components=3)
        pca.fit(df_PCA)
        result=pd.DataFrame(pca.transform(df_PCA), 
                            columns=['PCA%i' % i for i in range(3)], 
                            index=df_PCA.index)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(result['PCA0'], 
                   result['PCA1'], 
                   result['PCA2'], 
                   c=my_color, 
                   cmap='rainbow', 
                   s=2, marker="o",
                   linewidths=0)
        
        #genera legend del target
        labels = np.unique(self.df[self.label_y])
        handles = [plt.Line2D([],[],marker=".",ls="",
                                 color=scat.cmap(scat.norm(yi))) for yi in labels]               
        ax.legend(handles, labels)
        
        # make simple, bare axis lines through space:
        xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
         
        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(self.titulo)
        #plt.show()
        fig.tight_layout()
        y = self.df[self.label_y]
        return result, y

#Visualizar smv en 2d   
def plot_svm_2d(grid, X_test, Y_test):
    scaler1 = StandardScaler()
    scaler1.fit(X_test)
    X_test_scaled = scaler1.transform(X_test)
    
    
    pca1 = PCA(n_components=2)
    X_test_scaled_reduced = pca1.fit_transform(X_test_scaled)
    
    
    svm_model = SVC(kernel='rbf', C=float(grid.best_params_['SupVM__C']), 
                    gamma=float(grid.best_params_['SupVM__gamma']))
    
    classify = svm_model.fit(X_test_scaled_reduced, Y_test)
    
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    
    def make_meshgrid(x, y, h=.1):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))#,
                             #np.arange(z_min, z_max, h))
        return xx, yy
    
    X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    fig, ax = plt.subplots(figsize=(12,9))
    fig.patch.set_facecolor('white')
    cdict1={0:'lime',1:'deeppink'}
    
    Y_tar_list = Y_test.tolist()
    yl1= [int(target1) for target1 in Y_tar_list]
    labels1=yl1
     
    labl1={0:'Malignant',1:'Benign'}
    marker1={0:'*',1:'d'}
    alpha1={0:.8, 1:0.5}
    
    for l1 in np.unique(labels1):
        ix1=np.where(labels1==l1)
        ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])
    
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
               edgecolors='navy', label='Support Vectors')
    
    plot_contours(ax, classify, xx, yy,cmap='seismic', alpha=0.4)
    plt.legend(fontsize=15)
    
    plt.xlabel("1st Principal Component",fontsize=14)
    plt.ylabel("2nd Principal Component",fontsize=14)
    plt.show()

#Tune the Number of Selected Features
def RKFold(X,y):
    # define the evaluation method
    cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
    # define the pipeline to evaluate
    model = LinearRegression()
    fs = SelectKBest(score_func=mutual_info_regression)
    pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
    # define the grid
    grid = dict()
    grid['sel__k'] = [i for i in range(2, X.shape[1]+1)]
    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
    # perform the search
    results = search.fit(X, y)
    # summarize best
    print('Best MAE: %.5f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.5f with: %r" % (mean, param))
    return results.best_params_['sel__k']

#entrenamiento cross-validation con hiperparámetros
def Gridsearchcv(X_train, X_test, y_train, y_test):
    ############
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', PipelineHelper([
            # ('svc', SVC(probability=True)),
            # ('brf', BalancedRandomForestClassifier()),
            ('gb', GradientBoostingClassifier())
        ])),
    ])

    params = {
    'clf__selected_model': pipe.named_steps['clf'].generate({
        # # #gb 3780
        "gb__learning_rate": [0.1],
        "gb__max_depth":[8,9,10],
        "gb__max_features":["sqrt"],
        "gb__subsample":[0.8],
        "gb__n_estimators":[100],

        # # #BalancedRandomForestClassifier
        # 'brf__criterion': ['gini', 'entropy'],
        # 'brf__max_depth' : [int(x) for x in np.linspace(1, 45, num = 3)],
        # 'brf__max_features' : ['auto', 'sqrt', 'log2'],

        # # # #svm 
        # 'svc__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
        # 'svc__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50],
        # 'svc__kernel': ['rbf'],
        
        }),
    }
    scoring = {
        'F1': make_scorer(f1_score, average='micro'),
        'pr': make_scorer(precision_score, average='micro'),
        'rc': make_scorer(recall_score, average='micro'),
        'acc': make_scorer(accuracy_score)
        }
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    #n_iter: 30,60, 100
    grid = RandomizedSearchCV(
        pipe, 
        params,
        refit = 'acc',
        cv = cv, 
        verbose = 3, 
        n_jobs=-1,
        n_iter = 1,
        scoring= scoring,
        return_train_score = True
        )

    grid.fit(X_train, y_train)
    df_grid=pd.DataFrame(grid.cv_results_)
    df_grid = df_grid.sort_values(by=['mean_test_F1'],ascending=False)
    df_grid = df_grid[[
        'param_clf__selected_model',
        'params',
        'mean_fit_time',
        'std_fit_time',

        'mean_test_pr',
        'std_test_pr',
        'rank_test_pr',
        
        'mean_test_rc',
        'std_test_rc',
        'rank_test_rc',
        
        'mean_test_F1', 
        'std_test_F1', 
        'rank_test_F1',
        
        'mean_test_acc', 
        'std_test_acc', 
        'rank_test_acc'
    ]]

    print("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.best_estimator_.predict(X_test) 
    report = classification_report(y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(y_test, grid_predictions))

    return grid, df_grid, report