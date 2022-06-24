import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from calcul_indices import NDVI, NDRE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor



def count_nan(img):
    return np.isnan(img).sum()

def RMSE(X,y):
    return round(mean_squared_error(y,X, squared=False),2)

def R2(X,y):
    return round(r2_score(y,X), 2)




def indices_mean_split(data,func, printer=False):
    """ calcul le NDRE et NDVI pixel a pixel, puis une fonction func (moyenne/std) pour une image, puis split le jeu """
    
    for mois in data.columns[data.columns.str.contains("ms")]:
        data["NDRE_"+ mois] = data[mois].apply(NDRE).apply(func)
        data["NDVI_"+ mois] = data[mois].apply(NDVI).apply(func)
        
    data = data.drop(["juin_ms", "sept_ms","oct_ms", "nov_ms"], axis=1)
        
    if printer is True:
        for mois in data.columns[data.columns.str.contains("ms")]:
            print(mois)
            print("--")
            print(f"nan NDRE : {data[mois].apply(count_nan).sum()}")
            print(f"min NDRE : {data[mois].apply(np.min).min()}")
            print(f"max NDRE : {data[mois].apply(np.max).max()}")

            print(f"nan NDVI : {data[mois].apply(count_nan).sum()}")
            print(f"min NDVI : {data[mois].apply(np.min).min()}")
            print(f"max NDVI : {data[mois].apply(np.max).max()}")
            print('---------------------------------------')
    
    train = data[data.split == "train"]
    val = data[data.split == "val"]
    test = data[data.split == "test"]
    
    X_train = train[train.columns[train.columns.str.contains("ms")]]
    y_train = train["jourF"]

    X_val = val[val.columns[val.columns.str.contains("ms")]]
    y_val = val["jourF"]

    X_test = test[test.columns[test.columns.str.contains("ms")]]
    y_test = test["jourF"]
    
    return X_train, y_train, X_val, y_val, X_test, y_test






def indices_mean(data,func, printer=False):
    """ calcul le NDRE et NDVI pixel a pixel, puis une fonction func (moyenne/std) pour une imag """
    
    for mois in data.columns[data.columns.str.contains("ms")]:
        data["NDRE_"+ mois] = data[mois].apply(NDRE).apply(func)
        data["NDVI_"+ mois] = data[mois].apply(NDVI).apply(func)
        
    data = data.drop(["juin_ms", "sept_ms","oct_ms", "nov_ms"], axis=1)
        
    if printer is True:
        for mois in data.columns[data.columns.str.contains("ms")]:
            print(mois)
            print("--")
            print(f"nan NDRE : {data[mois].apply(count_nan).sum()}")
            print(f"min NDRE : {data[mois].apply(np.min).min()}")
            print(f"max NDRE : {data[mois].apply(np.max).max()}")

            print(f"nan NDVI : {data[mois].apply(count_nan).sum()}")
            print(f"min NDVI : {data[mois].apply(np.min).min()}")
            print(f"max NDVI : {data[mois].apply(np.max).max()}")
            print('---------------------------------------')
    
    return data




def opti_lasso(X_train, y_train, X_val, y_val):
    """ Optimise le paramètre de régularisation d'une régression lasso, la range est défini  (0.0000001, 0.1, 0.0001) """
    
    alpha_value, MSE_score = [], []

    for n in np.arange(0.0000001, 0.1, 0.0001):
        lasso = Lasso(alpha=n)
        lasso.fit(X_train, y_train)
        MSE = mean_squared_error(y_val, lasso.predict(X_val))
        alpha_value.append(n)
        MSE_score.append(MSE)

    return alpha_value[MSE_score.index(min(MSE_score))], min(MSE_score)


def opti_RF(X_train, y_train, X_val, y_val):
    """ Optimise le nombre d'arbre d'une régression par RF, le nombre de seuil est fixé au maxium a 10 """
    
    number_tree, MSE_score = [], []

    for nestimator in np.arange(100, 500, 50):
        RF = RandomForestRegressor(n_estimators=nestimator, max_depth=10, n_jobs=-1, )
        RF.fit(X_train, y_train)
        MSE = mean_squared_error(y_val, RF.predict(X_val))
        number_tree.append(nestimator)
        MSE_score.append(MSE)

    return number_tree[MSE_score.index(min(MSE_score))], min(MSE_score)
    
    

def plot_pred(modele, X, y):   
    """ Plot les observations et prédictions d'un modèle """

    for model,color in zip(modele, ["orange", "skyblue", "darkorchid"]):
        plt.scatter(y=y,x=model.predict(X), c=color)

    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'grey', lw=2) 
    plt.xlabel("prédiction")
    plt.ylabel("observation")
    plt.legend(list(map(str,modele)), bbox_to_anchor = (0.65, 1.25))