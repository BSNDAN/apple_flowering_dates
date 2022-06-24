import os

import cv2
import gdal
import imageio
import numpy as np
import pandas as pd

from osgeo import gdalconst, ogr, osr


def raster_to_img(rast):  
    """ Lire image tiff et la transformer en numpy array (x,y,canal) """
    raster = [rast.GetRasterBand(i+1).ReadAsArray() for i in range(rast.RasterCount)]
    return np.dstack(raster)


def patch(data ,image, i, long):  
    """A partir de coordonnées d'un centre d'arbre, créer un patch de longueur 'long' """
    l = int(long/2)
    X, y = int(data.loc[i]["X"]), int(data.loc[i]["y"])
    return image[X-l:X+l, y-l:y+l]


def patch_to_list(capteur, trees):
    """ Stack l'ensemble des patch d'un capteur (MS/RGB) stocké dans une liste (split) """
    
    os.chdir(os.path.join("/Volumes/disque/Mosaic/Patch", capteur))
    
    split,nom  = [], [] 
    
    for tuiles in os.listdir():
        
        if capteur in tuiles:
            
            nom.append(tuiles)
            os.chdir(tuiles)
            img = []

            for ligne in range(trees.shape[0]):
                
                line = trees.loc[ligne]
                
                if capteur == "rgb": 
                    img.append(np.load(f"{line.id_tree}.npy"))
                
                elif capteur == "ms": 
                    img.append(np.load(f"{line.id_tree}.npy"))
                
                else : 
                    print("capteur inconnue")
                    break
             
            split.append(img)
            
            print(f"{tuiles} done")
            print("-----------")
            
            os.chdir("..")
    
    return split, nom


def tiles_to_list(capteur, data):
    """ Prends en entrée un capteur (RGB/MS) et va chercher les tuiles correspondantes pour les stocker (month_img) ainsi que leur nom (id_tree), le numéro de tiles (identifiant) et le mois correspondant"""
    
    os.chdir(os.path.join("/Volumes/disque/Mosaic/tiles", capteur))
    
    month_img,name  = [], [] 
    
    for mois in os.listdir():
        
        if capteur in mois:
            
            name.append(mois)
            os.chdir(mois)
            img, id_tree,identifiant = [], [], []

            for ligne in range(data.shape[0]):
                
                line = data.loc[ligne]
                
                if capteur == "rgb":
                    for i in range(16):
                        img.append(np.load(f"{line.id_tree}_T{i}.npy")), id_tree.append(line.id_tree), identifiant.append(i+1)
                    
                elif capteur == "ms": 
                    for i in range(36):
                        img.append(np.load(f"{line.id_tree}_T{i}.npy")), id_tree.append(line.id_tree), identifiant.append(i+1)
                        
                else : 
                    print("capteur inconnue")
                    break
             
            month_img.append(img)
            
            print(f"{mois} done")
            print("-----------")
            
            os.chdir("..")
    
    return id_tree, month_img, name, identifiant


def splitter(jeu, mois, data):
    """  """

    d_ = data[data.split == jeu].reset_index(drop=True)
    
    X_ = d_[mois] #récupérer le jeu correspondant dans le df pandas
    X_data = np.array([X_[idx] for idx in X_.index]) #transformer series(100,100,3) en np(len,100,100,3)
    
    y_data = d_["jourF"]
        
    print(f"{jeu} done")
    print("----------------")
    return X_data, y_data



def patch_df(capteur, split=2):
    """Pour un capteur donné (RGB/MS), renvoie un dataframe composé de tous les patchs"""
    
    os.chdir("/Volumes/disque/Mosaic") 
    trees_patch = pd.read_csv("SIG-terrain/trees_{s}.csv".format(s = split)).iloc[:,1:]
    
    print("chargement patch :")
    print("-------------------")
    patching, nom = patch_to_list(capteur, trees_patch)
    print("chargement terminé")
    print("-------------------")
    print("-------------------")
    
    #ajouter les patch au dataframe pour une meilleur gestion des data
    for idx,name in enumerate(nom):
        trees_patch[name] = patching[idx]
        
    #transformer les array uint32 en int32 pour pouvoir calculer les NDVI et NDRE 
    for mois in trees_patch.columns[trees_patch.columns.str.contains("ms")]:
        trees_patch[mois] = trees_patch[mois].apply(np.int32)
        
        
    return trees_patch


def tiles_df(capteur, split=2):
    """Pour un capteur donné (RGB/MS), renvoie un dataframe composé de toutes les tuiles"""
    
    os.chdir("/Volumes/disque/Mosaic") 
    trees_tiles = pd.read_csv("SIG-terrain/trees_{s}.csv".format(s = split)).iloc[:,1:]
    
    print("chargement tiles :")
    print("-------------------")
    id_tree, month_img, name, identifiant = tiles_to_list(capteur, trees_tiles)
    print("chargement terminé")
    print("-------------------")
    print("-------------------")
    
    df_ms = pd.DataFrame({"id_tree" : id_tree, "tiles" : identifiant, name[0]: month_img[0], name[1] : month_img[1], name[2] : month_img[2], name[3] : month_img[3]})
        
    trees_tiles = trees_tiles.merge(df_ms, on="id_tree")
    
    #transformer les array uint32 en int32 pour pouvoir calculer les NDVI et NDRE 
    for mois in trees_tiles.columns[trees_tiles.columns.str.contains("ms")]:
        trees_tiles[mois] = trees_tiles[mois].apply(np.int32)
        
    return trees_tiles