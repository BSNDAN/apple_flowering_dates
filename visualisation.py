import os

import cv2
import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from gestion_image import raster_to_img, patch, patch_to_list, tiles_to_list, patch_df, tiles_df

################################################### Choix de la résolution ###################################################

def rand_plot_RGB(data, img, nb, resolution):   
    
    """Pour une image en RGB donné, affiche un nombre d'arbre tiré aléatoirement (nb) dans un espace de résolution défini"""
    
    rand = [random.randint(0,data.shape[0]) for i in range(nb)]

    for num_arbre in rand:
        plt.figure(figsize=(15,20))
        for i, res in enumerate(resolution):
            plt.subplot(nb,len(resolution),i+1)
            plt.axis('off')
            plt.imshow(patch(data, img, num_arbre, res))
            plt.title(f"arbre {num_arbre}, résolution {res} pixels")
        plt.show()
        
        
def rand_plot_MS(data, img, nb, resolution):   
    
    """Pour une image en MS donné, affiche un nombre d'arbre tiré aléatoirement (nb) dans un espace de résolution défini"""
    
    rand = [random.randint(0,data.shape[0]) for i in range(nb)]

    for num_arbre in rand:
        plt.figure(figsize=(15,20))
        for i, res in enumerate(resolution):
            
            image = patch(data, img, num_arbre, res)
            B = image[:,:,0]
            G = image[:,:,1]
            R = image[:,:,3]
            rgb = cv2.normalize(np.dstack((R,G,B)),  np.zeros((res,res)), 50, 255, cv2.NORM_MINMAX)
            
            plt.subplot(nb,len(resolution),i+1)
            plt.axis('off')
            plt.imshow(rgb)
            plt.title(f"arbre {num_arbre}, résolution {res} pixels")
        plt.show()
        
        
def visu_resolution(capteur, mois, nb, resolution):
    
    """ Pour un capteur (RGB/MS) donné et pour un mois donné, affiche un nombre d'arbre tiré aléatoirement (nb) dans un espace de résolution défini"""
    
    os.chdir("/Volumes/disque/Mosaic")
    
    trees = pd.read_csv("SIG-terrain/trees_1.csv").iloc[:,1:]
    
    ds = gdal.Open("ortho_" + capteur + "/" + mois + ".tif")
    geot = ds.GetGeoTransform() 
    trees["X"], trees["y"] = ((trees["northing"] - geot[3])/geot[5]), ((trees["easting"] - geot[0])/geot[1])
                                                                       
    img = raster_to_img(ds)
    
    if capteur == "rgb":
        return rand_plot_RGB(trees, img, nb, resolution) 
    
    elif capteur == "ms":
        return rand_plot_MS(trees, img, nb, resolution) 
    
    else : print("nothing to plot")
          
        
    
################################################### Visualisation temporelle ###################################################


def time_plot_RGB(data, num):  
    arbre, col, mois, i = data.loc[num], ["juin_rgb", "sept_rgb", "oct_rgb", "nov_rgb"], ["Juin", "septembre", "octobre", "novembre"], 1
        
    plt.figure(figsize=(20,8))
    for col, mois in zip(col, mois):
        plt.subplot(1,4,i)
        plt.imshow(arbre[col])
        plt.axis("off")
        plt.title(f"arbre {num}, {mois}")
        i+=1
        
        
def time_plot_MS(data, num):  
    
    arbre, col, mois, i = data.loc[num], ["juin_ms","sept_ms", "oct_ms", "nov_ms"], ["Juin","septembre", "octobre", "novembre"], 1 
        
    plt.figure(figsize=(20,8))
    for col, mois in zip(col, mois):
        img = arbre[col]
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,3]       
        rgb = cv2.normalize(np.dstack((R,G,B)),  np.zeros((200,200)), 50, 255, cv2.NORM_MINMAX)
        
        plt.subplot(1,4,i)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"arbre {num}, {mois}")
        i+=1
        
        
################################################### Visualisation patch et tiles ###################################################


def tiles_rgb_plot(capteur, index, mois):
    
    trees_patch = patch_df(capteur)
    trees_tiles = tiles_df(capteur)
    
    ident = trees_patch.loc[index]["id_tree"]
    
    plt.figure(figsize=(12,8))
    plt.imshow(trees_patch[trees_patch["id_tree"] == ident][mois].iloc[0])
    plt.axis("off")
    plt.show()
    
    plt.figure(figsize=(18,12))
    for i in range(16):
        plt.subplot(5,5,i+1)
        plt.imshow(trees_tiles[trees_tiles["id_tree"] == ident][mois].reset_index(drop=True).loc[i])
        plt.axis("off")
        
        
def tiles_ms_plot(capteur, index, mois):
    
    trees_patch = patch_df("ms")
    trees_tiles = tiles_df("ms")
    
    ident = trees_patch.loc[index]["id_tree"]
   
    img = trees_patch[trees_patch["id_tree"] == ident][mois].iloc[0]
    
    B,G,R = img[:,:,0],img[:,:,1], img[:,:,3]
    rgb = cv2.normalize(np.dstack((R,G,B)),  np.zeros((200,200)), 50, 255, cv2.NORM_MINMAX)
    
    plt.figure(figsize=(12,8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()
    
    plt.figure(figsize=(18,12))
    for i in range(36):
        plt.subplot(6,6,i+1)
        img = trees_tiles[trees_tiles["id_tree"] == ident][mois].reset_index(drop=True).loc[i]
        B,G,R = img[:,:,0],img[:,:,1], img[:,:,3]
        rgb = cv2.normalize(np.dstack((R,G,B)),  np.zeros((200,200)), 50, 255, cv2.NORM_MINMAX)
        plt.imshow(rgb)
        plt.axis("off")