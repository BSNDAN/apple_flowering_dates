import os 

import gdal
import numpy as np
import pandas as pd

from gestion_image import raster_to_img, patch, patch_df

################################################### Patch ###################################################

def import_mosaics(capteur, res, data):
    """ import toutes les mosaiques d'un capteur (rgb/ms), tranforme chaque raster (mosaique) en array, retourne une liste de patch """ 

    nom, split = [], []
    
    os.chdir("/Volumes/disque/Mosaic")
    for name in os.listdir("ortho_" + capteur):
        if 'tif' in name:
            nom.append(name)
            ds = gdal.Open("ortho_" + capteur + "/" + name)
            geot = ds.GetGeoTransform() 
            data["X"], data["y"] = ((data["northing"] - geot[3])/geot[5]), ((data["easting"] - geot[0])/geot[1])

            img = raster_to_img(ds)
            split.append([patch(data, img, i, res) for i in range(data.shape[0])])
            print(f"{name} : done")
            print("________________")
            
    return nom, split



def export_patch(capteur, data):
     """ exporte les patchs issues d'un dataframeen format numpy """
    
    for mois in data.columns[data.columns.str.contains(capteur)]:
        
        chemin = os.path.join("/Volumes/disque/Mosaic/Patch", capteur, mois)
        os.mkdir(chemin)
        os.chdir(chemin)
        
        for ligne in range(data.shape[0]):
            np.save(f"{data.loc[ligne].id_tree}", data.loc[ligne][mois])
            
        os.chdir("..")
        
     
    
def patching(capteur, res, split=2):
    """ créer des patchs de résolution "res" et les exportes das in fichier """
    
    os.chdir("/Volumes/disque/Mosaic") 
    trees = pd.read_csv("SIG-terrain/trees_{s}.csv".format(s = split)).iloc[:,1:]
    
    nom, split = import_mosaics(capteur, res, trees)
    print("import mosaics done")
    
    nom = [nom[i].replace(".tif", "") for i in range(len(nom))]
    
    for idx,name in enumerate(nom):
        trees[name] = split[idx]
        
    export_patch(capteur, trees)
    print("-------------------")
    print("export patch done")
        
################################################### Tiles ###################################################       
        
def create_tiles(data, mois, pool_size, stride):    
    """Générer des tiles à partir de patch, les patchs sont d'une taille 'au pool size' et il y a un décalage de 'strides' entre 2 tiles d'un meme patch"""
    
    df = pd.DataFrame(columns=["id_tree", "tiles"])

    for ligne in range(data.shape[0]): 

        line = data.loc[ligne]
        img, tiles, tiles2 = line[mois], [], []

        for i in np.arange(img.shape[0], step=stride):
            for j in np.arange(img.shape[0], step=stride):
                mat = img[i:i+pool_size, j:j+pool_size]
                if mat.shape == (pool_size, pool_size, mat.shape[2]):
                    tiles.append(mat)
                tiles2.append(tiles)
        
        df.loc[ligne] = [line["id_tree"],tiles]       
    return df
        
        
def export_tiles(capteur, data):
    """ exporte les tiles issues d'un dataframe en format numpy """
    
    for mois in data.columns[data.columns.str.contains(capteur)]:
        
        chemin = os.path.join("/Volumes/disque/Mosaic/tiles", capteur, mois)
        os.mkdir(chemin)
        os.chdir(chemin)
        for ligne in range(data.shape[0]):
            line = data.loc[ligne]
            
            if capteur == "rgb" :            
                for i in range(16):
                    np.save(f"{line.id_tree}_T{i}", line[mois][i])
                    
            else : 
                for i in range(36):
                    np.save(f"{line.id_tree}_T{i}", line[mois][i])
        os.chdir("..")
        print(f"export {mois} : done")
        print("----------------------")
        
        
def tiling(capteur, pool_size, stride):   
    """ créer des tiles à partir de tiles  et les exportes danxs in fichier """
    trees, tile = patch_df(capteur), []

    for mois in trees.columns[trees.columns.str.contains(capteur)]:
        tile.append(create_tiles(trees, mois,pool_size, stride))

    data = pd.concat([tile[0], tile[1]["tiles"], tile[2]["tiles"], tile[3]["tiles"]], axis=1)
    name = list(trees.columns[trees.columns.str.contains(capteur)])
    name.insert(0, "id_tree")
    data.columns = name
    export_tiles(capteur, data)
    
    
    
    
    
