import numpy as np 


def NDRE(img):
    """ calcul le NDRE d'une image pixel a pixel """
    return (img[:,:,5] - img[:,:,4]) / (img[:,:,5] + img[:,:,4])

def NDVI(img):
    """ calcul le NDVI d'une image pixel a pixel """
    return (img[:,:,5] - img[:,:,3]) / (img[:,:,5] + img[:,:,3])

def mean_NDRE(img):
    """ calcul la moyenne de chacun des canaux puis calcul le NDRE """
    return (img[:,:,5].mean() - img[:,:,4].mean()) / (img[:,:,5].mean() + img[:,:,4].mean())

def mean_NDVI(img): 
    """ calcul la moyenne de chacun des canaux puis calcul le NDVI """
    return (img[:,:,5].mean() - img[:,:,3].mean()) / (img[:,:,5].mean() + img[:,:,3].mean())

def mean_PIR(img):
    """ calcul la moyenne du canal Proche Infra-rouge 850 nm """
    return img[:,:,5].mean()

def mean_RE(img):
    """ calcul la moyenne du canal Red Edge 730 nm """
    return img[:,:,4].mean()

def mean_R(img):
    """ calcul la moyenne du canal Rouge 675 nm """
    return img[:,:,3].mean()

def mean_V2(img):
    """ calcul la moyenne du canal vert 570nm """
    return img[:,:,2].mean()

def mean_V1(img):
    """ calcul la moyenne du canal vert 530nm """
    return img[:,:,1].mean()

def mean_B(img):
    """ calcul la moyenne du canal vert 450nm """
    return img[:,:,0].mean()