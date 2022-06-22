import numpy as np 


def NDRE(img):
    return (img[:,:,5] - img[:,:,4]) / (img[:,:,5] + img[:,:,4])

def NDVI(img): 
    return (img[:,:,5] - img[:,:,3]) / (img[:,:,5] + img[:,:,3])

def mean_NDRE(img):
    return (img[:,:,5].mean() - img[:,:,4].mean()) / (img[:,:,5].mean() + img[:,:,4].mean())

def mean_NDVI(img): 
    return (img[:,:,5].mean() - img[:,:,3].mean()) / (img[:,:,5].mean() + img[:,:,3].mean())

def mean_GNDVI(img):
    return ((img[:,:,5].mean() - img[:,:,2].mean())/(img[:,:,5].mean() + img[:,:,2].mean()))

def mean_MCARI2(img):
    return ((1.5*(2.5*(img[:,:,5].mean() - img[:,:,3].mean()) - 1.3*(img[:,:,5].mean() - img[:,:,2].mean())))/ (np.square(2*img[:,:,5].mean() + 1)-(6*img[:,:,5].mean() - 5*np.sqrt(img[:,:,3].mean()))-0.5))
     
def mean_PRI(img):
    return ((img[:,:,2].mean() - img[:,:,1].mean())/(img[:,:,2].mean() + img[:,:,1].mean()))

def mean_RERVI(img):
    return (img[:,:,5].mean()/img[:,:,4].mean())
    
def mean_REDVI(img):
    return (img[:,:,5].mean() - img[:,:,4].mean())

def mean_RVI(img):
    return (img[:,:,3].mean()/img[:,:,5].mean())

def mean_OSAVI(img):
    return ((img[:,:,5].mean() - img[:,:,3].mean())/(img[:,:,5].mean() + img[:,:,3].mean() + 0.16))

def mean_RDVI(img):
    return ((img[:,:,5].mean() - img[:,:,3].mean())/np.sqrt((img[:,:,5].mean() + img[:,:,3].mean())))
    
def mean_EVI(img):
    return (2.4*(img[:,:,5].mean() - img[:,:,4].mean())/(img[:,:,5].mean() + img[:,:,4].mean() + 1))


def mean_PIR(img):
    return img[:,:,5].mean()

def mean_RE(img):
    return img[:,:,4].mean()

def mean_R(img):
    return img[:,:,3].mean()

def mean_V2(img):
    return img[:,:,2].mean()

def mean_V1(img):
    return img[:,:,1].mean()

def mean_B(img):
    return img[:,:,0].mean()