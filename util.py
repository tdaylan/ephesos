import numpy as np

import allesfitter
from allesfitter.priors.estimate_noise_wrap import estimate_noise_wrap

import sys, os, h5py, fnmatch

from astropy.io import fits
from astropy.coordinates import SkyCoord

# own modules
from tdpy.util import summgene

import astroquery
import astroquery.mast


def read_qlop(path, pathcsvv=None, stdvcons=None):
    
    print('Reading QLP light curve from %s...' % path)
    objtfile = h5py.File(path, 'r')
    time = objtfile['LightCurve/BJD'][()] + 2457000.
    tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
    flux = 10**(-(tmag - np.nanmedian(tmag)) / 2.5)
    flux /= np.nanmedian(flux) 
    arry = np.empty((flux.size, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    if stdvcons is None:
        stdvcons = 1e-3
        print('Assuming a constant photometric precision of %g for the QLP light curve.' % stdvcons)
    stdv = np.zeros_like(flux) + stdvcons
    arry[:, 2] = stdv
    
    # filter out bad data
    indx = np.where((objtfile['LightCurve/QFLAG'][()] == 0) & np.isfinite(flux) & np.isfinite(time) & np.isfinite(stdv))[0]
    arry = arry[indx, :]
    if not np.isfinite(arry).all():
        print('arry')
        summgene(arry)
        raise Exception('Light curve is not finite')
    if arry.shape[0] == 0:
        print('arry')
        summgene(arry)
        raise Exception('Light curve has no data points')

    if pathcsvv is not None:
        print('Writing to %s...' % pathcsvv)
        np.savetxt(pathcsvv, arry, delimiter=',')

    return arry


def retr_indxtimetran(time, epoc, peri, duramask):
    
    listindxtimemask = []
    for n in range(-20000, 20000):
        timeinit = epoc + n * peri - duramask / 2.
        timefinl = epoc + n * peri + duramask / 2.
        indxtimemask = np.where((time > timeinit) & (time < timefinl))[0]
        listindxtimemask.append(indxtimemask)
        
    indxtimemask = np.concatenate(listindxtimemask)
    numbtime = time.size
    indxtime = np.arange(numbtime)
    
    return indxtimemask
    

def down_spoclcur(strgobjt, pathdown, boollcuronly=True):
    
    obsTable = astroquery.mast.Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', objectname=strgobjt)
    dataProducts = astroquery.mast.Observations.get_product_list(obsTable[0])
    if boollcuronly:
        want = dataProducts['description'] == 'Light curves'
    else:
        want = np.arange(len(dataProducts))
    manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathdown)
    pathdown = manifest['Local Path'][0]
    
    return pathdown


def fold(arry, epoc, peri):
    
    time = arry[:, 0]
    phas = (((time - epoc) % peri) / peri + 0.25) % 1.
    indx = np.argsort(phas)
    arryfold = np.copy(arry)
    arryfold = arryfold[indx, :]
    arryfold[:, 0] = epoc + (phas[indx] - 0.25) * peri

    return arryfold


def flbn(arry, epoc, peri, numbbins, indxtimeflag=None):
    
    arryflbn = np.empty((numbbins, 3))
    indxbins = np.arange(numbbins)
    time = arry[:, 0]
    
    phas = (((time - epoc) % peri) / peri + 0.25) % 1.
    binsphas = np.linspace(0., 1., numbbins + 1)
    meanphas = (binsphas[:-1] + binsphas[1:]) / 2.
    if indxtimeflag is None:
        numbtime = time.size
        indxtimeflag = np.array([])
    
    for k in indxbins:
        indxtime = np.where((phas < binsphas[k+1]) & (phas > binsphas[k]))[0]
        indxtime = np.setdiff1d(indxtime, indxtimeflag)
        arryflbn[k, 1] = np.mean(arry[indxtime, 1])
        arryflbn[k, 2] = np.mean(arry[indxtime, 2]) / np.sqrt(indxtime.size)
        
    arryflbn[:, 0] = epoc + peri * (meanphas - 0.25)
    
    indxgood = np.where(np.isfinite(arryflbn[:, 1]))[0]
    arryflbn = arryflbn[indxgood, :]

    if not np.isfinite(arryflbn).all():
        raise Exception('')
    
    return arryflbn

    
def read_tesskplr_fold(pathfold, pathwrit, typeinst='tess', strgtype='PDCSAP_FLUX'):
    
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error
    '''

    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    listarry = []
    for path in listpath:
        arry = read_tesskplr_file(pathfold + path + '/' + path + '_lc.fits', typeinst=typeinst, strgtype=strgtype)
        listarry.append(arry)
    
    # merge sectors
    arry = np.concatenate(listarry, axis=0)
    
    # sort in time
    indxsort = np.argsort(arry[:, 0])
    arry = arry[indxsort, :]
    
    # save
    pathoutp = '%s/%s.csv' % (pathwrit, typeinst)
    np.savetxt(pathoutp, arry, delimiter=',')

    return arry 


def read_tesskplr_file(path, typeinst='tess', strgtype='PDCSAP_FLUX', boolmask=True):
    
    '''
    Reads a TESS or Kepler light curve file and returns a data cube with time, flux and flux error
    '''
    
    listhdun = fits.open(path)
    time = listhdun[1].data['TIME'] + 2457000
    if typeinst == 'TESS':
        time += 2457000
    if typeinst == 'kplr':
        time += 2454833
    
    flux = listhdun[1].data[strgtype]
    if strgtype == 'PDCSAP_FLUX':
        stdv = listhdun[1].data['PDCSAP_FLUX_ERR']
    else:
        stdv = flux * 1e-2
        
    indxgood = listhdun[1].data['QUALITY'] == 0
    if boolmask:
        # filtering
        time = time[indxgood]
        flux = flux[indxgood]
        stdv = stdv[indxgood]
    
    numbtime = time.size
    arry = np.empty((numbtime, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    arry[:, 2] = stdv
    arry = arry[~np.any(np.isnan(arry), axis=1)]
    arry[:, 2] /= np.mean(arry[:, 1])
    arry[:, 1] /= np.mean(arry[:, 1])
    
    if boolmask:
        return arry
    else:
        return arry, indxgood


def retr_fracrtsa(fracrprs, fracsars):
    
    fracrtsa = (fracrprs + 1.) / fracsars
    
    return fracrtsa


def retr_fracsars(fracrprs, fracrtsa):
    
    fracsars = (fracrprs + 1.) / fracrtsa
    
    return fracsars

