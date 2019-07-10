from __init__ import *
import os
import astroquery
import astroquery.mast
#import astroquery.mast.Observations
import allesfitter
import sys, os, h5py
from allesfitter.priors.estimate_noise_wrap import estimate_noise_wrap
from astropy.io import fits
import fnmatch, numpy as np
from tdpy.util import summgene
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

def init_alle(tici, dictsett=None, dictpara=None, strgtarg=None, pathfold=None, pathtmpt=None, liststrgdata=['SPOC'], strgalleextn=None):
    
    '''
    Creates folders, downloads TESS (SPOC or QLP) data, edits settings.csv and params.csv as required and runs allesfitter
    '''

    if pathfold is None:
        pathfold = os.environ['TESSTARG_DATA_PATH'] + '/'
    if pathtmpt is None:
        pathtmpt = os.environ['TESSTARG_DATA_PATH'] + '/tmpt/'
    if strgtarg is None:
        strgtarg = 'tici_%d' % tici

    pathtarg = pathfold + strgtarg + '/'
    
    if 'SPOC' in liststrgdata:
        
        # download data
        obsTable = astroquery.mast.Observations.query_criteria(target_name=tici, \
                                                               obs_collection='TESS', \
                                                               dataproduct_type='timeseries', \
                                               )
        listpath = []
        for k in range(len(obsTable)):
            dataProducts = astroquery.mast.Observations.get_product_list(obsTable[k])
            want = (dataProducts['obs_collection'] == 'TESS') * (dataProducts['dataproduct_type'] == 'timeseries')
            for k in range(len(dataProducts['productFilename'])):
                if not dataProducts['productFilename'][k].endswith('_lc.fits'):
                    want[k] = 0
            listpath.append(pathtarg + dataProducts[want]['productFilename'].data[0]) 
            manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathtarg)
        
        if len(obsTable) == 0:
            #print 'No TESS light curve has been found.'
            return
        else:
            print 'Found TESS SPOC data.'
    elif 'QLOP' in liststrgdata:
        print 'Reading the QLP data on the target...'
        catalogData = Catalogs.query_object(tici, catalog="TIC")
        rasc = catalogData[0]['ra']
        decl = catalogData[0]['dec']
        sector_table = Tesscut.get_sectors(SkyCoord(rasc, decl, unit="deg"))
        listsect = sector_table['sector'].data
        listcami = sector_table['camera'].data
        listccdi = sector_table['ccd'].data
        for m, sect in enumerate(listsect):
            path = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/LC/' % (listsect[m], listcami[m], strgccdi[m])
            pathqlop = path + str(tici) + '.h5'
            time, flux, stdvflux = read_qlop(pathqlop)

    # read the files to make the CSV file
    if 'SPOC' in liststrgdata:
        pathdown = pathtarg + 'mastDownload/TESS/'
        arry = read_tesskplr_fold(pathdown, pathalle)
        pathoutp = '%sTESS.csv' % pathalle
        np.savetxt(pathoutp, arry, delimiter=',')
    
    # construct target folder structure
    pathalle = pathtarg + 'allesfit'
    if strgalleextn is not None:
        pathalle += '_' + strgalleextn
    pathalle += '/'
    cmnd = 'mkdir -p %s %s' % (pathtarg, pathalle)
    print cmnd
    os.system(cmnd)
    
    sing_alle(pathalle, dictsett=dictsett, dictpara=dictpara, pathtmpt=pathtmpt)


def read_qlop(path, stdvcons=1e-3):
    
    print 'Reading QLP light curve...'
    objtfile = h5py.File(path, 'r')
    time = objtfile['LightCurve/BJD'][()] + 2457000.
    tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
    flux = 10**(-(tmag - np.median(tmag)) / 2.5)
    flux /= np.median(flux) 
    arry = np.empty((flux.size, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    print 'Assuming a constant photometric precision of %g for the QLP light curve.' % stdvcons
    stdv = np.zeros_like(flux) + stdvcons
    arry[:, 2] = stdv
   
    # filter out bad data
    indx = np.where(objtfile['LightCurve/QFLAG'][()] == 0)[0]
    print 'arry'
    summgene(arry)
    
    arry = arry[indx, :]
    
    print 'arry'
    summgene(arry)
    
    return arry


def sing_alle(pathalle, dictsett=None, dictpara=None, pathtmpt=None):
    
    '''
    Edits settings.csv and params.csv as required and runs allesfitter
    '''

    # make the changes to params.csv and settings.csv
    for a in range(2):
        if a == 0:
            strgfile = 'settings.csv'
        if a == 1:
            strgfile = 'params.csv'
        
        pathfile = pathalle + strgfile
        
        print 'Working on %s...' % strgfile
        
        if not os.path.exists(pathfile):
            # settings.csv or params.csv has not been created before. Initializing them based on the input dictsett or dictpara
            print '%s does not exist at %s%s.' % (strgfile, pathalle, strgfile)
            cmnd = 'cp %s%s %s' % (pathtmpt, strgfile, pathalle)
            print cmnd
            os.system(cmnd)
            if a == 0:
                dicttemp = dictsett
            if a == 1:
                dicttemp = dictpara
            booldoit = dicttemp is not None
        else:
            print 'Found previously existing %s%s...' % (pathalle, strgfile)
            if a == 0:
                booldoit = False 
            if a == 1:
                pathfilepost = pathalle + 'results/mcmc_table.csv'
                if os.path.exists(pathfilepost):
                    # read median values of the previous posterior and write them to the initial value of the params.csv
                    booldoit = True
                    dicttemp = {}
                    objtfilepost = open(pathfilepost, 'r')
                    listlinepost = []
                    for linepost in objtfilepost:
                        listlinepost.append(linepost)
                    objtfilepost.close()
                    objtfile = open(pathfile, 'r')
                    for line in objtfile:
                        linesplt = line.split(',')
                        for linepost in listlinepost:
                            linespltpost = linepost.split(',')
                            if linespltpost[0] == linesplt[0]:
                                dicttemp[linespltpost[0]] = [linespltpost[1]] + linesplt[2:]
                    objtfile.close()
                else:
                    booldoit = False
        
        if booldoit:
            # edit settings.csv or params.csv with those specified in dictsett or dictpara
            ## replace lines that already exist
            objtfile = open(pathfile, 'r')
            listline = []
            for line in objtfile:
                listline.append(line)
            objtfile.close()
            os.system('rm %s' % pathfile)
            objtfile = open(pathfile, 'w')
            numbstrg = len(dicttemp)
            numbline = len(listline)
            numbfond = np.zeros((numbline, numbstrg))
            for m, line in enumerate(listline):
                linesplt = line.split(',')
                for k, strgtemp in enumerate(dicttemp):
                    if linesplt[0] == strgtemp:
                        numbfond[m, k] += 1
                indx = np.where(numbfond[m, :] > 0)[0]
                if indx.size > 1:
                    raise Exception('')
                if indx.size > 0:
                    strgtemp = dicttemp.keys()[np.asscalar(indx)]
                    lineneww = ','.join([strgtemp] + dicttemp[strgtemp])# + '\n'
                else:
                    lineneww = line
                objtfile.write(lineneww)
                    
            ## add lines that do not already exist
            for k, strgtemp in enumerate(dicttemp):
                if np.sum(numbfond[:, k], 0) > 1:
                    raise Exception('')
                if np.sum(numbfond[:, k], 0) == 0:
                    objtfile.write(','.join([strgtemp] + dicttemp[strgtemp]) + '\n')
            objtfile.close()
    allesfitter.show_initial_guess(pathalle)
    allesfitter.mcmc_fit(pathalle)
    allesfitter.mcmc_output(pathalle)
    objtalle = allesfitter.allesclass(pathalle)
    
    return objtalle


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
    #for k in range(time.size):
    #    print '%d %15f %15g %10g' % (k, time[k], phas[k], arry[k, 1])
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

    
def call_alle():
    '''
    Convenience routine for calling allesfitter
    argument 1:
        plotnest -- calls allesfitter.ns_output()
        sampnest -- calls allesfitter.ns_fit()
        plotmcmc -- calls allesfitter.mcmc_output()
        sampmcmc -- calls allesfitter.mcmc_fit()
    argument 2: name of the planet
    argument 3: kind of the run
        global
        occultation
        eccentricity
    argument 4: type of the run
        wouttess
        withtess
        onlytess
        
    If only the first two arguments are provided, remakes all plots of the given planet
    '''

    if sys.argv[1] == 'plotnest':
        func = allesfitter.ns_output
    if sys.argv[1] == 'plotmcmc':
        func = allesfitter.mcmc_output
    if sys.argv[1] == 'sampprio':
        func = estimate_noise_wrap
    if sys.argv[1] == 'sampnest':
        func = allesfitter.ns_fit
    if sys.argv[1] == 'sampmcmc':
        func = allesfitter.mcmc_fit
    
    pathbase = os.environ['TESSTARG_DATA_PATH'] + '/'
    
    if sys.argv[1] == 'plot' and len(sys.argv) == 3:
        # make plotting automatic
        for strgkind in ['global', 'occulation', 'eccentricity']:
            for strgtype in ['wouttess', 'withtess', 'onlytes']:
                path = pathbase + sys.argv[2] + '/allesfit_' + strgkind + '/allesfit_' + strgtype
                if sys.argv[1].endswith('nest'):
                    path += '_ns/'
                else:
                    path += '_mc/'
                func(path)
    elif sys.argv[1] == 'viol':
        path = pathbase + sys.argv[2] + '/allesfit_' + sys.argv[3] + '/'
        allesfitter.postprocessing.plot_viol.proc_post(path)
    else:
        path = pathbase + sys.argv[2] + '/' + sys.argv[3] + '/'
        #path = pathbase + sys.argv[2] + '/allesfit_' + sys.argv[3] + '/allesfit_' + sys.argv[4]
        #if sys.argv[1].endswith('nest'):
        #    path += '_ns/'
        #else:
        #    path += '_mc/'
        func(path)


def read_tesskplr_fold(pathfold, pathwrit, typeinst='tess', strgtype='PDCSAP_FLUX'):
    
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error
    '''

    print 'pathfold'
    print pathfold
    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    print 'listpath'
    print listpath
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
