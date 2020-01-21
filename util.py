import numpy as np

import sys, os, h5py, fnmatch
import pickle

import emcee

import astropy as ap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.io import fits

import multiprocessing

import scipy as sp
import scipy.interpolate

# own modules
import tdpy.util
from tdpy.util import summgene

import astroquery
import astroquery.mast

import matplotlib.pyplot as plt



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


def retr_indxtimetran(time, epoc, peri, duramask, booloutt=False):
    
    listindxtimetran = []
    for n in range(-20000, 20000):
        timeinit = epoc + n * peri - duramask / 2.
        timefinl = epoc + n * peri + duramask / 2.
        indxtimetran = np.where((time > timeinit) & (time < timefinl))[0]
        listindxtimetran.append(indxtimetran)
    indxtimetran = np.concatenate(listindxtimetran)
    indxtimetran = np.unique(indxtimetran)
    
    if booloutt:
        indxtimeretr = np.setdiff1d(np.arange(time.size), indxtimetran)
    else:
        indxtimeretr = indxtimetran
    
    return indxtimeretr
    

def retr_timeedge(time):

    difftime = time[1:] - time[:-1]
    indxtimebrek = np.where(difftime > 1.)[0]
    timeedge = [0, np.inf]
    for k in indxtimebrek:
        timeedge.append((time[k] + time[k+1]) / 2.)
    timeedge = np.array(timeedge)
    timeedge = np.sort(timeedge)
    
    return timeedge


def detr_lcur(time, lcur, epocmask=None, perimask=None, duramask=None, verbtype=1):
    
    if verbtype > 0:
        print('Detrending the light curve...')
    
    timeedge = retr_timeedge(time)

    numbedge = len(timeedge)
    numbregi = numbedge - 1
    
    if verbtype > 0:
        print('numbregi')
        print(numbregi)
        print('timeedge')
        print(timeedge)

    indxregi = np.arange(numbregi)
    lcurdetrregi = [[] for i in indxregi]
    indxtimeregi = [[] for i in indxregi]
    indxtimeregioutt = [[] for i in indxregi]
    listobjtspln = [[] for i in indxregi]
    for i in indxregi:
        # find times inside the region
        indxtimeregi[i] = np.where((time >= timeedge[i]) & (time <= timeedge[i+1]))[0]
        timeregi = time[indxtimeregi[i]]
        lcurregi = lcur[indxtimeregi[i]]
        
        # mask out the transits
        if epocmask is not None and perimask is not None and duramask is not None:
            # find the out-of-transit times
            indxtimetran = []
            for k in range(epocmask.size):
                indxtimetran.append(retr_indxtimetran(timeregi, epocmask[k], perimask[k], duramask[k]))
            indxtimetran = np.concatenate(indxtimetran)
            indxtimeregioutt[i] = np.setdiff1d(np.arange(timeregi.size), indxtimetran)
        else:
            indxtimeregioutt[i] = np.arange(timeregi.size)
        
        # fit the spline
        objtspln = scipy.interpolate.UnivariateSpline(timeregi[indxtimeregioutt[i]], lcurregi[indxtimeregioutt[i]])
        lcurdetrregi[i] = lcurregi - objtspln(timeregi) + 1.
        listobjtspln[i] = objtspln
    
    return lcurdetrregi, indxtimeregi, indxtimeregioutt, listobjtspln


def retr_data(strgdata, strgmast, pathdata, boolsapp):
    
    # download data
    if strgdata != 'tcat':
        pathlcurspoc = pathdata + 'mastDownload/TESS/'
        if not os.path.exists(pathlcurspoc):
            print('Trying to download SPOC data with keyword: %s' % strgmast)
            listpathdown = tesstarg.util.down_spoclcur(pathdata, strgmast)
            print('listpathdown')
            print(listpathdown)
        else:
            print('SPOC folder already exists at %s. Will not attempt at downloading SPOC data...' % pathlcurspoc)
   
    # determine type of data to be used for allesfitter analysis
    if strgdata is None:
        if os.path.exists(pathlcurspoc):
            if boolsapp:
                strgdata = 'sapp'
            else:
                strgdata = 'pdcc' 
        else:
            strgdata = 'qlop'
    
    if strgdata != 'tcat':
        listpathlcur = []
        if strgdata == 'sapp' or strgdata == 'pdcc':
            listpathlcurinte = []
            for extn in os.listdir(pathlcurspoc):
                pathlcurinte = pathlcurspoc + extn + '/'
                listpathlcurinte.append(pathlcurinte)
                pathlcur = pathlcurinte + fnmatch.filter(os.listdir(pathlcurinte), '*_lc.fits')[0]
                listpathlcur.append(pathlcur)
        if strgdata == 'qlop':
            pathlcurqlop = pathdata + 'qlop/'
            print('Searching for QLP light curve(s) in %s...' % pathlcurqlop)
            os.system('mkdir -p %s' % pathlcurqlop)
            listtemp = fnmatch.filter(os.listdir(pathlcurqlop), 'sector-*')
            if len(listtemp) > 0:
                listpathlcur.extend(pathlcurqlop + listtemp)
                print('Found QLP light curves:')
                for temp in listtemp:
                    print(temp)
    else:
        print('Will run TCAT on the object...')
        tcat.main.main( \
             ticitarg=int(strgmast), \
             labltarg=strgtarg, \
             strgtarg=strgtarg, \
            )
    ## make sure the list of paths to sector files are time-sorted
    listpathlcur.sort()
    
    listpathsapp = []
    listpathpdcc = []
   
    # merge light curves from different sectors
    numbsect = len(listpathlcur)
    indxsect = np.arange(numbsect)
    listarrylcursapp = [[] for o in indxsect] 
    listarrylcurpdcc = [[] for o in indxsect] 
    listarrylcur = []
    for o, pathlcur in enumerate(listpathlcur):
        if strgdata == 'tcat':
            arrylcur = np.loadtxt(pathdata + 'band.csv', delimiter=',', skiprows=9)
        elif strgdata == 'qlop':
            arrylcur = read_qlop(pathlcur, typeinst='tess', boolmask=True)
        else:
            listarrylcursapp[o], indxtimequalgood, indxtimenanngood = read_tesskplr_file(pathlcur, typeinst='tess', strgtype='SAP_FLUX')
            listarrylcurpdcc[o], indxtimequalgood, indxtimenanngood = read_tesskplr_file(pathlcur, typeinst='tess', strgtype='PDCSAP_FLUX')
            if strgdata == 'sapp':
                arrylcur = listarrylcursapp[o]
            else:
                arrylcur = listarrylcurpdcc[o]
        listarrylcur.append(arrylcur)
    print('%d sectors of data retrieved.' % numbsect)
    arrylcur = np.concatenate(listarrylcur, 0)
    arrylcursapp = np.concatenate(listarrylcursapp, 0)
    arrylcurpdcc = np.concatenate(listarrylcurpdcc, 0)
    
    return arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc
   

def down_spoclcur(pathdownbase, strgmast, boollcuronly=True):
    
    if strgmast is None:
        raise Exception('strgmast should not be None.')

    obsTable = astroquery.mast.Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', objectname=strgmast)
    
    catalogData = astroquery.mast.Catalogs.query_object(strgmast, catalog="TIC")
    rasc = catalogData[0]['ra']
    decl = catalogData[0]['dec']
    strgtici = '%s' % catalogData[0]['ID']

    listpathdown = []
    for tabl in obsTable:
        #for keys in tabl:
        #    print('tabl')
        #    print(tabl)
        #    print('keys')
        #    print(keys)
        print('tabl[target_name]')
        print(tabl['target_name'])
        if tabl['target_name'] == '%s' % strgtici:
            dataProducts = astroquery.mast.Observations.get_product_list(tabl)
            #for strg in dataProducts:
            #    print strg
            #    print
            #print('dataProducts')
            #print(dataProducts)
            #print(dataProducts.keys())
            print('dataProducts[description]')
            #print(astropy.table.Table.read(dataProducts))
            
            print(type(dataProducts['description']))
            desc = np.array([dataProducts['description'][a] for a in range(len(dataProducts['description']))])
            print('boollcuronly')
            print(boollcuronly)
            if boollcuronly:
                want = np.where(desc == 'Light curves')[0]
            else:
                want = np.arange(len(dataProducts))
            if want.size > 0:
                print('desc')
                print(desc)
                print('want')
                print(want)
                manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathdownbase)
                pathdown = manifest['Local Path'][0]
                listpathdown.append(pathdown)
    
    if len(listpathdown) == 0:
        print('No SPOC data is found...')

    return listpathdown


def fold_lcur(arry, epoc, peri, boolxdattime=False, boolsort=True, phasshft=0.5):
    
    phas = (((arry[:, 0] - epoc) % peri) / peri + phasshft) % 1. - phasshft
    
    arryfold = np.copy(arry)
    arryfold[:, 0] = phas
    
    if boolsort:
        indx = np.argsort(phas)
        arryfold = arryfold[indx, :]
    
    if boolxdattime:
        arryfold[:, 0] *= peri

    if not np.isfinite(arryfold).all():
        raise Exception('')
    
    return arryfold


def rebn_lcur(arry, numbbins):
    
    arryrebn = np.empty((numbbins, 3)) + np.nan
    indxbins = np.arange(numbbins)
    xdat = arry[:, 0]
    binsxdat = np.linspace(np.amin(xdat), np.amax(xdat), numbbins + 1)
    meanxdat = (binsxdat[:-1] + binsxdat[1:]) / 2.
    numbxdat = meanxdat.size
    arryrebn[:, 0] = meanxdat

    for k in indxbins:
        indxxdat = np.where((xdat < binsxdat[k+1]) & (xdat > binsxdat[k]))[0]
        arryrebn[k, 1] = np.mean(arry[indxxdat, 1])
        arryrebn[k, 2] = np.std(arry[indxxdat, 1]) / np.sqrt(indxxdat.size)
    
    return arryrebn

    
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


def read_tesskplr_file(path, typeinst='tess', strgtype='PDCSAP_FLUX', boolmaskqual=True, boolmasknann=True):
    
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
    
        
    indxtimequalgood = np.where(listhdun[1].data['QUALITY'] == 0)[0]
    if boolmaskqual:
        # filtering for good quality
        time = time[indxtimequalgood]
        flux = flux[indxtimequalgood]
        stdv = stdv[indxtimequalgood]
    
    numbtime = time.size
    arry = np.empty((numbtime, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    arry[:, 2] = stdv
    
    indxtimenanngood = np.where(~np.any(np.isnan(arry), axis=1))[0]
    if boolmasknann:
        arry = arry[indxtimenanngood, :]
    
    # normalize
    arry[:, 2] /= np.mean(arry[:, 1])
    arry[:, 1] /= np.mean(arry[:, 1])
    
    return arry, indxtimequalgood, indxtimenanngood


def retr_fracrtsa(fracrprs, fracsars):
    
    fracrtsa = (fracrprs + 1.) / fracsars
    
    return fracrtsa


def retr_fracsars(fracrprs, fracrtsa):
    
    fracsars = (fracrprs + 1.) / fracrtsa
    
    return fracsars


def retr_datamock(numbplan=100, numbnois=100, numbtime=100, dept=1e-2, nois=1e-3, numbbinsphas=1000, pathplot=None, boollabltime=False, boolflbn=False):
    
    '''
    Function to generate mock light curves.
    numbplan: number of data samples containing signal, i.e., planet transits
    numbnois: number of background data samples, i.e., without planet transits
    numbtime: number of time bins
    pathplot: path in which plots are to be generated
    boollabltime: Boolean flag to label each time bin separately (e.g., for using with LSTM)
    '''
    
    # total number of data samples
    numbdata = numbplan + numbnois
    
    indxdata = np.arange(numbdata)
    indxtime = np.arange(numbtime)
    
    minmtime = 0.
    maxmtime = 27.3

    time = np.linspace(minmtime, maxmtime, numbtime)
    
    minmperi = 1.
    maxmperi = 10.
    minmdept = 1e-3
    maxmdept = 1e-2
    minmepoc = np.amin(time)
    maxmepoc = np.amax(time)
    minmdura = 0.125
    maxmdura = 0.375

    # input data
    flux = np.zeros((numbdata, numbtime))
    
    # planet transit properties
    ## durations
    duraplan = minmdura * np.ones(numbplan) + (maxmdura - minmdura) * np.random.rand(numbplan)
    ## phas  
    epocplan = minmepoc * np.ones(numbplan) + (maxmepoc - minmepoc) * np.random.rand(numbplan)
    ## periods
    periplan = minmperi * np.ones(numbplan) + (maxmperi - minmperi) * np.random.rand(numbplan)
    ##depths
    deptplan = minmdept * np.ones(numbplan) + (maxmdept - minmdept) * np.random.rand(numbplan)

    # input signal data
    fluxplan = np.zeros((numbplan, numbtime))
    indxplan = np.arange(numbplan)
    for k in indxplan:
        phas = (time - epocplan[k]) / periplan[k]
        for n in range(-1000, 1000):
            indxphastran = np.where(abs(phas - n) < duraplan[k] / periplan[k])[0]
            fluxplan[k, indxphastran] -= deptplan[k]
    
    # place the signal data
    flux[:numbplan, :] = fluxplan

    # label the data
    if boollabltime:
        outp = np.zeros((numbdata, numbtime))
        outp[np.where(flux == dept[0])] = 1.
    else:
        outp = np.zeros(numbdata)
        outp[:numbplan] = 1
    
    # add noise to all data
    flux += nois * np.random.randn(numbtime * numbdata).reshape((numbdata, numbtime))
    
    # adjust the units of periods from number of time bins to days
    peri = peri / 24. / 30.
    
    # assign random periods to non-planet data
    peritemp = np.empty(numbdata)
    peritemp[:numbplan] = peri
    peritemp[numbplan:] = 1. + np.random.rand(numbnois) * 9.
    peri = peritemp

    # phase-fold and bin
    if boolflbn:
        binsphas = np.linspace(0., 1., numbbinsphas + 1)
        fluxflbn = np.empty((numbdata, numbbinsphas))
        phas = np.empty((numbdata, numbtime))
        for k in indxdata:
            phas[k, :] = ((time - epoc[k]) / peri[k] + 0.25) % 1.
            for n in indxphas:
                indxtime = np.where((phas < binsphas[n+1]) & (phas > binsphas[n]))[0]
                fluxflbn[k, n] = np.mean(flux[k, indxtime])
        inpt = fluxflbn
        xdat = meanphas
    else:
        inpt = flux
        xdat = time

    if pathplot != None:
        # generate plots if pathplot is set
        print ('Plotting the data set...')
        for k in indxdata:
            figr, axis = plt.subplots() # figr is unused
            axis.plot(indxtime, inpt[k, :])
            axis.set_title(outp[k])
            if k < numbplan:
                for t in indxtime:
                    if (t - phas[k]) % peri[k] == 0:
                        axis.axvline(t, ls='--', alpha=0.3, color='grey')
            axis.set_xlabel('$t$')
            axis.set_ylabel('$f(t)$')
            path = pathplot + 'data_%04d.pdf' % k
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    # randomize the data set
    numpstat = np.random.get_state()
    np.random.seed(0)
    indxdatarand = np.random.choice(indxdata, size=numbdata, replace=False)
    inpt = inpt[indxdatarand, :]
    outp = outp[indxdatarand]
    peri = peri[indxdatarand]
    np.random.set_state(numpstat)

    return inpt, xdat, outp, peri, epoc


def retr_dataete6(nois=None, numbdata=None, boolnorm=True, boolplot=True, booldtrd=True):
    
    print('Retrieving TESS data...')
    os.system('mkdir -p %s' % pathdata)
    
    liststrgtype = ['plan', 'ebin', 'bebs', 'star']
    numbtype = len(liststrgtype)
    
    numbtime = 20610
    if numbdata == None:
        numbdata = 40
    
    if numbdata % 4 != 0:
        raise Exception('Number of data samples must be a multiple of 4.')

    indxdata = np.arange(numbdata)

    pathtime = pathdata + 'save_time.npy'
    pathflux = pathdata + 'save_flux.npy'
    pathlabl = pathdata + 'save_labl.npy'
    pathtici = pathdata + 'save_tici.npy'
    pathperi = pathdata + 'save_peri.npy'

    if not os.path.exists(pathtime):
        
        time = np.empty((numbdata, numbtime))
        flux = np.empty((numbdata, numbtime))
        labl = np.empty(numbdata)
        tici = np.empty(numbdata)
        peri = [[] for k in indxdata]
        
        k = 0
        for a, strgtype in enumerate(liststrgtype):
            
            pathdatatype = pathdata + strgtype
            
            if strgtype == 'plan':
                strgtypelong = 'Planets'
                strgtypeextn = 'planet'
                numblineoffs = 33
            if strgtype == 'ebin':
                strgtypelong = 'EBs'
                strgtypeextn = 'eb'
                numblineoffs = 33
            if strgtype == 'bebs':
                strgtypelong = 'BackEBs'
                strgtypeextn = 'backeb'
                numblineoffs = 38
            if strgtype == 'star':
                strgtypelong = 'Stars'
                strgtypeextn = 'star'
                numblineoffs = 20

            pathdest = pathdatatype + '/ete6_%s_data.txt' % strgtypeextn
            
            # download labels
            if not os.path.exists(pathdest):
                strgcmnd = 'wget -P %s https://archive.stsci.edu/missions/tess/ete-6/ground_truth/ete6_%s_data.txt 2>&1 /dev/null' \
                                                                                                            % (pathdatatype, strgtypeextn)
                os.system(strgcmnd)

            # parse labels
            datatrue = np.loadtxt(pathdest, skiprows=numblineoffs)

            listticitemp = fnmatch.filter(os.listdir(pathdata + 'bulk/' + strgtypelong), '*')
            listtici = []
            for ticitemp in listticitemp:
                listtici.append(ticitemp.split('_')[1].split('.')[0])
            numbtici = len(listtici)

            print ('Found %d TIC IDs' % numbtici)

            for l, strgtici in enumerate(listtici):
                
                if l == numbdata / 4:
                    break
                #strgtici = str(strgtici).zfill(16)
                #namefile = 'tess2019128220341-' + strgtici + '-0016-s_lc.fits'
                #strginte = strgtici[0:2] + '/' + strgtici[2:5] + '/' + strgtici[5:8] + '/' + strgtici[8:11]
                #pathdestfile = pathdatatype + '/ete6_%s_data.txt' % strgtypeextn
                #
                ## download file
                #if not os.path.exists(pathdestfile):
                #    strgcmnd = 'wget -P ' + pathdatatype + ' http://archive.stsci.edu/missions/tess/ete-6/tid/' + strginte + '/' + namefile + ' 2>&1 /dev/null'
                #    os.system(strgcmnd)

                
                #pathdestfile = pathdatatype + '/' + namefile
                #listhdun = ap.io.fits.open(pathdestfile)
                #listhdun.info()
                #time[k, :] = listhdun['LIGHTCURVE'].data['TIME']
                #flux[k, :] = listhdun['LIGHTCURVE'].data['SAP_FLUX']
                #time[k, :] = np.arange([0]listhdun['LIGHTCURVE'].data['TIME']
                
                # parse bulk download file
                pathdestfile = pathdata + 'bulk/' + strgtypelong + '/' + strgtypelong + '_' + strgtici + '.txt'
                data = np.loadtxt(pathdestfile)
                
                flux[k, :] = data
                labl[k] = a
                tici[k] = int(strgtici)
                
                indxdatathis = np.where(datatrue[:, 0] == tici[k])[0]
                if a != 3:
                    peri[k] = datatrue[indxdatathis, 9]
                else:
                    peri[k] = [0.]
                
                if peri[k][0] == 0.:
                    peri[k] = -3. - np.random.rand(1) * 5.
                
                print (peri[k][0])
                
                k += 1

        print ('Writing to %s...' % pathtime)
        np.save(pathtime, time)
        print ('Writing to %s...' % pathflux)
        np.save(pathflux, flux)
        print ('Writing to %s...' % pathlabl)
        np.save(pathlabl, labl)
        print ('Writing to %s...' % pathtici)
        np.save(pathtici, tici)
        print ('Writing to %s...' % pathperi)
        np.save(pathperi, peri)
    else:
        
        print ('Reading from %s...' % pathtime)
        time = np.load(pathtime)
        print ('Reading from %s...' % pathflux)
        flux = np.load(pathflux)
        print ('Reading from %s...' % pathlabl)
        labl = np.load(pathlabl)
        print ('Reading from %s...' % pathtici)
        tici = np.load(pathtici)
        print ('Reading from %s...' % pathperi)
        peri = np.load(pathperi)
    
    if numbdata < time.shape[0]:
        time = time[:numbdata, :]
        flux = flux[:numbdata, :]
        labl = labl[:numbdata]
        tici = tici[:numbdata]
        peri = peri[:numbdata]

    if nois is not None:
        print ('Adding noise...')
        flux += nois * np.random.randn(numbtime * numbdata).reshape((numbdata, numbtime))
   
    if boolnorm:
        print ('Normalizing...')
        for k in indxdata:
            flux[k, :] /= np.mean(flux[k, :])
    
    time = np.tile(np.arange(numbtime), (numbdata, 1))
    if boolplot:
        figr, axis = plt.subplots()
        for k in range(numbdata):
            if labl[k] == 0:
                colr = 'blue'
            if labl[k] == 1:
                colr = 'g'
            if labl[k] == 2:
                colr = 'r'
            if labl[k] == 3:
                colr = 'y'
            axis.plot(time[0, :], flux[k, :], alpha=0.5, color=colr)
        pathplot = pathdata + 'lcur.pdf'
        print ('Writing to %s...' % pathplot)
        plt.savefig(pathplot)
        plt.close()
    
    indxrand = np.random.random_integers(numbdata)
    
    if booldtrd:
        fluxbsln = np.empty_like(flux)
        for k in indxdata:
            timeknotmaxm = np.amax(time[k, :] - 1)
            timeknotminm = np.amin(time[k, :] + 1)
            timeknot = np.arange(timeknotminm, timeknotmaxm, 30.)
            fluxbsln[k, :] = sp.interpolate.LSQUnivariateSpline(time[k, :], flux[k, :], timeknot)(time[k, :])
    
    flux = flux - fluxbsln

    return time, flux, labl, tici, peri


def retr_datatess(boolflbn=True, boolplot=True):
    
    print('Retrieving TESS data...')

    pathdata = os.environ['EXOP_DATA_PATH'] + '/tess/original_data/'
    pathdatalcurflbn = os.environ['EXOP_DATA_PATH'] + '/tess/plot/lcurflbn/'
    os.system('mkdir -p %s' % pathdatalcurflbn)
    
    numbphasnann = 1000
    
    pathsave = os.environ['EXOP_DATA_PATH'] + '/tess/save_tess.pickle'
    #if True or not os.path.exists(pathsave):
    if not os.path.exists(pathsave):
        
        listfiledval = fnmatch.filter(os.listdir(pathdata), '*_dvt.fits')
        pathcsvf = pathdata + 'tess_tce_sector1_2.txt'
        objtfile = open(pathcsvf)
        liststrgtcee = []
        listlabl = []
        listdisp = []
        listtici = []
        
        listphas = []
        listflux = []
        listlegd = []
        listitoi = []
        cntrtcee = 0
        cntrtceefoun = 0
        cntrtceespoc = 0
        cntrtceeuniq = 0
        listticiused = []
        liststrgitoiused = []
        for k, line in enumerate(objtfile):
            
            if k == 0:
                continue
            
            # temp
            #if k > 1000:
            #    break
            
            cntrtcee += 1

            linesplt = line.split('\t')
            tici = linesplt[0]
            strgitoi = linesplt[1].split('.')[1]
            disp = linesplt[2]
            strgtcee = '%s_%s' % (tici, strgitoi)
            
            legd = '%s, %s' % (strgtcee, disp)

            if linesplt[-5] != 'spoc':
                continue
    
            cntrtceespoc += 1

            if int(linesplt[0]) in listticiused and strgitoi in liststrgitoiused:
                continue
    
            cntrtceeuniq += 1

            indxhdun = int(strgitoi)
            boolfoun = False
            boolgood = False
            try:
                path = pathdata + 'tess2018206190142-s0001-s0001-%016d-00106_dvt.fits' % int(tici)
                listhdun = fits.open(path)
                boolfoun = True
                
                if indxhdun >= len(listhdun) - 1:
                    raise Exception('')

                boolgood = True
            except:
                pass
                if boolfoun:
                    print('There are not enough HDUs in the SPOC file for TIC ID %d...' % int(tici))
                else:
                    pass
                    #print 'Could not find TIC ID %d...' % int(tici)
            
            if boolgood:
                
                listtici.append(int(tici))
                liststrgtcee.append(strgtcee)
                listdisp.append(disp)
                if disp == 'PC' or disp == 'P' or disp == 'KP':
                    listlabl.append(1)
                else:
                    listlabl.append(0)
                listlegd.append(legd) 
                liststrgitoiused.append(strgitoi)
                #for keys in listhdun[2].data:
                #    print(keys)
                #listhdun[-1].data['PDCSAP_FLUX']
                # temp

                phasraww = ((listhdun[indxhdun].data['PHASE'] - listhdun[indxhdun].header['TEPOCH']) % listhdun[indxhdun].header['TPERIOD'] + 0.25) % 1.
                fluxraww = listhdun[indxhdun].data['LC_INIT']
                
                numbbins = 100
                indxbins = np.arange(numbbins)
                phasbins = np.linspace(0., 1., numbbins + 1)
                phas = (phasbins[1:] + phasbins[:-1]) / 2.
                flux = np.empty_like(phas)
                booltrsh = False
                for j in indxbins:
                    indx = np.where((phasraww < phasbins[j+1]) & (phasraww > phasbins[j]))[0]
                    flux[j] = np.mean(fluxraww[indx])
                    if indx.size == 0:
                        booltrsh = True
                

                if cntrtceefoun != 0 and (flux - listflux[-1] == 0).all():
                    raise Exception('Two phase curves are the same!')
    
                indxnann = np.where(~np.isfinite(flux))[0]
                for n in indxnann:
                    indxminm = max(0, n - numbphasnann)
                    indxmaxm = min(flux.size, n + numbphasnann)
                    flux[n] = np.nanmedian(flux[indxminm:indxmaxm])
                indxnann = np.where(~np.isfinite(flux))[0]

                cntrtceefoun += 1
                listphas.append(np.copy(phas))
                listflux.append(np.copy(flux))
                listitoi.append(indxhdun)
                listticiused.append(int(tici))

                listhdun.close()
        
            print
            print
        
        listticiused = np.array(listticiused)
        listticiused = np.unique(listticiused)

        numbobjt = listticiused.size

        numbtici = len(listtici)
        numbticiused = len(listticiused)
        indxtici = np.arange(numbtici)
        
        listlabl = np.array(listlabl)
        listphas = np.vstack(listphas)
        listflux = np.vstack(listflux)
        
        listdata = [listphas, listflux, listlabl, listlegd, listtici, listitoi]

        print('Writing to %s...' % pathsave)
        objtfile = open(pathsave, 'wb')
        pickle.dump(listdata, objtfile, protocol=pickle.HIGHEST_PROTOCOL)
        objtfile.close()
    
        assert len(listlabl) > 0
    
    else:
        objtfile = open(pathsave, 'r')
        print('Reading from %s...' % pathsave)
        listdata = pickle.load(objtfile)
        objtfile.close()
    
    if boolplot:
        
        numbplotfram = 1

        listphas = listdata[0]
        listflux = listdata[1]
        listlabl = listdata[2]
        listlegd = listdata[3]
        listtici = listdata[4]
        listitoi = listdata[5]

        cntr = 0
        for k in range(len(listdata[0])):
            
            if k % numbplotfram == 0:
                figr, axis = plt.subplots(figsize=(12, 6))

            axis.scatter(listphas[k], listflux[k], s=2)
            axis.set_title(listlegd[k])
            path = pathdatalcurflbn + 'lcurflbn_012%d_%d.png' % (listtici[k], listitoi[k])
            if (k % numbplotfram == numbplotfram - 1 or k == len(listdata[0]) - 1) and not os.path.exists(path):
                axis.set_xlabel('Phase')
                axis.set_ylabel('$\Delta f$')
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                cntr += 1
        cmnd = 'convert %stran_*.png tran.gif' % pathdata
        # temp
        #os.system(cmnd)
    
    listtemp = np.where(~np.isfinite(listdata[1]))
    indxphasbadd = listtemp[1]
    indxdatabadd = listtemp[0]

    return listdata


def retr_limtpara(scalpara, minmpara, maxmpara, meanpara, stdvpara):
    
    numbpara = scalpara.size
    limtpara = np.empty((2, numbpara))
    indxpara = np.arange(numbpara)
    for n in indxpara:
        if scalpara[n] == 'self':
            limtpara[0, n] = minmpara[n]
            limtpara[1, n] = maxmpara[n]
        if scalpara[n] == 'gaus':
            limtpara[0, n] = meanpara[n] - 10 * stdvpara[n]
            limtpara[1, n] = meanpara[n] + 10 * stdvpara[n]
    
    return limtpara


def samp(gdat, pathimag, numbsampwalk, numbsampburnwalk, retr_modl, retr_lpos, listlablpara, scalpara, \
                                                    minmpara, maxmpara, meanpara, stdvpara, numbdata, diagmode=True, strgmodl=None):
        
    if strgmodl is None:
        strgmodl = ''
    else:
        strgmodl = '_' + strgmodl

    numbpara = len(listlablpara)
    indxpara = np.arange(numbpara)
    numbdoff = numbdata - numbpara
    
    # plotting
    ## plot limits 
    limtpara = retr_limtpara(scalpara, minmpara, maxmpara, meanpara, stdvpara)

    ## plot bins
    numbbins = 20
    indxbins = np.arange(numbbins)
    binspara = np.empty((numbbins + 1, numbpara)) 
    for k in indxpara:
        binspara[:, k] = np.linspace(limtpara[0, k], limtpara[1, k], numbbins + 1)
    meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
    dictllik = [gdat]
    
    # initialize
    numbwalk = 2 * numbpara
    indxwalk = np.arange(numbwalk)
    parainit = [[] for k in indxwalk]
    meanparainit = (limtpara[0, :] + limtpara[1, :]) / 2.
    stdvnorm = (limtpara[1, :] - limtpara[0, :]) / 100.
    for k in indxwalk:
        print('limtpara')
        print(limtpara)
        print('meanparainit')
        print(meanparainit)
        print('stdvnorm')
        print(stdvnorm)
        parainit[k]  = (scipy.stats.truncnorm.rvs((limtpara[0, :] - meanparainit) / stdvnorm, \
                                                            (limtpara[1, :] - meanparainit) / stdvnorm)) * stdvnorm + meanparainit

    numbsamp = numbsampwalk * numbwalk
    indxsamp = np.arange(numbsamp)
    numbsampburn = numbsampburnwalk * numbwalk
    if diagmode:
        if numbsampwalk == 0:
            raise Exception('')
    initindxtranmodl, inittimetranmodl, initindxtranmodlproj, inittimetranmodlproj = retr_modl(gdat, meanparainit)
    listvarb = [[initindxtranmodl], [inittimetranmodl], [initindxtranmodlproj], [inittimetranmodlproj]]
    # temp
    #objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik, pool=multiprocessing.Pool())
    objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik)
    if numbsampburnwalk > 0:
        parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampburnwalk, progress=True)
        objtsamp.reset()
    else:
        parainitburn = parainit
    objtsamp.run_mcmc(parainitburn, numbsampwalk, progress=True)
    objtsave = objtsamp
    
    parapost = objtsave.flatchain
    indxsampwalk = np.arange(numbsampwalk)

    if numbsamp != numbsampwalk * numbwalk:
        raise Exception('')

    listsamp = objtsave.flatchain
    listllik = objtsave.flatlnprobability
    
    listlpos = objtsave.lnprobability
    chi2 = -2. * listlpos
    
    # plot the posterior
    ## parameter
    ### trace
    figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
    print('numbwalk')
    print(numbwalk)
    print('objtsave.lnprobability')
    summgene(objtsave.lnprobability)
    print('indxsampwalk')
    summgene(indxsampwalk)
    for i in indxwalk:
        axis[0].plot(indxsampwalk, objtsave.lnprobability[i, :])
    axis[0].set_ylabel('logL')
    listlablparafull = []
    for k in indxpara:
        for i in indxwalk:
            axis[k+1].plot(indxsampwalk, objtsave.chain[i, :, k])
        labl = listlablpara[k][0]
        if listlablpara[k][1] != '':
            labl += ' [%s]' % listlablpara[k][1]
        listlablparafull.append(labl)
        axis[k+1].set_ylabel(labl)
    path = pathimag + 'trac%s.png' % (strgmodl)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
        
    indxsampmlik = np.argmax(listllik)
    listparamlik = listsamp[indxsampmlik, :]
    #print('Saving the maximum likelihood to %s...' % pathmlik)
    #np.savetxt(pathmlik, listparamlik, delimiter=',')
    
    strgplot = 'post' + strgmodl
    print('path')
    print(path)
    tdpy.mcmc.plot_grid(pathimag, strgplot, listsamp, listlablparafull, listvarbdraw=[meanpara.flatten()], numbbinsplot=numbbins)
    
    print('Minimum chi2: ')
    print(np.amin(chi2))
    print('Minimum chi2 per dof: ')
    print(np.amin(chi2) / numbdoff)
    print('Maximum aposterior: ')
    print(np.amax(listlpos))
    
    return parapost



