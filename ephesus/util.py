import sys
import os

import numpy as np

from tqdm import tqdm

import json

import time as timemodu

from numba import jit, prange
import h5py
import fnmatch

import pandas as pd

import astropy
import astropy as ap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.timeseries

import celerite

import multiprocessing
from functools import partial

import scipy as sp
import scipy.interpolate

import astroquery
import astroquery.mast

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

# own modules
import tdpy
from tdpy import summgene


def quer_mast(request):

    from urllib.parse import quote as urlencode
    import http.client as httplib 

    server='mast.stsci.edu'

    # Grab Python Version
    version = '.'.join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {'Content-type': 'application/x-www-form-urlencoded',
               'Accept': 'text/plain',
               'User-agent':'python-requests/'+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request('POST', '/api/v0/invoke', 'request='+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content


def xmat_tici(listtici):
    
    if len(listtici) == 0:
        raise Exception('')
    
    # make sure the input is a python list of strings
    if isinstance(listtici[0], str):
        if isinstance(listtici, np.ndarray):
            listtici = list(listtici)
    else:
        if isinstance(listtici, list):
            listtici = np.array(listtici)
        if isinstance(listtici, np.ndarray):
            listtici = listtici.astype(str)
        listtici = list(listtici)

    request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':'rad, mass', \
                                                                        'filters':[{'paramName':'ID', 'values':listtici}]}}
    headers, outString = quer_mast(request)
    dictquer = json.loads(outString)['data']
    
    return dictquer


def retr_dictpopltic8(typepopl, numbsyst=None, typeverb=1):
    '''
    Get a dictionary of the sources in the TIC8 with the fields in the TIC8.
    
    Keyword arguments   
        typepopl: type of the population
            'tessprms2min': 2-minute TESS targets obtained by merging the SPOC 2-min bulk downloads
            'ticiprmsffimhcon': TESS targets with contamination larger than
            'ticiprmsffimm060': TESS targets brighter than mag 6.0
            'ticiprmsffimm100': TESS targets brighter than mag 10.0
            'ticiprmsffimm140': TESS targets brighter than mag 14.0

    Returns a dictionary with keys:
        rasc: RA
        decl: declination
        tmag: TESS magnitude
        radistar: radius of the star
        massstar: mass of the star
    '''
    
    if typeverb > 0:
        print('Retrieving a dictionary of TIC8 for population %s...' % typepopl)
    
    if typepopl.startswith('tess') or typepopl.startswith('tici'):
        if typepopl[4:].startswith('yr01'):
            listtsec = np.arange(1, 14) # [1-13]
        elif typepopl[4:].startswith('yr02'):
            listtsec = np.arange(13, 27) # [13-26]
        elif typepopl[4:].startswith('yr03'):
            listtsec = np.arange(27, 40) # [27-39]
        elif typepopl[4:].startswith('yr04'):
            listtsec = np.arange(40, 56) # [40-55]
        elif typepopl[4:].startswith('yr04'):
            listtsec = np.arange(56, 70) # [56-69]
        elif typepopl[4:].startswith('prms'):
            listtsec = np.arange(1, 27)
        elif typepopl[4:].endswith('e1ms'):
            listtsec = np.arange(27, 56)
        elif typepopl[4:].endswith('e2ms'):
            listtsec = np.arange(56, 70)
        else:
            raise Exception('')
        numbtsec = len(listtsec)
        indxtsec = np.arange(numbtsec)

    pathlistticidata = os.environ['EPHESUS_DATA_PATH'] + '/data/listticidata/'
    os.system('mkdir -p %s' % pathlistticidata)

    path = pathlistticidata + 'listticidata_%s.csv' % typepopl
    if not os.path.exists(path):
        
        # dictionary of strings that will be keys of the output dictionary
        dictstrg = dict()
        dictstrg['ID'] = 'tici'
        dictstrg['ra'] = 'rascstar'
        dictstrg['dec'] = 'declstar'
        dictstrg['Tmag'] = 'tmag'
        dictstrg['rad'] = 'radistar'
        dictstrg['mass'] = 'massstar'
        dictstrg['Teff'] = 'tmptstar'
        dictstrg['logg'] = 'loggstar'
        dictstrg['MH'] = 'metastar'
        liststrg = list(dictstrg.keys())
        
        print('typepopl')
        print(typepopl)
        if typepopl.startswith('tessprms'):
            
            if typepopl[8:12] == '20sc':
                strgurll = '_20s_'
                labltemp = '20-second'
            if typepopl[8:12] == '2min':
                strgurll = '_'
                labltemp = '2-minute'

            dictquer = dict()
            listtici = []
            for o in indxtsec:
                if typepopl.endswith('bulk'):
                    pathtess = os.environ['TESS_DATA_PATH'] + '/data/lcur/sector-%02d' % listtsec[o]
                    listnamefile = fnmatch.filter(os.listdir(pathtess), '*.fits')
                    listticitsec = []
                    for namefile in listnamefile:
                        listticitsec.append(str(int(namefile.split('-')[2])))
                    listticitsec = np.array(listticitsec)
                else:
                    url = 'https://tess.mit.edu/wp-content/uploads/all_targets%sS%03d_v1.csv' % (strgurll, listtsec[o])
                    c = pd.read_csv(url, header=5)
                    listticitsec = c['TICID'].values
                    listticitsec = listticitsec.astype(str)
                numbtargtsec = listticitsec.size
                
                if typeverb > 0:
                    print('%d observed %s targets in Sector %d...' % (numbtargtsec, labltemp, listtsec[o]))
                
                if numbtargtsec > 0:
                    dictquertemp = xmat_tici(listticitsec)
                
                if o == 0:
                    dictquerinte = dict()
                    for name in dictstrg.keys():
                        dictquerinte[dictstrg[name]] = [[] for o in indxtsec]
                
                for name in dictstrg.keys():
                    for k in range(len(dictquertemp)):
                        dictquerinte[dictstrg[name]][o].append(dictquertemp[k][name])

            print('Concatenating arrays from different sectors...')
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = np.concatenate(dictquerinte[dictstrg[name]])
            
            u, indxuniq, cnts = np.unique(dictquer['tici'], return_index=True, return_counts=True)
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = dictquer[dictstrg[name]][indxuniq]
            dictquer['numbtsec'] = cnts

        elif typepopl.startswith('tici'):
            if typepopl.endswith('hcon'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass, contratio', \
                                                             'filters':[{'paramName':'contratio', 'values':[{"min":10., "max":1e3}]}]}}
            if typepopl.endswith('m060'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":6.0}]}]}}
            if typepopl.endswith('m100'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":10.0}]}]}}
            if typepopl.endswith('m140'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":14.0}]}]}}
            headers, outString = quer_mast(request)
            listdictquer = json.loads(outString)['data']
            if typeverb > 0:
                print('%d matches...' % len(listdictquer))
            dictquer = dict()
            for name in listdictquer[0].keys():
                if name == 'ID':
                    namedict = 'tici'
                if name == 'Tmag':
                    namedict = 'tmag'
                if name == 'ra':
                    namedict = 'rascstar'
                if name == 'dec':
                    namedict = 'declstar'
                if name == 'rad':
                    namedict = 'radistar'
                if name == 'mass':
                    namedict = 'massstar'
                dictquer[namedict] = np.empty(len(listdictquer))
                for k in range(len(listdictquer)):
                    dictquer[namedict][k] = listdictquer[k][name]
        else:
            print('Unrecognized population name: %s' % typepopl)
            raise Exception('')
        
        numbtarg = dictquer['radistar'].size
            
        if typeverb > 0:
            print('%d targets...' % numbtarg)
            print('Writing to %s...' % path)
        #columns = ['tici', 'radi', 'mass']
        pd.DataFrame.from_dict(dictquer).to_csv(path, index=False)#, columns=columns)
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        dictquer = pd.read_csv(path, nrows=numbsyst).to_dict(orient='list')
        
        for name in dictquer.keys():
            dictquer[name] = np.array(dictquer[name])

    #if gdat.typedata == 'simuinje':
    #    indx = np.where((~np.isfinite(gdat.dictfeat['true']['ssys']['massstar'])) | (~np.isfinite(gdat.dictfeat['true']['ssys']['radistar'])))[0]
    #    gdat.dictfeat['true']['ssys']['radistar'][indx] = 1.
    #    gdat.dictfeat['true']['ssys']['massstar'][indx] = 1.
    #    gdat.dictfeat['true']['totl']['tmag'] = dicttic8['tmag']
        
    return dictquer


def retr_objtlinefade(x, y, colr='black', initalph=1., alphfinl=0.):
    
    colr = get_color(colr)
    cdict = {'red':   ((0.,colr[0],colr[0]),(1.,colr[0],colr[0])),
             'green': ((0.,colr[1],colr[1]),(1.,colr[1],colr[1])),
             'blue':  ((0.,colr[2],colr[2]),(1.,colr[2],colr[2])),
             'alpha': ((0.,initalph, initalph), (1., alphfinl, alphfinl))}
    
    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")
   
    segments = np.zeros((Npts-1,2,2))
    segments[0][0] = [x[0], y[0]]
    for i in range(1,Npts-1):
        pt = [x[i], y[i]]
        segments[i-1][1] = pt
        segments[i][0] = pt 
    segments[-1][1] = [x[-1], y[-1]]

    individual_cm = mpl.colors.LinearSegmentedColormap('indv1', cdict)
    lc = mpl.collections.LineCollection(segments, cmap=individual_cm)
    lc.set_array(np.linspace(0.,1.,len(segments)))
    
    return lc


def retr_liststrgcomp(numbcomp):
    
    liststrgcomp = np.array(['b', 'c', 'd', 'e', 'f', 'g'])[:numbcomp]

    return liststrgcomp


def retr_listcolrcomp(numbcomp):
    
    listcolrcomp = np.array(['magenta', 'orange', 'red', 'green', 'purple', 'cyan'])[:numbcomp]

    return listcolrcomp


def plot_orbt( \
              # path to write the plot
              path, \
              # radius of the planets [R_E]
              radicomp, \
              # sum of radius of planet and star divided by the semi-major axis
              rsmacomp, \
              # epoc of the planets [BJD]
              epoc, \
              # orbital periods of the planets [days]
              peri, \
              # cosine of the inclination
              cosi, \
              # type of visualization: 
              ## 'realblac': dark background, black planets
              ## 'realblaclcur': dark backgound, luminous planets, with light curves 
              ## 'realcolrlcur': dark background, colored planets, with light curves 
              ## 'cartcolr': bright background, colored planets
              typevisu, \
              
              # radius of the star [R_S]
              radistar=1., \
              # mass of the star [M_S]
              massstar=1., \
              # Boolean flag to produce an animation
              boolanim=False, \

              # angle of view with respect to the orbital plane [deg]
              anglpers=5., \

              # size of the figure
              sizefigr=(8, 8), \
              listcolrcomp=None, \
              liststrgcomp=None, \
              boolsingside=True, \
              ## file type of the plot
              typefileplot='pdf', \

              # verbosity level
              typeverb=1, \
             ):

    dictfact = retr_factconv()
    
    mpl.use('Agg')

    numbcomp = len(radicomp)
    
    if isinstance(radicomp, list):
        radicomp = np.array(radicomp)

    if isinstance(rsmacomp, list):
        rsmacomp = np.array(rsmacomp)

    if isinstance(epoc, list):
        epoc = np.array(epoc)

    if isinstance(peri, list):
        peri = np.array(pericomp)

    if isinstance(cosi, list):
        cosi = np.array(cosi)

    if listcolrcomp is None:
        listcolrcomp = retr_listcolrcomp(numbcomp)

    if liststrgcomp is None:
        liststrgcomp = retr_liststrgcomp(numbcomp)
    
    # semi-major axes of the planets [AU]
    smax = (radicomp / dictfact['rsre'] + radistar) / dictfact['aurs'] / rsmacomp
    indxcomp = np.arange(numbcomp)
    
    # perspective factor
    factpers = np.sin(anglpers * np.pi / 180.)

    ## scale factor for the star
    factstar = 5.
    
    ## scale factor for the planets
    factplan = 20.
    
    # maximum y-axis value
    maxmyaxi = 0.05

    if typevisu == 'cartmerc':
        # Mercury
        smaxmerc = 0.387 # [AU]
        radicompmerc = 0.3829 # [R_E]
    
    # scaled radius of the star [AU]
    radistarscal = radistar / dictfact['aurs'] * factstar
    
    time = np.arange(0., 30., 2. / 60. / 24.)
    numbtime = time.size
    indxtime = np.arange(numbtime)
   
    if boolanim:
        numbiter = min(500, numbtime)
    else:
        numbiter = 1
    indxiter = np.arange(numbiter)
    
    xposmaxm = smax
    yposmaxm = factpers * xposmaxm
    numbtimequad = 10
    
    if typevisu == 'realblaclcur':
        numbtimespan = 100

    # get transit model based on TESS ephemerides
    rratcomp = radicomp / radistar
    
    rflxtranmodl = retr_rflxtranmodl(time, pericomp=peri, epocmtracomp=epoc, rsmacomp=rsmacomp, cosicomp=cosi, rratcomp=rratcomp)['rflx'] - 1.
    
    lcur = rflxtranmodl + np.random.randn(numbtime) * 1e-6
    ylimrflx = [np.amin(lcur), np.amax(lcur)]
    
    phas = np.random.rand(numbcomp)[None, :] * 2. * np.pi + 2. * np.pi * time[:, None] / peri[None, :]
    yposelli = yposmaxm[None, :] * np.sin(phas)
    xposelli = xposmaxm[None, :] * np.cos(phas)
    
    # time indices for iterations
    indxtimeiter = np.linspace(0., numbtime - numbtime / numbiter, numbiter).astype(int)
    
    if typevisu.startswith('cart'):
        colrstar = 'k'
        colrface = 'w'
        plt.style.use('default')
    else:
        colrface = 'k'
        colrstar = 'w'
        plt.style.use('dark_background')
    
    if boolanim:
        gdat.cmndmakeanim = 'convert -delay 5'
        listpathtemp = []
    for k in indxiter:
        
        if typevisu == 'realblaclcur':
            numbrows = 2
        else:
            numbrows = 1
        figr, axis = plt.subplots(figsize=sizefigr)

        ### lower half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 180, 360, fc=colrstar, zorder=1, edgecolor=colrstar)
        axis.add_artist(w1)
        
        for jj, j in enumerate(indxcomp):
            xposellishft = np.roll(xposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
            yposellishft = np.roll(yposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
        
            # trailing lines
            if typevisu.startswith('cart'):
                objt = retr_objtlinefade(xposellishft, yposellishft, colr=listcolrcomp[j], initalph=1., alphfinl=0.)
                axis.add_collection(objt)
            
            # add planets
            if typevisu.startswith('cart'):
                colrplan = listcolrcomp[j]
                # add planet labels
                axis.text(.6 + 0.03 * jj, 0.1, liststrgcomp[j], color=listcolrcomp[j], transform=axis.transAxes)
        
            if typevisu.startswith('real'):
                if typevisu == 'realillu':
                    colrplan = 'k'
                else:
                    colrplan = 'black'
            radi = radicomp[j] / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((xposelli[indxtimeiter[k], j], yposelli[indxtimeiter[k], j], 0), radius=radi, color=colrplan, zorder=3)
            axis.add_artist(w1)
            
        ## upper half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 0, 180, fc=colrstar, zorder=4, edgecolor=colrstar)
        axis.add_artist(w1)
        
        if typevisu == 'cartmerc':
            ## add Mercury
            axis.text(.387, 0.01, 'Mercury', color='grey', ha='right')
            radi = radicompmerc / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((smaxmerc, 0), radius=radi, color='grey')
            axis.add_artist(w1)
        
        # temperature axis
        #axistwin = axis.twiny()
        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        
        # temperature contours
        #for tmpt in [500., 700,]:
        #    smaj = tmpt
        #    axis.axvline(smaj, ls='--')
        
        axis.get_yaxis().set_visible(False)
        axis.set_aspect('equal')
        
        if typevisu == 'cartmerc':
            maxmxaxi = max(1.2 * np.amax(smaxcomp), 0.4)
        else:
            maxmxaxi = 1.2 * np.amax(smaxcomp)
        
        if boolsingside:
            minmxaxi = 0.
        else:
            minmxaxi = -maxmxaxi

        axis.set_xlim([minmxaxi, maxmxaxi])
        axis.set_ylim([-maxmyaxi, maxmyaxi])
        axis.set_xlabel('Distance from the star [AU]')
        
        if typevisu == 'realblaclcur':
            print('indxtimeiter[k]')
            print(indxtimeiter[k])
            minmindxtime = max(0, indxtimeiter[k]-numbtimespan)
            print('minmindxtime')
            print(minmindxtime)
            xtmp = time[minmindxtime:indxtimeiter[k]]
            if len(xtmp) == 0:
                continue
            print('xtmp')
            print(xtmp)
            timescal = 2 * maxmxaxi * (xtmp - np.amin(xtmp)) / (np.amax(xtmp) - np.amin(xtmp)) - maxmxaxi
            print('timescal')
            print(timescal)
            axis.scatter(timescal, 10000. * lcur[minmindxtime:indxtimeiter[k]] + maxmyaxi * 0.8, rasterized=True, color='cyan', s=0.5)
            print('time[minmindxtime:indxtimeiter[k]]')
            summgene(time[minmindxtime:indxtimeiter[k]])
            print('lcur[minmindxtime:indxtimeiter[k]]')
            summgene(lcur[minmindxtime:indxtimeiter[k]])
            print('')

        #plt.subplots_adjust()
        #axis.legend()
        
        if boolanim:
            pathtemp = '%s_%s_%04d.%s' % (path, typevisu, k, gdat.typefileplot)
        else:
            pathtemp = '%s_%s.%s' % (path, typevisu, gdat.typefileplot)
        print('Writing to %s...' % pathtemp)
        plt.savefig(pathtemp)
        plt.close()
        
        if boolanim:
            listpathtemp.append(pathtemp)
            gdat.cmndmakeanim += ' %s' % pathtemp 
    if boolanim:
        gdat.cmndmakeanim += ' %s_%s.gif' % (path, typevisu)
        #os.system(gdat.cmndmakeanim)
        for pathtemp in listpathtemp:
            gdat.cmnddeleimag = 'rm %s' % pathtemp
            os.system(gdat.cmnddeleimag)


def retr_dictpoplrvel():
    
    if typeverb > 0:
        print('Reading Sauls Gaia high RV catalog...')
    path = os.environ['TROIA_DATA_PATH'] + '/data/Gaia_high_RV_errors.txt'
    for line in open(path):
        listnamesaul = line[:-1].split('\t')
        break
    if typeverb > 0:
        print('Reading from %s...' % path)
    data = np.loadtxt(path, skiprows=1)
    dictcatl = dict()
    dictcatl['rascstar'] = data[:, 0]
    dictcatl['declstar'] = data[:, 1]
    dictcatl['stdvrvel'] = data[:, -4]
    
    return dictcatl


def retr_dicthostplan(namepopl, typeverb=1):
    
    pathlygo = os.environ['EPHESUS_DATA_PATH'] + '/'
    path = pathlygo + 'data/dicthost%s.csv' % namepopl
    if os.path.exists(path):
        if typeverb > 0:
            print('Reading from %s...' % path)
        dicthost = pd.read_csv(path).to_dict(orient='list')
        for name in dicthost.keys():
            dicthost[name] = np.array(dicthost[name])
        
    else:
        dicthost = dict()
        if namepopl == 'toii':
            dictplan = retr_dicttoii()
        else:
            dictplan = retr_dictexar()
        listnamestar = np.unique(dictplan['namestar'])
        dicthost['namestar'] = listnamestar
        numbstar = listnamestar.size
        indxstar = np.arange(numbstar)
        
        listnamefeatstar = ['numbplanstar', 'numbplantranstar', 'radistar', 'massstar']
        listnamefeatcomp = ['epocmtracomp', 'pericomp', 'duratrantotl', 'radicomp', 'masscomp']
        for namefeat in listnamefeatstar:
            dicthost[namefeat] = np.empty(numbstar)
        for namefeat in listnamefeatcomp:
            dicthost[namefeat] = [[] for k in indxstar]
        for k in indxstar:
            indx = np.where(dictplan['namestar'] == listnamestar[k])[0]
            for namefeat in listnamefeatstar:
                dicthost[namefeat][k] = dictplan[namefeat][indx[0]]
            for namefeat in listnamefeatcomp:
                dicthost[namefeat][k] = dictplan[namefeat][indx]
                
        print('Writing to %s...' % path)
        pd.DataFrame.from_dict(dicthost).to_csv(path, index=False)

    return dicthost


def retr_dicttoii(toiitarg=None, boolreplexar=False, typeverb=1, strgelem='plan'):
    
    dictfact = retr_factconv()
    
    pathlygo = os.environ['EPHESUS_DATA_PATH'] + '/'
    pathexof = pathlygo + 'data/exofop_toilists.csv'
    if typeverb > 0:
        print('Reading from %s...' % pathexof)
    objtexof = pd.read_csv(pathexof, skiprows=0)
    
    strgradielem = 'radi' + strgelem
    strgstdvradi = 'stdv' + strgradielem
    strgmasselem = 'mass' + strgelem
    strgstdvmass = 'stdv' + strgmasselem
    
    dicttoii = {}
    dicttoii['toii'] = objtexof['TOI'].values
    numbcomp = dicttoii['toii'].size
    indxcomp = np.arange(numbcomp)
    toiitargexof = np.empty(numbcomp, dtype=object)
    for k in indxcomp:
        toiitargexof[k] = int(dicttoii['toii'][k])
        
    if toiitarg is None:
        indxcomp = np.arange(numbcomp)
    else:
        indxcomp = np.where(toiitargexof == toiitarg)[0]
    
    dicttoii['toii'] = dicttoii['toii'][indxcomp]
    
    numbcomp = indxcomp.size
    
    if indxcomp.size == 0:
        if typeverb > 0:
            print('The host name, %s, was not found in the ExoFOP TOI Catalog.' % toiitarg)
        return None
    else:
        dicttoii['namestar'] = np.empty(numbcomp, dtype=object)
        dicttoii['nametoii'] = np.empty(numbcomp, dtype=object)
        for kk, k in enumerate(indxcomp):
            dicttoii['nametoii'][kk] = 'TOI-' + str(dicttoii['toii'][kk])
            dicttoii['namestar'][kk] = 'TOI-' + str(dicttoii['toii'][kk])[:-3]
        
        dicttoii['dept'] = objtexof['Depth (ppm)'].values[indxcomp] * 1e-3 # [ppt]
        dicttoii['rrat'] = np.sqrt(dicttoii['dept'] * 1e-3)
        dicttoii[strgradielem] = objtexof['Planet Radius (R_Earth)'][indxcomp].values
        dicttoii['stdvradi' + strgelem] = objtexof['Planet Radius (R_Earth) err'][indxcomp].values
        
        rascstarstrg = objtexof['RA'][indxcomp].values
        declstarstrg = objtexof['Dec'][indxcomp].values
        dicttoii['rascstar'] = np.empty_like(dicttoii[strgradielem])
        dicttoii['declstar'] = np.empty_like(dicttoii[strgradielem])
        for k in range(dicttoii[strgradielem].size):
            objt = astropy.coordinates.SkyCoord('%s %s' % (rascstarstrg[k], declstarstrg[k]), unit=(astropy.units.hourangle, astropy.units.deg))
            dicttoii['rascstar'][k] = objt.ra.degree
            dicttoii['declstar'][k] = objt.dec.degree

        # a string holding the comments
        dicttoii['strgcomm'] = np.empty(numbcomp, dtype=object)
        dicttoii['strgcomm'][:] = objtexof['Comments'][indxcomp].values
        
        #objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar']*astropy.units.degree, \
        #                                       dec=dicttoii['declstar']*astropy.units.degree, frame='icrs')
        
        objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar'], \
                                               dec=dicttoii['declstar'], frame='icrs', unit='deg')
        
        # transit duration
        dicttoii['duratrantotl'] = objtexof['Duration (hours)'].values[indxcomp] # [hours]
        
        # galactic longitude
        dicttoii['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dicttoii['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dicttoii['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dicttoii['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        # SNR
        dicttoii['s2nr'] = objtexof['Planet SNR'][indxcomp].values
        
        dicttoii['numbobsvtime'] = objtexof['Time Series Observations'][indxcomp].values
        dicttoii['numbobsvspec'] = objtexof['Spectroscopy Observations'][indxcomp].values
        dicttoii['numbobsvimag'] = objtexof['Imaging Observations'][indxcomp].values
        # alert year
        dicttoii['yearaler'] = objtexof['Date TOI Alerted (UTC)'][indxcomp].values
        for k in range(len(dicttoii['yearaler'])):
            dicttoii['yearaler'][k] = astropy.time.Time(dicttoii['yearaler'][k] + ' 00:00:00.000').decimalyear
        dicttoii['yearaler'] = dicttoii['yearaler'].astype(float)

        dicttoii['tsmmacwg'] = objtexof['ACWG TSM'][indxcomp].values
        dicttoii['esmmacwg'] = objtexof['ACWG ESM'][indxcomp].values
    
        dicttoii['facidisc'] = np.empty(numbcomp, dtype=object)
        dicttoii['facidisc'][:] = 'Transiting Exoplanet Survey Satellite (TESS)'
        
        dicttoii['peri'+strgelem] = objtexof['Period (days)'][indxcomp].values
        dicttoii['peri'+strgelem][np.where(dicttoii['peri'+strgelem] == 0)] = np.nan

        dicttoii['epocmtra'+strgelem] = objtexof['Epoch (BJD)'][indxcomp].values

        dicttoii['tmagsyst'] = objtexof['TESS Mag'][indxcomp].values
        dicttoii['stdvtmagsyst'] = objtexof['TESS Mag err'][indxcomp].values

        # transit duty cycle
        dicttoii['dcyc'] = dicttoii['duratrantotl'] / dicttoii['peri'+strgelem] / 24.
        
        boolfrst = np.zeros(numbcomp)
        dicttoii['numbplanstar'] = np.zeros(numbcomp)
        
        liststrgfeatstartici = ['massstar', 'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'distsyst', 'metastar', 'radistar', 'tmptstar', 'loggstar']
        liststrgfeatstarticiinhe = ['mass', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'd', 'MH', 'rad', 'Teff', 'logg']
        
        numbstrgfeatstartici = len(liststrgfeatstartici)
        indxstrgfeatstartici = np.arange(numbstrgfeatstartici)

        for strgfeat in liststrgfeatstartici:
            dicttoii[strgfeat] = np.zeros(numbcomp)
            dicttoii['stdv' + strgfeat] = np.zeros(numbcomp)
        
        ## crossmatch with TIC
        print('Retrieving TIC columns of TOI hosts...')
        dicttoii['tici'] = objtexof['TIC ID'][indxcomp].values
        listticiuniq = np.unique(dicttoii['tici'].astype(str))
        request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':"*", \
                                                              'filters':[{'paramName':'ID', 'values':list(listticiuniq)}]}}
        headers, outString = quer_mast(request)
        listdictquer = json.loads(outString)['data']
        for k in range(len(listdictquer)):
            indxtemp = np.where(dicttoii['tici'] == listdictquer[k]['ID'])[0]
            if indxtemp.size == 0:
                raise Exception('')
            for n in indxstrgfeatstartici:
                dicttoii[liststrgfeatstartici[n]][indxtemp] = listdictquer[k][liststrgfeatstarticiinhe[n]]
                dicttoii['stdv' + liststrgfeatstartici[n]][indxtemp] = listdictquer[k]['e_' + liststrgfeatstarticiinhe[n]]
        
        dicttoii['typedisptess'] = objtexof['TESS Disposition'][indxcomp].values
        dicttoii['boolfpos'] = objtexof['TFOPWG Disposition'][indxcomp].values == 'FP'
        
        # augment
        dicttoii['numbplanstar'] = np.empty(numbcomp)
        boolfrst = np.zeros(numbcomp, dtype=bool)
        for kk, k in enumerate(indxcomp):
            indxcompthis = np.where(dicttoii['namestar'][kk] == dicttoii['namestar'])[0]
            if kk == indxcompthis[0]:
                boolfrst[kk] = True
            dicttoii['numbplanstar'][kk] = indxcompthis.size
        
        dicttoii['numbplantranstar'] = dicttoii['numbplanstar']
        dicttoii['lumistar'] = dicttoii['radistar']**2 * (dicttoii['tmptstar'] / 5778.)**4
        dicttoii['stdvlumistar'] = dicttoii['lumistar'] * np.sqrt((2 * dicttoii['stdvradistar'] / dicttoii['radistar'])**2 + \
                                                                        (4 * dicttoii['stdvtmptstar'] / dicttoii['tmptstar'])**2)

        # mass from radii
        path = pathlygo + 'exofop_toi_mass_saved.csv'
        if not os.path.exists(path):
            dicttemp = dict()
            dicttemp[strgmasselem] = np.ones_like(dicttoii[strgradielem]) + np.nan
            dicttemp['stdvmass' + strgelem] = np.ones_like(dicttoii[strgradielem]) + np.nan
            
            numbsamppopl = 10
            indx = np.where(np.isfinite(dicttoii[strgradielem]))[0]
            for n in tqdm(range(indx.size)):
                k = indx[n]
                meanvarb = dicttoii[strgradielem][k]
                stdvvarb = dicttoii['stdvradi' + strgelem][k]
                
                # if radius uncertainty is not available, assume that it is small, so the mass uncertainty will be dominated by population uncertainty
                if not np.isfinite(stdvvarb):
                    stdvvarb = 1e-3 * dicttoii[strgradielem][k]
                else:
                    stdvvarb = dicttoii['stdvradi' + strgelem][k]
                
                # sample from a truncated Gaussian
                listradicomp = tdpy.samp_gaustrun(1000, dicttoii[strgradielem][k], stdvvarb, 0., np.inf)
                
                # estimate the mass from samples
                listmassplan = retr_massfromradi(listradicomp)
                
                dicttemp[strgmasselem][k] = np.mean(listmassplan)
                dicttemp['stdvmass' + strgelem][k] = np.std(listmassplan)
                
            if typeverb > 0:
                print('Writing to %s...' % path)
            pd.DataFrame.from_dict(dicttemp).to_csv(path, index=False)
        else:
            if typeverb > 0:
                print('Reading from %s...' % path)
            dicttemp = pd.read_csv(path).to_dict(orient='list')
            
            for name in dicttemp:
                dicttemp[name] = np.array(dicttemp[name])
                if toiitarg is not None:
                    dicttemp[name] = dicttemp[name][indxcomp]

        dicttoii[strgmasselem] = dicttemp['mass' + strgelem]
        
        dicttoii['stdvmass' + strgelem] = dicttemp['stdvmass' + strgelem]
        
        dicttoii['masstotl'] = dicttoii['massstar'] + dicttoii[strgmasselem] / dictfact['msme']
        dicttoii['smax'+strgelem] = retr_smaxkepl(dicttoii['peri'+strgelem], dicttoii['masstotl'])
        
        dicttoii['inso'] = dicttoii['lumistar'] / dicttoii['smax'+strgelem]**2
        
        dicttoii['tmptplan'] = dicttoii['tmptstar'] * np.sqrt(dicttoii['radistar'] / dicttoii['smax'+strgelem] / 2. / dictfact['aurs'])
        # temp check if factor of 2 is right
        dicttoii['stdvtmptplan'] = np.sqrt((dicttoii['stdvtmptstar'] / dicttoii['tmptstar'])**2 + \
                                                        0.5 * (dicttoii['stdvradistar'] / dicttoii['radistar'])**2) / np.sqrt(2.)
        
        dicttoii['densplan'] = 5.51 * dicttoii[strgmasselem] / dicttoii[strgradielem]**3 # [g/cm^3]
        dicttoii['booltran'] = np.ones_like(dicttoii['toii'], dtype=bool)
    
        dicttoii['vesc'] = retr_vesc(dicttoii[strgmasselem], dicttoii[strgradielem])
        print('temp: vsiistar and projoblq are NaNs')
        dicttoii['vsiistar'] = np.ones(numbcomp) + np.nan
        dicttoii['projoblq'] = np.ones(numbcomp) + np.nan
        
        # replace confirmed planet features
        if boolreplexar:
            dictexar = retr_dictexar()
            listdisptess = objtexof['TESS Disposition'][indxcomp].values.astype(str)
            listdisptfop = objtexof['TFOPWG Disposition'][indxcomp].values.astype(str)
            indxexofcpla = np.where((listdisptfop == 'CP') & (listdisptess == 'PC'))[0]
            listticicpla = dicttoii['tici'][indxexofcpla]
            numbticicpla = len(listticicpla)
            indxticicpla = np.arange(numbticicpla)
            for k in indxticicpla:
                indxexartici = np.where((dictexar['tici'] == int(listticicpla[k])) & \
                                                    (dictexar['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]
                indxexoftici = np.where(dicttoii['tici'] == int(listticicpla[k]))[0]
                for strg in dictexar.keys():
                    if indxexartici.size > 0:
                        dicttoii[strg] = np.delete(dicttoii[strg], indxexoftici)
                    dicttoii[strg] = np.concatenate((dicttoii[strg], dictexar[strg][indxexartici]))

        # calculate TSM and ESM
        calc_tsmmesmm(dicttoii, strgelem=strgelem)
    
        # turn zero TSM ACWG or ESM ACWG into NaN
        indx = np.where(dicttoii['tsmmacwg'] == 0)[0]
        dicttoii['tsmmacwg'][indx] = np.nan
        
        indx = np.where(dicttoii['esmmacwg'] == 0)[0]
        dicttoii['esmmacwg'][indx] = np.nan

    return dicttoii


def calc_tsmmesmm(dictpopl, strgelem='plan', boolsamp=False):
    
    if boolsamp:
        numbsamp = 1000
    else:
        numbsamp = 1

    strgradielem = 'radi' + strgelem
    strgmasselem = 'mass' + strgelem
    
    numbcomp = dictpopl[strgmasselem].size
    listtsmm = np.empty((numbsamp, numbcomp)) + np.nan
    listesmm = np.empty((numbsamp, numbcomp)) + np.nan
    
    for n in range(numbcomp):
        
        if not np.isfinite(dictpopl['tmptplan'][n]):
            continue
        
        if not np.isfinite(dictpopl[strgradielem][n]):
            continue
        
        if boolsamp:
            if not np.isfinite(dictpopl['stdvradi' + strgelem][n]):
                stdv = dictpopl[strgradielem][n]
            else:
                stdv = dictpopl['stdvradi' + strgelem][n]
            listradicomp = tdpy.samp_gaustrun(numbsamp, dictpopl[strgradielem][n], stdv, 0., np.inf)
            
            listmassplan = tdpy.samp_gaustrun(numbsamp, dictpopl[strgmasselem][n], dictpopl['stdvmass' + strgelem][n], 0., np.inf)

            if not np.isfinite(dictpopl['stdvtmptplan'][n]):
                stdv = dictpopl['tmptplan'][n]
            else:
                stdv = dictpopl['stdvtmptplan'][n]
            listtmptplan = tdpy.samp_gaustrun(numbsamp, dictpopl['tmptplan'][n], stdv, 0., np.inf)
            
            if not np.isfinite(dictpopl['stdvradistar'][n]):
                stdv = dictpopl['radistar'][n]
            else:
                stdv = dictpopl['stdvradistar'][n]
            listradistar = tdpy.samp_gaustrun(numbsamp, dictpopl['radistar'][n], stdv, 0., np.inf)
            
            listkmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['kmagsyst'][n], dictpopl['stdvkmagsyst'][n])
            listjmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['jmagsyst'][n], dictpopl['stdvjmagsyst'][n])
            listtmptstar = tdpy.samp_gaustrun(numbsamp, dictpopl['tmptstar'][n], dictpopl['stdvtmptstar'][n], 0., np.inf)
        
        else:
            listradicomp = dictpopl[strgradielem][None, n]
            listtmptplan = dictpopl['tmptplan'][None, n]
            listmassplan = dictpopl[strgmasselem][None, n]
            listradistar = dictpopl['radistar'][None, n]
            listkmagsyst = dictpopl['kmagsyst'][None, n]
            listjmagsyst = dictpopl['jmagsyst'][None, n]
            listtmptstar = dictpopl['tmptstar'][None, n]
        
        # TSM
        listtsmm[:, n] = retr_tsmm(listradicomp, listtmptplan, listmassplan, listradistar, listjmagsyst)

        # ESM
        listesmm[:, n] = retr_esmm(listtmptplan, listtmptstar, listradicomp, listradistar, listkmagsyst)
        
        #if (listesmm[:, n] < 1e-10).any():
        #    print('listradicomp')
        #    summgene(listradicomp)
        #    print('listtmptplan')
        #    summgene(listtmptplan)
        #    print('listmassplan')
        #    summgene(listmassplan)
        #    print('listradistar')
        #    summgene(listradistar)
        #    print('listkmagsyst')
        #    summgene(listkmagsyst)
        #    print('listjmagsyst')
        #    summgene(listjmagsyst)
        #    print('listtmptstar')
        #    summgene(listtmptstar)
        #    print('listesmm[:, n]')
        #    summgene(listesmm[:, n])
        #    raise Exception('')
    dictpopl['tsmm'] = np.nanmedian(listtsmm, 0)
    dictpopl['stdvtsmm'] = np.nanstd(listtsmm, 0)
    dictpopl['esmm'] = np.nanmedian(listesmm, 0)
    dictpopl['stdvesmm'] = np.nanstd(listesmm, 0)
    
    #print('listesmm')
    #summgene(listesmm)
    #print('dictpopl[tsmm]')
    #summgene(dictpopl['tsmm'])
    #print('dictpopl[esmm]')
    #summgene(dictpopl['esmm'])
    #print('dictpopl[stdvtsmm]')
    #summgene(dictpopl['stdvtsmm'])
    #print('dictpopl[stdvesmm]')
    #summgene(dictpopl['stdvesmm'])
    #raise Exception('')


def retr_reso(listperi, maxmordr=10):
    
    if np.where(listperi == 0)[0].size > 0:
        raise Exception('')

    numbsamp = listperi.shape[0]
    numbcomp = listperi.shape[1]
    indxcomp = np.arange(numbcomp)
    listratiperi = np.zeros((numbsamp, numbcomp, numbcomp))
    intgreso = np.zeros((numbcomp, numbcomp, 2))
    for j in indxcomp:
        for jj in indxcomp:
            if j >= jj:
                continue
                
            rati = listperi[:, j] / listperi[:, jj]
            #print('listperi')
            #print(listperi)
            #print('rati')
            #print(rati)
            if rati < 1:
                listratiperi[:, j, jj] = 1. / rati
            else:
                listratiperi[:, j, jj] = rati

            minmdiff = 1e100
            for a in range(1, maxmordr):
                for aa in range(1, maxmordr):
                    diff = abs(float(a) / aa - listratiperi[:, j, jj])
                    if np.mean(diff) < minmdiff:
                        minmdiff = np.mean(diff)
                        minmreso = a, aa
            intgreso[j, jj, :] = minmreso
            #print('minmdiff') 
            #print(minmdiff)
            #print('minmreso')
            #print(minmreso)
            #print
    
    return intgreso, listratiperi


def retr_dilu(tmpttarg, tmptcomp, strgwlentype='tess'):
    
    if strgwlentype != 'tess':
        raise Exception('')
    else:
        binswlen = np.linspace(0.6, 1.)
    meanwlen = (binswlen[1:] + binswlen[:-1]) / 2.
    diffwlen = (binswlen[1:] - binswlen[:-1]) / 2.
    
    fluxtarg = tdpy.retr_specbbod(tmpttarg, meanwlen)
    fluxtarg = np.sum(diffwlen * fluxtarg)
    
    fluxcomp = tdpy.retr_specbbod(tmptcomp, meanwlen)
    fluxcomp = np.sum(diffwlen * fluxcomp)
    
    dilu = 1. - fluxtarg / (fluxtarg + fluxcomp)
    
    return dilu


def anim_tmptdete(timefull, lcurfull, meantimetmpt, lcurtmpt, pathimag, listindxtimeposimaxm, corrprod, corr, strgextn='', \
                  ## file type of the plot
                  typefileplot='pdf', \
                  colr=None):
    
    numbtimefull = timefull.size
    numbtimekern = lcurtmpt.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    
    listpath = []
    gdat.cmndmakeanim = 'convert -delay 20'
    
    numbtimeanim = min(200, numbtimefullruns)
    indxtimefullrunsanim = np.random.choice(indxtimefullruns, size=numbtimeanim, replace=False)
    indxtimefullrunsanim = np.sort(indxtimefullrunsanim)

    for tt in indxtimefullrunsanim:
        
        path = pathimag + 'lcur%s_%08d.%s' % (strgextn, tt, typefileplot)
        listpath.append(path)
        if not os.path.exists(path):
            plot_tmptdete(timefull, lcurfull, tt, meantimetmpt, lcurtmpt, path, listindxtimeposimaxm, corrprod, corr)
        gdat.cmndmakeanim += ' %s' % path
    
    pathanim = pathimag + 'lcur%s.gif' % strgextn
    gdat.cmndmakeanim += ' %s' % pathanim
    print('gdat.cmndmakeanim')
    print(gdat.cmndmakeanim)
    os.system(gdat.cmndmakeanim)
    gdat.cmnddeleimag = 'rm'
    for path in listpath:
        gdat.cmnddeleimag += ' ' + path
    os.system(gdat.cmnddeleimag)


def plot_tmptdete(timefull, lcurfull, tt, meantimetmpt, lcurtmpt, path, listindxtimeposimaxm, corrprod, corr):
    
    numbtimekern = lcurtmpt.size
    indxtimekern = np.arange(numbtimekern)
    numbtimefull = lcurfull.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    difftime = timefull[1] - timefull[0]
    
    figr, axis = plt.subplots(5, 1, figsize=(8, 11))
    
    # plot the whole light curve
    proc_axiscorr(timefull, lcurfull, axis[0], listindxtimeposimaxm)
    
    # plot zoomed-in light curve
    minmindx = max(0, tt - int(numbtimekern / 4))
    maxmindx = min(numbtimefullruns - 1, tt + int(5. * numbtimekern / 4))
    indxtime = np.arange(minmindx, maxmindx + 1)
    print('indxtime')
    summgene(indxtime)
    proc_axiscorr(timefull, lcurfull, axis[1], listindxtimeposimaxm, indxtime=indxtime)
    
    # plot template
    axis[2].plot(timefull[0] + meantimetmpt + tt * difftime, lcurtmpt, color='b', marker='v')
    axis[2].set_ylabel('Template')
    axis[2].set_xlim(axis[1].get_xlim())

    # plot correlation
    axis[3].plot(timefull[0] + meantimetmpt + tt * difftime, corrprod[tt, :], color='red', marker='o')
    axis[3].set_ylabel('Correlation')
    axis[3].set_xlim(axis[1].get_xlim())
    
    # plot the whole total correlation
    print('indxtimefullruns')
    summgene(indxtimefullruns)
    print('timefull')
    summgene(timefull)
    print('corr')
    summgene(corr)
    axis[4].plot(timefull[indxtimefullruns], corr, color='m', marker='o', ms=1, rasterized=True)
    axis[4].set_ylabel('Total correlation')
    
    titl = 'C = %.3g' % corr[tt]
    axis[0].set_title(titl)

    limtydat = axis[0].get_ylim()
    axis[0].fill_between(timefull[indxtimekern+tt], limtydat[0], limtydat[1], alpha=0.4)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    

def proc_axiscorr(time, lcur, axis, listindxtimeposimaxm, indxtime=None, colr='k', timeoffs=2457000):
    
    if indxtime is None:
        indxtimetemp = np.arange(time.size)
    else:
        indxtimetemp = indxtime
    axis.plot(time[indxtimetemp], lcur[indxtimetemp], ls='', marker='o', color=colr, rasterized=True, ms=0.5)
    maxmydat = axis.get_ylim()[1]
    for kk in range(len(listindxtimeposimaxm)):
        if listindxtimeposimaxm[kk] in indxtimetemp:
            axis.plot(time[listindxtimeposimaxm[kk]], maxmydat, marker='v', color='b')
    #print('timeoffs')
    #print(timeoffs)
    #axis.set_xlabel('Time [BJD-%d]' % timeoffs)
    axis.set_ylabel('Relative flux')
    

def retr_lcurmodl_flarsing(meantime, timeflar, amplflar, scalrise, scalfall):
    
    numbtime = meantime.size
    if numbtime == 0:
        raise Exception('')
    indxtime = np.arange(numbtime)
    indxtimerise = np.where(meantime < timeflar)[0]
    indxtimefall = np.setdiff1d(indxtime, indxtimerise)
    lcur = np.empty_like(meantime)
    lcur[indxtimerise] = np.exp((meantime[indxtimerise] - timeflar) / scalrise)
    lcur[indxtimefall] = np.exp(-(meantime[indxtimefall] - timeflar) / scalfall)
    lcur *= amplflar / np.amax(lcur) 
    
    return lcur


def srch_flar(time, lcur, typeverb=1, strgextn='', numbkern=3, minmscalfalltmpt=None, maxmscalfalltmpt=None, \
                                                                    pathimag=None, boolplot=True, boolanim=False, thrs=None):

    minmtime = np.amin(time)
    timeflartmpt = 0.
    amplflartmpt = 1.
    scalrisetmpt = 0. / 24.
    difftime = np.amin(time[1:] - time[:-1])
    
    if minmscalfalltmpt is None:
        minmscalfalltmpt = 3 * difftime
    
    if maxmscalfalltmpt is None:
        maxmscalfalltmpt = 3. / 24.
    
    if typeverb > 1:
        print('lcurtmpt')
        summgene(lcurtmpt)
    
    indxscalfall = np.arange(numbkern)
    listscalfalltmpt = np.linspace(minmscalfalltmpt, maxmscalfalltmpt, numbkern)
    print('listscalfalltmpt')
    print(listscalfalltmpt)
    listcorr = []
    listlcurtmpt = [[] for k in indxscalfall]
    meantimetmpt = [[] for k in indxscalfall]
    for k in indxscalfall:
        numbtimekern = 3 * int(listscalfalltmpt[k] / difftime)
        print('numbtimekern')
        print(numbtimekern)
        meantimetmpt[k] = np.arange(numbtimekern) * difftime
        print('meantimetmpt[k]')
        summgene(meantimetmpt[k])
        if numbtimekern == 0:
            raise Exception('')
        listlcurtmpt[k] = retr_lcurmodl_flarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
        if not np.isfinite(listlcurtmpt[k]).all():
            raise Exception('')
        
    corr, listindxtimeposimaxm, timefull, lcurfull = corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, thrs=thrs, boolanim=boolanim, boolplot=boolplot, \
                                                                                            typeverb=typeverb, strgextn=strgextn, pathimag=pathimag)

    #corr, listindxtimeposimaxm, timefull, rflxfull = corr_tmpt(gdat.timethis, gdat.rflxthis, gdat.listtimetmpt, gdat.listdflxtmpt, \
    #                                                                    thrs=gdat.thrstmpt, boolanim=gdat.boolanimtmpt, boolplot=gdat.boolplottmpt, \
     #                                                               typeverb=gdat.typeverb, strgextn=gdat.strgextnthis, pathimag=gdat.pathtargimag)
                
    return corr, listindxtimeposimaxm, meantimetmpt, timefull, lcurfull


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def corr_arryprod(lcurtemp, lcurtmpt, numbkern):
    
    # correlate
    corrprod = [[] for k in range(numbkern)]
    for k in range(numbkern):
        corrprod[k] = lcurtmpt[k] * lcurtemp[k]
    
    return corrprod


#@jit(parallel=True)
def corr_copy(indxtimefullruns, lcurstan, indxtimekern, numbkern):
    '''
    Make a matrix with rows as the shifted and windowed copies of the time series.
    '''
    print('corr_copy()')
    listlcurtemp = [[] for k in range(numbkern)]
    for k in range(numbkern):
        numbtimefullruns = indxtimefullruns[k].size
        numbtimekern = indxtimekern[k].size
        listlcurtemp[k] = np.empty((numbtimefullruns, numbtimekern))
        print('k')
        print(k)
        print('numbtimefullruns')
        print(numbtimefullruns)
        print('numbtimekern')
        print(numbtimekern)

        for t in range(numbtimefullruns):
            listlcurtemp[k][t, :] = lcurstan[indxtimefullruns[k][t]+indxtimekern[k]]
        print('listlcurtemp[k]')
        summgene(listlcurtemp[k])
    print('')
    return listlcurtemp


def corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, typeverb=2, thrs=None, strgextn='', pathimag=None, boolplot=True, \
              ## file type of the plot
              typefileplot='pdf', \
              boolanim=False, \
             ):
    
    timeoffs = np.amin(time) // 1000
    timeoffs *= 1000
    time -= timeoffs
    
    if typeverb > 1:
        timeinit = timemodu.time()
    
    print('corr_tmpt()')
    
    if lcur.ndim > 1:
        raise Exception('')
    
    for lcurtmpt in listlcurtmpt:
        if not np.isfinite(lcurtmpt).all():
            raise Exception('')

    if not np.isfinite(lcur).all():
        raise Exception('')

    numbtime = lcur.size
    
    numbkern = len(listlcurtmpt)
    indxkern = np.arange(numbkern)
    
    # count gaps
    difftime = time[1:] - time[:-1]
    minmdifftime = np.amin(difftime)
    difftimesort = np.sort(difftime)[::-1]
    print('difftimesort')
    for k in range(difftimesort.size):
        print(difftimesort[k] / minmdifftime)
        if k == 20:
             break
    
    boolthrsauto = thrs is None
    
    print('temp: setting boolthrsauto')
    thrs = 1.
    boolthrsauto = False
    
    # number of time samples in the kernel
    numbtimekern = np.empty(numbkern, dtype=int)
    indxtimekern = [[] for k in indxkern]
    
    # take out the mean
    listlcurtmptstan = [[] for k in indxkern]
    for k in indxkern:
        listlcurtmptstan[k] = np.copy(listlcurtmpt[k])
        listlcurtmptstan[k] -= np.mean(listlcurtmptstan[k])
        numbtimekern[k] = listlcurtmptstan[k].size
        indxtimekern[k] = np.arange(numbtimekern[k])

    minmtimechun = 3 * 3. / 24. / 60. # [days]
    print('minmdifftime * 24 * 60')
    print(minmdifftime * 24 * 60)
    listlcurfull = []
    indxtimebndr = np.where(difftime > minmtimechun)[0]
    indxtimebndr = np.concatenate([np.array([0]), indxtimebndr, np.array([numbtime - 1])])
    numbchun = indxtimebndr.size - 1
    indxchun = np.arange(numbchun)
    corrchun = [[[] for k in indxkern] for l in indxchun]
    listindxtimeposimaxm = [[[] for k in indxkern] for l in indxchun]
    listlcurchun = [[] for l in indxchun]
    listtimechun = [[] for l in indxchun]
    print('indxtimebndr')
    print(indxtimebndr)
    print('numbchun')
    print(numbchun)
    for l in indxchun:
        
        print('Chunk %d...' % l)
        
        minmindxtimeminm = 0
        minmtime = time[indxtimebndr[l]+1]
        print('minmtime')
        print(minmtime)
        print('indxtimebndr[l]')
        print(indxtimebndr[l])
        maxmtime = time[indxtimebndr[l+1]]
        print('maxmtime')
        print(maxmtime)
        numb = int(round((maxmtime - minmtime) / minmdifftime))
        print('numb')
        print(numb)
        
        if numb == 0:
            print('Skipping due to chunk with single point...')
            continue

        timechun = np.linspace(minmtime, maxmtime, numb)
        listtimechun[l] = timechun
        print('timechun')
        summgene(timechun)
        
        if float(indxtimebndr[l+1] - indxtimebndr[l]) / numb < 0.8:
            print('Skipping due to undersampled chunk...')
            continue

        numbtimefull = timechun.size
        print('numbtimefull')
        print(numbtimefull)
        
        indxtimechun = np.arange(indxtimebndr[l], indxtimebndr[l+1] + 1)
        
        print('time[indxtimechun]')
        summgene(time[indxtimechun])
        
        # interpolate
        lcurchun = scipy.interpolate.interp1d(time[indxtimechun], lcur[indxtimechun])(timechun)
        
        if indxtimechun.size != timechun.size:
            print('time[indxtimechun]')
            if timechun.size < 50:
                for timetemp in time[indxtimechun]:
                    print(timetemp)
            summgene(time[indxtimechun])
            print('timechun')
            if timechun.size < 50:
                for timetemp in timechun:
                    print(timetemp)
            summgene(timechun)
            #raise Exception('')

        # take out the mean
        lcurchun -= np.mean(lcurchun)
        
        listlcurchun[l] = lcurchun
        
        # size of the full grid minus the kernel size
        numbtimefullruns = np.empty(numbkern, dtype=int)
        indxtimefullruns = [[] for k in indxkern]
        
        # find the correlation
        for k in indxkern:
            print('Kernel %d...' % k)
            
            if numb < numbtimekern[k]:
                print('Skipping due to chunk shorther than the kernel...')
                continue
            
            # find the total correlation (along the time delay axis)
            corrchun[l][k] = scipy.signal.correlate(lcurchun, listlcurtmptstan[k], mode='valid')
            print('corrchun[l][k]')
            summgene(corrchun[l][k])
        
            numbtimefullruns[k] = numbtimefull - numbtimekern[k] + 1
            indxtimefullruns[k] = np.arange(numbtimefullruns[k])
        
            print('numbtimekern[k]')
            print(numbtimekern[k])
            print('numbtimefullruns[k]')
            print(numbtimefullruns[k])

            if boolthrsauto:
                perclowrcorr = np.percentile(corr[k],  1.)
                percupprcorr = np.percentile(corr[k], 99.)
                indx = np.where((corr[k] < percupprcorr) & (corr[k] > perclowrcorr))[0]
                medicorr = np.median(corr[k])
                thrs = np.std(corr[k][indx]) * 7. + medicorr

            if not np.isfinite(corrchun[l][k]).all():
                raise Exception('')

            # determine the threshold on the maximum correlation
            if typeverb > 1:
                print('thrs')
                print(thrs)

            # find triggers
            listindxtimeposi = np.where(corrchun[l][k] > thrs)[0]
            if typeverb > 1:
                print('listindxtimeposi')
                summgene(listindxtimeposi)
            
            # cluster triggers
            listtemp = []
            listindxtimeposiptch = []
            for kk in range(len(listindxtimeposi)):
                listtemp.append(listindxtimeposi[kk])
                if kk == len(listindxtimeposi) - 1 or listindxtimeposi[kk] != listindxtimeposi[kk+1] - 1:
                    listindxtimeposiptch.append(np.array(listtemp))
                    listtemp = []
            
            if typeverb > 1:
                print('listindxtimeposiptch')
                summgene(listindxtimeposiptch)

            listindxtimeposimaxm[l][k] = np.empty(len(listindxtimeposiptch), dtype=int)
            for kk in range(len(listindxtimeposiptch)):
                indxtemp = np.argmax(corrchun[l][k][listindxtimeposiptch[kk]])
                listindxtimeposimaxm[l][k][kk] = listindxtimeposiptch[kk][indxtemp]
            
            if typeverb > 1:
                print('listindxtimeposimaxm[l][k]')
                summgene(listindxtimeposimaxm[l][k])
            
            if boolplot or boolanim:
                strgextntotl = strgextn + '_kn%02d' % k
        
            if boolplot:
                if numbtimefullruns[k] <= 0:
                    continue
                numbdeteplot = min(len(listindxtimeposimaxm[l][k]), 10)
                figr, axis = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                
                #proc_axiscorr(time[indxtimechun], lcur[indxtimechun], axis[0], listindxtimeposimaxm[l][k])
                proc_axiscorr(timechun, lcurchun, axis[0], listindxtimeposimaxm[l][k])
                
                axis[1].plot(timechun[indxtimefullruns[k]], corrchun[l][k], color='m', ls='', marker='o', ms=1, rasterized=True)
                axis[1].set_ylabel('C')
                axis[1].set_xlabel('Time [BJD-%d]' % timeoffs)
                
                path = pathimag + 'lcurflar_ch%02d%s.%s' % (l, strgextntotl, gdat.typefileplot)
                plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                for n in range(numbdeteplot):
                    figr, axis = plt.subplots(figsize=(8, 4), sharex=True)
                    for i in range(numbdeteplot):
                        indxtimeplot = indxtimekern[k] + listindxtimeposimaxm[l][k][i]
                        proc_axiscorr(timechun, lcurchun, axis, listindxtimeposimaxm[l][k], indxtime=indxtimeplot, timeoffs=timeoffs)
                    path = pathimag + 'lcurflar_ch%02d%s_det.%s' % (l, strgextntotl, gdat.typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
            print('Done with the plot...')
            if False and boolanim:
                path = pathimag + 'lcur%s.gif' % strgextntotl
                if not os.path.exists(path):
                    anim_tmptdete(timefull, lcurfull, meantimetmpt[k], listlcurtmpt[k], pathimag, \
                                                                listindxtimeposimaxm[l][k], corrprod[k], corrchun[l][k], strgextn=strgextntotl)
                else:
                    print('Skipping animation for kernel %d...' % k)
    if typeverb > 1:
        print('Delta T (corr_tmpt, rest): %g' % (timemodu.time() - timeinit))

    return corrchun, listindxtimeposimaxm, listtimechun, listlcurchun


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


# transits
def retr_indxtran(time, epoc, peri, duratrantotl=None):
    '''
    Find the transit indices for a given time axis, epoch, period, and optionally transit duration.
    '''

    if np.isfinite(peri):
        if duratrantotl is None:
            duratemp = 0.
        else:
            duratemp = duratrantotl
        intgminm = np.ceil((np.amin(time) - epoc - duratemp / 48.) / peri)
        intgmaxm = np.ceil((np.amax(time) - epoc - duratemp / 48.) / peri)
        indxtran = np.arange(intgminm, intgmaxm)
    else:
        indxtran = np.arange(1)
    
    return indxtran


def retr_indxtimetran(time, epoc, peri, \
                      # total transit duration [hours]
                      duratrantotl, \
                      duratranfull=None, \
                      # type of the in-transit phase interval
                      typeineg=None, \
                      booloutt=False, boolseco=False):
    '''
    Return the indices of times during transit.
    '''

    if not np.isfinite(time).all():
        raise Exception('')
    
    if not np.isfinite(duratrantotl).all():
        print('duratrantotl')
        print(duratrantotl)
        raise Exception('')
    
    indxtran = retr_indxtran(time, epoc, peri, duratrantotl)

    if boolseco:
        offs = 0.5
    else:
        offs = 0.

    listindxtimetran = []
    for n in indxtran:
        timetotlinit = epoc + (n + offs) * peri - duratrantotl / 48.
        timetotlfinl = epoc + (n + offs) * peri + duratrantotl / 48.
        if duratranfull is not None:
            timefullinit = epoc + (n + offs) * peri - duratranfull / 48.
            timefullfinl = epoc + (n + offs) * peri + duratranfull / 48.
            timeingrhalf = (timetotlinit + timefullinit) / 2.
            timeeggrhalf = (timetotlfinl + timefullfinl) / 2.
            if typeineg == 'inge':
                indxtime = np.where((time > timetotlinit) & (time < timefullinit) | (time > timefullfinl) & (time < timetotlfinl))[0]
            if typeineg == 'ingr':
                indxtime = np.where((time > timetotlinit) & (time < timefullinit))[0]
            if typeineg == 'eggr':
                indxtime = np.where((time > timefullfinl) & (time < timetotlfinl))[0]
            if typeineg == 'ingrinit':
                indxtime = np.where((time > timetotlinit) & (time < timeingrhalf))[0]
            if typeineg == 'ingrfinl':
                indxtime = np.where((time > timeingrhalf) & (time < timefullinit))[0]
            if typeineg == 'eggrinit':
                indxtime = np.where((time > timefullfinl) & (time < timeeggrhalf))[0]
            if typeineg == 'eggrfinl':
                indxtime = np.where((time > timeeggrhalf) & (time < timetotlfinl))[0]
        else:
            indxtime = np.where((time > timetotlinit) & (time < timetotlfinl))[0]
        listindxtimetran.append(indxtime)
    if len(listindxtimetran) > 0:
        indxtimetran = np.concatenate(listindxtimetran)
        indxtimetran = np.unique(indxtimetran)
    else:
        indxtimetran = np.array([])

    if booloutt:
        indxtimeretr = np.setdiff1d(np.arange(time.size), indxtimetran)
    else:
        indxtimeretr = indxtimetran
    
    return indxtimeretr
    

def retr_timeedge(time, lcur, timebrekregi, \
                  # Boolean flag to add breaks at discontinuties
                  booladdddiscbdtr, \
                  timescal, \
                 ):
    
    difftime = time[1:] - time[:-1]
    indxtimebrekregi = np.where(difftime > timebrekregi)[0]
    
    if booladdddiscbdtr:
        listindxtimebrekregiaddi = []
        dif1 = lcur[:-1] - lcur[1:]
        indxtimechec = np.where(dif1 > 20. * np.std(dif1))[0]
        for k in indxtimechec:
            if np.mean(lcur[-3+k:k]) - np.mean(lcur[k:k+3]) < np.std(np.concatenate((lcur[-3+k:k], lcur[k:k+3]))):
                listindxtimebrekregiaddi.append(k)

            #diff = lcur[k] - lcur[k-1]
            #if abs(diff) > 5 * np.std(lcur[k-3:k]) and abs(diff) > 5 * np.std(lcur[k:k+3]):
            #    listindxtimebrekregiaddi.append(k)
            #    #print('k')
            #    #print(k)
            #    #print('diff')
            #    #print(diff)
            #    #print('np.std(lcur[k:k+3])')
            #    #print(np.std(lcur[k:k+3]))
            #    #print('np.std(lcur[k-3:k])')
            #    #print(np.std(lcur[k-3:k]))
            #    #print('')
        listindxtimebrekregiaddi = np.array(listindxtimebrekregiaddi, dtype=int)
        indxtimebrekregi = np.concatenate([indxtimebrekregi, listindxtimebrekregiaddi])
        indxtimebrekregi = np.unique(indxtimebrekregi)

    timeedge = [0, np.inf]
    for k in indxtimebrekregi:
        timeedgeprim = (time[k] + time[k+1]) / 2.
        timeedge.append(timeedgeprim)
    timeedge = np.array(timeedge)
    timeedge = np.sort(timeedge)

    return timeedge


def retr_tsecticibase(tici, typeverb=1):
    '''
    Retrieve the list of sectors for which SPOC light curves are available for target, using the predownloaded database of light curves
    as opposed to individual download calls over internet.
    '''
    
    pathbase = os.environ['TESS_DATA_PATH'] + '/data/lcur/'
    path = pathbase + 'tsec/tsec_spoc_%016d.csv' % tici
    if not os.path.exists(path):
        listtsecsele = np.arange(1, 60)
        listpath = []
        listtsec = []
        strgtagg = '*-%016d-*.fits' % tici
        for tsec in listtsecsele:
            pathtemp = pathbase + 'sector-%02d/' % tsec
            listpathtemp = fnmatch.filter(os.listdir(pathtemp), strgtagg)
            
            if len(listpathtemp) > 0:
                listpath.append(pathtemp + listpathtemp[0])
                listtsec.append(tsec)
        
        listtsec = np.array(listtsec).astype(int)
        print('Writing to %s...' % path)
        objtfile = open(path, 'w')
        for k in range(len(listpath)):
            objtfile.write('%d,%s\n' % (listtsec[k], listpath[k]))
        objtfile.close()
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        objtfile = open(path, 'r')
        listtsec = []
        listpath = []
        for line in objtfile:
            linesplt = line.split(',')
            listtsec.append(linesplt[0])
            listpath.append(linesplt[1][:-1])
        listtsec = np.array(listtsec).astype(int)
        objtfile.close()
    
    return listtsec, listpath


def retr_lliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.log_likelihood(lcur)


def retr_gradlliknegagpro(listparagpro, lcur, objtgpro):
    '''
    Compute the gradient of the negative loglikelihood of the GP model
    '''
    
    objtgpro.set_parameter_vector(listparagpro)
    
    return -objtgpro.grad_log_likelihood(lcur)[1]


def bdtr_tser( \
              # time grid
              time, \
              
              # dependent variable
              lcur, \
              
              # standard-deviation of the dependent variable
              stdvlcur, \

              # epoc, period, and duration of mask
              epocmask=None, perimask=None, duramask=None, \
              
              # Boolean flag to break the time-series into regions
              boolbrekregi=False, \

              # minimum gap to break the time-series into regions
              timebrekregi=None, \
              
              # Boolean flag to add breaks at vertical discontinuties
              booladdddiscbdtr=True, \
              
              # baseline detrend type
              ## 'gpro':
              ## 'medi':
              ## 'spln':
              typebdtr=None, \
              
              # order of the spline
              ordrspln=None, \
              
              # time scale of the spline detrending
              timescalbdtrspln=None, \
              
              # time scale of the median detrending
              timescalbdtrmedi=None, \
              
              # verbosity level
              typeverb=1, \
              
             ):
    '''
    Detrend the baseline of time-series data.
    '''
    
    if typebdtr is None:
        typebdtr = 'gpro'
    
    if boolbrekregi and timebrekregi is None:
        timebrekregi = 0.1 # [day]
    if ordrspln is None:
        ordrspln = 3
    if timescalbdtrspln is None:
        timescalbdtrspln = 0.5 # [days]
    if timescalbdtrmedi is None:
        timescalbdtrmedi = 0.5 # [days]
    
    if typebdtr == 'spln' or typebdtr == 'gpro':
        timescal = timescalbdtrspln
    else:
        timescal = timescalbdtrmedi
    if typeverb > 0:
        print('Detrending the light curve with at a time scale of %.g days...' % timescal)
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits while detrending...')
        
    if boolbrekregi:
        # determine the times at which the light curve will be broken into pieces
        timeedge = retr_timeedge(time, lcur, timebrekregi, booladdddiscbdtr, timescal)
        numbedge = len(timeedge)
        numbregi = numbedge - 1
    else:
        timeedge = [np.amin(time), np.amax(time)]
        numbregi = 1
    
    if typeverb > 1:
        print('timebrekregi')
        print(timebrekregi)
        print('Number of regions: %d' % numbregi)
        print('Times at the edges of the regions:')
        print(timeedge)

    indxregi = np.arange(numbregi)
    lcurbdtrregi = [[] for i in indxregi]
    indxtimeregi = [[] for i in indxregi]
    indxtimeregioutt = [[] for i in indxregi]
    listobjtspln = [[] for i in indxregi]
    for i in indxregi:
        if typeverb > 1:
            print('Region %d' % i)
        # find times inside the region
        indxtimeregi[i] = np.where((time >= timeedge[i]) & (time <= timeedge[i+1]))[0]
        timeregi = time[indxtimeregi[i]]
        lcurregi = lcur[indxtimeregi[i]]
        stdvlcurregi = stdvlcur[indxtimeregi[i]]
        
        # mask out the transits
        if epocmask is not None and len(epocmask) > 0 and duramask is not None and perimask is not None:
            # find the out-of-transit times
            indxtimetran = []
            for k in range(epocmask.size):
                if np.isfinite(duramask[k]):
                    indxtimetran.append(retr_indxtimetran(timeregi, epocmask[k], perimask[k], duramask[k]))
            
            indxtimetran = np.concatenate(indxtimetran)
            indxtimeregioutt[i] = np.setdiff1d(np.arange(timeregi.size), indxtimetran)
        else:
            indxtimeregioutt[i] = np.arange(timeregi.size)
            
        if typeverb > 1:
            print('lcurregi[indxtimeregioutt[i]]')
            summgene(lcurregi[indxtimeregioutt[i]])
        
        if typebdtr == 'medi':
            listobjtspln = None
            size = int(timescalbdtrmedi / np.amin(timeregi[1:] - timeregi[:-1]))
            if size == 0:
                print('timescalbdtrmedi')
                print(timescalbdtrmedi)
                print('np.amin(timeregi[1:] - timeregi[:-1])')
                print(np.amin(timeregi[1:] - timeregi[:-1]))
                print('lcurregi')
                summgene(lcurregi)
                raise Exception('')
            lcurbdtrregi[i] = 1. + lcurregi - scipy.ndimage.median_filter(lcurregi, size=size)
        
        if typebdtr == 'gpro':
            # fit a Gaussian Process (GP) model to the data as baseline
            ## construct the kernel object
            objtkern = celerite.terms.Matern32Term(log_sigma=np.log(np.std(lcurregi[indxtimeregioutt[i]])), log_rho=np.log(timescalbdtrspln))
            
            ## construct the GP model object
            objtgpro = celerite.GP(objtkern, mean=np.mean(lcurregi[indxtimeregioutt[i]]))
            
            # compute the covariance matrix
            objtgpro.compute(timeregi[indxtimeregioutt[i]], yerr=stdvlcurregi[indxtimeregioutt[i]])
            
            # get the initial parameters of the GP model
            #parainit = objtgpro.get_parameter_vector()
            
            # get the bounds on the GP model parameters
            #limtparagpro = objtgpro.get_parameter_bounds()
            
            # minimize the negative loglikelihood
            #objtmini = scipy.optimize.minimize(retr_lliknegagpro, parainit, jac=retr_gradlliknegagpro, method="L-BFGS-B", bounds=limtparagpro, args=(lcurregi[indxtimeregioutt[i]], objtgpro))
            
            #print('GP Matern 3/2 parameters with maximum likelihood:')
            #print(objtmini.x)

            # update the GP model with the parameters that minimize the negative loglikelihood
            #objtgpro.set_parameter_vector(objtmini.x)
            
            # get the GP model mean baseline
            lcurbase = objtgpro.predict(lcurregi[indxtimeregioutt[i]], t=timeregi, return_cov=False, return_var=False)#[0]
            
            # subtract the baseline from the data
            lcurbdtrregi[i] = 1. + lcurregi - lcurbase

            listobjtspln[i] = objtgpro
        if typebdtr == 'spln':
            # fit the spline
            if lcurregi[indxtimeregioutt[i]].size > 0:
                if timeregi[indxtimeregioutt[i]].size < 4:
                    print('Warning! Only %d points available for spline! This will result in a trivial baseline-detrended light curve (all 1s).' \
                                                                                                                % timeregi[indxtimeregioutt[i]].size)
                    print('numbregi')
                    print(numbregi)
                    print('indxtimeregioutt[i]')
                    summgene(indxtimeregioutt[i])
                    for ii in indxregi:
                        print('indxtimeregioutt[ii]')
                        summgene(indxtimeregioutt[ii])
                        
                    #raise Exception('')

                    listobjtspln[i] = None
                    lcurbdtrregi[i] = np.ones_like(lcurregi)
                else:
                    
                    minmtime = np.amin(timeregi[indxtimeregioutt[i]])
                    maxmtime = np.amax(timeregi[indxtimeregioutt[i]])
                    numbknot = int((maxmtime - minmtime) / timescalbdtrspln) + 1
                    
                    timeknot = np.linspace(minmtime, maxmtime, numbknot)
                    #timeknot = timeknot[1:-1]
                    numbknot = timeknot.size

                    indxknotregi = np.digitize(timeregi[indxtimeregioutt[i]], timeknot) - 1

                    if typeverb > 1:
                        print('minmtime')
                        print(minmtime)
                        print('maxmtime')
                        print(maxmtime)
                        print('timescalbdtrspln')
                        print(timescalbdtrspln)
                        print('%d knots used (exclduing the end points).' % (numbknot))
                        if numbknot > 1:
                            print('Knot separation: %.3g hours' % (24 * (timeknot[1] - timeknot[0])))
                    
                    if numbknot > 0:
                        objtspln = scipy.interpolate.LSQUnivariateSpline(timeregi[indxtimeregioutt[i]], lcurregi[indxtimeregioutt[i]], timeknot, k=ordrspln)
                        lcurbdtrregi[i] = lcurregi - objtspln(timeregi) + 1.
                        listobjtspln[i] = objtspln
                    else:
                        lcurbdtrregi[i] = lcurregi - np.median(lcurregi) + 1.
                        listobjtspln[i] = None
                    
            else:
                lcurbdtrregi[i] = lcurregi
                listobjtspln[i] = None
            
            if typeverb > 1:
                print('lcurbdtrregi[i]')
                summgene(lcurbdtrregi[i])
                print('')

    return lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge


def retr_brgtlmdk(cosg, coeflmdk, typelmdk='quad'):
    
    if typelmdk == 'linr':
        brgtlmdk = 1. - coeflmdk[0] * (1. - cosg)
    
    if typelmdk == 'quad':
        brgtlmdk = 1. - coeflmdk[0] * (1. - cosg) - coeflmdk[1] * (1. - cosg)**2
    
    if typelmdk == 'nlin':
        brgtlmdk = 1. - coeflmdk[0] * (1. - cosg) - coeflmdk[1] * (1. - cosg)**2
    
    if typelmdk == 'none':
        brgtlmdk = np.ones_like(cosg)

    return brgtlmdk


def retr_logg(radi, mass):
    
    logg = mass / radi**2

    return logg


def retr_noislsst(magtinpt):
    
    nois = np.zeros_like(magtinpt) + np.inf
    indx = np.where((magtinpt < 20.) & (magtinpt > 15.))
    nois[indx] = 6.
    indx = np.where((magtinpt >= 20.) & (magtinpt < 24.))
    nois[indx] = 6. * 10**((magtinpt[indx] - 20.) / 3.)
    
    return nois


def retr_noistess(magtinpt):
    
    nois = np.array([40., 40., 40., 90.,200.,700., 3e3, 2e4]) * 1e-3 # [ppt]
    magt = np.array([ 2.,  4.,  6.,  8., 10., 12., 14., 16.])
    objtspln = scipy.interpolate.interp1d(magt, nois, fill_value='extrapolate')
    nois = objtspln(magtinpt)
    
    return nois


def retr_tmag(gdat, cntp):
    
    tmag = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    #tmag = -2.5 * np.log10(mlikfluxtemp) + 20.424
    
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return tmag


@jit(nopython=True)
def srch_pbox_work_loop(m, phas, phasdiff, dydchalf):
    
    phasoffs = phas - phasdiff[m]
    
    if phasdiff[m] < dydchalf:
        booltemp = (phasoffs < dydchalf) | (1. - phas < dydchalf - phasoffs)
    elif 1. - phasdiff[m] < dydchalf:
        booltemp = (1. - phas - phasdiff[m] < dydchalf) | (phas < dydchalf - phasoffs)
    else:
        booltemp = np.abs(phasoffs) < dydchalf
    
    indxitra = np.where(booltemp)[0]
    
    return indxitra


def srch_pbox_work(listperi, listarrytser, listdcyc, listepoc, listduratrantotllevl, i):
    
    numbperi = len(listperi[i])
    numbdcyc = len(listdcyc[0])
    
    numblevlrebn = len(listduratrantotllevl)
    indxlevlrebn = np.arange(numblevlrebn)
    
    #conschi2 = np.sum(weig * arrytser[:, 1]**2)
    #listtermchi2 = np.empty(numbperi)
    
    rflxitraminm = np.zeros(numbperi) + 1e100
    dcycmaxm = np.zeros(numbperi)
    epocmaxm = np.zeros(numbperi)
    
    listphas = [[] for b in indxlevlrebn]
    for k in tqdm(range(len(listperi[i]))):
        
        peri = listperi[i][k]
        
        for b in indxlevlrebn:
            listphas[b] = (listarrytser[b][:, 0] % peri) / peri
        
        for l in range(len(listdcyc[k])):
            
            b = np.digitize(listdcyc[k][l] * peri * 24., listduratrantotllevl) - 1
            #b = 0
            
            #print('listduratrantotllevl')
            #print(listduratrantotllevl)
            #print('listdcyc[k][l] * peri * 24.')
            #print(listdcyc[k][l] * peri * 24.)
            #print('b')
            #print(b)

            dydchalf = listdcyc[k][l] / 2.

            phasdiff = (listepoc[k][l] % peri) / peri
            
            #print('listphas[b]')
            #summgene(listphas[b])
            #print('')
            
            for m in range(len(listepoc[k][l])):
                
                indxitra = srch_pbox_work_loop(m, listphas[b], phasdiff, dydchalf)
                
                if indxitra.size == 0:
                    continue
    
                rflxitra = np.mean(listarrytser[b][:, 1][indxitra])
                
                if rflxitra < rflxitraminm[k]:
                    rflxitraminm[k] = rflxitra
                    dcycmaxm[k] = listdcyc[k][l]
                    epocmaxm[k] = listepoc[k][l][m]
                
                if not np.isfinite(rflxitra):
                    print('b')
                    print(b)
                    print('listarrytser[b][:, 1]')
                    summgene(listarrytser[b][:, 1])
                    #print('deptcomp')
                    #print(dept)
                    #print('np.std(rflx[indxitra])')
                    #summgene(np.std(rflx[indxitra]))
                    #print('rflx[indxitra]')
                    #summgene(rflx[indxitra])
                    raise Exception('')
                    
                #timechecloop[0][k, l, m] = timemodu.time()
                #print('pericomp')
                #print(peri)
                #print('dcyc')
                #print(dcyc)
                #print('epocmtracomp')
                #print(epoc)
                #print('phasdiff')
                #summgene(phasdiff)
                #print('phasoffs')
                #summgene(phasoffs)
                
                #print('booltemp')
                #summgene(booltemp)
                #print('indxitra')
                #summgene(indxitra)
                #print('deptcomp')
                #print(dept)
                #print('stdv')
                #print(stdv)
                #terr = np.sum(weig[indxitra])
                #ters = np.sum(weig[indxitra] * rflx[indxitra])
                #termchi2 = ters**2 / terr / (1. - terr)
                #print('ters')
                #print(ters)
                #print('terr')
                #print(terr)
                #print('deptcomp')
                #print(dept)
                #print('indxitra')
                #summgene(indxitra)
                #print('s2nr')
                #print(s2nr)
                #print('')
                
                #if True:
                if False:
                    figr, axis = plt.subplots(2, 1, figsize=(8, 8))
                    axis[0].plot(listarrytser[b][:, 0], listarrytser[b][:, 1], color='b', ls='', marker='o', rasterized=True, ms=0.3)
                    axis[0].plot(listarrytser[b][:, 0][indxitra], listarrytser[b][:, 1][indxitra], color='r', ls='', marker='o', ms=2., rasterized=True)
                    axis[0].axhline(1., ls='-.', alpha=0.3, color='k')
                    axis[0].set_xlabel('Time [BJD]')
                    
                    axis[1].plot(listphas[b], listarrytser[b][:, 1], color='b', ls='', marker='o', rasterized=True, ms=0.3)
                    axis[1].plot(listphas[b][indxitra], listarrytser[b][:, 1][indxitra], color='r', ls='', marker='o', ms=2., rasterized=True)
                    axis[1].plot(np.mean(listphas[b][indxitra]), rflxitra, color='g', ls='', marker='o', ms=4., rasterized=True)
                    axis[1].axhline(1., ls='-.', alpha=0.3, color='k')
                    axis[1].set_xlabel('Phase')
                    titl = '$P$=%.3f, $T_0$=%.3f, $q_{tr}$=%.3g, $f$=%.6g' % (peri, listepoc[k][l][m], listdcyc[k][l], rflxitra)
                    axis[0].set_title(titl, usetex=False)
                    path = '/Users/tdaylan/Documents/work/data/troia/toyy_tessprms2min_TESS/mock0001/imag/rflx_tria_diag_%04d%04d.pdf' % (l, m)
                    print('Writing to %s...' % path)
                    plt.savefig(path, usetex=False)
                    plt.close()
        
    return rflxitraminm, dcycmaxm, epocmaxm


def retr_stdvwind(ydat, sizewind, boolcuttpeak=True):
    '''
    Return the standard deviation of a series inside a running windown.
    '''
    
    numbdata = ydat.size
    
    if sizewind % 2 != 1 or sizewind > numbdata:
        raise Exception('')

    sizewindhalf = int((sizewind - 1) / 2)
    
    indxdata = np.arange(numbdata)
    
    stdv = np.empty_like(ydat)
    for k in indxdata:
        


        minmindx = max(0, k - sizewindhalf)
        maxmindx = min(numbdata - 1, k + sizewindhalf)
        
        if boolcuttpeak:
            indxdatawind = np.arange(minmindx, maxmindx+1)
            #indxdatawind = indxdatawind[np.where(ydat[indxdatawind] < np.percentile(ydat[indxdatawind], 99.999))]
            indxdatawind = indxdatawind[np.where(ydat[indxdatawind] != np.amax(ydat[indxdatawind]))]
        
        else:
            if k > minmindx and k+1 < maxmindx:
                indxdatawind = np.concatenate((np.arange(minmindx, k), np.arange(k+1, maxmindx+1)))
            elif k > minmindx:
                indxdatawind = np.arange(minmindx, k)
            elif k+1 < maxmindx:
                indxdatawind = np.arange(k+1, maxmindx+1)

        stdv[k] = np.std(ydat[indxdatawind])
    
    return stdv


def srch_pbox(arry, \
              
              ### maximum number of transiting objects
              maxmnumbpbox=1, \
              
              ticitarg=None, \
              
              dicttlsqinpt=None, \
              booltlsq=False, \
              
              # minimum period
              minmperi=None, \

              # maximum period
              maxmperi=None, \

              # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
              factduracade=2., \

              # factor by which to oversample the frequency grid
              factosam=1., \
              
              # Boolean flag to search for positive boxes
              boolsrchposi=False, \

              # number of duty cycle samples  
              numbdcyc=3, \
              
              # spread in the logarithm of duty cycle
              deltlogtdcyc=None, \
              
              # density of the star
              densstar=None, \

              # epoc steps divided by trial duration
              factdeltepocdura=0.5, \

              # detection threshold
              thrssdee=7.1, \
              
              # number of processes
              numbproc=None, \
              
              # Boolean flag to enable multiprocessing
              boolprocmult=False, \
              
              # string extension to output files
              strgextn='', \
              # path where the output data will be stored
              pathdata=None, \

              # plotting
              ## path where the output images will be stored
              pathimag=None, \
              ## file type of the plot
              typefileplot='pdf', \
              ## figure size
              figrsizeydobskin=(8, 2.5), \
              ## time offset
              timeoffs=0, \
              ## data transparency
              alphraww=0.2, \
              
              # verbosity level
              typeverb=1, \
              
              # Boolean flag to turn on diagnostic mode
              booldiag=True, \

              # Boolean flag to force rerun and overwrite previous data and plots 
              boolover=True, \

             ):
    '''
    Search for periodic boxes in time-series data.
    '''
    
    boolproc = False
    listnameplot = ['sigr', 'resisigr', 'stdvresisigr', 'sdeecomp', 'rflx', 'pcur']
    if pathdata is None:
        boolproc = True
    else:
        if strgextn == '':
            pathsave = pathdata + 'pbox.csv'
        else:
            pathsave = pathdata + 'pbox_%s.csv' % strgextn
        if not os.path.exists(pathsave):
            boolproc = True
        
        dictpathplot = dict()
        for strg in listnameplot:
            dictpathplot[strg] = []
            
        if os.path.exists(pathsave):
            if typeverb > 0:
                print('Reading from %s...' % pathsave)
            
            dictpboxoutp = pd.read_csv(pathsave).to_dict(orient='list')
            for name in dictpboxoutp.keys():
                dictpboxoutp[name] = np.array(dictpboxoutp[name])
                if len(dictpboxoutp[name]) == 0:
                    dictpboxoutp[name] = np.array([])
            
            if not pathimag is None:
                for strg in listnameplot:
                    for j in range(len(dictpboxoutp['pericomp'])):
                        dictpathplot[strg].append(pathimag + strg + '_pbox_tce%d_%s.%s' % (j, strgextn, typefileplot))
         
                        if not os.path.exists(dictpathplot[strg][j]):
                            boolproc = True
            
    if boolproc:
        dictpboxoutp = dict()
        if pathimag is not None:
            for name in listnameplot:
                dictpboxoutp['listpathplot%s' % name] = []
    
        print('Searching for periodic boxes in time-series data...')
        
        print('factosam')
        print(factosam)
        if booltlsq:
            import transitleastsquares
            if dicttlsqinpt is None:
                dicttlsqinpt = dict()
        
        # setup TLS
        # temp
        #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(ticitarg))
        
        dictpboxinte = dict()
        liststrgvarbsave = ['pericomp', 'epocmtracomp', 'deptcomp', 'duracomp', 'sdeecomp']
        for strg in liststrgvarbsave:
            dictpboxoutp[strg] = []
        
        arrysrch = np.copy(arry)
        if boolsrchposi:
            arrysrch[:, 1] = 2. - arrysrch[:, 1]

        j = 0
        
        timeinit = timemodu.time()

        dictfact = retr_factconv()
        
        numbtime = arrysrch[:, 0].size
        
        minmtime = np.amin(arrysrch[:, 0])
        maxmtime = np.amax(arrysrch[:, 0])
        #arrysrch[:, 0] -= minmtime

        delttime = maxmtime - minmtime
        deltfreq = 0.1 / delttime / factosam
        
        print('Initial:')
        print('minmperi')
        print(minmperi)
        print('maxmperi')
        print(maxmperi)
        #raise Exception('')

        if maxmperi is None:
            minmfreq = 2. / delttime
        else:
            minmfreq = 1. / maxmperi

        if minmperi is None:
            maxmfreq = 1. / 0.5 # 0.5 days
        else:
            maxmfreq = 1. / minmperi

        listfreq = np.arange(minmfreq, maxmfreq, deltfreq)
        listperi = 1. / listfreq
        
        if pathimag is not None:
            numbtimeplot = 100000
            timemodlplot = np.linspace(minmtime, maxmtime, numbtimeplot)
        
        numbperi = listperi.size
        if numbperi < 3:
            print('maxmperi')
            print(maxmperi)
            print('minmperi')
            print(minmperi)
            print('numbperi')
            print(numbperi)
            raise Exception('')

        indxperi = np.arange(numbperi)
        minmperi = np.amin(listperi)
        maxmperi = np.amax(listperi)
        print('minmperi')
        print(minmperi)
        print('maxmperi')
        print(maxmperi)
        
        indxdcyc = np.arange(numbdcyc)
        listdcyc = [[] for k in indxperi]
        listperilogt = np.log10(listperi)
        
        if deltlogtdcyc is None:
            deltlogtdcyc = np.log10(2.)
        
        # assuming Solar density
        maxmdcyclogt = -2. / 3. * listperilogt - 1. + deltlogtdcyc
        if densstar is not None:
            maxmdcyclogt += -1. / 3. * np.log10(densstar)

        minmdcyclogt = maxmdcyclogt - 2. * deltlogtdcyc
        for k in indxperi:
            listdcyc[k] = np.logspace(minmdcyclogt[k], maxmdcyclogt[k], numbdcyc)
        print('Trial transit duty cycles at the smallest period')
        print(listdcyc[-1])
        print('Trial transit durations at the smallest period [hr]')
        print(listdcyc[-1] * listperi[-1] * 24)
        print('Trial transit duty cycles at the largest period')
        print(listdcyc[0])
        print('Trial transit durations at the largest period [hr]')
        print(listdcyc[0] * listperi[0] * 24)

        # cadence
        cade = np.amin(arrysrch[1:, 0] - arrysrch[:-1, 0]) * 24. # [hr]
        
        # minimum transit duration
        minmduratrantotl = listdcyc[-1][0] * listperi[-1] * 24
        
        # maximum transit duration
        maxmduratrantotl = listdcyc[0][-1] * listperi[0] * 24
        
        if minmduratrantotl < factduracade * cade:
            print('Either the minimum transit duration is too small or the cadence is too large.')
            print('minmduratrantotl')
            print(minmduratrantotl)
            print('factduracade')
            print(factduracade)
            print('cade [hr]')
            print(cade)
            raise Exception('')
        
        # number of rebinned data sets
        numblevlrebn = 10
        indxlevlrebn = np.arange(numblevlrebn)
        
        # list of transit durations when rebinned data sets will be used
        listduratrantotllevl = np.linspace(minmduratrantotl, maxmduratrantotl, numblevlrebn)
        
        print('listduratrantotllevl')
        print(listduratrantotllevl)
        
        # rebinned data sets
        print('Number of data points: %d...' % numbtime)
        listarrysrch = []
        for b in indxlevlrebn:
            delt = listduratrantotllevl[b] / 24. / factduracade
            arryrebn = rebn_tser(arrysrch, delt=delt)
            indx = np.where(np.isfinite(arryrebn[:, 1]))[0]
            print('Number of data points in binned data set for Delta time %g [min]: %d' % (delt * 24. * 60., arryrebn.shape[0]))
            arryrebn = arryrebn[indx, :]
            listarrysrch.append(arryrebn)
            print('Number of data points in binned data set for Delta time %g [min]: %d' % (delt * 24. * 60., arryrebn.shape[0]))
            print('')
        listepoc = [[[] for l in range(numbdcyc)] for k in indxperi]
        numbtria = np.zeros(numbperi, dtype=int)
        for k in indxperi:
            for l in indxdcyc:
                diffepoc = max(cade / 24., factdeltepocdura * listperi[k] * listdcyc[k][l])
                listepoc[k][l] = np.arange(minmtime, minmtime + listperi[k], diffepoc)
                numbtria[k] += len(listepoc[k][l])
                
        dflx = arrysrch[:, 1] - 1.
        stdvdflx = arrysrch[:, 2]
        varidflx = stdvdflx**2
        
        print('Number of trial periods: %d...' % numbperi)
        print('Number of trial computations for the smallest period: %d...' % numbtria[-1])
        print('Number of trial computations for the largest period: %d...' % numbtria[0])
        print('Total number of trial computations: %d...' % np.sum(numbtria))

        while True:
            
            if maxmnumbpbox is not None and j >= maxmnumbpbox:
                break
            
            print('j')
            print(j)

            # mask out the detected transit
            if j > 0:
                ## remove previously detected periodic box from the rebinned data
                pericomp = [dictpboxoutp['pericomp'][j]]
                epocmtracomp = [dictpboxoutp['epocmtracomp'][j]]
                radicomp = [dictfact['rsre'] * np.sqrt(dictpboxoutp['deptcomp'][j] * 1e-3)]
                cosicomp = [0]
                rsmacomp = [retr_rsmacomp(dictpboxoutp['pericomp'][j], dictpboxoutp['duracomp'][j], cosicomp[0])]
                    
                for b in indxlevlrebn:
                    ## evaluate model at all resolutions
                    dictoutp = retr_rflxtranmodl(listarrysrch[b][:, 0], pericomp=pericomp, epocmtracomp=epocmtracomp, \
                                                                                        rsmacomp=rsmacomp, cosicomp=cosicomp, rratcomp=rratcomp)
                    ## subtract it from data
                    listarrysrch[b][:, 1] -= (dictoutp['rflx'][b] - 1.)
                
                    if (dictpboxinte['rflx'][b] == 1.).all():
                        raise Exception('')

            if booltlsq:
                objtmodltlsq = transitleastsquares.transitleastsquares(arrysrch[:, 0], lcurpboxmeta)
                objtresu = objtmodltlsq.power(\
                                              # temp
                                              #u=ab, \
                                              **dicttlsqinpt, \
                                              #use_threads=1, \
                                             )

                dictpbox = dict()
                dictpboxinte['listperi'] = objtresu.periods
                dictpboxinte['listsigr'] = objtresu.power
                
                dictpboxoutp['pericomp'].append(objtresu.period)
                dictpboxoutp['epocmtracomp'].append(objtresu.T0)
                dictpboxoutp['duracomp'].append(objtresu.duration)
                dictpboxoutp['deptcomp'].append(objtresu.depth * 1e3)
                dictpboxoutp['sdeecomp'].append(objtresu.SDE)
                dictpboxoutp['prfp'].append(objtresu.FAP)
                
                if objtresu.SDE < thrssdee:
                    break
                
                dictpboxinte['rflxtsermodl'] = objtresu.model_lightcurve_model
                
                if pathimag is not None:
                    dictpboxinte['listtimetran'] = objtresu.transit_times
                    dictpboxinte['timemodl'] = objtresu.model_lightcurve_time
                    dictpboxinte['phasmodl'] = objtresu.model_folded_phase
                    dictpboxinte['rflxpsermodl'] = objtresu.model_folded_model
                    dictpboxinte['phasdata'] = objtresu.folded_phase
                    dictpboxinte['rflxpserdata'] = objtresu.folded_y

            else:
                
                if boolprocmult:
                    
                    if numbproc is None:
                        #numbproc = multiprocessing.cpu_count() - 1
                        numbproc = int(0.8 * multiprocessing.cpu_count())
                    
                    print('Generating %d processes...' % numbproc)
                    
                    objtpool = multiprocessing.Pool(numbproc)
                    numbproc = objtpool._processes
                    indxproc = np.arange(numbproc)

                    listperiproc = [[] for i in indxproc]
                    
                    binsperiproc = tdpy.icdf_powr(np.linspace(0., 1., numbproc + 1)[1:-1], np.amin(listperi), np.amax(listperi), 1.97)
                    binsperiproc = np.concatenate((np.array([-np.inf]), binsperiproc, np.array([np.inf])))
                    indxprocperi = np.digitize(listperi, binsperiproc, right=False) - 1
                    for i in indxproc:
                        indx = np.where(indxprocperi == i)[0]
                        listperiproc[i] = listperi[indx]
                    data = objtpool.map(partial(srch_pbox_work, listperiproc, listarrysrch, listdcyc, listepoc, listduratrantotllevl), indxproc)
                    listrflxitra = np.concatenate([data[k][0] for k in indxproc])
                    listdeptmaxm = np.concatenate([data[k][1] for k in indxproc])
                    listdcycmaxm = np.concatenate([data[k][2] for k in indxproc])
                    listepocmaxm = np.concatenate([data[k][3] for k in indxproc])
                else:
                    listrflxitra, listdcycmaxm, listepocmaxm = srch_pbox_work([listperi], listarrysrch, listdcyc, listepoc, listduratrantotllevl, 0)
                
                listdept = (np.median(listarrysrch[b][:, 1]) - listrflxitra) * 1e3 # [ppt])
                listsigr = listdept
                if (~np.isfinite(listsigr)).any():
                    raise Exception('')

                sizekern = 51
                listresisigr = listsigr - scipy.ndimage.median_filter(listsigr, size=sizekern)
                #listresisigr = listresisigr**2
                liststdvresisigr = retr_stdvwind(listresisigr, sizekern, boolcuttpeak=True)
                listsdee = listresisigr / liststdvresisigr
                #listsdee -= np.amin(listsdee)
                
                indxperimpow = np.argmax(listsdee)
                sdee = listsdee[indxperimpow]
                
                if not np.isfinite(sdee):
                    print('Warning! SDE is infinite! Making it zero.')
                    sdee = 0.
                    print('arry')
                    summgene(arry)
                    for b in indxlevlrebn:
                        print('listarrysrch[b]')
                        summgene(listarrysrch[b])
                    print('listsigr')
                    summgene(listsigr)
                    indxperizerostdv = np.where(liststdvresisigr == 0)[0]
                    print('indxperizerostdv')
                    summgene(indxperizerostdv)
                    print('liststdvresisigr')
                    summgene(liststdvresisigr)
                    #raise Exception('')

                dictpboxoutp['sdeecomp'].append(sdee)
                dictpboxoutp['pericomp'].append(listperi[indxperimpow])
                dictpboxoutp['duracomp'].append(24. * listdcycmaxm[indxperimpow] * listperi[indxperimpow]) # [hours]
                dictpboxoutp['epocmtracomp'].append(listepocmaxm[indxperimpow])
                dictpboxoutp['deptcomp'].append(listdept[indxperimpow])
                
                print('sdeecomp')
                print(sdee)

                # best-fit orbit
                dictpboxinte['listperi'] = listperi
                
                print('temp: assuming power is SNR')
                dictpboxinte['listsigr'] = listsigr
                dictpboxinte['listresisigr'] = listresisigr
                dictpboxinte['liststdvresisigr'] = liststdvresisigr
                dictpboxinte['listsdeecomp'] = listsdee
                
                # to be deleted because these are rebinned and model may be all 1s
                #if booldiag and (dictpboxinte['rflxtsermodl'][b] == 1).all():
                #    print('listarrysrch[b][:, 0]')
                #    summgene(listarrysrch[b][:, 0])
                #    print('radistar')
                #    print(radistar)
                #    print('pericomp')
                #    print(pericomp)
                #    print('epocmtracomp')
                #    print(epocmtracomp)
                #    print('rsmacomp')
                #    print(rsmacomp)
                #    print('cosicomp')
                #    print(cosicomp)
                #    print('radicomp')
                #    print(radicomp)
                #    print('dictpboxinte[rflxtsermodl[b]]')
                #    summgene(dictpboxinte['rflxtsermodl'][b])
                #    raise Exception('')

                if pathimag is not None:
                    for strg in listnameplot:
                        for j in range(len(dictpboxoutp['pericomp'])):
                            pathplot = pathimag + strg + '_pbox_tce%d_%s.%s' % (j, strgextn, typefileplot)
                            dictpathplot[strg].append(pathplot)
            
                    pericomp = [dictpboxoutp['pericomp'][j]]
                    epocmtracomp = [dictpboxoutp['epocmtracomp'][j]]
                    cosicomp = [0]
                    rsmacomp = [retr_rsmacomp(dictpboxoutp['pericomp'][j], dictpboxoutp['duracomp'][j], cosicomp[0])]
                    rratcomp = [np.sqrt(dictpboxoutp['deptcomp'][j] * 1e-3)]
                    dictoutp = retr_rflxtranmodl(timemodlplot, pericomp=pericomp, epocmtracomp=epocmtracomp, \
                                                                                            rsmacomp=rsmacomp, cosicomp=cosicomp, rratcomp=rratcomp, typesyst='psys')
                    dictpboxinte['rflxtsermodl'] = dictoutp['rflx']
                    
                    arrymetamodl = np.zeros((numbtimeplot, 3))
                    arrymetamodl[:, 0] = timemodlplot
                    arrymetamodl[:, 1] = dictpboxinte['rflxtsermodl']
                    arrypsermodl = fold_tser(arrymetamodl, dictpboxoutp['epocmtracomp'][j], dictpboxoutp['pericomp'][j], phasshft=0.5)
                    arrypserdata = fold_tser(listarrysrch[0], dictpboxoutp['epocmtracomp'][j], dictpboxoutp['pericomp'][j], phasshft=0.5)
                        
                    dictpboxinte['timedata'] = listarrysrch[0][:, 0]
                    dictpboxinte['rflxtserdata'] = listarrysrch[0][:, 1]
                    dictpboxinte['phasdata'] = arrypserdata[:, 0]
                    dictpboxinte['rflxpserdata'] = arrypserdata[:, 1]

                    dictpboxinte['timemodl'] = arrymetamodl[:, 0]
                    dictpboxinte['phasmodl'] = arrypsermodl[:, 0]
                    dictpboxinte['rflxpsermodl'] = arrypsermodl[:, 1]
                
                    print('boolsrchposi')
                    print(boolsrchposi)
                    if boolsrchposi:
                        dictpboxinte['rflxpsermodl'] = 2. - dictpboxinte['rflxpsermodl']
                        dictpboxinte['rflxtsermodl'] = 2. - dictpboxinte['rflxtsermodl']
                        dictpboxinte['rflxpserdata'] = 2. - dictpboxinte['rflxpserdata']
           
            if pathimag is not None:
                strgtitl = 'P=%.3f d, $T_0$=%.3f, Dep=%.2g ppt, Dur=%.2g hr, SDE=%.3g' % \
                            (dictpboxoutp['pericomp'][j], dictpboxoutp['epocmtracomp'][j], dictpboxoutp['deptcomp'][j], dictpboxoutp['duracomp'][j], dictpboxoutp['sdeecomp'][j])
                
                # plot power spectra
                for a in range(4):
                    if a == 0:
                        strg = 'sigr'
                    if a == 1:
                        strg = 'resisigr'
                    if a == 2:
                        strg = 'stdvresisigr'
                    if a == 3:
                        strg = 'sdeecomp'

                    figr, axis = plt.subplots(figsize=figrsizeydobskin)
                    
                    axis.axvline(dictpboxoutp['pericomp'][j], alpha=0.4, lw=3)
                    minmxaxi = np.amin(dictpboxinte['listperi'])
                    maxmxaxi = np.amax(dictpboxinte['listperi'])
                    for n in range(2, 10):
                        xpos = n * dictpboxoutp['pericomp'][j]
                        if xpos > maxmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    for n in range(2, 10):
                        xpos = dictpboxoutp['pericomp'][j] / n
                        if xpos < minmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    
                    axis.set_ylabel('Power')
                    axis.set_xlabel('Period [days]')
                    axis.set_xscale('log')
                    axis.plot(dictpboxinte['listperi'], dictpboxinte['list' + strg], color='black', lw=0.5)
                    axis.set_title(strgtitl)
                    plt.subplots_adjust(bottom=0.2)
                    path = dictpathplot[strg][j]
                    dictpboxoutp['listpathplot%s' % strg].append(path)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                # plot data and model time-series
                figr, axis = plt.subplots(figsize=figrsizeydobskin)
                lcurpboxmeta = listarrysrch[0][:, 1]
                if boolsrchposi:
                    lcurpboxmetatemp = 2. - lcurpboxmeta
                else:
                    lcurpboxmetatemp = lcurpboxmeta
                axis.plot(listarrysrch[0][:, 0] - timeoffs, lcurpboxmetatemp, alpha=alphraww, marker='o', ms=1, ls='', color='grey')
                axis.plot(dictpboxinte['timemodl'] - timeoffs, dictpboxinte['rflxtsermodl'], color='b')
                if timeoffs == 0:
                    axis.set_xlabel('Time [days]')
                else:
                    axis.set_xlabel('Time [BJD-%d]' % timeoffs)
                axis.set_ylabel('Relative flux');
                if j == 0:
                    ylimtserinit = axis.get_ylim()
                else:
                    axis.set_ylim(ylimtserinit)
                axis.set_title(strgtitl)
                plt.subplots_adjust(bottom=0.2)
                path = dictpathplot['rflx'][j]
                dictpboxoutp['listpathplotrflx'].append(path)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()

                # plot data and model phase-series
                figr, axis = plt.subplots(figsize=figrsizeydobskin)
                axis.plot(dictpboxinte['phasdata'], dictpboxinte['rflxpserdata'], marker='o', ms=1, ls='', alpha=alphraww, color='grey')
                axis.plot(dictpboxinte['phasmodl'], dictpboxinte['rflxpsermodl'], color='b')
                axis.set_xlabel('Phase')
                axis.set_ylabel('Relative flux');
                if j == 0:
                    ylimpserinit = axis.get_ylim()
                else:
                    axis.set_ylim(ylimpserinit)
                axis.set_title(strgtitl)
                plt.subplots_adjust(bottom=0.2)
                path = dictpathplot['pcur'][j]
                dictpboxoutp['listpathplotpcur'].append(path)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()
            
            print('dictpboxoutp[sdee]')
            print(dictpboxoutp['sdeecomp'])
            print('thrssdee')
            print(thrssdee)
            j += 1
        
            if sdee < thrssdee or indxperimpow == listsdee.size - 1:
                break
        
        # make the BLS features arrays
        for name in dictpboxoutp.keys():
            dictpboxoutp[name] = np.array(dictpboxoutp[name])
        
        print('dictpboxoutp')
        print(dictpboxoutp)
        pd.DataFrame.from_dict(dictpboxoutp).to_csv(pathsave, index=False)
                
        timefinl = timemodu.time()
        timetotl = timefinl - timeinit
        timeredu = timetotl / numbtime / np.sum(numbtria)
        
        print('srch_pbox() took %.3g seconds in total and %g ns per observation and trial.' % (timetotl, timeredu * 1e9))

    return dictpboxoutp


def retr_subp(dictpopl, dictnumbsamp, dictindxsamp, namepoplinit, namepoplfinl, indx):
    
    if isinstance(indx, list):
        raise Exception('')

    if len(indx) == 0:
        indx = np.array([], dtype=int)

    dictindxsamp[namepoplinit][namepoplfinl] = indx
    
    if indx.size == 0:
        print('Warning! indx is zero.')

    dictpopl[namepoplfinl] = dict()
    for name in dictpopl[namepoplinit].keys():
        
        #print('')
        #print('')
        #print('')
        #print('')
        #print('')
        #print('')
        #print('')
        #print('')
        #print('')
        #print('dictpopl[namepoplfinl]')
        #summgene(dictpopl[namepoplfinl])
        #print('')
        #print('dictindxsamp[namepoplinit]')
        #summgene(dictindxsamp[namepoplinit])
        #print('')
        #print('name')
        #print(name)
        #print('')
        #print('dictpopl[namepoplinit][name]')
        #summgene(dictpopl[namepoplinit][name])
        #print('')
        #print('dictindxsamp[namepoplinit][namepoplfinl] (this is indx)')
        #summgene(dictindxsamp[namepoplinit][namepoplfinl])
        
        if indx.size > 0:
            dictpopl[namepoplfinl][name] = dictpopl[namepoplinit][name][dictindxsamp[namepoplinit][namepoplfinl]]
        else:
            dictpopl[namepoplfinl][name] = np.array([])

    dictnumbsamp[namepoplfinl] = dictindxsamp[namepoplinit][namepoplfinl].size
    dictindxsamp[namepoplfinl] = dict()


def retr_dictpoplstarcomp( \
                          # type of target systems
                          typesyst, \
                          
                          # type of the population of target systems
                          typepoplsyst, \
                          
                          # number of systems
                          numbsyst=None, \
                          
                          # epochs of mid-transits
                          epocmtracomp=0.5, \
                          
                          # offset for mid-transit epochs
                          timeepoc=None, \
                          
                          # type of sampling of orbital period and semi-major axes
                          ## 'smax': semi-major axes are sampled first and then orbital periods are calculated
                          ## 'peri': orbital periods are sampled first and then semi-major axies are calculated
                          typesamporbtcomp='smax', \

                          # minimum ratio of semi-major axis to radius of the host star
                          minmsmaxradistar=3., \
                          
                          # maximum ratio of semi-major axis to radius of the host star
                          maxmsmaxradistar=1e4, \
                          
                          # minimum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          minmpericomp=0.1, \
                          
                          # maximum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          maxmpericomp=1000., \
                          
                          # Boolean flag to include exomoons
                          boolinclmoon=False, \
                          
                          # Boolean flag to make the generative model produce Suns
                          booltoyysunn=False, \
                          
                          # Boolean flag to diagnose
                          booldiag=True, \
                          
                         ):
    '''
    Sample a synthetic population of the features of companions (e.g., exoplanets )and the companions to companions (e.g., exomoons) 
    hosted by a specified or random population of stellar systems.
    '''
    
    print('typesyst')
    print(typesyst)
    print('typepoplsyst')
    print(typepoplsyst)
    
    # dictionary keys of the populations
    namepoplstartotl = 'star' + typepoplsyst + 'totl'
    namepoplstaroccu = 'star' + typepoplsyst + 'occu'
    namepoplcomptotl = 'compstar' + typepoplsyst + 'totl'
    namepoplcomptran = 'compstar' + typepoplsyst + 'tran'
    
    namepoplmoontotl = 'mooncompstar' + typepoplsyst + 'totl'
    
    # Boolean flag indicating if the system is a compact object transiting a stellar companion
    boolsystcosctran = typesyst == 'cosctran'
    # Boolean flag indicating if the system is a compact object with stellar companion
    boolsystcosc = typesyst == 'cosc' or typesyst == 'cosctran'

    dictpoplstar = dict()
    dictstarnumbsamp = dict()
    dictstarindxsamp = dict()
    dictstarnumbsamp[namepoplstartotl] = dict()
    dictstarindxsamp[namepoplstartotl] = dict()
    
    dictpoplcomp = dict()
    dictcompnumbsamp = dict()
    dictcompindxsamp = dict()
    dictcompnumbsamp[namepoplcomptotl] = dict()
    dictcompindxsamp[namepoplcomptotl] = dict()
    dictpoplcomp[namepoplcomptotl] = dict()
    
    dictpoplmoon = dict()
    dictmoonnumbsamp = dict()
    dictmoonindxsamp = dict()
    dictmoonnumbsamp[namepoplmoontotl] = dict()
    dictmoonindxsamp[namepoplmoontotl] = dict()
    dictpoplmoon[namepoplmoontotl] = dict()
    
    dictfact = retr_factconv()
    
    # get the features of the star population
    if typepoplsyst.startswith('tess') or typepoplsyst.startswith('tici'):
        dictpoplstar[namepoplstartotl] = retr_dictpopltic8(typepoplsyst, numbsyst=numbsyst)
        
        print('Removing stars that do not have radii or masses...')
        indx = np.where(np.isfinite(dictpoplstar[namepoplstartotl]['radistar']) & \
                        np.isfinite(dictpoplstar[namepoplstartotl]['massstar']))[0]
        for name in dictpoplstar[namepoplstartotl].keys():
            dictpoplstar[namepoplstartotl][name] = dictpoplstar[namepoplstartotl][name][indx]

        if (dictpoplstar[namepoplstartotl]['rascstar'] > 1e4).any():
            raise Exception('')

        if (dictpoplstar[namepoplstartotl]['radistar'] == 0.).any():
            raise Exception('')

        dictpoplstar[namepoplstartotl]['densstar'] = 1.41 * dictpoplstar[namepoplstartotl]['massstar'] / dictpoplstar[namepoplstartotl]['radistar']**3
        dictpoplstar[namepoplstartotl]['idenstar'] = dictpoplstar[namepoplstartotl]['tici']
    

    elif typepoplsyst == 'gene':
        
        if numbsyst is None:
            numbsyst = 10000
        
        dictpoplstar[namepoplstartotl] = dict()
        
        dictpoplstar[namepoplstartotl]['idenstar'] = np.arange(numbsyst)
        
        dictpoplstar[namepoplstartotl]['distsyst'] = tdpy.icdf_powr(np.random.rand(numbsyst), 100., 7000., -2.)
        
        if booltoyysunn:
            dictpoplstar[namepoplstartotl]['radistar'] = np.ones(numbsyst)
            dictpoplstar[namepoplstartotl]['massstar'] = np.ones(numbsyst)
            dictpoplstar[namepoplstartotl]['densstar'] = 1.4 * np.ones(numbsyst)
        else:
            dictpoplstar[namepoplstartotl]['massstar'] = tdpy.icdf_powr(np.random.rand(numbsyst), 0.1, 10., 2.)
            dictpoplstar[namepoplstartotl]['densstar'] = 1.4 * (1. / dictpoplstar[namepoplstartotl]['massstar'])**(0.7)
            dictpoplstar[namepoplstartotl]['radistar'] = (1.4 * dictpoplstar[namepoplstartotl]['massstar'] / dictpoplstar[namepoplstartotl]['densstar'])**(1. / 3.)
        
        dictpoplstar[namepoplstartotl]['lumistar'] = dictpoplstar[namepoplstartotl]['massstar']**4
        
        dictpoplstar[namepoplstartotl]['tmag'] = 1. * (-2.5) * np.log10(dictpoplstar[namepoplstartotl]['lumistar'] / dictpoplstar[namepoplstartotl]['distsyst']**2)
        
        if typepoplsyst == 'lsstwfds':
            dictpoplstar[namepoplstartotl]['rmag'] = -2.5 * np.log10(dictpoplstar[namepoplstartotl]['lumistar'] / dictpoplstar[namepoplstartotl]['distsyst']**2)
            
            indx = np.where((dictpoplstar[namepoplstartotl]['rmag'] < 24.) & (dictpoplstar[namepoplstartotl]['rmag'] > 15.))[0]
            for name in ['distsyst', 'rmag', 'massstar', 'densstar', 'radistar', 'lumistar']:
                dictpoplstar[namepoplstartotl][name] = dictpoplstar[namepoplstartotl][name][indx]

    else:
        print('typepoplsyst')
        print(typepoplsyst)
        raise Exception('')
    
    dictstarnumbsamp[namepoplstartotl] = dictpoplstar[namepoplstartotl]['radistar'].size

    # probability of occurence
    if boolsystcosc or typesyst == 'sbin':
        dictpoplstar[namepoplstartotl]['numbcompstarmean'] = np.empty_like(dictpoplstar[namepoplstartotl]['radistar']) + np.nan
    
    if typesyst == 'psys' or typesyst == 'psysmoon':
        
        #masstemp = np.copy(dictpoplstar[namepoplstartotl]['massstar'])
        #masstemp[np.where(~np.isfinite(masstemp))] = 1.
        
        # mean number of companions per star
        dictpoplstar[namepoplstartotl]['numbcompstarmean'] = 0.5 * dictpoplstar[namepoplstartotl]['massstar']**(-1.)
        
        # number of companions per star
        dictpoplstar[namepoplstartotl]['numbcompstar'] = np.random.poisson(dictpoplstar[namepoplstartotl]['numbcompstarmean'])
    
    elif typesyst == 'psyspcur' or typesyst == 'cosc' or typesyst == 'sbin':
        # number of companions per star
        dictpoplstar[namepoplstartotl]['numbcompstar'] = np.ones(dictpoplstar[namepoplstartotl]['radistar'].size).astype(int)
    else:
        print('typesyst')
        print(typesyst)
        raise Exception('')

    # Boolean flag of occurence
    dictpoplstar[namepoplstartotl]['booloccu'] = dictpoplstar[namepoplstartotl]['numbcompstar'] > 0
    
    # subpopulation where companions occur
    indx = np.where(dictpoplstar[namepoplstartotl]['booloccu'])[0]
    print('namepoplstartotl')
    print(namepoplstartotl)
    print('namepoplstaroccu')
    print(namepoplstaroccu)
    retr_subp(dictpoplstar, dictstarnumbsamp, dictstarindxsamp, namepoplstartotl, namepoplstaroccu, indx)
    
    if boolsystcosc:
        minmmasscomp = 5. # [Solar mass]
        maxmmasscomp = 200. # [Solar mass]
    elif typesyst == 'psys' or typesyst == 'psyspcur' or typesyst == 'psysmoon':
        if typesyst == 'psys' or typesyst == 'psysmoon':
            # ~ Mars mass
            minmmasscomp = 0.1 # [Earth mass]
        if typesyst == 'psyspcur':
            # ~ Jupiter mass
            minmmasscomp = 300. # [Earth mass]
        # Deuterium burning mass
        maxmmasscomp = 4400. # [Earth mass]
    elif typesyst == 'sbin':
        minmmasscomp = 0.5 # [Earth mass]
        maxmmasscomp = 1000. # [Earth mass]
    else:
        print('typesyst')
        print(typesyst)
        raise Exception('')
    print('Sampling companion features...')
    
    numbsyst = len(dictpoplstar[namepoplstartotl]['radistar'])
    indxsyst = np.arange(numbsyst)

    # indices of companions for each star
    indxcompstar = [[] for k in indxsyst]
    cntr = 0
    for k in range(len(dictpoplstar[namepoplstartotl]['radistar'])):
        indxcompstar[k] = np.arange(cntr, cntr + dictpoplstar[namepoplstartotl]['numbcompstar'][k]).astype(int)
        cntr += dictpoplstar[namepoplstartotl]['numbcompstar'][k]
    dictcompnumbsamp[namepoplcomptotl] = cntr
    

    


    # prepare to load star features into component features
    for name in list(dictpoplstar[namepoplstartotl].keys()):
        dictpoplcomp[namepoplcomptotl][name] = np.empty(dictcompnumbsamp[namepoplcomptotl])
    
    # total mass
    dictpoplstar[namepoplstartotl]['massstarcomp'] = np.empty(dictpoplstar[namepoplstartotl]['radistar'].size)
    
    listnamecomp = ['pericomp', 'cosicomp', 'masscomp', 'smaxcomp', 'epocmtracomp', 'radistar', 'massstarcomp']
    if typesyst == 'psysmoon':
        listnamecomp += ['masscompmoon']
    if not boolsystcosc:
        listnamecomp += ['radicomp', 'denscomp']
    for name in listnamecomp:
        dictpoplcomp[namepoplcomptotl][name] = np.empty(dictcompnumbsamp[namepoplcomptotl])

    if booldiag:
        cntr = 0
        for k in range(len(indxcompstar)):
            cntr += indxcompstar[k].size
        if cntr != dictcompnumbsamp[namepoplcomptotl]:
            raise Exception('')
    
    dictpoplstar[namepoplstartotl]['massstarcomp'] = np.copy(dictpoplstar[namepoplstartotl]['massstar'])

    numbstar = dictpoplstar[namepoplstartotl]['radistar'].size
    for k in tqdm(range(numbstar)):
        
        if dictpoplstar[namepoplstartotl]['numbcompstar'][k] == 0:
            continue

        # cosine of orbital inclinations
        dictpoplcomp[namepoplcomptotl]['cosicomp'][indxcompstar[k]] = np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
        
        # load star features into component features
        for name in dictpoplstar[namepoplstartotl].keys():
            dictpoplcomp[namepoplcomptotl][name][indxcompstar[k]] = dictpoplstar[namepoplstartotl][name][k]
        
        # companion mass
        dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]] = tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), \
                                                                                                                                            minmmasscomp, maxmmasscomp, 2.)
        
        if typesyst == 'psys' or typesyst == 'psyspcur' or typesyst == 'psysmoon' or typesyst == 'sbin':
            # companion radius
            dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k]] = retr_radifrommass(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
    
            # companion density
            dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k]] = 5.51 * dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]] / \
                                                                                                            dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k]]**3
        
        # total mass
        if boolsystcosc or typesyst == 'sbin':
            dictpoplstar[namepoplstartotl]['massstarcomp'][k] += np.sum(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
        if typesyst == 'psys' or typesyst == 'psyspcur' or typesyst == 'psysmoon':
            dictpoplstar[namepoplstartotl]['massstarcomp'] = dictpoplstar[namepoplstartotl]['massstar']
        
        if typesamporbtcomp == 'peri':
        
            dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]] = tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), \
                                                                                                                                                minmpericomp, maxmpericomp, 2.)
            dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]] = retr_smaxkepl(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]], \
                                                                                                                        dictpoplstar[namepoplstartotl]['massstarcomp'][k])
        else:
            # semi-major axes
            #if np.isfinite(dictpoplstar[namepoplstartotl]['densstar'][k]):
            #    densstar = dictpoplstar[namepoplstartotl]['densstar'][k]
            #else:
            #    densstar = 1.
            #dictpoplcomp[namepoplcomptotl]['radiroch'][k] = retr_radiroch(radistar, densstar, denscomp)
            #minmsmax = 2. * dictpoplcomp[namepoplcomptotl]['radiroch'][k]
            dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]] = dictpoplstar[namepoplstartotl]['radistar'][k] * \
                                                                                            tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), \
                                                                                            minmsmaxradistar, maxmsmaxradistar, 2.) / dictfact['aurs']
            
            dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]] = retr_perikepl(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]], \
                                                                                                                        dictpoplstar[namepoplstartotl]['massstarcomp'][k])
    
        
        if booldiag:
            if not np.isfinite(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]]).all():
                
                print('dictpoplcomp[namepoplcomptotl][masscomp][indxcompstar[k]]')
                print(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
                print('dictpoplcomp[namepoplcomptotl][massstarcomp][indxcompstar[k]]')
                print(dictpoplcomp[namepoplcomptotl]['massstarcomp'][indxcompstar[k]])
                print('dictpoplcomp[namepoplcomptotl][smaxcomp][indxcompstar[k]]')
                print(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]])
                print('dictpoplcomp[namepoplcomptotl][pericomp][indxcompstar[k]]')
                print(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]])
                raise Exception('')

        # epochs
        if epocmtracomp is not None:
            dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = np.full(dictpoplstar[namepoplstartotl]['numbcompstar'][k], epocmtracomp)
        else:
            dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = 1e8 * np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
        if timeepoc is not None:
            dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = dictpoplcomp[namepoplcomptotl]['epocmtracomp'][k] + dictpoplcomp[namepoplcomptotl]['pericomp'][k] * \
                                                             np.round((dictpoplcomp[namepoplcomptotl]['epocmtracomp'][k] - timeepoc) / dictpoplcomp[namepoplcomptotl]['pericomp'][k])
    
    if typesyst == 'psysmoon':
        # initialize the total mass of the companion + moons system as the mass of the companion
        dictpoplcomp[namepoplcomptotl]['masscompmoon'] = np.copy(dictpoplcomp[namepoplcomptotl]['masscomp'])
                
    dictpoplcomp[namepoplcomptotl]['rsum'] = dictpoplcomp[namepoplcomptotl]['radistar']
    if not boolsystcosc:
        dictpoplcomp[namepoplcomptotl]['rsum'] += dictpoplcomp[namepoplcomptotl]['radicomp'] / dictfact['rsre']    
    dictpoplcomp[namepoplcomptotl]['rsmacomp'] = dictpoplcomp[namepoplcomptotl]['rsum'] / dictpoplcomp[namepoplcomptotl]['smaxcomp'] / dictfact['aurs']
    
    # orbital inclinations of the companions
    dictpoplcomp[namepoplcomptotl]['inclcomp'] = 180. / np.pi * np.arccos(dictpoplcomp[namepoplcomptotl]['cosicomp'])
    
    # Boolean flag indicating whether a companion is transiting
    dictpoplcomp[namepoplcomptotl]['booltran'] = dictpoplcomp[namepoplcomptotl]['rsmacomp'] > dictpoplcomp[namepoplcomptotl]['cosicomp']

    # subpopulation where object transits
    indx = np.where(dictpoplcomp[namepoplcomptotl]['booltran'])[0]
    retr_subp(dictpoplcomp, dictcompnumbsamp, dictcompindxsamp, namepoplcomptotl, namepoplcomptran, indx)

    # transit duration
    dictpoplcomp[namepoplcomptran]['duratrantotl'] = retr_duratrantotl(dictpoplcomp[namepoplcomptran]['pericomp'], \
                                                                   dictpoplcomp[namepoplcomptran]['rsmacomp'], \
                                                                   dictpoplcomp[namepoplcomptran]['cosicomp'])
    dictpoplcomp[namepoplcomptran]['dcyc'] = dictpoplcomp[namepoplcomptran]['duratrantotl'] / dictpoplcomp[namepoplcomptran]['pericomp'] / 24.
    if typesyst == 'psys':
        dictpoplcomp[namepoplcomptran]['rrat'] = dictpoplcomp[namepoplcomptran]['radicomp'] / dictpoplcomp[namepoplcomptran]['radistar'] / dictfact['rsre']
        dictpoplcomp[namepoplcomptran]['dept'] = 1e3 * dictpoplcomp[namepoplcomptran]['rrat']**2 # [ppt]
    if boolsystcosc:
        dictpoplcomp[namepoplcomptran]['amplslen'] = retr_amplslen(dictpoplcomp[namepoplcomptran]['pericomp'], dictpoplcomp[namepoplcomptran]['radistar'], \
                                                                            dictpoplcomp[namepoplcomptran]['masscomp'], dictpoplcomp[namepoplcomptran]['massstar'])
    
    dictcompnumbsamp[namepoplcomptotl] = dictpoplcomp[namepoplcomptotl]['radistar'].size
    
    indxmooncompstar = [[[] for j in indxcompstar[k]] for k in indxsyst]
    if typesyst == 'psysmoon':
        # Hill radius of the companion
        dictpoplcomp[namepoplcomptotl]['radihill'] = retr_radihill(dictpoplcomp[namepoplcomptotl]['smaxcomp'], dictpoplcomp[namepoplcomptotl]['masscomp'] / dictfact['msme'], \
                                                                                                                            dictpoplcomp[namepoplcomptotl]['massstar'])
    
        # maximum semi-major axis of the moons 
        dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'] = 0.5 * dictpoplcomp[namepoplcomptotl]['radihill']
            
        # mean number of moons per companion
        dictpoplcomp[namepoplcomptotl]['numbmooncompmean'] = 0.5 * dictpoplcomp[namepoplcomptotl]['masscomp']**(-1.)
        
        # number of moons per companion
        dictpoplcomp[namepoplcomptotl]['numbmooncomp'] = np.random.poisson(dictpoplcomp[namepoplcomptotl]['numbmooncompmean'])
        
        cntr = 0
        for k in range(len(dictpoplstar[namepoplstartotl]['radistar'])):
            for j in range(len(indxcompstar[k])):
                indxmooncompstar[k][j] = np.arange(cntr, cntr + dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k][j]]).astype(int)
                cntr += dictpoplcomp[namepoplcomptotl]['numbmooncomp'][j]
        dictmoonnumbsamp[namepoplmoontotl] = cntr
    
        numbmoontotl = np.sum(dictpoplcomp[namepoplcomptotl]['numbmooncomp'])
        
        # prepare to load component features into moon features
        for name in list(dictpoplcomp[namepoplcomptotl].keys()):
            dictpoplmoon[namepoplmoontotl][name] = np.empty(dictmoonnumbsamp[namepoplmoontotl])
    
        for k in tqdm(range(numbstar)):
            
            if dictpoplstar[namepoplstartotl]['numbcompstar'][k] == 0:
                continue
            
            numbcomp = dictpoplstar[namepoplstartotl]['numbcompstar'][k]
            
            # number of exomoons to the companion
            numbmoon = dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k]]
            for name in ['radi', 'mass', 'dens', 'peri', 'epocmtra', 'smax', 'minmsmax']:
                dictpoplmoon[namepoplmoontotl][name+'moon'] = np.empty(numbmoontotl)
            
            indxmoon = [[] for j in indxcompstar[k]]
            for j in range(indxcompstar[k].size):
                
                #print('')
                #print('')
                #print('')
                #print('')
                #print('j')
                #print(j)
                #print('indxcompstar[k]')
                #print(indxcompstar[k])
                #print('numbmoon')
                #print(numbmoon)
                #print('indxmoon')
                #print(indxmoon)
                #print('indxmooncompstar[k][j]')
                #print(indxmooncompstar[k][j])
                #print('')
                
                if numbmoon[j] == 0:
                    continue

                indxmoon[j] = np.arange(numbmoon[j])
                # properties of the moons
                ## radii [R_E]
                dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]] = dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]] * \
                                                                tdpy.icdf_powr(np.random.rand(dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k][j]]), 0.05, 0.3, 2.)
                
                ## mass [M_E]
                dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] = retr_massfromradi(dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]])
                
                ## densities [g/cm^3]
                dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]] = dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] / \
                                                                                                                dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]]**3
                
                # minimum semi-major axes for the moons
                dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]] = retr_radiroch(dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]], \
                                                                                                       dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k][j]], \
                                                                                                       dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                
                # semi-major axes of the moons
                for jj in indxmoon[j]:
                    dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j][jj]] = tdpy.icdf_powr(np.random.rand(), \
                                                                                                            dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j][jj]], \
                                                                                                            dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]], 2.)
                
                # add the moon masses to the total mass of the companion + moons system
                dictpoplcomp[namepoplcomptotl]['masscompmoon'][indxcompstar[k][j]] += np.sum(dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                
                # orbital period of the moons
                dictpoplmoon[namepoplmoontotl]['perimoon'][indxmooncompstar[k][j]] = retr_perikepl(dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]], \
                                                                                                dictpoplcomp[namepoplcomptotl]['masscompmoon'][indxcompstar[k][j]] / dictfact['msme'])
                
                # load component features into moon features
                for name in dictpoplcomp[namepoplcomptotl].keys():
                    dictpoplmoon[namepoplmoontotl][name][indxmooncompstar[k][j]] = dictpoplcomp[namepoplcomptotl][name][indxcompstar[k][j]]
                
            # mid-transit times of the moons
            dictpoplmoon[namepoplmoontotl]['epocmtramoon'] = 1e8 * np.random.rand(numbmoontotl)
            
            if (dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]] > dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k][j]] / 1.2).any():
                
                print('dictpoplcomp[namepoplcomptotl][smaxcomp][indxcompstar[k][j]]')
                print(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k][j]])
                print('dictpoplmoon[namepoplmoontotl][smaxmoon][indxmooncompstar[k][j]]')
                print(dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]])
                
                print('dictpoplcomp[namepoplcomptotl][maxmsmaxmoon][indxcompstar[k][j]]')
                print(dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]])
                print('dictpoplcomp[namepoplcomptotl][radihill][indxcompstar[k][j]]')
                print(dictpoplcomp[namepoplcomptotl]['radihill'][indxcompstar[k][j]])
                
                print('dictpoplcomp[namepoplcomptotl][radicomp][indxcompstar[k][j]]')
                print(dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]])
                print('dictpoplcomp[namepoplcomptotl][denscomp][indxcompstar[k][j]]')
                print(dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k][j]])
                print('dictpoplmoon[namepoplmoontotl][densmoon][indxmooncompstar[k][j]]')
                print(dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                
                print('dictpoplmoon[namepoplmoontotl][minmsmaxmoon][indxmooncompstar[k][j]]')
                print(dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]])
                raise Exception('')
    
    return dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumbsamp, dictcompindxsamp, indxcompstar, indxmooncompstar
       

def retr_listtablobsv(strgmast):
    
    if strgmast is None:
        raise Exception('strgmast should not be None.')

    listtablobsv = astroquery.mast.Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', objectname=strgmast)

    if len(listtablobsv) == 0:
        print('No SPOC data is found...')
    
    return listtablobsv


def setp_coorstrgmast(rasctarg=None, decltarg=None, strgmast=None):
    
    if strgmast is not None and (rasctarg is not None or decltarg is not None) or strgmast is None and (rasctarg is None or decltarg is None):
        raise Exception('')

    # determine RA and DEC if not already provided
    if rasctarg is None:
        rasctarg, decltarg = retr_rascdeclfromstrgmast(strgmast)
    
    # determine strgmast if not already provided
    if strgmast is None:
        strgmast = '%g %g' % (rasctarg, decltarg)

    return strgmast, rasctarg, decltarg


def retr_rascdeclfromstrgmast(strgmast):

    print('Querying the TIC using the key %s, in order to get the RA and DEC of the closest TIC source...' % strgmast)
    listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC')
    #listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC', radius='40s')
    rasctarg = listdictcatl[0]['ra']
    decltarg = listdictcatl[0]['dec']
    print('TIC, RA, and DEC of the closest match are %d, %.5g, and %.5g' % (int(listdictcatl[0]['ID']), rasctarg, decltarg))
    
    return rasctarg, decltarg

    
def retr_listtsec(strgmast):
    '''
    Retrieve the list of sectors, cameras, and CCDs for which TESS data are available for the target.
    '''
    
    print('Calling TESSCut with keyword %s to get the list of sectors for which TESS data are available...' % strgmast)
    tabltesscutt = astroquery.mast.Tesscut.get_sectors(coordinates=strgmast, radius=0)

    listtsec = np.array(tabltesscutt['sector'])
    listtcam = np.array(tabltesscutt['camera'])
    listtccd = np.array(tabltesscutt['ccd'])
   
    return listtsec, listtcam, listtccd


def retr_booltpxf(listtsec, listtsecspoc):

    ## number of sectors for which TESS data are available
    numbtsec = len(listtsec)
    ## Boolean flags to indicate that TPFs exist at 2-min
    booltpxf = np.zeros(numbtsec, dtype=bool)
    for k, tsec in enumerate(listtsec):
        if tsec in listtsecspoc:
            booltpxf[k] = True
    
    return booltpxf


def exec_lspe(arrylcur, pathimag=None, pathdata=None, strgextn='', factnyqt=None, \
              minmfreq=0.1, \
              maxmfreq=None, \
              factosam=3., \
              ## file type of the plot
              typefileplot='pdf', \
              # verbosity level
              typeverb=0, \
             ):
    '''
    Calculate the LS periodogram of a time-series.
    '''
    
    if maxmfreq is not None and factnyqt is not None:
        raise Exception('')
    
    dictlspeoutp = dict()
    
    if pathimag is not None:
        pathplot = pathimag + 'lspe_%s.%s' % (strgextn, typefileplot)

    if pathdata is not None:
        pathcsvv = pathdata + 'spec_lspe_%s.csv' % strgextn
    
    if pathdata is None or not os.path.exists(pathcsvv) or pathimag is not None and not os.path.exists(pathplot):
        print('Calculating LS periodogram...')
        
        # factor by which the maximum frequency is compared to the Nyquist frequency
        if factnyqt is None:
            factnyqt = 1.
        
        time = arrylcur[:, 0]
        lcur = arrylcur[:, 1]
        numbtime = time.size
        minmtime = np.amin(time)
        maxmtime = np.amax(time)
        delttime = maxmtime - minmtime
        freqnyqt = numbtime / delttime / 2.
        
        if minmfreq is None:
            minmfreq = 1. / delttime
        
        if maxmfreq is None:
            maxmfreq = factnyqt * freqnyqt
        
        # determine the frequency sampling resolution with N samples per line
        deltfreq = minmfreq / factosam / 2.
        freq = np.arange(minmfreq, maxmfreq, deltfreq)
        peri = 1. / freq
        
        objtlspe = astropy.timeseries.LombScargle(time, lcur, nterms=1)

        powr = objtlspe.power(freq)
        
        if pathdata is not None:
            arry = np.empty((peri.size, 2))
            arry[:, 0] = peri
            arry[:, 1] = powr
            print('Writing to %s...' % pathcsvv)
            np.savetxt(pathcsvv, arry, delimiter=',')
    
    else:
        if typeverb > 0:
            print('Reading from %s...' % pathcsvv)
        arry = np.loadtxt(pathcsvv, delimiter=',')
        peri = arry[:, 0]
        powr = arry[:, 1]
    
    #listindxperipeak, _ = scipy.signal.find_peaks(powr)
    #indxperimpow = listindxperipeak[0]
    indxperimpow = np.argmax(powr)
    
    perimpow = peri[indxperimpow]
    powrmpow = powr[indxperimpow]

    if pathimag is not None:
        if not os.path.exists(pathplot):
            figr, axis = plt.subplots(figsize=(8., 4.5))
            axis.plot(peri, powr, color='k')
            
            axis.axvline(perimpow, alpha=0.4, lw=3)
            minmxaxi = np.amin(peri)
            maxmxaxi = np.amax(peri)
            for n in range(2, 10):
                xpos = n * perimpow
                if xpos > maxmxaxi:
                    break
                axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
            for n in range(2, 10):
                xpos = perimpow / n
                if xpos < minmxaxi:
                    break
                axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
            
            strgtitl = 'Maximum power of %.3g at %.3f days' % (powrmpow, perimpow)
            
            listprob = [0.05]
            powrfals = objtlspe.false_alarm_level(listprob)
            for p in range(len(listprob)):
                axis.axhline(powrfals[p], ls='--')

            axis.set_xscale('log')
            axis.set_xlabel('Period [days]')
            axis.set_ylabel('Normalized Power')
            axis.set_title(strgtitl)
            print('Writing to %s...' % pathplot)
            plt.savefig(pathplot)
            plt.close()
        dictlspeoutp['pathplot'] = pathplot

    dictlspeoutp['perimpow'] = perimpow
    dictlspeoutp['powrmpow'] = powrmpow
    
    return dictlspeoutp


def plot_lcur( \
              
              pathimag, \
              
              strgextn, \
              
              # dictionary holding the model time-series
              dictmodl=None, \
              
              # the times at twhich the data time-series have been collected
              timedata=None, \
              
              # data time-series
              lcurdata=None, \
              
              timedatabind=None, \
              
              lcurdatabind=None, \
              
              lcurdatastdvbind=None, \
              
              # Boolean flag to break the line of the model when separation is very large
              boolbrekmodl=True, \
              
              # Boolean flag to ignore any existing files and overwrite
              boolwritover=False, \
              
              # label for the vertical axis, including the unit
              lablyaxi=None, \
              
              # size of the figure
              sizefigr=None, \
              
              timeoffs=0., \
              
              limtxaxi=None, \
              
              limtyaxi=None, \
              
              # title for the plot
              strgtitl='', \
              
              # Boolean flag to diagnose
              booldiag=True, \
                      
              ## file type of the plot
              typefileplot='pdf', \
             ):
    '''
    Plot a list of data and model time-series
    '''
    
    if strgextn == '':
        raise Exception('')
    
    path = pathimag + 'lcur_%s.%s' % (strgextn, typefileplot)
    
    # skip plotting
    if not boolwritover and os.path.exists(path):
        return path
    
    boollegd = False
    
    if sizefigr is None:
        sizefigr = [8., 2.5]

    figr, axis = plt.subplots(figsize=sizefigr)
    
    # raw data
    if timedata is not None:
        axis.plot(timedata - timeoffs, lcurdata, color='grey', ls='', marker='o', ms=1, rasterized=True)
    
    # binned data
    if timedatabind is not None:
        axis.errorbar(timedatabind, lcurdatabind, yerr=lcurdatastdvbind, color='k', ls='', marker='o', ms=2)
    
    # model
    if dictmodl is not None:
        
        k = 0
        for attr in dictmodl:
            if 'lsty' in dictmodl[attr]:
                ls = dictmodl[attr]['lsty']
            else:
                ls = None
            
            if 'colr' in dictmodl[attr]:
                color = dictmodl[attr]['colr']
            else:
                color = None
                
            if 'alph' in dictmodl[attr]:
                alpha = dictmodl[attr]['alph']
            else:
                alpha = None
                
            if boolbrekmodl:
                diftimemodl = dictmodl[attr]['time'][1:] - dictmodl[attr]['time'][:-1]
                
                indxtimebrekregi = np.where(diftimemodl > 2 * np.amin(diftimemodl))[0] + 1
                indxtimebrekregi = np.concatenate([np.array([0]), indxtimebrekregi, np.array([dictmodl[attr]['time'].size - 1])])
                numbtimebrekregi = indxtimebrekregi.size
                numbtimechun = numbtimebrekregi - 1

                xdat = []
                ydat = []
                for n in range(numbtimechun):
                    xdat.append(dictmodl[attr]['time'][indxtimebrekregi[n]:indxtimebrekregi[n+1]])
                    ydat.append(dictmodl[attr]['lcur'][indxtimebrekregi[n]:indxtimebrekregi[n+1]])
                    
            else:
                xdat = [dictmodl[attr]['time']]
                ydat = [dictmodl[attr]['lcur']]
            numbchun = len(xdat)
            
            for n in range(numbchun):
                if n == 0 and 'labl' in dictmodl[attr]:
                    label = dictmodl[attr]['labl']
                    boollegd = True
                else:
                    label = None
                
                axis.plot(xdat[n] - timeoffs, ydat[n], color=color, lw=1, label=label, ls=ls, alpha=alpha)
            k += 1

    if timeoffs == 0:
        axis.set_xlabel('Time [days]')
    else:
        axis.set_xlabel('Time [BJD-%d]' % timeoffs)
    
    if limtxaxi is not None:
        if not np.isfinite(limtxaxi).all():
            print('limtxaxi')
            print(limtxaxi)
            raise Exception('')

        axis.set_xlim(limtxaxi)

    if limtyaxi is not None:
        axis.set_ylim(limtyaxi)

    if lablyaxi is None:
        lablyaxi = 'Relative flux'
    
    axis.set_ylabel(lablyaxi)
    axis.set_title(strgtitl)
    
    if boollegd:
        axis.legend()

    plt.subplots_adjust(bottom=0.2)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    return path


def plot_pcur(pathimag, arrylcur=None, arrypcur=None, arrypcurbind=None, phascent=0., boolhour=False, epoc=None, peri=None, strgextn='', \
              ## file type of the plot
              typefileplot='pdf', \
                                                            boolbind=True, timespan=None, booltime=False, numbbins=100, limtxdat=None):
    
    if arrypcur is None:
        arrypcur = fold_tser(arrylcur, epoc, peri)
    if arrypcurbind is None and boolbind:
        arrypcurbind = rebn_tser(arrypcur, numbbins)
        
    # phase on the horizontal axis
    figr, axis = plt.subplots(1, 1, figsize=(8, 4))
    
    # time on the horizontal axis
    if booltime:
        lablxaxi = 'Time [hours]'
        if boolhour:
            fact = 24.
        else:
            fact = 1.
        xdat = arrypcur[:, 0] * peri * fact
        if boolbind:
            xdatbind = arrypcurbind[:, 0] * peri * fact
    else:
        lablxaxi = 'Phase'
        xdat = arrypcur[:, 0]
        if boolbind:
            xdatbind = arrypcurbind[:, 0]
    axis.set_xlabel(lablxaxi)
    
    axis.plot(xdat, arrypcur[:, 1], color='grey', alpha=0.2, marker='o', ls='', ms=0.5, rasterized=True)
    if boolbind:
        axis.plot(xdatbind, arrypcurbind[:, 1], color='k', marker='o', ls='', ms=2)
    
    axis.set_ylabel('Relative Flux')
    
    # adjust the x-axis limits
    if limtxdat is not None:
        axis.set_xlim(limtxdat)
    
    plt.subplots_adjust(hspace=0., bottom=0.25, left=0.25)
    path = pathimag + 'pcur%s.%s' % (strgextn, typefileplot)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
            

def fold_tser(arry, epoc, peri, boolxdattime=False, boolsort=True, phasshft=0.5, booldiag=True):
    
    phas = (((arry[:, 0] - epoc) % peri) / peri + phasshft) % 1. - phasshft
    
    arryfold = np.empty_like(arry)
    arryfold[:, 0, ...] = phas
    arryfold[:, 1:3, ...] = arry[:, 1:3, ...]
    
    if boolsort:
        indx = np.argsort(phas)
        arryfold = arryfold[indx, :, ...]
    
    if boolxdattime:
        arryfold[:, 0, ...] *= peri

    return arryfold


def rebn_tser(arry, numbbins=None, delt=None, binsxdat=None):
    
    if not (numbbins is None and delt is None and binsxdat is not None or \
            numbbins is not None and delt is None and binsxdat is None or \
            numbbins is None and delt is not None and binsxdat is None):
        raise Exception('')
    
    if arry.shape[0] == 0:
        print('Warning! Trying to bin an empty time-series...')
        return arry
    
    xdat = arry[:, 0]
    if numbbins is not None:
        arryrebn = np.empty((numbbins, 3)) + np.nan
        binsxdat = np.linspace(np.amin(xdat), np.amax(xdat), numbbins + 1)
    if delt is not None:
        binsxdat = np.arange(np.amin(xdat), np.amax(xdat) + delt, delt)
    if delt is not None or binsxdat is not None:
        numbbins = binsxdat.size - 1
        arryrebn = np.empty((numbbins, 3)) + np.nan

    meanxdat = (binsxdat[:-1] + binsxdat[1:]) / 2.
    arryrebn[:, 0] = meanxdat

    indxbins = np.arange(numbbins)
    for k in indxbins:
        indxxdat = np.where((xdat < binsxdat[k+1]) & (xdat > binsxdat[k]))[0]
        if indxxdat.size == 0:
            arryrebn[k, 1] = np.nan
            arryrebn[k, 2] = np.nan
        else:
            arryrebn[k, 1] = np.mean(arry[indxxdat, 1])
            stdvfrst  = np.sqrt(np.nansum(arry[indxxdat, 2]**2)) / indxxdat.size
            stdvseco = np.std(arry[indxxdat, 1])
            arryrebn[k, 2] = np.sqrt(stdvfrst**2 + stdvseco**2)
    
    return arryrebn

    
def read_tesskplr_fold(pathfold, pathwrit, boolmaskqual=True, typeinst='tess', strgtypelcur='PDCSAP_FLUX', boolnorm=None):
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error.
    '''

    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    listarry = []
    for path in listpath:
        arry = read_tesskplr_file(pathfold + path + '/' + path + '_lc.fits', typeinst=typeinst, strgtypelcur=strgtypelcur, boolmaskqual=boolmaskqual, boolnorm=boolnorm)
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


def read_tesskplr_file(path, typeinst='tess', strgtypelcur='PDCSAP_FLUX', boolmaskqual=True, boolmasknann=True, boolnorm=None):
    '''
    Read a TESS or Kepler light curve file and returns a data cube with time, flux and flux error.
    '''
    
    if boolnorm is None:
        boolnorm = True

    print('Reading from %s...' % path)
    listhdun = fits.open(path)
    
    tsec = listhdun[0].header['SECTOR']
    tcam = listhdun[0].header['CAMERA']
    tccd = listhdun[0].header['CCD']
    
    # indices of times where the quality flag is not raised (i.e., good quality)
    indxtimequalgood = np.where((listhdun[1].data['QUALITY'] == 0) & np.isfinite(listhdun[1].data['TIME']))[0]
    time = listhdun[1].data['TIME']

    # Boolean flag indicating whether the target file is a light curve or target pixel file
    boollcur = 'lc.fits' in path

    if boollcur:
        strgtype = strgtypelcur

        time = listhdun[1].data['TIME'] + 2457000
        if typeinst == 'TESS':
            time += 2457000
        if typeinst == 'kplr':
            time += 2454833
    
        flux = listhdun[1].data[strgtype]
        stdv = listhdun[1].data[strgtype+'_ERR']
        #print(listhdun[1].data.names)
        
        if boolmaskqual:
            # filtering for good quality
            time = time[indxtimequalgood]
            flux = flux[indxtimequalgood, ...]
            if boollcur:
                stdv = stdv[indxtimequalgood]
    
        numbtime = time.size
        arry = np.empty((numbtime, 3))
        arry[:, 0] = time
        arry[:, 1] = flux
        arry[:, 2] = stdv
        
        #indxtimenanngood = np.where(~np.any(np.isnan(arry), axis=1))[0]
        #if boolmasknann:
        #    arry = arry[indxtimenanngood, :]
    
        #print('HACKING, SKIPPING NORMALIZATION FOR SPOC DATA')
        # normalize
        if boolnorm:
            factnorm = np.median(arry[:, 1])
            arry[:, 1] /= np.median(arry[:, 1])
            arry[:, 2] /= np.median(arry[:, 1])
        
        return arry, tsec, tcam, tccd
    else:
        return listhdun, indxtimequalgood, tsec, tcam, tccd


#def retr_fracrtsa(fracrprs, fracsars):
#    
#    fracrtsa = (fracrprs + 1.) / fracsars
#    
#    return fracrtsa
#
#
#def retr_fracsars(fracrprs, fracrtsa):
#    
#    fracsars = (fracrprs + 1.) / fracrtsa
#    
#    return fracsars


def retr_rflxmodlrise(time, timerise, coeflinerise, coefquadrise, coefline):
    
    timeoffs = time - timerise
    
    dflxline = coefline * timeoffs
    
    indxpost = np.where(timeoffs > 0)[0]
    dflxrise = np.zeros_like(time)
    dflxrise[indxpost] = coeflinerise * timeoffs[indxpost] + coefquadrise * timeoffs[indxpost]**2
    
    rflx = 1. + dflxrise + dflxline
    
    return rflx, dflxline, dflxrise


def retr_boolnocccomp(gdat, j, t, boolinve=False):
    '''
    Return a grid of Booleans indicating which prid points are not occulted
    '''
    
    if gdat.typecoor == 'plan':
        xposgridcomp = gdat.xposgridplan[j]
        yposgridcomp = gdat.yposgridplan[j]
    
    if gdat.typecoor == 'star':
        xposgridcomp = gdat.xposgridstar - gdat.xposcomp[j][t]
        yposgridcomp = gdat.yposgridstar - gdat.yposcomp[j]
    
    if gdat.typesyst == 'psys' or gdat.typesyst == 'psyspcur' or \
                    gdat.typesyst == 'plandiskedgehori' or gdat.typesyst == 'plandiskedgevert' or gdat.typesyst == 'plandiskface' or gdat.typesyst == 'sbin':
        distcomp = np.sqrt(xposgridcomp**2 + yposgridcomp**2)
        
        if boolinve:
            boolnocccomp = distcomp < gdat.rratcomp[j]
        else:
            boolnocccomp = distcomp > gdat.rratcomp[j]
    else:
        print('gdat.typesyst')
        print(gdat.typesyst)
        raise Exception('')

    if gdat.typesyst == 'plandiskedgehori':
        if boolinve:
            booldisk = (xposgridcomp / 1.75 / gdat.rratcomp[j])**2 + (yposgridcomp / 0.2 / gdat.rratcomp[j])**2 < 1.
        else:
            booldisk = (xposgridcomp / 1.75 / gdat.rratcomp[j])**2 + (yposgridcomp / 0.2 / gdat.rratcomp[j])**2 > 1.
        boolnocccomp = boolnocccomp & booldisk
       
    if gdat.typesyst == 'plandiskedgevert':
        if boolinve:
            booldisk = (yposgridcomp / 1.75 / gdat.rratcomp[j])**2 + (xposgridcomp / 0.2 / gdat.rratcomp[j])**2 < 1.
        else:
            booldisk = (yposgridcomp / 1.75 / gdat.rratcomp[j])**2 + (xposgridcomp / 0.2 / gdat.rratcomp[j])**2 > 1.
        boolnocccomp = boolnocccomp & booldisk
       
    if gdat.typesyst == 'plandiskface':
        if boolinve:
            boolnocccomp = boolnocccomp & ((distcomp > 1.5 * gdat.rratcomp[j]) & (distcomp < 1.75 * gdat.rratcomp[j]))
        else:
            boolnocccomp = boolnocccomp & ((distcomp > 1.5 * gdat.rratcomp[j]) & (distcomp < 1.75 * gdat.rratcomp[j]))
    
    return boolnocccomp


def retr_fluxstartrantotl(gdat, t, boolrofi):
    '''
    Calculate the total flux of a star on the grid
    '''
    fluxstartran = retr_fluxstartran(gdat, t, boolrofi)
    
    fluxstartrantotl = np.sum(fluxstartran)

    return fluxstartrantotl


def retr_fluxstartran(gdat, t, boolrofi=None):
    '''
    Calculate the relative flux from a brightness map
    '''
    if boolrofi is not None:
        indxgridrofi = np.where(boolrofi)
    
    if gdat.typecoor == 'plan':
        if boolrofi is not None:
            dist = gdat.diststargridplan[indxgridrofi]
        else:
            dist = gdat.diststargridplan
        cosg = np.sqrt(1. - dist**2)
        fluxstartran = retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)
    if gdat.typecoor == 'star':
        if boolrofi is not None:
            fluxstartran = gdat.brgtgridstar[indxgridrofi]
        else:
            fluxstartran = gdat.brgtgridstar
            
    return fluxstartran


def anim_fluxtotl(gdat, t, j=None):
    
    #print('gdat.pathfoldanim')
    #print(gdat.pathfoldanim)
    #print('os.path.exists(gdat.pathgiff)')
    #print(os.path.exists(gdat.pathgiff))
    #print(os.path.exists(gdat.pathgiff))
    #print('')
    if gdat.pathfoldanim is not None and not os.path.exists(gdat.pathgiff):
        
        gdat.pathtime = gdat.pathfoldanim + 'imag%s%s_%04d.%s' % (gdat.strgextn, gdat.strgcompmoon, t, gdat.typefileplot)
        
        gdat.cmndmakeanim += ' %s' % gdat.pathtime
        gdat.cmnddeleimag += ' %s' % gdat.pathtime
        
        figr, axis = plt.subplots(figsize=(4, 3))
        
        if gdat.typecoor == 'plan':
            
            if gdat.typesyst != 'cosc':
                # Booleans indicating whether planet grid points are NOT occulted
                gdat.boolgridplannocc = retr_boolnocccomp(gdat, j, t)
                                
                # Booleans indicating whether planet grid points are bright
                gdat.boolgridplanbrgt = gdat.boolgridplannocc & gdat.boolgridplanstar

                # indices of the planet grid points that are bright
                gdat.indxgridplanbrgt = np.where(gdat.boolgridplanbrgt)
            
                # brightness of the planet grid points
                brgttemp = np.zeros_like(gdat.xposgridplan[j])

                cosg = np.sqrt(1. - gdat.diststargridplan[gdat.indxgridplanbrgt]**2)
                brgttemp[gdat.indxgridplanbrgt] = retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)
            
            if gdat.typesyst == 'cosc':
                brtglens = retr_brgtlens(gdat, t)
                print('brtglens')
                summgene(brtglens)
                brgttemp = brtglens

        if gdat.typecoor == 'star':
            brgttemp = np.zeros_like(gdat.brgtgridstar)
            brgttemp[gdat.boolgridstarbrgt] = gdat.brgtgridstar[gdat.boolgridstarbrgt]
        
        imag = axis.imshow(brgttemp, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmbrgtstar)
        
        axis.axis('off')
        
        titl = '$R_S = %.3g R_{\odot}$' % gdat.radistar
        axis.set_title(titl)
        
        if gdat.titlvisu is not None:
            axis.set_title(gdat.titlvisu)

        print('Writing to %s...' % gdat.pathtime)
        plt.savefig(gdat.pathtime, dpi=200)
        plt.close()
    

def retr_brgtlens(gdat, t):
    
    gdat.xposgridstarplan = gdat.xposgridstar - gdat.xposstar[0][t]
    gdat.yposgridstarplan = gdat.yposgridstar - gdat.yposstar[0]
    
    gdat.xposgridstarnorm = gdat.xposgridstarplan / gdat.radieins[0]
    gdat.yposgridstarnorm = gdat.yposgridstarplan / gdat.radieins[0]
    
    gdat.distgridstarnorm = np.sqrt(gdat.xposgridstarnorm**2 + gdat.yposgridstarnorm**2)
    
    xposfrst = 0.5 * (gdat.xposgridstarplan + np.sqrt(gdat.xposgridstarplan**2 + 4. * gdat.radieins[0]**2))
    yposfrst = 0.5 * (gdat.yposgridstarplan + np.sqrt(gdat.yposgridstarplan**2 + 4. * gdat.radieins[0]**2))

    xposseco = 0.5 * (gdat.xposgridstarplan - np.sqrt(gdat.xposgridstarplan**2 + 4. * gdat.radieins[0]**2))
    yposseco = 0.5 * (gdat.yposgridstarplan - np.sqrt(gdat.yposgridstarplan**2 + 4. * gdat.radieins[0]**2))

    magnfrst = (gdat.distgridstarnorm**2 + 2.) / (2. * gdat.distgridstarnorm * np.sqrt(gdat.distgridstarnorm**2 + 4.)) + 0.5
    magnseco = (gdat.distgridstarnorm**2 + 2.) / (2. * gdat.distgridstarnorm * np.sqrt(gdat.distgridstarnorm**2 + 4.)) - 0.5
    
    print('xposfrst.flatten()')
    summgene(xposfrst.flatten())
    print('yposfrst.flatten()')
    summgene(yposfrst.flatten())
    print('magnfrst.flatten()')
    summgene(magnfrst.flatten())
    print('gdat.binsxposgridplan[0]')
    summgene(gdat.binsxposgridplan[0])
    print('gdat.binsyposgridplan[0]')
    summgene(gdat.binsyposgridplan[0])
    flux, _, _, _ = scipy.stats.binned_statistic_2d(xposfrst.flatten(), yposfrst.flatten(), magnfrst.flatten(), statistic='sum', bins=[gdat.binsxposgridplan[0], gdat.binsyposgridplan[0]])

    #magn = np.abs(1. / (1. - (gdat.radieins[0] / gdat.distgridplan[0][np.where(boolgridplanstar)])**4.))
    
    return flux


def retr_rflxtranmodl( \
                      # times at which to evaluate the relative flux
                      time, \
                      # companions
                      ## orbital periods of the companions
                      pericomp, \
                      ## mid-transit epochs of the companions
                      epocmtracomp, \
                      
                      # minimum radius ratio tolerated
                      tolerrat=None, \

                      # a factor to multiply the number of time samples collected for interpolation
                      facttimereso=None, \

                      # the spatial resolution of the grid over which the planet's brightness and occultation are evaluated
                      resoplan=None, \

                      ## orbital inclination
                      inclcomp=None, \
                      ## cosine of the orbital inclination
                      cosicomp=None, \
                      
                      ## radius ratio for the companions
                      rratcomp=None, \
                      ## eccentricity of the orbit
                      eccecomp=0., \
                      ## sine of 
                      sinwcomp=0., \
                      
                      ## radii of the companions
                      radicomp=None, \
                      ## mass of the companions
                      masscomp=None, \
                      
                      ## type of the system
                      typesyst='psys', \
                      
                      # type of limb-darkening
                      typelmdk='quad', \
                      # limb-darkening coefficient(s)
                      coeflmdk=None, \
                      # radius of the host star in Solar radius
                      radistar=None, \
                      # mass of the host star in Solar mass
                      massstar=None, \

                      ## sum of stellar and companion radius
                      rsmacomp=None, \
                      
                      # Boolean flag to evaluate by interpolating a single transit
                      boolintp=None, \

                      # type of coordinate system
                      ## 'star': centered on the star
                      ## 'plan': centered on the planet
                      typecoor='plan', \
                     
                      # type of the brightness of the companion
                      typeplanbrgt='dark', \

                      # moons
                      ## radii
                      radimoon=None, \
                      ## orbital periods
                      perimoon=None, \
                      ## mid-transit epochs
                      epocmtramoon=None, \
                      ## sum of planetary and moon radius
                      rsmamoon=None, \
                      ## cosine of the orbital inclination
                      cosimoon=None, \
                      ## eccentricity of the orbit
                      eccemoon=None, \
                      ## sine of 
                      sinwmoon=None, \
                      
                      # spots
                      ## rotation period at the equator
                      perispot=None, \

                      # path to animate the integration in
                      pathfoldanim=None, \

                      # title for the visuals
                      titlvisu=None, \

                      # string for the animation
                      strgextn='', \

                      # Boolean flag to ignore any existing plot and overwrite
                      boolwritover=False, \
                      
                      # Boolean flag to diagnose
                      booldiag=True, \
                      
                      ## file type of the plot
                      typefileplot='pdf', \

                      # verbosity level
                      typeverb=1, \
                     ):
    '''
    Calculate the relative flux light curve of a star due to list of transiting companions and their orbiting moons.
    When limb-darkening and/or moons are turned on, the result is interpolated based on star-to-companion radius, companion-to-moon radius.
    '''
    timeinit = timemodu.time()
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    dictinpt = dict(locals())
    for attr, valu in dictinpt.items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    if typeverb > 0:
        print('Estimating the light curve via retr_rflxtranmodl()...')
        
    if isinstance(pericomp, list):
        pericomp = np.array(pericomp)

    if isinstance(epocmtracomp, list):
        epocmtracomp = np.array(epocmtracomp)

    if isinstance(radicomp, list):
        radicomp = np.array(radicomp)

    if isinstance(rsmacomp, list):
        rsmacomp = np.array(rsmacomp)

    if isinstance(gdat.rratcomp, list):
        gdat.rratcomp = np.array(gdat.rratcomp)

    if isinstance(masscomp, list):
        masscomp = np.array(masscomp)

    if isinstance(inclcomp, list):
        inclcomp = np.array(inclcomp)

    if isinstance(cosicomp, list):
        cosicomp = np.array(cosicomp)

    if isinstance(eccecomp, list):
        eccecomp = np.array(eccecomp)

    if isinstance(sinwcomp, list):
        sinwcomp = np.array(sinwcomp)
    
    # determine the input type
    if rsmacomp is not None:
        typeinpt = 'rsmacomp'
    else:
        typeinpt = 'masstotlradistar'
    
    if typeverb > 1:
        print('typeinpt')
        print(typeinpt)

    if gdat.tolerrat is None:
        gdat.tolerrat = 3e-3
    
    if gdat.facttimereso is None:
        gdat.facttimereso = 1.
   
    if gdat.resoplan is None:
        gdat.resoplan = 0.05

    # check inputs
    if booldiag:
        
        if gdat.typesyst != 'cosc' and not ((gdat.radistar is not None and radicomp is not None) or gdat.rratcomp is not None):
            print('gdat.radistar')
            print(gdat.radistar)
            print('radicomp')
            print(radicomp)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            raise Exception('')
        
        if not ((masscomp is not None and massstar is not None) or rsmacomp is not None):
            print('typesyst')
            print(typesyst)
            print('masscomp')
            print(masscomp)
            print('massstar')
            print(massstar)
            print('rsmacomp')
            print(rsmacomp)
            raise Exception('')

        # inclination
        if inclcomp is not None and cosicomp is not None:
            raise Exception('')
        
        # eccentricity

        # periods, radii, and masses
        if typeinpt == 'masstotlradistar':
            if gdat.radistar is None or not np.isfinite(gdat.radistar):
                print('gdat.radistar')
                print(gdat.radistar)
                raise Exception('')
        if typeinpt == 'rsmacomp':
            if not np.isfinite(rsmacomp) or rsmacomp is None:
                print('rsmacomp')
                print(rsmacomp)
                raise Exception('')
        if gdat.typesyst == 'cosc':
            if massstar is None:
                raise Exception('')
        
            if masscomp is None:
                raise Exception('')
    
    # Boolean flag to return separate light curves for the companion and moon
    boolcompmoon = radimoon is not None

    if boolcompmoon and boolintp:
        raise Exception('')

    if gdat.coeflmdk is None:
        # limb-darkening coefficient 
        if gdat.typelmdk == 'linr':
            # linear
            gdat.coeflmdk = 0.6
        if gdat.typelmdk == 'quad':
            # quadratic
            gdat.coeflmdk=[0.4, 0.25]
        if gdat.typelmdk == 'nlin':
            # nonlinear
            gdat.coeflmdk=[0.2, 0.2, 0.2]
    
    if inclcomp is not None:
        cosicomp = np.cos(inclcomp * np.pi / 180.)
    
    if gdat.typesyst == 'cosc':
        if radicomp is not None:
            if typeverb > 0:
                print('Radius provided for the compact object...')
        else:
            radicomp = np.array([0.])

    numbcomp = pericomp.size
    if perimoon is not None:
        indxcomp = np.arange(numbcomp)
        numbmoon = np.empty(numbcomp, dtype=int)
        for j in indxcomp:
            numbmoon[j] = len(perimoon[j])
        indxmoon = [np.arange(numbmoon[j]) for j in indxcomp]
    
    # output dictionary
    dictoutp = dict()

    if typeverb > 0:
        if typeverb > 1:
            print('pericomp')
            print(pericomp)
            print('epocmtracomp')
            print(epocmtracomp)
    
    if booldiag:
        if time.ndim != 1:
            raise Exception('')
    
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    numbtime = time.size
    indxtime = np.arange(numbtime)

    dictfact = retr_factconv()
    
    if gdat.pathfoldanim is not None and gdat.strgextn != '':
        gdat.strgextn = '_' + gdat.strgextn
    
    if gdat.rratcomp is not None:
        radicomp = gdat.rratcomp
    else:
        gdat.rratcomp = radicomp / gdat.radistar / dictfact['rsre']
        if typeverb > 1:
            print('gdat.radistar')
            print(gdat.radistar)
    
    if typeinpt == 'rsmacomp':
        
        gdat.radistar = 1.
        radicomp = gdat.rratcomp * dictfact['rsre']
        if typeverb > 0:
            print('gdat.radistar [RS]')
            print(gdat.radistar)
            print('rsmacomp')
            print(rsmacomp)
            print('radicomp [RE]')
            print(radicomp)
        smaxcomp = 1. / (gdat.radistar + radicomp / dictfact['rsre']) / rsmacomp / dictfact['aurs']
    
    if typeinpt == 'masstotlradistar':
        
        if gdat.typesyst == 'psys' or gdat.typesyst == 'psyspcur':
            masscompsolr = masscomp / dictfact['msme']
        elif gdat.typesyst == 'cosc' or gdat.typesyst == 'sbin':
            masscompsolr = masscomp
        else:
            print('gdat.typesyst')
            print(gdat.typesyst)
            raise Exception('')

        ## total mass of the system
        masstotl = massstar + np.sum(masscompsolr)
        
        ## semi-major axis
        smaxcomp = retr_smaxkepl(pericomp, masstotl)
        
        print('Estimating rsmacomp...')
        rsmacomp = (gdat.radistar + radicomp / dictfact['rsre']) / smaxcomp / dictfact['aurs']
        
        if perimoon is not None:
            smaxmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            for j in indxcomp:
                smaxmoon[j] = retr_smaxkepl(perimoon[j], masscompsolr[j])
                
                for jj in indxmoon[j]:
                    if smaxmoon[j] <= radicomp[j] / dictfact['rsre'] / dictfact['aurs']:
                        print('smaxmoon[j] [AU]')
                        print(smaxmoon[j])
                        print('masscomp[j] [MES]')
                        print(masscomp[j])
                        print('masscompsolr[j]')
                        print(masscompsolr[j])
                        print('perimoon[j]')
                        print(perimoon[j])
                        print('radicomp[j]')
                        print(radicomp[j])
                        raise Exception('')
    
    numbcomp = pericomp.size
    indxcomp = np.arange(numbcomp)
    
    if eccecomp is None:
        eccecomp = np.zeros(numbcomp)
    if sinwcomp is None:
        sinwcomp = np.zeros(numbcomp)
    
    if typeverb > 1:
        print('time')
        summgene(time)
        print('radicomp [RE]')
        print(radicomp)
        print('cosicomp')
        print(cosicomp)
        print('rsmacomp')
        print(rsmacomp)
        print('smaxcomp [AU]')
        print(smaxcomp)
        
        print('indxcomp')
        print(indxcomp)
        if perimoon is not None:
            print('perimoon [days]')
            print(perimoon)
            print('radimoon [RE]')
            print(radimoon)
            print('smaxmoon [AU]')
            for smaxmoontemp in smaxmoon:
                print(smaxmoontemp)
    
    if boolintp is None:
        if perimoon is not None or numbcomp > 1 or gdat.perispot is not None:
            boolintp = False
        else:
            boolintp = True

    if typeverb > 0:
        print('typesyst')
        print(typesyst)
        print('boolintp')
        print(boolintp)
        print('typecoor')
        print(typecoor)

    
    duratrantotl = retr_duratrantotl(pericomp, rsmacomp, cosicomp) / 24.
    
    dictoutp['duratrantotl'] = duratrantotl
        
    if gdat.typesyst == 'cosc':
        amplslen = retr_amplslen(pericomp, radistar, masscomp, massstar)
        dictoutp['amplslen'] = amplslen

    if boolintp:
        phascomp = [[] for j in indxcomp]
        timecomp = [[] for j in indxcomp]
    
        if typeverb > 0:
            print('duratrantotl [days]')
            print(duratrantotl)

        if gdat.typesyst != 'cosc':
            if booldiag and gdat.rratcomp is None:
                print('gdat.typesyst')
                print(gdat.typesyst)
                raise Exception('')
            if booldiag:
                if gdat.typesyst == 'sbin' and (gdat.rratcomp == 0).any():
                    raise Exception('')

            duratranfull = retr_duratranfull(pericomp, rsmacomp, cosicomp, gdat.rratcomp) / 24.
        
        numbtimecomp = [[] for j in indxcomp]
        indxtimecomp = [[] for j in indxcomp]
        numbtimecompingr = [[] for j in indxcomp]
        numbtimecompfull = [[] for j in indxcomp]
        if gdat.typesyst == 'cosc':
            numbtimecomp[0] = int(gdat.facttimereso * 75)
            numbtimecompfull[0] = int(gdat.facttimereso * 75)
        else:
            for j in indxcomp:
                numbtimecompingr[j] = int(gdat.facttimereso * 25)
                if np.isfinite(duratranfull[j]):
                    numbtimecompfull[j] = int(numbtimecompingr[j] * (duratrantotl[j] - duratranfull[j]) / duratrantotl[j])
                else:
                    numbtimecompfull[j] = 0
                numbtimecomp[j] = numbtimecompfull[j] + 2 * numbtimecompingr[j]
        
        for j in indxcomp:
            indxtimecomp[j] = np.arange(numbtimecomp[j])
        
        maxphastrantotl = duratrantotl / pericomp / 2.
    
    phas = [[] for j in indxcomp]
    for j in indxcomp:
        
        if typeverb > 1:
            print('j')
            print(j)
        
        phas[j] = ((time - epocmtracomp[j]) / pericomp[j] + 0.5) % 1. - 0.5
        
        if booldiag:
            if np.isscalar(phas[j]):
                raise Exception('')
        
        if typeverb > 1:
            print('phas[j]')
            summgene(phas[j])

        if boolintp:
            if not np.isfinite(duratrantotl[j]) or duratrantotl[j] == 0.:
                continue

            if gdat.typesyst == 'cosc':
                timecompinen = -0.5 * duratrantotl[j]
                timecompegst = 0.5 * duratrantotl[j]
            else:
                timecompinst = -duratrantotl[j] / 2.
                timecompegen = duratrantotl[j] / 2.
                if np.isfinite(duratranfull[j]):
                    timecompinen = timecompinst + 0.5 * (duratrantotl[j] - duratranfull[j])
                    timecompegst = timecompegen - 0.5 * (duratrantotl[j] - duratranfull[j])
                else:
                    timecompinen = 0.
                    timecompegst = 0.
            listtimetemp = []
            if gdat.typesyst == 'cosc':
                listtimetemp.append(np.linspace(timecompinen, timecompegst, numbtimecompfull[j]))
            else:
                listtimetemp += [np.linspace(timecompinst, timecompinen, numbtimecompingr[j])]
                if numbtimecompfull[j] > 0:
                    listtimetemp.append(np.linspace(timecompinen, timecompegst, numbtimecompfull[j] + 1)[1:])
                listtimetemp.append(np.linspace(timecompegst, timecompegen, numbtimecompingr[j] + 1)[1:])
            timecomp[j] = np.concatenate(listtimetemp)
            
            phascomp[j] = (timecomp[j] / pericomp[j] + 0.5) % 1. - 0.5
            
            if booldiag:
                if (timecomp[j][1:] - timecomp[j][:-1] <= 0).any() or not np.isfinite(timecomp[j]).all():
                    print('')
                    print('')
                    print('')
                    print('gdat.typesyst')
                    print(gdat.typesyst)
                    print('pericomp')
                    print(pericomp)
                    print('rsmacomp')
                    print(rsmacomp)
                    print('cosicomp')
                    print(cosicomp)
                    print('duratrantotl[j]')
                    print(duratrantotl[j])
                    print('duratranfull[j]')
                    print(duratranfull[j])
                    print('numbtimecompfull[j]')
                    print(numbtimecompfull[j])
                    print('timecompinst')
                    print(timecompinst)
                    print('timecompinen')
                    print(timecompinen)
                    print('timecompegst')
                    print(timecompegst)
                    print('timecompegen')
                    print(timecompegen)
                    print('np.isfinite(timecomp[j]).all()')
                    print(np.isfinite(timecomp[j]).all())
                    print('listtimetemp')
                    print(listtimetemp)
                    print('timecomp[j]')
                    print(timecomp[j])
                    raise Exception('')
                if not np.isfinite(phascomp[j]).all():
                    print('timecomp[j]')
                    summgene(timecomp[j])
                    print('duratrantotl')
                    print(duratrantotl)
                    raise Exception('')
        
                if np.isscalar(phascomp[j]):
                    raise Exception('')
        
    # grids
    if gdat.typesyst != 'cosc' and not (gdat.rratcomp > gdat.tolerrat).all():
        print('WARNING! At least one of the occulter radii is smaller than the grid resolution. The output will be unreliable.')
    
    if gdat.typesyst == 'cosc':
        diffgrid = 0.01
        gdat.wdthslen = np.array([0.1])
        gdat.radieins = 0.04273 * np.sqrt(masscomp * smaxcomp) / radistar

        if gdat.booldiag:
            if numbcomp > 1:
                raise Exception('')

    elif not (gdat.rratcomp > gdat.tolerrat).any():
        diffgrid = 0.01
    else:
        diffgrid = gdat.resoplan * np.amin(gdat.rratcomp[gdat.rratcomp > gdat.tolerrat])
    ## planet
    if gdat.typecoor == 'plan':
        gdat.xposgridplan = [[] for j in indxcomp]
        gdat.yposgridplan = [[] for j in indxcomp]
        if gdat.typesyst == 'cosc':
            gdat.distgridplan = [[] for j in indxcomp]
            gdat.binsxposgridplan = [[] for j in indxcomp]
            gdat.binsyposgridplan = [[] for j in indxcomp]
        
        for j in indxcomp:
            if gdat.typesyst == 'cosc':
                arryplan = np.arange(-gdat.wdthslen[j] - 2. * diffgrid, gdat.wdthslen[j] + 3. * diffgrid, diffgrid)
            else:
                arryplan = np.arange(-gdat.rratcomp[j] - 2. * diffgrid, gdat.rratcomp[j] + 3. * diffgrid, diffgrid)
            gdat.xposgridplan[j], gdat.yposgridplan[j] = np.meshgrid(arryplan, arryplan)
    
            if gdat.typesyst == 'cosc':
                # distance to the points in the planet grid
                gdat.distgridplan[j] = np.sqrt(gdat.xposgridplan[j]**2 + gdat.yposgridplan[j]**2)
    
                if gdat.typesyst == 'cosc':
                    arryplan = np.arange(-gdat.wdthslen[j] - 2.5 * diffgrid, gdat.wdthslen[j] + 3.5 * diffgrid, diffgrid)
                else:
                    arryplan = np.arange(-gdat.rratcomp[j] - 2.5 * diffgrid, gdat.rratcomp[j] + 3.5 * diffgrid, diffgrid)
                gdat.binsxposgridplan[j] = arryplan
                gdat.binsyposgridplan[j] = arryplan#np.meshgrid(arryplan, arryplan)
    
    ## star
    arrystar = np.arange(-1. - 2. * diffgrid, 1. + 3. * diffgrid, diffgrid)
    gdat.xposgridstar, gdat.yposgridstar = np.meshgrid(arrystar, arrystar)
    
    # distance to the points in the star grid
    gdat.distgridstar = np.sqrt(gdat.xposgridstar**2 + gdat.yposgridstar**2)
    
    # Booleans indicating whether star grid points are within the star
    gdat.boolgridstarstar = gdat.distgridstar < 1.
    
    # indices of the star grid points within the star
    gdat.indxgridstarstar = np.where(gdat.boolgridstarstar)
    
    # stellar brightness in the star grid
    gdat.brgtgridstar = np.zeros_like(gdat.xposgridstar)
    cosg = np.sqrt(1. - gdat.distgridstar[gdat.indxgridstarstar]**2)
    gdat.brgtgridstar[gdat.indxgridstarstar] = retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)
    
    # maximum stellar brightness
    if gdat.pathfoldanim is not None:
        gdat.maxmbrgtstar = retr_brgtlmdk(1., gdat.coeflmdk, typelmdk=gdat.typelmdk)

    # total (unocculted) stellar birghtness
    gdat.brgtstarnocc = np.sum(gdat.brgtgridstar)
    
    if boolintp:
        fluxtotlcomp = [np.full(numbtimecomp[j], gdat.brgtstarnocc) for j in indxcomp]
    
    if gdat.typecoor == 'star':
        gdat.xposcomp = [[] for j in indxcomp]
        gdat.yposcomp = [[] for j in indxcomp]
        if perimoon is not None:
            xposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            yposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
        
    if gdat.typecoor == 'plan':
        gdat.xposstar = [[] for j in indxcomp]
        gdat.yposstar = [[] for j in indxcomp]
        
    for j in indxcomp:
        
        if boolintp and not np.isfinite(duratrantotl[j]) or duratrantotl[j] == 0.:
            continue
        
        if boolintp:
            phastemp = phascomp[j]
        else:
            phastemp = phas[j]
        
        if booldiag:
            if phastemp.size == 1:
                print('boolintp')
                print(boolintp)
                raise Exception('')

        xpos = smaxcomp[j] * np.sin(2. * np.pi * phastemp) / gdat.radistar * dictfact['aurs']
        ypos = cosicomp[j] * smaxcomp[j] / gdat.radistar * dictfact['aurs']
        
        if gdat.typecoor == 'star':
            gdat.xposcomp[j] = xpos
            gdat.yposcomp[j] = ypos
            
            if perimoon is not None:
                for jj in indxmoon[j]:
                    xposmoon[j][jj] = gdat.xposcomp[j] + smaxmoon[j][jj] * np.cos(2. * np.pi * (time - epocmtramoon[j][jj]) / perimoon[j][jj]) / gdat.radistar * dictfact['aurs']
                    yposmoon[j][jj] = gdat.yposcomp[j] + smaxmoon[j][jj] * np.sin(2. * np.pi * (time - epocmtramoon[j][jj]) / perimoon[j][jj]) / gdat.radistar * dictfact['aurs']
                
        if gdat.typecoor == 'plan':
            gdat.xposstar[j] = -xpos
            gdat.yposstar[j] = -ypos
    
    if boolcompmoon:
        numbitermoon = 2
    else:
        numbitermoon = 1
    
    fluxtotl = np.full(numbtime, gdat.brgtstarnocc)
    
    for a in range(numbitermoon):
        
        #print('a')
        #print(a)

        if a == 0:
            gdat.strgcompmoon = ''
        else:
            gdat.strgcompmoon = '_onlycomp'
        
        if gdat.pathfoldanim is not None:
            gdat.pathgiff = gdat.pathfoldanim + 'anim%s%s.gif' % (gdat.strgextn, gdat.strgcompmoon)
            # make animation
            gdat.cmndmakeanim = 'convert -delay 5 -density 200'
            gdat.cmnddeleimag = 'rm'
        
        if boolintp:
        
            for j in indxcomp:
                if not np.isfinite(duratrantotl[j]) or duratrantotl[j] == 0.:
                    continue
                 
                if gdat.typesyst != 'cosc' and gdat.rratcomp[j] < gdat.tolerrat:
                    continue
                    
                for t in indxtimecomp[j]:

                    if gdat.typecoor == 'plan':
                        
                        if gdat.typesyst != 'cosc' and (np.sqrt(gdat.xposstar[j][t]**2 + gdat.yposstar[j]**2) < 1. + gdat.rratcomp[j]):
                            booleval = True
                        elif gdat.typesyst == 'cosc' and (np.sqrt(gdat.xposstar[j][t]**2 + gdat.yposstar[j]**2) < 1. + gdat.wdthslen[j]):
                            booleval = True
                        else:
                            booleval = False

                    if gdat.typecoor == 'star':
                        if gdat.typesyst != 'cosc' and (np.sqrt(gdat.xposcomp[j][t]**2 + gdat.yposcomp[j]**2) < 1. + gdat.rratcomp[j]):
                            booleval = True
                        elif gdat.typesyst == 'cosc' and (np.sqrt(gdat.xposcomp[j][t]**2 + gdat.yposcomp[j]**2) < 1. + gdat.wdthslen[j]):
                            booleval = True
                        else:
                            booleval = False
    
                    if booleval:
                        
                        if gdat.typecoor == 'plan':
                            
                            if gdat.typesyst != 'cosc':
                                # Booleans indicating whether planet grid points are occulted
                                gdat.boolgridoccu = retr_boolnocccomp(gdat, j, t, boolinve=True)
                            
                            # distance from the points in the planet grid to the star
                            gdat.diststargridplan = np.sqrt((gdat.xposgridplan[j] - gdat.xposstar[j][t])**2 + (gdat.yposgridplan[j] - gdat.yposstar[j])**2)
                        
                            # Booleans indicating whether planet grid points are within the star
                            gdat.boolgridplanstar = gdat.diststargridplan < 1.
                            
                            if gdat.typesyst != 'cosc':
                                # Booleans indicating whether planet grid points are within the star and occulted
                                booloccu = gdat.boolgridplanstar & gdat.boolgridoccu
                            
                                # stellar flux occulted
                                deltflux = -retr_fluxstartrantotl(gdat, t, booloccu)
                            
                                fluxtotlcomp[j][t] += deltflux
                        
                            else:
                                # stellar flux in the planet grid
                                fluxstartrantotl = retr_fluxstartrantotl(gdat, t, gdat.boolgridplanstar)
                                
                                if booldiag and not np.isfinite(fluxstartrantotl).all():
                                    print('')
                                    print('')
                                    print('')
                                    print('fluxstartran')
                                    summgene(fluxstartran)
                                    raise Exception('')
                                
                                brtglens = retr_brgtlens(gdat, t)
                                
                                np.set_printoptions(precision=1, linewidth=180)

                                print('brtglens')
                                print(brtglens)
                                print('')
                                print('')
                                print('')
                                fluxtotlcomp[j][t] += np.sum(brtglens) - fluxstartrantotl

                        if gdat.typecoor == 'star':
                            # Booleans indicating whether planet grid points are NOT occulted
                            boolnocccomp = retr_boolnocccomp(gdat, j, t)
                            
                            gdat.boolgridstarbrgt = gdat.boolgridstarstar & boolnocccomp
        
                            fluxtotlcomp[j][t] = retr_fluxstartrantotl(gdat, t, gdat.boolgridstarbrgt)
                            
                        anim_fluxtotl(gdat, t, j=j)
    
                        if booldiag:
                            if abs(fluxtotlcomp[j][t]) > 1e20:
                                print('jt')
                                print(j, t)
                                print('fluxtotlcomp[j]')
                                summgene(fluxtotlcomp[j])
                                raise Exception('')

        else:
            
            for t in tqdm(range(numbtime)):
                cntrtemp = 0
                booleval = False
                
                for j in indxcomp:
                
                    if gdat.rratcomp[j] < gdat.tolerrat:
                        continue

                    if abs(phas[j][t]) > 0.25:
                        continue
            
                    if gdat.typesyst != 'cosc' and (np.sqrt(gdat.xposcomp[j][t]**2 + gdat.yposcomp[j]**2) < 1. + gdat.rratcomp[j]):
                        booleval = True
                        cntrtemp += 1

                    if cntrtemp == 1:
                        gdat.boolgridstarbrgt = np.copy(gdat.boolgridstarstar)
                                    
                    if booleval:
                        boolnocccomp = retr_boolnocccomp(gdat, j, t)
                        gdat.boolgridstarbrgt = gdat.boolgridstarbrgt & boolnocccomp
    
                    
                    if perimoon is not None and a == 0:

                        for jj in indxmoon[j]:
                            
                            if np.sqrt(xposmoon[j][jj][t]**2 + yposmoon[j][jj][t]**2) < 1. + rratmoon[j][jj]:
                                
                                booleval = True
                                cntrtemp += 1

                                if cntrtemp == 1:
                                    gdat.boolgridstarbrgt = np.copy(gdat.boolgridstarstar)
                                    
                                xposgridmoon = gdat.xposgridstar - xposmoon[j][jj][t]
                                yposgridmoon = gdat.yposgridstar - yposmoon[j][jj][t]
                                
                                distmoon = np.sqrt(xposgridmoon**2 + yposgridmoon**2)
                                boolnoccmoon = distmoon > rratmoon[j][jj]
                                
                                gdat.boolgridstarbrgt = gdat.boolgridstarbrgt & boolnoccmoon
            
                if booleval:
                    fluxtotl[t] = retr_fluxstartrantotl(gdat, t, gdat.boolgridstarbrgt)
                    anim_fluxtotl(gdat, t)
                        
                    
        if gdat.pathfoldanim is not None and not os.path.exists(gdat.pathgiff):
            # make the animation
            gdat.cmndmakeanim += ' %s' % gdat.pathgiff
            print('Writing to %s...' % gdat.pathgiff)
            print(gdat.cmndmakeanim)
            os.system(gdat.cmndmakeanim)
            
            # delete images
            os.system(gdat.cmnddeleimag)

    if boolintp:
        for j in indxcomp:
            
            if not np.isfinite(duratrantotl[j]) or duratrantotl[j] == 0.:
                continue
            
            indxphasintr = np.where(abs(phas[j]) < maxphastrantotl[j])[0]
            
            #try:
            diff = gdat.brgtstarnocc - scipy.interpolate.interp1d(phascomp[j], fluxtotlcomp[j])(phas[j][indxphasintr])
            #except:
            #    print('')
            #    print('')
            #    print('')
            #    print('gdat.typesyst')
            #    print(gdat.typesyst)
            #    print('phas[j]')
            #    summgene(phas[j])
            #    print('phas[j][indxphasintr]')
            #    summgene(phas[j][indxphasintr])
            #    print('phascomp[j]')
            #    summgene(phascomp[j])
            #    print('fluxtotlcomp[j]')
            #    summgene(fluxtotlcomp[j])
            #    raise Exception('')

            fluxtotl[indxphasintr] -= diff
        
    rflxtranmodl = fluxtotl / np.amax(fluxtotl)
    print('rflxtranmodl')
    summgene(rflxtranmodl)
    dictoutp['rflx'] = rflxtranmodl
    
    if masscomp is not None and massstar is not None and radistar is not None:
        densstar = 1.4 * massstar / radistar**3
        deptbeam = 1e-3 * retr_deptbeam(pericomp, massstar, masscomp)
        deptelli = 1e-3 * retr_deptelli(pericomp, densstar, massstar, masscomp)
        dictoutp['rflxbeam'] = [[] for j in indxcomp]
        dictoutp['rflxelli'] = [[] for j in indxcomp]
        print('dictoutp[rflx]')
        summgene(dictoutp['rflx'])

        for j in indxcomp:
            
            dictoutp['rflxslen'] = np.copy(dictoutp['rflx'])
            
            dictoutp['rflxbeam'][j] = 1. + deptbeam * np.sin(phas[j])
            dictoutp['rflxelli'][j] = 1. + deptelli * np.sin(2. * phas[j])
        
            dictoutp['rflx'] += dictoutp['rflxelli'][j] - 2.
            dictoutp['rflx'] += dictoutp['rflxbeam'][j]

            print('dictoutp[rflx]')
            summgene(dictoutp['rflx'])


    if boolcompmoon:
        rflxtranmodlcomp /= np.amax(rflxtranmodlcomp)
        dictoutp['rflxcomp'] = rflxtranmodlcomp
        rflxtranmodlmoon = 1. + rflxtranmodl - rflxtranmodlcomp
        dictoutp['rflxmoon'] = rflxtranmodlmoon

    dictoutp['timetotl'] = timemodu.time() - timeinit
    dictoutp['timeredu'] = dictoutp['timetotl'] / numbtime

    if booldiag and dictoutp['timeredu'] > 1e-1:
        print('Took too long to execute...')
        #raise Exception('')
    
    if booldiag and not np.isfinite(dictoutp['rflx']).all():
        print('dictoutp[rflx] is not all finite.')
        print('dictoutp[rflx]')
        summgene(dictoutp['rflx'])
        print('boolintp')
        print(boolintp)
        print('gdat.brgtstarnocc')
        print(gdat.brgtstarnocc)
        print('gdat.typesyst')
        print(gdat.typesyst)
        print('boolcompmoon')
        print(boolcompmoon)
        #print('rflxtranmodlcomp')
        #summgene(rflxtranmodlcomp)
        print('fluxtotl')
        summgene(fluxtotl)
        print('pericomp')
        print(pericomp)
        print('masscomp')
        print(masscomp)
        print('massstar')
        print(massstar)
        if typeinpt == 'masstotlradistar':
            print('masstotl')
            print(masstotl)
        print('smaxcomp')
        print(smaxcomp)
        print('gdat.rratcomp')
        print(gdat.rratcomp)
        print('cosicomp')
        print(cosicomp)
        print('rsmacomp')
        print(rsmacomp)
        print('duratrantotl')
        print(duratrantotl)
        raise Exception('')
    
    if typeverb > 0:
        print('retr_rflxtranmodl ran in %.3g seconds and %g ms per 1000 time samples.' % (dictoutp['timetotl'], dictoutp['timeredu'] * 1e6))
        print('')

    return dictoutp


def retr_radifrommass( \
                      # list of planet masses in units of Earth mass
                      listmassplan, \
                      # type of radius-mass model
                      strgtype='mine', \
                      ):
    '''
    Estimate planetary radii from samples of masses.
    '''
    
    if len(listmassplan) == 0:
        raise Exception('')

    if strgtype == 'mine':
        # interpolate masses
        listradi = np.empty_like(listmassplan)
        
        indx = np.where(listmassplan < 2.)[0]
        listradi[indx] = listmassplan[indx]**0.28
        
        indx = np.where((listmassplan > 2.) & (listmassplan < 130.))[0]
        listradi[indx] = 5. * (listmassplan[indx] / 20.)**(-0.59)
        
        indx = np.where((listmassplan > 130.) & (listmassplan < 2.66e4))[0]
        listradi[indx] = 10. * (listmassplan[indx] / 1e5)**(-0.04)
        
        indx = np.where(listmassplan > 2.66e4)[0]
        listradi[indx] = 20. * (listmassplan[indx] / 5e4)**0.88
    
    return listradi


def retr_massfromradi( \
                      # list of planet radius in units of Earth radius
                      listradicomp, \
                      # type of radius-mass model
                      strgtype='mine', \
                      # verbosity level
                      typeverb=1, \
                      ):
    '''
    Estimate planetary mass from samples of radii.
    '''
    
    if len(listradicomp) == 0:
        raise Exception('')


    if strgtype == 'mine':
        # get interpolation data
        path = os.environ['EPHESUS_DATA_PATH'] + '/data/massfromradi.csv'
        if os.path.exists(path):
            if typeverb > 0:
                print('Reading from %s...' % path)
            arry = np.loadtxt(path)
        else:
            # features of confirmed exoplanets
            dictpopl = dict()
            dictpopl['totl'] = retr_dictexar()
            ## planets with good measured radii and masses
            #indx = []
            #for n  in range(dictpopl['totl'][strgstrgrefrmasselem].size):
            #    if not ('Calculated Value' in dictpopl['totl'][strgstrgrefrmasselem][n] or \
            #            'Calculated Value' in dictpopl['totl']['strgrefrradicomp'][n]):
            #        indx.append(n)
            #indxmeas = np.array(indx)
            #indxgood = np.where(dictpopl['totl']['stdvmasscomp'] / dictpopl['totl']['stdvmasscomp'] > 5.)[0]
            #indx = np.setdiff1d(indxmeas, indxgood)
            #retr_subp(dictpopl, dictnumbsamp, dictindxsamp, 'totl', 'gmas', indxgood)
            
            minmradi = np.nanmin(dictpopl['totl']['radicomp'])
            maxmradi = np.nanmax(dictpopl['totl']['radicomp'])
            binsradi = np.linspace(minmradi, 24., 15)
            meanradi = (binsradi[1:] + binsradi[:-1]) / 2.
            arry = np.empty((meanradi.size, 5))
            arry[:, 0] = meanradi
            for l in range(meanradi.size):
                indx = np.where((dictpopl['totl']['radicomp'] > binsradi[l]) & (dictpopl['totl']['radicomp'] < binsradi[l+1]) & \
                                                                                            (dictpopl['totl']['masscomp'] / dictpopl['totl']['stdvmasscomp'] > 5.))[0]
                arry[l, 1] = np.nanmedian(dictpopl['totl']['masscomp'][indx])
                arry[l, 2] = np.nanstd(dictpopl['totl']['masscomp'][indx])
                arry[l, 3] = np.nanmedian(dictpopl['totl']['densplan'][indx])
                arry[l, 4] = np.nanstd(dictpopl['totl']['densplan'][indx])
            
            print('Writing to %s...' % path)
            np.savetxt(path, arry, fmt='%8.4g')

        # interpolate masses
        listmass = np.interp(listradicomp, arry[:, 0], arry[:, 1])
        liststdvmass = np.interp(listradicomp, arry[:, 0], arry[:, 2])
    
    if strgtype == 'wolf2016':
        # (Wolgang+2016 Table 1)
        listmass = (2.7 * (listradicomp * 11.2)**1.3 + np.random.randn(listradicomp.size) * 1.9) / 317.907
        listmass = np.maximum(listmass, np.zeros_like(listmass))
    
    return listmass


def retr_tmptplandayynigh(tmptirra, epsi):
    '''
    Estimate the dayside and nightside temperatures [K] of a planet given its irradiation temperature in K and recirculation efficiency.
    '''
    
    tmptdayy = tmptirra * (2. / 3. - 5. / 12. * epsi)**.25
    tmptnigh = tmptirra * (epsi / 4.)**.25
    
    return tmptdayy, tmptnigh


def retr_esmm(tmptplanequi, tmptstar, radicomp, radistar, kmag):
    
    tmptplanirra = tmptplanequi
    tmptplandayy, tmptplannigh = retr_tmptplandayynigh(tmptplanirra, 0.1)
    esmm = 1.1e3 * tdpy.util.retr_specbbod(tmptplandayy, 7.5) / tdpy.util.retr_specbbod(tmptstar, 7.5) * (radicomp / radistar)*2 * 10**(-kmag / 5.)

    return esmm


def retr_tsmm(radicomp, tmptplan, massplan, radistar, jmag):
    
    tsmm = 1.53 / 1.2 * radicomp**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)
    
    return tsmm


def retr_scalheig(tmptplan, massplan, radicomp):
    
    # tied to Jupier's scale height for H/He at 110 K   
    scalheig = 27. * (tmptplan / 160.) / (massplan / radicomp**2) / 71398. # [R_J]

    return scalheig


def retr_rflxfromdmag(dmag, stdvdmag=None):
    
    rflx = 10**(-dmag / 2.5)

    if stdvdmag is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
    
    return rflx, stdvrflx


def retr_dictexar( \
                  strgexar=None, \
                  # verbosity level
                  typeverb=1, \
                  strgelem='plan', \
                 ):
    
    strgradielem = 'radi' + strgelem
    strgstdvradi = 'stdv' + strgradielem
    strgmasselem = 'mass' + strgelem
    strgstdvmass = 'stdv' + strgmasselem
    
    strgstrgrefrradielem = 'strgrefrradi' + strgelem
    strgstrgrefrmasselem = 'strgrefrmass' + strgelem

    # get NASA Exoplanet Archive data
    path = os.environ['EPHESUS_DATA_PATH'] + '/data/PSCompPars_2022.05.17_13.05.01.csv'
    if typeverb > 0:
        print('Reading from %s...' % path)
    objtexar = pd.read_csv(path, skiprows=316)
    if strgexar is None:
        indx = np.arange(objtexar['hostname'].size)
        #indx = np.where(objtexar['default_flag'].values == 1)[0]
    else:
        indx = np.where(objtexar['hostname'] == strgexar)[0]
        #indx = np.where((objtexar['hostname'] == strgexar) & (objtexar['default_flag'].values == 1))[0]
    
    dictfact = retr_factconv()
    
    if indx.size == 0:
        print('The target name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgexar)
        return None
    else:
        dictexar = {}
        dictexar['namestar'] = objtexar['hostname'][indx].values
        dictexar['nameplan'] = objtexar['pl_name'][indx].values
        
        numbplanexar = len(dictexar['nameplan'])
        indxplanexar = np.arange(numbplanexar)

        listticitemp = objtexar['tic_id'][indx].values
        dictexar['tici'] = np.empty(numbplanexar, dtype=int)
        for k in indxplanexar:
            if isinstance(listticitemp[k], str):
                dictexar['tici'][k] = listticitemp[k][4:]
            else:
                dictexar['tici'][k] = 0
        
        dictexar['rascstar'] = objtexar['ra'][indx].values
        dictexar['declstar'] = objtexar['dec'][indx].values
        
        # err1 have positive values or zero
        # err2 have negative values or zero
        
        dictexar['toii'] = np.empty(numbplanexar, dtype=object)
        
        # discovery method
        dictexar['methdisc'] = objtexar['discoverymethod'][indx].values
        
        # discovery facility
        dictexar['facidisc'] = objtexar['disc_facility'][indx].values
        
        # discovery year
        dictexar['yeardisc'] = objtexar['disc_year'][indx].values
        
        dictexar['inso'] = objtexar['pl_insol'][indx].values
        dictexar['pericomp'] = objtexar['pl_orbper'][indx].values # [days]
        dictexar['smaxcomp'] = objtexar['pl_orbsmax'][indx].values # [AU]
        dictexar['epocmtracomp'] = objtexar['pl_tranmid'][indx].values # [BJD]
        dictexar['cosicomp'] = np.cos(objtexar['pl_orbincl'][indx].values / 180. * np.pi)
        dictexar['duratrantotl'] = objtexar['pl_trandur'][indx].values # [hour]
        dictexar['dept'] = 10. * objtexar['pl_trandep'][indx].values # ppt
        
        # to be deleted
        #dictexar['boolfpos'] = np.zeros(numbplanexar, dtype=bool)
        
        dictexar['booltran'] = objtexar['tran_flag'][indx].values
        
        # mass provenance
        dictexar['strgprovmass'] = objtexar['pl_bmassprov'][indx].values
        
        dictexar['booltran'] = dictexar['booltran'].astype(bool)

        # radius reference
        dictexar[strgstrgrefrradielem] = objtexar['pl_rade_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrradielem].size):
            if isinstance(dictexar[strgstrgrefrradielem][a], float) and not np.isfinite(dictexar[strgstrgrefrradielem][a]):
                dictexar[strgstrgrefrradielem][a] = ''
        
        # mass reference
        dictexar[strgstrgrefrmasselem] = objtexar['pl_bmasse_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrmasselem].size):
            if isinstance(dictexar[strgstrgrefrmasselem][a], float) and not np.isfinite(dictexar[strgstrgrefrmasselem][a]):
                dictexar[strgstrgrefrmasselem][a] = ''

        for strg in ['radistar', 'massstar', 'tmptstar', 'loggstar', strgradielem, strgmasselem, 'tmptplan', 'tagestar', \
                     'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'tmagsyst', 'metastar', 'distsyst', 'lumistar']:
            strgvarbexar = None
            if strg.endswith('syst'):
                strgvarbexar = 'sy_'
                if strg[:-4].endswith('mag'):
                    strgvarbexar += '%smag' % strg[0]
                if strg[:-4] == 'dist':
                    strgvarbexar += 'dist'
            if strg.endswith('star'):
                strgvarbexar = 'st_'
                if strg[:-4] == 'logg':
                    strgvarbexar += 'logg'
                if strg[:-4] == 'tage':
                    strgvarbexar += 'age'
                if strg[:-4] == 'meta':
                    strgvarbexar += 'met'
                if strg[:-4] == 'radi':
                    strgvarbexar += 'rad'
                if strg[:-4] == 'mass':
                    strgvarbexar += 'mass'
                if strg[:-4] == 'tmpt':
                    strgvarbexar += 'teff'
                if strg[:-4] == 'lumi':
                    strgvarbexar += 'lum'
            if strg.endswith('plan') or strg.endswith(strgelem):
                strgvarbexar = 'pl_'
                if strg[:-4].endswith('mag'):
                    strgvarbexar += '%smag' % strg[0]
                if strg[:-4] == 'tmpt':
                    strgvarbexar += 'eqt'
                if strg[:-4] == 'radi':
                    strgvarbexar += 'rade'
                if strg[:-4] == 'mass':
                    strgvarbexar += 'bmasse'
            if strgvarbexar is None:
                print('strg')
                print(strg)
                raise Exception('')
            dictexar[strg] = objtexar[strgvarbexar][indx].values
            dictexar['stdv%s' % strg] = (objtexar['%serr1' % strgvarbexar][indx].values - objtexar['%serr2' % strgvarbexar][indx].values) / 2.
       
        dictexar['vesc'] = retr_vesc(dictexar[strgmasselem], dictexar[strgradielem])
        dictexar['masstotl'] = dictexar['massstar'] + dictexar[strgmasselem] / dictfact['msme']
        
        dictexar['densplan'] = objtexar['pl_dens'][indx].values # [g/cm3]
        dictexar['vsiistar'] = objtexar['st_vsin'][indx].values # [km/s]
        dictexar['projoblq'] = objtexar['pl_projobliq'][indx].values # [deg]
        
        dictexar['numbplanstar'] = np.empty(numbplanexar)
        dictexar['numbplantranstar'] = np.empty(numbplanexar, dtype=int)
        boolfrst = np.zeros(numbplanexar, dtype=bool)
        #dictexar['booltrantotl'] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexar['namestar']):
            indxexarstar = np.where(namestar == dictexar['namestar'])[0]
            if k == indxexarstar[0]:
                boolfrst[k] = True
            dictexar['numbplanstar'][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexar['namestar']) & dictexar['booltran'])[0]
            dictexar['numbplantranstar'][k] = indxexarstartran.size
            #dictexar['booltrantotl'][k] = dictexar['booltran'][indxexarstar].all()
        
        objticrs = astropy.coordinates.SkyCoord(ra=dictexar['rascstar'], \
                                               dec=dictexar['declstar'], frame='icrs', unit='deg')
        
        # transit duty cycle
        dictexar['dcyc'] = dictexar['duratrantotl'] / dictexar['pericomp'] / 24.
        
        # galactic longitude
        dictexar['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dictexar['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dictexar['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dictexar['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        dictexar['rrat'] = dictexar[strgradielem] / dictexar['radistar'] / dictfact['rsre']
        

        # calculate TSM and ESM
        calc_tsmmesmm(dictexar, strgelem=strgelem)
        
    return dictexar


# physics

def retr_vesc(massplan, radicomp):
    
    vesc = 11.2 * np.sqrt(massplan / radicomp) # km/s

    return vesc


def retr_rs2a(rsmacomp, rrat):
    
    rs2a = rsmacomp / (1. + rrat)
    
    return rs2a


#def retr_rsmacomp(radicomp, radistar, smax):
#    
#    dictfact = retr_factconv()
#    rsmacomp = (radistar + radicomp / dictfact['rsre']) / (smax * dictfact['aurs'])
#    
#    return rsmacomp
#
#
def retr_rsmacomp(peri, dura, cosi):
    
    rsmacomp = np.sqrt(np.sin(dura * np.pi / peri / 24.)**2 + cosi**2)
    
    return rsmacomp


def retr_duratranfull(
                      # orbital period [days]
                      pericomp, \
                      # sum of the radii of th star and the companion divided by the semi-major axis
                      rsmacomp, \
                      # cosine of the orbital inclination
                      cosicomp, \
                      # radius ratio of the companion and the star
                      rratcomp, \
                     ):
    '''
    Return the full transit duration in hours.
    '''    
    
    # radius of the star minus the radius of the companion
    rdiacomp = rsmacomp * (1. - 2. / (1. + rratcomp))

    fact = rdiacomp**2 - cosicomp**2
    
    duratranfull = np.full_like(pericomp, np.nan)
    indxtran = np.where(fact > 0)
    
    # sine of inclination
    sinicomp = np.sqrt(1. - cosicomp[indxtran]**2)
    
    #duratranfull = 24. * peri / np.pi * np.arcsin(rs2a / sinicomp * np.sqrt((1. - rrat)**2 - imfa**2)) # [hours]
    duratranfull[indxtran] = 24. * pericomp[indxtran] / np.pi * np.arcsin(np.sqrt(fact[indxtran]) / sinicomp) # [hours]

    return duratranfull 


def retr_duratrantotl( \
                      # orbital period [days]
                      pericomp, \
                      # sum of the radii of th star and the companion divided by the semi-major axis
                      rsmacomp, \
                      # cosine of the orbital inclination
                      cosicomp, \
                     ):
    '''
    Return the total transit duration in hours.
    '''    
    
    fact = rsmacomp**2 - cosicomp**2
    
    indxtran = np.where(fact > 0)
    
    duratrantotl = np.full_like(pericomp, np.nan)
        
    # sine of inclination
    sinicomp = np.sqrt(1. - cosicomp**2)
    
    duratrantotl = 24. * pericomp / np.pi * np.arcsin(np.sqrt(fact) / sinicomp) # [hours]
    
    return duratrantotl


def retr_imfa(cosi, rs2a, ecce, sinw):
    
    imfa = cosi / rs2a * (1. - ecce)**2 / (1. + ecce * sinw)

    return imfa


def retr_deptbeam(peri, massstar, masscomp):
    '''
    Calculate the beaming amplitude.
    '''
    
    deptbeam = 2.8 * peri**(-1. / 3.) * (massstar + masscomp)**(-2. / 3.) * masscomp # [ppt]
    
    return deptbeam


def retr_deptelli(peri, densstar, massstar, masscomp):
    '''
    Calculate the ellipsoidal variation amplitude.
    '''
    
    deptelli = 18.9 * peri**(-2.) / densstar * (1. / (1. + massstar / masscomp)) # [ppt]
    
    return deptelli


def retr_masscomp(amplslen, peri):
    
    print('temp: this mass calculation is an approximation.')
    masscomp = 1e-3 * amplslen / 7.15e-5 / gdat.radistar**(-2.) / peri**(2. / 3.) / (gdat.massstar)**(1. / 3.)
    
    return masscomp


def retr_amplslen( \
                  # orbital period [days]
                  peri, \
                  # radistar: radius of the star [Solar radius]
                  radistar, \
                  # mass of the companion [Solar mass]
                  masscomp, \
                  # mass of the star [Solar mass]
                  massstar, \
                 ):
    '''
    Calculate the self-lensing amplitude.
    '''
    
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.) * 1e3 # [ppt]

    return amplslen


def retr_smaxkepl(peri, masstotl):
    '''
    Get the semi-major axis of a Keplerian orbit (in AU) from the orbital period (in days) and total mass (in Solar masses).

    Arguments
        peri: orbital period [days]
        masstotl: total mass of the system [Solar Masses]
    Returns
        smax: the semi-major axis of a Keplerian orbit [AU]
    '''
    
    smax = (7.496e-6 * masstotl * peri**2)**(1. / 3.) # [AU]
    
    return smax


def retr_perikepl(smax, masstotl):
    '''
    Get the period of a Keplerian orbit (in days) from the semi-major axis (in AU) and total mass (in Solar masses).

    Arguments
        smax: the semi-major axis of a Keplerian orbit [AU]
        masstotl: total mass of the system [Solar Masses]
    Returns
        peri: orbital period [days]
    '''
    
    peri = np.sqrt(smax**3 / 7.496e-6 / masstotl)
    
    return peri


def retr_radiroch(radistar, densstar, denscomp):
    '''
    Return the Roche limit.

    Arguments
        radistar: radius of the primary star
        densstar: density of the primary star
        denscomp: density of the companion
    '''    
    radiroch = radistar * (2. * densstar / denscomp)**(1. / 3.)
    
    return radiroch


def retr_radihill(smax, masscomp, massstar):
    '''
    Return the Hill radius of a companion.

    Arguments
        peri: orbital period
        rsmacomp: the sum of radii of the two bodies divided by the semi-major axis
        cosi: cosine of the inclination
    '''    
    radihill = smax * (masscomp / 3. / massstar)**(1. / 3.) # [AU]
    
    return radihill


# massplan in M_E
# massstar in M_S
def retr_rvelsema(peri, massplan, massstar, incl, ecce):
    
    dictfact = retr_factconv()
    
    rvelsema = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massplan + massstar * dictfact['msme'])**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]

    return rvelsema


# semi-amplitude of radial velocity of a two-body
# masstar in M_S
# massplan in M_J
# peri in days
# incl in degrees
def retr_rvel(time, epoc, peri, massplan, massstar, incl, ecce, arguperi):
    
    phas = (time - epoc) / peri
    phas = phas % 1.
    #consgrav = 2.35e-49
    #cons = 1.898e27
    #masstotl = massplan + massstar
    #smax = 
    #ampl = np.sqrt(consgrav / masstotl / smax / (1. - ecce**2))
    #rvel = cons * ampl * mass * np.sin(np.pi * incl / 180.) * (np.cos(np.pi * arguperi / 180. + 2. * np.pi * phas) + ecce * np.cos(np.pi * arguperi / 180.))

    ampl = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massstar + 9.548e-4 * massplan)**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]
    rvel = ampl * (np.cos(np.pi * arguperi / 180. + 2. * np.pi * phas) + ecce * np.cos(np.pi * arguperi / 180.))

    return rvel


def retr_factconv():
    
    dictfact = dict()

    dictfact['rsrj'] = 9.731
    dictfact['rjre'] = 11.21
    dictfact['rsre'] = dictfact['rsrj'] * dictfact['rjre']
    dictfact['msmj'] = 1048.
    dictfact['mjme'] = 317.8
    dictfact['msme'] = dictfact['msmj'] * dictfact['mjme']
    dictfact['aurs'] = 215.
    dictfact['pcau'] = 206265.

    return dictfact


def retr_alphelli(u, g):
    
    alphelli = 0.15 * (15 + u) * (1 + g) / (3 - u)
    
    return alphelli


def plot_anim():

    pathbase = os.environ['PEXO_DATA_PATH'] + '/imag/'
    radistar = 0.9
    
    booldark = True
    
    boolsingside = False
    boolanim = True
                  ## 'realblac': dark background, black planet
                  ## 'realblaclcur': dark backgound, black planet, light curve
                  ## 'realcolrlcur': dark backgound, colored planet, light curve
                  ## 'cartcolr': cartoon backgound
    listtypevisu = ['realblac', 'realblaclcur', 'realcolrlcur', 'cartcolr']
    listtypevisu = ['realblac', 'cartcolr']
    path = pathbase + 'orbt'
    
    for a in range(2):
    
        radicomp = [1.6, 2.1, 2.7, 3.1]
        rsmacomp = [0.0895, 0.0647, 0.0375, 0.03043]
        epoc = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
        peri = [3.8, 6.2, 14.2, 19.6]
        cosi = [0., 0., 0., 0.]
        
        if a == 1:
            radicomp += [2.0]
            rsmacomp += [0.88 / (215. * 0.1758)]
            epoc += [2458793.2786]
            peri += [29.54115]
            cosi += [0.]
        
        for typevisu in listtypevisu:
            
            if a == 0:
                continue
    
            pexo.main.plot_orbt( \
                                path, \
                                radicomp, \
                                rsmacomp, \
                                epoc, \
                                peri, \
                                cosi, \
                                typevisu, \
                                radistar=radistar, \
                                boolsingside=boolsingside, \
                                boolanim=boolanim, \
                               )
        



