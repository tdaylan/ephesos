import sys
import os

import numpy as np

from tqdm import tqdm

import json

import time as timemodu

from numba import jit, prange
import h5py
import fnmatch

import astropy
import astropy as ap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.timeseries

import pandas as pd

import multiprocessing
from functools import partial

import scipy as sp
import scipy.interpolate

import astroquery
import astroquery.mast

import matplotlib.pyplot as plt

# own modules
import tdpy
from tdpy import summgene
import lygos
import hattusa


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
    """
    Get a dictionary of the sources in the TIC8 with the fields in the TIC8
    
    Keyword arguments   
        typepopl: type of the population
            'ffimhcon': TESS targets with contamination larger than
            'ffimm135': TESS targets brighter than mag 13.5
            'tessnomi2min': 2-minute TESS targets obtained by merging the SPOC 2-min bulk downloads

    Returns a dictionary with keys:
        rasc: RA
        decl: declination
        tmag: TESS magnitude
        radistar: radius of the star
        massstar: mass of the star
    """
    
    if typeverb > 0:
        print('Retrieving a dictionary of TIC8 for population %s...' % typepopl)
    
    if typepopl.startswith('tess'):
        if typepopl[4:].startswith('nomi'):
            listtsec = np.arange(1, 27)
        elif typepopl[4:].endswith('extd'):
            listtsec = np.arange(27, 39)
        else:
            listtsec = [int(typepopl[-2:])]
        numbtsec = len(listtsec)
        indxtsec = np.arange(numbtsec)

    pathlistticidata = os.environ['EPHESUS_DATA_PATH'] + '/data/listticidata/'
    os.system('mkdir -p %s' % pathlistticidata)

    path = pathlistticidata + 'listticidata_%s.csv' % typepopl
    if not os.path.exists(path):
        
        # dictionary of strings that will be keys of the output dictionary
        dictstrg = dict()
        dictstrg['ID'] = 'tici'
        dictstrg['ra'] = 'rasc'
        dictstrg['dec'] = 'decl'
        dictstrg['Tmag'] = 'tmag'
        dictstrg['rad'] = 'radistar'
        dictstrg['mass'] = 'massstar'
        dictstrg['Teff'] = 'tmptstar'
        dictstrg['logg'] = 'loggstar'
        dictstrg['MH'] = 'metastar'
        liststrg = list(dictstrg.keys())
        
        print('typepopl')
        print(typepopl)
        if typepopl.startswith('tessnomi'):
            
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
            
            u, indxuniq = np.unique(dictquer['tici'], return_index=True)
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = dictquer[dictstrg[name]][indxuniq]

            numbtarg = dictquer['radistar'].size
            if typeverb > 0:
                print('%d observed 2-min targets...' % numbtarg)
            
        elif typepopl.startswith('ffim'):
            if typepopl == 'ffimhcon':
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                #"columns":"c.*", \
                'columns':'rad, mass, ID, contratio', \
                                                             'filters':[{'paramName':'contratio', 'values':[{"min":10., "max":1e3}]}]}}
            if typepopl == 'ffimm135':
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                #"columns":"c.*", \
                'columns':'rad, mass, ID', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":5., "max":8.5}]}]}}
            headers, outString = quer_mast(request)
            listdictquer = json.loads(outString)['data']
            if typeverb > 0:
                print('%d matches...' % len(listdictquer))
            dictquer = dict()
            for name in listdictquer[0].keys():
                dictquer[name] = np.empty(len(listdictquer))
                for k in range(len(listdictquer)):
                    dictquer[name][k] = listdictquer[k][name]
        else:
            print('Unrecognized population name: %s' % typepopl)
            raise Exception('')
        
        if typeverb > 0:
            #print('%d targets...' % numbtarg)
            print('Writing to %s...' % path)
        pd.DataFrame.from_dict(dictquer).to_csv(path)
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        dictquer = pd.read_csv(path).to_dict(orient='list')
        
        for name in dictquer.keys():
            dictquer[name] = np.array(dictquer[name])
        del dictquer['Unnamed: 0']

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


def plot_orbt( \
              # path to write the plot
              path, \
              # radius of the planets [R_E]
              radiplan, \
              # sum of radius of planet and star divided by the semi-major axis
              rsma, \
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
              typefileplot='pdf', \

              # verbosity level
              typeverb=1, \
             ):

    dictfact = retr_factconv()
    
    mpl.use('Agg')

    numbcomp = len(radiplan)
    
    if isinstance(radiplan, list):
        radiplan = np.array(radiplan)

    if isinstance(rsma, list):
        rsma = np.array(rsma)

    if isinstance(epoc, list):
        epoc = np.array(epoc)

    if isinstance(peri, list):
        peri = np.array(peri)

    if isinstance(cosi, list):
        cosi = np.array(cosi)

    if listcolrcomp is None:
        listcolrcomp = retr_listcolrcomp(numbcomp)

    if liststrgcomp is None:
        liststrgcomp = retr_liststrgcomp(numbcomp)
    
    # semi-major axes of the planets [AU]
    smax = (radiplan / dictfact['rsre'] + radistar) / dictfact['aurs'] / rsma
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
        radiplanmerc = 0.3829 # [R_E]
    
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
    rrat = radiplan / radistar
    
    rflxtranmodl = retr_rflxmodl(time, peri, epoc, radiplan, radistar, rsma, cosi) - 1.
    
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
        cmnd = 'convert -delay 5'
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
            radi = radiplan[j] / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((xposelli[indxtimeiter[k], j], yposelli[indxtimeiter[k], j], 0), radius=radi, color=colrplan, zorder=3)
            axis.add_artist(w1)
            
        ## upper half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 0, 180, fc=colrstar, zorder=4, edgecolor=colrstar)
        axis.add_artist(w1)
        
        if typevisu == 'cartmerc':
            ## add Mercury
            axis.text(.387, 0.01, 'Mercury', color='grey', ha='right')
            radi = radiplanmerc / dictfact['rsre'] / dictfact['aurs'] * factplan
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
            maxmxaxi = max(1.2 * np.amax(smax), 0.4)
        else:
            maxmxaxi = 1.2 * np.amax(smax)
        
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
            pathtemp = '%s_%s_%04d.%s' % (path, typevisu, k, typefileplot)
        else:
            pathtemp = '%s_%s.%s' % (path, typevisu, typefileplot)
        print('Writing to %s...' % pathtemp)
        plt.savefig(pathtemp)
        plt.close()
        
        if boolanim:
            listpathtemp.append(pathtemp)
            cmnd += ' %s' % pathtemp 
    if boolanim:
        cmnd += ' %s_%s.gif' % (path, typevisu)
        os.system(cmnd)
        for pathtemp in listpathtemp:
            cmnd = 'rm %s' % pathtemp
            os.system(cmnd)


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
    dictcatl['rasc'] = data[:, 0]
    dictcatl['decl'] = data[:, 1]
    dictcatl['stdvrvel'] = data[:, -4]
    
    return dictcatl


def retr_dicthostplan(namepopl):
    
    pathlygo = os.environ['LYGOS_DATA_PATH'] + '/'
    path = pathlygo + 'data/dicthost%s.csv' % namepopl
    if os.path.exists(path):
        print('Reading from %s...' % path)
        dicthost = pd.read_csv(path).to_dict(orient='list')
        del dicthost['Unnamed: 0']
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
        listnamefeatcomp = ['epoc', 'peri', 'duratran', 'radicomp', 'masscomp']
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
        pd.DataFrame.from_dict(dicthost).to_csv(path)

    return dicthost


def retr_dicttoii(toiitarg=None, boolreplexar=False, typeverb=1):
    
    dictfact = retr_factconv()
    
    pathlygo = os.environ['LYGOS_DATA_PATH'] + '/'
    pathexof = pathlygo + 'data/exofop_tess_tois.csv'
    if typeverb > 0:
        print('Reading from %s...' % pathexof)
    objtexof = pd.read_csv(pathexof, skiprows=0)
    
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
        dicttoii['nameplan'] = np.empty(numbcomp, dtype=object)
        for kk, k in enumerate(indxcomp):
            dicttoii['nameplan'][kk] = 'TOI-' + str(dicttoii['toii'][kk])
            dicttoii['namestar'][kk] = 'TOI-' + str(dicttoii['toii'][kk])[:-3]
        
        dicttoii['dept'] = objtexof['Depth (ppm)'].values[indxcomp] * 1e-3 # [ppt]
        dicttoii['rrat'] = np.sqrt(dicttoii['dept'] * 1e-3)
        dicttoii['radicomp'] = objtexof['Planet Radius (R_Earth)'][indxcomp].values
        dicttoii['stdvradicomp'] = objtexof['Planet Radius (R_Earth) err'][indxcomp].values
        
        rascstarstrg = objtexof['RA'][indxcomp].values
        declstarstrg = objtexof['Dec'][indxcomp].values
        dicttoii['rascstar'] = np.empty_like(dicttoii['radicomp'])
        dicttoii['declstar'] = np.empty_like(dicttoii['radicomp'])
        for k in range(dicttoii['radicomp'].size):
            objt = astropy.coordinates.SkyCoord('%s %s' % (rascstarstrg[k], declstarstrg[k]), unit=(astropy.units.hourangle, astropy.units.deg))
            dicttoii['rascstar'][k] = objt.ra.degree
            dicttoii['declstar'][k] = objt.dec.degree

        dicttoii['strgcomm'] = np.empty(numbcomp, dtype=object)
        dicttoii['strgcomm'][:] = objtexof['Comments'][indxcomp].values
        
        #objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar']*astropy.units.degree, \
        #                                       dec=dicttoii['declstar']*astropy.units.degree, frame='icrs')
        
        objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar'], \
                                               dec=dicttoii['declstar'], frame='icrs', unit='deg')
        
        # transit duration
        dicttoii['duratran'] = objtexof['Duration (hours)'].values[indxcomp] # [hours]
        
        # galactic longitude
        dicttoii['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dicttoii['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dicttoii['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dicttoii['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        dicttoii['tsmmacwg'] = objtexof['ACWG TSM'][indxcomp].values
        dicttoii['esmmacwg'] = objtexof['ACWG ESM'][indxcomp].values
    
        dicttoii['facidisc'] = np.empty(numbcomp, dtype=object)
        dicttoii['facidisc'][:] = 'Transiting Exoplanet Survey Satellite (TESS)'
        
        dicttoii['peri'] = objtexof['Period (days)'][indxcomp].values
        dicttoii['peri'][np.where(dicttoii['peri'] == 0)] = np.nan

        dicttoii['epoc'] = objtexof['Epoch (BJD)'][indxcomp].values

        dicttoii['tmagsyst'] = objtexof['TESS Mag'][indxcomp].values
        dicttoii['stdvtmagsyst'] = objtexof['TESS Mag err'][indxcomp].values

        # transit duty cycle
        dicttoii['dcyc'] = dicttoii['duratran'] / dicttoii['peri'] / 24.
        
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
            dicttemp['masscomp'] = np.ones_like(dicttoii['radicomp']) + np.nan
            dicttemp['stdvmasscomp'] = np.ones_like(dicttoii['radicomp']) + np.nan
            
            numbsamppopl = 10
            indx = np.where(np.isfinite(dicttoii['radicomp']))[0]
            for n in tqdm(range(indx.size)):
                k = indx[n]
                meanvarb = dicttoii['radicomp'][k]
                stdvvarb = dicttoii['stdvradicomp'][k]
                
                # if radius uncertainty is not available, assume that it is small, so the mass uncertainty will be dominated by population uncertainty
                if not np.isfinite(stdvvarb):
                    stdvvarb = 1e-3 * dicttoii['radicomp'][k]
                else:
                    stdvvarb = dicttoii['stdvradicomp'][k]
                
                # sample from a truncated Gaussian
                listradiplan = tdpy.samp_gaustrun(1000, dicttoii['radicomp'][k], stdvvarb, 0., np.inf)
                
                # estimate the mass from samples
                listmassplan = retr_massfromradi(listradiplan)
                
                dicttemp['masscomp'][k] = np.mean(listmassplan)
                dicttemp['stdvmasscomp'][k] = np.std(listmassplan)
                
            if typeverb > 0:
                print('Writing to %s...' % path)
            pd.DataFrame.from_dict(dicttemp).to_csv(path)
        else:
            if typeverb > 0:
                print('Reading from %s...' % path)
            dicttemp = pd.read_csv(path).to_dict(orient='list')
            
            for name in dicttemp:
                dicttemp[name] = np.array(dicttemp[name])
                if toiitarg is not None:
                    dicttemp[name] = dicttemp[name][indxcomp]

        dicttoii['masscomp'] = dicttemp['masscomp']
        
        dicttoii['stdvmasscomp'] = dicttemp['stdvmasscomp']
        
        dicttoii['masstotl'] = dicttoii['massstar'] + dicttoii['masscomp'] / dictfact['msme']
        dicttoii['smax'] = retr_smaxkepl(dicttoii['peri'], dicttoii['masstotl'])
        
        dicttoii['inso'] = dicttoii['lumistar'] / dicttoii['smax']**2
        
        dicttoii['tmptplan'] = dicttoii['tmptstar'] * np.sqrt(dicttoii['radistar'] / dicttoii['smax'] / 2. / dictfact['aurs'])
        # temp check if factor of 2 is right
        dicttoii['stdvtmptplan'] = np.sqrt((dicttoii['stdvtmptstar'] / dicttoii['tmptstar'])**2 + \
                                                        0.5 * (dicttoii['stdvradistar'] / dicttoii['radistar'])**2) / np.sqrt(2.)
        
        dicttoii['densplan'] = 5.51 * dicttoii['masscomp'] / dicttoii['radicomp']**3 # [g/cm^3]
        dicttoii['booltran'] = np.ones_like(dicttoii['toii'], dtype=bool)
    
        dicttoii['vesc'] = retr_vesc(dicttoii['masscomp'], dicttoii['radicomp'])
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
        calc_tsmmesmm(dicttoii)
    
        # turn zero TSM ACWG or ESM ACWG into NaN
        indx = np.where(dicttoii['tsmmacwg'] == 0)[0]
        dicttoii['tsmmacwg'][indx] = np.nan
        
        print('dicttoii[tsmmacwg]')
        summgene(dicttoii['tsmmacwg'])
        print('indx where it is 0')
        summgene(indx)
        
        indx = np.where(dicttoii['esmmacwg'] == 0)[0]
        dicttoii['esmmacwg'][indx] = np.nan

        print('dicttoii[esmmacwg]')
        summgene(dicttoii['esmmacwg'])
        print('indx where it is 0')
        summgene(indx)
        
    return dicttoii


def calc_tsmmesmm(dictpopl, boolsamp=False):
    
    if boolsamp:
        numbsamp = 1000
    else:
        numbsamp = 1

    numbcomp = dictpopl['radicomp'].size
    listtsmm = np.empty((numbsamp, numbcomp)) + np.nan
    listesmm = np.empty((numbsamp, numbcomp)) + np.nan
    
    for n in range(numbcomp):
        
        if not np.isfinite(dictpopl['tmptplan'][n]):
            continue
        
        if not np.isfinite(dictpopl['radicomp'][n]):
            continue
        
        if boolsamp:
            if not np.isfinite(dictpopl['stdvradicomp'][n]):
                stdv = dictpopl['radicomp'][n]
            else:
                stdv = dictpopl['stdvradicomp'][n]
            listradiplan = tdpy.samp_gaustrun(numbsamp, dictpopl['radicomp'][n], stdv, 0., np.inf)
            
            listmassplan = tdpy.samp_gaustrun(numbsamp, dictpopl['masscomp'][n], dictpopl['stdvmasscomp'][n], 0., np.inf)

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
            listradiplan = dictpopl['radicomp'][None, n]
            listtmptplan = dictpopl['tmptplan'][None, n]
            listmassplan = dictpopl['masscomp'][None, n]
            listradistar = dictpopl['radistar'][None, n]
            listkmagsyst = dictpopl['kmagsyst'][None, n]
            listjmagsyst = dictpopl['jmagsyst'][None, n]
            listtmptstar = dictpopl['tmptstar'][None, n]
        
        # TSM
        listtsmm[:, n] = retr_tsmm(listradiplan, listtmptplan, listmassplan, listradistar, listjmagsyst)

        # ESM
        listesmm[:, n] = retr_esmm(listtmptplan, listtmptstar, listradiplan, listradistar, listkmagsyst)
        
        #if (listesmm[:, n] < 1e-10).any():
        #    print('listradiplan')
        #    summgene(listradiplan)
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


def anim_tmptdete(timefull, lcurfull, meantimetmpt, lcurtmpt, pathimag, listindxtimeposimaxm, corrprod, corr, strgextn='', colr=None):
    
    numbtimefull = timefull.size
    numbtimekern = lcurtmpt.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    
    listpath = []
    cmnd = 'convert -delay 20'
    
    numbtimeanim = min(200, numbtimefullruns)
    indxtimefullrunsanim = np.random.choice(indxtimefullruns, size=numbtimeanim, replace=False)
    indxtimefullrunsanim = np.sort(indxtimefullrunsanim)

    for tt in indxtimefullrunsanim:
        
        path = pathimag + 'lcur%s_%08d.pdf' % (strgextn, tt)
        listpath.append(path)
        if not os.path.exists(path):
            plot_tmptdete(timefull, lcurfull, tt, meantimetmpt, lcurtmpt, path, listindxtimeposimaxm, corrprod, corr)
        cmnd += ' %s' % path
    
    pathanim = pathimag + 'lcur%s.gif' % strgextn
    cmnd += ' %s' % pathanim
    print('cmnd')
    print(cmnd)
    os.system(cmnd)
    cmnd = 'rm'
    for path in listpath:
        cmnd += ' ' + path
    os.system(cmnd)


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
    

def srch_flar(time, lcur, typeverb=1, strgextn='', numbkern=3, minmscalfalltmpt=None, maxmscalfalltmpt=None, \
                                                                    pathimag=None, boolplot=True, boolanim=False, thrs=None):

    minmtime = np.amin(time)
    timeflartmpt = 0.
    amplflartmpt = 1.
    scalrisetmpt = 0. / 24.
    difftime = np.amin(time[1:] - time[:-1])
    
    print('time')
    summgene(time)
    print('difftime')
    print(difftime)
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
        listlcurtmpt[k] = hattusa.retr_lcurmodl_flarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
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
    """
    Make a matrix with rows as the shifted and windowed copies of the time series.
    """
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


def corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, typeverb=2, thrs=None, strgextn='', pathimag=None, boolplot=True, boolanim=False):
    
    timeoffs = np.amin(time) // 1000
    timeoffs *= 1000
    time -= timeoffs
    
    if typeverb > 1:
        timeinit = timemodu.time()
    
    print('corr_tmpt()')
    print('time')
    summgene(time)
    
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
                
                path = pathimag + 'lcurflar_ch%02d%s.pdf' % (l, strgextntotl)
                plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                for n in range(numbdeteplot):
                    figr, axis = plt.subplots(figsize=(8, 4), sharex=True)
                    for i in range(numbdeteplot):
                        indxtimeplot = indxtimekern[k] + listindxtimeposimaxm[l][k][i]
                        proc_axiscorr(timechun, lcurchun, axis, listindxtimeposimaxm[l][k], indxtime=indxtimeplot, timeoffs=timeoffs)
                    path = pathimag + 'lcurflar_ch%02d%s_det.pdf' % (l, strgextntotl)
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
def retr_indxtran(time, epoc, peri, duratotl=None):
    '''
    Find the transit indices for a given time axis, epoch, period, and optionally transit duration
    '''

    if np.isfinite(peri):
        if duratotl is None:
            duratemp = 0.
        else:
            duratemp = duratotl
        intgminm = np.ceil((np.amin(time) - epoc - duratemp / 48.) / peri)
        intgmaxm = np.ceil((np.amax(time) - epoc - duratemp / 48.) / peri)
        indxtran = np.arange(intgminm, intgmaxm + 1)
    else:
        indxtran = np.arange(1)
    
    return indxtran


def retr_indxtimetran(time, epoc, peri, duratotl, durafull=None, \
                      # type of the in-transit phase interval
                      typeineg=None, \
                      booloutt=False, boolseco=False):
    '''
    Return the indices of times during transit
    Duration is in hours.
    '''

    if not np.isfinite(time).all():
        raise Exception('')
    
    if not np.isfinite(duratotl).all():
        print('duratotl')
        print(duratotl)
        raise Exception('')
    
    indxtran = retr_indxtran(time, epoc, peri, duratotl)

    if boolseco:
        offs = 0.5
    else:
        offs = 0.

    listindxtimetran = []
    for n in indxtran:
        timetotlinit = epoc + (n + offs) * peri - duratotl / 48.
        timetotlfinl = epoc + (n + offs) * peri + duratotl / 48.
        if durafull is not None:
            timefullinit = epoc + (n + offs) * peri - durafull / 48.
            timefullfinl = epoc + (n + offs) * peri + durafull / 48.
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
    indxtimetran = np.concatenate(listindxtimetran)
    indxtimetran = np.unique(indxtimetran)
    
    if booloutt:
        indxtimeretr = np.setdiff1d(np.arange(time.size), indxtimetran)
    else:
        indxtimeretr = indxtimetran
    
    return indxtimeretr
    

def retr_timeedge(time, lcur, timebrek, \
                  # Boolean flag to add breaks at discontinuties
                  booladdddiscbdtr, \
                  timescal, \
                 ):
    
    difftime = time[1:] - time[:-1]
    indxtimebrek = np.where(difftime > timebrek)[0]
    
    if booladdddiscbdtr:
        listindxtimebrekaddi = []
        for k in range(3, len(time) - 3):
            diff = lcur[k] - lcur[k-1]
            if abs(diff) > 5 * np.std(lcur[k-3:k]) and abs(diff) > 5 * np.std(lcur[k:k+3]):
                listindxtimebrekaddi.append(k)
                #print('k')
                #print(k)
                #print('diff')
                #print(diff)
                #print('np.std(lcur[k:k+3])')
                #print(np.std(lcur[k:k+3]))
                #print('np.std(lcur[k-3:k])')
                #print(np.std(lcur[k-3:k]))
                #print('')
        listindxtimebrekaddi = np.array(listindxtimebrekaddi, dtype=int)
        indxtimebrek = np.concatenate([indxtimebrek, listindxtimebrekaddi])
        indxtimebrek = np.unique(indxtimebrek)

    timeedge = [0, np.inf]
    for k in indxtimebrek:
        timeedgeprim = (time[k] + time[k+1]) / 2.
        timeedge.append(timeedgeprim)
    timeedge = np.array(timeedge)
    timeedge = np.sort(timeedge)

    return timeedge


def retr_tsecticibase(tici):
    '''
    Retrieve the list of sectors for which SPOC light curves are available for target, using the predownloaded database of light curves
    as opposed to individual download calls over internet
    '''
    pathbase = os.environ['TESS_DATA_PATH'] + '/data/lcur/'
    path = pathbase + 'tsec/tsec_spoc_%016d.csv' % tici
    if not os.path.exists(path):
        listtsecsele = np.arange(1, 50)
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


def bdtr_tser( \
              # time grid
              time, \
              # dependent variable
              lcur, \
              
              # epoc, period, and duration of mask
              epocmask=None, perimask=None, duramask=None, \
              
              # verbosity level
              typeverb=1, \
              
              # minimum gap to break the time-series into regions
              timebrek=None, \
              
              # Boolean flag to add breaks at vertical discontinuties
              booladdddiscbdtr=False, \
              
              # baseline detrend type
              ## 'medi':
              ## 'spln':
              typebdtr=None, \
              
              # order of the spline
              ordrspln=None, \
              # time scale of the spline detrending
              timescalbdtrspln=None, \
              # time scale of the median detrending
              timescalbdtrmedi=None, \
             ):
    '''
    Baseline-detrend a time-series.
    '''
    
    if typebdtr is None:
        typebdtr = 'spln'
    if timebrek is None:
        timebrek = 0.1 # [day]
    if ordrspln is None:
        ordrspln = 3
    if timescalbdtrspln is None:
        timescalbdtrspln = 0.5
    if timescalbdtrmedi is None:
        timescalbdtrmedi = 0.5 # [hour]
    
    if typebdtr == 'spln':
        timescal = timescalbdtrspln
    else:
        timescal = timescalbdtrmedi
    if typeverb > 0:
        print('Detrending the light curve with at a time scale of %.g days...' % timescal)
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits while detrending...')
        
    # determine the times at which the light curve will be broken into pieces
    timeedge = retr_timeedge(time, lcur, timebrek, booladdddiscbdtr, timescal)

    numbedge = len(timeedge)
    numbregi = numbedge - 1
    indxregi = np.arange(numbregi)
    lcurbdtrregi = [[] for i in indxregi]
    indxtimeregi = [[] for i in indxregi]
    indxtimeregioutt = [[] for i in indxregi]
    listobjtspln = [[] for i in indxregi]
    for i in indxregi:
        if typeverb > 1:
            print('i')
            print(i)
        # find times inside the region
        indxtimeregi[i] = np.where((time >= timeedge[i]) & (time <= timeedge[i+1]))[0]
        timeregi = time[indxtimeregi[i]]
        lcurregi = lcur[indxtimeregi[i]]
        
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
        
        if typebdtr == 'spln':
            if typeverb > 1:
                print('lcurregi[indxtimeregioutt[i]]')
                summgene(lcurregi[indxtimeregioutt[i]])
            # fit the spline
            if lcurregi[indxtimeregioutt[i]].size > 0:
                if timeregi[indxtimeregioutt[i]].size < 4:
                    print('Warning! Only %d points available for spline! This will result in a trivial baseline-detrended light curve (all 1s).' \
                                                                                                                % timeregi[indxtimeregioutt[i]].size)
                    listobjtspln[i] = None
                    lcurbdtrregi[i] = np.ones_like(lcurregi)
                else:
                    
                    minmtime = np.amin(timeregi[indxtimeregioutt[i]])
                    maxmtime = np.amax(timeregi[indxtimeregioutt[i]])
                    numbknot = int((maxmtime - minmtime) / timescalbdtrspln)
                    
                    timeknot = np.linspace(minmtime, maxmtime, numbknot)
                    timeknot = timeknot[1:-1]
                    
                    if numbknot >= 4:
                        print('Region %d. %d knots used.' % (i, timeknot.size))
                        if numbknot > 1:
                            print('Knot separation: %.3g hours' % (24 * (timeknot[1] - timeknot[0])))
                    
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


def retr_brgtlmdk(cosg, coe1, coe2, typelmdk='quad'):
    
    if typelmdk == 'quad':
        brgtlmdk = 1. - coe1 * (1. - cosg) - coe2 * (1. - cosg)**2
    
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


def srch_pbox_work(listperi, listarrytser, listdcyc, listepoc, listduratranlevl, i):
    
    numbperi = len(listperi[i])
    numbdcyc = len(listdcyc[0])
    
    numblevlrebn = len(listduratranlevl)
    indxlevlrebn = np.arange(numblevlrebn)
    
    listminmtime = np.empty(numblevlrebn)
    listmedirflx = np.empty(numblevlrebn)
    for b in indxlevlrebn:
        #stdvrflx = 0.01 + 0 * arrytser[:, 1]
        #vari = stdvrflx**2
        #weig = 1. / vari / np.sum(1. / vari)
        listminmtime[b] = np.amin(listarrytser[b][:, 0])
        listmedirflx[b] = np.median(listarrytser[b][:, 1])

    #conschi2 = np.sum(weig * arrytser[:, 1]**2)
    #listtermchi2 = np.empty(numbperi)
    
    lists2nr = np.zeros(numbperi) - 1e100
    
    listdeptmaxm = np.empty(numbperi)
    listdcycmaxm = np.empty(numbperi)
    listepocmaxm = np.empty(numbperi)
    
    listphas = [[] for b in indxlevlrebn]
    for k in tqdm(range(len(listperi[i]))):
        
        peri = listperi[i][k]
        
        for b in indxlevlrebn:
            listphas[b] = (listarrytser[b][:, 0] % peri) / peri
        
        for l in range(len(listdcyc[k])):
            
            b = np.digitize(listdcyc[k][l] * peri * 24., listduratranlevl) - 1
            #b = 0
            
            #print('listduratranlevl')
            #print(listduratranlevl)
            #print('listdcyc[k][l] * peri * 24.')
            #print(listdcyc[k][l] * peri * 24.)
            #print('b')
            #print(b)

            dydchalf = listdcyc[k][l] / 2.

            phasdiff = (listepoc[k][l] - listminmtime[b]) / peri
            
            #print('listphas[b]')
            #summgene(listphas[b])
            #print('')
            
            for m in range(len(listepoc[k][l])):
                
                indxitra = srch_pbox_work_loop(m, listphas[b], phasdiff, dydchalf)
                
                if indxitra.size == 0:
                    continue
    
                dept = listmedirflx[b] - np.mean(listarrytser[b][:, 1][indxitra])
                
                stdv = np.std(listarrytser[b][:, 1][indxitra])
                
                if stdv != 0:
                    s2nr = dept / stdv
                else:
                    s2nr = 1.
                
                if s2nr > lists2nr[k]:
                    lists2nr[k] = s2nr
                    listdeptmaxm[k] = dept
                    listdcycmaxm[k] = listdcyc[k][l]
                    listepocmaxm[k] = listepoc[k][l][m]
                
                if not np.isfinite(s2nr):
                    print('b')
                    print(b)
                    print('listarrytser[b][:, 1]')
                    summgene(listarrytser[b][:, 1])
                    #print('dept')
                    #print(dept)
                    #print('np.std(rflx[indxitra])')
                    #summgene(np.std(rflx[indxitra]))
                    #print('rflx[indxitra]')
                    #summgene(rflx[indxitra])
                    raise Exception('')
                    
                #timechecloop[0][k, l, m] = timemodu.time()
                #print('peri')
                #print(peri)
                #print('dcyc')
                #print(dcyc)
                #print('epoc')
                #print(epoc)
                #print('phasdiff')
                #summgene(phasdiff)
                #print('phasoffs')
                #summgene(phasoffs)
                
                #print('booltemp')
                #summgene(booltemp)
                #print('indxitra')
                #summgene(indxitra)
                #print('dept')
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
                #print('dept')
                #print(dept)
                #print('indxitra')
                #summgene(indxitra)
                #print('s2nr')
                #print(s2nr)
                #print('')

                #figr, axis = plt.subplots(2, 1, figsize=(8, 8))
                #axis[0].plot(time, rflx, color='b', ls='', marker='o', rasterized=True, ms=0.3)
                #axis[0].plot(time[indxitra], rflx[indxitra], color='r', ls='', marker='o', ms=0.3, rasterized=True)
                #indxtran = retr_indxtran(time, epoc, peri)
                #for indx in indxtran:
                #    axis[0].axvline(epoc + peri * indx, ls='--', alpha=0.5, lw=0.5)
                ##axis[1].set_xlabel('Time [BJD]')
                #
                #axis[1].plot(phas, rflx, color='b', ls='', marker='o', rasterized=True, ms=0.3)
                #axis[1].plot(phas[indxitra], rflx[indxitra], color='r', ls='', marker='o', ms=0.3, rasterized=True)
                ##axis[1].set_xlabel('Phase')
                #titl = '$P$=%.3g days, $T_0$=%g BJD, $q_{tr}$=%.3g, $D_{tr}$=%.3g, SNR=%.3g' % (peri, epoc, dcyc, dept, s2nr)
                ##axis[0].set_title(titl, usetex=False)
                #path = '/Users/tdaylan/Documents/work/data/troia/tessnomi2min_mock/TIC198486253/imag/rflx_tria_diag_%04d%04d.pdf' % (l, m)
                #print('Writing to %s...' % path)
                #plt.savefig(path, usetex=False)
                #plt.close()
        
        if not np.isfinite(lists2nr[k]):
            raise Exception('')

    listdeptmaxm *= 1e3  # [ppt]
    
    #listdept = listsigr / np.sqrt(terr * (1. - terr)) * 1e3 # [ppt]
    
    return lists2nr, listdeptmaxm, listdcycmaxm, listepocmaxm


def srch_pbox(arry, \
              
              # Boolean flag to search for positive boxes
              boolpuls=False, \
              
              ### maximum number of transiting objects
              maxmnumbpbox=1, \
              
              ticitarg=None, \
              
              dicttlsqinpt=None, \
              booltlsq=False, \
              
              # minimum period
              minmperi=None, \

              # maximum period
              maxmperi=None, \

              # factor by which to oversample the frequency grid
              factosam=1., \
                
              # minimum duty cycle
              minmdcyc=0.01, \
              
              # Boolean flag to enable multiprocessing
              boolprocmult=False, \
              
              # number of processes
              numbproc=None, \
              
              # detection threshold
              thrssdee=7.1, \
              
              # string extension to output files
              strgextn='', \
              # path where the output data will be stored
              pathdata=None, \

              # plotting
              ## path where the output images will be stored
              pathimag=None, \
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

             ):
    '''
    Search for periodic boxes in time-series data
    '''
    
    boolproc = False
    if pathdata is None:
        boolproc = True
    else:
        if strgextn == '':
            pathsave = pathdata + 'pbox.csv'
        else:
            pathsave = pathdata + 'pbox_%s.csv' % strgextn
        if not os.path.exists(pathsave):
            boolproc = True
    
    if not boolproc:
        if typeverb > 0:
            print('Reading %s...' % pathsave)
        
        dictpboxoutp = pd.read_csv(pathsave).to_dict(orient='list')
        for name in dictpboxoutp.keys():
            dictpboxoutp[name] = np.array(dictpboxoutp[name])
            if len(dictpboxoutp[name]) == 0:
                dictpboxoutp[name] = np.array([])
        
    else:
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
        
        dictpboxoutp = dict()
        dictpboxinte = dict()
        liststrgvarbsave = ['peri', 'epoc', 'dept', 'dura', 'sdee']
        for strg in liststrgvarbsave:
            dictpboxoutp[strg] = []
        
        arrysrch = np.copy(arry)
        if boolpuls:
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
        
        numbdcyc = 4
        indxdcyc = np.arange(numbdcyc)
        listdcyc = [[] for k in indxperi]
        listperilogt = np.log10(listperi)
        minmdcyclogt = -2. / 3. * listperilogt - 1. - 0.5
        maxmdcyclogt = -2. / 3. * listperilogt - 1. + 0.5
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
        minmduratran = listdcyc[-1][0] * listperi[-1] * 24
        
        # maximum transit duration
        maxmduratran = listdcyc[0][-1] * listperi[0] * 24
        
        if minmduratran < 5. * cade:
            print('Either the minimum transit duration is too small or the cadence is too large.')
            raise Exception('')
        
        # number of rebinned data sets
        numblevlrebn = 10
        indxlevlrebn = np.arange(numblevlrebn)
        
        # list of transit durations when rebinned data sets will be used
        listduratranlevl = np.linspace(minmduratran, maxmduratran, numblevlrebn)
        
        print('listduratranlevl')
        print(listduratranlevl)
        
        # rebinned data sets
        print('Number of data points: %d...' % numbtime)
        listarrysrch = []
        for b in indxlevlrebn:
            delt = 0.5 * listduratranlevl[b] / 24.
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
                diffepoc = max(cade / 24., 0.5 * listperi[k] * listdcyc[k][l])
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
                arrysrch[:, 1] -= (dictpboxinte['rflxtsermodl'] - 1.)
                
                if (dictpboxinte['rflxtsermodl'] == 1.).all():
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
                
                dictpboxoutp['peri'].append(objtresu.period)
                dictpboxoutp['epoc'].append(objtresu.T0)
                dictpboxoutp['dura'].append(objtresu.duration)
                dictpboxoutp['dept'].append(objtresu.depth * 1e3)
                dictpboxoutp['sdee'].append(objtresu.SDE)
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
                    data = objtpool.map(partial(srch_pbox_work, listperiproc, listarrysrch, listdcyc, listepoc, listduratranlevl), indxproc)
                    listsigr = np.concatenate([data[k][0] for k in indxproc])
                    listdeptmaxm = np.concatenate([data[k][1] for k in indxproc])
                    listdcycmaxm = np.concatenate([data[k][2] for k in indxproc])
                    listepocmaxm = np.concatenate([data[k][3] for k in indxproc])
                else:
                    listsigr, listdeptmaxm, listdcycmaxm, listepocmaxm = srch_pbox_work([listperi], listarrysrch, listdcyc, listepoc, listduratranlevl, 0)
                
                if (~np.isfinite(listsigr)).any():
                    raise Exception('')

                sizekern = 9
                resisigr = listsigr - scipy.ndimage.median_filter(listsigr, size=sizekern)
                indx = np.where(resisigr < np.percentile(resisigr, 95.))
                listsdee = resisigr / np.std(resisigr[indx])
                listsdee -= np.amin(listsdee)
                
                indxperimpow = np.argmax(listsdee)
                sdee = listsdee[indxperimpow]
                
                if not np.isfinite(sdee):
                    raise Exception('')

                dictpboxoutp['sdee'].append(sdee)
                dictpboxoutp['peri'].append(listperi[indxperimpow])
                dictpboxoutp['dura'].append(24. * listdcycmaxm[indxperimpow] * listperi[indxperimpow]) # [hours]
                dictpboxoutp['epoc'].append(listepocmaxm[indxperimpow])
                dictpboxoutp['dept'].append(listdeptmaxm[indxperimpow])
                
                print('sdee')
                print(sdee)

                # best-fit orbit
                dictpboxinte['listperi'] = listperi
                
                print('temp: assuming power is SNR')
                dictpboxinte['listsdee'] = listsdee
                dictpboxinte['listsigr'] = listsigr
                
                radistar = 1.
                pericomp = [dictpboxoutp['peri'][j]]
                epoccomp = [dictpboxoutp['epoc'][j]]
                radicomp = [dictfact['rsre'] * np.sqrt(dictpboxoutp['dept'][j] * 1e-3)]
                cosicomp = [0]
                rsma = [retr_rsma(dictpboxoutp['peri'][j], dictpboxoutp['dura'][j], cosicomp[0])]
                print('rsma')
                print(rsma)
                print('cosicomp')
                print(cosicomp)
                print('radicomp')
                print(radicomp)
                print('epoccomp')
                print(epoccomp)
                dictpboxinte['rflxtsermodl'] = [[] for b in indxlevlrebn]
                print('radicomp')
                print(radicomp)
                for b in indxlevlrebn:
                    dictpboxinte['rflxtsermodl'][b] = retr_rflxtranmodl(listarrysrch[b][:, 0], radistar, pericomp=pericomp, epoccomp=epoccomp, \
                                                                                            rsma=rsma, cosicomp=cosicomp, radicomp=radicomp, booltrap=False)
                    
                    if booldiag and (dictpboxinte['rflxtsermodl'][b] == 1).all():
                        raise Exception('')

                if pathimag is not None:
                    arrymetamodl = np.zeros((numbtimeplot, 3))
                    arrymetamodl[:, 0] = timemodlplot
                    arrymetamodl[:, 1] = retr_rflxtranmodl(timemodlplot, radistar, pericomp=pericomp, epoccomp=epoccomp, \
                                                                                            rsma=rsma, cosicomp=cosicomp, radicomp=radicomp, booltrap=False)
                    arrypsermodl = fold_tser(arrymetamodl, dictpboxoutp['epoc'][j], dictpboxoutp['peri'][j], phasshft=0.5)
                    arrypserdata = fold_tser(listarrysrch[b], dictpboxoutp['epoc'][j], dictpboxoutp['peri'][j], phasshft=0.5)
                        
                    dictpboxinte['timedata'] = listarrysrch[0][:, 0]
                    dictpboxinte['rflxtserdata'] = listarrysrch[0][:, 1]
                    dictpboxinte['phasdata'] = arrypserdata[:, 0]
                    dictpboxinte['rflxpserdata'] = arrypserdata[:, 1]

                    dictpboxinte['timemodl'] = arrymetamodl[:, 0]
                    dictpboxinte['phasmodl'] = arrypsermodl[:, 0]
                    dictpboxinte['rflxpsermodl'] = arrypsermodl[:, 1]
                
            if boolpuls:
                dictpboxinte['rflxtsermodl'] = 2. - dictpboxinte['rflxtsermodl']
                if pathimag is not None:
                    arrymetamodl[:, 1] = 2. - arrymetamodl[:, 1]
                    dictpboxinte['rflxpserdata'] = 2. - dictpboxinte['rflxpserdata']
                    dictpboxinte['rflxpsermodl'] = 2. - dictpboxinte['rflxpsermodl']
           
            if pathimag is not None:
                strgtitl = 'P=%.3g d, Dep=%.2g ppt, Dur=%.2g hr, SDE=%.3g' % \
                            (dictpboxoutp['peri'][j], dictpboxoutp['dept'][j], dictpboxoutp['dura'][j], dictpboxoutp['sdee'][j])
                for a in range(2):
                    if a == 0:
                        strg = 'sigr'
                    else:
                        strg = 'sdee'

                    # plot TLS power spectrum
                    figr, axis = plt.subplots(figsize=figrsizeydobskin)
                    
                    axis.axvline(dictpboxoutp['peri'][j], alpha=0.4, lw=3)
                    minmxaxi = np.amin(dictpboxinte['listperi'])
                    maxmxaxi = np.amax(dictpboxinte['listperi'])
                    for n in range(2, 10):
                        xpos = n * dictpboxoutp['peri'][j]
                        if xpos > maxmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    for n in range(2, 10):
                        xpos = dictpboxoutp['peri'][j] / n
                        if xpos < minmxaxi:
                            break
                        axis.axvline(xpos, alpha=0.4, lw=1, linestyle='dashed')
                    
                    axis.set_ylabel('Power')
                    axis.set_xlabel('Period [days]')
                    axis.set_xscale('log')
                    axis.plot(dictpboxinte['listperi'], dictpboxinte['list' + strg], color='black', lw=0.5)
                    axis.set_title(strgtitl)
                    plt.subplots_adjust(bottom=0.2)
                    path = pathimag + strg + '_pbox_tce%d_%s.%s' % (j, strgextn, typefileplot)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
                
                # plot light curve + TLS model
                figr, axis = plt.subplots(figsize=figrsizeydobskin)
                lcurpboxmeta = listarrysrch[0][:, 1]
                if boolpuls:
                    lcurpboxmetatemp = 2. - lcurpboxmeta
                else:
                    lcurpboxmetatemp = lcurpboxmeta
                axis.plot(listarrysrch[0][:, 0] - timeoffs, lcurpboxmetatemp, alpha=alphraww, marker='o', ms=1, ls='', color='grey')
                axis.plot(dictpboxinte['timemodl'] - timeoffs, arrymetamodl[:, 1], color='b')
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
                path = pathimag + 'rflx_pbox_tce%d_%s.%s' % (j, strgextn, typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()

                # plot phase curve + TLS model
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
                path = pathimag + 'pcur_pbox_tce%d_%s.%s' % (j, strgextn, typefileplot)
                print('Writing to %s...' % path)
                plt.savefig(path, dpi=200)
                plt.close()
            
            print('dictpboxoutp[sdee]')
            print(dictpboxoutp['sdee'])
            print('thrssdee')
            print(thrssdee)
            j += 1
        
            if sdee < thrssdee:
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


def retr_lcurtess( \
              # keyword string for MAST search
              strgmast=None, \
              
              # TIC ID
              ticitarg=None, \

              # RA for tesscut search
              rasctarg=None, \
    
              # DEC for tesscut search
              decltarg=None, \
    
              # Boolean flag to only consider FFI data
              boolffimonly=False, \

              # Boolean flag to only consider TPF data (2-min or 20-sec)
              booltpxfonly=False, \

              ## Boolean flag to use 20-sec TPF when available
              boolfasttpxf=True, \
              
              # a string label for the target to be optionally used for lygos if the target is identified with a RA&DEC
              labltarg=None, \

              ## subset of sectors to retrieve
              listtsecsele=None, \
              
              # input dictionary to lygos
              dictlygoinpt=dict(), \

              ## Boolean flag to apply quality mask
              boolmaskqual=True, \
         
              # TPF light curve extraction pipeline (FFIs are always extracted by lygos)
              ## 'lygos': lygos
              ## 'SPOC': SPOC
              typelcurtpxftess='SPOC', \
              #typelcurtpxftess='lygos', \
              
              ## type of SPOC light curve: 'PDC', 'SAP'
              typedataspoc='PDC', \
              
             ):
    '''
    Pipeline to retrieve TESS light curve of a target
    '''
    
    print('typelcurtpxftess')
    print(typelcurtpxftess)
    
    #strgmast, rasctarg, decltarg = setp_coorstrgmast(rasctarg, decltarg, strgmast)
    
    if not (strgmast is not None and ticitarg is None and rasctarg is None and decltarg is None or \
            strgmast is None and ticitarg is not None and rasctarg is None and decltarg is None or \
            strgmast is None and ticitarg is None and rasctarg is not None and decltarg is not None):
                    print('strgmast')
                    print(strgmast)
                    print('ticitarg')
                    print(ticitarg)
                    print('rasctarg')
                    print(rasctarg)
                    print('decltarg')
                    print(decltarg)
                    raise Exception('')
    
    # determine the MAST keyword to be used for the target
    if strgmast is not None:
        strgmasttemp = strgmast
    elif rasctarg is not None:
        strgmasttemp = '%g %g' % (rasctarg, decltarg)
    else:
        strgmasttemp = 'TIC %d' % ticitarg
    
    # determine the TIC ID to be used to search for available sectors
    ticitsec = None
    if ticitarg is None:
        listdictcatl = astroquery.mast.Catalogs.query_object(strgmasttemp, catalog='TIC', radius='40s')
        if len(listdictcatl) > 0 and listdictcatl[0]['dstArcSec'] < 1.:
            ticitsec = int(listdictcatl[0]['ID'])
        else:
            print('Warning! No TIC match to the MAST keyword: %s' % strgmasttemp)
    else:
        ticitsec = ticitarg
    print('ticitsec')
    print(ticitsec)

    if not booltpxfonly:
        # get the list of sectors for which TESS FFI data are available
        listtsecffim, temp, temp = retr_listtsec(strgmasttemp)
    
    # get the list of sectors for which TESS SPOC data are available
    print('typelcurtpxftess')
    print(typelcurtpxftess)
    if typelcurtpxftess == 'lygos' or ticitsec is None:
        listtsecspoc = np.array([], dtype=int)
        listpathspoc = np.array([])
    else:
        print('Retrieving the list of available TESS sectors for which there is SPOC light curve data...')
        listtsecspoc, listpathspoc = retr_tsecticibase(ticitsec)
    
    numbtsecspoc = listtsecspoc.size
    indxtsecspoc = np.arange(numbtsecspoc)

    if not booltpxfonly:
        # merge SPOC and FFI sector lists
        listtsecmerg = np.unique(np.concatenate((listtsecffim, listtsecspoc)))
    else:
        listtsecmerg = listtsecspoc

    # filter the list of sectors using the desired list of sectors, if any
    if listtsecsele is not None:
        listtsec = []
        for tsec in listtsecsele:
            if tsec in listtsecmerg:
                listtsec.append(tsec)
        listtsec = np.array(listtsec)
    else:
        listtsec = listtsecmerg
    
    print('listtsec')
    print(listtsec)
    
    numbtsec = len(listtsec)
    indxtsec = np.arange(numbtsec)

    listtcam = np.empty(numbtsec, dtype=int)
    listtccd = np.empty(numbtsec, dtype=int)
    
    # determine whether sectors have 2-minute cadence data
    booltpxf = retr_booltpxf(listtsec, listtsecspoc)
    print('booltpxf')
    print(booltpxf)
    
    if typelcurtpxftess == 'lygos':
        boollygo = np.ones(numbtsec, dtype=bool)
        booltpxflygo = True
        listtseclygo = listtsec
    if typelcurtpxftess == 'SPOC':
        boollygo = ~booltpxf
        booltpxflygo = False
        listtseclygo = listtsec[boollygo]
    listtseclygo = listtsec[boollygo]
    
    print('booltpxflygo')
    print(booltpxflygo)
    print('listtseclygo')
    print(listtseclygo)
    
    listarrylcur = [[] for o in indxtsec]
    if len(listtseclygo) > 0:
        
        dictlygoinpt['strgmast'] = strgmast
        dictlygoinpt['ticitarg'] = ticitarg
        dictlygoinpt['rasctarg'] = rasctarg
        dictlygoinpt['decltarg'] = decltarg
        dictlygoinpt['labltarg'] = labltarg
        dictlygoinpt['listtsecsele'] = listtseclygo
        dictlygoinpt['booltpxflygo'] = booltpxflygo
        if not 'boolmaskqual' in dictlygoinpt:
            dictlygoinpt['boolmaskqual'] = boolmaskqual
        
        print('Will run lygos on the target...')
        dictlygooutp = lygos.main.init( \
                                       **dictlygoinpt, \
                                      )
        
        print('listtsec')
        print(listtsec)
        for o, tseclygo in enumerate(listtsec):
            indx = np.where(dictlygooutp['listtsec'] == tseclygo)[0]
            print('indx')
            print(indx)
            if indx.size > 0:
                if len(dictlygooutp['listarry'][indx[0]]) == 0:
                    print('listtseclygo')
                    print(listtseclygo)
                    print('booltpxflygo')
                    print(booltpxflygo)
                    #raise Exception('')
                else:
                    listarrylcur[o] = dictlygooutp['listarry'][indx[0]]
                    listtcam[o] = dictlygooutp['listtcam'][indx[0]]
                    listtccd[o] = dictlygooutp['listtccd'][indx[0]]
    
    listarrylcursapp = None
    listarrylcurpdcc = None
    arrylcursapp = None
    arrylcurpdcc = None
    
    #listpathdownspoclcur = []
    print('listtsecspoc')
    print(listtsecspoc)
    if len(listtsecspoc) > 0 and typelcurtpxftess == 'SPOC':
        
        pathdatatess = os.environ['TESS_DATA_PATH'] + '/'
        # download data from MAST
        os.system('mkdir -p %s' % pathdatatess)
        
        print('Downloading SPOC data products...')
        
        listhdundataspoc = [[] for o in indxtsecspoc]
        #listpathdownspoc = []

        # get observation tables
        #listtablobsv = retr_listtablobsv(strgmasttemp)
        #listprodspoc = []
        #for k, tablobsv in enumerate(listtablobsv):
        #    
        #    listprodspoctemp = astroquery.mast.Observations.get_product_list(tablobsv)
        #    
        #    if listtablobsv['distance'][k] > 0:
        #        continue

        #    strgdesc = 'Light curves'
        #    listprodspoctemp = astroquery.mast.Observations.filter_products(listprodspoctemp, description=strgdesc)
        #    for a in range(len(listprodspoctemp)):
        #        boolfasttemp = listprodspoctemp[a]['obs_id'].endswith('fast')
        #        if not boolfasttemp:
        #            tsec = int(listprodspoctemp[a]['obs_id'].split('-')[1][1:])
        #            listprodspoc.append(listprodspoctemp)
        #
        #for k in range(len(listprodspoc)):
        #    manifest = astroquery.mast.Observations.download_products(listprodspoc[k], download_dir=pathdatatess)
        #    
        #    # to move files to an upstream folder
        #    #pathnest = manifest['Local Path'][0]
        #    #pathnestfold = pathnest.split('/')[-1]
        #    #cmnd = 'mv %s %s' % (pathnest, pathdatatess)
        #    #print('cmnd')
        #    #print(cmnd)
        #    ##os.system(cmnd)
        #    #cmnd = 'rmdir %s' % (pathnestfold)
        #    #print('cmnd')
        #    #print(cmnd)
        #    ##os.system(cmnd)

        #    #listpathdownspoclcur.append(manifest['Local Path'][0])

        ## make sure the list of paths to sector files are time-sorted
        #listpathdownspoc.sort()
        #listpathdownspoclcur.sort()
        
        ## read SPOC light curves
        if typedataspoc == 'SAP':
            print('Reading the SAP light curves...')
        else:
            print('Reading the PDC light curves...')
        listarrylcursapp = [[] for o in indxtsec] 
        listarrylcurpdcc = [[] for o in indxtsec] 
        for o in indxtsec:
            if not boollygo[o]:
                
                indx = np.where(listtsec[o] == listtsecspoc)[0][0]
                path = listpathspoc[indx]
                #path = listpathdownspoclcur[indx]
                listarrylcursapp[indx], indxtimequalgood, indxtimenanngood, listtsec[o], listtcam[o], listtccd[o] = \
                                                       read_tesskplr_file(path, typeinst='tess', strgtype='SAP_FLUX', boolmaskqual=boolmaskqual)
                listarrylcurpdcc[indx], indxtimequalgood, indxtimenanngood, listtsec[o], listtcam[o], listtccd[o] = \
                                                       read_tesskplr_file(path, typeinst='tess', strgtype='PDCSAP_FLUX', boolmaskqual=boolmaskqual)
            
                if typedataspoc == 'SAP':
                    arrylcur = listarrylcursapp[indx]
                else:
                    arrylcur = listarrylcurpdcc[indx]
                listarrylcur[o] = arrylcur
            
        if numbtsec == 0:
            print('No data have been retrieved.' % (numbtsec, strgtemp))
        else:
            if numbtsec == 1:
                strgtemp = ''
            else:
                strgtemp = 's'
            print('%d sector%s of data retrieved.' % (numbtsec, strgtemp))
        
        # merge light curves from different sectors
        arrylcursapp = np.concatenate([arry for arry in listarrylcursapp if len(arry) > 0], 0)
        arrylcurpdcc = np.concatenate([arry for arry in listarrylcurpdcc if len(arry) > 0], 0)
    
    # merge light curves from different sectors
    if [] in listarrylcur:
        print('listarrylcur contains an empty element. Will remove it.')
        listarrylcurtemp = []
        listindxtsecgood = []
        for o in indxtsec:
            if len(listarrylcur[o]) > 0:
                listindxtsecgood.append(o)
        listindxtsecgood = np.array(listindxtsecgood, dtype=int)
        print('listindxtsecgood')
        summgene(listindxtsecgood)
        listtsec = listtsec[listindxtsecgood]
        listtcam = listtcam[listindxtsecgood]
        listtccd = listtccd[listindxtsecgood]
        for indxtsecgood in listindxtsecgood:
            listarrylcurtemp.append(listarrylcur[indxtsecgood])
        listarrylcur = listarrylcurtemp

    if len(listarrylcur) > 0:
        arrylcur = np.concatenate(listarrylcur, 0)
        
        #if not np.isfinite(arrylcur).all():
        #    indxbadd = np.where(~np.isfinite(arrylcur))[0]
        #    print('arrylcur')
        #    summgene(arrylcur)
        #    print('indxbadd')
        #    summgene(indxbadd)
        #    raise Exception('')
    else:
        arrylcur = []

    for o, tseclygo in enumerate(listtsec):
        if not np.isfinite(listarrylcur[o][:, 1]).all():
            print('listtsec')
            print(listtsec)
            print('tseclygo')
            print(tseclygo)
            indxbadd = np.where(~np.isfinite(listarrylcur[o][:, 1]))[0]
            print('listarrylcur[o][:, 1]')
            summgene(listarrylcur[o][:, 1])
            print('indxbadd')
            summgene(indxbadd)
            raise Exception('')
    
    return arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc, listtsec, listtcam, listtccd, listpathspoc
   

def retr_subp(dictpopl, dictnumbsamp, dictindxsamp, namepoplinit, namepoplfinl, indx):
    
    dictindxsamp[namepoplinit][namepoplfinl] = indx
    dictpopl[namepoplfinl] = dict()
    for name in dictpopl[namepoplinit].keys():
        dictpopl[namepoplfinl][name] = dictpopl[namepoplinit][name][dictindxsamp[namepoplinit][namepoplfinl]]
    dictnumbsamp[namepoplfinl] = dictindxsamp[namepoplinit][namepoplfinl].size
    dictindxsamp[namepoplfinl] = dict()


def retr_dictpoplstarcomp(typepoplstar, typecomp, numbsyst=None, timeepoc=None):
    '''
    Get a dictionary with features of stars and their companions.
    '''

    dictpopl = dict()
    dictnumbsamp = dict()
    dictindxsamp = dict()
    dictindxsamp['star'] = dict()
    
    dictfact = retr_factconv()
    
    # get the features of the star population
    if typepoplstar == 'tessnomi2min':
        dictpopl['star'] = retr_dictpopltic8(typepoplstar, numbsyst=numbsyst)
        dictpopl['star']['densstar'] = 1.41 * dictpopl['star']['massstar'] / dictpopl['star']['radistar']**3
        dictpopl['star']['idenstar'] = dictpopl['star']['tici']
    elif typepoplstar == 'lsstwfds' or typepoplstar == 'tessexm2':
        dictpopl['star'] = dict()
        
        if numbsyst is None:
            if typepoplstar == 'tessexm2':
                numbsyst = 10000000
            if typepoplstar == 'lsstwfds':
                numbsyst = 1000000
        
        dictpopl['star']['idenstar'] = np.arange(numbsyst)
        
        dictpopl['star']['distsyst'] = tdpy.icdf_powr(np.random.rand(numbsyst), 100., 7000., -2.)
        dictpopl['star']['massstar'] = tdpy.icdf_powr(np.random.rand(numbsyst), 0.1, 10., 2.)
        
        dictpopl['star']['densstar'] = 1.4 * (1. / dictpopl['star']['massstar'])**(0.7)
        dictpopl['star']['radistar'] = (1.4 * dictpopl['star']['massstar'] / dictpopl['star']['densstar'])**(1. / 3.)
        
        dictpopl['star']['lumistar'] = dictpopl['star']['massstar']**4
        
        if typepoplstar == 'tessexm2':
            dictpopl['star']['tmag'] = tdpy.icdf_powr(np.random.rand(numbsyst), 6., 14., -2.)
        if typepoplstar == 'lsstwfds':
            dictpopl['star']['rmag'] = -2.5 * np.log10(dictpopl['star']['lumistar'] / dictpopl['star']['distsyst']**2)
            
            indx = np.where((dictpopl['star']['rmag'] < 24.) & (dictpopl['star']['rmag'] > 15.))[0]
            for name in ['distsyst', 'rmag', 'massstar', 'densstar', 'radistar', 'lumistar']:
                dictpopl['star'][name] = dictpopl['star'][name][indx]

    else:
        raise Exception('')
    
    dictnumbsamp['star'] = dictpopl['star']['radistar'].size

    # probability of occurence
    if typecomp == 'cosc':
        dictpopl['star']['numbcompstarmean'] = tdpy.samp_gaustrun(dictnumbsamp['star'], 1e-6, 1e-4, 0, np.inf)
        
        dictpopl['star']['rsum'] = dictpopl['star']['radistar']
    if typecomp == 'psys':
        
        masstemp = np.copy(dictpopl['star']['massstar'])
        masstemp[np.where(~np.isfinite(masstemp))] = 1.

        dictpopl['star']['numbcompstarmean'] = 0.5 * masstemp**(-1.)
        
        dictpopl['star']['rsum'] = dictpopl['star']['radistar']
        #dictpopl['star']['rsum'] = dictpopl['star']['radistar'] + dictpopl['star']['radicomp'] / dictfact['rsre']
        
    # number of companions per star
    dictpopl['star']['numbcompstar'] = np.random.poisson(dictpopl['star']['numbcompstarmean'])
    
    # Boolean flag of occurence
    dictpopl['star']['booloccu'] = dictpopl['star']['numbcompstar'] > 0
    
    # subpopulation where companions occur
    indx = np.where(dictpopl['star']['booloccu'])[0]
    retr_subp(dictpopl, dictnumbsamp, dictindxsamp, 'star', 'starcomp', indx)
    
    if typecomp == 'cosc':
        minmmasscomp = 5. # [Solar mass]
        maxmmasscomp = 200. # [Solar mass]
    if typecomp == 'psys':
        minmmasscomp = 0.5 # [Earth mass]
        maxmmasscomp = 1000. # [Earth mass]
    
    print('Sampling companion features...')
    
    for name in ['cosi', 'masscomp', 'radicomp', 'smax', 'epoc']:
        dictpopl['starcomp'][name] = [[] for k in range(indx.size)]
    for k in range(indx.size):
        
        if dictpopl['star']['numbcompstar'][k] == 0:
            continue

        # cosine of orbital inclinations
        dictpopl['starcomp']['cosi'][k] = np.random.rand(dictpopl['star']['numbcompstar'][k])
    
        # companion mass
        dictpopl['starcomp']['masscomp'][k] = tdpy.util.icdf_powr(np.random.rand(dictpopl['star']['numbcompstar'][k]), minmmasscomp, maxmmasscomp, 2.)
        
        # companion radius
        if typecomp == 'psys':
            dictpopl['starcomp']['radicomp'][k] = retr_radifrommass(dictpopl['starcomp']['masscomp'][k])
    
        # semi-major axes
        #if np.isfinite(dictpopl['comp']['densstar'][k]):
        #    densstar = dictpopl['comp']['densstar'][k]
        #else:
        #    densstar = 1.
        #dictpopl['comp']['radiroch'][k] = retr_radiroch(radistar, densstar, denscomp)
        #minmsmax = 2. * dictpopl['comp']['radiroch'][k]
        
        minmsmax = 3.# * dictpopl['comp']['radistar'][k]
        dictpopl['starcomp']['smax'][k] = dictpopl['starcomp']['radistar'][k] * tdpy.util.icdf_powr(np.random.rand(), minmsmax, 1e4, 2.) / dictfact['aurs']
        
        # epochs
        dictpopl['starcomp']['epoc'][k] = 1e8 * np.random.rand(dictpopl['star']['numbcompstar'][k])
        if timeepoc is not None:
            dictpopl['starcomp']['epoc'][k] = dictpopl['starcomp']['epoc'][k] + dictpopl['starcomp']['peri'][k] * \
                                                                        np.round((dictpopl['starcomp']['epoc'][k] - timeepoc) / dictpopl['starcomp']['peri'][k])
    
    dictpopl['starcomp']['rsma'] = dictpopl['comp']['radistar'] / dictpopl['comp']['smax'] / dictfact['aurs']
    
    # load star features into component features
    dictpopl['comp'] = dict()
    for name in list(dictpopl['star'].keys()):
        dictpopl['comp'][name] = np.empty(dictnumbsamp['comp'])
    cntr = 0
    for k in range(dictnumbsamp['star']):
        numb = dictpopl['star']['numbcompstar'][k]
        for name in list(dictpopl['star'].keys()):
            dictpopl['comp'][name][cntr:cntr+numb] = dictpopl['star'][name][k]
        cntr += numb
    
    dictnumbsamp['comp'] = np.sum(dictpopl['star']['numbcompstar'])

    # orbital inclinations
    dictpopl['comp']['incl'] = 180. / np.pi * np.arccos(dictpopl['comp']['cosi'])
    
    # total mass
    if typecomp == 'cosc':
        dictpopl['star']['masstotl'] = dictpopl['star']['massbhol'] + dictpopl['star']['massstar']
    if typecomp == 'psys':
        dictpopl['star']['masstotl'] = dictpopl['star']['massstar']
    
    print('Estimating orbital periods...')
    dictpopl['comp']['peri'] = retr_perikepl(dictpopl['comp']['smax'], dictpopl['comp']['masstotl'])
    
    # subpopulation where object transits
    dictindxsamp['comp'] = dict()
    indx = np.where(dictpopl['comp']['rsma'] > dictpopl['comp']['cosi'])[0]
    retr_subp(dictpopl, dictnumbsamp, dictindxsamp, 'comp', 'comptran', indx)

    # transit duration
    dictpopl['comptran']['duratran'] = retr_duratran(dictpopl['comptran']['peri'], \
                                                                   dictpopl['comptran']['rsma'], \
                                                                   dictpopl['comptran']['cosi'])
    dictpopl['comptran']['dcyc'] = dictpopl['comptran']['duratran'] / dictpopl['comptran']['peri'] / 24.
    if typecomp == 'psys':
        dictpopl['comptran']['rrat'] = dictpopl['comptran']['radicomp'] / dictpopl['comptran']['radistar'] / dictfact['rsre']
        dictpopl['comptran']['dept'] = 1e3 * dictpopl['comptran']['rrat']**2 # [ppt]
    if typecomp == 'cosc':
        dictpopl['comptran']['amplslen'] = retr_amplslen(dictpopl['comptran']['peri'], dictpopl['comptran']['radistar'], \
                                                                            dictpopl['comptran']['massbhol'], dictpopl['comptran']['massstar'])
        
    return dictpopl, dictnumbsamp, dictindxsamp
       

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
    Retrieve the list of sectors, cameras, and CCDs for which TESS data are available for the target
    '''
    
    print('Calling TESSCut with keyword %s to get the list of sectors for which TESS data are available...' % strgmast)
    tabltesscutt = astroquery.mast.Tesscut.get_sectors(strgmast, radius=0)

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


def exec_lspe(arrylcur, pathimag=None, pathdata=None, strgextn='', factnyqt=None, maxmfreq=None, factosam=1.):
    '''
    Calculate the LS periodogram of a time-series
    '''
    
    if maxmfreq is not None and factnyqt is not None:
        raise Exception('')
    
    if pathdata is not None:
        path = pathdata + 'spec_lspe_%s.csv' % strgextn
    
    if pathdata is not None and os.path.exists(path):
        print('Reading from %s...' % path)
        arry = np.loadtxt(path, delimiter=',')
        peri = arry[:, 0]
        powr = arry[:, 1]
    
    else:
        print('Calculating LS periodogram...')
        
        # factor by which the maximum frequency is compared to the Nyquist frequency
        if factnyqt is None:
            factnyqt = 2.
        
        time = arrylcur[:, 0]
        lcur = arrylcur[:, 1]
        numbtime = time.size
        minmtime = np.amin(time)
        maxmtime = np.amax(time)
        delttime = maxmtime - minmtime
        minmfreq = 1. / delttime
        freqnyqt = numbtime / delttime / 2.
        
        if maxmfreq is None:
            maxmfreq = factnyqt * freqnyqt
        
        # determine the frequency sampling resolution with N samples per line
        deltfreq = minmfreq / factosam / 2.
        freq = np.arange(minmfreq, maxmfreq, deltfreq)
        peri = 1. / freq
        
        powr = astropy.timeseries.LombScargle(time, lcur, nterms=2).power(freq)
        
        if pathdata is not None:
            arry = np.empty((peri.size, 2))
            arry[:, 0] = peri
            arry[:, 1] = powr
            print('Writing to %s...' % path)
            np.savetxt(path, arry, delimiter=',')
    indxperimpow = np.argmax(powr)
    perimpow = peri[indxperimpow]
    powrmpow = powr[indxperimpow]

    if pathimag is not None:
        path = pathimag + 'lspe_%s.pdf' % strgextn
        if not os.path.exists(path):
            figr, axis = plt.subplots(figsize=(8, 4))
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
                    
            axis.set_xscale('log')
            axis.set_xlabel('Period [days]')
            axis.set_ylabel('Power')
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    return perimpow, powrmpow


def plot_lcur(pathimag, strgextn, dictmodl=None, timedata=None, lcurdata=None, \
              # break the line of the model when separation is very large
              boolbrekmodl=True, \
              timedatabind=None, lcurdatabind=None, lcurdatastdvbind=None, \
              # Boolean flag to ignore any existing plot and overwrite
              boolwritover=False, \
              # size of the figure
              sizefigr=None, \
              timeoffs=0., \
              limtxaxi=None, \
              limtyaxi=None, \
              titl='', listcolrmodl=None):
    
    if strgextn == '':
        raise Exception('')
    
    path = pathimag + 'lcur_%s.pdf' % strgextn
    
    # skip plotting
    if not boolwritover and os.path.exists(path):
        return
    
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
        if listcolrmodl is None:
            listcolrmodl = ['r', 'b', 'g', 'c', 'm', 'orange', 'olive']
        k = 0
        for attr in dictmodl:
            if boolbrekmodl:
                diftimemodl = dictmodl[attr]['time'][1:] - dictmodl[attr]['time'][:-1]
                indxtimebrek = np.where(diftimemodl > 2 * np.amin(diftimemodl))[0] + 1
                indxtimebrek = np.concatenate([np.array([0]), indxtimebrek, np.array([dictmodl[attr]['time'].size - 1])])
                numbtimebrek = indxtimebrek.size
                numbtimechun = numbtimebrek - 1

                xdat = []
                ydat = []
                for n in range(numbtimechun):
                    xdat.append(dictmodl[attr]['time'][indxtimebrek[n]:indxtimebrek[n+1]])
                    ydat.append(dictmodl[attr]['lcur'][indxtimebrek[n]:indxtimebrek[n+1]])
                    
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
                axis.plot(xdat[n] - timeoffs, ydat[n], color=listcolrmodl[k], lw=1, label=label)
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

    axis.set_ylabel('Relative flux')
    axis.set_title(titl)
    
    if boollegd:
        axis.legend()

    plt.subplots_adjust(bottom=0.2)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    return path


def plot_pcur(pathimag, arrylcur=None, arrypcur=None, arrypcurbind=None, phascent=0., boolhour=False, epoc=None, peri=None, strgextn='', \
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
    path = pathimag + 'pcur%s.pdf' % strgextn
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
            arryrebn[k, 2] = np.sqrt(np.nansum(arry[indxxdat, 2]**2)) / indxxdat.size
    
    return arryrebn

    
def read_tesskplr_fold(pathfold, pathwrit, boolmaskqual=True, typeinst='tess', strgtype='PDCSAP_FLUX'):
    
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error
    '''

    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    listarry = []
    for path in listpath:
        arry = read_tesskplr_file(pathfold + path + '/' + path + '_lc.fits', typeinst=typeinst, strgtype=strgtype, boolmaskqual=boolmaskqual)
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
    
    boollcur = path.endswith('_lc.fits')

    tsec = listhdun[0].header['SECTOR']
    tcam = listhdun[0].header['CAMERA']
    tccd = listhdun[0].header['CCD']
        
    if boollcur:
        time = listhdun[1].data['TIME'] + 2457000
        if typeinst == 'TESS':
            time += 2457000
        if typeinst == 'kplr':
            time += 2454833
        flux = listhdun[1].data[strgtype]
        stdv = listhdun[1].data[strgtype+'_ERR']
    
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
    
        #print('HACKING, SKIPPING NORMALIZATION FOR SPOC DATA')
        # normalize
        arry[:, 2] /= np.median(arry[:, 1])
        arry[:, 1] /= np.median(arry[:, 1])
    
        return arry, indxtimequalgood, indxtimenanngood, tsec, tcam, tccd
    else:
        return listhdun, tsec, tcam, tccd


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


def retr_rsma(radiplan, radistar, smax):
    
    dictfact = retr_factconv()
    rsma = (radistar + radiplan / dictfact['rsre']) / (smax * dictfact['aurs'])
    return rsma


def retr_rflxmodlrise(time, timerise, coeflinerise, coefquadrise, coefline):
    
    timeoffs = time - timerise
    
    dflxline = coefline * timeoffs
    
    indxpost = np.where(timeoffs > 0)[0]
    dflxrise = np.zeros_like(time)
    dflxrise[indxpost] = coeflinerise * timeoffs[indxpost] + coefquadrise * timeoffs[indxpost]**2
    
    rflx = 1. + dflxrise + dflxline
    
    return rflx, dflxline, dflxrise


def retr_rflxmodlcosc( \
                      # time axis
                      time, \
                      ## epoch of the orbit
                      epoc, \
                      ## orbital period in days
                      peri, \
                      ## radius of the star in Solar radius
                      radistar, \
                      ## mass of the companion in Solar mass
                      masscomp, \
                      ## mass of the star in Solar mass
                      massstar, \
                      ## inclination of the orbit in degrees
                      incl, \
                      # Boolean flag to diagnose
                      booldiag=True, \

                     ):
    
    # phase
    phas = ((time - epoc) / peri) % 1.
    
    # conversion constants
    dictfact = retr_factconv()
    
    ## total mass
    masstotl = masscomp + massstar
    ## semi-major axis
    smax = retr_smaxkepl(peri, masstotl)
    ## radius of the star divided by the semi-major axis
    rsma = radistar / smax / dictfact['aurs']
    ## cosine of the inclination angle
    cosi = np.cos(incl / 180. * np.pi)
    
    ## self-lensing
    ### duration
    duratran = retr_duratran(peri, rsma, cosi)
    ### amplitude
    amplslen = retr_amplslen(peri, radistar, masscomp, massstar)
    ### signal
    dflxslen = np.zeros_like(time)
    if np.isfinite(duratran):
        indxtimetran = retr_indxtimetran(time, epoc, peri, duratran)
        dflxslen[indxtimetran] += 1e-3 * amplslen
    
    ## ellipsoidal variation
    ### density of the star in g/cm3
    densstar = 1.41 * massstar / radistar**3
    ### amplitude
    amplelli = retr_amplelli(peri, densstar, massstar, masscomp)
    ### signal
    dflxelli = -1e-3 * amplelli * np.cos(4. * np.pi * phas) 
    
    ## beaming
    amplbeam = retr_amplbeam(peri, massstar, masscomp)
    ### signal
    dflxbeam = 1e-3 * amplbeam * np.sin(2. * np.pi * phas)
    
    ## total relative flux
    rflxtotl = 1. + dflxslen + dflxelli + dflxbeam
    
    dictoutpbhol = dict()
    dictoutpbhol['amplslen'] = amplslen
    dictoutpbhol['duratran'] = duratran
    dictoutpbhol['rflxelli'] = dflxelli + 1.
    dictoutpbhol['rflxbeam'] = dflxbeam + 1.
    dictoutpbhol['rflxslen'] = dflxslen + 1.
    
    if booldiag and not np.isfinite(rflxtotl).all():
        print('peri')
        print(peri)
        print('masscomp')
        print(masscomp)
        print('massstar')
        print(massstar)
        print('masstotl')
        print(masstotl)
        print('smax')
        print(smax)
        print('cosi')
        print(cosi)
        print('rsma')
        print(rsma)
        print('duratran')
        print(duratran)
        raise Exception('')

    return rflxtotl, dictoutpbhol


def retr_rflxtranmodl( \
                      # times at which to evaluate the relative flux
                      time, \
                      # host
                      ## radius of the host star
                      radistar, \
                      # companions
                      ## orbital periods of the companions
                      pericomp, \
                      ## mid-transit epochs of the companions
                      epoccomp, \
                      
                      ## orbital inclination
                      inclcomp=None, \
                      ## cosine of the orbital inclination
                      cosicomp=None, \
                      
                      ## eccentricity of the orbit
                      eccecomp=0., \
                      ## sine of 
                      sinwcomp=0., \
                      
                      ## radii of the companions
                      radicomp=None, \
                      ## mass of the companions
                      masscomp=None, \
                      
                      ## type of the companion
                      typecomp='plan', \
                      
                      # type of limb-darkening
                      typelmdk='none', \
                      # lineaer limd-darkening coefficient 
                      coeflmdklinr=0.2, \
                      # quadratic limd-darkening coefficient 
                      coeflmdkquad=0.2, \
                      # mass of star
                      massstar=None, \

                      ## sum of stellar and companion radius
                      rsma=None, \
                      
                      # moons
                      ## radii
                      radimoon=None, \
                      ## orbital periods
                      perimoon=None, \
                      ## mid-transit epochs
                      epocmoon=None, \
                      ## sum of planetary and moon radius
                      rsmamoon=None, \
                      ## cosine of the orbital inclination
                      cosimoon=None, \
                      ## eccentricity of the orbit
                      eccemoon=None, \
                      ## sine of 
                      sinwmoon=None, \
                      # Boolean flag to model transits as trapezoid
                      booltrap=True, \
                      
                      # Boolean flag to return separate light curves for the companion and moon
                      boolcompmoon=False, \

                      # path to animate the integration in
                      pathanim=None, \

                      # string for the animation
                      strgextn='', \

                      # verbosity level
                      typeverb=1, \
                     ):
    '''
    Calculate the relative flux light curve of a star due to list of transiting companions and their orbiting moons.
    When limb-darkening and/or moons are turned on, the result is interpolated based on star-to-companion radius, companion-to-moon radius
    '''
    timeinit = timemodu.time()

    if isinstance(pericomp, list):
        pericomp = np.array(pericomp)

    if isinstance(epoccomp, list):
        epoccomp = np.array(epoccomp)

    if isinstance(radicomp, list):
        radicomp = np.array(radicomp)

    if isinstance(rsma, list):
        rsma = np.array(rsma)

    if isinstance(cosicomp, list):
        cosicomp = np.array(cosicomp)

    if isinstance(eccecomp, list):
        eccecomp = np.array(eccecomp)

    if isinstance(sinwcomp, list):
        sinwcomp = np.array(sinwcomp)
    
    if inclcomp is not None and cosicomp is not None:
        raise Exception('')
    
    if inclcomp is not None:
        cosicomp = np.cos(inclcomp * np.pi / 180.)

    numbcomp = pericomp.size
    if perimoon is not None:
        indxcomp = np.arange(numbcomp)
        numbmoon = np.empty(numbcomp, dtype=int)
        for j in indxcomp:
            numbmoon[j] = len(perimoon[j])
        indxmoon = [np.arange(numbmoon[j]) for j in indxcomp]
    
    if rsma is not None:
        typeinpt = 'rsma'
    else:
        typeinpt = 'perimass'
    
    # output dictionary
    dictoutp = dict()

    # type of calculation
    if typelmdk == 'none' and perimoon is None:
        typecalc = 'trap'
    else:
        typecalc = 'inte'

    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    numbtime = time.size
    
    dictfact = retr_factconv()
    
    if pathanim is not None and strgextn != '':
        strgextn = '_' + strgextn
    
    print('typeinpt')
    print(typeinpt)

    if typeinpt == 'rsma':
        smaxcomp = (radistar + radicomp / dictfact['rsre']) / rsma / dictfact['aurs']
    elif typeinpt == 'perimass':
        smaxcomp = retr_smaxkepl(pericomp, massstar)
        #masstotl = massstar + np.sum(masscomp)
        #smaxcomp = retr_smaxkepl(pericomp, masstotl)
        
        if perimoon is not None:
            smaxmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            for j in indxcomp:
                smaxmoon[j] = retr_smaxkepl(perimoon[j], masscomp[j] / dictfact['msme'])
                
                for jj in indxmoon[j]:
                    if smaxmoon[j] * dictfact['aurs'] * dictfact['rsre'] <= radicomp[j]:
                        print('smaxmoon[j] * dictfact[aurs] * dictfact[rsre]')
                        print(smaxmoon[j] * dictfact['aurs'] * dictfact['rsre'])
                        print('masscomp[j]')
                        print(masscomp[j])
                        print('perimoon[j]')
                        print(perimoon[j])
                        print('masscomp[j] / dictfact[msme]')
                        print(masscomp[j] / dictfact['msme'])
                        print('radicomp[j]')
                        print(radicomp[j])
                        raise Exception('')
   
    numbcomp = radicomp.size
    indxcomp = np.arange(numbcomp)
    
    if eccecomp is None:
        eccecomp = np.zeros(numbcomp)
    if sinwcomp is None:
        sinwcomp = np.zeros(numbcomp)
    
    radistareart = radistar * dictfact['rsre']
        
    rs2a = radistar / smaxcomp / dictfact['aurs']
    imfa = retr_imfa(cosicomp, rs2a, eccecomp, sinwcomp)
    sini = np.sqrt(1. - cosicomp**2)
    rrat = radicomp / radistareart
    dept = rrat**2 * 1e3
    
    if booltrap:
        durafull = retr_duratranfull(pericomp, rs2a, sini, rrat, imfa)
        duratotl = retr_duratrantotl(pericomp, rs2a, sini, rrat, imfa)
        duraineg = (duratotl - durafull) / 2.
        durafullhalf = durafull / 2.
    else:
        duratotl = retr_duratran(pericomp, rsma, cosicomp)
    duratotlhalf = duratotl / 2.

    ## Boolean flag that indicates whether there is any transit
    booltran = np.isfinite(duratotl)
    
    if typeverb > 1:
        print('time')
        summgene(time)
        
        print('radistar')
        print(radistar)
        
        print('epoccomp')
        print(epoccomp)
        print('pericomp')
        print(pericomp)
        print('radicomp')
        print(radicomp)
        print('cosicomp')
        print(cosicomp)
        print('radicomp')
        print(radicomp)
        print('smaxcomp')
        print(smaxcomp)
        print('smaxcomp * dictfact[aurs] [R_S]')
        print(smaxcomp * dictfact['aurs'])
        print('masscomp')
        print(masscomp)
        print('rsma')
        print(rsma)
        print('imfa')
        print(imfa)
        print('rrat')
        print(rrat)
        print('rs2a')
        print(rs2a)
        if typecalc != 'inte':
            print('duratotl')
            print(duratotl)
            if booltrap:
                print('durafull')
                print(durafull)
        print('booltran')
        print(booltran)
        print('indxcomp')
        print(indxcomp)
        if perimoon is not None:
            print('perimoon')
            print(perimoon)
            print('radimoon')
            print(radimoon)
            print('smaxmoon * dictfact[aurs] * dictfact[rsre] [R_E]')
            for smaxmoontemp in smaxmoon:
                print(smaxmoontemp * dictfact['aurs'] * dictfact['rsre'])
    
    if typecalc == 'inte':
        masstotl = np.sum(masscomp) / dictfact['msme'] + massstar
        
        numbtime = time.size
        indxtime = np.arange(numbtime)
        
        # grid
        numbside = 1000
        arry = np.linspace(-1., 1., numbside) * radistareart
        xposgrid, yposgrid = np.meshgrid(arry, arry)
        
        # distance to the center of the star
        diststar = np.sqrt(xposgrid**2 + yposgrid**2)
        
        # grid of stellar brightness
        brgt = np.zeros_like(xposgrid)
        boolstar = diststar < radistareart
        indxgridstar = np.where(boolstar)
        
        brgt[indxgridstar] = retr_brgtlmdk(1. - diststar[indxgridstar] / radistareart, coeflmdklinr, coeflmdkquad, typelmdk=typelmdk)
        
        maxmbrgt = np.amax(brgt)
        
        rflxtranmodl = np.sum(brgt) * np.ones(numbtime)
        
        boolnocccomp = [[] for j in indxcomp]
        xposcomp = [[] for j in indxcomp]
        yposcomp = [[] for j in indxcomp]
        phascomp = [[] for j in indxcomp]
        if perimoon is not None:
            boolnoccmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            xposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            yposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
        
        for j in indxcomp:
            #velocomp = 0.017 * np.sqrt(masstotl / smaxcomp[j]) # [AU/day]
            phascomp[j] = ((time - epoccomp[j]) / pericomp[j]) % 1.
            xposcomp[j] = smaxcomp[j] * np.sin(2. * np.pi * phascomp[j]) * dictfact['aurs'] * dictfact['rsre']
            yposcomp[j] = np.cos(inclcomp[j] * np.pi / 180.) * smaxcomp[j]

            if perimoon is not None:
                for jj in indxmoon[j]:
                    #velomoon = 2.9e-5 * np.sqrt(masscomp[j] / smaxcomp[j]) # [AU/day]
                    xposmoon[j][jj] = xposcomp[j] + smaxmoon[j][jj] * np.cos(2. * np.pi * (time - epocmoon[j][jj]) / perimoon[j][jj]) * dictfact['aurs'] * dictfact['rsre']
                    yposmoon[j][jj] = yposcomp[j] + smaxmoon[j][jj] * np.sin(2. * np.pi * (time - epocmoon[j][jj]) / perimoon[j][jj]) * dictfact['aurs'] * dictfact['rsre']
                    
        if boolcompmoon:
            numbiter = 2
        else:
            numbiter = 1

        for a in range(numbiter):
            
            if pathanim is not None:
                pathtime = [[] for t in indxtime]

            #print('a')
            #print(a)

            if a == 0:
                strgcompmoon = ''
            else:
                strgcompmoon = '_onlycomp'
                rflxtranmodlcomp = np.sum(brgt) * np.ones(numbtime)
            
            if pathanim is not None:
                cmnd = 'convert -delay 5 -density 200'
                pathgiff = pathanim + 'anim%s%s.gif' % (strgextn, strgcompmoon)

            for t in indxtime:
                
                #print('t')
                #print(t)
                #print('radistareart')
                #print(radistareart)
                
                boolnocc = np.copy(boolstar)
                booleval = False
                for j in indxcomp:
                    
                    #print('xposcomp[j][t]')
                    #print(xposcomp[j][t])
                    #print('radicomp[j]')
                    #print(radicomp[j])
                    
                    if phascomp[j][t] > 0.25 and phascomp[j][t] < 0.75:
                        continue

                    if np.sqrt(xposcomp[j][t]**2 + yposcomp[j]**2) < radistareart + radicomp[j]:
                    
                        #print('t: %d, j: %d, evaluating companion' % (t, j))
                        
                        booleval = True

                        xposgridcomp = xposgrid - xposcomp[j][t]
                        yposgridcomp = yposgrid - yposcomp[j]
                        
                        if typecomp == 'plan' or typecomp == 'plandiskedgehori' or typecomp == 'plandiskedgevert' or typecomp == 'plandiskface':
                            distcomp = np.sqrt(xposgridcomp**2 + yposgridcomp**2)
                            boolnocccomp[j] = distcomp > radicomp[j]
                        
                        if typecomp == 'plandiskedgehori':
                            booldisk = (xposgridcomp / 1.75 / radicomp[j])**2 + (yposgridcomp / 0.2 / radicomp[j])**2 > 1.
                            boolnocccomp[j] = boolnocccomp[j] & booldisk
                           
                        if typecomp == 'plandiskedgevert':
                            booldisk = (yposgridcomp / 1.75 / radicomp[j])**2 + (xposgridcomp / 0.2 / radicomp[j])**2 > 1.
                            boolnocccomp[j] = boolnocccomp[j] & booldisk
                           
                        if typecomp == 'plandiskface':
                            boolnocccomp[j] = boolnocccomp[j] & ((distcomp > 1.5 * radicomp[j]) & (distcomp < 1.75 * radicomp[j]))
                        
                        #print('np.sum(boolnocccomp[j])')
                        #print(np.sum(boolnocccomp[j]))
                        
                        boolnocc = boolnocc & boolnocccomp[j]
                    
                if perimoon is not None and a == 0:
                    for j in indxcomp:
                        if phascomp[j][t] > 0.25 and phascomp[j][t] < 0.75:
                            continue

                        for jj in indxmoon[j]:
                            
                            #print('xposmoon[j][jj][t]')
                            #print(xposmoon[j][jj][t])
                            #print('radimoon[j][jj]')
                            #print(radimoon[j][jj])
                            if np.sqrt(xposmoon[j][jj][t]**2 + yposmoon[j][jj][t]**2) < radistareart + radimoon[j][jj]:
                                
                                booleval = True
                                
                                #print('t: %d, j: %d, jj: %d, evaluating moon' % (t, j, jj))
                                xposgridmoon = xposgrid - xposmoon[j][jj][t]
                                yposgridmoon = yposgrid - yposmoon[j][jj][t]
                                
                                distmoon = np.sqrt(xposgridmoon**2 + yposgridmoon**2)
                                boolnoccmoon[j][jj] = distmoon > radimoon[j][jj]
                                
                                #print('np.sum(boolnoccmoon[j][jj])')
                                #print(np.sum(boolnoccmoon[j][jj]))
                                
                                boolnocc = boolnocc & boolnoccmoon[j][jj]
                
                if booleval:
                    
                    #print('np.sum(boolnocc)')
                    #print(np.sum(boolnocc))
                    #print('')
                    
                    indxgridnocc = np.where(boolnocc)
                    

                    if a == 0:
                        rflxtranmodl[t] = np.sum(brgt[indxgridnocc])
                    else:
                        rflxtranmodlcomp[t] = np.sum(brgt[indxgridnocc])
                        
                    if pathanim is not None and not os.path.exists(pathgiff):
                        pathtime[t] = pathanim + 'imag%s%s_%04d.pdf' % (strgextn, strgcompmoon, t)
                        cmnd+= ' %s' % pathtime[t]
                        figr, axis = plt.subplots(figsize=(4, 3))
                        brgttemp = np.zeros_like(brgt)
                        brgttemp[boolnocc] = brgt[boolnocc]
                        imag = axis.imshow(brgttemp, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=maxmbrgt)
                        axis.axis('off')
                        print('Writing to %s...' % pathtime[t])
                        plt.savefig(pathtime[t], dpi=200)
                        plt.close()
                
                #print('')
            
            if pathanim is not None:
                # make animation
                cmnd += ' %s' % pathgiff
                print('Writing to %s...' % pathgiff)
                os.system(cmnd)
                
                # delete images
                cmnd = 'rm'
                for t in indxtime:
                    if len(pathtime[t]) > 0:
                        cmnd += ' %s' % pathtime[t]
                os.system(cmnd)

        rflxtranmodl /= np.amax(rflxtranmodl)
        dictoutp['rflx'] = rflxtranmodl
        
        if boolcompmoon:
            rflxtranmodlcomp /= np.amax(rflxtranmodlcomp)
            dictoutp['rflxcomp'] = rflxtranmodlcomp
            rflxtranmodlmoon = 1. + rflxtranmodl - rflxtranmodlcomp
            dictoutp['rflxmoon'] = rflxtranmodlmoon

    else:
        
        rflxtranmodl = np.ones_like(time)
    
        for j in indxcomp:
            if booltran[j]:
                    
                minmindxtran = int(np.floor((minmtime - epoccomp[j]) / pericomp[j]))
                maxmindxtran = int(np.ceil((maxmtime - epoccomp[j]) / pericomp[j]))
                indxtranthis = np.arange(minmindxtran, maxmindxtran + 1)
                
                if typeverb > 1:
                    print('minmindxtran')
                    print(minmindxtran)
                    print('maxmindxtran')
                    print(maxmindxtran)
                    print('indxtranthis')
                    print(indxtranthis)

                for n in indxtranthis:
                    timetran = epoccomp[j] + pericomp[j] * n
                    timeshft = time - timetran
                    timeshftnega = -timeshft
                    timeshftabso = abs(timeshft)
                    
                    indxtimetotl = np.where(timeshftabso < duratotlhalf[j] / 24.)[0]
                    if booltrap:
                        indxtimefull = indxtimetotl[np.where(timeshftabso[indxtimetotl] < durafullhalf[j] / 24.)]
                        indxtimeinre = indxtimetotl[np.where((timeshftnega[indxtimetotl] < duratotlhalf[j] / 24.) & \
                                                                                            (timeshftnega[indxtimetotl] > durafullhalf[j] / 24.))]
                        indxtimeegre = indxtimetotl[np.where((timeshft[indxtimetotl] < duratotlhalf[j] / 24.) & (timeshft[indxtimetotl] > durafullhalf[j] / 24.))]
                    
                        rflxtranmodl[indxtimeinre] += 1e-3 * dept[j] * ((timeshftnega[indxtimeinre] - duratotlhalf[j] / 24.) / duraineg[j] / 24.)
                        rflxtranmodl[indxtimeegre] += 1e-3 * dept[j] * ((timeshft[indxtimeegre] - duratotlhalf[j] / 24.) / duraineg[j])
                        rflxtranmodl[indxtimefull] -= 1e-3 * dept[j]
                    else:
                        rflxtranmodl[indxtimetotl] -= 1e-3 * dept[j]
                    
                    if typeverb > 1:
                        print('n')
                        print(n)
                        print('timetran')
                        summgene(timetran)
                        print('timeshft[indxtimetotl]')
                        summgene(timeshft[indxtimetotl])
                        print('timeshftabso[indxtimetotl]')
                        summgene(timeshftabso[indxtimetotl])
                        print('timeshftnega[indxtimetotl]')
                        summgene(timeshftnega[indxtimetotl])
                        print('indxtimetotl')
                        summgene(indxtimetotl)
                        if booltrap:
                            print('duratotlhalf[j]')
                            print(duratotlhalf[j])
                            print('durafullhalf[j]')
                            print(durafullhalf[j])
                            print('indxtimefull')
                            summgene(indxtimefull)
                            print('indxtimeinre')
                            summgene(indxtimeinre)
                            print('indxtimeegre')
                            summgene(indxtimeegre)
        dictoutp['rflx'] = rflxtranmodl
    
    timetotl = timemodu.time() - timeinit
    timeredu = timetotl / numbtime
    print('retr_rflxtranmodl ran in %.3g seconds and %g ns per time sample.' % (timetotl, timeredu * 1e9))

    return dictoutp


def retr_radifrommass( \
                      # list of planet masses in units of Earth mass
                      listmassplan, \
                      # type of radius-mass model
                      strgtype='mine', \
                      ):
    '''
    Estimate planetary radii from samples of masses
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
                      listradiplan, \
                      # type of radius-mass model
                      strgtype='mine', \
                      ):
    '''
    Estimate planetary mass from samples of radii.
    '''
    
    if len(listradiplan) == 0:
        raise Exception('')


    if strgtype == 'mine':
        # get interpolation data
        path = os.environ['EPHESUS_DATA_PATH'] + '/data/massfromradi.csv'
        print('Reading from %s...' % path)
        arry = np.loadtxt(path)
        
        # interpolate masses
        listmass = np.interp(listradiplan, arry[:, 0], arry[:, 1])
        liststdvmass = np.interp(listradiplan, arry[:, 0], arry[:, 2])
    
    if strgtype == 'wolf2016':
        # (Wolgang+2016 Table 1)
        listmass = (2.7 * (listradiplan * 11.2)**1.3 + np.random.randn(listradiplan.size) * 1.9) / 317.907
        listmass = np.maximum(listmass, np.zeros_like(listmass))
    
    return listmass


def retr_tmptplandayynigh(tmptirra, epsi):
    '''
    Estimate the dayside and nightside temperatures [K] of a planet given its irradiation temperature in K and recirculation efficiency.
    '''
    
    tmptdayy = tmptirra * (2. / 3. - 5. / 12. * epsi)**.25
    tmptnigh = tmptirra * (epsi / 4.)**.25
    
    return tmptdayy, tmptnigh


def retr_esmm(tmptplanequi, tmptstar, radiplan, radistar, kmag):
    
    tmptplanirra = tmptplanequi
    tmptplandayy, tmptplannigh = retr_tmptplandayynigh(tmptplanirra, 0.1)
    esmm = 1.1e3 * tdpy.util.retr_specbbod(tmptplandayy, 7.5) / tdpy.util.retr_specbbod(tmptstar, 7.5) * (radiplan / radistar)*2 * 10**(-kmag / 5.)

    return esmm


def retr_tsmm(radiplan, tmptplan, massplan, radistar, jmag):
    
    tsmm = 1.53 / 1.2 * radiplan**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)
    
    return tsmm


def retr_scalheig(tmptplan, massplan, radiplan):
    
    # tied to Jupier's scale height for H/He at 110 K   
    scalheig = 27. * (tmptplan / 160.) / (massplan / radiplan**2) / 71398. # [R_J]

    return scalheig


def retr_rflxfromdmag(dmag, stdvdmag=None):
    
    rflx = 10**(-dmag / 2.5)

    if stdvdmag is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
    
    return rflx, stdvrflx


def retr_lcur_mock(numbplan=100, numbnois=100, numbtime=100, dept=1., nois=1e-3, numbbinsphas=1000, pathplot=None, boollabltime=False, boolflbn=False):
    
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
    minmdept = 0.1 # [ppt]
    maxmdept = 10. # [ppt]
    minmepoc = np.amin(time)
    maxmepoc = np.amax(time)
    minmdura = 3. # [hour]
    maxmdura = 4. # [hour]

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
            indxphastran = np.where(abs(phas - n) < duraplan[k] / periplan[k] / 24.)[0]
            fluxplan[k, indxphastran] -= 1e-3 * deptplan[k]
    
    # place the signal data
    flux[:numbplan, :] = fluxplan

    # label the data
    if boollabltime:
        outp = np.zeros((numbdata, numbtime))
        outp[np.where(flux == 1e-3 * dept[0])] = 1.
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


def retr_dictexar(strgexar=None):
    
    # get NASA Exoplanet Archive data
    path = os.environ['EPHESUS_DATA_PATH'] + '/data/PSCompPars_2021.10.02_11.57.17.csv'
    print('Reading %s...' % path)
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
        dictexar['facidisc'] = objtexar['disc_facility'][indx].values
        
        dictexar['inso'] = objtexar['pl_insol'][indx].values
        dictexar['peri'] = objtexar['pl_orbper'][indx].values # [days]
        dictexar['smax'] = objtexar['pl_orbsmax'][indx].values # [AU]
        dictexar['epoc'] = objtexar['pl_tranmid'][indx].values # [BJD]
        dictexar['cosi'] = np.cos(objtexar['pl_orbincl'][indx].values / 180. * np.pi)
        dictexar['duratran'] = objtexar['pl_trandur'][indx].values # [hour]
        dictexar['dept'] = 10. * objtexar['pl_trandep'][indx].values # ppt
        
        # to be deleted
        #dictexar['boolfpos'] = np.zeros(numbplanexar, dtype=bool)
        
        dictexar['booltran'] = objtexar['tran_flag'][indx].values
        
        # mass provenance
        dictexar['strgprovmass'] = objtexar['pl_bmassprov'][indx].values
        
        dictexar['booltran'] = dictexar['booltran'].astype(bool)

        # radius reference
        dictexar['strgrefrradiplan'] = objtexar['pl_rade_reflink'][indx].values
        for a in range(dictexar['strgrefrradiplan'].size):
            if isinstance(dictexar['strgrefrradiplan'][a], float) and not np.isfinite(dictexar['strgrefrradiplan'][a]):
                dictexar['strgrefrradiplan'][a] = ''
        
        # mass reference
        dictexar['strgrefrmassplan'] = objtexar['pl_bmasse_reflink'][indx].values
        for a in range(dictexar['strgrefrmassplan'].size):
            if isinstance(dictexar['strgrefrmassplan'][a], float) and not np.isfinite(dictexar['strgrefrmassplan'][a]):
                dictexar['strgrefrmassplan'][a] = ''

        for strg in ['radistar', 'massstar', 'tmptstar', 'loggstar', 'radicomp', 'masscomp', 'tmptplan', 'tagestar', \
                     'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'tmagsyst', 'metastar', 'distsyst', 'lumistar']:
            strgexar = None
            if strg.endswith('syst'):
                strgexar = 'sy_'
                if strg[:-4].endswith('mag'):
                    strgexar += '%smag' % strg[0]
                if strg[:-4] == 'dist':
                    strgexar += 'dist'
            if strg.endswith('star'):
                strgexar = 'st_'
                if strg[:-4] == 'logg':
                    strgexar += 'logg'
                if strg[:-4] == 'tage':
                    strgexar += 'age'
                if strg[:-4] == 'meta':
                    strgexar += 'met'
                if strg[:-4] == 'radi':
                    strgexar += 'rad'
                if strg[:-4] == 'mass':
                    strgexar += 'mass'
                if strg[:-4] == 'tmpt':
                    strgexar += 'teff'
                if strg[:-4] == 'lumi':
                    strgexar += 'lum'
            if strg.endswith('plan'):
                strgexar = 'pl_'
                if strg[:-4].endswith('mag'):
                    strgexar += '%smag' % strg[0]
                if strg[:-4] == 'tmpt':
                    strgexar += 'eqt'
                if strg[:-4] == 'radi':
                    strgexar += 'rade'
                if strg[:-4] == 'mass':
                    strgexar += 'bmasse'
            if strgexar is None:
                raise Exception('')
            dictexar[strg] = objtexar[strgexar][indx].values
            dictexar['stdv%s' % strg] = (objtexar['%serr1' % strgexar][indx].values - objtexar['%serr2' % strgexar][indx].values) / 2.
       
        dictexar['vesc'] = retr_vesc(dictexar['masscomp'], dictexar['radicomp'])
        dictexar['masstotl'] = dictexar['massstar'] + dictexar['masscomp'] / dictfact['msme']
        
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
        dictexar['dcyc'] = dictexar['duratran'] / dictexar['peri'] / 24.
        
        # galactic longitude
        dictexar['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dictexar['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dictexar['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dictexar['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        dictexar['rrat'] = dictexar['radicomp'] / dictexar['radistar'] / dictfact['rsre']
        

        # calculate TSM and ESM
        calc_tsmmesmm(dictexar)
        
    return dictexar


# physics

def retr_vesc(massplan, radiplan):
    
    vesc = 11.2 * np.sqrt(massplan / radiplan) # km/s

    return vesc


def retr_rs2a(rsma, rrat):
    
    rs2a = rsma / (1. + rrat)
    
    return rs2a


def retr_rsma(peri, dura, cosi):
    
    rsma = np.sqrt(np.sin(dura * np.pi / peri / 24.)**2 + cosi**2)
    
    return rsma


def retr_duratranfull(peri, rs2a, sini, rrat, imfa):
    
    durafull = 24. * peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. - rrat)**2 - imfa**2)) # [hours]

    return durafull 


def retr_duratrantotl(peri, rs2a, sini, rrat, imfa):
    
    duratotl = 24. * peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. + rrat)**2 - imfa**2)) # [hours]
    
    return duratotl

    
def retr_imfa(cosi, rs2a, ecce, sinw):
    
    imfa = cosi / rs2a * (1. - ecce)**2 / (1. + ecce * sinw)

    return imfa


def retr_amplbeam(peri, massstar, masscomp):
    
    '''Calculates the beaming amplitude'''
    
    amplbeam = 2.8 * peri**(-1. / 3.) * (massstar + masscomp)**(-2. / 3.) * masscomp # [ppt]
    
    return amplbeam


def retr_amplelli(peri, densstar, massstar, masscomp):
    
    '''Calculates the ellipsoidal variation amplitude'''
    
    amplelli = 18.9 * peri**(-2.) / densstar * (1. / (1. + massstar / masscomp)) # [ppt]
    
    return amplelli


def retr_masscomp(amplslen, peri):
    
    print('temp: this mass calculation is an approximation.')
    masscomp = 1e-3 * amplslen / 7.15e-5 / gdat.radistar**(-2.) / peri**(2. / 3.) / (gdat.massstar)**(1. / 3.)
    
    return masscomp


def retr_amplslen(peri, radistar, masscomp, massstar):
    
    """
    Calculate the self-lensing amplitude.

    Arguments
        peri: orbital period [days]
        radistar: radius of the star [Solar radius]
        masscomp: mass of the companion [Solar mass]
        massstar: mass of the star [Solar mass]

    Returns
        amplslen: the fractional amplitude of the self-lensing
    """
    
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.) * 1e3 # [ppt]

    return amplslen


def retr_smaxkepl(peri, masstotl):
    
    """
    Get the semi-major axis of a Keplerian orbit (in AU) from the orbital period (in days) and total mass (in Solar masses).

    Arguments
        peri: orbital period [days]
        masstotl: total mass of the system [Solar Masses]
    Returns
        smax: the semi-major axis of a Keplerian orbit [AU]
    """
    
    smax = (7.496e-6 * masstotl * peri**2)**(1. / 3.) # [AU]
    
    return smax


def retr_perikepl(smax, masstotl):
    
    """
    Get the period of a Keplerian orbit (in days) from the semi-major axis (in AU) and total mass (in Solar masses).

    Arguments
        smax: the semi-major axis of a Keplerian orbit [AU]
        masstotl: total mass of the system [Solar Masses]
    Returns
        peri: orbital period [days]
    """
    
    peri = np.sqrt(smax**3 / 7.496e-6 / masstotl)
    
    return peri


def retr_duratran(peri, rsma, cosi):
    """
    Return the transit duration in the unit of the input orbital period (peri).

    Arguments
        peri: orbital period
        rsma: the sum of radii of the two bodies divided by the semi-major axis
        cosi: cosine of the inclination
    """    
    
    dura = 24. * peri / np.pi * np.arcsin(np.sqrt(rsma**2 - cosi**2)) # [hours]
    
    return dura


def retr_radiroch(radistar, densstar, denscomp):
    """
    Return the Roche limit

    Arguments
        radistar: radius of the primary star
        densstar: density of the primary star
        denscomp: density of the companion
    """    
    radiroch = radistar * (2. * densstar / denscomp)**(1. / 3.)
    
    return radiroch


def retr_radihill(smax, masscomp, massstar):
    """
    Return the Hill radius of a companion

    Arguments
        peri: orbital period
        rsma: the sum of radii of the two bodies divided by the semi-major axis
        cosi: cosine of the inclination
    """    
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
    
        radiplan = [1.6, 2.1, 2.7, 3.1]
        rsma = [0.0895, 0.0647, 0.0375, 0.03043]
        epoc = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
        peri = [3.8, 6.2, 14.2, 19.6]
        cosi = [0., 0., 0., 0.]
        
        if a == 1:
            radiplan += [2.0]
            rsma += [0.88 / (215. * 0.1758)]
            epoc += [2458793.2786]
            peri += [29.54115]
            cosi += [0.]
        
        for typevisu in listtypevisu:
            
            if a == 0:
                continue
    
            pexo.main.plot_orbt( \
                                path, \
                                radiplan, \
                                rsma, \
                                epoc, \
                                peri, \
                                cosi, \
                                typevisu, \
                                radistar=radistar, \
                                boolsingside=boolsingside, \
                                boolanim=boolanim, \
                                #typefileplot='png', \
                               )
        



