import numpy as np

import time as timemodu

from numba import jit, prange
import sys, os, h5py, fnmatch
import pickle

import emcee

import astropy as ap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.io import fits

import multiprocessing

import mr_forecast

import transitleastsquares

import astropy

import scipy as sp
import scipy.interpolate

# own modules
import tdpy.util
from tdpy.util import summgene

import lygos.main

import astroquery
import astroquery.mast

import matplotlib.pyplot as plt


def retr_lcurflar(meantime, indxtime, listtimeflar, listamplflar, listscalrise, listscalfall):
    
    if meantime.size == 0:
        raise Exception('')
    
    lcur = np.zeros_like(meantime)
    numbflar = len(listtimeflar)
    for k in np.arange(numbflar):
        lcur += retr_lcurflarsing(meantime, listtimeflar[k], listamplflar[k], listscalrise[k], listscalfall[k])

    return lcur


def retr_lcurflarsing(meantime, timeflar, amplflar, scalrise, scalfall):
    
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


def anim_tmptdete(timefull, lcurfull, meantimetmpt, lcurtmpt, pathimag, listindxtimeposimaxm, corrprod, corr, strgextn='', colr=None):
    
    numbtimefull = timefull.size
    numbtimekern = lcurtmpt.size
    numbtimefullruns = numbtimefull - numbtimekern
    indxtimefullruns = np.arange(numbtimefullruns)
    
    print('anim_tmptdete()')
    print('timefull')
    summgene(timefull)
    print('lcurfull')
    summgene(lcurfull)
    print('meantimetmpt')
    summgene(meantimetmpt)
    print('lcurtmpt')
    summgene(lcurtmpt)
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
    proc_axiscorr(timefull, lcurfull, axis[0], listindxtimeposimaxm, corr)
    
    # plot zoomed-in light curve
    print('plot_tmptdete()')
    print('indxtimekern')
    summgene(indxtimekern)
    minmindx = max(0, tt - int(numbtimekern / 4))
    maxmindx = min(numbtimefullruns - 1, tt + int(5. * numbtimekern / 4))
    indxtime = np.arange(minmindx, maxmindx + 1)
    print('indxtime')
    summgene(indxtime)
    proc_axiscorr(timefull, lcurfull, axis[1], listindxtimeposimaxm, corr, indxtime=indxtime)
    
    print('tt')
    print(tt)
    print('meantimetmpt + tt * difftime')
    summgene(meantimetmpt + tt * difftime)
    # plot template
    axis[2].plot(timefull[0] + meantimetmpt + tt * difftime, lcurtmpt, color='k', marker='D')
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
    

def proc_axiscorr(time, lcur, axis, listindxtimeposimaxm, corr, indxtime=None, colr='k'):
    
    if indxtime is None:
        indxtimetemp = np.arange(time.size)
    else:
        indxtimetemp = indxtime
    axis.plot(time[indxtimetemp], lcur[indxtimetemp], ls='', marker='o', color=colr, rasterized=True, ms=0.5)
    maxmydat = axis.get_ylim()[1]
    for kk in range(len(listindxtimeposimaxm)):
        if listindxtimeposimaxm[kk] in indxtimetemp:
            axis.plot(time[listindxtimeposimaxm[kk]], maxmydat, marker='D', color=colr)
            #axis.text(time[listindxtimeposimaxm[kk]], maxmydat, '%.3g' % corr[listindxtimeposimaxm[kk]], color='k', va='center', \
            #                                                ha='center', size=2, rasterized=False)
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Relative flux')
    

def find_flar(time, lcur, verbtype=1, strgextn='', numbkern=3, minmscalfalltmpt=None, maxmscalfalltmpt=None, \
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
    
    if verbtype > 1:
        print('lcurtmpt')
        summgene(lcurtmpt)
    
    indxscalfall = np.arange(numbkern)
    listscalfalltmpt = np.linspace(minmscalfalltmpt, maxmscalfalltmpt, numbkern)
    
    listcorr = []
    listlcurtmpt = [[] for k in indxscalfall]
    meantimetmpt = [[] for k in indxscalfall]
    for k in indxscalfall:
        numbtimekern = 3 * int(listscalfalltmpt[k] / difftime)
        meantimetmpt[k] = np.arange(numbtimekern) * difftime
        listlcurtmpt[k] = retr_lcurflarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
        if not np.isfinite(listlcurtmpt[k]).all():
            raise Exception('')
        
    corr, listindxtimeposimaxm, timefull, lcurfull = corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, thrs=thrs, boolanim=boolanim, boolplot=boolplot, \
                                                                                            verbtype=verbtype, strgextn=strgextn, pathimag=pathimag)

    return corr, listindxtimeposimaxm, meantimetmpt, timefull, lcurfull


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def corr_arryprod(lcurtemp, lcurtmpt, numbkern):
    
    # correlate
    corrprod = [[] for k in range(numbkern)]
    for k in prange(numbkern):
        corrprod[k] = lcurtmpt[k] * lcurtemp[k]
    
    return corrprod


#@jit(parallel=True)
def corr_copy(indxtimefullruns, lcurstan, indxtimekern, numbkern):
    
    print('corr_copy()')
    # make windowed and shifted copies of the light curve
    listlcurtemp = [[] for k in range(numbkern)]
    for k in prange(numbkern):
        numbtimefullruns = indxtimefullruns[k].size
        numbtimekern = indxtimekern[k].size
        listlcurtemp[k] = np.empty((numbtimefullruns, numbtimekern))
        print('k')
        print(k)
        print('numbtimefullruns')
        print(numbtimefullruns)
        print('numbtimekern')
        print(numbtimekern)

        for t in prange(numbtimefullruns):
            listlcurtemp[k][t, :] = lcurstan[indxtimefullruns[k][t]+indxtimekern[k]]
        print('listlcurtemp[k]')
        summgene(listlcurtemp[k])
    print('')
    return listlcurtemp


def corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, verbtype=2, thrs=None, strgextn='', pathimag=None, boolplot=True, boolanim=False):
    
    if verbtype > 1:
        timeinit = timemodu.time()
    
    if lcur.ndim > 1:
        raise Exception('')
    
    for lcurtmpt in listlcurtmpt:
        if not np.isfinite(lcurtmpt).all():
            raise Exception('')

    if not np.isfinite(lcur).all():
        raise Exception('')

    numbtime = lcur.size
    
    # construct the full grid
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    timefull = np.linspace(minmtime, maxmtime, int(round((maxmtime - minmtime) / np.amin(time[1:] - time[:-1]))))
    print('timefull')
    summgene(timefull)
    numbtimefull = timefull.size
    lcurfull = np.zeros_like(timefull)
    indx = np.digitize(time, timefull) - 1
    lcurfull[indx] = lcur
    print('lcurfull')
    summgene(lcurfull)
    
    numbkern = len(listlcurtmpt)
    indxkern = np.arange(numbkern)
    numbtimekern = np.empty(numbkern, dtype=int)
    numbtimefullruns = np.empty(numbkern, dtype=int)
    corr = [[] for k in indxkern]
    corrprod = [[] for k in indxkern]
    indxtimekern = [[] for k in indxkern]
    indxtimefullruns = [[] for k in indxkern]
    listindxtimeposimaxm = [[] for k in indxkern]
    for k in indxkern:
        numbtimekern[k] = listlcurtmpt[k].size
        listlcurtmpt[k] -= np.mean(listlcurtmpt[k])
        listlcurtmpt[k] /= np.std(listlcurtmpt[k])
        indxtimekern[k] = np.arange(numbtimekern[k])
        numbtimefullruns[k] = numbtimefull - numbtimekern[k]
        indxtimefullruns[k] = np.arange(numbtimefullruns[k])
    
    # standardize
    lcurstan = lcurfull / np.std(lcur - np.mean(lcur))
    
    if verbtype > 1:
        print('Delta T (corr_tmpt, initial): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    listlcurtemp = corr_copy(indxtimefullruns, lcurstan, indxtimekern, numbkern)
    if verbtype > 1:
        print('Delta T (corr_tmpt, copy): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    corrprod = corr_arryprod(listlcurtemp, listlcurtmpt, numbkern)
    if verbtype > 1:
        print('Delta T (corr_tmpt, corr_prod): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    boolthrsauto = thrs is None
    
    for k in indxkern:
        # find the total correlation (along the time delay axis)
        corr[k] = np.sum(corrprod[k], 1)
        print('k')
        print(k)
        print('listlcurtemp[k]')
        summgene(listlcurtemp[k])
        print('corrprod[k]')
        summgene(corrprod[k])
        print('corr[k]')
        summgene(corr[k])
        if boolthrsauto:
            perclowrcorr = np.percentile(corr[k],  1.)
            percupprcorr = np.percentile(corr[k], 99.)
            indx = np.where((corr[k] < percupprcorr) & (corr[k] > perclowrcorr))[0]
            medicorr = np.median(corr[k])
            thrs = np.std(corr[k][indx]) * 7. + medicorr

        if not np.isfinite(corr[k]).all():
            raise Exception('')

        if verbtype > 1:
            print('corr[k]')
            summgene(corr[k])
    
        # determine the threshold on the maximum correlation
        if verbtype > 1:
            print('thrs')
            print(thrs)

        # find triggers
        listindxtimeposi = np.where(corr[k] > thrs)[0]
        if verbtype > 1:
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
        
        if verbtype > 1:
            print('listindxtimeposiptch')
            summgene(listindxtimeposiptch)

        listindxtimeposimaxm[k] = np.empty(len(listindxtimeposiptch), dtype=int)
        for kk in range(len(listindxtimeposiptch)):
            indxtemp = np.argmax(corr[k][listindxtimeposiptch[kk]])
            listindxtimeposimaxm[k][kk] = listindxtimeposiptch[kk][indxtemp]
        
        if verbtype > 1:
            print('listindxtimeposimaxm[k]')
            summgene(listindxtimeposimaxm[k])
        
        if boolplot or boolanim:
            strganim = strgextn + '_kn%02d' % k
        
        if boolplot:
            numbdeteplot = min(len(listindxtimeposimaxm[k]), 10)
            indxtimefullruns = np.arange(numbtimefullruns[k])
            numbfram = 3 + numbdeteplot
            figr, axis = plt.subplots(numbfram, 1, figsize=(8, numbfram*3))
            proc_axiscorr(timefull, lcurfull, axis[0], listindxtimeposimaxm[k], corr[k])
            
            axis[1].plot(time[indxtimefullruns], corr[k], color='m', ls='', marker='o', ms=1, rasterized=True)
            axis[1].plot(time[indxtimefullruns[listindxtimeposi]], corr[k][listindxtimeposi], color='r', ls='', marker='o', ms=1, rasterized=True)
            axis[1].set_ylabel('C')
            
            axis[2].axvline(thrs, alpha=0.4, color='r', lw=3, ls='-.')
            if boolthrsauto:
                axis[2].axvline(percupprcorr, alpha=0.4, lw=3, ls='--')
                axis[2].axvline(perclowrcorr, alpha=0.4, lw=3, ls='--')
                axis[2].axvline(medicorr, alpha=0.4, lw=3, ls='-')
            axis[2].set_ylabel(r'N')
            axis[2].set_xlabel('C')
            axis[2].set_yscale('log')
            axis[2].hist(corr[k], color='black', bins=100)
            
            for i in range(numbdeteplot):
                indxtimeplot = indxtimekern[k] + listindxtimeposimaxm[k][i]
                proc_axiscorr(timefull, lcurfull, axis[3+i], listindxtimeposimaxm[k], corr[k], indxtime=indxtimeplot)
            
            path = pathimag + 'lcurflardepr%s.pdf' % (strganim)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        if boolanim:
            path = pathimag + 'lcur%s.gif' % strganim
            if not os.path.exists(path):
                anim_tmptdete(timefull, lcurfull, meantimetmpt[k], listlcurtmpt[k], pathimag, \
                                                            listindxtimeposimaxm[k], corrprod[k], corr[k], strgextn=strganim)
            else:
                print('Skipping animation for kernel %d...' % k)
    if verbtype > 1:
        print('Delta T (corr_tmpt, rest): %g' % (timemodu.time() - timeinit))

    return corr, listindxtimeposimaxm, timefull, lcurfull


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

def retr_indxtimetran(time, epoc, peri, duramask, booloutt=False):
    
    if not np.isfinite(time).all():
        raise Exception('')
    listindxtimetran = []
    
    if np.isfinite(peri):
        intgminm = np.floor((np.amin(time) - epoc - duramask / 2.) / peri)
        intgmaxm = np.ceil((np.amax(time) - epoc - duramask / 2.) / peri)
        arry = np.arange(intgminm, intgmaxm + 1)
    else:
        arry = np.arange(1)

    for n in arry:
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
    

def retr_timeedge(time, lcur, durabrek, \
                  # Boolean flag to add breaks at discontinuties
                  booladdddiscbdtr=False, \
                 ):

    difftime = time[1:] - time[:-1]
    indxtimebrek = np.where(difftime > durabrek)[0]
    
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
        timeedge.append((time[k] + time[k+1]) / 2.)
    timeedge = np.array(timeedge)
    timeedge = np.sort(timeedge)
    
    return timeedge


def bdtr_tser(time, lcur, epocmask=None, perimask=None, duramask=None, verbtype=1, \
              
              # break
              durabrek=None, \
              booladdddiscbdtr=False, \
              # baseline detrend type
              bdtrtype=None, \
              # spline
              ordrspln=None, \
              scaltimespln=None, \
              # median filter
              durakernbdtrmedi=None, \
             ):
    
    if bdtrtype is None:
        bdtrtype = 'spln'
    if durabrek is None:
        durabrek = 0.1
    if ordrspln is None:
        ordrspln = 3
    if scaltimespln is None:
        scaltimespln = 0.5
    if durakernbdtrmedi is None:
        durakernbdtrmedi = 1.
    
    if verbtype > 0:
        print('Detrending the light curve...')
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits while detrending...')
   
    # determine the times at which the light curve will be broken into pieces
    timeedge = retr_timeedge(time, lcur, durabrek, booladdddiscbdtr)

    numbedge = len(timeedge)
    numbregi = numbedge - 1
    indxregi = np.arange(numbregi)
    lcurbdtrregi = [[] for i in indxregi]
    indxtimeregi = [[] for i in indxregi]
    indxtimeregioutt = [[] for i in indxregi]
    listobjtspln = [[] for i in indxregi]
    for i in indxregi:
        if verbtype > 1:
            print('i')
            print(i)
        # find times inside the region
        indxtimeregi[i] = np.where((time >= timeedge[i]) & (time <= timeedge[i+1]))[0]
        timeregi = time[indxtimeregi[i]]
        lcurregi = lcur[indxtimeregi[i]]
        
        # mask out the transits
        if epocmask is not None and len(epocmask) > 0:
            # find the out-of-transit times
            indxtimetran = []
            for k in range(epocmask.size):
                indxtimetran.append(retr_indxtimetran(timeregi, epocmask[k], perimask[k], duramask[k]))
            indxtimetran = np.concatenate(indxtimetran)
            indxtimeregioutt[i] = np.setdiff1d(np.arange(timeregi.size), indxtimetran)
        else:
            indxtimeregioutt[i] = np.arange(timeregi.size)
        
        if bdtrtype == 'medi':
            listobjtspln = None
            size = int(durakernbdtrmedi / np.amin(timeregi[1:] - timeregi[:-1]))
            lcurbdtrregi[i] = 1. + lcurregi - scipy.ndimage.median_filter(lcurregi, size=size)
        
        if bdtrtype == 'spln':
            if verbtype > 1:
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
                    numbknot = int((maxmtime - minmtime) / scaltimespln)
                    timeknot = np.linspace(minmtime, maxmtime, numbknot)
                    timeknot = timeknot[1:-1]
                    print('Knot separation: %.3g hours' % (24 * (timeknot[1] - timeknot[0])))
                    objtspln = scipy.interpolate.LSQUnivariateSpline(timeregi[indxtimeregioutt[i]], lcurregi[indxtimeregioutt[i]], timeknot, \
                                                                         k=ordrspln)
                    
                    lcurbdtrregi[i] = lcurregi - objtspln(timeregi) + 1.
                    listobjtspln[i] = objtspln
            else:
                lcurbdtrregi[i] = lcurregi
                listobjtspln[i] = None
            
            if verbtype > 1:
                print('lcurbdtrregi[i]')
                summgene(lcurbdtrregi[i])
                print('')

    return lcurbdtrregi, indxtimeregi, indxtimeregioutt, listobjtspln, timeedge


def retr_logg(radi, mass):
    
    logg = mass / radi**2

    return logg


def exec_tlsq(arry, pathimag, numbplan=None, maxmnumbplantlsq=None, strgextn='', thrssdee=7.1, \
                                 ticitarg=None, dicttlsqinpt=None, \
                                 strgplotextn='pdf', figrsize=(4., 3.), figrsizeydobskin=(8, 2.5), alphraww=0.2, \
                                 ):
    
    # setup TLS
    # temp
    #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(ticitarg))
    
    liststrgvarb = ['peri', 'epoc', 'dept', 'dura', 'powr', 'listperi']

    j = 0
    dicttlsq = {}
    for strgvarb in liststrgvarb:
        dicttlsq[strgvarb] = []
    while True:
        
        if maxmnumbplantlsq is not None and j >= maxmnumbplantlsq:
            break

        # mask
        if j == 0:
            timetlsqmeta = arry[:, 0]
            lcurtlsqmeta = arry[:, 1]
        else:
            # mask out the detected transit
            listtimetrantemp = objtresu.transit_times
            indxtimetran = []
            for timetrantemp in listtimetrantemp:
                indxtimetran.append(np.where(abs(timetlsqmeta - timetrantemp) < objtresu.duration / 2.)[0])
            indxtimetran = np.concatenate(indxtimetran)
            if indxtimetran.size != np.unique(indxtimetran).size:
                raise Exception('')
            indxtimegood = np.setdiff1d(np.arange(timetlsqmeta.size), indxtimetran)
            timetlsqmeta = timetlsqmeta[indxtimegood]
            lcurtlsqmeta = lcurtlsqmeta[indxtimegood]
        
        # transit search
        print('timetlsqmeta')
        summgene(timetlsqmeta)
        print('lcurtlsqmeta')
        summgene(lcurtlsqmeta)

        objtmodltlsq = transitleastsquares.transitleastsquares(timetlsqmeta, lcurtlsqmeta)
        
        print('dicttlsqinpt')
        print(dicttlsqinpt)
        objtresu = objtmodltlsq.power(\
                                      # temp
                                      #u=ab, \
                                      **dicttlsqinpt, \
                                      #use_threads=1, \
                                     )

        print('objtresu.period')
        print(objtresu.period)
        print('objtresu.T0')
        print(objtresu.T0)
        print('objtresu.duration')
        print(objtresu.duration)
        print('objtresu.depth')
        print(objtresu.depth)
        print('np.amax(objtresu.power)')
        print(np.amax(objtresu.power))
        print('objtresu.SDE')
        print(objtresu.SDE)
        print('FAP: %g' % objtresu.FAP) 
        
        #filelogg.write('SDE: %g\n' % objtresu.SDE)
        #filelogg.write('Period: %g\n' % objtresu.period)
        #filelogg.write('Depth: %g\n' % objtresu.depth)
        #filelogg.write('Duration: %g\n' % objtresu.duration)
        #filelogg.write('\n')
        #filelogg.close()
        
        # temp check how to do BLS instead of TLS
        #gdat.objtresu = model.power()
        #gdat.listsdee[gdat.indxlcurthis] = gdat.objtresu.SDE
        #gdat.fittperimaxmthis = gdat.objtresu.period
        #gdat.fittperimaxm.append(gdat.fittperimaxmthis)
        #gdat.peri = gdat.objtresu.periods
        #gdat.dept = gdat.objtresu.depth
        #gdat.blssamplslen = 1 -  gdat.dept
        #print('gdat.blssamplslen')
        #print(gdat.blssamplslen)
        #gdat.blssmasscomp = retr_masscomp(gdat, gdat.blssamplslen, 8.964)
        #print('gdat.blssmasscomp')
        #print(gdat.blssmasscomp)
        #gdat.dura = gdat.objtresu.duration
        #gdat.powr = gdat.objtresu.power
        #gdat.timetran = gdat.objtresu.transit_times
        #gdat.phasmodl = gdat.objtresu.model_folded_phase
        #gdat.pcurmodl = 2. - gdat.objtresu.model_folded_model
        #gdat.phasdata = gdat.objtresu.folded_phase
        #gdat.pcurdata = 2. - gdat.objtresu.folded_y
    
        # plot TLS power spectrum
        figr, axis = plt.subplots(figsize=figrsize)
        axis.axvline(objtresu.period, alpha=0.4, lw=3)
        axis.set_xlim(np.min(objtresu.periods), np.max(objtresu.periods))
        for n in range(2, 10):
            axis.axvline(n*objtresu.period, alpha=0.4, lw=1, linestyle='dashed')
            axis.axvline(objtresu.period / n, alpha=0.4, lw=1, linestyle='dashed')
        axis.set_ylabel(r'SDE')
        axis.set_xlabel('Period [days]')
        axis.plot(objtresu.periods, objtresu.power, color='black', lw=0.5)
        axis.set_xlim(0, max(objtresu.periods));

        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'sdee_tlsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # plot light curve + TLS model
        figr, axis = plt.subplots(figsize=figrsizeydobskin)
        axis.plot(timetlsqmeta, lcurtlsqmeta, alpha=alphraww, marker='o', ms=1, ls='', color='grey', rasterized=True)
        axis.plot(objtresu.model_lightcurve_time, objtresu.model_lightcurve_model, color='b')
        axis.set_xlabel('Time [days]')
        axis.set_ylabel('Relative flux');
        if j == 0:
            ylimtserinit = axis.get_ylim()
        else:
            axis.set_ylim(ylimtserinit)
        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'rflx_tlsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # plot phase curve + TLS model
        figr, axis = plt.subplots(figsize=figrsizeydobskin)
        axis.plot(objtresu.folded_phase, objtresu.folded_y, marker='o', ms=1, ls='', alpha=alphraww, color='grey', rasterized=True)
        axis.plot(objtresu.model_folded_phase, objtresu.model_folded_model, color='b')
        axis.set_xlabel('Phase')
        axis.set_ylabel('Relative flux');
        if j == 0:
            ylimpserinit = axis.get_ylim()
        else:
            axis.set_ylim(ylimpserinit)
        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'pcur_tlsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        dicttlsq['listperi'].append(objtresu.periods)
        dicttlsq['powr'].append(objtresu.power)
        print('objtresu.SDE')
        print(objtresu.SDE)
        if objtresu.SDE > thrssdee:
            dicttlsq['peri'].append(objtresu.period)
            dicttlsq['epoc'].append(objtresu.T0)
            dicttlsq['dura'].append(objtresu.duration)
            dicttlsq['dept'].append(objtresu.depth)
        else:
            break
        
        j += 1
    
    for strgvarb in liststrgvarb:
        dicttlsq[strgvarb] = np.array(dicttlsq[strgvarb])
        print('strgvarb')
        print(strgvarb)
        print('dicttlsq[strgvarb]')
        summgene(dicttlsq[strgvarb])
        print('')

    return dicttlsq


def writ_brgtcatl():
    
    catalog_data = astroquery.mast.Catalogs.query_criteria(catalog='TIC', radius=1e12, Tmag=[-15,6])
    rasc = np.array(catalog_data[:]['ra'])
    decl = np.array(catalog_data[:]['dec'])
    kici = np.array(catalog_data[:]['KIC'])
    tmag = np.array(catalog_data[:]['Tmag'])
    indx = np.where(kici != -999)[0]
    rasc = rasc[indx]
    decl = decl[indx]
    kici = kici[indx]
    tmag = tmag[indx]
    path = 'kic.txt'
    numbtarg = rasc.size
    data = np.empty((numbtarg, 2))
    data[:, 0] = kici
    data[:, 1] = tmag
    np.savetxt(path, data, fmt=['%20d', '%20g'])


def retr_rascdeclfromstrgmast(strgmast):

    print('strgmast')
    print(strgmast)
    print('Querying the TIC to get the RA and DEC of the closest TIC source...')
    listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC')
    #listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC', radius='40s')
    rasctarg = listdictcatl[0]['ra']
    decltarg = listdictcatl[0]['dec']
    
    return rasctarg, decltarg

    
def retr_lcurtess( \
              # path for downloading data
              pathtarg, \
              
              # keyword string for MAST search, where the closest target's RA and DEC will be fed into the tesscut search
              strgmast=None, \
    
              # RA for tesscut search
              rasctarg=None, \
    
              # DEC for tesscut search
              decltarg=None, \
    
              # type of data: 'SPOC': SPOC when available, lygos otherwise; 'lygos': lygos-only
              typedatatess='SPOC', \
              
              # type of SPOC data to be used
              typedataspoc='PDC', \

              # lygos
              labltarg=None, \
              strgtarg=None, \
              maxmnumbstarlygo=None, \
              ## number of pixels on a side to use with lygos
              numbside=11, \

              # SPOC
              ## subset of sectors to retrieve
              listtsecsele=None, \
              ## Boolean flag to apply quality mask
              boolmaskqual=True, \
              ## Boolean flag to only use 20-sec data when SPOC light curves with both 2-min and 20-sec data exist
              boolfastonly=True, \

             ):
    
    print('typedatatess')
    print(typedatatess)
    
    strgmast, rasctarg, decltarg = setp_coorstrgmast(rasctarg, decltarg, strgmast)

    # get the list of sectors for which TESS SPOC data are available
    listtsec, listtcam, listtccd = retr_listtsec(rasctarg, decltarg)
    numbtsec = len(listtsec)
    indxtsec = np.arange(numbtsec)

    # get the list of sectors for which TESS SPOC data are available
    print('Retrieving the list of available TESS sectors for which there is SPOC data...')
    # list of TESS sectors for which SPOC data are available
    listtsecspoc = []
    # get observation tables
    listtablobsv = retr_listtablobsv(strgmast)
    numbtabl = len(listtablobsv)
    indxtabl = np.arange(numbtabl)
    listlistproddata = [[] for k in indxtabl]
    for k, tablobsv in enumerate(listtablobsv):
        # tablobsv[target_name] is TIC ID
        if strgmast.startswith('TIC ') and tablobsv['target_name'] != strgmast[4:]:
            continue
        listlistproddata[k] = astroquery.mast.Observations.get_product_list(tablobsv)
        listlistproddata[k] = astroquery.mast.Observations.filter_products(listlistproddata[k], description='Light curves')
        for a in range(len(listlistproddata[k])):
            if listlistproddata[k][a]['description'] == 'Light curves':
               #typeprod == 'tpxf' and listproddatatemp['description'][a] == 'Target pixel files':
                tsec = int(listlistproddata[k][a]['obs_id'].split('-')[1][1:])
                listtsecspoc.append(tsec) 
    
    listtsecspoc = np.array(listtsecspoc)
    listtsecspoc = np.sort(listtsecspoc)
    
    print('listtsecspoc')
    print(listtsecspoc)
    
    # determine whether sectors have 2-minute cadence data
    booltpxf = retr_booltpxf(listtsec, listtsecspoc)

    boollygo = ~booltpxf
    listtseclygo = listtsec[boollygo]
    if typedatatess == 'lygos':
        listtseclygo = listtsec
    else:
        listtseclygo = listtsec[boollygo]
    
    listarrylcur = [[] for o in indxtsec]
    if len(listtseclygo) > 0:
        print('Will run lygos on the object...')
        dictoutp = lygos.main.init( \
                                       
                                       strgmast=strgmast, \
                                       labltarg=labltarg, \
                                       strgtarg=strgtarg, \
                                       maxmnumbstar=maxmnumbstarlygo, \
                                       listtsecsele=listtseclygo, \
                                      )
        for o, tseclygo in enumerate(listtsec):
            if tseclygo in listtseclygo:
                listarrylcur[o] = dictoutp['listarry'][o]
        arrylcursapp = None
        arrylcurpdcc = None
        listarrylcursapp = None
        listarrylcurpdcc = None

        listtsec = dictoutp['listtsec']
        listtcam = dictoutp['listtcam']
        listtccd = dictoutp['listtccd']
        
    listtsecspoc = np.setdiff1d(listtsec, listtseclygo)
    
    if len(listtsecspoc) > 0:
        
        # retrieve SPOC data from MAST
        ## get cutout images
        objtskyy = astropy.coordinates.SkyCoord(rasctarg, decltarg, unit='deg')
        listhdundata = astroquery.mast.Tesscut.get_cutouts(objtskyy, numbside)
        # parse cutout HDUs
        listtsec = []
        listicam = []
        listiccd = []
        listobjtwcss = []
        for hdundata in listhdundata:
            listtsec.append(hdundata[0].header['SECTOR'])
            listicam.append(hdundata[0].header['CAMERA'])
            listiccd.append(hdundata[0].header['CCD'])
            listobjtwcss.append(astropy.wcs.WCS(hdundata[2].header))
        
        listtsec = np.array(listtsec)

        # download data from MAST
        os.system('mkdir -p %s' % pathtarg)
        
        # select sector
        if listtsecsele is None:
            listtsecsele = listtsec
        
        #listproddatatemp = []
        #for k, proddata in enumerate(listproddata):
        #    if k in indxprodsele:
        #        listproddatatemp.append(proddata)
        #listproddata = listproddatatemp
        
        pathmasttess = pathtarg + 'mastDownload/TESS/'
        listpathdown = [[] for o in listtsec]
        for k in indxtabl:
            if len(listlistproddata[k]) > 0:
                tsec = int(listlistproddata[k][0]['obs_id'].split('-')[1][1:])
                if tsec in listtsecsele:
                    #if listlistproddata[k][a]['description'] == 'Light curves' and tsec in indxtsecsele:
                    manifest = astroquery.mast.Observations.download_products(listlistproddata[k], download_dir=pathtarg)
                    pathdown = manifest['Local Path'][0]
                    print('tsec')
                    print(tsec)
                    print('listtsec')
                    print(listtsec)
                    o = np.where(tsec == listtsec)[0][0]
                    listpathdown[o] = pathdown

        ## make sure the list of paths to sector files are time-sorted
        #listpathdown.sort()
        
                #indxsectfile = np.where(listtsec == hdundata[0].header['SECTOR'])[0][0]
        #if typedataspoc == 'SAP' or typedataspoc == 'PDC':
        #    for namefold in os.listdir(pathmasttess):
        #        
        #        # eliminate those 2-min data for which 20-sec data already exist
        #        if boolfastonly and namefold.endswith('-s'):
        #            if os.path.exists(pathmasttess + namefold[:-2] + '-a_fast'):
        #                continue    
        #        
        #        if namefold.endswith('-s') or namefold.endswith('-a_fast'):
        #            pathlcurinte = pathmasttess + namefold + '/'
        #            pathlcur = pathlcurinte + fnmatch.filter(os.listdir(pathlcurinte), '*lc.fits')[0]
        #            listpathlcur.append(pathlcur)
        #
        
        listpathsapp = []
        listpathpdcc = []
        
        # merge light curves from different sectors
        listtsec = np.empty(numbtsec, dtype=int)
        listtcam = np.empty(numbtsec, dtype=int)
        listtccd = np.empty(numbtsec, dtype=int)
        listarrylcursapp = [[] for o in indxtsec] 
        listarrylcurpdcc = [[] for o in indxtsec] 
        for o in indxtsec:
            if booltpxf[o]:
                path = listpathdown[o]
                listarrylcursapp[o], indxtimequalgood, indxtimenanngood, listtsec[o], listtcam[o], listtccd[o] = \
                                                       read_tesskplr_file(path, typeinst='tess', strgtype='SAP_FLUX', boolmaskqual=boolmaskqual)
                listarrylcurpdcc[o], indxtimequalgood, indxtimenanngood, listtsec[o], listtcam[o], listtccd[o] = \
                                                       read_tesskplr_file(path, typeinst='tess', strgtype='PDCSAP_FLUX', boolmaskqual=boolmaskqual)
            
                if typedataspoc == 'SAP':
                    arrylcur = listarrylcursapp[o]
                else:
                    arrylcur = listarrylcurpdcc[o]
                listarrylcur[o] = arrylcur
            
        if numbtsec == 0:
            print('No data have been retrieved.' % (numbtsec, strgtemp))
        else:
            if numbtsec == 1:
                strgtemp = ''
            else:
                strgtemp = 's'
            print('%d sector%s of data retrieved.' % (numbtsec, strgtemp))
        arrylcur = np.concatenate(listarrylcur, 0)
        
        arrylcursapp = np.concatenate([arry for arry in listarrylcursapp if len(arry) > 0], 0)
        arrylcurpdcc = np.concatenate([arry for arry in listarrylcurpdcc if len(arry) > 0], 0)
        
    return arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc, listtsec, listtcam, listtccd
   

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


def retr_listtsec(rasctarg, decltarg):

    # check all available TESS (FFI) data 
    objtskyy = astropy.coordinates.SkyCoord(rasctarg, decltarg, unit='deg')
    print('Calling TESSCut to get available sectors for the RA and DEC...')
    tabltesscutt = astroquery.mast.Tesscut.get_sectors(objtskyy, radius=0)
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


# plotting

def plot_lspe(pathimag, arrylcur, strgextn=''):
    
    from astropy.timeseries import LombScargle
    
    time = arrylcur[:, 0]
    lcur = arrylcur[:, 1]
    maxmperi = (np.amax(time) - np.amin(time)) / 4.
    minmperi = np.amin(time[1:] - time[:-1]) * 4.
    peri = np.linspace(minmperi, maxmperi, 500)
    freq = 1. / peri
    
    powr = LombScargle(time, lcur, nterms=2).power(freq)
    figr, axis = plt.subplots(figsize=(8, 4))
    axis.plot(peri, powr, color='k')
    
    axis.set_xlabel('Period [days]')
    axis.set_ylabel('Power')

    plt.savefig(pathimag + 'lspe_%s.pdf' % strgextn)
    plt.close()
    
    listperi = peri[np.argsort(powr)[::-1][:5]]

    return listperi


def plot_lcur(pathimag, dictmodl=None, timedata=None, lcurdata=None, \
                                        timedatabind=None, lcurdatabind=None, lcurdatastdvbind=None, boolover=True, \
                                        strgextn='', titl='', listcolrmodl=None):
    
    path = pathimag + 'lcur_%s.pdf' % strgextn
    
    # skip plotting
    if not boolover and os.path.exists(path):
        return

    figr, axis = plt.subplots(figsize=(8, 4))
    
    # model
    if dictmodl is not None:
        if listcolrmodl is None:
            listcolrmodl = [None] * len(dictmodl)
        k = 0
        for attr in dictmodl:
            axis.plot(dictmodl[attr]['time'], dictmodl[attr]['lcur'], color=listcolrmodl[k])
            k += 1

    # raw data
    if timedata is not None:
        axis.plot(timedata, lcurdata, color='grey', ls='', marker='o', ms=1, rasterized=True)
    
    # binned data
    if timedatabind is not None:
        axis.errorbar(timedatabind, lcurdatabind, yerr=lcurdatastdvbind, color='k', ls='', marker='o', ms=2)
    
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Relative flux')
    axis.set_title(titl)
    
    print(f'Writing to {path}...')
    plt.savefig(path)
    plt.close()


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
            

def fold_tser(arry, epoc, peri, boolxdattime=False, boolsort=True, phasshft=0.5):
    
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


def rebn_tser(arry, numbbins=None, delt=None):
    
    if numbbins is None and delt is None or numbbins is not None and delt is not None:
        raise Exception('')
    
    if arry.shape[0] == 0:
        print('Warning! Trying to bin an empty time-series...')
        return arry
    
    xdat = arry[:, 0]
    if numbbins is not None and delt is None:
        arryrebn = np.empty((numbbins, 3)) + np.nan
        binsxdat = np.linspace(np.amin(xdat), np.amax(xdat), numbbins + 1)
    else:
        binsxdat = np.arange(np.amin(xdat), np.amax(xdat) + delt, delt)
        numbbins = binsxdat.size - 1
        arryrebn = np.empty((numbbins, 3)) + np.nan

    meanxdat = (binsxdat[:-1] + binsxdat[1:]) / 2.
    numbxdat = meanxdat.size
    arryrebn[:, 0] = meanxdat

    indxbins = np.arange(numbbins)
    for k in indxbins:
        indxxdat = np.where((xdat < binsxdat[k+1]) & (xdat > binsxdat[k]))[0]
        arryrebn[k, 1] = np.mean(arry[indxxdat, 1])
        arryrebn[k, 2] = np.std(arry[indxxdat, 1]) / np.sqrt(indxxdat.size)
    
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
    time = listhdun[1].data['TIME'] + 2457000
    if typeinst == 'TESS':
        time += 2457000
    if typeinst == 'kplr':
        time += 2454833
    
    flux = listhdun[1].data[strgtype]
    if strgtype == 'PDCSAP_FLUX':
        stdv = listhdun[1].data['PDCSAP_FLUX_ERR']
    else:
        stdv = flux * 0.
    
    tsec = listhdun[0].header['SECTOR']
    tcam = listhdun[0].header['CAMERA']
    tccd = listhdun[0].header['CCD']
        
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


def retr_fracrtsa(fracrprs, fracsars):
    
    fracrtsa = (fracrprs + 1.) / fracsars
    
    return fracrtsa


def retr_fracsars(fracrprs, fracrtsa):
    
    fracsars = (fracrprs + 1.) / fracrtsa
    
    return fracsars


def retr_massfromradi(radiplan, strgtype='chenkipp2016'):

    if strgtype == 'chenkipp2016':
        listmass = mr_forecast.Rpost2M(radiplan, unit='Jupiter', classify='Yes')
    
    if strgtype == 'wolf2016':
        # (Wolgang+2016 Table 1)
        listmass = (2.7 * (radiplan * 11.2)**1.3 + np.random.randn(radiplan.size) * 1.9) / 317.907
        listmass = np.maximum(listmass, np.zeros_like(listmass))
    
    return listmass


def retr_esmm(tmptplanequb, tmptstar, radiplan, radistar, kmag):
    
    tmptplandayy = 1.1 * tmptplanequb
    esmm = 4.29e6 * tdpy.util.retr_specbbod(tmptplandayy, 7.5) / tdpy.util.retr_specbbod(tmptstar, 7.5) * (radiplan / radistar)*2 * 10**(-kmag / 5.)

    return esmm


def retr_tsmm(radiplan, tmptplan, massplan, radistar, jmag):

    tsmm = 700. * radiplan**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)

    return tsmm


def retr_magttess(gdat, cntp):
    
    magt = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #mliklcurtemp = -2.5 * np.log10(mlikfluxtemp) + 20.424
    #gdat.magtrefr = -2.5 * np.log10(gdat.refrrflx[o] / 1.5e4 / 30. / 60.) + 10
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return magt


def retr_scalheig(tmptplan, massplan, radiplan):
    
    # tied to Jupier's scale height for H/He at 110 K   
    scalheig = 27. * (tmptplan / 160.) / (massplan / radiplan**2) / 71398. # [R_J]

    return scalheig


def retr_rflxfromdmag(dmag, stdvdmag=None):
    
    rflx = 10**(-dmag / 2.5)

    if stdvdmag is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
    
    return rflx, stdvrflx


def retr_lcur_mock(numbplan=100, numbnois=100, numbtime=100, dept=1e-2, nois=1e-3, numbbinsphas=1000, pathplot=None, boollabltime=False, boolflbn=False):
    
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


# physics

def retr_vesc(massplan, radiplan):
    
    vesc = 59.5 * np.sqrt(massplan / radiplan) # km/s

    return vesc


def retr_dura(peri, rsma, cosi):
    
    dura = peri / np.pi * np.arcsin(np.sqrt(rsma**2 - cosi**2))
    
    return dura


# massplan in M_J
# massstar in M_S
def retr_rvelsema(peri, massplan, massstar, incl, ecce):
    
    rvelsema = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massstar + 9.548e-4 * massplan)**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]

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
    
    factrsrj = 9.731
    factrjre = 11.21
    
    factmsmj = 1048.
    factmjme = 317.8
    
    factaurj = 2093.
    
    return factrsrj, factmsmj, factrjre, factmjme, factaurj


def retr_smaxkepl(peri, masstotl):
    
    smax = (7.496e-6 * masstotl * peri**2)**(1. / 3.)
    
    return smax

    

