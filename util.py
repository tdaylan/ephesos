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

import scipy as sp
import scipy.interpolate

# own modules
import tdpy.util
from tdpy.util import summgene

import pandora.main

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


def anim_flardete(time, lcur, meantimetmpt, lcurtmpt, pathimag, listindxtimeposimaxm, corrprod, corr, strgextn='', colr=None):
    
    numbtimekern = lcurtmpt.size
    indxtimekern = np.arange(numbtimekern)
    numbtime = lcur.size
    numbtimeruns = numbtime - numbtimekern
    indxtimeruns = np.arange(numbtimeruns)
    difftime = time[1] - time[0]
    
    listpath = []
    cmnd = 'convert -delay 20'
    
    numbtimeanim = min(40, numbtimeruns)
    indxtimerunsanim = np.random.choice(indxtimeruns, size=numbtimeanim, replace=False)
    indxtimerunsanim = np.sort(indxtimerunsanim)

    for tt in indxtimerunsanim:
        
        path = pathimag + 'lcur%s_%08d.pdf' % (strgextn, tt)
        listpath.append(path)
        if not os.path.exists(path):
            figr, axis = plt.subplots(5, 1, figsize=(8, 11))
            retr_axislcur(time, lcur, axis[0], listindxtimeposimaxm, corr)
            indxtime = indxtimekern + tt
            retr_axislcur(time, lcur, axis[1], listindxtimeposimaxm, corr, indxtime=indxtime)
            axis[2].plot(meantimetmpt + tt * difftime, lcurtmpt, color='k', marker='D')
            axis[2].set_ylabel('Template')
            axis[3].plot(meantimetmpt + tt * difftime, corrprod[tt, :], color='red', marker='o')
            axis[3].set_ylabel('Correlation')
            axis[4].plot(time[indxtimeruns], corr, color='m', marker='o', ms=1, rasterized=True)
            axis[4].set_ylabel('Maximum correlation')
            
            titl = 'C = %.3g' % corr[tt]
            axis[0].set_title(titl)

            limtydat = axis[0].get_ylim()
            axis[0].fill_between(time[indxtimekern+tt], limtydat[0], limtydat[1], alpha=0.4)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
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


def retr_axislcur(time, lcur, axis, listindxtimeposimaxm, corr, indxtime=None, colr='k'):
    
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
    print('listscalfalltmpt')
    print(listscalfalltmpt)
    
    listcorr = []
    listlcurtmpt = [[] for k in indxscalfall]
    meantimetmpt = [[] for k in indxscalfall]
    for k in indxscalfall:
        numbtimekern = 3 * int(listscalfalltmpt[k] / difftime)
        meantimetmpt[k] = np.arange(numbtimekern) * difftime
        listlcurtmpt[k] = retr_lcurflarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
        if not np.isfinite(listlcurtmpt[k]).all():
            raise Exception('')
        
    corr, listindxtimeposimaxm = corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, thrs=thrs, boolanim=boolanim, boolplot=boolplot, \
                                                                                            verbtype=verbtype, strgextn=strgextn, pathimag=pathimag)

    return corr, listindxtimeposimaxm, meantimetmpt


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def corr_arryprod(lcurtemp, lcurtmpt, numbkern):
    
    # correlate
    corrprod = [[] for k in range(numbkern)]
    for k in prange(numbkern):
        corrprod[k] = lcurtmpt[k] * lcurtemp[k]
    
    return corrprod


#@jit(parallel=True)
def corr_copy(indxtimeruns, lcurstan, indxtimekern, numbkern):
    
    # make windowed copies of the light curve
    lcurtemp = [[] for k in range(numbkern)]
    for k in prange(numbkern):
        numbtimeruns = indxtimeruns[k].size
        numbtimekern = indxtimekern[k].size
        lcurtemp[k] = np.empty((numbtimeruns, numbtimekern))
        for t in prange(numbtimeruns):
            lcurtemp[k][t, :] = lcurstan[indxtimeruns[k][t]+indxtimekern[k]]
   
    return lcurtemp


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
    numbkern = len(listlcurtmpt)
    indxkern = np.arange(numbkern)
    numbtimekern = np.empty(numbkern, dtype=int)
    numbtimeruns = np.empty(numbkern, dtype=int)
    corr = [[] for k in indxkern]
    corrprod = [[] for k in indxkern]
    indxtimekern = [[] for k in indxkern]
    indxtimeruns = [[] for k in indxkern]
    listindxtimeposimaxm = [[] for k in indxkern]
    for k in indxkern:
        numbtimekern[k] = listlcurtmpt[k].size
        listlcurtmpt[k] -= np.mean(listlcurtmpt[k])
        listlcurtmpt[k] /= np.std(listlcurtmpt[k])
        indxtimekern[k] = np.arange(numbtimekern[k])
        numbtimeruns[k] = numbtime - numbtimekern[k]
        indxtimeruns[k] = np.arange(numbtimeruns[k])
    
    lcurstan = lcur - np.mean(lcur)
    lcurstan /= np.std(lcurstan)
    
    if verbtype > 1:
        print('Delta T (corr_tmpt, initial): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    listlcurtemp = corr_copy(indxtimeruns, lcurstan, indxtimekern, numbkern)
    
    if verbtype > 1:
        print('Delta T (corr_tmpt, copy): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    corrprod = corr_arryprod(listlcurtemp, listlcurtmpt, numbkern)

    if verbtype > 1:
        print('Delta T (corr_tmpt, corr_prod): %g' % (timemodu.time() - timeinit))
        timeinit = timemodu.time()
    
    boolthrsauto = thrs is None
    
    for k in indxkern:
        # find maximum correlation (maximum along the time delay axis)
        #corr[k] = np.amax(corrprod[k], 1)
        # find the total correlation (along the time delay axis)
        corr[k] = np.sum(corrprod[k], 1)
    
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
            indxtimeruns = np.arange(numbtimeruns[k])
            numbfram = 3 + numbdeteplot
            figr, axis = plt.subplots(numbfram, 1, figsize=(8, numbfram*3))
            retr_axislcur(time, lcur, axis[0], listindxtimeposimaxm[k], corr[k])
            axis[1].plot(time[indxtimeruns], corr[k], color='m', ls='', marker='o', ms=1, rasterized=True)
            axis[1].plot(time[indxtimeruns[listindxtimeposi]], corr[k][listindxtimeposi], color='r', ls='', marker='o', ms=1, rasterized=True)
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
                retr_axislcur(time, lcur, axis[3+i], listindxtimeposimaxm[k], corr[k], indxtime=indxtimeplot)
            path = pathimag + 'lcurflardete%s.pdf' % (strganim)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        if boolanim:
            path = pathimag + 'lcur%s.gif' % strganim
            if not os.path.exists(path):
                anim_flardete(time, lcur, meantimetmpt[k], listlcurtmpt[k], pathimag, listindxtimeposimaxm[k], corrprod[k], corr[k], strgextn=strganim)
    
    if verbtype > 1:
        print('Delta T (corr_tmpt, rest): %g' % (timemodu.time() - timeinit))

    return corr, listindxtimeposimaxm


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
    intgminm = np.floor((np.amin(time) - epoc - duramask / 2.) / peri)
    intgmaxm = np.ceil((np.amax(time) - epoc - duramask / 2.) / peri)
    for n in np.arange(intgminm, intgmaxm + 1):
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
    

def retr_timeedge(time, lcur, durabrek, booladdddiscbdtr):

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


def bdtr_lcur(time, lcur, epocmask=None, perimask=None, duramask=None, verbtype=1, \
              
              # break
              durabrek=0.1, \
              booladdddiscbdtr=False, \
              # baseline detrend type
              bdtrtype='spln', \
              # spline
              ordrspln=3, \
              weigsplnbdtr=1e0, \
              # median filter
              durakernbdtrmedi=1., \
             ):
    
    if verbtype > 0:
        print('Detrending the light curve...')
        if bdtrtype == 'spln':
            print('weigsplnbdtr')
            print(weigsplnbdtr)
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits...')
        else:
            print('Not using a specific ephemeris to mask out transits...')
   
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
        if epocmask is not None and perimask is not None and duramask is not None:
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
                    objtspln = scipy.interpolate.UnivariateSpline(timeregi[indxtimeregioutt[i]], lcurregi[indxtimeregioutt[i]], \
                                                                                        k=ordrspln, s=indxtimeregioutt[i].size*weigsplnbdtr)
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


def exec_tlss(arry, pathimag, numbplan=None, maxmnumbplantlss=None, \
                                    ticitarg=None, strgplotextn='pdf', figrsize=(4., 3.), figrsizeydobskin=(8, 2.5)):
    
    # setup TLS
    # temp
    #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(ticitarg))
    
    liststrgvarb = ['peri', 'epoc', 'dept', 'dura']

    j = 0
    dicttlss = {}
    for strgvarb in liststrgvarb:
        dicttlss[strgvarb] = []
    while True:
        
        # mask
        if j == 0:
            timetlssmeta = arry[:, 0]
            lcurtlssmeta = arry[:, 1]
        else:
            # mask out the detected transit
            listtimetrantemp = results.transit_times
            indxtimetran = []
            for timetrantemp in listtimetrantemp:
                indxtimetran.append(np.where(abs(timetlssmeta - timetrantemp) < results.duration / 2.)[0])
            indxtimetran = np.concatenate(indxtimetran)
            if indxtimetran.size != np.unique(indxtimetran).size:
                raise Exception('')
            indxtimegood = np.setdiff1d(np.arange(timetlssmeta.size), indxtimetran)
            timetlssmeta = timetlssmeta[indxtimegood]
            lcurtlssmeta = lcurtlssmeta[indxtimegood]
        
        # transit search
        print('timetlssmeta')
        summgene(timetlssmeta)
        print('lcurtlssmeta')
        summgene(lcurtlssmeta)

        objtmodltlss = transitleastsquares.transitleastsquares(timetlssmeta, lcurtlssmeta)
        #results = objtmodltlss.power(u=ab, use_threads=1)
        results = objtmodltlss.power(period_min=9.8, period_max=10., transit_depth_min=1)
        
        print('results.period')
        print(results.period)
        print('results.T0')
        print(results.T0)
        print('results.duration')
        print(results.duration)
        print('results.depth')
        print(results.depth)
        print('np.amax(results.power)')
        print(np.amax(results.power))
        print('results.SDE')
        print(results.SDE)
        print('FAP: %g' % results.FAP) 
        
        # plot TLS power spectrum
        figr, axis = plt.subplots(figsize=figrsize)
        axis.axvline(results.period, alpha=0.4, lw=3)
        axis.set_xlim(np.min(results.periods), np.max(results.periods))
        for n in range(2, 10):
            axis.axvline(n*results.period, alpha=0.4, lw=1, linestyle='dashed')
            axis.axvline(results.period / n, alpha=0.4, lw=1, linestyle='dashed')
        axis.set_ylabel(r'SDE')
        axis.set_xlabel('Period (days)')
        axis.plot(results.periods, results.power, color='black', lw=0.5)
        axis.set_xlim(0, max(results.periods));
        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'sdeetls%d.%s' % (j, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # plot light curve + TLS model
        figr, axis = plt.subplots(figsize=figrsizeydobskin)
        axis.scatter(timetlssmeta, lcurtlssmeta, alpha=0.5, s = 0.8, zorder=0)
        axis.plot(results.model_lightcurve_time, results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
        axis.set_xlabel('Time (days)')
        axis.set_ylabel('Relative flux');
        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'lcurtls%d.%s' % (j, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # plot phase curve + TLS model
        figr, axis = plt.subplots(figsize=figrsizeydobskin)
        axis.plot(results.model_folded_phase, results.model_folded_model, color='red')
        axis.scatter(results.folded_phase, results.folded_y, s=0.8, alpha=0.5, zorder=2)
        axis.set_xlabel('Phase')
        axis.set_ylabel('Relative flux');
        plt.subplots_adjust(bottom=0.2)
        path = pathimag + 'pcurtls%d.%s' % (j, strgplotextn)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        if numbplan is None:
            if results.SDE > 7.1 and not (maxmnumbplantlss is not None and j >= maxmnumbplantlss):
                dicttlss['peri'].append(results.period)
                dicttlss['epoc'].append(results.T0)
                dicttlss['dura'].append(results.duration)
                dicttlss['dept'].append(results.depth)
            else:
                break
        else:
            if j == numbplan:
                break
        j += 1
    
    for strgvarb in liststrgvarb:
        dicttlss[strgvarb] = np.array(dicttlss[strgvarb])
    
    return dicttlss


def writ_brgtcatl():
    
    catalog_data = astroquery.mast.Catalogs.query_criteria(catalog="TIC", radius=1e12, Tmag=[-15,6])
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


def retr_data(datatype, strgmast, pathtarg, \
              
              boolmaskqual=True, \
              
              # Pandora
              ticitarg=None, \
              strgtarg=None, \
              labltarg=None, \
              sectsele=None, \
              maxmnumbstarpand=None, \
             ):
    
    # download data
    if datatype != 'pand':
        pathmasttess = pathtarg + 'mastDownload/TESS/'
        if not os.path.exists(pathmasttess):
            print('Trying to download SPOC data with keyword: %s' % strgmast)
            listpathdown = down_spoclcur(pathtarg, strgmast, sectsele=sectsele)
            print('listpathdown')
            print(listpathdown)
        else:
            print('SPOC folder already exists at %s. Will not attempt at downloading SPOC data...' % pathmasttess)
   
    # determine type of data to be used for allesfitter analysis
    if datatype is None:
        if os.path.exists(pathmasttess):
            if boolsapp:
                datatype = 'sapp'
            else:
                datatype = 'pdcc' 
        else:
            datatype = 'pand'
    
    print('datatype')
    print(datatype)

    if datatype != 'pand':
        listpathlcur = []
        if datatype == 'sapp' or datatype == 'pdcc':
            for namefile in os.listdir(pathmasttess):
                if namefile.endswith('-s'):
                    pathlcurinte = pathmasttess + namefile + '/'
                    pathlcur = pathlcurinte + fnmatch.filter(os.listdir(pathlcurinte), '*_lc.fits')[0]
                    listpathlcur.append(pathlcur)

        if datatype == 'qlop':
            pathlcurqlop = pathtarg + 'qlop/'
            print('Searching for QLP light curve(s) in %s...' % pathlcurqlop)
            os.system('mkdir -p %s' % pathlcurqlop)
            listtemp = fnmatch.filter(os.listdir(pathlcurqlop), 'sector-*')
            if len(listtemp) > 0:
                listpathlcur.extend(pathlcurqlop + listtemp)
                print('Found QLP light curves:')
                for temp in listtemp:
                    print(temp)

        ## make sure the list of paths to sector files are time-sorted
        listpathlcur.sort()
        
        listpathsapp = []
        listpathpdcc = []
    
        # merge light curves from different sectors
        numbsect = len(listpathlcur)
        indxsect = np.arange(numbsect)
        listisec = np.empty(numbsect, dtype=int)
        listicam = np.empty(numbsect, dtype=int)
        listiccd = np.empty(numbsect, dtype=int)
        listarrylcursapp = [[] for o in indxsect] 
        listarrylcurpdcc = [[] for o in indxsect] 
        listarrylcur = []
        for o, pathlcur in enumerate(listpathlcur):
            if datatype == 'qlop':
                arrylcur = read_qlop(pathlcur, typeinst='tess', boolmask=True)
            else:
                listarrylcursapp[o], indxtimequalgood, indxtimenanngood, listisec[o], listicam[o], listiccd[o] = \
                                                       read_tesskplr_file(pathlcur, typeinst='tess', strgtype='SAP_FLUX', boolmaskqual=boolmaskqual)
                listarrylcurpdcc[o], indxtimequalgood, indxtimenanngood, listisec[o], listicam[o], listiccd[o] = \
                                                       read_tesskplr_file(pathlcur, typeinst='tess', strgtype='PDCSAP_FLUX', boolmaskqual=boolmaskqual)
                
                if datatype == 'sapp':
                    arrylcur = listarrylcursapp[o]
                else:
                    arrylcur = listarrylcurpdcc[o]
            listarrylcur.append(arrylcur)
        print('%d sectors of data retrieved.' % numbsect)
        arrylcur = np.concatenate(listarrylcur, 0)
        arrylcursapp = np.concatenate(listarrylcursapp, 0)
        arrylcurpdcc = np.concatenate(listarrylcurpdcc, 0)
    
    else:
        print('Will run pandora on the object...')
        listarrylcur, listmeta = pandora.main.main( \
                                       ticitarg=ticitarg, \
                                       labltarg=labltarg, \
                                       strgmast=strgmast, \
                                       strgtarg=strgtarg, \
                                       maxmnumbstar=maxmnumbstarpand, \
                                      )
        arrylcur = np.concatenate(listarrylcur, 0) 
        arrylcursapp = None
        arrylcurpdcc = None
        listarrylcursapp = None
        listarrylcurpdcc = None

        listisec, listicam, listiccd = listmeta
        
    return datatype, arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc, listisec, listicam, listiccd
   

def down_spoclcur(pathdownbase, strgmast, boollcuronly=True, sectsele=None):
    
    if strgmast is None:
        raise Exception('strgmast should not be None.')

    obsTable = astroquery.mast.Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', objectname=strgmast)
    print('strgmast')
    print(strgmast)
    print('obsTable')
    print(obsTable)
    
    #obid = np.array([obsTable['obs_id'][a] for a in range(len(obsTable['obs_id']))])
    #print('obid')
    #print(obid)
    
    catalogData = astroquery.mast.Catalogs.query_object(strgmast, catalog="TIC")
    rasc = catalogData[0]['ra']
    decl = catalogData[0]['dec']
    strgtici = '%s' % catalogData[0]['ID']
    
    print('obsTable')
    print(obsTable)
    print('')
    for valu in obsTable:
        print('valu')
        print(valu)
        print('')
    print('')
    #namefile = obsTable[0]['obs_id']

    listpathdown = []
    for tabl in obsTable:
        if tabl['target_name'] == '%s' % strgtici:
            dataProducts = astroquery.mast.Observations.get_product_list(tabl)
            
            # number of data products
            #numbprod = len(dataProducts['description']
            numbprod = len(dataProducts)
            indxprod = np.arange(numbprod)
            
            print('dataProducts')
            print(dataProducts)
            print('')
            for valu in dataProducts:
                print('valu')
                print(valu)
                print('')
            print('')
            # select light curve data products
            desc = np.array([dataProducts['description'][a] for a in indxprod])
            if boollcuronly:
                indxprodlcur = np.where(desc == 'Light curves')[0]
            else:
                indxprodlcur = indxprod
            
            # select sector
            if sectsele is not None:
                obid = np.array([dataProducts['obs_id'][a] for a in indxprod])
                print('obid')
                print(obid)
                indxprodsect = []
                for a in indxprod:
                    sect = int(obid[a].split('-')[1][1:])
                    if sect == sectsele:
                        indxprodsect.append(a)
                indxprodsect = np.array(indxprodsect)
            else:
                indxprodsect = indxprod
            
            indxprodsele = np.intersect1d(indxprodsect, indxprodlcur)
            print('indxprodsect')
            print(indxprodsect)
            print('indxprodlcur')
            print(indxprodlcur)
            print('indxprodsele')
            print(indxprodsele)
            print('indxprod')
            print(indxprod)
            print('')
            print('')
            print('')
            print('')

            if indxprodsele.size > 0:
                manifest = astroquery.mast.Observations.download_products(dataProducts[indxprodsele], download_dir=pathdownbase)
                pathdown = manifest['Local Path'][0]
                listpathdown.append(pathdown)
        else:
            print('The TIC ID of the nearest target is not the target name in the observation table.')
            print('tabl[target_name]')
            print(tabl['target_name'])
            print('strgtici')
            print(strgtici)
            print('')
    if len(listpathdown) == 0:
        print('No SPOC data is found...')

    return listpathdown


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

    plt.savefig(pathimag + 'lspe%s.pdf' % strgextn)
    plt.close()
    
    listperi = peri[np.argsort(powr)[::-1][:5]]

    return listperi


def plot_lcur(pathimag, timemodl=None, lcurmodl=None, timedata=None, lcurdata=None, \
                                        timedatabind=None, lcurdatabind=None, lcurdatastdvbind=None, \
                                        strgextn='', titl=''):

    figr, axis = plt.subplots(figsize=(8, 4))
    
    # model
    if timemodl is not None:
        axis.plot(timemodl, lcurmodl, color='b')
    
    # raw data
    if timedata is not None:
        axis.plot(timedata, lcurdata, color='grey', ls='', marker='o', ms=0.5, rasterized=True)
    
    # binned data
    if timedatabind is not None:
        axis.errorbar(timedatabind, lcurdatabind, yerr=lcurdatastdvbind, color='k', ls='', marker='o', ms=2)
    
    axis.set_xlabel('Time [days]')
    axis.set_ylabel('Relative flux')
    axis.set_title(titl)
    
    path = pathimag + f'lcurdata%s.pdf' % strgextn
    print(f'Writing to {path}...')
    plt.savefig(path)
    plt.close()


def plot_pcur(pathimag, arrylcur=None, arrypcur=None, arrypcurbind=None, phascent=0., boolhour=False, epoc=None, peri=None, strgextn='', \
                                                            boolbind=True, timespan=None, booltime=False, numbbins=100, limtxdat=None):
    
    if arrypcur is None:
        arrypcur = fold_lcur(arrylcur, epoc, peri)
    if arrypcurbind is None and boolbind:
        arrypcurbind = rebn_lcur(arrypcur, numbbins)
        
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
        print('Assuming a constant uncertainty of 1 percent for the SAP data.')
        stdv = flux * 1e-2
    
    isec = listhdun[0].header['SECTOR']
    icam = listhdun[0].header['CAMERA']
    iccd = listhdun[0].header['CCD']
        
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
    
    return arry, indxtimequalgood, indxtimenanngood, isec, icam, iccd


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
            path = pathdatalcurflbn + 'lcurflbn_012%d_%d.pdf' % (listtici[k], listitoi[k])
            if (k % numbplotfram == numbplotfram - 1 or k == len(listdata[0]) - 1) and not os.path.exists(path):
                axis.set_xlabel('Phase')
                axis.set_ylabel('$\Delta f$')
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                cntr += 1
        cmnd = 'convert %stran_*.pdf tran.gif' % pathdata
        # temp
        #os.system(cmnd)
    
    listtemp = np.where(~np.isfinite(listdata[1]))
    indxphasbadd = listtemp[1]
    indxdatabadd = listtemp[0]

    return listdata


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

    tsmm = radiplan**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)

    return tsmm


def retr_magttess(gdat, cntp):
    
    magt = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #mliklcurtemp = -2.5 * np.log10(mlikfluxtemp) + 20.424
    #gdat.magtrefr = -2.5 * np.log10(gdat.refrrflx[o] / 1.5e4 / 30. / 60.) + 10
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return magt


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

def retr_dura(peri, rsma, cosi):
    
    dura = peri / np.pi * np.arcsin(np.sqrt(rsma**2 - cosi**2))
    
    return dura


# massplan in M_J
# massstar in M_S
def retr_radvsema(peri, massplan, massstar, incl, ecce):
    
    radvsema = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massstar + 9.548e-4 * massplan)**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]

    return radvsema


# semi-amplitude of radial velocity of a two-body
# masstar in M_S
# massplan in M_J
# peri in days
# incl in degrees
def retr_radv(time, epoc, peri, massplan, massstar, incl, ecce, arguperi):
    
    phas = (time - epoc) / peri
    phas = phas % 1.
    #consgrav = 2.35e-49
    #cons = 1.898e27
    #masstotl = massplan + massstar
    #smax = 
    #ampl = np.sqrt(consgrav / masstotl / smax / (1. - ecce**2))
    #radv = cons * ampl * mass * np.sin(np.pi * incl / 180.) * (np.cos(np.pi * arguperi / 180. + 2. * np.pi * phas) + ecce * np.cos(np.pi * arguperi / 180.))

    ampl = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massstar + 9.548e-4 * massplan)**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]
    radv = ampl * (np.cos(np.pi * arguperi / 180. + 2. * np.pi * phas) + ecce * np.cos(np.pi * arguperi / 180.))

    return radv


def retr_smaxkepl(peri, masstotl):
    
    smax = (7.496e-6 * masstotl * peri**2)**(1. / 3.)
    
    return smax

    

