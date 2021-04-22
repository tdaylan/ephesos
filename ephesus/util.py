import sys
import os

import numpy as np

from tqdm import tqdm

import time as timemodu

#from numba import jit, prange
import h5py
import fnmatch

import astropy
import astropy as ap
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.io import fits

import pandas as pd

import multiprocessing

import scipy as sp
import scipy.interpolate

import astroquery
import astroquery.mast

import matplotlib.pyplot as plt

# own modules
import tdpy
from tdpy.util import summgene
import lygos
import miletos
import hattusa


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
    

def proc_axiscorr(time, lcur, axis, listindxtimeposimaxm, corr, indxtime=None, colr='k', timeoffs=2457000):
    
    if indxtime is None:
        indxtimetemp = np.arange(time.size)
    else:
        indxtimetemp = indxtime
    axis.plot(time[indxtimetemp], lcur[indxtimetemp], ls='', marker='o', color=colr, rasterized=True, ms=0.5)
    maxmydat = axis.get_ylim()[1]
    for kk in range(len(listindxtimeposimaxm)):
        if listindxtimeposimaxm[kk] in indxtimetemp:
            axis.plot(time[listindxtimeposimaxm[kk]], maxmydat, marker='v', color='b')
            #axis.text(time[listindxtimeposimaxm[kk]], maxmydat, '%.3g' % corr[listindxtimeposimaxm[kk]], color='k', va='center', \
            #                                                ha='center', size=2, rasterized=False)
    axis.set_xlabel('Time [BJD]')
    #print('timeoffs')
    #print(timeoffs)
    #axis.set_xlabel('Time [BJD-%d]' % timeoffs)
    axis.set_ylabel('Relative flux')
    

def srch_flar(time, lcur, verbtype=1, strgextn='', numbkern=3, minmscalfalltmpt=None, maxmscalfalltmpt=None, \
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
        listlcurtmpt[k] = hattusa.retr_lcurflarsing(meantimetmpt[k], timeflartmpt, amplflartmpt, scalrisetmpt, listscalfalltmpt[k])
        if not np.isfinite(listlcurtmpt[k]).all():
            raise Exception('')
        
    corr, listindxtimeposimaxm, timefull, lcurfull = corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, thrs=thrs, boolanim=boolanim, boolplot=boolplot, \
                                                                                            verbtype=verbtype, strgextn=strgextn, pathimag=pathimag)

    #corr, listindxtimeposimaxm, timefull, rflxfull = ephesus.corr_tmpt(gdat.timethis, gdat.rflxthis, gdat.listtimetmpt, gdat.listdflxtmpt, \
    #                                                                    thrs=gdat.thrstmpt, boolanim=gdat.boolanimtmpt, boolplot=gdat.boolplottmpt, \
     #                                                               verbtype=gdat.verbtype, strgextn=gdat.strgextnthis, pathimag=gdat.pathtargimag)
                
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


def corr_tmpt(time, lcur, meantimetmpt, listlcurtmpt, verbtype=2, thrs=None, strgextn='', pathimag=None, boolplot=True, boolanim=False):
    
    timeoffs = np.amin(time) // 1000
    timeoffs *= 1000
    time -= timeoffs
    
    if verbtype > 1:
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
    
    # count gaps
    difftime = time[1:] - time[:-1]
    minmdifftime = np.amin(difftime)
    difftimesort = np.sort(difftime)[::-1]
    print('difftimesort')
    for k in range(difftimesort.size):
        print(difftimesort[k] / minmdifftime)
        if k == 20:
             break

    # construct the "full" grid, i.e., regularly sampled even during the data gaps
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
    
    ## size of the full grid minus the kernel size
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
    
    print('numbtimekern')
    print(numbtimekern)

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
            
            axis[1].plot(timefull[indxtimefullruns], corr[k], color='m', ls='', marker='o', ms=1, rasterized=True)
            axis[1].plot(timefull[indxtimefullruns[listindxtimeposi]], corr[k][listindxtimeposi], color='r', ls='', marker='o', ms=1, rasterized=True)
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
                proc_axiscorr(timefull, lcurfull, axis[3+i], listindxtimeposimaxm[k], corr[k], indxtime=indxtimeplot, timeoffs=timeoffs)
            
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

def retr_indxtimetran(time, epoc, peri, dura, booloutt=False, boolseco=False):
    
    if not np.isfinite(time).all():
        raise Exception('')
    
    if not np.isfinite(dura).all():
        raise Exception('')
    
    listindxtimetran = []
    
    if np.isfinite(peri):
        intgminm = np.floor((np.amin(time) - epoc - dura / 2.) / peri)
        intgmaxm = np.ceil((np.amax(time) - epoc - dura / 2.) / peri)
        arry = np.arange(intgminm, intgmaxm + 1)
    else:
        arry = np.arange(1)

    if boolseco:
        offs = 0.5
    else:
        offs = 0.

    for n in arry:
        timeinit = epoc + (n + offs) * peri - dura / 2.
        timefinl = epoc + (n + offs) * peri + dura / 2.
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
                  booladdddiscbdtr, \
                  timescal, \
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
        timeedgeprim = (time[k] + time[k+1]) / 2.
        timeedge.append(timeedgeprim)
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
              timescalspln=None, \
              # median filter
              durakernbdtrmedi=None, \
             ):
    
    if bdtrtype is None:
        bdtrtype = 'spln'
    if durabrek is None:
        durabrek = 0.1
    if ordrspln is None:
        ordrspln = 3
    if timescalspln is None:
        timescalspln = 0.5
    if durakernbdtrmedi is None:
        durakernbdtrmedi = 1.
    
    if bdtrtype == 'spln':
        timescal = timescalspln
    else:
        timescal = durakernbdtrmedi

    if verbtype > 0:
        print('Detrending the light curve with at a time scale of %.g days...' % timescal)
        if epocmask is not None:
            print('Using a specific ephemeris to mask out transits while detrending...')
        
    # determine the times at which the light curve will be broken into pieces
    timeedge = retr_timeedge(time, lcur, durabrek, booladdddiscbdtr, timescal)

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
                    numbknot = int((maxmtime - minmtime) / timescalspln)
                    
                    timeknot = np.linspace(minmtime, maxmtime, numbknot)
                    timeknot = timeknot[1:-1]
                    #print('Knot separation: %.3g hours' % (24 * (timeknot[1] - timeknot[0])))
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


def retr_noistess(magtinpt):
    
    nois = np.array([40., 40., 40., 90.,200.,700., 3e3, 2e4]) * 1e-6
    magt = np.array([ 2.,  4.,  6.,  8., 10., 12., 14., 16.])
    objtspln = scipy.interpolate.interp1d(magt, nois, fill_value='extrapolate')
    nois = objtspln(magtinpt)
    
    return nois


def retr_magttess(gdat, cntp):
    
    magt = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #mliklcurtemp = -2.5 * np.log10(mlikfluxtemp) + 20.424
    #gdat.magtrefr = -2.5 * np.log10(gdat.refrrflx[o] / 1.5e4 / 30. / 60.) + 10
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return magt


def exec_blsq(arrytser, minmdcyc=0.001):
    
    dictblsqoutp = dict()
    
    numbtime = arrytser.shape[0]
    minmtime = np.amin(arrytser[:, 0])
    maxmtime = np.amax(arrytser[:, 0])
    indxtime = np.arange(numbtime)
    timeinit = timemodu.time()
    
    factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = retr_factconv()
    
    minmperi = 0.3
    maxmperi = (maxmtime - minmtime) / 3.
    print('maxmperi')
    print(maxmperi)
    numbtranmaxm = (maxmtime - minmtime) / minmperi
    diffperi = (maxmtime - minmtime) / numbtranmaxm
    numbperi = 5 * int((maxmperi - minmperi) / diffperi)
    print('numbperi')
    print(numbperi)
    listperi = np.linspace(minmperi, maxmperi, numbperi)
    
    numbdcyc = 4
    maxmdcyc = 0.1
    listdcyc = np.linspace(minmdcyc, maxmdcyc, numbdcyc)
    
    numboffs = 3
    minmoffs = 0.
    maxmoffs = numboffs / (1 + numboffs)
    listoffs = np.linspace(minmoffs, maxmoffs, numboffs)
    
    dflx = arrytser[:, 1] - 1.
    stdvdflx = arrytser[:, 2]
    varidflx = stdvdflx**2
    
    phasshft = 0.5

    indxperi = np.arange(numbperi)
    indxdcyc = np.arange(numbdcyc)
    indxoffs = np.arange(numboffs)
    listdept = np.zeros((numbperi, numbdcyc, numboffs))
    s2nr = np.zeros((numbperi, numbdcyc, numboffs))
    
    numbphas = 1000
    indxphas = np.arange(numbphas)
    binsphas = np.linspace(-phasshft, phasshft, numbphas + 1)
    meanphas = (binsphas[1:] + binsphas[:-1]) / 2.

    arrytserextn = arrytser[:, :, None, None] + 0 * listdcyc[None, None, :, None] * listoffs[None, None, None, :]
    for k in tqdm(range(numbperi)):
        
        arrypcur = fold_tser(arrytser, minmtime, listperi[k], boolsort=False, booldiagmode=False, phasshft=phasshft)
        
        # rebin the phase curve
        arrypcur = rebn_tser(arrypcur, binsxdat=binsphas)

        vari = arrypcur[:, 2]**2
        weig = 1. / vari
        stdvdepttemp = np.sqrt(np.nanmean(vari))
        for l in indxdcyc:
            for m in indxoffs:
                
                booltemp = (arrypcur[:, 0] + listdcyc[l] / 2. - listoffs[m]) % 1. < listdcyc[l]
                indxitra = np.where(booltemp)[0]
                if indxitra.size == 0:
                    continue
                indxotra = np.where(~booltemp)[0]
                deptitra = np.nansum(arrypcur[indxitra, 1] * weig[indxitra]) / np.nansum(weig[indxitra])
                deptotra = np.nansum(arrypcur[indxotra, 1] * weig[indxotra]) / np.nansum(weig[indxotra])
                #deptitra = np.mean(arrypcur[indxitra, 1])
                #deptotra = np.mean(arrypcur[indxotra, 1])
                depttemp = deptotra - deptitra
                s2nrtemp = depttemp / stdvdepttemp
                listdept[k, l, m] = depttemp
                s2nr[k, l, m] = s2nrtemp
                
                if False:
                #if True:
                #if k == 11 and l == 6 and m == 121:
                    print('listdcyc[l]')
                    print(listdcyc[l])
                    print('listoffs[m]')
                    print(listoffs[m])
                    print('arrypcur[:, 0]')
                    print(arrypcur[:, 0])
                    print('arrypcur[:, 1]')
                    print(arrypcur[:, 1])
                    
                    print('arrypcur[:, 1] * booltemp.astype(int)')
                    print(arrypcur[:, 1] * booltemp.astype(int))
                    print('indxitra')
                    summgene(indxitra)
                    print('arrypcur[indxitra, 1]')
                    summgene(arrypcur[indxitra, 1])
                    print('arrypcur[indxitra, 2]')
                    summgene(arrypcur[indxitra, 2])
                    print('weig[indxitra]')
                    summgene(weig[indxitra])
                    print('arrypcur[indxotra, 1]')
                    summgene(arrypcur[indxotra, 1])
                    print('arrypcur[indxotra, 2]')
                    summgene(arrypcur[indxotra, 2])
                    print('weig[indxotra]')
                    summgene(weig[indxotra])
                    print('deptitra')
                    print(deptitra)
                    print('deptotra')
                    print(deptotra)
                    print('depttemp')
                    print(depttemp)
                    print('stdvdepttemp')
                    print(stdvdepttemp)
                    print('s2nrtemp')
                    print(s2nrtemp)
                    print('')
    
    timefinl = timemodu.time()
    timetotl = timefinl - timeinit
    timeredu = timetotl / numbtime / numbperi / numboffs / numbdcyc
    
    indx = np.unravel_index(np.nanargmax(s2nr), s2nr.shape)
    
    dictblsqoutp['s2nr'] = np.nanmax(s2nr)
    dictblsqoutp['peri'] = listperi[indx[0]]
    dictblsqoutp['dura'] = listdcyc[indx[1]] * listperi[indx[0]]
    dictblsqoutp['epoc'] = minmtime + listoffs[indx[2]] * listperi[indx[0]]
    dictblsqoutp['dept'] = listdept[indx]
    
    print('temp: assuming SDE == SNR')
    dictblsqoutp['sdee'] = dictblsqoutp['s2nr']

    s2nrperi = np.empty_like(listperi)
    for k in indxperi:
        indx = np.unravel_index(np.nanargmax(s2nr[k, :, :]), s2nr[k, :, :].shape)
        s2nrperi[k] = s2nr[k, :, :][indx]
    
    # best-fit orbit
    cosi = 0
    rsma = retr_rsma(dictblsqoutp['peri'], dictblsqoutp['dura'], cosi)
    rrat = np.sqrt(dictblsqoutp['dept'])

    print('exec_blsq() took %.3g seconds in total and %g ns per observation and trial.' % (timetotl, timeredu * 1e9))
    dictblsqoutp['listperi'] = listperi
    
    print('temp: assuming power is SNR')
    dictblsqoutp['listpowr'] = s2nrperi
    
    numbtimemodl = 100000
    arrytsermodl = np.empty((numbtimemodl, 3))
    arrytsermodl[:, 0] = np.linspace(minmtime, maxmtime, 100000)
    arrytsermodl[:, 1] = retr_rflxtranmodl(arrytsermodl[:, 0], [dictblsqoutp['peri']], [dictblsqoutp['epoc']], \
                                        [rrat], 1. / factrsre, [rsma], [cosi], booltrap=False)

    arrypsermodl = fold_tser(arrytsermodl, dictblsqoutp['epoc'], dictblsqoutp['peri'], phasshft=phasshft)
    arrypserdata = fold_tser(arrytser, dictblsqoutp['epoc'], dictblsqoutp['peri'], phasshft=phasshft)
            
    dictblsqoutp['timedata'] = arrytser[:, 0]
    dictblsqoutp['rflxtserdata'] = arrytser[:, 1]
    dictblsqoutp['phasdata'] = arrypserdata[:, 0]
    dictblsqoutp['rflxpserdata'] = arrypserdata[:, 1]

    dictblsqoutp['timemodl'] = arrytsermodl[:, 0]
    dictblsqoutp['rflxtsermodl'] = arrytsermodl[:, 1]
    dictblsqoutp['phasmodl'] = arrypsermodl[:, 0]
    dictblsqoutp['rflxpsermodl'] = arrypsermodl[:, 1]
    
    print('dictblsqoutp')
    print(dictblsqoutp)

    return dictblsqoutp


def srch_pbox(arry, \
              # folder in which plots will be generated
              pathimag=None, \
              numbplan=None, \
              strgextn='', \
              thrssdee=7.1, \
              boolpuls=False, \
              ### maximum number of transiting objects
              maxmnumbtobj=None, \
              ### input dictionary for BLS
              dictblsqinpt=dict(), \
              ticitarg=None, \
              dicttlsqinpt=None, \
              booltlsq=False, \
              # plotting
              strgplotextn='pdf', \
              figrsize=(4., 3.), \
              figrsizeydobskin=(8, 2.5), \
              alphraww=0.2, \
             ):
    """
    Search for periodic boxes in time-series data
    """
    
    print('Searching for periodic boxes in time-series data...')
    print('booltlsq')
    print(booltlsq)
    print('boolpuls')
    print(boolpuls)

    if booltlsq:
        import transitleastsquares
        if dicttlsqinpt is None:
            dicttlsqinpt = dict()
    
    # setup TLS
    # temp
    #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = transitleastsquares.catalog_info(TIC_ID=int(ticitarg))
    
    liststrgvarb = ['peri', 'epoc', 'dept', 'dura', 'sdee']
    #liststrgvarb = ['peri', 'epoc', 'dept', 'dura', 'listpowr', 'listperi', 'sdee']
    
    arrysrch = np.copy(arry)
    if boolpuls:
        arrysrch[:, 1] = 2. - arrysrch[:, 1]

    j = 0
    dictsrchpboxoutp = {}
    dictsrchpboxoutp['listdictblsqoutp'] = []

    while True:
        
        print('j')
        print(j)
        if maxmnumbtobj is not None and j >= maxmnumbtobj:
            break

        # mask out the detected transit
        if j == 0:
            arrymeta = np.copy(arrysrch)
        else:
            arrymeta -= dictblsqoutp['rflxtsermodl']

        # transit search
        timeblsqmeta = arrymeta[:, 0]
        lcurblsqmeta = arrymeta[:, 1]
        if booltlsq:
            objtmodltlsq = transitleastsquares.transitleastsquares(timeblsqmeta, lcurblsqmeta)
            objtresu = objtmodltlsq.power(\
                                          # temp
                                          #u=ab, \
                                          **dicttlsqinpt, \
                                          #use_threads=1, \
                                         )

            # temp check how to do BLS instead of TLS
            dictblsq = dict()
            dictblsqoutp['listperi'] = objtresu.periods
            dictblsqoutp['listpowr'] = objtresu.power
            
            dictblsqoutp['peri'] = objtresu.period
            dictblsqoutp['epoc'] = objtresu.T0
            dictblsqoutp['dura'] = objtresu.duration
            dictblsqoutp['dept'] = objtresu.depth
            dictblsqoutp['sdee'] = objtresu.SDE
            
            dictblsqoutp['prfp'] = objtresu.FAP
            
            dictblsqoutp['listtimetran'] = objtresu.transit_times
            
            dictblsqoutp['timemodl'] = objtresu.model_lightcurve_time
            dictblsqoutp['phasmodl'] = objtresu.model_folded_phase
            dictblsqoutp['rflxpsermodl'] = objtresu.model_folded_model
            dictblsqoutp['rflxtsermodl'] = objtresu.model_lightcurve_model
            dictblsqoutp['phasdata'] = objtresu.folded_phase
            dictblsqoutp['rflxpserdata'] = objtresu.folded_y

        else:
            dictblsqoutp = exec_blsq(arrymeta, **dictblsqinpt)
        
        if boolpuls:
            dictblsqoutp['timemodl'] = 2. - dictblsqoutp['timemodl']
            dictblsqoutp['phasmodl'] = 2. - dictblsqoutp['phasmodl']
            dictblsqoutp['phasdata'] = 2. - dictblsqoutp['phasdata']
            dictblsqoutp['rflxpserdata'] = 2. - dictblsqoutp['rflxpserdata']
            dictblsqoutp['rflxpsermodl'] = 2. - dictblsqoutp['rflxpsermodl']
            dictblsqoutp['rflxtsermodl'] = 2. - dictblsqoutp['rflxtsermodl']
        
        if pathimag is not None:
            strgtitl = 'P=%.3g d, Dep=%.3g ppm, Dur=%.3g d, SDE=%.3g' % \
                        (dictblsqoutp['peri'], dictblsqoutp['dept'], dictblsqoutp['dura'], dictblsqoutp['sdee'])
            # plot TLS power spectrum
            figr, axis = plt.subplots(figsize=figrsize)
            axis.axvline(dictblsqoutp['peri'], alpha=0.4, lw=3)
            axis.set_xlim(np.min(dictblsqoutp['listperi']), np.max(dictblsqoutp['listperi']))
            for n in range(2, 10):
                axis.axvline(n * dictblsqoutp['peri'], alpha=0.4, lw=1, linestyle='dashed')
                axis.axvline(dictblsqoutp['peri'] / n, alpha=0.4, lw=1, linestyle='dashed')
            axis.set_ylabel(r'SDE')
            axis.set_xlabel('Period [days]')
            axis.plot(dictblsqoutp['listperi'], dictblsqoutp['listpowr'], color='black', lw=0.5)
            axis.set_xlim(0, max(dictblsqoutp['listperi']));
            axis.set_title(strgtitl)
            plt.subplots_adjust(bottom=0.2)
            path = pathimag + 'sdee_blsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot light curve + TLS model
            figr, axis = plt.subplots(figsize=figrsizeydobskin)
            if boolpuls:
                timeblsqmetatemp = 2. - timeblsqmeta
                lcurblsqmetatemp = 2. - lcurblsqmeta
            axis.plot(timeblsqmeta, lcurblsqmeta, alpha=alphraww, marker='o', ms=1, ls='', color='grey', rasterized=True)
            axis.plot(dictblsqoutp['timemodl'], dictblsqoutp['rflxtsermodl'], color='b')
            axis.set_xlabel('Time [days]')
            axis.set_ylabel('Relative flux');
            if j == 0:
                ylimtserinit = axis.get_ylim()
            else:
                axis.set_ylim(ylimtserinit)
            axis.set_title(strgtitl)
            plt.subplots_adjust(bottom=0.2)
            path = pathimag + 'rflx_blsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # plot phase curve + TLS model
            figr, axis = plt.subplots(figsize=figrsizeydobskin)
            axis.plot(dictblsqoutp['phasdata'], dictblsqoutp['rflxpserdata'], marker='o', ms=1, ls='', alpha=alphraww, color='grey', rasterized=True)
            axis.plot(dictblsqoutp['phasmodl'], dictblsqoutp['rflxpsermodl'], color='b')
            axis.set_xlabel('Phase')
            axis.set_ylabel('Relative flux');
            if j == 0:
                ylimpserinit = axis.get_ylim()
            else:
                axis.set_ylim(ylimpserinit)
            axis.set_title(strgtitl)
            plt.subplots_adjust(bottom=0.2)
            path = pathimag + 'pcur_blsq_tce%d_%s.%s' % (j, strgextn, strgplotextn)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        if dictblsqoutp['sdee'] > thrssdee:
            dictsrchpboxoutp['listdictblsqoutp'].append(dictblsqoutp)
        else:
            break
        j += 1
       
    # merge output BLS dictionaries of different TCEs
    for name in liststrgvarb:
        dictsrchpboxoutp[name] = []
        for k in range(len(dictsrchpboxoutp['listdictblsqoutp'])):
            dictsrchpboxoutp[name].append(dictsrchpboxoutp['listdictblsqoutp'][k][name])
        dictsrchpboxoutp[name] = np.array(dictsrchpboxoutp[name])
    
    return dictsrchpboxoutp


def retr_rascdeclfromstrgmast(strgmast):

    print('Querying the TIC using the key %s, in order to get the RA and DEC of the closest TIC source...' % strgmast)
    listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC')
    #listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC', radius='40s')
    rasctarg = listdictcatl[0]['ra']
    decltarg = listdictcatl[0]['dec']
    print('TIC, RA, and DEC of the closest match are %d, %.5g, and %.5g' % (int(listdictcatl[0]['ID']), rasctarg, decltarg))
    
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
    
              # type of data: 
              ## 'SPOC': SPOC when available, lygos otherwise
              ## 'lygos': lygos-only
              ## 'lygos-best'
              typedatatess='SPOC', \
              
              # type of SPOC data to be used
              typedataspoc='PDC', \

              ## subset of sectors to retrieve
              listtsecsele=None, \
              
              ## number of pixels on a side to use with lygos
              numbside=11, \
              
              # lygos
              labltarg=None, \
              strgtarg=None, \
              dictlygoinpt=dict(), \

              ## Boolean flag to apply quality mask
              boolmaskqual=True, \
              ## Boolean flag to only use 20-sec data when SPOC light curves with both 2-min and 20-sec data exist
              boolfastonly=True, \

             ):
    
    print('typedatatess')
    print(typedatatess)
    
    strgmast, rasctarg, decltarg = setp_coorstrgmast(rasctarg, decltarg, strgmast)

    # get the list of sectors for which TESS SPOC data are available
    listtsecffim, temp, temp = retr_listtsec(rasctarg, decltarg)

    listtsecspoc = []
        
    # get the list of sectors for which TESS SPOC TPFs are available
    print('Retrieving the list of available TESS sectors for which there is SPOC TPF data...')
    # get observation tables
    listtablobsv = retr_listtablobsv(strgmast)
    listprodspoc = []
    for k, tablobsv in enumerate(listtablobsv):
        
        listprodspoctemp = astroquery.mast.Observations.get_product_list(tablobsv)
        
        if listtablobsv['distance'][k] > 0:
            continue

        strgdesc = 'Light curves'
        listprodspoctemp = astroquery.mast.Observations.filter_products(listprodspoctemp, description=strgdesc)
        for a in range(len(listprodspoctemp)):
            boolfasttemp = listprodspoctemp[a]['obs_id'].endswith('fast')
            if not boolfasttemp:
                tsec = int(listprodspoctemp[a]['obs_id'].split('-')[1][1:])
                listtsecspoc.append(tsec) 
                listprodspoc.append(listprodspoctemp)
    
    listtsecspoc = np.array(listtsecspoc)
    listtsecspoc = np.sort(listtsecspoc)
    
    print('listtsecspoc')
    print(listtsecspoc)
    
    numbtsecspoc = listtsecspoc.size
    indxtsecspoc = np.arange(numbtsecspoc)

    # merge SPOC and FFI sector lists
    listtsec = np.unique(np.concatenate((listtsecffim, listtsecspoc)))
        
    # filter the list of sectors using the desired list of sectors, if any
    if listtsecsele is not None:
        listtsecsele = np.array(listtsecsele)
        for tsec in listtsecsele:
            if tsec not in listtsec:
                print('listtsec')
                print(listtsec)
                print('listtsecsele')
                print(listtsecsele)
                raise Exception('Selected sector is not in the list of available sectors.')
    else:
        listtsecsele = listtsec
    
    print('listtsecsele')
    print(listtsecsele)
    
    numbtsec = len(listtsecsele)
    indxtsec = np.arange(numbtsec)

    listtcam = np.empty(numbtsec, dtype=int)
    listtccd = np.empty(numbtsec, dtype=int)
    
    # determine whether sectors have 2-minute cadence data
    booltpxf = retr_booltpxf(listtsecsele, listtsecspoc)
    print('booltpxf')
    print(booltpxf)
    
    boollygo = ~booltpxf
    listtseclygo = listtsecsele[boollygo]
    if typedatatess == 'lygos' or typedatatess == 'lygos-best':
        listtseclygo = listtsecsele
        if typedatatess == 'lygos':
            booltpxflygo = False
        if typedatatess == 'lygos-best':
            booltpxflygo = True
    if typedatatess == 'SPOC':
        booltpxflygo = False
        listtseclygo = listtsecsele[boollygo]
    
    print('booltpxflygo')
    print(booltpxflygo)
    print('listtseclygo')
    print(listtseclygo)
    
    listarrylcur = [[] for o in indxtsec]
    if len(listtseclygo) > 0:
        print('Will run lygos on the object...')
        dictlygooutp = lygos.main.init( \
                                       
                                       strgmast=strgmast, \
                                       labltarg=labltarg, \
                                       strgtarg=strgtarg, \
                                       listtsecsele=listtseclygo, \
                                       
                                       # lygos-specific
                                       booltpxflygo=booltpxflygo, \
                                       **dictlygoinpt, \

                                      )
        print('listtsecsele')
        print(listtsecsele)
        print('dictlygooutp[listtsec]')
        print(dictlygooutp['listtsec'])
        for o, tseclygo in enumerate(listtsecsele):
            indx = np.where(dictlygooutp['listtsec'] == tseclygo)[0]
            if indx.size > 0:

                print('o')
                print(o)
                print('tseclygo')
                print(tseclygo)
                print('dictlygooutp[listarry]')
                summgene(dictlygooutp['listarry'])
                listarrylcur[o] = dictlygooutp['listarry'][indx[0]]
                listtcam[o] = dictlygooutp['listtcam'][indx[0]]
                listtccd[o] = dictlygooutp['listtccd'][indx[0]]
    
    listarrylcursapp = None
    listarrylcurpdcc = None
    arrylcursapp = None
    arrylcurpdcc = None
    
    print('listtsecspoc')
    print(listtsecspoc)
    if len(listtsecspoc) > 0 and not booltpxflygo:
        
        # download data from MAST
        os.system('mkdir -p %s' % pathtarg)
        
        print('Downloading SPOC data products...')
        
        listhdundataspoc = [[] for o in indxtsecspoc]
        listpathdownspoc = []

        listpathdownspoclcur = []
        for k in range(len(listprodspoc)):
            manifest = astroquery.mast.Observations.download_products(listprodspoc[k], download_dir=pathtarg)
            listpathdownspoclcur.append(manifest['Local Path'][0])

        ## make sure the list of paths to sector files are time-sorted
        listpathdownspoc.sort()
        listpathdownspoclcur.sort()
        
        ## read SPOC light curves
        if typedataspoc == 'SAP':
            print('Reading the SAP light curves...')
        else:
            print('Reading the PDC light curves...')
        listarrylcursapp = [[] for o in indxtsec] 
        listarrylcurpdcc = [[] for o in indxtsec] 
        for o in indxtsec:
            if not boollygo[o]:
                
                indx = np.where(listtsecsele[o] == listtsecspoc)[0][0]
                print('listtsecsele')
                print(listtsecsele)
                print('o')
                print(o)
                print('listtsecspoc')
                print(listtsecspoc)
                print('indx')
                print(indx)
                path = listpathdownspoclcur[indx]
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
    arrylcur = np.concatenate(listarrylcur, 0)
        
    for o, tseclygo in enumerate(listtsecsele):
        for k in range(3):
            if not np.isfinite(listarrylcur[o][:, k]).all():
                print('listtsecsele')
                print(listtsecsele)
                print('tseclygo')
                print(tseclygo)
                print('k')
                print(k)
                indxbadd = np.where(~np.isfinite(listarrylcur[o][:, k]))[0]
                print('listarrylcur[o][:, k]')
                summgene(listarrylcur[o][:, k])
                print('indxbadd')
                summgene(indxbadd)
                raise Exception('')
    if not np.isfinite(arrylcur).all():
        indxbadd = np.where(~np.isfinite(arrylcur))[0]
        print('arrylcur')
        summgene(arrylcur)
        print('indxbadd')
        summgene(indxbadd)
        raise Exception('')
                
    return arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc, listtsecsele, listtcam, listtccd
   

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
    print('Calling TESSCut to get available sectors for the RA and DEC (%.5g, %.5g)...' % (rasctarg, decltarg))
    tabltesscutt = astroquery.mast.Tesscut.get_sectors(objtskyy, radius=0)

    listtsec = np.array(tabltesscutt['sector'])
    listtcam = np.array(tabltesscutt['camera'])
    listtccd = np.array(tabltesscutt['ccd'])
   
    print('temp')
    if rasctarg == 122.989831564958:
        listtsec = np.array([7, 34])
        listtcam = np.array([1, 1])
        listtccd = np.array([3, 4])
    
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
    peri = np.linspace(minmperi, maxmperi, 10000)
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


def plot_lcur(pathimag, strgextn, dictmodl=None, timedata=None, lcurdata=None, \
              # break the line of the model when separation is very large
              boolbrekmodl=True, \
              timedatabind=None, lcurdatabind=None, lcurdatastdvbind=None, boolwritover=True, \
              titl='', listcolrmodl=None):
    
    if strgextn == '':
        raise Exception('')
    
    path = pathimag + 'lcur%s.pdf' % strgextn
    
    # skip plotting
    if not boolwritover and os.path.exists(path):
        return
    
    figr, axis = plt.subplots(figsize=(8, 4))
    
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
                axis.plot(xdat[n], ydat[n], color=listcolrmodl[k], lw=2)
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
    
    plt.subplots_adjust(bottom=0.15)
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
            

def fold_tser(arry, epoc, peri, boolxdattime=False, boolsort=True, phasshft=0.5, booldiagmode=True):
    
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


def retr_rflxtranmodl(time, peri, epoc, radiplan, radistar, rsma, cosi, ecce=0., sinw=0., booltrap=False):
    
    timeinit = timemodu.time()

    if isinstance(peri, list):
        peri = np.array(peri)

    if isinstance(epoc, list):
        epoc = np.array(epoc)

    if isinstance(radiplan, list):
        radiplan = np.array(radiplan)

    if isinstance(rsma, list):
        rsma = np.array(rsma)

    if isinstance(cosi, list):
        cosi = np.array(cosi)

    if isinstance(ecce, list):
        ecce = np.array(ecce)

    if isinstance(sinw, list):
        sinw = np.array(sinw)
    
    boolinptphys = True#False

    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    numbtime = time.size
    
    factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = retr_factconv()
    
    if boolinptphys:
        smax = (radistar + radiplan / factrsre) / rsma / factaurs
    
    rs2a = radistar / smax / factaurs
    imfa = retr_imfa(cosi, rs2a, ecce, sinw)
    sini = np.sqrt(1. - cosi**2)
    rrat = radiplan / factrsre / radistar
    dept = rrat**2
    
    rflxtranmodl = np.ones_like(time)
    
    if booltrap:
        durafull = retr_duratranfull(peri, rs2a, sini, rrat, imfa)
        duratotl = retr_duratrantotl(peri, rs2a, sini, rrat, imfa)
        duraineg = (duratotl - durafull) / 2.
        durafullhalf = durafull / 2.
    else:
        duratotl = retr_duratran(peri, rsma, cosi)
    duratotlhalf = duratotl / 2.

    # Boolean flag that indicates whether there is any transit
    booltran = np.isfinite(duratotl)
    
    numbplan = radiplan.size
    indxplan = np.arange(numbplan)
    
    if False:
    #if True:
        print('time')
        summgene(time)
        print('epoc')
        print(epoc)
        print('peri')
        print(peri)
        print('cosi')
        print(cosi)
        print('rsma')
        print(rsma)
        print('imfa')
        print(imfa)
        print('radistar')
        print(radistar)
        print('rrat')
        print(rrat)
        print('rs2a')
        print(rs2a)
        print('duratotl')
        print(duratotl)
        print('durafull')
        print(durafull)
        print('booltran')
        print(booltran)
        print('indxplan')
        print(indxplan)
    
    for j in indxplan:
        
        if booltran[j]:
            
            minmindxtran = int(np.floor((epoc[j] - minmtime) / peri[j]))
            maxmindxtran = int(np.ceil((maxmtime - epoc[j]) / peri[j]))
            indxtranthis = np.arange(minmindxtran, maxmindxtran + 1)
            
            for n in indxtranthis:
                timetran = epoc[j] + peri[j] * n
                timeshft = time - timetran
                timeshftnega = -timeshft
                timeshftabso = abs(timeshft)
                
                indxtimetotl = np.where(timeshftabso < duratotlhalf[j])[0]
                if booltrap:
                    indxtimefull = indxtimetotl[np.where(timeshftabso[indxtimetotl] < durafullhalf[j])]
                    indxtimeinre = indxtimetotl[np.where((timeshftnega[indxtimetotl] < duratotlhalf[j]) & (timeshftnega[indxtimetotl] > durafullhalf[j]))]
                    indxtimeegre = indxtimetotl[np.where((timeshft[indxtimetotl] < duratotlhalf[j]) & (timeshft[indxtimetotl] > durafullhalf[j]))]
                
                    rflxtranmodl[indxtimeinre] += dept[j] * ((timeshftnega[indxtimeinre] - duratotlhalf[j]) / duraineg[j])
                    rflxtranmodl[indxtimeegre] += dept[j] * ((timeshft[indxtimeegre] - duratotlhalf[j]) / duraineg[j])
                    rflxtranmodl[indxtimefull] -= dept[j]
                else:
                    rflxtranmodl[indxtimetotl] -= dept[j]
                
                if False:
                #if True:
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
                    print('duratotlhalf[j]')
                    print(duratotlhalf[j])
                    print('durafullhalf[j]')
                    print(durafullhalf[j])
                    print('indxtimetotl')
                    summgene(indxtimetotl)
                    print('indxtimefull')
                    summgene(indxtimefull)
                    print('indxtimeinre')
                    summgene(indxtimeinre)
                    print('indxtimeegre')
                    summgene(indxtimeegre)
    
    timetotl = timemodu.time() - timeinit
    timeredu = timetotl / numbtime
    print('retr_rflxtranmodl() took %.3g seconds in total and %g ns per time sample.' % (timetotl, timeredu * 1e9))

    return rflxtranmodl


def retr_massfromradi(listradiplan, strgtype='chenkipp2016', \
                      # Boolean flag indicating 
                      boolinptsamp=False, \
                      ):
    
    '''listradiplan in units of Earth radius'''

    if strgtype == 'chenkipp2016':
        import mr_forecast
        indxgood = np.where((listradiplan > 0.1) & (listradiplan < 100.))[0]
        if indxgood.size < listradiplan.size:
            print('retr_massfromradi(): planet radius inappropriate for mr_forecast. Truncating...')
            listradiplan = listradiplan[indxgood]
        
        if len(listradiplan) > 0:
            if boolinptsamp:
                listmass = mr_forecast.Rpost2M(listradiplan)
            else:
                listmass = np.empty_like(listradiplan)
                print('listradiplan')
                summgene(listradiplan)
                for k in range(len(listradiplan)):
                    listmass[k] = mr_forecast.Rpost2M(listradiplan[k, None])
                
            #listmass = mr_forecast.Rpost2M(listradiplan, unit='Jupiter', classify='Yes')
        else:
            listmass = np.ones_like(listradiplan) + np.nan

    if strgtype == 'wolf2016':
        # (Wolgang+2016 Table 1)
        listmass = (2.7 * (listradiplan * 11.2)**1.3 + np.random.randn(listradiplan.size) * 1.9) / 317.907
        listmass = np.maximum(listmass, np.zeros_like(listmass))
    
    return listmass


def retr_esmm(tmptplanequb, tmptstar, radiplan, radistar, kmag):
    
    tmptplandayy = 1.1 * tmptplanequb
    esmm = 4.29e6 * tdpy.util.retr_specbbod(tmptplandayy, 7.5) / tdpy.util.retr_specbbod(tmptstar, 7.5) * (radiplan / radistar)*2 * 10**(-kmag / 5.)

    return esmm


def retr_tsmm(radiplan, tmptplan, massplan, radistar, jmag):
    
    tsmm = 1.53 * radiplan**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)
    
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


def retr_dictexar(strgexar=None):
    
    # get NASA Exoplanet Archive data
    path = os.environ['EPHESUS_DATA_PATH'] + '/data/PSCompPars_2021.04.07_18.46.54.csv'
    print('Reading %s...' % path)
    objtexar = pd.read_csv(path, skiprows=316)
    if strgexar is None:
        indx = np.arange(objtexar['hostname'].size)
        #indx = np.where(objtexar['default_flag'].values == 1)[0]
    else:
        indx = np.where(objtexar['hostname'] == strgexar)[0]
        #indx = np.where((objtexar['hostname'] == strgexar) & (objtexar['default_flag'].values == 1))[0]
    
    factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = retr_factconv()
    
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
        dictexar['duratran'] = objtexar['pl_trandur'][indx].values / 24. # [day]
        dictexar['dept'] = objtexar['pl_trandep'][indx].values / 100. # dimensionless
        
        dictexar['boolfpos'] = np.zeros(numbplanexar, dtype=bool)
        
        dictexar['booltran'] = objtexar['tran_flag'][indx].values
        dictexar['booltran'] = dictexar['booltran'].astype(bool)
        
        for strg in ['radistar', 'massstar', 'tmptstar', 'loggstar', 'radiplan', 'massplan', 'tmptplan', \
                     'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'metastar', 'distsyst', 'lumistar']:
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
       
        dictexar['vesc'] = retr_vesc(dictexar['massplan'], dictexar['radiplan'])
        dictexar['masstotl'] = dictexar['massstar'] + dictexar['massplan'] / factmsme
        
        dictexar['densplan'] = objtexar['pl_dens'][indx].values / 5.51 # [d_E]
        dictexar['vsiistar'] = objtexar['st_vsin'][indx].values # [km/s]
        dictexar['projoblq'] = objtexar['pl_projobliq'][indx].values # [deg]
        
        dictexar['numbplanstar'] = np.empty(numbplanexar)
        dictexar['numbplantranstar'] = np.empty(numbplanexar)
        dictexar['boolfrst'] = np.zeros(numbplanexar, dtype=bool)
        #dictexar['booltrantotl'] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexar['namestar']):
            indxexarstar = np.where(namestar == dictexar['namestar'])[0]
            if k == indxexarstar[0]:
                dictexar['boolfrst'][k] = True
            dictexar['numbplanstar'][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexar['namestar']) & dictexar['booltran'])[0]
            dictexar['numbplantranstar'][k] = indxexarstartran.size
            #dictexar['booltrantotl'][k] = dictexar['booltran'][indxexarstar].all()
        
        dictexar['rrat'] = dictexar['radiplan'] / dictexar['radistar'] / factrsre
        
    return dictexar


# physics

def retr_vesc(massplan, radiplan):
    
    vesc = 59.5 * np.sqrt(massplan / radiplan) # km/s

    return vesc


def retr_rs2a(rsma, rrat):
    
    rs2a = rsma / (1. + rrat)
    
    return rs2a


def retr_rsma(peri, dura, cosi):
    
    rsma = np.sqrt(np.sin(dura * np.pi / peri)**2 + cosi**2)
    
    return rsma


def retr_duratranfull(peri, rs2a, sini, rrat, imfa):
    
    durafull = peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. - rrat)**2 - imfa**2))

    return durafull 


def retr_duratrantotl(peri, rs2a, sini, rrat, imfa):
    
    duratotl = peri / np.pi * np.arcsin(rs2a / sini * np.sqrt((1. + rrat)**2 - imfa**2))
    
    return duratotl

    
def retr_imfa(cosi, rs2a, ecce, sinw):
    
    imfa = cosi / rs2a * (1. - ecce)**2 / (1. + ecce * sinw)

    return imfa


def retr_amplbeam(peri, massstar, masscomp):
    
    '''Calculates the beaming amplitude'''
    
    amplbeam = 2.8e-3 * peri**(-1. / 3.) * (massstar + masscomp)**(-2. / 3.) * masscomp
    
    return amplbeam


def retr_amplelli(peri, densstar, massstar, masscomp):
    
    '''Calculates the ellipsoidal variation amplitude'''
    
    amplelli = 1.89e-2 * peri**(-2.) / densstar * (1. / (1. + massstar / masscomp))
    
    return amplelli


def retr_masscomp(amplslen, peri):
    
    print('temp: this mass calculation is an approximation.')
    masscomp = amplslen / 7.15e-5 / gdat.radistar**(-2.) / peri**(2. / 3.) / (gdat.massstar)**(1. / 3.)
    
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
    
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.)

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


def retr_duratran(peri, rsma, cosi):
    """
    Return the transit duration in the unit of the input orbital period (peri).

    Arguments
        peri: orbital period
        rsma: the sum of radii of the two bodies divided by the semi-major axis
        cosi: cosine of the inclination
    """    
    
    dura = peri / np.pi * np.arcsin(np.sqrt(rsma**2 - cosi**2))
    
    return dura


# massplan in M_E
# massstar in M_S
def retr_rvelsema(peri, massplan, massstar, incl, ecce):
    
    factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = retr_factconv()
    
    rvelsema = 203. * peri**(-1. / 3.) * massplan * np.sin(incl / 180. * np.pi) / \
                                                    (massplan + massstar * factmsme)**(2. / 3.) / np.sqrt(1. - ecce**2) # [m/s]

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
    factrsre = factrsrj * factrjre
    
    factmsmj = 1048.
    factmjme = 317.8
    factmsme = factmsmj * factmjme

    factaurs = 215.
    
    return factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs


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
        



