import sys
import os

from tqdm import tqdm

import time as modutime

import multiprocessing

import numpy as np

import scipy as sp
import scipy.interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy
import astropy as ap
import astropy.timeseries

import astroquery
import astroquery.mast

import miletos
import nicomedia
import chalcedon
import tdpy
from tdpy import summgene


def retr_listtypesyst():
    
    listtypesyst = [ \
                    'PlanetarySystem', \
                    'PlanetarySystemEmittingCompanion', \
                    'PlanetarySystemWithMoons', \
                    'PlanetarySystemWithRingsFaceOn', \
                    'PlanetarySystemWithRingsInclinedVertical', \
                    'PlanetarySystemWithRingsInclinedHorizontal', \
                   ]
    
    return listtypesyst


def retr_boolgridouts(gdat, j, typecoor, typeoccu='comp'):
    '''
    Return a grid of Booleans on either grid indicating which prid points are not occulted by the companion or the primary
    typeoccu == 'comp' returns the Booleans outside the companion
    typeoccu == 'prim' returns the Booleans outside the primary
    '''
    
    if gdat.boolsystpsys:
        if typeoccu == 'comp':
            if typecoor == 'comp':
                boolnocccomp = gdat.distfromcompgridcomp[j] > gdat.rratcomp[j]

                if gdat.typesyst == 'PlanetarySystemWithRingsInclinedHorizontal':
                    boolnoccdisk = (gdat.xposgridcomp[j] / 1.75 / gdat.rratcomp[j])**2 + (gdat.yposgridcomp[j] / 0.2 / gdat.rratcomp[j])**2 > 1.
                    boolnocccomp = boolnocccomp & boolnoccdisk
                   
                elif gdat.typesyst == 'PlanetarySystemWithRingsInclinedVertical':
                    boolnoccdisk = (gdat.yposgridcomp[j] / 1.75 / gdat.rratcomp[j])**2 + (gdat.xposgridcomp[j] / 0.2 / gdat.rratcomp[j])**2 > 1.
                    boolnocccomp = boolnocccomp & boolnoccdisk
                   
                elif gdat.typesyst == 'PlanetarySystemWithRingsFaceOn':
                    boolnoccdisk = ~((gdat.distfromcompgridcomp[j] > 1.5 * gdat.rratcomp[j]) & (gdat.distfromcompgridcomp[j] < 1.75 * gdat.rratcomp[j]))
                    boolnocccomp = boolnocccomp & boolnoccdisk
    
            else:
                gdat.distfromcompgridstar[j] = np.sqrt((gdat.xposgridstar - gdat.xposcompgridstar[j])**2 + (gdat.yposgridstar - gdat.yposcompgridstar[j])**2)
                boolnocccomp = gdat.distfromcompgridstar[j] > gdat.rratcomp[j]
                
        if typeoccu == 'star':
            if typecoor == 'comp':
                # distance to the primary in the grid of the companion j
                boolnocccomp = gdat.distfromprimgridcomp[j] > 1.
            else:
                raise Exception('')
    elif gdat.typesyst == 'turkey':
        positemp = np.vstack([gdat.xposgridshft.flatten(), gdat.yposgridshft.flatten()]).T
        indx = np.where((abs(positemp[:, 0]) < gdat.maxmxposturkmesh) & (abs(positemp[:, 1]) < gdat.maxmyposturkmesh))[0]

        boolnocccomp = np.ones(gdat.yposgridshft.size, dtype=bool)
        boolnocccomp[indx] = scipy.interpolate.griddata(gdat.positurkmesh, gdat.valuturkmesh, positemp[indx, :], fill_value=0.) < 0.5

        boolnocccomp = boolnocccomp.reshape(gdat.xposgridshft.shape)

    else:
        print('')
        print('')
        print('')
        print('gdat.typesyst')
        print(gdat.typesyst)
        raise Exception('gdat.typesyst undefined.')
    
    return boolnocccomp


def retr_lumistartran(gdat, typecoor, boolrofi, j=None):
    '''
    Calculate the total flux of a star on the grid
    '''
    indxgridrofi = np.where(boolrofi)
    lumistartran = retr_lumistartranrofi(gdat, typecoor, indxgridrofi, j)
    
    lumistartran = np.sum(lumistartran)

    return lumistartran


def retr_lumistartranrofi(gdat, typecoor, indxgridrofi, j=None):
    '''
    Calculate the relative flux from a brightness map
    '''
    
    if typecoor == 'comp' or typecoor == 'sour':
        if typecoor == 'comp':
            dist = gdat.distfromprimgridcomp[j]
            areapixl = gdat.areapixlcomp
        if typecoor == 'sour':
            dist = gdat.diststargridsour
            areapixl = gdat.areapixlsour
        
        if indxgridrofi is not None:
            dist = dist[indxgridrofi]

        cosg = np.sqrt(1. - dist**2)
        fluxstartran = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk) * areapixl
        
        if gdat.booldiag and (abs(fluxstartran) > 1e10).any() or not np.isfinite(fluxstartran).all():
            print('dist')
            summgene(dist)
            print('fluxstartran')
            print(fluxstartran)
            print('typecoor')
            print(typecoor)
            raise Exception('')

    if typecoor == 'star':
        fluxstartran = gdat.brgtprimgridprim[indxgridrofi]
            
    return fluxstartran


def retr_pathbaseanimfram(gdat, namevarbanim):
    
    pathbaseanimfram = gdat.pathvisu + '%s%s%s' % (namevarbanim, gdat.strgextn, gdat.strgcompmoon)
    
    return pathbaseanimfram
   
   
def retr_pathanimfram(gdat, namevarbanim, t, boolimaglfov, j):
    
    pathbaseanimfram = retr_pathbaseanimfram(gdat, namevarbanim)
    
    if boolimaglfov:
        path = pathbaseanimfram + '_Diagram%04d.%s' % (t, gdat.typefileplot)
    else:
        if j is None:
            strgextn = ''
        else:
            strgextn = '_co%02d' % j
        path = pathbaseanimfram + '_%s%s_Frame%04d.%s' % (gdat.typecoor, strgextn, t, gdat.typefileplot)
    
    return path
   
   
def make_imag(gdat, t, typecolr='real', typemrkr='none', j=None):
    
    # temp (boolimaglfov will probably be deleted)
    boolimaglfov = False

    # to be deleted?
    #if gdat.booldiag:
    #    if j is None and typecoor == 'comp':
    #        raise Exception('')

    for namevarbanim in gdat.listnamevarbfram:
        
        #if gdat.indxframthis != 0 and namevarbanim in ['posifrstphotlens', 'posisecophotlens', 'cntsfrstlens', 'cntssecolens']:
        #    continue
        
        boolexst = os.path.exists(gdat.pathgiff[namevarbanim])
        if not boolexst:
            
            path = retr_pathanimfram(gdat, namevarbanim, t, boolimaglfov, j)
            
            if not boolimaglfov:
                gdat.cmndmakeanim[namevarbanim] += ' %s' % path
                gdat.cmnddeleimag[namevarbanim] += ' %s' % path
        
            figr, axis = plt.subplots(figsize=(6, 6))
            
            if namevarbanim == 'flux':
                if not boolimaglfov:
            
                    if gdat.booldiag:
                        if not hasattr(gdat, 'boolevalflux'):
                            raise Exception('')
            
                    if gdat.boolevalflux:
                        if gdat.boolsystpsys:
                            if gdat.typecoor == 'comp':
                                
                                # brightness on the companion grid points
                                brgttemp = np.zeros_like(gdat.xposgridcomp[j])
                                
                                # calculate and load the brightness over pixels not occulted by the companion
                                brgttemp[gdat.indxgridcompprimnocc] = calc_brgtprimgridcomp(gdat, j, t)[gdat.indxgridcompprimnocc]
                                
                                # load the brightness over pixels not occulted by the primary
                                if gdat.typebrgtcomp != 'dark':
                                    brgttemp[gdat.indxplannoccgridcomp[j]] = gdat.brgtcomp
                        
                            elif gdat.typecoor == 'star':
                                # brightness on the primary grid points
                                brgttemp = np.zeros_like(gdat.brgtprimgridprim)
                                
                                # load the brightness of pixels from the pre-calculated primary grid, which are not occulted by the companion
                                brgttemp[gdat.boolgridstarlght] = gdat.brgtprimgridprim[gdat.boolgridstarlght]
                            else:
                                print('')
                                print('')
                                print('')
                                print('typecoor')
                                print(typecoor)
                                print('gdat.typesyst')
                                print(gdat.typesyst)
                                raise Exception('Could not define brgttemp.')
                        
                        elif gdat.typesyst == 'CompactObjectStellarCompanion':
                            brgttemp = gdat.brgtlens
                        else:
                            print('')
                            print('')
                            print('')
                            print('gdat.typesyst')
                            print(gdat.typesyst)
                            raise Exception('Could not define brgttemp.')
                    else:
                        if gdat.typecoor == 'comp':
                            # primary brightness in the companion grid
                            brgttemp = np.zeros_like(gdat.xposgridcomp[j])
                            
                        if gdat.typecoor == 'star':
                            brgttemp = gdat.brgtprimgridprim

                    cmap = 'magma'
                    #cmap = 'Blues_r'
                    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white","blue"])
                    vmax = gdat.maxmlumistar
                    imag = axis.imshow(brgttemp, origin='lower', interpolation='nearest', cmap=cmap, vmin=0., vmax=vmax)
        
                if gdat.boolshowlcuranim:
                    axistser = figr.add_axes([0.2, 0.1, 0.6, 0.25], frameon=False)
                    
                    gdat.timeanimprev = 0.5 * (gdat.time[-1] - gdat.time[0])
                    
                    print('t')
                    print(t)
                    print('gdat.time')
                    summgene(gdat.time)
                    print('')

                    gdat.numbtimeprev = int(gdat.timeanimprev / (gdat.time[t] - gdat.time[t-1]))
                    if j is None:
                        indxtimeinit = max(0, t - gdat.numbtimeprev)
                        xdat = gdat.time[indxtimeinit:t+1]
                        ydat = gdat.lumisyst[indxtimeinit:t+1]
                        xlim = [gdat.time[t] - gdat.timeanimprev, gdat.time[t]]
                    else:
                        xdat = gdat.listphaseval[j][:t+1]
                        ydat = gdat.lumisysteval[j][:t+1]
                        xlim = [-0.25, 0.75]
                    
                    axistser.plot(xdat, ydat, marker='', color='firebrick', ls='-', lw=1)
                    
                    sprd = np.amax(gdat.rratcomp)**2
                    ylim = gdat.lumistarnocc * np.array([1. - 3. * sprd, 1. + sprd])
                    axistser.set_ylim(ylim)
                    
                    axistser.set_xlim(xlim)
                    
                    if gdat.booldiag:
                        if gdat.typesyst == 'PlanetarySystem':
                            if ylim[0] > np.amin(ydat):
                                raise Exception('')
                            if ylim[1] < np.amin(ydat):
                                raise Exception('')
                    
                    if not boolimaglfov:
                        axistser.axis('off')

            if namevarbanim == 'posifrstphotlens':
    
                imag = axis.scatter(gdat.xposfrst, gdat.yposfrst, s=0.001)

            if namevarbanim == 'posisecophotlens':
    
                imag = axis.scatter(gdat.xposseco, gdat.yposseco, s=0.001)
            

            if namevarbanim == 'brgtgridsour':
                gdat.brgtgridsourimag = np.zeros_like(gdat.xposgridsour)
                gdat.brgtgridsourimag[gdat.indxgridsourstar]
                imag = axis.imshow(gdat.brgtgridsourimag, origin='lower', interpolation='nearest', cmap=cmap, vmin=0., vmax=gdat.maxmbrgtsour)
            
            if namevarbanim == 'fluxfrstlens':
                imag = axis.imshow(gdat.fluxfrstlens, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmlumistar)
            
            if namevarbanim == 'fluxsecolens':
                imag = axis.imshow(gdat.fluxsecolens, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmlumistar)
            
            if namevarbanim == 'cntsfrstlens':
                imag = axis.imshow(gdat.cntsfrstlens, origin='lower', interpolation='nearest', cmap='magma')
            
            if namevarbanim == 'cntssecolens':
                imag = axis.imshow(gdat.cntssecolens, origin='lower', interpolation='nearest', cmap='magma')
            
            axis.set_aspect('equal')
            if not boolimaglfov:
                axis.axis('off')
            
            if boolimaglfov:
                if gdat.booldiag:
                    if j is not None and boolimaglfov:
                        raise Exception('')
                
                axis.set_xlim(gdat.limtposiimag)
                axis.set_ylim(gdat.limtposiimag)

                if typecoor == 'star':
            
                    axis.add_patch(plt.Circle((0, 0), 1., color='orange'))
                    for j in gdat.indxcomp:
                        axis.add_patch(plt.Circle((gdat.xposcompgridstar[j], gdat.yposcompgridstar[j]), gdat.radicomp[j], color=gdat.listcolrcomp[j]))
                        #axis.plot(gdat.dictvarborbt['posicompgridprim'][:, j, 0], gdat.dictvarborbt['posicompgridprim'][:, j, 1], ls='-', color=gdat.listcolrcomp[j])
                        
                        for ou in gdat.indxsegmfade:
                            axis.plot(gdat.dictvarborbt['posicompgridprim'][gdat.indxtimesegmfade[ou], j, 0], \
                                      gdat.dictvarborbt['posicompgridprim'][gdat.indxtimesegmfade[ou], j, 1], alpha=gdat.listalphline[j][ou], \
                                                                                                                                            ls='-', color=gdat.listcolrcomp[j])
                            
                        #lc = mpl.collections.LineCollection(segments=gdat.listsegm, lw=1, array=gdat.listalphline[j], color=gdat.listcolrcomp[j])#cmap=myred, lw=3)
                        #line = axis.add_collection(lc)




            if gdat.strgtitl is not None:
                axis.set_title(gdat.strgtitl)
            else:
                # temp
                strgtitl = ''
                axis.set_title(strgtitl)
            
            if gdat.boolshowlcuranim:
                bbox = dict(boxstyle='round', ec='white', fc='white')
                if j is not None:
                    strgtextinit = 'Time from midtransit'
                    facttime, strgtextfinl = tdpy.retr_timeunitdays(abs(gdat.listphaseval[j][t] * gdat.pericomp[j]))
                    timemtra = gdat.listphaseval[j][t] * gdat.pericomp[j] * facttime
                    if gdat.typelang == 'Turkish':
                        strgtextinit = gdat.dictturk[strgtextinit]
                        strgtextfinl = gdat.dictturk[strgtextfinl]
                    strgtext = '%s: %.2f %s \n Phase: %.3g' % (strgtextinit, timemtra, strgtextfinl, gdat.listphaseval[j][t])
                else:
                    timemtra = gdat.time[t]
                    strgtextinit = 'Time'
                    strgtextfinl = 'days'
                    strgtext = '%s: %.2f %s' % (strgtextinit, timemtra, strgtextfinl)
                axis.text(0.5, 0.95, strgtext, bbox=bbox, transform=axis.transAxes, color='firebrick', ha='center')
            
            #tdpy.sign_code(axis, 'ephesos')
            
            print('Writing to %s...' % path)
            if namevarbanim == 'posifrstphotlens' or namevarbanim == 'posisecophotlens':
                plt.savefig(path, dpi=400)
            else:
                plt.savefig(path, dpi=200)
            
            plt.close()
    
    if not boolimaglfov:
        gdat.indxframthis += 1

    
def calc_brgtprimgridcomp(gdat, j, t):
    
    ## determine the pixels over which the stellar brightness will be calculated
    if abs(gdat.listphaseval[j][t]) < 0.25:
        ## when the companion is in front of the primary
        
        # Booleans indicating where the primary is not occulted in the companion grid
        gdat.boolgridcompprimnocc = gdat.boolgridcompoutscomp[j] & gdat.boolgridcompinsdprim

        # indices of the companion grid where the primary is not occulted
        gdat.indxgridcompprimnocc = np.where(gdat.boolgridcompprimnocc)
    else:
        ## when the companion is behind the primary
        gdat.indxgridcompprimnocc = np.where(gdat.boolgridcompinsdprim)
    
    cosg = np.sqrt(1. - gdat.distfromprimgridcomp[j][gdat.indxgridcompprimnocc]**2)
    brgtprim = np.zeros_like(gdat.distfromprimgridcomp[j])
    brgtprim[gdat.indxgridcompprimnocc] = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk) * gdat.areapixlstar
    
    return brgtprim


def retr_brgtlens(gdat, t):
    
    # brightness in the sources plane
    gdat.brgtgridsour = retr_lumistartran(gdat, 'sour', gdat.indxgridsourstar)
    print('gdat.brgtgridsour')
    summgene(gdat.brgtgridsour)

    #print('gdat.brgtgridsour')
    #summgene(gdat.brgtgridsour)
    #gdat.brgtgridsourtotl = np.sum(gdat.brgtgridsour)
    #print('gdat.brgtgridsourtotl')
    #print(gdat.brgtgridsourtotl)
    
    # magnified brightness on the source plane
    gdat.brgtgridsourmagn = (gdat.brgtgridsour * gdat.magn[gdat.indxgridsourstar]).flatten()
    print('gdat.brgtgridsourmagn')
    summgene(gdat.brgtgridsourmagn)

    gdat.arrygridsour = np.vstack([gdat.xposgridsour[gdat.indxgridsourstar].flatten(), \
                                   gdat.yposgridsour[gdat.indxgridsourstar].flatten()]).T
    
    print('gdat.arrygridsour')
    summgene(gdat.arrygridsour)
    print('gdat.arrygridsourintp')
    summgene(gdat.arrygridsourintp)
    gdat.brgtintp = scipy.interpolate.griddata(gdat.arrygridsour, gdat.brgtgridsourmagn, gdat.arrygridsourintp, fill_value=0.)
    gdat.brgtintp = gdat.brgtintp.reshape((gdat.numbsidegridcomp[0], gdat.numbsidegridcomp[0]))
    print('gdat.brgtintp')
    summgene(gdat.brgtintp)

    if gdat.booldiag:
        if (gdat.brgtintp < 0).any():
            raise Exception('')
    
    flux = gdat.brgtintp# * gdat.magn
    
    print('flux')
    summgene(flux)

    if gdat.booldiag:

        if (not np.isfinite(flux).all() or (flux == 0.).all()):
            print('')
            print('')
            print('')
            print('')
            print('')
            raise Exception('')

    return flux


def funcanomderi(anomecce, ecce):
    '''
    Derivative of g(E) as defined in Eq 42 in Murray and Correia in Exoplanets (ed Sara Seager)
    '''
    
    funcanomderi = 1. - ecce * np.cos(anomecce)
    
    return funcanomderi


def func_anom(anommean, anomecce, ecce):
    '''
    g(E) as defined in Eq 42 in Murray and Correia in Exoplanets (ed Sara Seager)
    '''
    
    funcanom = anomecce - ecce * np.sin(anomecce) - anommean
    
    return funcanom


def retr_anomdist(phas, smax, ecce, booldiag=False):
    '''
    Calculate the mean, eccentric, true anomaly, and distance-to-planet from phase, orbits's semi-major axis and eccentricity
    '''
    
    ecce = 0.

    # Mean anomaly
    anommean = 2. * np.pi * phas
  
    # eccentric anomaly
    anomecce = anommean
    errr = abs(func_anom(anommean, anomecce, ecce))
    
    # temp -- this part is not well tested because it is not entering the while loop
    while errr > 1e-10:
        funcanom = func_anom(anommean, anomecce, ecce)
        funcanomderi = funcanomderi(anomecce, ecce)
        
        # trial eccentric anomaly
        anomecce = anomecce - funcanom / funcanomderi
        
        # g(E) at the trial eccentric anomaly
        errr = abs(func_anom(anommean, anomecce, ecce))
        
    # distance from the  star
    dist = smax * (1. - ecce * np.cos(anomecce))
    
    # true anomaly
    anomtrueinit = np.arccos((np.cos(anomecce) - ecce) / (1. - ecce * np.cos(anomecce)))
    if anomecce > np.pi:
        anomtrue = anomtrueinit + 2. * abs(np.pi - anomtrueinit)
    elif anomecce < 0.:
        anomtrue = -anomtrueinit
    else:
        anomtrue = anomtrueinit

    if booldiag:
        if ecce < 0.01:
            if abs(2. * np.pi * phas - anomtrue) > 0.01:
                print('')
                print('')
                print('')
                print('ecce')
                print(ecce)
                print('phas')
                print(phas)
                print('anommean')
                print(anommean)
                print('anomecce')
                print(anomecce)
                print('anomtrueinit')
                print(anomtrueinit)
                print('anomtrue')
                print(anomtrue)
                raise Exception('anomtrue calculation is anomalous.')

    return anommean, anomecce, anomtrue, dist


def retr_posifromphas_efes(gdat, j, t, phas):
    '''
    Calculate body positions from phase
    '''
    
    anommean, anomecce, anomtrue, dist = retr_anomdist(phas, gdat.smaxcomp[j], gdat.eccecomp[j], booldiag=gdat.booldiag)
    
    # initial position in a frame where the the orbit lies in the x-y plane and the semi-major axis is aligned with the x-axis
    xposinit = dist * np.cos(anomtrue)
    yposinit = dist * np.sin(anomtrue)
    zposinit = 0.

    # find the position in the observer's frame (x, y, z) by performing three Euler rotations about Z, X, and Z axis on the position in the original frame (x_0, y_0, 0)
    # where the three rotations correspond to the the three orbital elements: argument of periapse (arpacomp), inclination (inclcomp), and longitude of the ascending node (loancomp)
    # pi / 2 terms shift the definition of anomtrue by 90 degrees to bring true anomy == 0 (and phase == 0) to inferior conjunction
    xpos = dist * (np.cos(gdat.loancomp[j]) * np.cos(gdat.arpacomp[j] + anomtrue + np.pi / 2) - \
                   np.sin(gdat.loancomp[j]) * np.sin(gdat.arpacomp[j] + anomtrue + np.pi / 2) * np.cos(np.pi / 180. * gdat.inclcomp[j]))
    ypos = dist * (np.sin(gdat.loancomp[j]) * np.cos(gdat.arpacomp[j] + anomtrue + np.pi / 2) + \
                   np.cos(gdat.loancomp[j]) * np.sin(gdat.arpacomp[j] + anomtrue + np.pi / 2) * np.cos(np.pi / 180. * gdat.inclcomp[j]))
    zpos = dist * np.sin(gdat.arpacomp[j] + anomtrue + np.pi / 2) * np.sin(np.pi / 180. * gdat.inclcomp[j])
    
    if gdat.booldiag:
        if not np.isfinite(xpos) or not np.isfinite(ypos) or not np.isfinite(zpos):
            print('')
            print('')
            print('')
            print('gdat.listphaseval[j][t]')
            print(gdat.listphaseval[j][t])
            print('phas')
            print(phas)
            print('gdat.eccecomp[j]')
            print(gdat.eccecomp[j])
            print('gdat.smaxcomp[j]')
            print(gdat.smaxcomp[j])
            print('gdat.loancomp[j]')
            print(gdat.loancomp[j])
            print('anommean')
            print(anommean)
            print('anomecce')
            print(anomecce)
            print('anomtrue')
            print(anomtrue)
            print('dist')
            print(dist)
            print('gdat.arpacomp')
            print(gdat.arpacomp)
            print('gdat.inclcomp')
            print(gdat.inclcomp)
            print('gdat.loancomp')
            print(gdat.loancomp)
            print('xpos')
            print(xpos)
            print('ypos')
            print(ypos)
            print('zpos')
            print(zpos)
            raise Exception('Transformed coordinates are not finite')
    
    return xpos, ypos, zpos, anommean, anomecce, anomtrue
    

def retr_boolevalflux(gdat, j, typecoor):

    if gdat.typesyst == 'PlanetarySystemEmittingCompanion':
        
        boolevalflux = True

    else:
        # this is neeed because the pre-ingress is undersampled
        factfudg = 1.2

        if typecoor == 'comp':
            if gdat.typesyst.startswith('PlanetarySystemWithRings') and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + 30. * gdat.rratcomp[j]):
                boolevalflux = True
            elif gdat.boolsystpsys and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + factfudg * gdat.rratcomp[j]):
                boolevalflux = True
            elif gdat.typesyst == 'CompactObjectStellarCompanion' and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + gdat.wdthslen[j]):
                boolevalflux = True
            else:
                boolevalflux = False
        elif typecoor == 'star':
            if gdat.typesyst.startswith('PlanetarySystemWithRings') and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + 3. * gdat.rratcomp[j]):
                boolevalflux = True
            elif gdat.boolsystpsys and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + factfudg * gdat.rratcomp[j]):
                boolevalflux = True
            elif gdat.typesyst == 'CompactObjectStellarCompanion' and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + gdat.wdthslen[j]):
                boolevalflux = True
            else:
                boolevalflux = False
        else:
            raise Exception('')
    
    if gdat.typeverb > 1:
        print('boolevalflux')
        print(boolevalflux)

    return boolevalflux


def proc_phaseval(gdat, j, t):
    '''
    Compute the system brightness at index t of the evaluation time axis using the companion grid for companion j
    '''

    xpos, ypos, zpos, anommean, anomecce, anomtrue = retr_posifromphas_efes(gdat, j, t, gdat.listphaseval[j][t])
    
    # position of the primary in the grid of the companion j
    gdat.xposstargridcomp[j] = -xpos
    gdat.yposstargridcomp[j] = -ypos
    gdat.zposstargridcomp[j] = -zpos
    
    gdat.listposiprimgridcompeval[j][t, 0] = gdat.xposstargridcomp[j]
    gdat.listposiprimgridcompeval[j][t, 1] = gdat.yposstargridcomp[j]
    gdat.listposiprimgridcompeval[j][t, 2] = gdat.zposstargridcomp[j]

    # start with the brightness of the primary
    gdat.lumisysteval[j][t] = gdat.lumistarnocc
    
    # decide whether to evaluate brightness terms due to companions or moons
    gdat.boolevalflux = retr_boolevalflux(gdat, j, 'comp')
    
    if gdat.boolevalflux:
        
        # distance to the primary in the grid of the companion j
        gdat.distfromprimgridcomp[j] = np.sqrt((gdat.xposgridcomp[j] - gdat.xposstargridcomp[j])**2 + \
                                        (gdat.yposgridcomp[j] - gdat.yposstargridcomp[j])**2)
        
        # Booleans indicating whether the points in the grid of the companion j, are inside the primary
        gdat.boolgridcompinsdprim = gdat.distfromprimgridcomp[j] < 1.
        
        if gdat.boolsystpsys:
            
            if abs(gdat.listphaseval[j][t]) < 0.25 or (gdat.boolmakeimaglfov or gdat.boolmakeanim):
                
                # Booleans indicating whether companion grid points are inside the star and occulted
                gdat.boolgridcompstaroccu = gdat.boolgridcompinsdprim & gdat.boolgridcompinsdcomp[j]
                
                # stellar flux occulted
                deltlumi = -retr_lumistartran(gdat, 'comp', gdat.boolgridcompstaroccu, j)
                
                if gdat.booldiag:
                    if gdat.typesyst == 'PlanetarySystem':
                        if deltlumi < -3. * gdat.lumistarnocc * np.amax(gdat.rratcomp)**2:
                            print('')
                            print('')
                            print('')
                            raise Exception('')
                
                if abs(gdat.listphaseval[j][t]) < 0.25:
                    gdat.lumisysteval[j][t] += deltlumi
                
        # brightness of the companion
        if gdat.typebrgtcomp != 'dark':
            
            if abs(gdat.listphaseval[j][t]) < 0.25:
                gdat.indxplannoccgridcomp[j] = gdat.indxplangridcomp[j]
            else:

                # Booleans indicating the region outside the star in the companion grid
                gdat.booloutsstargridcomp = retr_boolgridouts(gdat, j, 'comp', typeoccu='star')
                
                # Booleans indicating the planetary region region outside the star in the companion grid
                gdat.boolplannoccgridcomp = gdat.booloutsstargridcomp & gdat.boolplangridcomp[j]

                gdat.indxplannoccgridcomp[j] = np.where(gdat.boolplannoccgridcomp)
            
            # calculate the brightness of the planet
            # cosine of gamma
            cosg = np.sqrt(1. - gdat.distfromcompgridcomp[j][gdat.indxplannoccgridcomp[j]]**2)
            
            ## brightness on the companion before limb-darkening
            if gdat.typebrgtcomp == 'isot':
                brgtraww = gdat.ratibrgtcomp# * np.ones_like(gdat.xposstargridcomp[j])[gdat.indxplannoccgridcomp[j]]
            else:
            
                # transform to planet coordinate
                xposgridsphr = gdat.xposstargridcomp[j]
                yposgridsphr = gdat.zposstargridcomp[j]
                zposgridsphr = gdat.yposstargridcomp[j]
                
                # find spherical coordinates in the planet coordinate
                thet = -0.5 * np.pi + np.arccos(zposgridsphr / np.sqrt(xposgridsphr**2 + yposgridsphr**2 + zposgridsphr**2))
                phii = 0.5 * np.pi - np.arctan2(yposgridsphr, xposgridsphr)
                
                if gdat.booldiag:
                    if gdat.ratibrgtcomp is None:
                        print('')
                        print('')
                        print('')
                        print('gdat.typesyst')
                        print(gdat.typesyst)
                        print('gdat.typebrgtcomp')
                        print(gdat.typebrgtcomp)
                        print('gdat.ratibrgtcomp')
                        print(gdat.ratibrgtcomp)
                        print('gdat.latigridsphr')
                        print(gdat.latigridsphr)
                        print('thet')
                        print(thet)
                        raise Exception('gdat.ratibrgtcomp is None')
                
                brgtraww = gdat.ratibrgtcomp * np.cos(thet + gdat.latigridsphr[j][gdat.indxplannoccgridcomp[j]])
            
                # longitudes of the unocculted pixels of the revolved (and tidally-locked) planet
                gdat.longgridsphrrota = phii + gdat.longgridsphr[j][gdat.indxplannoccgridcomp[j]]

                if gdat.typebrgtcomp == 'heated_rdis':
                    brgtraww *= (0.55 + 0.45 * np.sin(gdat.longgridsphrrota + np.pi * gdat.offsphascomp[j] / 180.))
                elif gdat.typebrgtcomp == 'heated_sliced':
                    indxslic = (gdat.numbslic * ((gdat.longgridsphrrota % (2. * np.pi)) / np.pi / 2.)).astype(int)
                    brgtraww *= gdat.brgtsliccomp[indxslic]
            
            gdat.brgtcomp = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, brgtraww=brgtraww, typelmdk=gdat.typelmdk) * gdat.areapixlcomp
            
            gdat.lumisysteval[j][t] += np.sum(gdat.brgtcomp)
                
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            # distance from the points in the source grid to the star
            gdat.diststargridsour = np.sqrt((gdat.xposgridsour - gdat.xposstargridcomp[j])**2 + (gdat.yposgridsour - gdat.yposstargridcomp[j])**2)
            
            # Booleans indicating whether source grid points are inside the star
            gdat.boolstargridsour = gdat.diststargridsour < 1.
            
            if gdat.booldiag:
                if not gdat.boolstargridsour.any():
                    print('gdat.xposgridsour')
                    summgene(gdat.xposgridsour)
                    print('gdat.yposgridsour')
                    summgene(gdat.yposgridsour)
                    print('gdat.xposstargridcomp[j]')
                    print(gdat.xposstargridcomp[j])
                    print('gdat.yposstargridcomp[j]')
                    print(gdat.yposstargridcomp[j])
                    print('gdat.boolstargridsour')
                    summgene(gdat.boolstargridsour)
                    print('')

            gdat.indxgridsourstar = np.where(gdat.boolstargridsour)
                
            # calculate the lensed brightness inside the companion grid
            gdat.brgtlens = retr_brgtlens(gdat, t)
            
            # calculate the brightness inside the companion grid
            gdat.indxgridcompinsdprim = np.where(gdat.boolgridcompinsdprim)
            gdat.lumistarplan = np.sum(retr_lumistartran(gdat, 'comp', gdat.indxgridcompinsdprim, j))
            
            print('gdat.lumistarplan')
            print(gdat.lumistarplan)
            print('gdat.lumistarnocc - gdat.lumistarplan')
            print(gdat.lumistarnocc - gdat.lumistarplan)

            fluxtotlfram = np.sum(gdat.brgtlens) + gdat.lumistarnocc - gdat.lumistarplan
            
            print('fluxtotlfram')
            print(fluxtotlfram)

            if booldiag:
                if fluxtotlfram / gdat.lumistarnocc < 0.5 or fluxtotlfram <= 0.:
                    print('')
                    print('')
                    print('')
                    print('gdat.brgtlens')
                    summgene(gdat.brgtlens)
                    print('fluxtotlfram')
                    print(fluxtotlfram)
                    print('gdat.lumistarnocc')
                    print(gdat.lumistarnocc)
                    raise Exception('fluxtotlfram / gdat.lumistarnocc < 0.5 or fluxtotlfram <= 0.')
                if fluxtotlfram == 0.:
                    print('')
                    print('')
                    print('')
                    print('fluxtotlfram')
                    print(fluxtotlfram)
                    raise Exception('fluxtotlfram == 0.')
            
            gdat.lumisysteval[j][t] += fluxtotlfram
        
        if gdat.typeverb > 1:
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
        
        if gdat.booldiag:
            if not np.isfinite(gdat.lumisysteval[j][t]):
                print('')
                print('')
                print('')
                print('jt')
                print(j, t)
                print('gdat.boolevalflux')
                print(gdat.boolevalflux)
                print('gdat.lumisysteval[j][t]')
                summgene(gdat.lumisysteval[j][t])
                raise Exception('not np.isfinite(gdat.lumisysteval[j][t])')
        
    if gdat.booldiag:
        if not isinstance(gdat.lumisysteval[j][t], float):
            print('gdat.lumisysteval[j][t]')
            print(gdat.lumisysteval[j][t])
            raise Exception('len(gdat.lumisysteval[j][t]) == 0 or np.isscalar(gdat.lumisysteval[j][t])')
    
        if gdat.typesyst == 'PlanetarySystem':
            if gdat.lumisysteval[j][t] > gdat.lumistarnocc or gdat.lumisysteval[j][t] < gdat.lumistarnocc - 3 * gdat.lumistarnocc * np.amax(gdat.rratcomp)**2:
                print('')
                print('')
                print('')
                print('gdat.lumisysteval[j][t]')
                print(gdat.lumisysteval[j][t])
                print('gdat.lumistarnocc')
                print(gdat.lumistarnocc)
                raise Exception('gdat.lumisysteval[j][t] > gdat.lumistarnocc or gdat.lumisysteval[j][t] < gdat.lumistarnocc - 3 * gdat.lumistarnocc * np.amax(gdat.rratcomp)**2')


def eval_modl( \
              # times in days at which to evaluate the relative flux
              time, \
              
              # type of the model system
              ## 'PlanetarySystemEmittingCompanion': planetary system with phase curve
              typesyst, \
              
              # parametrization of orbital parameters relevant to transits
              ## orbital periods of the companions
              pericomp, \
              
              ## mid-transit epochs of the companions
              epocmtracomp, \
              
              ## sum of stellar and companion radius
              rsmacomp=None, \
              
              ## cosine of the orbital inclination (alternative to inclcomp)
              cosicomp=None, \
              
              ## sine of 
              sinwcomp=0., \
              
              ## cosine of 
              coswcomp=0., \
              
              # orbital elements
              ## semi-major axis of the orbit
              smaxcomp=None, \

              ## eccentricity of the orbit
              eccecomp=None, \
              
              ## argument of periapse
              arpacomp=None, \

              ## orbital inclination
              inclcomp=None, \
              
              ## longitude of the ascending node
              loancomp=None, \

              ## radius ratio for the companions
              rratcomp=None, \
              
              ## radii of the companions
              radicomp=None, \
              
              ## mass of the companions
              masscomp=None, \
              
              ## type of the brightness of the companions
              ### it is only functional if typesyst is 'PlanetarySystemEmittingCompanion'
              ### 'dark': companion completely dark
              ### 'heated_rdis': companion is an externally heated body with heat redistribution (efficiency and phase offset)
              ### 'heated_sliced': companion has a heat distribution determined by the input temperatures of longitudinal slices
              ### 'isot': companion is an internally heated, isothermal body
              typebrgtcomp=None, \
              
              ## Boolean flag to include occultor crossings
              boolmodlplancros=False, \

              ## phase offset for the sinusoidal model
              offsphascomp=None, \

              ## temperature of the slices of the companion
              tmptsliccomp=None, \

              ## ratio of substellar brightness on the planet to that on the surface of the star
              ratibrgtcomp=None, \
              
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
              
              # type of model for lensing
              ## 'phdy': calculated photodynamically via ray tracing
              ## 'Gaussian': Gaussian centered at conjuction
              typemodllens='phdy', \

              # spots
              ## rotation period at the equator
              perispot=None, \

              # type of limb-darkening
              typelmdk='quadkipp', \
              ## limb-darkening coefficient(s)
              coeflmdk=None, \
              
              # radius of the host star in Solar radius
              radistar=None, \
              
              # mass of the host star in Solar mass
              massstar=None, \
            
              # String indicating the type of estimation
              ## 'intg': integrate the surface brightness
              ## 'simpboxx': constant drop in-transit
              ## 'simptrap': trapezoidal drop in-transit
              typecalc='intg', \

              # Boolean flag to evaluate by interpolating a single transit
              boolintp=None, \

              # type of grid used to sample the brightness
              ## 'star': centered on the star
              ## 'comp': centered on the companion
              ## 'bary': quisi-inertial frame at the barycenter
              typecoor=None, \
             
              # a string indicating the type of normalization
              ## 'none': no detrending
              ## 'medi': by median
              ## 'nocc': by unocculted stellar brightness
              ## 'edgeleft': left edge
              ## 'maxm': maximum value
              typenorm='nocc', \
              
              # Boolean flag to use tqdm to report the percentage of completion
              booltqdm=False, \

              # path for visuals
              pathvisu=None, \
              
              # label for the unit of time
              lablunittime='BJD', \

              # Boolean flag to make an animation
              boolmakeanim=False, \
              
              # Boolean flag to show the light curve on the animation
              boolshowlcuranim=True, \

              # title of the animation
              strgtitl=None, \

              # string to be appended to the file name of the visuals
              strgextn='', \
              
              # resolution controls
              ## phase interval between phase samples during inegress and egress
              diffphasineg=None, \

              ## phase interval between phase samples during full transits and eclipses
              diffphasintr=None, \

              ## phase interval between phase samples elsewhere
              diffphaspcur=None, \

              ## the spatial resolution of the companion grid
              diffgridcomp=None, \
              
              # Boolean flag to check if the computation can be accelerated
              boolfast=True, \
              
              # the maximum factor by which the primary will get tidally deformed
              maxmfactellp=None, \

              # Boolean flag to calculate the light curve of the system
              boolcalclcur=True, \
              
              # Boolean flag to additionally calculate the angular distance between the companions
              boolcalcdistcomp=False, \
              
              # Boolean flag to plot the minimum angular distance between the companions
              boolplotdistcomp=False, \

              ## Boolean flag to make a large-FOV image
              boolmakeimaglfov=False, \
              
              ## type of coloring for the large-FOV image
              ### 'real': dark background, body colors consistent with temperature and reflection
              ### 'cart': white background, pastel-colored stars, companions, and moons
              typecolrimaglfov='real', \
              
              ## type of the enhacements for the large-FOV image
              ### 'none': no coloring, tail, or label
              ### 'colr': no tail or label, colored
              ### 'tail': with tail
              ### 'taillabl': with tail and label
              typemrkrimaglfov='taillabl', \

              ## Boolean flag to add Mercury in the large-FOV image
              boolinclmercimaglfov=False, \
                
              # type of plot background
              typeplotback='black', \

              # type of light curve plot
              ## 'inst': inset
              ## 'lowr': lower panel
              typeplotlcurposi='inst', \
              
              # type of the limit of the light curve visualized
              ## 'wind': window
              ## 'tran': around-transit
              typeplotlcurlimt='wind', \
              
              # type of language for text
              typelang='English', \

              # Boolean flag to ignore any existing plot and overwrite
              boolwritover=False, \
              
              # Boolean flag to diagnose
              booldiag=True, \
              
              # Boolean flag to diagnose by raising an exception when the model output is all 1s
              booldiagtran=False, \
              
              ## file type of the plot
              typefileplot='png', \

              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
             ):
    '''
    Calculate the flux of a system of potentially lensing and transiting stars, planets, and compact objects.
    When limb-darkening and/or moons are turned on, the result is interpolated based on star-to-companion radius, companion-to-moon radius.
    '''
    timeinit = modutime.time()
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    dictinpt = dict(locals())
    for attr, valu in dictinpt.items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # output dictionary
    dictefes = dict()
    dictefes['dictinpt'] = dictinpt

    # Boolean flag to profile (track) time budget
    gdat.boolproftime = True
    
    # Boolean flag to halt execution if the time budget is excessive (only to be used during diagnosis)
    gdat.boolstopproftime = True
    
    if typeverb > 1:
        print('Estimating the light curve via eval_modl()...')
        
    if isinstance(gdat.pericomp, list):
        gdat.pericomp = np.array(gdat.pericomp)

    if isinstance(gdat.epocmtracomp, list):
        gdat.epocmtracomp = np.array(gdat.epocmtracomp)

    if isinstance(gdat.radicomp, list):
        gdat.radicomp = np.array(gdat.radicomp)

    if isinstance(gdat.rsmacomp, list):
        gdat.rsmacomp = np.array(gdat.rsmacomp)

    if isinstance(gdat.rratcomp, list):
        gdat.rratcomp = np.array(gdat.rratcomp)
    
    if isinstance(gdat.masscomp, list):
        gdat.masscomp = np.array(gdat.masscomp)

    if isinstance(gdat.inclcomp, list):
        gdat.inclcomp = np.array(gdat.inclcomp)

    if isinstance(gdat.offsphascomp, list):
        gdat.offsphascomp = np.array(gdat.offsphascomp)

    if isinstance(gdat.cosicomp, list):
        gdat.cosicomp = np.array(gdat.cosicomp)
    
    if isinstance(gdat.eccecomp, list):
        gdat.eccecomp = np.array(gdat.eccecomp)

    if isinstance(gdat.sinwcomp, list):
        gdat.sinwcomp = np.array(gdat.sinwcomp)
    
    gdat.numbcomp = gdat.pericomp.size
    gdat.indxcomp = np.arange(gdat.numbcomp)
    
    if gdat.loancomp is None:
        gdat.loancomp = np.full(gdat.numbcomp, fill_value=np.pi)
    
    if gdat.arpacomp is None:
        gdat.arpacomp = np.zeros(gdat.numbcomp)
    
    if gdat.cosicomp is None and gdat.inclcomp is None:
        raise Exception('gdat.cosicomp and gdat.inclcomp should not be None at the same time.')
    elif gdat.cosicomp is not None and gdat.inclcomp is None:
        gdat.intgcompflip = np.random.randint(2, size=gdat.numbcomp) - 1
    
    if gdat.typesyst == 'PlanetarySystemEmittingCompanion':
        
        if gdat.typebrgtcomp == 'dark':
            raise Exception('gdat.typebrgtcomp cannot be dark when typesyst is PlanetarySystemEmittingCompanion')
        
        if gdat.offsphascomp is None:
            gdat.offsphascomp = np.zeros(gdat.numbcomp)

    if gdat.typebrgtcomp is None:
        if gdat.typesyst == 'PlanetarySystemEmittingCompanion':
            gdat.typebrgtcomp = 'heated_rdis'
        else:
            gdat.typebrgtcomp = 'dark'

    if gdat.typecoor is None:
        if gdat.numbcomp == 1 and gdat.typesyst != 'PlanetarySystemWithMoons' or not gdat.boolmodlplancros:
            gdat.typecoor = 'comp'
        else:
            gdat.typecoor = 'star'

    gdat.boolsystpsys = gdat.typesyst.startswith('PlanetarySystem')
    
    if gdat.typeplotback == 'black':
        plt.style.use('dark_background')
        
    if gdat.boolcalcdistcomp:
        dictefes['distcomp'] = np.empty((gdat.numbcomp, gdat.numbcomp))

    # check inputs
    if gdat.booldiag:
        if (gdat.boolmakeanim or gdat.boolmakeimaglfov) and gdat.pathvisu is None:
            
            raise Exception('')

        if gdat.diffphasineg is not None and not isinstance(gdat.diffphasineg, float):
            print('gdat.diffphasineg')
            print(gdat.diffphasineg)
            raise Exception('')
        
        if gdat.time.size == 0:
            print('')
            print('')
            print('')
            raise Exception('gdat.time.size == 0')

        if gdat.diffphaspcur is not None and not isinstance(gdat.diffphaspcur, float):
            print('gdat.diffphaspcur')
            print(gdat.diffphaspcur)
            raise Exception('')

        if np.isscalar(gdat.rratcomp):
            raise Exception('')
        
        
        if gdat.typesyst.startswith('PlanetarySystemEmittingCompanion') and (gdat.rratcomp is None or not np.isfinite(gdat.rratcomp).all()):
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.rratcomp')
            summgene(gdat.rratcomp)
            raise Exception('rratcomp is None or at least one element in it is infinite despite the system having planetary companions.')
        
        if gdat.rratcomp is not None and (gdat.rratcomp < 0).any():
            print('')
            print('')
            print('')
            print('gdat.rratcomp')
            summgene(gdat.rratcomp)
            raise Exception('rratcomp is outside the physical limits.')
        
        if gdat.boolintp and gdat.typecoor == 'star':
            raise Exception('')

        if gdat.boolintp and gdat.numbcomp > 1:
            raise Exception('')

        if gdat.boolsystpsys and not ((gdat.radistar is not None and gdat.radicomp is not None) or gdat.rratcomp is not None):
            print('')
            print('')
            print('')
            print('gdat.radistar')
            print(gdat.radistar)
            print('gdat.radicomp')
            print(gdat.radicomp)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            raise Exception('gdat.boolsystpsys and not ((gdat.radistar is not None and gdat.radicomp is not None) or gdat.rratcomp is not None)')
        
        if gdat.rsmacomp is None and not (gdat.masscomp is not None and gdat.massstar is not None):
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.rsmacomp')
            print(gdat.rsmacomp)
            print('gdat.masscomp')
            print(gdat.masscomp)
            print('gdat.massstar')
            print(gdat.massstar)
            raise Exception('')

        if gdat.typesyst == 'CompactObjectStellarCompanion':
            
            if gdat.typecoor == 'star':
                raise Exception('')
            if gdat.massstar is None or gdat.masscomp is None:
                raise Exception('')
        else:
            if gdat.rratcomp.ndim == 2 and gdat.coeflmdk.ndim == 2 and gdat.rratcomp.shape[-1] != gdat.rratcomp.shape[-1]:
                print('')
                print('')
                print('')
                print('gdat.rratcomp')
                summgene(gdat.rratcomp)
                print('gdat.coeflmdk')
                summgene(gdat.coeflmdk)
                print('rratcomp and coeflmdk should either be one or two dimensional and if two dimensional, \
                                                        their energy axis (second one) should have the same length.')
                raise Exception('')

        # inclination
        if gdat.inclcomp is not None and gdat.cosicomp is not None:
            raise Exception('')
        
        if gdat.typesyst != 'CompactObjectStellarCompanion' and (gdat.rsmacomp is None or not np.isfinite(gdat.rsmacomp).all()):
            print('')
            print('')
            print('')
            print('gdat.rsmacomp')
            print(gdat.rsmacomp)
            raise Exception('gdat.rsmacomp is None or not np.isfinite(gdat.rsmacomp).all()')
        
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            if gdat.massstar is None:
                raise Exception('')
        
            if gdat.masscomp is None:
                raise Exception('')
    
    if gdat.boolmakeimaglfov:
        gdat.listcolrcomp = ['b', 'r', 'g', 'orange', 'cyan', 'magenta', 'olive', 'yellow']
    
    if gdat.boolfast and gdat.boolsystpsys and gdat.rratcomp.ndim == 2:
        print('np.std(gdat.rratcomp)')
        print(np.std(gdat.rratcomp))
        print('gdat.rratcomp')
        summgene(gdat.rratcomp)
        if np.std(gdat.rratcomp) < 0.05 and gdat.numbcomp == 1:
            gdat.boolrscllcur = True
            print('Rescaling the white light curve instead of calculating the light curve in each wavelength channel...')
            gdat.rratcompsave = np.copy(gdat.rratcomp)
            gdat.meanrratcomp = np.mean(gdat.rratcomp, 1)
            gdat.rratcomp = gdat.meanrratcomp
        else:
            print('Not rescaling the white light curve instead of calculating the light curve in each wavelength channel...')
            gdat.boolrscllcur = False
    
    # Boolean flag to return separate light curves for the companion and moon
    boolcompmoon = radimoon is not None
            
    if gdat.coeflmdk is None:
        # limb-darkening coefficient 
        if gdat.typelmdk == 'linr':
            # linear
            gdat.coeflmdk = 0.6
        if gdat.typelmdk == 'quad' or gdat.typelmdk == 'quadkipp':
            # quadratic
            gdat.coeflmdk=[0.4, 0.25]
        if gdat.typelmdk == 'nlin':
            # nonlinear
            gdat.coeflmdk=[0.2, 0.2, 0.2]
    elif gdat.typelmdk == 'quadkipp':
        # convert the Kipping basis to quadratic
        gdat.coeflmdk = nicomedia.retr_coeflmdkfromkipp(gdat.coeflmdk[0], gdat.coeflmdk[1])
    
    if gdat.inclcomp is not None:
        gdat.cosicomp = np.cos(gdat.inclcomp * np.pi / 180.)
    elif gdat.cosicomp is not None:
        gdat.inclcomp = np.arccos(gdat.cosicomp) * 180. / np.pi # [deg]
    
    if typeverb > 1:
        print('gdat.typesyst')
        print(gdat.typesyst)
        print('gdat.typecoor')
        print(gdat.typecoor)
        print('gdat.boolintp')
        print(gdat.boolintp)
    
    if gdat.typesyst == 'CompactObjectStellarCompanion':
        if typeverb > 1:
            print('typemodllens')
            print(typemodllens)

        if gdat.radicomp is not None:
            if typeverb > 0:
                print('Warning from ephesos! A radius was provided for the compact object...')
        else:
            gdat.radicomp = np.array([0.])

    if gdat.perimoon is not None:
        gdat.indxcomp = np.arange(gdat.numbcomp)
        numbmoon = np.empty(gdat.numbcomp, dtype=int)
        for j in gdat.indxcomp:
            numbmoon[j] = len(gdat.perimoon[j])
        indxmoon = [np.arange(numbmoon[j]) for j in gdat.indxcomp]
    
    if typeverb > 1:
        print('gdat.pericomp')
        print(gdat.pericomp)
        print('gdat.epocmtracomp')
        print(gdat.epocmtracomp)
    
    if gdat.booldiag:
        if gdat.time.ndim != 1:
            print('gdat.time')
            summgene(gdat.time)
            raise Exception('')
    
    minmtime = np.amin(gdat.time)
    maxmtime = np.amax(gdat.time)
    gdat.numbtime = gdat.time.size
    gdat.indxtime = np.arange(gdat.numbtime)

    gdat.dictvarborbt = dict()
    gdat.dictvarborbt['posicompgridprim'] = np.empty((gdat.numbtime, gdat.numbcomp, 3))
    gdat.dictvarborbt['anomtrue'] = np.empty((gdat.numbtime, gdat.numbcomp))

    gdat.listposiprimgridcomp = np.empty((gdat.numbtime, gdat.numbcomp, 3))
    
    gdat.dictfact = tdpy.retr_factconv()
    
    if gdat.pathvisu is not None:
        
        # path for animations
        gdat.dictturk = tdpy.retr_dictturk()

        if gdat.boolmakeanim:
            gdat.indxframthis = 0

            gdat.pathgiff = dict()
            gdat.cmndmakeanim = dict()
            gdat.cmnddeleimag = dict()
        
        if gdat.boolmakeanim or gdat.boolmakeimaglfov:
            gdat.listnamevarbfram = ['flux']
            if gdat.typesyst == 'CompactObjectStellarCompanion':
                gdat.listnamevarbfram += ['posifrstphotlens', 'posisecophotlens', 'fluxfrstlens', 'fluxsecolens']#, 'brgtgridsour' 'cntsfrstlens', 'cntssecolens']
    
        if gdat.strgextn != '':
            gdat.strgextn = '_' + gdat.strgextn
    
    if gdat.rratcomp is not None:
        gdat.radicomp = gdat.rratcomp
        if gdat.radistar is not None:
            raise Exception('')
    else:
        gdat.rratcomp = gdat.radicomp / gdat.radistar / gdat.dictfact['rsre']
        if typeverb > 1:
            print('gdat.radistar')
            print(gdat.radistar)
    
    #gdat.radistar = 1.
    
    if typeverb > 1:
        print('gdat.coeflmdk')
        print(gdat.coeflmdk)
        print('gdat.rsmacomp')
        print(gdat.rsmacomp)
        print('gdat.cosicomp')
        print(gdat.cosicomp)
    
    if gdat.rsmacomp is not None:
        gdat.smaxcomp = (1. + gdat.rratcomp) / gdat.rsmacomp
    
    if gdat.typesyst == 'PlanetarySystemEmittingCompanion':
    
        if gdat.maxmfactellp is None:
            gdat.maxmfactellp = 1.2
        
        if gdat.typebrgtcomp == 'heated_rdis' or gdat.typebrgtcomp == 'heated_sliced':
            
            if gdat.ratibrgtcomp is not None:
                print('')
                print('')
                print('')
                print('gdat.ratibrgtcomp')
                print(gdat.ratibrgtcomp)
                raise Exception('A brightness ratio (typebrgtcomp) cannot be provided when the companion is passively heated, which already determines typebrgtcomp.')
            gdat.ratibrgtcomp = (1. / gdat.smaxcomp)**2
        
        elif gdat.typebrgtcomp == 'isot':
            if gdat.ratibrgtcomp is None:
                gdat.ratibrgtcomp = 1.
        else:
            raise Exception('')
        
    if gdat.masscomp is not None and gdat.massstar is not None:
        
        if gdat.typesyst == 'PlanetarySystemEmittingCompanion' or gdat.typesyst == 'PlanetarySystemEmittingCompanion':
            gdat.masscompsolr = gdat.masscomp / gdat.dictfact['msme']
        elif gdat.typesyst == 'CompactObjectStellarCompanion' or gdat.typesyst == 'StellarBinary':
            gdat.masscompsolr = gdat.masscomp
        else:
            print('gdat.typesyst')
            print(gdat.typesyst)
            raise Exception('')

        if gdat.typeverb > 1:
            print('gdat.masscompsolr')
            print(gdat.masscompsolr)
            print('gdat.massstar')
            print(gdat.massstar)
        
        ## total mass of the system
        gdat.masstotl = gdat.massstar + np.sum(gdat.masscompsolr) # [M_S]
        
        ## semi-major axis
        gdat.smaxcompasun = nicomedia.retr_smaxkepl(gdat.pericomp, gdat.masstotl) # [AU]
        
        gdat.smaxcomp = gdat.smaxcompasun * gdat.dictfact['aurs'] / gdat.radistar # [R_*]

        if gdat.perimoon is not None:
            smaxmoon = [[[] for jj in indxmoon[j]] for j in gdat.indxcomp]
            for j in gdat.indxcomp:
                smaxmoon[j] = nicomedia.retr_smaxkepl(gdat.perimoon[j], gdat.masscompsolr[j])
                
                for jj in indxmoon[j]:
                    if smaxmoon[j] <= gdat.radicomp[j] / gdat.dictfact['rsre'] / gdat.dictfact['aurs']:
                        print('smaxmoon[j] [AU]')
                        print(smaxmoon[j])
                        print('gdat.masscomp[j] [MES]')
                        print(gdat.masscomp[j])
                        print('gdat.masscompsolr[j]')
                        print(gdat.masscompsolr[j])
                        print('gdat.perimoon[j]')
                        print(gdat.perimoon[j])
                        print('gdat.radicomp[j]')
                        print(gdat.radicomp[j])
                        raise Exception('')
    
    if gdat.eccecomp is None:
        gdat.eccecomp = np.zeros(gdat.numbcomp)
    if gdat.sinwcomp is None:
        gdat.sinwcomp = np.zeros(gdat.numbcomp)
    
    if typeverb > 1:
        print('gdat.smaxcomp [R_star]')
        print(gdat.smaxcomp)
        print('gdat.indxcomp')
        print(gdat.indxcomp)
        if gdat.perimoon is not None:
            print('gdat.perimoon [days]')
            print(gdat.perimoon)
            print('radimoon [RE]')
            print(radimoon)
            print('smaxmoon [AU]')
            for smaxmoontemp in smaxmoon:
                print(smaxmoontemp)
    
    if gdat.boolintp is None:
        if gdat.perimoon is not None or gdat.numbcomp > 1 and gdat.boolmodlplancros or gdat.perispot is not None:
            if typeverb > 1:
                print('Either the model has moon, stellar spots, or multiple companions.')
                print('Will evaluate the model at each time (as opposed to interpolating phase curves)...')
            gdat.boolintp = False
        else:
            if typeverb > 1:
                print('The model only has a single companion. Will interpolate the phase curve (as opposed to evaluating the model at each time)...')
            gdat.boolintp = True

    if boolcompmoon and gdat.boolintp:
        raise Exception('')

    if gdat.rsmacomp is None:
        gdat.rsmacomp = (gdat.rratcomp + gdat.radistar) / gdat.smaxcomp

    if gdat.booldiag:
        if gdat.rsmacomp is None:
            print('')
            print('')
            print('')
            print('gdat.smaxcomp')
            print(gdat.smaxcomp)
            raise Exception(' gdat.rsmacomp is Non')

    gdat.duratrantotl = nicomedia.retr_duratrantotl(gdat.pericomp, gdat.rsmacomp, gdat.cosicomp, booldiag=gdat.booldiag) / 24.
    
    dictefes['duratrantotl'] = gdat.duratrantotl
        
    if gdat.typesyst == 'CompactObjectStellarCompanion':
        if gdat.typemodllens == 'gaus':
            gdat.dcyctrantotlhalf = gdat.smaxcomp / gdat.radistar / gdat.cosicomp
        if typeverb > 1:
            print('gdat.masscomp')
            print(gdat.masscomp)
            print('gdat.massstar')
            print(gdat.massstar)
        amplslen = chalcedon.retr_amplslen(gdat.pericomp, gdat.radistar, gdat.masscomp, gdat.massstar)
        dictefes['amplslen'] = amplslen
    
    if gdat.boolintp:
        timecomp = [[] for j in gdat.indxcomp]
    
        if typeverb > 1:
            print('gdat.duratrantotl [days]')
            print(gdat.duratrantotl)

        if gdat.boolsystpsys:
            if gdat.booldiag:
                if gdat.pericomp.size != gdat.rsmacomp.size:
                    print('gdat.rsmacomp')
                    summgene(gdat.rsmacomp)
                    print('gdat.pericomp')
                    summgene(gdat.pericomp)
                    print('gdat.rratcomp')
                    summgene(gdat.rratcomp)
                    print('gdat.cosicomp')
                    summgene(gdat.cosicomp)
                    print('gdat.typesyst')
                    print(gdat.typesyst)
                    raise Exception('')
                if gdat.rratcomp is None:
                    print('gdat.typesyst')
                    print(gdat.typesyst)
                    raise Exception('')
                if gdat.typesyst == 'StellarBinary' and (gdat.rratcomp == 0).any():
                    raise Exception('')
            
            if gdat.boolfast and gdat.rratcomp.ndim == 2 and gdat.boolrscllcur:
                rratcomp = gdat.meanrratcomp
            else:
                rratcomp = gdat.rratcomp
            gdat.duratranfull = nicomedia.retr_duratranfull(gdat.pericomp, gdat.rsmacomp, gdat.cosicomp, rratcomp) / 24.
            dictefes['duratranfull'] = gdat.duratranfull
        
        if typeverb > 1:
            print('gdat.duratranfull [days]')
            print(gdat.duratranfull)

        if gdat.diffphasineg is None:
            if typesyst == 'CompactObjectStellarCompanion':
                gdat.diffphasineg = 0.00005
            else:
                if gdat.numbcomp == 1 and gdat.rratcomp[0] > 0.5:
                    gdat.diffphasineg = 0.01
                else:
                    gdat.diffphasineg = 0.0003
        
        if gdat.diffphaspcur is None:
            gdat.diffphaspcur = 0.02
        
        if gdat.diffphasintr is None:
            gdat.diffphasintr = np.empty(gdat.numbcomp)
            for j in gdat.indxcomp:
                if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                    gdat.diffphasintr[j] = 0.0005
                else:
                    gdat.diffphasintr[j] = 0.0001
        
        if typeverb > 1:
            if np.isfinite(gdat.duratranfull):
                print('gdat.diffphasineg')
                print(gdat.diffphasineg)
            print('gdat.diffphasintr')
            print(gdat.diffphasintr)
            print('gdat.diffphaspcur')
            print(gdat.diffphaspcur)
    
    if gdat.typecoor == 'star' or gdat.boolmakeimaglfov:
        gdat.distfromcompgridstar = [[] for j in gdat.indxcomp]
    
    if gdat.typecoor == 'comp':
        # distance to the primary in the grid of the companion j
        gdat.distfromprimgridcomp = [[] for j in gdat.indxcomp]
    
    gdat.phascomp = [[] for j in gdat.indxcomp]
    if gdat.perimoon is not None:
        gdat.phasmoon = [[] for j in gdat.indxcomp]
    for j in gdat.indxcomp:
        
        if typeverb > 1:
            print('j')
            print(j)
        
        gdat.phascomp[j] = ((gdat.time - gdat.epocmtracomp[j]) / gdat.pericomp[j] + 0.25) % 1. - 0.25
        
        if gdat.perimoon is not None:
            gdat.phasmoon[j] = ((gdat.time - gdat.epocmtramoon[j]) / gdat.perimoon[j] + 0.25) % 1. - 0.25
        
        if gdat.booldiag:
            if np.isscalar(gdat.phascomp[j]):
                raise Exception('')
        
        if typeverb > 1:
            print('gdat.phascomp[j]')
            summgene(gdat.phascomp[j])
    
    if gdat.booldiag:
        if gdat.boolsystpsys and len(gdat.rratcomp) == 0:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            raise Exception('gdat.rratcomp is empty.')
    
    if gdat.boolcalclcur:
        if typecalc == 'simpboxx':
            for j in gdat.indxcomp:
                indxtimetran = np.where((gdat.phascomp[j] < gdat.duratrantotl[j] / gdat.pericomp[j]) | (gdat.phascomp[j] > 1. -  gdat.duratrantotl[j] / gdat.pericomp[j]))[0]
                rflxtranmodl = np.ones_like(gdat.phascomp)
                rflxtranmodl[indxtimetran] -= gdat.rratcomp**2
        elif len(rratcomp) == 0:
            rflxtranmodl = np.ones(gdat.numbtime)
        else:
            
            if gdat.boolmakeanim:
                gdat.diffgridstar = 1e-3
                if gdat.diffgridcomp is None:
                    gdat.diffgridcomp = 5e-4

            elif gdat.typesyst == 'CompactObjectStellarCompanion':
                
                gdat.radieins = chalcedon.retr_radieinssbin(gdat.masscomp, gdat.smaxcompasun) / gdat.radistar
                gdat.wdthslen = np.minimum(2. * gdat.radieins, np.ones_like(gdat.masscomp))
                
                if gdat.typeverb > 1:
                    print('gdat.masscomp')
                    print(gdat.masscomp)
                    print('gdat.smaxcomp')
                    print(gdat.smaxcomp)
                    print('gdat.smaxcompasun')
                    print(gdat.smaxcompasun)
                    print('gdat.radieins')
                    print(gdat.radieins)
                    print('gdat.wdthslen')
                    print(gdat.wdthslen)
                
                gdat.factresowdthslen = 0.01
                gdat.diffgridcomp = gdat.factresowdthslen * gdat.wdthslen[0]
                if gdat.diffgridcomp < 1e-3:
                    if gdat.typeverb > 0:
                        print('The grid resolution needed to resolve the Einstein radius is %g, which is too small. Limiting the grid resolution at 0.001.' % \
                                                                                                                                                            gdat.diffgrid)
                    gdat.diffgrid = 1e-3
                gdat.factosampsour = 0.1
                gdat.factsizesour = 1.
                
                if gdat.booldiag:
                    if gdat.numbcomp > 1:
                        raise Exception('')

            else:
            
                gdat.diffgridstar = 1e-3
                
                if gdat.booldiag:
                    if len(gdat.rratcomp) == 0:
                        print('')
                        print('')
                        print('')
                        print('gdat.typesyst')
                        print(gdat.typesyst)
                        print('gdat.rratcomp')
                        print(gdat.rratcomp)
                        raise Exception('len(gdat.rratcomp) == 0')

                if gdat.diffgridcomp is None:
                    gdat.diffgridcomp = 0.02
            
            if gdat.booldiag:
                if (gdat.rratcomp > 1).any():
                    print('At least one of the radius ratios is larger than unity.')
                    print('gdat.rratcomp')
                    print(gdat.rratcomp)
                
                if gdat.diffgridstar > 0.01:
                    print('')
                    print('')
                    print('')
                    print('Warning! The star grid resolution is too low.')
                    print('gdat.rratcomp')
                    print(gdat.rratcomp)
                    #raise Exception('')
                    
                if gdat.typecoor == 'star' and (gdat.diffgridstar > 0.1 or gdat.diffgridstar > 0.2 * np.amin(gdat.rratcomp)):
                    print('')
                    print('')
                    print('')
                    print('Warning! The star grid resolution is too low to resolve the smallest occultor.')
                    print('gdat.rratcomp')
                    print(gdat.rratcomp)
                    #raise Exception('')
                    
                if gdat.typecoor == 'comp' and gdat.diffgridcomp > 0.1:
                    print('')
                    print('')
                    print('')
                    print('Warning! The companion grid resolution is too low.')
                    print('gdat.diffgridcomp')
                    print(gdat.diffgridcomp)
                    #raise Exception('')

            if typeverb > 1:
                print('gdat.diffgridstar')
                print(gdat.diffgridstar)
                if gdat.typecoor == 'comp':
                    print('gdat.diffgridcomp')
                    print(gdat.diffgridcomp)
            
            gdat.areapixlstar = gdat.diffgridstar**2
            if gdat.typecoor == 'comp':
                gdat.areapixlcomp = gdat.diffgridcomp**2
            
            if gdat.typesyst == 'turkey':
                if gdat.numbcomp != 1:
                    raise Exception('')
                path = os.environ['EPHESOS_DATA_PATH'] + '/data/LightCurve/turkey.csv'
                print('Reading from %s...' % path)
                gdat.positurk = np.loadtxt(path, delimiter=',')
                
                print('Scaling and centering the template coordinates...')
                for a in range(2):
                    gdat.positurk[:, a] -= np.amin(gdat.positurk[:, a])
                # half of the diagonal of the rectangle
                halfdiag = 0.5 * np.sqrt((np.amax(gdat.positurk[:, 0]))**2 + (np.amax(gdat.positurk[:, 1]))**2)
                # normalize
                gdat.positurk *= gdat.rratcomp[0] / halfdiag
                # center
                gdat.positurk -= 0.5 * np.amax(gdat.positurk, 0)[None, :]
                
                diffturk = 1. * gdat.diffgrid
                diffturksmth = 2. * diffturk
                gdat.xposturk = np.arange(-5 * diffturk + np.amin(gdat.positurk[:, 0]), np.amax(gdat.positurk[:, 0]) + 5. * diffturk, diffturk)
                gdat.yposturk = np.arange(-5 * diffturk + np.amin(gdat.positurk[:, 1]), np.amax(gdat.positurk[:, 1]) + 5. * diffturk, diffturk)
                gdat.maxmxposturkmesh = np.amax(gdat.xposturk)
                gdat.maxmyposturkmesh = np.amax(gdat.yposturk)

                gdat.xposturkmesh, gdat.yposturkmesh = np.meshgrid(gdat.xposturk, gdat.yposturk)
                gdat.xposturkmeshflat = gdat.xposturkmesh.flatten()
                gdat.yposturkmeshflat = gdat.yposturkmesh.flatten()
                gdat.positurkmesh = np.vstack([gdat.xposturkmeshflat, gdat.yposturkmeshflat]).T
                
                gdat.valuturkmesh = np.exp(-((gdat.positurk[:, 0, None] - gdat.xposturkmeshflat[None, :]) / diffturksmth)**2 \
                                          - ((gdat.positurk[:, 1, None] - gdat.yposturkmeshflat[None, :]) / diffturksmth)**2)
                gdat.valuturkmesh = np.sum(gdat.valuturkmesh, 0)
                
                if not np.isfinite(gdat.valuturkmesh).all():
                    print('gdat.xposturkmeshflat')
                    summgene(gdat.xposturkmeshflat)
                    print('gdat.yposturkmeshflat')
                    summgene(gdat.yposturkmeshflat)
                    print('')
                    raise Exception('')
            
            if gdat.typecoor == 'star' and gdat.boolmakeanim or gdat.boolmakeimaglfov:
                if gdat.diffgridstar < 1e-3:
                    raise Exception('Images will be made, but the primary grid resolution is too low.')
                    
            ## planet
            if gdat.typecoor == 'comp':
                
                gdat.xposgridcomp = [[] for j in gdat.indxcomp]
                gdat.yposgridcomp = [[] for j in gdat.indxcomp]
                gdat.zposgridcomp = [[] for j in gdat.indxcomp]
                
                gdat.xposgridsphr = [[] for j in gdat.indxcomp]
                gdat.yposgridsphr = [[] for j in gdat.indxcomp]
                gdat.zposgridsphr = [[] for j in gdat.indxcomp]
                
                gdat.distfromcompgridcomp = [[] for j in gdat.indxcomp]
                gdat.indxplannoccgridcomp = [[] for j in gdat.indxcomp]
                gdat.numbsidegridcomp = [[] for j in gdat.indxcomp]
                gdat.latisinugridcomp = [[] for j in gdat.indxcomp]
                gdat.laticosigridcomp = [[] for j in gdat.indxcomp]
                gdat.boolplangridcomp = [[] for j in gdat.indxcomp]
                gdat.indxplangridcomp = [[] for j in gdat.indxcomp]

                gdat.longgridsphr = [[] for j in gdat.indxcomp]
                gdat.latigridsphr = [[] for j in gdat.indxcomp]
                
                if gdat.boolsystpsys:
                    
                    # Booleans indicating the region outside the planet in the companion grid
                    gdat.boolgridcompoutscomp = [[] for j in gdat.indxcomp]

                    # Booleans indicating the region inside the planet in the companion grid
                    gdat.boolgridcompinsdcomp = [[] for j in gdat.indxcomp]

                for j in gdat.indxcomp:
                    
                    if gdat.typesyst == 'CompactObjectStellarCompanion':
                        limtgridxpos = gdat.wdthslen[j]
                        limtgridypos = gdat.wdthslen[j]
                    elif gdat.typesyst.startswith('PlanetarySystemWithRings') or gdat.boolmakeanim:
                        limtgridxpos = 2. * gdat.rratcomp[j]
                        limtgridypos = 2. * gdat.rratcomp[j]
                    else:
                        limtgridxpos = gdat.rratcomp[j]
                        limtgridypos = gdat.rratcomp[j]
                    
                    if gdat.booldiag:
                        if limtgridxpos / gdat.diffgridcomp > 1e4:
                            print('')
                            print('')
                            print('')
                            print('gdat.typesyst')
                            print(gdat.typesyst)
                            print('gdat.rratcomp')
                            print(gdat.rratcomp)
                            raise Exception('limtgridxpos / gdat.diffgridcomp > 1e4')
                    
                    if gdat.boolmakeanim:
                        if gdat.diffgridcomp / limtgridxpos > 3e-2:
                            print('')
                            print('')
                            print('')
                            print('gdat.typecoor')
                            print(gdat.typecoor)
                            print('gdat.diffgridcomp')
                            print(gdat.diffgridcomp)
                            print('gdat.boolmakeanim')
                            print(gdat.boolmakeanim)
                            print('gdat.boolmakeimaglfov')
                            print(gdat.boolmakeimaglfov)
                            print('limtgridxpos')
                            print(limtgridxpos)
                            print('gdat.boolmakeanim')
                            print(gdat.boolmakeanim)
                            print('gdat.diffgridcomp / limtgridxpos')
                            print(gdat.diffgridcomp / limtgridxpos)
                            #raise Exception('Images could be made, but the companion grid resolution is too low.')
                            print('Images will be made, but the companion grid resolution is too low.')

                    arrycompxpos = np.arange(-limtgridxpos - 2. * gdat.diffgridcomp, limtgridxpos + 3. * gdat.diffgridcomp, gdat.diffgridcomp)
                    arrycompypos = np.arange(-limtgridypos - 2. * gdat.diffgridcomp, limtgridypos + 3. * gdat.diffgridcomp, gdat.diffgridcomp)
                    gdat.numbsidegridcomp[j] = arrycompxpos.size
                    
                    gdat.xposgridcomp[j], gdat.yposgridcomp[j] = np.meshgrid(arrycompypos, arrycompxpos)
                    
                    if gdat.typebrgtcomp != 'dark':
                        gdat.xposgridsphr[j] = gdat.xposgridcomp[j]
                        gdat.zposgridsphr[j] = gdat.yposgridcomp[j]
                        gdat.yposgridsphr[j] = np.empty_like(gdat.xposgridsphr[j])
                    
                        temp = gdat.rratcomp[j]**2 - gdat.xposgridsphr[j]**2 - gdat.zposgridsphr[j]**2
                        # indices of companion grid where this will not produce NaNs due to being outside of the companion
                        indxinsd = np.where(temp >= 0)
                        gdat.yposgridsphr[j][indxinsd] = np.sqrt(temp[indxinsd])
                        
                        # maybe to be deleted
                        #gdat.latigridcomp[j] = gdat.yposgridcomp[j] / gdat.rratcomp[j]
                        #gdat.latigridcomp[j] = np.sqrt(1. - gdat.latisinugridcomp[j]**2)
                        #gdat.zposgridcomp[j] = np.sqrt(gdat.rratcomp[j]*2 - gdat.xposgridcomp[j]**2  - gdat.yposgridcomp[j]**2)
                        
                        gdat.latigridsphr[j] = -0.5 * np.pi + \
                                                np.arccos(gdat.zposgridsphr[j] / np.sqrt(gdat.xposgridsphr[j]**2 + gdat.yposgridsphr[j]**2 + gdat.zposgridsphr[j]**2))
                        gdat.longgridsphr[j] = np.arctan2(gdat.yposgridsphr[j], gdat.xposgridsphr[j])
                        
                        #print('gdat.xposgridsphr[j]')
                        #summgene(gdat.xposgridsphr[j])
                        #print('gdat.yposgridsphr[j]')
                        #summgene(gdat.yposgridsphr[j])
                        #print('gdat.zposgridsphr[j]')
                        #summgene(gdat.zposgridsphr[j])
                        #print('gdat.longgridsphr[j]')
                        #summgene(gdat.longgridsphr[j])
                        #print('gdat.latigridsphr[j]')
                        #summgene(gdat.latigridsphr[j])
                        if gdat.tmptsliccomp is None:
                            gdat.tmptsliccomp = np.maximum(0.2 * np.random.randn(16) + 0.9, np.zeros(16))
                        
                        if gdat.typesyst == 'PlanetarySystemEmittingCompanion' and gdat.tmptsliccomp is None and gdat.typebrgtcomp == 'heated_sliced':
                            raise Exception('')

                        if gdat.tmptsliccomp is not None:
                            gdat.numbslic = len(gdat.tmptsliccomp)
                            gdat.brgtsliccomp = gdat.tmptsliccomp**4

                    gdat.distfromcompgridcomp[j] = np.sqrt(gdat.xposgridcomp[j]**2 + gdat.yposgridcomp[j]**2)
                    
                    gdat.boolplangridcomp[j] = gdat.distfromcompgridcomp[j] < gdat.rratcomp[j]
                    
                    gdat.indxplangridcomp[j] = np.where(gdat.boolplangridcomp[j])

                    if gdat.boolsystpsys:
                        
                        gdat.boolgridcompoutscomp[j] = retr_boolgridouts(gdat, j, gdat.typecoor, typeoccu='comp')
                    
                        if gdat.booldiag:
                            if gdat.boolgridcompoutscomp[j].ndim != 2:
                                raise Exception('')

                        gdat.boolgridcompinsdcomp[j] = ~gdat.boolgridcompoutscomp[j]

                    if typeverb > 1:
                        print('Number of pixels in the grid for companion %d: %d' % (j, gdat.xposgridcomp[j].size))
                        gdat.precphotflorcomp = 1e6 / gdat.xposgridcomp[j].size
                        print('Photometric precision floor achieved by this resolution: %g ppm' % gdat.precphotflorcomp)
                
            if gdat.typesyst == 'CompactObjectStellarCompanion' and gdat.typemodllens == 'phdy':
                gdat.diffgridsour = gdat.diffgrid / gdat.factosampsour
                
                gdat.areapixlsour = gdat.diffgridsour**2

                # grid for source plane
                arrysour = np.arange(-2. - 2. * gdat.diffgridsour, 2. + 3. * gdat.diffgridsour, gdat.diffgridsour)
                gdat.xposgridsoursing = arrysour
                gdat.yposgridsoursing = arrysour
                gdat.xposgridsour, gdat.yposgridsour = np.meshgrid(arrysour, arrysour)
                gdat.distgridsour = np.sqrt(gdat.xposgridsour**2 + gdat.yposgridsour**2)
                if typeverb > 1:
                    print('Number of pixels in the source grid: %d' % (gdat.xposgridsour.size))
                    print('gdat.xposgridsour')
                    summgene(gdat.xposgridsour)
                    print('gdat.yposgridsour')
                    summgene(gdat.yposgridsour)
                
                # source plane distance defined on the companion grid
                gdat.distsourgridcomp = gdat.distfromcompgridcomp[0] - gdat.radieins**2 / gdat.distfromcompgridcomp[0]
                
                # source plane distance defined on the source grid
                gdat.distsourgridsour = gdat.distgridsour - gdat.radieins**2 / gdat.distgridsour
                
                # normalized source plane distance (u)
                gdat.distsournormgridsour = gdat.distsourgridsour / gdat.radieins
                
                # magnification sampled on the source plane grid
                gdat.magn = abs((gdat.distsournormgridsour**2 + 2.) / (2. * gdat.distsournormgridsour * np.sqrt(gdat.distsournormgridsour**2 + 4.)) + 0.5)
                
                #for lll in range(gdat.magn.size):
                #    print('dist: %g magn: %g' % (gdat.distsournorm.flatten()[lll], gdat.magn.flatten()[lll]))
                
                # normalized source plane distance (u)
                gdat.distsournormgridcomp = gdat.distsourgridcomp / gdat.radieins
                
                # interpolate the background (source) brightness at that position
                gdat.xposintp = gdat.xposgridcomp[0] / gdat.distfromcompgridcomp[0] * gdat.distsournormgridcomp
                gdat.yposintp = gdat.yposgridcomp[0] / gdat.distfromcompgridcomp[0] * gdat.distsournormgridcomp
                
                if typeverb > 1:
                    print('gdat.xposintp')
                    summgene(gdat.xposintp)
                    print('gdat.yposintp')
                    summgene(gdat.yposintp)
            
                gdat.arrygridsourintp = np.vstack([gdat.xposintp.flatten(), gdat.yposintp.flatten()]).T

            
            ## star
            arrystar = np.arange(-1. - 2. * gdat.diffgridstar, 1. + 3. * gdat.diffgridstar, gdat.diffgridstar)
            gdat.xposgridstar, gdat.yposgridstar = np.meshgrid(arrystar, arrystar)
            gdat.numbpixlgridstar = gdat.xposgridstar.size
            gdat.precphotflorstar = 1e6 / gdat.numbpixlgridstar
            
            if gdat.typeverb > 0:
                if gdat.numbpixlgridstar > 1e3 and gdat.typecoor == 'star':
                    print('Warning! typecoor is %s. Too many points (%d) in the star grid due to resolution being too high (%g). It will take too long to evaluate the light curve.' % \
                                                                                                                            (gdat.typecoor, gdat.numbpixlgridstar, gdat.diffgridstar))

            if gdat.typeverb > 1:
                print('Number of pixels in the stellar grid: %d' % (gdat.numbpixlgridstar))
                print('Photometric precision floor achieved by this resolution: %g ppm' % gdat.precphotflorstar)

            # distance to the star in the star grid
            gdat.distgridstar = np.sqrt(gdat.xposgridstar**2 + gdat.yposgridstar**2)
            
            # Booleans indicating whether star grid points are inside the star
            gdat.boolgridstarinsdstar = gdat.distgridstar < 1.
            
            # indices of the star grid points inside the star
            gdat.indxgridstarstar = np.where(gdat.boolgridstarinsdstar)
            
            # stellar brightness in the star grid
            gdat.brgtprimgridprim = np.zeros_like(gdat.xposgridstar)
            cosg = np.sqrt(1. - gdat.distgridstar[gdat.indxgridstarstar]**2)
            gdat.brgtprimgridprim[gdat.indxgridstarstar] = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk) * gdat.areapixlstar
            
            if gdat.typesyst == 'CompactObjectStellarCompanion':
                
                #metrcomp = 3e-10 * gdat.xposgridsour[0].size * gdat.xposgridcomp[0].size
                metrcomp = 3e-10 * gdat.numbpixlgridstar * gdat.xposgridcomp[0].size
                
                if gdat.typeverb > 1:
                    print('Estimated execution time per time sample: %g ms' % metrcomp)
                    print('Estimated execution time: %g s' % (1e-3 * gdat.numbtime * metrcomp))
                
                if gdat.typecoor == 'star':
                    arry = np.arange(-1. - 2.5 * gdat.diffgridstar, 1. + 3.5 * gdat.diffgridstar, gdat.diffgridstar)
                    gdat.binsxposgridstar = arry
                    gdat.binsyposgridstar = arry
                if gdat.typecoor == 'comp':
                    arry = np.arange(-gdat.wdthslen[j] - 2.5 * gdat.diffgridcomp, gdat.wdthslen[j] + 3.5 * gdat.diffgridcomp, gdat.diffgridcomp)
                    gdat.binsxposgridcomp = arry
                    gdat.binsyposgridcomp = arry
            
                # maximum stellar brightness for source grid
                if gdat.pathvisu is not None:
                    gdat.maxmbrgtsour = nicomedia.retr_brgtlmdk(1., gdat.coeflmdk, typelmdk=gdat.typelmdk) * gdat.areapixlsour
            
            # maximum stellar brightness for planet and star grids
            if gdat.boolmakeanim or gdat.boolmakeimaglfov:
                gdat.maxmlumistar = nicomedia.retr_brgtlmdk(1., gdat.coeflmdk, typelmdk=gdat.typelmdk) * gdat.areapixlstar
            
            # total (unocculted) stellar birghtness
            gdat.lumistarnocc = np.sum(gdat.brgtprimgridprim)
            
            if gdat.typeverb > 1:
                print('gdat.lumistarnocc')
                print(gdat.lumistarnocc)
            if gdat.typeverb > 0:
                if gdat.booltqdm:
                    print('The model will be evaluated over %d time samples. tqdm will be used to indicate progress.' % gdat.numbtime)
            
            if gdat.typecoor == 'star' or gdat.boolmakeimaglfov:
                # coordinates of the companion in a grid where the star is stationary
                gdat.xposcompgridstar = [[] for j in gdat.indxcomp]
                gdat.yposcompgridstar = [[] for j in gdat.indxcomp]
                gdat.zposcompgridstar = [[] for j in gdat.indxcomp]
                if gdat.perimoon is not None:
                    gdat.xposmoon = [[[] for jj in indxmoon[j]] for j in gdat.indxcomp]
                    gdat.yposmoon = [[[] for jj in indxmoon[j]] for j in gdat.indxcomp]
                
            if gdat.typecoor == 'comp':
                # coordinates of the star in a list of grids where each companion is stationary, respectively
                gdat.xposstargridcomp = [[] for j in gdat.indxcomp]
                gdat.yposstargridcomp = [[] for j in gdat.indxcomp]
                gdat.zposstargridcomp = [[] for j in gdat.indxcomp]
            
            if gdat.boolintp:
                gdat.listphaseval = [[] for j in gdat.indxcomp]
                gdat.numbphaseval = np.empty(gdat.numbcomp, dtype=int)
                for j in gdat.indxcomp:
                    if np.isfinite(gdat.duratrantotl[j]):
                        
                        # array of durations of phase oversampled for the total transit
                        if gdat.typesyst.startswith('PlanetarySystemWithRings'):
                            gdat.phastrantotl = 2. * gdat.duratrantotl / gdat.pericomp
                        else:
                            gdat.phastrantotl = gdat.duratrantotl / gdat.pericomp
                        
                        if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                            # array of durations of phase oversampled for the full transit
                            if gdat.typesyst.startswith('PlanetarySystemWithRings'):
                                gdat.phastranfull = 2. * gdat.duratranfull / gdat.pericomp
                            else:
                                gdat.phastranfull = gdat.duratranfull / gdat.pericomp
                            
                            # array of durations of phase oversampled for the ingress and egress, inlcluding a fudge factor
                            deltphasineg = 1.1 * (gdat.phastrantotl - gdat.phastranfull) / 2.
                            
                            deltphasineghalf = 0.5 * deltphasineg
                            
                            # array of the phase of the ingress and egress
                            phasingr = (gdat.phastrantotl + gdat.phastranfull) / 4.
                            
                        else:
                            phasingr = gdat.phastrantotl / 2.
                            deltphasineghalf = np.zeros(gdat.numbcomp)
                        
                        # before ingress
                        gdat.listphaseval[j] = [np.arange(-0.25, -phasingr[j] - deltphasineghalf[j], gdat.diffphaspcur)]
                        
                        # ingress
                        if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                            gdat.listphaseval[j].append(np.arange(-phasingr[j] - deltphasineghalf[j], -phasingr[j] + deltphasineghalf[j], gdat.diffphasineg))
                        
                        # in transit, after ingress
                        gdat.listphaseval[j].append(np.arange(-phasingr[j] + deltphasineghalf[j], phasingr[j] - deltphasineghalf[j], gdat.diffphasintr[j]))
                        
                        # 
                        if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                            gdat.listphaseval[j].append(np.arange(phasingr[j] - deltphasineghalf[j], phasingr[j] + deltphasineghalf[j], gdat.diffphasineg))
                        
                        gdat.listphaseval[j].append(np.arange(phasingr[j] + deltphasineghalf[j], 0.5 - phasingr[j] - deltphasineghalf[j], gdat.diffphaspcur))
                        
                        if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                            gdat.listphaseval[j].append(np.arange(0.5 - phasingr[j] - deltphasineghalf[j], 0.5 - phasingr[j] + deltphasineghalf[j], gdat.diffphasineg))
                                                       
                        gdat.listphaseval[j].append(np.arange(0.5 - phasingr[j] + deltphasineghalf[j], 0.5 + phasingr[j] - deltphasineghalf[j], gdat.diffphasintr[j]))
                                                       
                        if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                            gdat.listphaseval[j].append(np.arange(0.5 + phasingr[j] - deltphasineghalf[j], 0.5 + phasingr[j] + deltphasineghalf[j], gdat.diffphasineg))
                        
                        gdat.listphaseval[j].append(np.arange(0.5 + phasingr[j] + deltphasineghalf[j], 0.75 + gdat.diffphaspcur, gdat.diffphaspcur))
                        
                        gdat.listphaseval[j] = np.concatenate(gdat.listphaseval[j])
                    else:
                        gdat.listphaseval[j] = np.arange(-0.25, 0.75 + gdat.diffphaspcur, gdat.diffphaspcur)
                    
                    if gdat.booldiag:
                        if (gdat.listphaseval[j] - np.sort(gdat.listphaseval[j]) != 0).any():
                            print('')
                            print('')
                            print('')
                            print('phasingr[j]')
                            print(phasingr[j])
                            print('deltphasineghalf[j]')
                            print(deltphasineghalf[j])
                            print('gdat.diffphasineg')
                            print(gdat.diffphasineg)
                            print('gdat.diffphaspcur')
                            print(gdat.diffphaspcur)
                            print('gdat.diffphasintr[j]')
                            print(gdat.diffphasintr[j])
                            print('gdat.listphaseval[j]')
                            for phaseval in gdat.listphaseval[j]:
                                print(phaseval)
                            print('Warning! gdat.listphaseval[j] are not sorted. Make sure this is not causing any issues down the pipeline.')
                    
                    gdat.numbphaseval[j] = gdat.listphaseval[j].size
                gdat.listposiprimgridcompeval = [np.empty((gdat.numbphaseval[j], 3)) for j in gdat.indxcomp]
            
                gdat.lumisysteval = [np.empty(gdat.numbphaseval[j]) for j in gdat.indxcomp]
            
            if boolcompmoon:
                numbitermoon = 2
            else:
                numbitermoon = 1
            
            gdat.lumisyst = np.full(gdat.numbtime, gdat.lumistarnocc)
            
            if gdat.typemodllens == 'gaus':
                if gdat.typesyst == 'CompactObjectStellarCompanion':
                    gdat.lumisyst += gdat.lumistarnocc * np.exp(-(gdat.phascomp[0] / gdat.dcyctrantotlhalf[j])**2)
            else:
                for a in range(numbitermoon):
                    
                    if a == 0:
                        gdat.strgcompmoon = ''
                    else:
                        gdat.strgcompmoon = '_onlycomp'
                    
                    if gdat.boolmakeanim:
                        for namevarbanim in gdat.listnamevarbfram:
                            gdat.pathgiff[namevarbanim] = gdat.pathvisu + 'anim%s%s%s.gif' % (namevarbanim, gdat.strgextn, gdat.strgcompmoon)
                            gdat.cmndmakeanim[namevarbanim] = 'convert -delay 5 -density 200'
                            gdat.cmnddeleimag[namevarbanim] = 'rm'
                        
                            pathbasedele = retr_pathbaseanimfram(gdat, namevarbanim)
                            cmnd = 'rm %s*' % pathbasedele
                            print(cmnd)
                            os.system(cmnd)
                    
                    # evaluate the brightness as a interpolate 
                    if gdat.boolintp:
                        
                        for j in gdat.indxcomp:
                            
                            if not np.isfinite(gdat.duratrantotl[j]) or gdat.duratrantotl[j] == 0.:
                                continue
                            
                            if gdat.booltqdm:
                                objttemp = tqdm(range(gdat.numbphaseval[j]))
                            else:
                                objttemp = range(gdat.numbphaseval[j])
                            
                            for t in objttemp:
                                
                                proc_phaseval(gdat, j, t)
                                
                                if gdat.boolmakeanim:
                                    make_imag(gdat, t, gdat.typecoor, j=j)
                            
                    else:
                        
                        if gdat.booltqdm:
                            objttemp = tqdm(range(gdat.numbtime))
                        else:
                            objttemp = range(gdat.numbtime)
                        
                        for t in objttemp:
                            proc_time(gdat, t)

                    if gdat.boolmakeanim:
                        
                        for namevarbanim in gdat.listnamevarbfram:
                            if not os.path.exists(gdat.pathgiff[namevarbanim]):
                                # make the animation
                                gdat.cmndmakeanim[namevarbanim] += ' %s' % gdat.pathgiff[namevarbanim]
                                print('Writing to %s...' % gdat.pathgiff[namevarbanim])
                                os.system(gdat.cmndmakeanim[namevarbanim])
                                
                                # delete images
                                os.system(gdat.cmnddeleimag[namevarbanim])

                if gdat.boolintp:
                    
                    for j in gdat.indxcomp:
                        
                        if not np.isfinite(gdat.duratrantotl[j]) or gdat.duratrantotl[j] == 0.:
                            continue
                        
                        if gdat.listphaseval[j].size > 1:
                            
                            if gdat.typesyst == 'CompactObjectStellarCompanion':
                                indxphaseval = gdat.indxtime
                            else:
                                indxphaseval = np.where((gdat.phascomp[j] >= np.amin(gdat.listphaseval[j])) & (gdat.phascomp[j] <= np.amax(gdat.listphaseval[j])))[0]
                                
                            if indxphaseval.size > 0:
                                intptemp = scipy.interpolate.interp1d(gdat.listphaseval[j], gdat.lumisysteval[j], fill_value=gdat.lumistarnocc, \
                                                                                                            bounds_error=False)(gdat.phascomp[j][indxphaseval])
                                
                                if gdat.typesyst == 'CompactObjectStellarCompanion':
                                    if gdat.booldiag:
                                        if np.amin(intptemp) / gdat.lumistarnocc > 1.1:
                                            raise Exception('')
                                    gdat.lumisyst[indxphaseval] = intptemp
                                else:
                                    diff = gdat.lumistarnocc - intptemp
                                    gdat.lumisyst[indxphaseval] -= diff

                                    if gdat.booldiag:
                                        if not np.isfinite(gdat.lumisyst).all() or (gdat.lumisyst < 0).any():
                                            print('')
                                            print('')
                                            print('')
                                            print('gdat.lumisyst')
                                            summgene(gdat.lumisyst)
                                            print('temp: suppressing the exception')
                                            #raise Exception('')
        
                        else:
                            gdat.lumisyst = np.full_like(gdat.phascomp[j], gdat.lumistarnocc)
            
            if gdat.booldiag:
                if not np.isfinite(gdat.lumisyst).all() or (gdat.lumisyst < 0).any():
                    print('')
                    print('')
                    print('')
                    print('gdat.lumisyst')
                    summgene(gdat.lumisyst)
                    print('temp: suppressing the exception')
                    #raise Exception('')
    
            # normalize the light curve
            if typenorm != 'none':
                
                if gdat.booldiag:
                    if gdat.typesyst == 'CompactObjectStellarCompanion':
                        if (gdat.lumisyst / gdat.lumistarnocc < 1. - 1e-6).any():
                            print('Warning! Flux decreased in a self-lensing light curve.')
                            print('gdat.lumisyst')
                            print(gdat.lumisyst)
                            #raise Exception('')

                if gdat.typeverb > 1:
                    if gdat.typesyst == 'CompactObjectStellarCompanion':
                        print('gdat.lumisyst')
                        summgene(gdat.lumisyst)
                    print('Normalizing the light curve...')
                
                if typenorm == 'medi':
                    fact = np.median(gdat.lumisyst)
                elif typenorm == 'nocc':
                    fact = gdat.lumistarnocc
                elif typenorm == 'maxm':
                    fact = np.amax(gdat.lumisyst)
                elif typenorm == 'edgeleft':
                    fact = gdat.lumisyst[0]
                rflxtranmodl = gdat.lumisyst / fact
                
                if gdat.booldiag:
                    if fact == 0.:
                        print('typenorm')
                        print(typenorm)
                        print('')
                        for j in gdat.indxcomp:
                            print('gdat.lumisysteval[j]')
                            summgene(gdat.lumisysteval[j])
                        print('Normalization involved division by 0.')
                        print('gdat.lumisyst')
                        summgene(gdat.lumisyst)
                        raise Exception('')
                    if gdat.typesyst == 'CompactObjectStellarCompanion':
                        if (rflxtranmodl < 0.9).any():
                            raise Exception('')

                #if (rflxtranmodl > 1e2).any():
                #    raise Exception('')

            if gdat.booldiag:
                if gdat.boolsystpsys and gdat.typesyst != 'PlanetarySystemEmittingCompanion' and np.amax(gdat.lumisyst) > gdat.lumistarnocc * (1. + 1e-6):
                    print('')
                    print('')
                    print('')
                    print('gdat.typesyst')
                    print(gdat.typesyst)
                    print('gdat.typelmdk')
                    print(gdat.typelmdk)
                    print('gdat.coeflmdk')
                    print(gdat.coeflmdk)
                    print('gdat.lumisyst')
                    summgene(gdat.lumisyst)
                    print('gdat.lumistarnocc')
                    print(gdat.lumistarnocc)
                    raise Exception('gdat.boolsystpsys and gdat.typesyst != psyspcur and np.amax(gdat.lumisyst) > gdat.lumistarnocc * (1. + 1e-6)')
            
                if False and np.amax(rflxtranmodl) > 1e6:
                    print('gdat.lumisyst')
                    summgene(gdat.lumisyst)
                    raise Exception('')

        dictefes['rflx'] = rflxtranmodl
        
        if gdat.masscomp is not None and gdat.massstar is not None and gdat.radistar is not None:
            densstar = 1.4 * gdat.massstar / gdat.radistar**3
            deptbeam = 1e-3 * nicomedia.retr_deptbeam(gdat.pericomp, gdat.massstar, gdat.masscomp)
            deptelli = 1e-3 * nicomedia.retr_deptelli(gdat.pericomp, densstar, gdat.massstar, gdat.masscomp)
            dictefes['rflxslen'] = [[] for j in gdat.indxcomp]
            dictefes['rflxbeam'] = [[] for j in gdat.indxcomp]
            dictefes['rflxelli'] = [[] for j in gdat.indxcomp]

            for j in gdat.indxcomp:
                
                dictefes['rflxslen'][j] = dictefes['rflx']
                
                dictefes['rflxbeam'][j] = 1. + deptbeam * np.sin(gdat.phascomp[j])
                dictefes['rflxelli'][j] = 1. + deptelli * np.sin(2. * gdat.phascomp[j])
                
                dictefes['rflx'] += dictefes['rflxelli'][j] - 2.
                dictefes['rflx'] += dictefes['rflxbeam'][j]
        
        if boolcompmoon:
            rflxtranmodlcomp /= np.amax(rflxtranmodlcomp)
            dictefes['rflxcomp'] = rflxtranmodlcomp
            rflxtranmodlmoon = 1. + rflxtranmodl - rflxtranmodlcomp
            dictefes['rflxmoon'] = rflxtranmodlmoon

        if gdat.boolfast and gdat.rratcomp.ndim == 2 and gdat.boolrscllcur:
            dictefes['rflx'] = 1. - gdat.rratcompsave[None, 0, :] * (1. - dictefes['rflx'][:, None])
        
        # create dummy energy axis
        if dictefes['rflx'].ndim == 1:
            dictefes['rflx'] = dictefes['rflx'][:, None]
        
        if (dictefes['rflx'] == 0).all():
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.numbcomp')
            print(gdat.numbcomp)
            print('gdat.pericomp')
            print(gdat.pericomp)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            print('gdat.rsmacomp')
            print(gdat.rsmacomp)
            print('gdat.cosicomp')
            print(gdat.cosicomp)
            print('gdat.duratrantotl')
            print(gdat.duratrantotl)
            print('gdat.time')
            summgene(gdat.time)
            for j in gdat.indxcomp:
                print('gdat.lumisysteval[j]')
                summgene(gdat.lumisysteval[j])
            print('dictefes[rflx]')
            summgene(dictefes['rflx'])
            raise Exception('')

        if gdat.booldiag:
            if not np.isfinite(dictefes['rflx']).all() or (dictefes['rflx'] < 0).any():
                print('')
                print('')
                print('')
                print('dictefes[rflx]')
                summgene(dictefes['rflx'])
                print('temp: suppressing the exception')
                #raise Exception('')
        
        if gdat.pathvisu is not None:
            
            # load the Efes dictionary
            for name in dictinpt:
                if name != 'gdat':
                    dictefes[name] = getattr(gdat, name)
            
            if gdat.booldiag:
                if (gdat.time - np.sort(gdat.time) != 0).any():
                    print('')
                    print('')
                    print('')
                    print('gdat.time')
                    summgene(gdat.time)
                    raise Exception('Time array is not sorted.')

                if (dictefes['time'] - np.sort(dictefes['time']) != 0).any():
                    print('')
                    print('')
                    print('')
                    print('dictefes[time]')
                    summgene(dictefes['time'])
                    print(dictefes['time'])
                    raise Exception('Time array is not sorted.')

            # plot the contents of the Efes dictionary
            plot_tser_dictefes(gdat.pathvisu, dictefes, '%s' % gdat.strgextn, lablunittime)
        
    if gdat.booldiag:
        if gdat.typecoor == 'star' and gdat.boolsystpsys and not (gdat.rratcomp > gdat.diffgridstar).all():
            print('')
            print('')
            print('')
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            print('WARNING! At least one of the occulter radii is smaller than the grid resolution. The output will be unreliable.')
    
    if gdat.boolmakeimaglfov and len(rratcomp) > 0:
        
        gdat.maxmsmaxcomp = np.amax(gdat.smaxcomp)
        gdat.limtposiimag = [-1.5 * gdat.maxmsmaxcomp, 1.5 * gdat.maxmsmaxcomp]
        numbimaglfov = 20
        if gdat.typecoor == 'star':
            maxm = gdat.numbtime - 1
        else:
            maxm = gdat.numbphaseval[0] - 1
        indxtimeimaglfov = np.linspace(0., maxm, numbimaglfov).astype(int)
        
        # if the computation was done in the companion grids over the evaluation times, then also calculate the positions in the global time grid
        if gdat.typecoor == 'comp':
            print('Original computation was done in the companion grids over the evaluation times. Additionally calculating the positions in the global time grid...')
            for t in gdat.indxtime:
                for j in gdat.indxcomp:
                    xpos, ypos, zpos, anommean, anomecce, anomtrue = retr_posifromphas_efes(gdat, j, t, gdat.phascomp[j][t])
                        
                    gdat.xposcompgridstar[j] = xpos
                    gdat.yposcompgridstar[j] = ypos
                    gdat.zposcompgridstar[j] = zpos
                    
                    gdat.dictvarborbt['posicompgridprim'][t, j, 0] = gdat.xposcompgridstar[j]
                    gdat.dictvarborbt['posicompgridprim'][t, j, 1] = gdat.yposcompgridstar[j]
                    gdat.dictvarborbt['posicompgridprim'][t, j, 2] = gdat.zposcompgridstar[j]
                    gdat.dictvarborbt['anomtrue'][t, j] = anomtrue
        
        if gdat.booldiag:
            liststrg = ['x', 'y', 'z']
            for j in gdat.indxcomp:
                for a in range(3):
                    if (gdat.pericomp[j] < (gdat.time[-1] - gdat.time[0])) and (not (gdat.dictvarborbt['posicompgridprim'][:, j, a] > 0).any() or 
                        not (gdat.dictvarborbt['posicompgridprim'][:, j, a] < 0).any()) and not (gdat.dictvarborbt['posicompgridprim'][:, j, a] == 0).all():
                        print('')
                        print('')
                        print('')
                        print('gdat.dictvarborbt[anomtrue][:, j]')
                        summgene(gdat.dictvarborbt['anomtrue'][:, j])
                        print('gdat.dictvarborbt[posicompgridprim][:, j, a]')
                        summgene(gdat.dictvarborbt['posicompgridprim'][:, j, a])
                        raise Exception('All values are one-sided for %s-axis!' % liststrg[a])
        
            if gdat.typecoor == 'star':
                for j in gdat.indxcomp:
                    if gdat.xposcompgridstar[j].size:
                        print('')
                        print('')
                        print('')
                        print('j')
                        print(j)
                        print('gdat.xposcompgridstar[j]')
                        summgene(gdat.xposcompgridstar[j])
                        raise Exception('gdat.xposcompgridstar[j] is empty!')


        #gdat.listsegm = [[] for j in gdat.indxcomp]
        gdat.numbsegmfade = 10
        gdat.listalphline = [np.empty(gdat.numbsegmfade) for j in gdat.indxcomp]
        gdat.indxsegmfade = np.arange(gdat.numbsegmfade)
        gdat.indxsegmfadetime = (gdat.indxtime / gdat.numbtime * gdat.numbsegmfade).astype(int)
        gdat.indxtimesegmfade = [[] for ou in gdat.indxsegmfade]
        for ou in gdat.indxsegmfade:
            gdat.indxtimesegmfade[ou] = np.where(ou == gdat.indxsegmfadetime)[0]
            
        for j in gdat.indxcomp:
            for ou in gdat.indxsegmfade:
                gdat.listalphline[j][ou] = np.mean(gdat.dictvarborbt['anomtrue'][gdat.indxtimesegmfade[ou], j])
            gdat.listalphline[j] -= np.amin(gdat.listalphline[j])
            gdat.listalphline[j] /= np.amax(gdat.listalphline[j])
            
            #for t in gdat.indxtime:
            #    gdat.listsegm[j].append([gdat.dictvarborbt['posicompgridprim'][t, j, 0], gdat.dictvarborbt['posicompgridprim'][t, j, 1]])

        #for t in indxtimeimaglfov:
        #    make_diag(gdat, t, typecolr=gdat.typecolrimaglfov, typemrkr='none')
        #    make_diag(gdat, t, typecolr=gdat.typecolrimaglfov, typemrkr='tail')
        #    make_diag(gdat, t, typecolr=gdat.typecolrimaglfov, typemrkr='taillabl')
    
    
    # planet-planet crossings
    if gdat.boolcalcdistcomp:
        gdat.listdistcomp = np.empty((gdat.numbtime, gdat.numbcomp, gdat.numbcomp)) + np.nan
        for j in gdat.indxcomp:
            for jj in gdat.indxcomp:
                if j < jj:
                    gdat.listdistcomp[:, j, jj] = np.sqrt((gdat.dictvarborbt['posicompgridprim'][:, jj, 0] - gdat.dictvarborbt['posicompgridprim'][:, j, 0])**2 + \
                                                          (gdat.dictvarborbt['posicompgridprim'][:, jj, 1] - gdat.dictvarborbt['posicompgridprim'][:, j, 1])**2)
        gdat.listminmcompdist = np.nanmin(np.nanmin(gdat.listdistcomp, axis=1), axis=1)
        gdat.minmcompdist = np.nanmin(gdat.listdistcomp)
        
        dictefes['minmcompdist'] = np.nanmin(gdat.listdistcomp)
        dictefes['numbppcr'] = np.where(gdat.minmcompdist < 0.1)[0].size
        dictefes['rateppcr'] = dictefes['numbppcr'] / (gdat.time[-1] - gdat.time[0])
    
        if gdat.boolplotdistcomp:
            gdat.timeoffs = tdpy.retr_offstime(gdat.time)
            
            dictmodl = dict()
            dictmodl['eval'] = dict()
            dictmodl['eval']['time'] = gdat.time
            dictmodl['eval']['tser'] = gdat.listminmcompdist
            for a in range(2):
                if a == 0:
                    limtyaxi = None
                else:
                    limtyaxi = [0., 0.15]
                strgextnthis = 'MinimumPlanetPlanetDistance%s_%d' % (gdat.strgextn, a)

                pathplot = miletos.plot_tser(gdat.pathvisu, \
                                             dictmodl=dictmodl, \
                                             timeoffs=gdat.timeoffs, \
                                             strgextn=strgextnthis, \
                                             limtyaxi=limtyaxi, \
                                             lablyaxi='Minimum planet-planet distance [R$_*$]', \
                                        )
        
    dictefes['timetotl'] = modutime.time() - timeinit
    dictefes['timeredu'] = dictefes['timetotl'] / gdat.numbtime

    if gdat.booldiag and dictefes['timeredu'] > 1e-1 and not gdat.boolmakeanim:
        print('Took too long to execute...')
        #raise Exception('')
    
    if gdat.booldiag:
        if gdat.boolcalclcur and not np.isfinite(dictefes['rflx']).all():
            print('')
            print('')
            print('')
            print('dictefes[rflx] is not all finite.')
            print('dictefes[rflx]')
            summgene(dictefes['rflx'])
            print('gdat.boolintp')
            print(gdat.boolintp)
            print('gdat.lumistarnocc')
            print(gdat.lumistarnocc)
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('boolcompmoon')
            print(boolcompmoon)
            #print('rflxtranmodlcomp')
            #summgene(rflxtranmodlcomp)
            print('gdat.pericomp')
            print(gdat.pericomp)
            print('gdat.masscomp')
            print(gdat.masscomp)
            print('gdat.massstar')
            print(gdat.massstar)
            print('gdat.smaxcomp')
            print(gdat.smaxcomp)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            print('gdat.cosicomp')
            print(gdat.cosicomp)
            print('gdat.rsmacomp')
            print(gdat.rsmacomp)
            print('gdat.duratrantotl')
            print(gdat.duratrantotl)
            raise Exception('')
    
    if typeverb > 0:
        print('eval_modl ran in %.3g seconds and %g ms per 1000 time samples.' % (dictefes['timetotl'], dictefes['timeredu'] * 1e6))
        print('')

    return dictefes


def proc_time(gdat, t):
    
    if gdat.boolproftime:
        timeinit = modutime.time()
    
    # Boolean flag to evaluate the flux at this time
    gdat.boolevalflux = False
    
    if gdat.boolcalclcur:
        gdat.boolgridstarlght = np.copy(gdat.boolgridstarinsdstar)

    for j in gdat.indxcomp:
        
        if gdat.typeverb > 1:
            print('j')
            print(j)
        
        xpos, ypos, zpos, anommean, anomecce, anomtrue = retr_posifromphas_efes(gdat, j, t, gdat.phascomp[j][t])
        
        gdat.xposcompgridstar[j] = xpos
        gdat.yposcompgridstar[j] = ypos
        gdat.zposcompgridstar[j] = zpos
        
        gdat.dictvarborbt['posicompgridprim'][t, j, 0] = gdat.xposcompgridstar[j]
        gdat.dictvarborbt['posicompgridprim'][t, j, 1] = gdat.yposcompgridstar[j]
        gdat.dictvarborbt['posicompgridprim'][t, j, 2] = gdat.zposcompgridstar[j]
        gdat.dictvarborbt['anomtrue'][t, j] = anomtrue
        
        if gdat.perimoon is not None:
            for jj in indxmoon[j]:
                listposimoonfromcomp = retr_posifromphas_efes(gdat, j, t, gdat.phasmoon[jj][t])
                gdat.xposmoon[j][jj] = gdat.xposcompgridstar[j] + \
                                smaxmoon[j][jj] * np.cos(2. * np.pi * (gdat.time - epocmtramoon[j][jj]) / gdat.perimoon[j][jj]) / gdat.radistar * gdat.dictfact['aurs']
                gdat.yposmoon[j][jj] = gdat.yposcompgridstar[j] + \
                                smaxmoon[j][jj] * np.sin(2. * np.pi * (gdat.time - epocmtramoon[j][jj]) / gdat.perimoon[j][jj]) / gdat.radistar * gdat.dictfact['aurs']

        
        if gdat.boolcalclcur:

            if abs(gdat.phascomp[j][t]) < 0.25:# or (gdat.boolmakeimaglfov or gdat.boolmakeanim):
    
                gdat.boolevalflux = gdat.boolevalflux or retr_boolevalflux(gdat, j, typecoor)

                if gdat.perimoon is not None and a == 0:

                    for jj in indxmoon[j]:
                        
                        if np.sqrt(gdat.xposmoon[j][jj][t]**2 + gdat.yposmoon[j][jj][t]**2) < 1. + rratmoon[j][jj]:
                            
                            boolevaltranflux = True

                            xposgridmoon = gdat.xposgridstar - gdat.xposmoon[j][jj][t]
                            yposgridmoon = gdat.yposgridstar - gdat.yposmoon[j][jj][t]
                            
                            distmoon = np.sqrt(xposgridmoon**2 + yposgridmoon**2)
                            boolnoccmoon = distmoon > rratmoon[j][jj]
                            
                            gdat.boolgridstarlght = gdat.boolgridstarlght & boolnoccmoon
    
                if gdat.boolevalflux:
                    
                    if typecoor == 'comp':
                        gdat.boolgridcompoutscomp = retr_boolgridouts(gdat, j, typecoor, typeoccu='comp')
                        gdat.boolgridcomplght = gdat.boolgridcompinsdprim & gdat.boolgridcompoutscomp
                    elif typecoor == 'star':
                        gdat.boolgridstaroutscomp = retr_boolgridouts(gdat, j, typecoor, typeoccu='comp')
                        gdat.boolgridstarlght = gdat.boolgridstarlght & gdat.boolgridstaroutscomp
                    else:
                        raise Exception('')
            
    if gdat.boolevalflux:
        if typecoor == 'comp':
            gdat.lumisyst[t] = retr_lumistartran(gdat, typecoor, gdat.boolgridcomplght, j)
        elif typecoor == 'star':
            gdat.lumisyst[t] = retr_lumistartran(gdat, typecoor, gdat.boolgridstarlght)
        
    if gdat.boolmakeanim:
        make_imag(gdat, t)
   
    if gdat.boolproftime:
        timeexec = modutime.time() - timeinit
        print('proc_time() took %s seconds.' % timeexec)
                    
    if gdat.boolstopproftime:
        if timeexec > 0.1:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.numbcomp')
            print(gdat.numbcomp)
            print('gdat.typecoor')
            print(gdat.typecoor)
            print('gdat.numbpixlgridstar')
            print(gdat.numbpixlgridstar)
            print('gdat.boolintp')
            print(gdat.boolintp)
            print('timeexec')
            print(timeexec)
            print('gdat.numbtime')
            print(gdat.numbtime)
            print('gdat.numbtime * timeexec')
            print(gdat.numbtime * timeexec)
            raise Exception('Computation has taken too long.')

                            
def plot_tser_dictefes( \
                       pathvisu, \
                       dictefes, strgextninpt, lablunittime, typetarg='', typefileplot='png', \
                       
                       # type of plot background
                       typeplotback='black', \

                      ):
    

    if typeplotback == 'white':
        colrbkgd = 'white'
        colrdraw = 'black'
    elif typeplotback == 'black':
        colrbkgd = 'black'
        colrdraw = 'white'
    
    dictlabl = dict()
    dictlabl['root'] = dict()
    dictlabl['unit'] = dict()
    dictlabl['totl'] = dict()
    
    numbcomp = dictefes['pericomp'].size
    indxcomp = np.arange(numbcomp)
    listnamevarbcomp = []

    for j in indxcomp:
        listnamevarbcomp += ['pericom%d' % j, 'epocmtracom%d' % j, 'cosicom%d' % j, 'rsmacom%d' % j] 
        #listnamevarbcomp += ['radicom%d' % j]
        listnamevarbcomp += ['typebrgtcom%d' % j]
        if dictefes['typesyst'] == 'PlanetarySystemEmittingCompanion':
            listnamevarbcomp += ['offsphascom%d' % j]
    
    listnamevarbsimu = []
    listnamevarbstar = []
    #listnamevarbstar += ['radistar']
    #listnamevarbstar += ['coeflmdklinr', 'coeflmdkquad']
    
    listnamevarbsyst = listnamevarbstar + listnamevarbcomp
    listnamevarbtotl = listnamevarbsyst + listnamevarbsimu
    
    # temp
    #listnamevarbtotl = list(dictefes['dictinpt'].keys())

    listlablpara, listscalpara, listlablroot, listlablunit, listlabltotl = tdpy.retr_listlablscalpara(listnamevarbtotl, boolmath=True)
    
    # turn lists of labels into dictionaries
    for k, strgvarb in enumerate(listnamevarbtotl):
        dictlabl['root'][strgvarb] = listlablroot[k]
        dictlabl['unit'][strgvarb] = listlablunit[k]
        dictlabl['totl'][strgvarb] = listlabltotl[k]
    
    dictmodl = dict()
    
    pathfoldanim = pathvisu
    
    # title for the plots
    strgtitl = retr_strgtitl(dictefes, listnamevarbcomp, dictlabl)
    
    #dicttemp['coeflmdk'] = np.array([dicttemp['coeflmdklinr'], dicttemp['coeflmdkquad']])
    
    # dictionary for the configuration
    dictmodl['eval'] = dict()
    dictmodl['eval']['time'] = dictefes['time'] # [BJD]
    
    numbtime = dictefes['rflx'].shape[0]
    numbener = dictefes['rflx'].shape[1]
    arrytser = np.empty((numbtime, numbener, 3))
    arrytser[:, 0, 0] = dictefes['time']
    arrytser[:, :, 1] = dictefes['rflx']
    arrypcur = [[] for j in indxcomp]
    for j in indxcomp:
        arrypcur[j] = miletos.fold_tser(arrytser, dictefes['epocmtracomp'][j], dictefes['pericomp'][j])

    #if dictlistvalubatc[namebatc]['vari'][nameparavari].size > 1:
    #    if not isinstance(dictlistvalubatc[namebatc]['vari'][nameparavari][k], str):
    #        dictmodl['eval']['labl'] = '%s = %.3g %s' % (dictlabl['root'][nameparavari], \
    #                            dictlistvalubatc[namebatc]['vari'][nameparavari][k], dictlabl['unit'][nameparavari])
    #    else:
    #        dictmodl['eval']['labl'] = '%s' % (dictlistvalubatc[namebatc]['vari'][nameparavari][k])
    
    listcolr = np.array(['g', 'b', 'firebrick', 'orange', 'olive'])

    duratrantotl = nicomedia.retr_duratrantotl(dictefes['pericomp'], dictefes['rsmacomp'], dictefes['cosicomp']) / 24. # [days]
    
    listxdatvert = [-0.5 * 24. * dictefes['duratrantotl'], 0.5 * 24. * dictefes['duratrantotl']] 
    if 'duratranfull' in dictefes:
        listxdatvert += [-0.5 * 24. * dictefes['duratranfull'], 0.5 * 24. * dictefes['duratranfull']]
    listxdatvert = np.array(listxdatvert)
    
    # title for the plots
    #for namevarbtotl in listnamevarbtotl:
    #    if namevarbtotl != nameparavari or dictlistvalubatc[namebatc]['vari'][nameparavari].size == 1:
    #        dictstrgtitl[namevarbtotl] = dictefes[namevarbtotl]
    #strgtitl = retr_strgtitl(dictstrgtitl, dictefes, listnamevarbcomp, dictlabl)
    lablyaxi = 'Relative flux - 1 [ppm]'
    
    if strgextninpt is None or strgextninpt == '':
        strgextnbase = '%s' % (dictefes['typesyst'])
        if typetarg != '':
            strgextnbase += '_%s' % typetarg
    else:
        strgextnbase = strgextninpt
    
    timeoffs = tdpy.retr_offstime(dictefes['time'])
        
    numbener = dictefes['rflx'].shape[1]
    indxener = np.arange(numbener)
    
    for e in indxener:
        
        if numbener > 0:
            strgener = '_e%03d' % e
        else:
            strgener = ''
        
        dictmodl['eval']['tser'] = 1e6 * (dictefes['rflx'][:, e] - 1)
        
        dictmodl['eval']['colr'] = 'b'
        
        # time-series
        strgextn = '%s%s' % (strgextnbase, strgener)
        pathplot = miletos.plot_tser(pathvisu, \
                                     dictmodl=dictmodl, \
                                     typefileplot=typefileplot, \
                                     timeoffs=timeoffs, \
                                     ydatcntr=0., \
                                     #listxdatvert=listxdatvert, \
                                    
                                     strgextn=strgextn, \
                                     lablyaxi=lablyaxi, \
                                     strgtitl=strgtitl, \
                                     #typesigncode='ephesos', \
                                    )
        
        if numbcomp == 1 and dictefes['typesyst'] == 'PlanetarySystemEmittingCompanion':
            for j in indxcomp:
                strgextnbasecomp = '%s%s_com%d' % (strgextnbase, strgener, j)
                
                epocmtra = dictefes['epocmtracomp'][j]
                peri = dictefes['pericomp'][j]
                duratrantotl = dictefes['duratrantotl'][j]
                
                print('Calling miletos.plot_tser() from ephesos.main for primary with boolfold=True and nophascntr specified. Companion %d, Energy %d' % (j, e))
                # horizontal zoom around the primary
                strgextn = '%s_prim' % (strgextnbasecomp)
                limtxaxi = np.array([-duratrantotl, duratrantotl]) / peri
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             epoc=epocmtra, \
                                             peri=peri, \
                                             typeplotback=typeplotback, \
                                             #typesigncode='ephesos', \
                                            )
                
                print('Calling miletos.plot_tser() from ephesos.main for secondary with boolfold=True and phascntr 0.5. Companion %d, Energy %d' % (j, e))
                # horizontal zoom around the secondary
                strgextn = '%s_seco' % (strgextnbasecomp)
                limtxaxi = 0.5 + np.array([-duratrantotl, duratrantotl]) / peri
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             phascntr=0.5, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             limtyaxi=[-0.2 * np.max(dictmodl['eval']['tser']), None], \
                                             epoc=epocmtra, \
                                             peri=peri, \
                                             typeplotback=typeplotback, \
                                             #typesigncode='ephesos', \
                                            )

                # full phase curve
                print('Calling miletos.plot_tser() from ephesos.main for full phase curve with boolfold=True and phascntr 0.25 specified. Companion %d, Energy %d' % (j, e))
                strgextn = '%s_pcur' % (strgextnbasecomp)
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             phascntr=0.25, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             epoc=epocmtra, \
                                             peri=peri, \
                                             typeplotback=typeplotback, \
                                             #typesigncode='ephesos', \
                                            )

                # vertical zoom onto the full phase curve
                strgextn = '%s_pcurzoom' % (strgextnbasecomp)
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             phascntr=0.25, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtyaxi=[-500, None], \
                                             epoc=epocmtra, \
                                             peri=peri, \
                                             typeplotback=typeplotback, \
                                             #typesigncode='ephesos', \
                                            )
                

def retr_strgtitl(dictefesinpt, listnamevarbcomp, dictlabl):
    '''
    Return the title of a plot with information about the system
    '''
    
    strgtitl = ''
    if 'radistar' in dictefesinpt and dictefesinpt['radistar'] is not None:
        strgtitl += '$R_*$ = %.1f $R_\odot$' % dictefesinpt['radistar']
    if dictefesinpt['typesyst'] == 'CompactObjectStellarCompanion' and 'massstar' in dictefesinpt:
        if len(strgtitl) > 0 and strgtitl[-2:] != ', ':
            strgtitl += ', '
        strgtitl += '$M_*$ = %.1f $M_\odot$' % dictefesinpt['massstar']
    
    for kk, name in enumerate(listnamevarbcomp):
        
        if name == 'epocmtracomp' or name == 'typebrgtcomp' or (not name[:-1] + 'p' in dictefesinpt and name in dictefesinpt):
            continue
        
        if name.startswith('epocmtracom'):
            continue

        if name.startswith('typebrgtcom'):
            continue

        if name in dictefesinpt:
            nameprim = name
        elif len(name.split('com')) == 2 and name.split('com')[1].isnumeric():
            nameprim = name.split('com')[0] + 'comp'
        else:
            print('name')
            print(name)
            raise Exception('')

        if dictefesinpt[nameprim] is None:
            print('')
            print('')
            print('')
            print('dictefesinpt[typesyst]')
            print(dictefesinpt['typesyst'])
            print('name')
            print(name)
            raise Exception('dictefesinpt[name] is None')

        #for j, valu in enumerate(dictefesinpt[nameprim]):
            
        valu = dictefesinpt[nameprim][int(name[-1])]

        if len(strgtitl) > 0 and strgtitl[-2:] != ', ':
            strgtitl += ', '
        
        if kk == 4:
            strgtitl += '\n'
        
        strgtitl += '%s = ' % dictlabl['root'][name]
        
        if name == 'typebrgtcomp':
            strgtitl += '%s' % (valu)
        else:
            strgtitl += '%.3g' % (valu)
        
        if name in dictlabl['unit'] and dictlabl['unit'][name] != '':
            strgtitl += ' %s' % dictlabl['unit'][name]
        
        #cntr += 1

    return strgtitl


