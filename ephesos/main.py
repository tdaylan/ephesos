import sys
import os

import numpy as np

from tqdm import tqdm

import json

import time as timemodu

from numba import jit, prange

import astropy
import astropy as ap
from astropy.io import fits
from astropy.io import fits
import astropy.timeseries

import multiprocessing

import scipy as sp
import scipy.interpolate

import miletos

import astroquery
import astroquery.mast

import matplotlib as mpl
import matplotlib.pyplot as plt

import nicomedia
import chalcedon

import tdpy
from tdpy import summgene


def prep_dist(gdat, j):
    
    # switch to the companion coordinate system
    if gdat.typecoor == 'comp':
        gdat.xposgridcompthis = gdat.xposgridcomp[j]
        gdat.yposgridcompthis = gdat.yposgridcomp[j]
    if gdat.typecoor == 'star':
        gdat.xposgridcompthis = gdat.xposgridstar - gdat.xposcompgridstar[j]
        gdat.yposgridcompthis = gdat.yposgridstar - gdat.yposcompgridstar[j]
    
    if gdat.boolsphr:
        # distance from the companion
        if gdat.typecoor == 'comp':
            gdat.distgridcompthis = gdat.distgridcomp[j]
        if gdat.typecoor == 'star':
            gdat.distgridcompthis = np.sqrt(gdat.xposgridcompthis**2 + gdat.yposgridcompthis**2)

        # distance from the star
        if gdat.typecoor == 'comp':
            gdat.distgridstarthis = np.sqrt((gdat.xposgridcomp[j] - gdat.xposstargridcomp)**2 + (gdat.yposgridcomp[j] - gdat.yposstargridcomp)**2)
        if gdat.typecoor == 'star':
            gdat.distgridstarthis = gdat.distgridstar 


def retr_boolnoccobjt(gdat, j, typeoccu='comp'):
    '''
    Return a grid of Booleans indicating which prid points are not occulted
    '''
    
    if gdat.boolsphr:
        if typeoccu == 'comp':
            boolnocccomp = gdat.distgridcompthis > gdat.rratcomp[j]
        
        if typeoccu == 'star':
            boolnocccomp = gdat.distgridstarthis > 1.
        
    elif gdat.typesyst == 'turkey':
        positemp = np.vstack([gdat.xposgridcompthis.flatten(), gdat.yposgridcompthis.flatten()]).T
        indx = np.where((abs(positemp[:, 0]) < gdat.maxmxposturkmesh) & (abs(positemp[:, 1]) < gdat.maxmyposturkmesh))[0]

        boolnocccomp = np.ones(gdat.yposgridcompthis.size, dtype=bool)
        boolnocccomp[indx] = scipy.interpolate.griddata(gdat.positurkmesh, gdat.valuturkmesh, positemp[indx, :], fill_value=0.) < 0.5

        boolnocccomp = boolnocccomp.reshape(gdat.xposgridcompthis.shape)

    elif gdat.typesyst == 'psysdiskedgehori':
        booldisk = (gdat.xposgridcompthis / 1.75 / gdat.rratcomp[j])**2 + (gdat.yposgridcompthis / 0.2 / gdat.rratcomp[j])**2 > 1.
        boolnocccomp = boolnocccomp & booldisk
       
    elif gdat.typesyst == 'psysdiskedgevert':
        booldisk = (gdat.yposgridcompthis / 1.75 / gdat.rratcomp[j])**2 + (gdat.xposgridcompthis / 0.2 / gdat.rratcomp[j])**2 > 1.
        boolnocccomp = boolnocccomp & booldisk
       
    elif gdat.typesyst == 'psysdiskface':
        boolnocccomp = boolnocccomp & ((gdat.distgridcompthis > 1.5 * gdat.rratcomp[j]) & (gdat.distgridcompthis < 1.75 * gdat.rratcomp[j]))
    
    else:
        print('gdat.typesyst')
        print(gdat.typesyst)
        raise Exception('')
    
    return boolnocccomp


def retr_fluxstartrantotl(gdat, typecoor, boolrofi):
    '''
    Calculate the total flux of a star on the grid
    '''
    indxgridrofi = np.where(boolrofi)
    fluxstartran = retr_fluxstartran(gdat, typecoor, indxgridrofi)
    
    fluxstartrantotl = np.sum(fluxstartran)

    return fluxstartrantotl


def retr_fluxstartran(gdat, typecoor, indxgridrofi=None):
    '''
    Calculate the relative flux from a brightness map
    '''
    
    if typecoor == 'comp' or typecoor == 'sour':
        if typecoor == 'comp':
            dist = gdat.diststargridcomp
            areagrid = gdat.areagrid
        if typecoor == 'sour':
            dist = gdat.diststargridsour
            areagrid = gdat.areagridsour
        
        if indxgridrofi is not None:
            dist = dist[indxgridrofi]

        cosg = np.sqrt(1. - dist**2)
        fluxstartran = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)# * areagrid
        
        if gdat.booldiag and (abs(fluxstartran) > 1e10).any() or not np.isfinite(fluxstartran).all():
            print('dist')
            summgene(dist)
            print('fluxstartran')
            print(fluxstartran)
            print('typecoor')
            print(typecoor)
            raise Exception('')

    if typecoor == 'star':
        fluxstartran = gdat.brgtgridstar[indxgridrofi]
            
    return fluxstartran


def make_framanim(gdat, t, phasthis, j=None):
    
    for namevarbanim in gdat.listnamevarbanim:
        
        if gdat.indxframthis != 0 and namevarbanim in ['posifrstphotlens', 'posisecophotlens', 'cntsfrstlens', 'cntssecolens']:
            continue

        if not os.path.exists(gdat.pathgiff[namevarbanim]):
            
            path = gdat.pathvisu + '%s%s%s_%04d.%s' % (namevarbanim, gdat.strgextn, gdat.strgcompmoon, t, gdat.typefileplot)
        
            gdat.cmndmakeanim[namevarbanim] += ' %s' % path
            gdat.cmnddeleimag[namevarbanim] += ' %s' % path
        
            figr, axis = plt.subplots(figsize=(6, 6))
            
            if namevarbanim == 'flux':
                if gdat.typecoor == 'comp':
                    
                    if gdat.boolsystpsys:
                        
                        # brightness on the companion grid points
                        brgttemp = np.zeros_like(gdat.xposgridcomp[j])
                        
                        # calculate the brightness due to primary
                        brgttemp[gdat.indxgridcompstarnocc] = gdat.brgtprim
                        
                        if gdat.typebrgtcomp != 'dark':
                            # companion brightness
                            brgttemp[gdat.indxplannoccgridcomp[j]] = gdat.brgtcomp
                
                if gdat.typecoor == 'star':
                    if gdat.boolsystpsys:
                        brgttemp = np.zeros_like(gdat.brgtgridstar)
                        brgttemp[gdat.boolgridstarbrgt] = gdat.brgtgridstar[gdat.boolgridstarbrgt]
                
                if gdat.typesyst == 'cosc':
                    brgttemp = gdat.brgtlens
                
                cmap = 'magma'
                #cmap = 'Blues_r'
                cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white","blue"])
                imag = axis.imshow(brgttemp, origin='lower', interpolation='nearest', cmap=cmap, vmin=0., vmax=gdat.maxmbrgtstar)
        
                if gdat.boolshowlcuranim:
                    axistser = figr.add_axes([0.2, 0.15, 0.6, 0.3], frameon=False)
                
                    if j is None:
                        axistser.plot(gdat.time[:t], gdat.fluxtotl[:t], marker='', color='firebrick', ls='-', lw=1)
                    
                    else:
                        phastemp = np.array(gdat.phascomp[j])
                        indx = np.argsort(phastemp)
                        axistser.plot(phastemp[indx], np.array(gdat.fluxtotlcomp[j])[indx], marker='', color='firebrick', ls='-', lw=1)
                    
                        minmydat = gdat.brgtstarnocc - 2. * gdat.rratcomp[j]**2 * gdat.brgtstarnocc
                        maxmydat = gdat.brgtstarnocc + 2. * gdat.rratcomp[j]**2 * gdat.brgtstarnocc
                        axistser.set_ylim([minmydat, maxmydat])
                        axistser.set_xlim([-0.25, 0.75])
                    
                        #print('gdat.phascomp[j]')
                        #summgene(gdat.phascomp[j])
                        #print('gdat.fluxtotlcomp[j]')
                        #summgene(gdat.fluxtotlcomp[j])
                    
                    #xlim = 2. * 0.5 * np.array([-gdat.duratrantotl[j] / gdat.pericomp[j], gdat.duratrantotl[j] / gdat.pericomp[j]])
                    #axistser.set_xlim(xlim)
                    
                    axistser.axis('off')

            if namevarbanim == 'posifrstphotlens':
    
                imag = axis.scatter(gdat.xposfrst, gdat.yposfrst, s=0.001)

            if namevarbanim == 'posisecophotlens':
    
                imag = axis.scatter(gdat.xposseco, gdat.yposseco, s=0.001)
            

            if namevarbanim == 'brgtgridsour':
                gdat.brgtgridsourimag = np.zeros_like(gdat.xposgridsour)
                gdat.brgtgridsourimag[gdat.indxgridsourstar]
                imag = axis.imshow(gdat.brgtgridsourimag, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmbrgtstarsour)
            
            if namevarbanim == 'fluxfrstlens':
                imag = axis.imshow(gdat.fluxfrstlens, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmbrgtstar)
            
            if namevarbanim == 'fluxsecolens':
                imag = axis.imshow(gdat.fluxsecolens, origin='lower', interpolation='nearest', cmap='magma', vmin=0., vmax=gdat.maxmbrgtstar)
            
            if namevarbanim == 'cntsfrstlens':
                imag = axis.imshow(gdat.cntsfrstlens, origin='lower', interpolation='nearest', cmap='magma')
            
            if namevarbanim == 'cntssecolens':
                imag = axis.imshow(gdat.cntssecolens, origin='lower', interpolation='nearest', cmap='magma')
            
            axis.set_aspect('equal')
            axis.axis('off')
            
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
                    #timemtra = gdat.phascomp[j][-1] * gdat.pericomp[j] * 24.
                    timemtra = gdat.phascomp[j][-1] * gdat.pericomp[j] * 24. * 60.
                    #strgtextfinl = 'hour'
                    strgtextfinl = 'minutes'
                    if gdat.typelang == 'Turkish':
                        strgtextinit = gdat.dictturk[strgtextinit]
                        strgtextfinl = gdat.dictturk[strgtextfinl]
                    strgtext = '%s: %.2f %s \n Phase: %.3g' % (strgtextinit, timemtra, strgtextfinl, gdat.phascomp[j][-1])
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
    
    gdat.indxframthis += 1

    
def calc_brgtprim(gdat, j, phasthis):
    
    ## determine the pixels over which the stellar brightness will be calculated
    if abs(phasthis) < 0.25:
        # Booleans indicating where the primary is not occulted in the companion grid
        gdat.boolstarnoccgridcomp = gdat.booloutsplangridcomp[j] & gdat.boolstargridcomp

        # indices of the companion grid where the primary is not occulted
        gdat.indxgridcompstarnocc = np.where(gdat.boolstarnoccgridcomp)
    else:
        gdat.indxgridcompstarnocc = gdat.boolstargridcomp
    cosg = np.sqrt(1. - gdat.diststargridcomp[gdat.indxgridcompstarnocc]**2)
    brgtprim = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)# * gdat.areagrid
    
    return brgtprim


def retr_brgtlens(gdat, t, phasthis):
    
    print('phasthis')
    print(phasthis)
    
    # brightness in the sources plane
    gdat.brgtgridsour = retr_fluxstartran(gdat, 'sour', gdat.indxgridsourstar)
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

        if (not np.isfinite(flux).all() or abs(phasthis) < 1e-3 and (flux == 0.).all()):
            print('')
            print('')
            print('')
            print('')
            print('')
            raise Exception('')

    return flux


def calc_posifromphas(gdat, j, phastemp):
    '''
    Calculate body positions from phase
    '''
    xpos = gdat.smaxcomp[j] * np.sin(2. * np.pi * phastemp)
    ypos = gdat.smaxcomp[j] * np.cos(2. * np.pi * phastemp) * gdat.cosicomp[j] * gdat.intgcompflip[j]
    zpos = gdat.smaxcomp[j] * np.cos(2. * np.pi * phastemp)
    
    if gdat.typecoor == 'star':
        gdat.xposcompgridstar[j] = xpos
        gdat.yposcompgridstar[j] = ypos
        gdat.zposcompgridstar[j] = zpos
        
        if gdat.perimoon is not None:
            for jj in indxmoon[j]:
                gdat.xposmoon[j][jj] = gdat.xposcompgridstar[j] + \
                                smaxmoon[j][jj] * np.cos(2. * np.pi * (gdat.time - epocmtramoon[j][jj]) / gdat.perimoon[j][jj]) / gdat.radistar * gdat.dictfact['aurs']
                gdat.yposmoon[j][jj] = gdat.yposcompgridstar[j] + \
                                smaxmoon[j][jj] * np.sin(2. * np.pi * (gdat.time - epocmtramoon[j][jj]) / gdat.perimoon[j][jj]) / gdat.radistar * gdat.dictfact['aurs']

    if gdat.typecoor == 'comp':
        gdat.xposstargridcomp[j] = -xpos
        gdat.yposstargridcomp[j] = -ypos
        gdat.zposstargridcomp[j] = -zpos
        

def proc_phas(gdat, j, t, phasthis):
    
    calc_posifromphas(gdat, j, phasthis)
    
    if gdat.typesyst == 'psyspcur':
        
        boolevaltranprim = True

    else:
        if gdat.typecoor == 'comp':
            
            if gdat.boolsystpsys and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + gdat.factwideanim * gdat.rratcomp[j]):
                boolevaltranprim = True
            elif gdat.typesyst == 'cosc' and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + gdat.factwideanim * gdat.wdthslen[j]):
                boolevaltranprim = True
            else:
                boolevaltranprim = False
            
            #if gdat.typesyst == 'cosc':
            #    print('temp')
            #    boolevaltranprim = True
        
        if gdat.typecoor == 'star':
            if gdat.boolsystpsys and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + gdat.factwideanim * gdat.rratcomp[j]):
                boolevaltranprim = True
            elif gdat.typesyst == 'cosc' and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + gdat.factwideanim * gdat.wdthslen[j]):
                boolevaltranprim = True
            else:
                boolevaltranprim = False

    #print('t')
    #print(t)
    #print('phasthis')
    #print(phasthis)
    #print('boolevaltranprim')
    #print(boolevaltranprim)
    #print('')

    if boolevaltranprim:
        
        if gdat.typecoor == 'comp':
            
            if gdat.typebndr == 'view' and gdat.maxmfactellp > 1:
                factelli = 1. + (gdat.maxmfactellp - 1.) * np.sin(2. * np.pi * phasthis)**2
            else:
                factelli = 1.

            # distance from the companion grid points to the star
            gdat.diststargridcomp = np.sqrt(((gdat.xposgridcomp[j] - gdat.xposstargridcomp[j]) / factelli)**2 + \
                                            ((gdat.yposgridcomp[j] - gdat.yposstargridcomp[j]))**2)
        
            # Booleans indicating whether companion grid points are within the star
            gdat.boolstargridcomp = gdat.diststargridcomp < 1.
        
        if gdat.boolsystpsys:
            prep_dist(gdat, j)
        
        # brightness of the primary
        if gdat.typebndr == 'view' and gdat.maxmfactellp > 1:
            gdat.brgtprim = calc_brgtprim(gdat, j, phasthis)
            fluxtotlcompthis = np.sum(gdat.brgtprim)
        else:
            gdat.brgtprim = gdat.brgtstarnocc
        fluxtotlcompthis = gdat.brgtprim

        if gdat.booldiag:
            if gdat.typesyst == 'psys':
                if fluxtotlcompthis > gdat.brgtstarnocc:
                    print('')
                    print('')
                    print('')
                    print('fluxtotlcompthis')
                    print(fluxtotlcompthis)
                    print('gdat.brgtstarnocc')
                    print(gdat.brgtstarnocc)
                    raise Exception('gdat.typesyst == psys and fluxtotlcompthis > gdat.brgtstarnocc')

        # brightness of the companion
        if gdat.typebrgtcomp != 'dark':
            if gdat.typecoor == 'comp':
                
                if abs(phasthis) > 0.25:
                    gdat.indxplannoccgridcomp[j] = gdat.indxplangridcomp[j]
                else:
                    # Booleans indicating the region outside the star in the companion grid
                    gdat.booloutsstargridcomp = retr_boolnoccobjt(gdat, j, typeoccu='star')
                    
                    # Booleans indicating the planetary region region outside the star in the companion grid
                    gdat.boolplannoccgridcomp = gdat.booloutsstargridcomp & gdat.boolplangridcomp[j]

                    gdat.indxplannoccgridcomp[j] = np.where(gdat.boolplannoccgridcomp)
               
                # calculate the brightness of the planet
                # cosine of gamma
                cosg = np.sqrt(1. - gdat.distgridcomp[j][gdat.indxplannoccgridcomp[j]]**2)
                
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
                
                gdat.brgtcomp = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, brgtraww=brgtraww, typelmdk=gdat.typelmdk)# * gdat.areagrid
                
                fluxtotlcompthis += np.sum(gdat.brgtcomp)
                
                #if abs(phasthis - 0.5) < 0.1:
                #    print('proc_phas()')
                #    print('phasthis')
                #    print(phasthis)
                #    #print('gdat.xposstargridcomp[j]')
                #    #print(gdat.xposstargridcomp[j])
                #    #print('gdat.yposstargridcomp[j]')
                #    #print(gdat.yposstargridcomp[j])
                #    print('phii')
                #    print(phii)
                #    print('thet')
                #    print(thet)
                #    print('gdat.longgridsphr[j]')
                #    summgene(gdat.longgridsphr[j])
                #    print('gdat.indxplannoccgridcomp[j][0]')
                #    summgene(gdat.indxplannoccgridcomp[j][0])
                #    print('gdat.indxplannoccgridcomp[j][1]')
                #    summgene(gdat.indxplannoccgridcomp[j][1])
                #    print('gdat.longgridsphr[j][gdat.indxplannoccgridcomp[j]])')
                #    summgene(gdat.longgridsphr[j][gdat.indxplannoccgridcomp[j]])
                #    print('gdat.latigridsphr[j][gdat.indxplannoccgridcomp[j]]')
                #    summgene(gdat.latigridsphr[j][gdat.indxplannoccgridcomp[j]])
                #    print('brgtraww')
                #    summgene(brgtraww)
                #    #print('gdat.brgtstarnocc')
                #    #print(gdat.brgtstarnocc)
                #    #print('np.sum(temp)')
                #    #print(np.sum(temp))
                #    print('')
                #    print('')
                #    print('')
        
        if gdat.booldiag:
            if gdat.typesyst == 'psys':
                if fluxtotlcompthis > gdat.brgtstarnocc:
                    print('')
                    print('')
                    print('')
                    print('fluxtotlcompthis')
                    print(fluxtotlcompthis)
                    print('gdat.brgtstarnocc')
                    print(gdat.brgtstarnocc)
                    raise Exception('gdat.typesyst == psys and fluxtotlcompthis > gdat.brgtstarnocc')

        if gdat.boolsystpsys:
            
            if abs(phasthis) < 0.25:
                
                if gdat.typecoor == 'comp':

                    if gdat.maxmfactellp == 1:
                        # Booleans indicating whether companion grid points are within the star and occulted
                        gdat.boolstaroccugridcomp = gdat.boolstargridcomp & gdat.boolinsicompgridcomp[j]
                        
                        # stellar flux occulted
                        deltflux = -retr_fluxstartrantotl(gdat, gdat.typecoor, gdat.boolstaroccugridcomp)
                        
                        fluxtotlcompthis += deltflux
                
                if gdat.typecoor == 'star':

                    # Booleans indicating whether companion grid points are NOT occulted
                    boolnocccomp = retr_boolnoccobjt(gdat, j)
                    
                    gdat.boolgridstarbrgt = gdat.boolgridstarstar & boolnocccomp
                    
                    fluxtotlcompthis += retr_fluxstartrantotl(gdat, gdat.typecoor, gdat.boolgridstarbrgt)
        
        if gdat.typesyst == 'psyslasr':
            gdat.boolgridcompthislasr = gdat.distgridcompthis < 0.5 * gdat.rratcomp[j]

    
        if gdat.typesyst == 'cosc':
            # distance from the points in the source grid to the star
            gdat.diststargridsour = np.sqrt((gdat.xposgridsour - gdat.xposstargridcomp[j])**2 + (gdat.yposgridsour - gdat.yposstargridcomp[j])**2)
            
            # Booleans indicating whether source grid points are within the star
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
                
            #if gdat.typecoor == 'comp':
            #    # distance from the points in the companion grid to the star
            #    gdat.diststargridcomp = np.sqrt((gdat.xposgridcomp[j] - gdat.xposstargridcomp[j])**2 + (gdat.yposgridcomp[j] - gdat.yposstargridcomp[j])**2)
            
            # calculate the lensed brightness within the companion grid
            gdat.brgtlens = retr_brgtlens(gdat, t, phasthis)
            
            print('np.sum(gdat.brgtlens)')
            print(np.sum(gdat.brgtlens))

            # calculate the brightness within the companion grid
            indxgridrofi = np.where(gdat.boolstargridcomp)
            gdat.brgtstarplan = np.sum(retr_fluxstartran(gdat, 'comp', indxgridrofi))
            
            print('gdat.brgtstarplan')
            print(gdat.brgtstarplan)
            print('gdat.brgtstarnocc - gdat.brgtstarplan')
            print(gdat.brgtstarnocc - gdat.brgtstarplan)

            fluxtotlfram = np.sum(gdat.brgtlens) + gdat.brgtstarnocc - gdat.brgtstarplan
            
            print('fluxtotlfram')
            print(fluxtotlfram)

            if booldiag:
                if fluxtotlfram / gdat.brgtstarnocc < 0.5 or fluxtotlfram <= 0.:
                    print('')
                    print('')
                    print('')
                    print('gdat.brgtlens')
                    summgene(gdat.brgtlens)
                    print('fluxtotlfram')
                    print(fluxtotlfram)
                    print('gdat.brgtstarnocc')
                    print(gdat.brgtstarnocc)
                    raise Exception('fluxtotlfram / gdat.brgtstarnocc < 0.5 or fluxtotlfram <= 0.')
                if fluxtotlfram == 0.:
                    print('')
                    print('')
                    print('')
                    print('fluxtotlfram')
                    print(fluxtotlfram)
                    raise Exception('fluxtotlfram == 0.')
            
            fluxtotlcompthis += fluxtotlfram
        
        if gdat.booldiag:
            if gdat.typesyst == 'psys':
                if fluxtotlcompthis > gdat.brgtstarnocc:
                    print('')
                    print('')
                    print('')
                    print('fluxtotlcompthis')
                    print(fluxtotlcompthis)
                    print('gdat.brgtstarnocc')
                    print(gdat.brgtstarnocc)
                    raise Exception('gdat.typesyst == psys and fluxtotlcompthis > gdat.brgtstarnocc')

        gdat.fluxtotlcomp[j].append(fluxtotlcompthis)
        
        gdat.phascomp[j].append(phasthis)

        if gdat.boolmakeanim:
            make_framanim(gdat, t, phasthis, j=j)
        
        if gdat.typeverb > 1:
            print('')
            print('')
            print('')
            print('')
            print('')
            print('')
        if gdat.booldiag:
            if abs(gdat.fluxtotlcomp[j][-1]) > 1e20:
                print('jt')
                print(j, t)
                print('gdat.fluxtotlcomp[j]')
                summgene(gdat.fluxtotlcomp[j])
                raise Exception('')


def eval_modl( \
              # times in days at which to evaluate the relative flux
              time, \
              
              # type of the model system
              ## 'psys': planetary system
              typesyst, \
              
              # companions
              ## orbital periods of the companions
              pericomp, \
              
              ## mid-transit epochs of the companions
              epocmtracomp, \
              
              ## sum of stellar and companion radius
              rsmacomp=None, \
              
              ## cosine of the orbital inclination
              cosicomp=None, \
              
              ## orbital inclination
              inclcomp=None, \
              
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
              
              ## type of the brightness of the companion
              ### it is only functional if typesyst is 'psyspcur'
              ### 'dark': companion completely dark
              ### 'heated_rdis': companion is an externally heated body with heat redistribution (efficiency and phase offset)
              ### 'heated_sliced': companion has a heat distribution determined by the input temperatures of longitudinal slices
              ### 'isot': companion is an internally heated, isothermal body
              typebrgtcomp='dark', \
              
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
              ## 'phdy': photodynamically calculated
              ## 'gaus': Gaussian
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

              # type of coordinate system
              ## 'star': centered on the star
              ## 'comp': centered on the companion
              typecoor=None, \
             
              # a string indicating the type of normalization
              ## 'none': no detrending
              ## 'medi': by median
              ## 'nocc': by unocculted stellar brightness
              ## 'edgeleft': left edge
              ## 'maxm': maximum value
              typenorm='nocc', \

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

              ## minimum radius ratio tolerated
              tolerrat=None, \

              ## the spatial resolution of the grid over which the planet's brightness and occultation are evaluated
              resoplan=None, \
              
              # Boolean flag to check if the computation can be accelerated
              boolfast=True, \
              
              # the maximum factor by which the primary will get tidally deformed
              maxmfactellp=None, \

              # type of visualization
              ## 'real': dark background
              ## 'cart': bright background, colored planets
              typevisu='real', \
              
              # type of the boundary of the grid
              ## 'calc': optimized for fast calculation
              ## 'view': large field-of-view unneccessary for calculation
              typebndr='calc', \
            
              # type of light curve plot
              ## 'inst': inset
              ## 'lowr': lower panel
              typeplotlcurposi='inst', \
              
              # type of the limit of the light curve visualized
              ## 'wind': window
              ## 'tran': around-transit
              typeplotlcurlimt='wind', \
              
              # type of the visualization of the planet
              # 'real': real
              # 'colr': cartoon
              # 'colrtail': cartoon with tail
              # 'colrtaillabl': cartoon with tail and label
              typeplotplan='real', \

              # Boolean flag to add Mercury
              boolinclmerc=False, \

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
    timeinit = timemodu.time()
    
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
    
    numbcomp = gdat.pericomp.size
    indxcomp = np.arange(numbcomp)
    
    gdat.intgcompflip = np.random.randint(2, size=numbcomp) - 1
    
    if gdat.typesyst == 'psyspcur':
        if gdat.offsphascomp is None:
            gdat.offsphascomp = np.zeros(numbcomp)

    if gdat.tolerrat is None:
        gdat.tolerrat = 3e-3
    
    if gdat.typecoor is None:
        if numbcomp == 1 and gdat.typesyst != 'psysmoon':
            gdat.typecoor = 'comp'
        else:
            gdat.typecoor = 'star'

    gdat.boolsystpsys = gdat.typesyst.startswith('psys')
    
    # check inputs
    if gdat.booldiag:
        
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

        if tolerrat is not None and not isinstance(tolerrat, float):
            print('tolerrat')
            print(tolerrat)
            raise Exception('')

        if resoplan is not None and not isinstance(resoplan, float):
            print('resoplan')
            print(resoplan)
            raise Exception('')
        
        if np.isscalar(gdat.rratcomp):
            raise Exception('')
        
        
        if gdat.typesyst.startswith('psys') and (gdat.rratcomp is None or not np.isfinite(gdat.rratcomp).all()):
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
        
        if boolintp and gdat.typecoor == 'star':
            raise Exception('')

        if boolintp and numbcomp > 1:
            raise Exception('')

        if gdat.boolsystpsys and not ((gdat.radistar is not None and gdat.radicomp is not None) or gdat.rratcomp is not None):
            print('gdat.radistar')
            print(gdat.radistar)
            print('gdat.radicomp')
            print(gdat.radicomp)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            raise Exception('')
        
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

        if gdat.typesyst == 'cosc':
            
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
        
        if gdat.typesyst != 'cosc' and (gdat.rsmacomp is None or not np.isfinite(gdat.rsmacomp).all()):
            print('')
            print('')
            print('')
            print('gdat.rsmacomp')
            print(gdat.rsmacomp)
            raise Exception('gdat.rsmacomp is None or not np.isfinite(gdat.rsmacomp).all()')
        
        if gdat.typesyst == 'cosc':
            if gdat.massstar is None:
                raise Exception('')
        
            if gdat.masscomp is None:
                raise Exception('')
    
    if gdat.boolfast and gdat.boolsystpsys and gdat.rratcomp.ndim == 2:
        print('np.std(gdat.rratcomp)')
        print(np.std(gdat.rratcomp))
        print('gdat.rratcomp')
        summgene(gdat.rratcomp)
        if np.std(gdat.rratcomp) < 0.05 and numbcomp == 1:
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
    
    if typeverb > 1:
        print('gdat.typesyst')
        print(gdat.typesyst)
        print('gdat.typecoor')
        print(gdat.typecoor)
        print('boolintp')
        print(boolintp)
    
    if gdat.typesyst == 'cosc':
        if typeverb > 1:
            print('typemodllens')
            print(typemodllens)

        if gdat.radicomp is not None:
            if typeverb > 0:
                print('Warning from ephesos! A radius was provided for the compact object...')
        else:
            gdat.radicomp = np.array([0.])

    if gdat.perimoon is not None:
        indxcomp = np.arange(numbcomp)
        numbmoon = np.empty(numbcomp, dtype=int)
        for j in indxcomp:
            numbmoon[j] = len(gdat.perimoon[j])
        indxmoon = [np.arange(numbmoon[j]) for j in indxcomp]
    
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
    numbtime = gdat.time.size
    indxtime = np.arange(numbtime)

    gdat.dictfact = tdpy.retr_factconv()
    
    if gdat.pathvisu is not None:
        
        # path for animations
        gdat.dictturk = tdpy.retr_dictturk()

        if gdat.boolmakeanim:
            gdat.indxframthis = 0

            gdat.pathgiff = dict()
            gdat.cmndmakeanim = dict()
            gdat.cmnddeleimag = dict()
            gdat.listnamevarbanim = ['flux']
            if gdat.typesyst == 'cosc':
                gdat.listnamevarbanim += ['posifrstphotlens', 'posisecophotlens', 'fluxfrstlens', 'fluxsecolens']#, 'brgtgridsour' 'cntsfrstlens', 'cntssecolens']
    
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
    
    gdat.radistar = 1.
    
    if typeverb > 1:
        print('gdat.coeflmdk')
        print(gdat.coeflmdk)
        print('gdat.radistar [RS]')
        print(gdat.radistar)
        print('gdat.rsmacomp')
        print(gdat.rsmacomp)
        print('gdat.cosicomp')
        print(gdat.cosicomp)
    
    if gdat.rsmacomp is not None:
        gdat.smaxcomp = (1. + gdat.rratcomp) / gdat.rsmacomp
    
    if gdat.typesyst == 'psyspcur':
    
        if gdat.maxmfactellp is None:
            gdat.maxmfactellp = 1.2
        
        if gdat.typebrgtcomp == 'heated_rdis' or gdat.typebrgtcomp == 'heated_sliced':
            if gdat.ratibrgtcomp is not None:
                raise Exception('A brightness ratio is provided for a passively heated companion.')
            gdat.ratibrgtcomp = (1. / gdat.smaxcomp)**2
        
            print('temp: fudge factor due to passband in the IR')
            gdat.ratibrgtcomp *= 5.
            
        if gdat.ratibrgtcomp is None:
            if gdat.typebrgtcomp == 'isot':
                gdat.ratibrgtcomp = 1.

    if gdat.masscomp is not None and gdat.massstar is not None:
        
        if gdat.typesyst == 'psys' or gdat.typesyst == 'psyspcur':
            gdat.masscompsolr = gdat.masscomp / gdat.dictfact['msme']
        elif gdat.typesyst == 'cosc' or gdat.typesyst == 'sbin':
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
            smaxmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            for j in indxcomp:
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
        gdat.eccecomp = np.zeros(numbcomp)
    if gdat.sinwcomp is None:
        gdat.sinwcomp = np.zeros(numbcomp)
    
    if typeverb > 1:
        print('gdat.smaxcomp [R_star]')
        print(gdat.smaxcomp)
        print('indxcomp')
        print(indxcomp)
        if gdat.perimoon is not None:
            print('gdat.perimoon [days]')
            print(gdat.perimoon)
            print('radimoon [RE]')
            print(radimoon)
            print('smaxmoon [AU]')
            for smaxmoontemp in smaxmoon:
                print(smaxmoontemp)
    
    if boolintp is None:
        if gdat.perimoon is not None or numbcomp > 1 or gdat.perispot is not None:
            if typeverb > 1:
                print('Either the model has moon, stellar spots, or multiple companions.')
                print('Will evaluate the model at each time (as opposed to interpolating phase curves)...')
            boolintp = False
        else:
            if typeverb > 1:
                print('The model only has a single companion. Will interpolate the phase curve (as opposed to evaluating the model at each time)...')
            boolintp = True

    if boolcompmoon and boolintp:
        raise Exception('')

    gdat.boolsphr = gdat.typesyst == 'psys' or gdat.typesyst == 'psyspcur' or gdat.typesyst == 'psysmoon' or \
                                                            gdat.typesyst == 'psysttvr' or gdat.typesyst == 'psyslasr' or \
                    gdat.typesyst == 'psysdiskedgehori' or gdat.typesyst == 'psysdiskedgevert' or gdat.typesyst == 'psysdiskface' or gdat.typesyst == 'sbin'
    
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
        
    if gdat.typesyst == 'cosc':
        if gdat.typemodllens == 'gaus':
            gdat.dcyctrantotlhalf = gdat.smaxcomp / gdat.radistar / gdat.cosicomp
        if typeverb > 1:
            print('gdat.masscomp')
            print(gdat.masscomp)
            print('gdat.massstar')
            print(gdat.massstar)
        amplslen = chalcedon.retr_amplslen(gdat.pericomp, gdat.radistar, gdat.masscomp, gdat.massstar)
        dictefes['amplslen'] = amplslen
    
    if boolintp:
        gdat.phascomp = [[] for j in indxcomp]
        timecomp = [[] for j in indxcomp]
    
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
                if gdat.typesyst == 'sbin' and (gdat.rratcomp == 0).any():
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

        numbtimecomp = [[] for j in indxcomp]
        indxtimecomp = [[] for j in indxcomp]
        
        if gdat.diffphasineg is None:
            if typesyst == 'cosc':
                gdat.diffphasineg = 0.00005
            else:
                if numbcomp == 1 and gdat.rratcomp[0] > 0.5:
                    gdat.diffphasineg = 0.01
                else:
                    gdat.diffphasineg = 0.0003
        
        if gdat.diffphaspcur is None:
            gdat.diffphaspcur = 0.02
        
        if gdat.diffphasintr is None:
            if gdat.boolsystpsys and np.isfinite(gdat.duratranfull):
                gdat.diffphasintr = 0.0005
            else:
                gdat.diffphasintr = 0.0001
        
        if typeverb > 1:
            if np.isfinite(gdat.duratranfull):
                print('gdat.diffphasineg')
                print(gdat.diffphasineg)
            print('gdat.diffphasintr')
            print(gdat.diffphasintr)
            print('gdat.diffphaspcur')
            print(gdat.diffphaspcur)

    phas = [[] for j in indxcomp]
    for j in indxcomp:
        
        if typeverb > 1:
            print('j')
            print(j)
        
        phas[j] = ((gdat.time - gdat.epocmtracomp[j]) / gdat.pericomp[j] + 0.25) % 1. - 0.25
        
        if gdat.booldiag:
            if np.isscalar(phas[j]):
                raise Exception('')
        
        if typeverb > 1:
            print('phas[j]')
            summgene(phas[j])
    
    if gdat.typeverb > 0:
        if gdat.boolsystpsys and not (gdat.rratcomp > gdat.tolerrat).all():
            print('gdat.tolerrat')
            print(gdat.tolerrat)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            print('WARNING! At least one of the occulter radii is smaller than the grid resolution. The output will be unreliable.')
    
        if gdat.boolsystpsys and len(gdat.rratcomp) == 0:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            print('gdat.tolerrat')
            print(gdat.tolerrat)
            print('gdat.rratcomp')
            print(gdat.rratcomp)
            raise Exception('gdat.rratcomp is empty.')
    
    if typecalc == 'simpboxx':
        
        for j in indxcomp:
            indxtime = np.where((phas[j] < gdat.duratrantotl[j] / gdat.pericomp[j]) | (phas[j] > 1. -  gdat.duratrantotl[j] / gdat.pericomp[j]))[0]
            rflxtranmodl = np.ones_like(phas)
            rflxtranmodl[indxtime] -= gdat.rratcomp**2
    
    else:
    
        if gdat.typesyst == 'cosc':
            
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
            gdat.diffgrid = gdat.factresowdthslen * gdat.wdthslen[0]
            if gdat.diffgrid < 1e-3:
                if gdat.typeverb > 0:
                    print('The grid resolution needed to resolve the Einstein radius is %g, which is too small. Limiting the grid resolution at 0.001.' % \
                                                                                                                                                        gdat.diffgrid)
                gdat.diffgrid = 1e-3
            gdat.factosampsour = 0.1
            gdat.factsizesour = 1.
            
            if gdat.booldiag:
                if numbcomp > 1:
                    raise Exception('')

        elif (gdat.rratcomp <= gdat.tolerrat).any():
            gdat.diffgrid = 0.001
        else:
        
            if gdat.resoplan is None:
                gdat.resoplan = 0.1
            
            gdat.diffgrid = min(0.02, gdat.resoplan * np.amin(gdat.rratcomp[gdat.rratcomp > gdat.tolerrat]))
        
        if gdat.booldiag:
            if (gdat.rratcomp > 1).any():
                print('At least one of the radius ratios is larger than unity.')
                print('gdat.rratcomp')
                print(gdat.rratcomp)
            if gdat.diffgrid > 0.2:
                print('')
                print('')
                print('')
                print('The grid resolution is too low.')
                print('gdat.tolerrat')
                print(gdat.tolerrat)
                print('gdat.rratcomp')
                print(gdat.rratcomp)
                print('gdat.resoplan')
                print(gdat.resoplan)
                print('gdat.diffgrid')
                print(gdat.diffgrid)
                raise Exception('')

        if typeverb > 1:
            print('gdat.diffgrid')
            print(gdat.diffgrid)
        
        gdat.areagrid = gdat.diffgrid**2
        
        if gdat.typesyst == 'turkey':
            if numbcomp != 1:
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

        ## planet
        if gdat.typecoor == 'comp':
            
            gdat.xposgridcomp = [[] for j in indxcomp]
            gdat.yposgridcomp = [[] for j in indxcomp]
            gdat.zposgridcomp = [[] for j in indxcomp]
            
            gdat.xposgridsphr = [[] for j in indxcomp]
            gdat.yposgridsphr = [[] for j in indxcomp]
            gdat.zposgridsphr = [[] for j in indxcomp]
            
            gdat.distgridcomp = [[] for j in indxcomp]
            gdat.indxplannoccgridcomp = [[] for j in indxcomp]
            gdat.numbsidegridcomp = [[] for j in indxcomp]
            gdat.latisinugridcomp = [[] for j in indxcomp]
            gdat.laticosigridcomp = [[] for j in indxcomp]
            gdat.boolplangridcomp = [[] for j in indxcomp]
            gdat.indxplangridcomp = [[] for j in indxcomp]

            gdat.longgridsphr = [[] for j in indxcomp]
            gdat.latigridsphr = [[] for j in indxcomp]
            
            if gdat.boolsphr:
                
                # Booleans indicating the region outside the planet in the companion grid
                gdat.booloutsplangridcomp = [[] for j in indxcomp]

                # Booleans indicating the region inside the planet in the companion grid
                gdat.boolinsicompgridcomp = [[] for j in indxcomp]

            for j in indxcomp:
                
                if gdat.typesyst == 'cosc':
                    limtgridxpos = gdat.wdthslen[j]
                    limtgridypos = gdat.wdthslen[j]
                else:
                    limtgridxpos = gdat.rratcomp[j] * 1.5
                    
                    if gdat.typebndr == 'view':
                        limtgridypos = (gdat.smaxcomp[j] + 1. * gdat.maxmfactellp) * 1.05
                    else:
                        limtgridypos = gdat.rratcomp[j]
                    
                arrycompxpos = np.arange(-limtgridxpos - 2. * gdat.diffgrid, limtgridxpos + 3. * gdat.diffgrid, gdat.diffgrid)
                arrycompypos = np.arange(-limtgridypos - 2. * gdat.diffgrid, limtgridypos + 3. * gdat.diffgrid, gdat.diffgrid)
                gdat.numbsidegridcomp[j] = arrycompxpos.size
                
                gdat.xposgridcomp[j], gdat.yposgridcomp[j] = np.meshgrid(arrycompypos, arrycompxpos)
                
                gdat.xposgridsphr[j] = gdat.xposgridcomp[j]
                gdat.zposgridsphr[j] = gdat.yposgridcomp[j]
                gdat.yposgridsphr[j] = np.sqrt(gdat.rratcomp[j]**2 - gdat.xposgridsphr[j]**2 - gdat.zposgridsphr[j]**2)
                
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
                
                if gdat.typesyst == 'psyspcur' and gdat.tmptsliccomp is None and gdat.typebrgtcomp == 'heated_sliced':
                    raise Exception('')

                if gdat.tmptsliccomp is not None:
                    gdat.numbslic = len(gdat.tmptsliccomp)
                    gdat.brgtsliccomp = gdat.tmptsliccomp**4

                gdat.distgridcomp[j] = np.sqrt(gdat.xposgridcomp[j]**2 + gdat.yposgridcomp[j]**2)
                
                gdat.boolplangridcomp[j] = gdat.distgridcomp[j] < gdat.rratcomp[j]
                
                gdat.indxplangridcomp[j] = np.where(gdat.boolplangridcomp[j])

                if gdat.boolsphr:
                    gdat.booloutsplangridcomp[j] = gdat.distgridcomp[j] > gdat.rratcomp[j]
                
                    gdat.boolinsicompgridcomp[j] = ~gdat.booloutsplangridcomp[j]

                if typeverb > 1:
                    print('Number of pixels in the grid for companion %d: %d' % (j, gdat.xposgridcomp[j].size))
                    gdat.precphotflorcomp = 1e6 / gdat.xposgridcomp[j].size
                    print('Photometric precision floor achieved by this resolution: %g ppm' % gdat.precphotflorcomp)
            
        if gdat.typesyst == 'cosc' and gdat.typemodllens == 'phdy':
            gdat.diffgridsour = gdat.diffgrid / gdat.factosampsour
            
            gdat.areagridsour = gdat.diffgridsour**2

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
            gdat.distsourgridcomp = gdat.distgridcomp[0] - gdat.radieins**2 / gdat.distgridcomp[0]
            
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
            gdat.xposintp = gdat.xposgridcomp[0] / gdat.distgridcomp[0] * gdat.distsournormgridcomp
            gdat.yposintp = gdat.yposgridcomp[0] / gdat.distgridcomp[0] * gdat.distsournormgridcomp
            
            if typeverb > 1:
                print('gdat.xposintp')
                summgene(gdat.xposintp)
                print('gdat.yposintp')
                summgene(gdat.yposintp)
        
            gdat.arrygridsourintp = np.vstack([gdat.xposintp.flatten(), gdat.yposintp.flatten()]).T

        
        ## star
        arrystar = np.arange(-1. - 2. * gdat.diffgrid, 1. + 3. * gdat.diffgrid, gdat.diffgrid)
        gdat.xposgridstar, gdat.yposgridstar = np.meshgrid(arrystar, arrystar)
        gdat.precphotflorstar = 1e6 / gdat.xposgridstar.size
        
        if gdat.typeverb > 1:
            print('Number of pixels in the stellar grid: %d' % (gdat.xposgridstar.size))
            print('Photometric precision floor achieved by this resolution: %g ppm' % gdat.precphotflorstar)

        # distance to the star in the star grid
        gdat.distgridstar = np.sqrt(gdat.xposgridstar**2 + gdat.yposgridstar**2)
        
        # Booleans indicating whether star grid points are within the star
        gdat.boolgridstarstar = gdat.distgridstar < 1.
        
        # indices of the star grid points within the star
        gdat.indxgridstarstar = np.where(gdat.boolgridstarstar)
        
        # stellar brightness in the star grid
        gdat.brgtgridstar = np.zeros_like(gdat.xposgridstar)
        cosg = np.sqrt(1. - gdat.distgridstar[gdat.indxgridstarstar]**2)
        gdat.brgtgridstar[gdat.indxgridstarstar] = nicomedia.retr_brgtlmdk(cosg, gdat.coeflmdk, typelmdk=gdat.typelmdk)# * gdat.areagrid
        
        if gdat.typesyst == 'cosc':
            
            #metrcomp = 3e-10 * gdat.xposgridsour[0].size * gdat.xposgridcomp[0].size
            metrcomp = 3e-10 * gdat.xposgridstar.size * gdat.xposgridcomp[0].size
            
            if gdat.typeverb > 1:
                print('Estimated execution time per time sample: %g ms' % metrcomp)
                print('Estimated execution time: %g s' % (1e-3 * numbtime * metrcomp))
            
            if gdat.typecoor == 'star':
                arry = np.arange(-1. - 2.5 * gdat.diffgrid, 1. + 3.5 * gdat.diffgrid, gdat.diffgrid)
                gdat.binsxposgridstar = arry
                gdat.binsyposgridstar = arry
            if gdat.typecoor == 'comp':
                arry = np.arange(-gdat.wdthslen[j] - 2.5 * gdat.diffgrid, gdat.wdthslen[j] + 3.5 * gdat.diffgrid, gdat.diffgrid)
                gdat.binsxposgridcomp = arry
                gdat.binsyposgridcomp = arry
        
            # maximum stellar brightness for source grid
            if gdat.pathvisu is not None:
                gdat.maxmbrgtstarsour = nicomedia.retr_brgtlmdk(1., gdat.coeflmdk, typelmdk=gdat.typelmdk)# * gdat.areagridsour
        
        if gdat.boolmakeanim:
            gdat.factwideanim = 5.
        else:
            gdat.factwideanim = 1.1

        # maximum stellar brightness for planet and star grids
        if gdat.boolmakeanim:
            gdat.maxmbrgtstar = nicomedia.retr_brgtlmdk(1., gdat.coeflmdk, typelmdk=gdat.typelmdk)# * gdat.areagrid
        
        # total (unocculted) stellar birghtness
        gdat.brgtstarnocc = np.sum(gdat.brgtgridstar)
        
        if gdat.typeverb > 1:
            print('gdat.brgtstarnocc')
            print(gdat.brgtstarnocc)
        
        if boolintp:
            gdat.fluxtotlcomp = [[] for j in indxcomp]
        
        if gdat.typecoor == 'star':
            gdat.xposcompgridstar = [[] for j in indxcomp]
            gdat.yposcompgridstar = [[] for j in indxcomp]
            gdat.zposcompgridstar = [[] for j in indxcomp]
            if gdat.perimoon is not None:
                gdat.xposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
                gdat.yposmoon = [[[] for jj in indxmoon[j]] for j in indxcomp]
            
        if gdat.typecoor == 'comp':
            gdat.xposstargridcomp = [[] for j in indxcomp]
            gdat.yposstargridcomp = [[] for j in indxcomp]
            gdat.zposstargridcomp = [[] for j in indxcomp]
        
        if boolintp:
            listphaseval = [[] for j in indxcomp]
            for j in indxcomp:
                if np.isfinite(gdat.duratrantotl[j]):
                    gdat.phastrantotl = gdat.duratrantotl / gdat.pericomp[j]
                    if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                        gdat.phastranfull = gdat.duratranfull[j] / gdat.pericomp[j]
                        # inlclude a fudge factor of 1.1
                        deltphasineg = 1.1 * (gdat.phastrantotl - gdat.phastranfull) / 2.
                        phasingr = (gdat.phastrantotl + gdat.phastranfull) / 4.
                        deltphasineghalf = 0.5 * deltphasineg
                    else:
                        phasingr = gdat.phastrantotl / 2.
                        deltphasineghalf = 0.
                    
                    listphaseval[j] = [np.arange(-0.25, -phasingr - deltphasineghalf, gdat.diffphaspcur)]
                    
                    if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                        listphaseval[j].append(np.arange(-phasingr - deltphasineghalf, -phasingr + deltphasineghalf, gdat.diffphasineg))

                    listphaseval[j].append(np.arange(-phasingr + deltphasineghalf, phasingr - deltphasineghalf, gdat.diffphasintr))
                                                   
                    if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                        listphaseval[j].append(np.arange(phasingr - deltphasineghalf, phasingr + deltphasineghalf, gdat.diffphasineg))
                    
                    listphaseval[j].append(np.arange(phasingr + deltphasineghalf, 0.5 - phasingr - deltphasineghalf, gdat.diffphaspcur))
                    
                    if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                        listphaseval[j].append(np.arange(0.5 - phasingr - deltphasineghalf, 0.5 - phasingr + deltphasineghalf, gdat.diffphasineg))
                                                   
                    listphaseval[j].append(np.arange(0.5 - phasingr + deltphasineghalf, 0.5 + phasingr - deltphasineghalf, gdat.diffphasintr))
                                                   
                    if gdat.boolsystpsys and np.isfinite(gdat.duratranfull[j]):
                        listphaseval[j].append(np.arange(0.5 + phasingr - deltphasineghalf, 0.5 + phasingr + deltphasineghalf, gdat.diffphasineg))
                    
                    listphaseval[j].append(np.arange(0.5 + phasingr + deltphasineghalf, 0.75 + gdat.diffphaspcur, gdat.diffphaspcur))
                    
                    listphaseval[j] = np.concatenate(listphaseval[j])
                else:
                    listphaseval[j] = np.arange(-0.25, 0.75 + gdat.diffphaspcur, gdat.diffphaspcur)
            
        if boolcompmoon:
            numbitermoon = 2
        else:
            numbitermoon = 1
        
        gdat.fluxtotl = np.full(numbtime, gdat.brgtstarnocc)
        
        if gdat.typemodllens == 'gaus':
            if gdat.typesyst == 'cosc':
                gdat.fluxtotl += gdat.brgtstarnocc * np.exp(-(phas[0] / gdat.dcyctrantotlhalf[j])**2)
        else:
            for a in range(numbitermoon):
                
                if a == 0:
                    gdat.strgcompmoon = ''
                else:
                    gdat.strgcompmoon = '_onlycomp'
                
                if gdat.boolmakeanim:
                    for namevarbanim in gdat.listnamevarbanim:
                        gdat.pathgiff[namevarbanim] = gdat.pathvisu + 'anim%s%s%s.gif' % (namevarbanim, gdat.strgextn, gdat.strgcompmoon)
                        gdat.cmndmakeanim[namevarbanim] = 'convert -delay 5 -density 200'
                        gdat.cmnddeleimag[namevarbanim] = 'rm'
                
                if boolintp:
                
                    for j in indxcomp:
                        if not np.isfinite(gdat.duratrantotl[j]) or gdat.duratrantotl[j] == 0.:
                            continue
                         
                        if gdat.boolsystpsys and gdat.rratcomp[j] < gdat.tolerrat:
                            continue
                        
                        for t, phasthis in enumerate(listphaseval[j]):
                            proc_phas(gdat, j, t, phasthis)
                            
                else:
                    
                    for t in tqdm(range(numbtime)):
                    #for t in range(numbtime):
                        
                        # Boolean flag to evaluate the flux at this time
                        boolevaltranflux = False
                        
                        gdat.boolgridstarbrgt = np.copy(gdat.boolgridstarstar)
                                            
                        for j in indxcomp:
                            
                            if gdat.typeverb > 1:
                                print('j')
                                print(j)
                            
                            if gdat.rratcomp[j] < gdat.tolerrat:
                                continue

                            if abs(phas[j][t]) > 0.25 and gdat.typebrgtcomp == 'dark':
                                continue
                    
                            calc_posifromphas(gdat, j, phas[j][t])
                            if gdat.boolsystpsys:
                                if gdat.typecoor == 'comp' and (np.sqrt(gdat.xposstargridcomp[j]**2 + gdat.yposstargridcomp[j]**2) < 1. + gdat.rratcomp[j]) or \
                                   gdat.typecoor == 'star' and (np.sqrt(gdat.xposcompgridstar[j]**2 + gdat.yposcompgridstar[j]**2) < 1. + gdat.rratcomp[j]):
                                        boolevaltranflux = True
                            
                            if gdat.typeverb > 1:
                                print('boolevaltranflux')
                                print(boolevaltranflux)

                            if boolevaltranflux:
                                prep_dist(gdat, j)
                                boolnocccomp = retr_boolnoccobjt(gdat, j)
                                gdat.boolgridstarbrgt = gdat.boolgridstarbrgt & boolnocccomp
            
                            if gdat.perimoon is not None and a == 0:

                                for jj in indxmoon[j]:
                                    
                                    if np.sqrt(gdat.xposmoon[j][jj][t]**2 + gdat.yposmoon[j][jj][t]**2) < 1. + rratmoon[j][jj]:
                                        
                                        boolevaltranflux = True

                                        xposgridmoon = gdat.xposgridstar - gdat.xposmoon[j][jj][t]
                                        yposgridmoon = gdat.yposgridstar - gdat.yposmoon[j][jj][t]
                                        
                                        distmoon = np.sqrt(xposgridmoon**2 + yposgridmoon**2)
                                        boolnoccmoon = distmoon > rratmoon[j][jj]
                                        
                                        gdat.boolgridstarbrgt = gdat.boolgridstarbrgt & boolnoccmoon
                    
                        if boolevaltranflux:
                            gdat.fluxtotl[t] = retr_fluxstartrantotl(gdat, gdat.typecoor, gdat.boolgridstarbrgt)
                            if gdat.boolmakeanim:
                                make_framanim(gdat, t, phas[j][t])
                                
                        if gdat.booldiag:
                            if t > 0:
                                if abs(gdat.fluxtotl[t] - gdat.fluxtotl[t-1]) / gdat.fluxtotl[t] > 0.002:
                                    raise Exception('Changed by more than 0.2 percent')
                                    
                            
                if gdat.boolmakeanim:

                    for namevarbanim in gdat.listnamevarbanim:
                        if not os.path.exists(gdat.pathgiff[namevarbanim]):
                            # make the animation
                            gdat.cmndmakeanim[namevarbanim] += ' %s' % gdat.pathgiff[namevarbanim]
                            print('Writing to %s...' % gdat.pathgiff[namevarbanim])
                            os.system(gdat.cmndmakeanim[namevarbanim])
                            
                            # delete images
                            os.system(gdat.cmnddeleimag[namevarbanim])

            if boolintp:
                
                for j in indxcomp:
                
                    gdat.fluxtotlcomp[j] = np.array(gdat.fluxtotlcomp[j])
                    gdat.phascomp[j] = np.array(gdat.phascomp[j])
                    numbtimecomp[j] = gdat.fluxtotlcomp[j].size
                    indxtimecomp[j] = np.arange(numbtimecomp[j])
                
                    if gdat.booldiag:
                        if gdat.booldiagtran:
                            if (gdat.fluxtotlcomp[j] == 1.).all():
                                raise Exception('')

                for j in indxcomp:
                    
                    if not np.isfinite(gdat.duratrantotl[j]) or gdat.duratrantotl[j] == 0.:
                        continue
                    
                    if gdat.phascomp[j].size > 1:
                        
                        if gdat.typesyst == 'cosc':
                            indxphaseval = indxtime
                        else:
                            indxphaseval = np.where((phas[j] >= np.amin(gdat.phascomp[j])) & (phas[j] <= np.amax(gdat.phascomp[j])))[0]
                            
                        if indxphaseval.size > 0:
                            intptemp = scipy.interpolate.interp1d(gdat.phascomp[j], gdat.fluxtotlcomp[j], fill_value=gdat.brgtstarnocc, \
                                                                                                        bounds_error=False)(phas[j][indxphaseval])
                            
                            if gdat.typesyst == 'cosc':
                                if gdat.booldiag:
                                    if np.amin(intptemp) / gdat.brgtstarnocc > 1.1:
                                        raise Exception('')
                                gdat.fluxtotl[indxphaseval] = intptemp
                            else:
                                diff = gdat.brgtstarnocc - intptemp
                                gdat.fluxtotl[indxphaseval] -= diff

                    else:
                        gdat.fluxtotl = np.full_like(phas[j], gdat.brgtstarnocc)
        
    if gdat.booldiag:
        if not np.isfinite(gdat.fluxtotl).all() or (gdat.fluxtotl < 0).any():
            print('')
            print('')
            print('')
            print('gdat.fluxtotl')
            summgene(gdat.fluxtotl)
            raise Exception('')
    
        # normalize the light curve
        if typenorm != 'none':
            
            if gdat.booldiag:
                if gdat.typesyst == 'cosc':
                    if (gdat.fluxtotl / gdat.brgtstarnocc < 1. - 1e-6).any():
                        print('Warning! Flux decreased in a self-lensing light curve.')
                        print('gdat.fluxtotl')
                        print(gdat.fluxtotl)
                        #raise Exception('')

            if gdat.typeverb > 1:
                if gdat.typesyst == 'cosc':
                    print('gdat.fluxtotl')
                    summgene(gdat.fluxtotl)
                print('Normalizing the light curve...')
            
            if typenorm == 'medi':
                fact = np.median(gdat.fluxtotl)
            elif typenorm == 'nocc':
                fact = gdat.brgtstarnocc
            elif typenorm == 'maxm':
                fact = np.amax(gdat.fluxtotl)
            elif typenorm == 'edgeleft':
                fact = gdat.fluxtotl[0]
            rflxtranmodl = gdat.fluxtotl / fact
            
            if gdat.booldiag:
                if fact == 0.:
                    print('typenorm')
                    print(typenorm)
                    print('')
                    for j in indxcomp:
                        print('gdat.fluxtotlcomp[j]')
                        summgene(gdat.fluxtotlcomp[j])
                    print('Normalization involved division by 0.')
                    print('gdat.fluxtotl')
                    summgene(gdat.fluxtotl)
                    raise Exception('')
                if gdat.typesyst == 'cosc':
                    if (rflxtranmodl < 0.9).any():
                        raise Exception('')

            #if (rflxtranmodl > 1e2).any():
            #    raise Exception('')

        if gdat.booldiag:
            if gdat.boolsystpsys and gdat.typesyst != 'psyspcur' and np.amax(gdat.fluxtotl) > gdat.brgtstarnocc * (1. + 1e-6):
                print('')
                print('')
                print('')
                print('gdat.typesyst')
                print(gdat.typesyst)
                print('gdat.typelmdk')
                print(gdat.typelmdk)
                print('gdat.coeflmdk')
                print(gdat.coeflmdk)
                print('gdat.fluxtotl')
                summgene(gdat.fluxtotl)
                print('gdat.brgtstarnocc')
                print(gdat.brgtstarnocc)
                raise Exception('gdat.boolsystpsys and gdat.typesyst != psyspcur and np.amax(gdat.fluxtotl) > gdat.brgtstarnocc * (1. + 1e-6)')
        
            if False and np.amax(rflxtranmodl) > 1e6:
                print('gdat.fluxtotl')
                summgene(gdat.fluxtotl)
                raise Exception('')

    dictefes['rflx'] = rflxtranmodl
    
    if gdat.booldiag:
        if gdat.booldiagtran:
            if (rflxtranmodl == 1.).all():
                if boolintp:
                    for j in indxcomp:
                        print('gdat.phascomp[j]')
                        summgene(gdat.phascomp[j])
                        print('gdat.fluxtotlcomp[j]')
                        summgene(gdat.fluxtotlcomp[j])
                raise Exception('')

    if gdat.masscomp is not None and gdat.massstar is not None and gdat.radistar is not None:
        densstar = 1.4 * gdat.massstar / gdat.radistar**3
        deptbeam = 1e-3 * nicomedia.retr_deptbeam(gdat.pericomp, gdat.massstar, gdat.masscomp)
        deptelli = 1e-3 * nicomedia.retr_deptelli(gdat.pericomp, densstar, gdat.massstar, gdat.masscomp)
        dictefes['rflxslen'] = [[] for j in indxcomp]
        dictefes['rflxbeam'] = [[] for j in indxcomp]
        dictefes['rflxelli'] = [[] for j in indxcomp]

        for j in indxcomp:
            
            dictefes['rflxslen'][j] = dictefes['rflx']
            
            dictefes['rflxbeam'][j] = 1. + deptbeam * np.sin(phas[j])
            dictefes['rflxelli'][j] = 1. + deptelli * np.sin(2. * phas[j])
            
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

    if gdat.booldiag:
        if not np.isfinite(dictefes['rflx']).all() or (dictefes['rflx'] < 0).any():
            print('')
            print('')
            print('')
            print('dictefes[rflx]')
            summgene(dictefes['rflx'])
            raise Exception('')
    
    if gdat.pathvisu is not None:
        #dictefes['time'] = time
        for name in dictinpt:
            if name != 'gdat':
                dictefes[name] = getattr(gdat, name)#dictinpt[name]
        plot_tser_dictefes(gdat.pathvisu, dictefes, '%s' % strgextn, lablunittime)
        
    dictefes['timetotl'] = timemodu.time() - timeinit
    dictefes['timeredu'] = dictefes['timetotl'] / numbtime

    if gdat.booldiag and dictefes['timeredu'] > 1e-1:
        print('Took too long to execute...')
        #raise Exception('')
    
    if gdat.booldiag and not np.isfinite(dictefes['rflx']).all():
        print('')
        print('')
        print('')
        print('dictefes[rflx] is not all finite.')
        print('dictefes[rflx]')
        summgene(dictefes['rflx'])
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


def plot_tser_dictefes(pathvisu, dictefes, strgextninpt, lablunittime, typetarg='', typefileplot='png'):

    dictlabl = dict()
    dictlabl['root'] = dict()
    dictlabl['unit'] = dict()
    dictlabl['totl'] = dict()
    
    numbcomp = dictefes['pericomp'].size
    indxcomp = np.arange(numbcomp)
    listnamevarbcomp = []

    for j in indxcomp:
        listnamevarbcomp += ['pericom%d' % j, 'epocmtracom%d' % j, 'cosicom%d' % j, 'rsmacom%d' % j] 
        listnamevarbcomp += ['radicom%d' % j]
        listnamevarbcomp += ['typebrgtcom%d' % j]
        if dictefes['typesyst'] == 'psyspcur':
            listnamevarbcomp += ['offsphascom%d' % j]
    
    listnamevarbsimu = ['tolerrat']#, 'diffphas']
    listnamevarbstar = ['radistar']
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
    
    print('strgtitl')
    print(strgtitl)

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
    
    #listcolr = ['g', 'b', 'firebrick', 'orange', 'olive']
    #dictmodl['eval']['colr'] = listcolr[k]

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
    #lablxaxi = 'Time from mid-transit [hours]'
    lablxaxi = 'Time [%s]' % lablunittime
    lablyaxi = 'Relative flux - 1 [ppm]'
    
    if strgextninpt is None or strgextninpt == '':
        strgextnbase = '%s' % (dictefes['typesyst'])
        if typetarg != '':
            strgextnbase += '_%s' % typetarg
    else:
        strgextnbase = strgextninpt
    
    numbener = dictefes['rflx'].shape[1]
    indxener = np.arange(numbener)
    for e in indxener:
        
        if numbener > 0:
            strgener = '_e%03d' % e
        else:
            strgener = ''
    
        dictmodl['eval']['lcur'] = 1e6 * (dictefes['rflx'][:, e] - 1)

        # time-series
        strgextn = '%s%s' % (strgextnbase, strgener)
        pathplot = miletos.plot_tser(pathvisu, \
                                     dictmodl=dictmodl, \
                                     typefileplot=typefileplot, \
                                     #listxdatvert=listxdatvert, \
                                     strgextn=strgextn, \
                                     lablxaxi=lablxaxi, \
                                     lablyaxi=lablyaxi, \
                                     strgtitl=strgtitl, \
                                     #typesigncode='ephesos', \
                                    )
        
        for j in indxcomp:
            strgextnbasecomp = '%s%s_com%d' % (strgextnbase, strgener, j)
            
            epoc = dictefes['epocmtracomp'][j]
            peri = dictefes['pericomp'][j]
            
            # phase curve
            strgextn = '%s_pcur' % (strgextnbasecomp)
            pathplot = miletos.plot_tser(pathvisu, \
                                         dictmodl=dictmodl, \
                                         boolfold=True, \
                                         typefileplot=typefileplot, \
                                         #listxdatvert=listxdatvert, \
                                         strgextn=strgextn, \
                                         lablxaxi=lablxaxi, \
                                         lablyaxi=lablyaxi, \
                                         strgtitl=strgtitl, \
                                         epoc=epoc, \
                                         peri=peri, \
                                         #typesigncode='ephesos', \
                                        )
            
            if dictefes['typesyst'] == 'psyspcur':
                
                # vertical zoom onto the phase curve
                strgextn = '%s_pcurzoom' % (strgextnbasecomp)
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtyaxi=[-500, None], \
                                             epoc=epoc, \
                                             peri=peri, \
                                             #typesigncode='ephesos', \
                                            )
                
                # horizontal zoom around the primary
                strgextn = '%s_prim' % (strgextnbasecomp)
                #limtxaxi = np.array([-24. * 0.7 * dictefes['duratrantotl'], 24. * 0.7 * dictefes['duratrantotl']])
                limtxaxi = np.array([-2, 2.])
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             epoc=epoc, \
                                             peri=peri, \
                                             #typesigncode='ephesos', \
                                            )
                
                # horizontal zoom around the secondary
                strgextn = '%s_seco' % (strgextnbasecomp)
                limtxaxi += 0.5 * dictefes['pericomp'][j] * 24.
                pathplot = miletos.plot_tser(pathvisu, \
                                             dictmodl=dictmodl, \
                                             boolfold=True, \
                                             typefileplot=typefileplot, \
                                             #listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             limtyaxi=[-500, None], \
                                             epoc=epoc, \
                                             peri=peri, \
                                             #typesigncode='ephesos', \
                                            )


def retr_strgtitl(dictefesinpt, listnamevarbcomp, dictlabl):
    '''
    Return the title of a plot with information about the system
    '''
    
    strgtitl = ''
    if 'radistar' in dictefesinpt:
        strgtitl += '$R_*$ = %.1f $R_\odot$' % dictefesinpt['radistar']
    if dictefesinpt['typesyst'] == 'cosc' and 'massstar' in dictefesinpt:
        if len(strgtitl) > 0 and strgtitl[-2:] != ', ':
            strgtitl += ', '
        strgtitl += '$M_*$ = %.1f $M_\odot$' % dictefesinpt['massstar']
    
    print('listnamevarbcomp')
    print(listnamevarbcomp)

    for kk, name in enumerate(listnamevarbcomp):
        
        if name == 'epocmtracomp' or name == 'typebrgtcomp' or (not name[:-1] + 'p' in dictefesinpt and name in dictefesinpt):
            continue
        
        if name.startswith('epocmtracom'):
            continue

        if name.startswith('typebrgtcom'):
            continue

        if name in dictefesinpt:
            nameprim = name
        else:
            nameprim = name[:-1] + 'p'
        
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
            
        print('nameprim')
        print(nameprim)
        print('dictefesinpt[nameprim]')
        print(dictefesinpt[nameprim])

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


