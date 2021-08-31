import sys

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy

def make_rflx( \
              # type of orbital architecture
              ## 'plan'
              ## 'bhol'
              ## 'planmoon'
              ## 'bholmoon'
              typeobar='planmoon', \
             ):
    '''
    Make relative flux light curves of stars with companions
    '''
    
    # time axis
    time = np.linspace(0., 30., 10000)
    
    # dictionary of conversion factors
    dictfact = ephesus.retr_factconv()
    
    # host
    ## radius of the star [R_S]
    radistar = 1.0
    ## total mass [M_S]
    massstar = 1.0
    
    # companions
    ## number of companions
    numbcomp = 2
    indxcomp = np.arange(numbcomp)
    
    ## orbital periods
    peri = np.array([3., 5.])
    ## mid-transit epochs
    epoc = np.array([200., 600.])
    ## cosine of inclination
    cosi = np.array([0.03, 0.05])
    if typeobar == 'planmoon' or typeobar == 'plan':
        ## radii of the planets [R_E]
        radicomp = np.array([1.2, 3.4])
        ## mass of the planets [M_E]
        masscomp = ephesus.retr_massfromradi(radicomp) / dictfact['msme'] # [M_S]
    
    # approximate total mass of the system
    masstotl = massstar

    ## semi-major axes [AU] 
    smax = ephesus.retr_smaxkepl(peri, masstotl)
    ## sum of radii divided by the semi-major axis
    rsma = ephesus.retr_rsma(radicomp, radistar, smax)
    
    if typeobar.endswith('moon'):
        # exomoons to companions
        
        numbmoon = np.empty(numbcomp, dtype=int)
        smaxmoon = [[] for j in indxcomp]
        perimoon = [[] for j in indxcomp]
        
        for j in indxcomp:
            # number of moons
            arry = np.arange(1, 10)
            prob = arry**(-2.)
            prob /= np.sum(prob)
            numbmoon[j] = np.random.choice(arry, p=prob)
        
            ## radii of the moons [R_E]
            radimoon = np.array([0.2, 0.4])
            ## mass of the moons [M_E]
            massmoon = ephesus.retr_massfromradi(radimoon)
            
            # Hill radius of the companion
            radihill = ephesus.retr_radihill(smax[j], masscomp[j], massstar)
            
            # minimum semi-major axis of the moons 
            minmsmaxmoon = radicomp[j] + radimoon[j]
            # maximum semi-major axis of the moons 
            maxmsmaxmoon = 0.5 * radihill
            # semi-major axes of the moons
            smaxmoon[j] = tdpy.icdf_powr(np.random.rand(numbmoon[j]), minmsmaxmoon, maxmsmaxmoon, 2.)
            # orbital period of the moons
            perimoon[j] = ephesus.retr_perikepl(smaxmoon[j], masstotl)
    
    # total mass
    masstotl = massstar + masscomp
    
    for j in indxcomp:
        if (smaxmoon[j] > smax[j] / 1.2).any():
            print('smax[j]')
            print(smax[j])
            print('smaxmoon[j]')
            print(smaxmoon[j])
            raise Exception('')
        
        # generate light curve
        rfxl = ephesus.retr_rflxtranmodl(time, radistar, peri, epoc, rsma, cosi, radicomp=radicomp, perimoon=perimoon)
        
        print('rfxl')
        summgene(rfxl)
    

globals().get(sys.argv[1])(*sys.argv[2:])
