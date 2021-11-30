import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy

def make_rflxmoon( \
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
    
    np.random.seed(0)

    # time axis
    time = np.linspace(0., 10., 10000)
    minmtime = np.amin(time)
    maxmtime = np.amax(time)

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
    pericomp = np.array([3., 5.])
    ## mid-transit epochs
    epoccomp = np.array([200., 600.])
    ## cosine of inclination
    cosicomp = np.array([0.03, 0.05])
    ## inclination
    inclcomp = 180. / np.pi * np.arccos(cosicomp)
    
    pathmoon = os.environ['EPHESUS_DATA_PATH'] + '/imag/moon/'
    os.system('mkdir -p %s' % pathmoon)
    if typeobar == 'planmoon' or typeobar == 'plan':
        ## planet properties
        ## radii [R_E]
        radicomp = np.array([8., 16.])
        ## masses [M_E]
        masscomp = ephesus.retr_massfromradi(radicomp)
        ## densities [d_E]
        denscomp = masscomp / radicomp**3

    # approximate total mass of the system
    masstotl = massstar

    ## semi-major axes [AU] 
    smaxcomp = ephesus.retr_smaxkepl(pericomp, masstotl)
    ## sum of radii divided by the semi-major axis
    #rsma = ephesus.retr_rsma(radicomp, radistar, smax)
    
    if typeobar.endswith('moon'):
        # exomoons to companions
        
        numbmoon = np.empty(numbcomp, dtype=int)
        radimoon = [[] for j in indxcomp]
        massmoon = [[] for j in indxcomp]
        densmoon = [[] for j in indxcomp]
        perimoon = [[] for j in indxcomp]
        epocmoon = [[] for j in indxcomp]
        smaxmoon = [[] for j in indxcomp]
        indxmoon = [[] for j in indxcomp]

        # Hill radius of the companion
        radihill = ephesus.retr_radihill(smaxcomp, masscomp / dictfact['msme'], massstar)
        # maximum semi-major axis of the moons 
        maxmsmaxmoon = 0.5 * radihill
        
        for j in indxcomp:
            # number of moons
            arry = np.arange(1, 10)
            prob = arry**(-2.)
            prob /= np.sum(prob)
            numbmoon[j] = np.random.choice(arry, p=prob)
            indxmoon[j] = np.arange(numbmoon[j])
            smaxmoon[j] = np.empty(numbmoon[j])
            # properties of the moons
            ## radii [R_E]
            radimoon[j] = radicomp[j] * tdpy.icdf_powr(np.random.rand(numbmoon[j]), 0.05, 0.3, 2.)
            ## mass [M_E]
            massmoon[j] = ephesus.retr_massfromradi(radimoon[j])
            ## densities [d_E]
            densmoon[j] = massmoon[j] / radimoon[j]**3
            # minimum semi-major axes
            minmsmaxmoon = ephesus.retr_radiroch(radicomp[j], denscomp[j], densmoon[j])
            
            # semi-major axes of the moons
            for jj in indxmoon[j]:
                smaxmoon[j][jj] = tdpy.icdf_powr(np.random.rand(), minmsmaxmoon[jj], maxmsmaxmoon[j], 2.)
            # orbital period of the moons
            perimoon[j] = ephesus.retr_perikepl(smaxmoon[j], np.sum(masscomp) / dictfact['msme'])
            # mid-transit times of the moons
            epocmoon[j] = tdpy.icdf_self(np.random.rand(numbmoon[j]), minmtime, maxmtime)
        
        #print('smaxcomp')
        #print(smaxcomp)
        #print('radicomp')
        #print(radicomp)
        #print('pericomp')
        #print(pericomp)
        #print('radihill')
        #print(radihill)
        #print('minmsmaxmoon')
        #print(minmsmaxmoon)
        #print('maxmsmaxmoon')
        #print(maxmsmaxmoon)
        #print('masstotl')
        #print(masstotl)
        #print('smaxmoon')
        #print(smaxmoon)
        #print('perimoon')
        #print(perimoon)
        #print('')
        #raise Exception('')

    if (smaxmoon[j] > smaxcomp[j] / 1.2).any():
        print('smaxcomp[j]')
        print(smaxcomp[j])
        print('radihill[j]')
        print(radihill[j])
        print('smaxmoon[j]')
        print(smaxmoon[j])
        raise Exception('')
    
    strgextn = '%s' % (typeobar)
    
    # generate light curve
    dictoutp = ephesus.retr_rflxtranmodl(time, radistar, pericomp, epoccomp, inclcomp=inclcomp, massstar=massstar, \
                        radicomp=radicomp, masscomp=masscomp, \
                        perimoon=perimoon, epocmoon=epocmoon, radimoon=radimoon, \
                        boolcompmoon=True, \
                        pathanim=pathmoon, \
                        strgextn=strgextn, \
                        )
    
    dictmodl = dict()
    dictmodl[strgextn] = dict()
    
    dictmodl[strgextn]['time'] = time
    dictmodl[strgextn]['lcur'] = dictoutp['rflx']
    boolwritover = True
    pathplot = ephesus.plot_lcur(pathmoon, dictmodl=dictmodl, strgextn=strgextn, \
                                    boolwritover=boolwritover, \
                                    #titl=titlraww, \
                                )
    
    dictmodl[strgextn]['lcur'] = dictoutp['rflxcomp']
    pathplot = ephesus.plot_lcur(pathmoon, dictmodl=dictmodl, strgextn=strgextn+'_comp', \
                                    boolwritover=boolwritover, \
                                    #titl=titlraww, \
                                )
    
    dictmodl[strgextn]['lcur'] = dictoutp['rflxmoon']
    pathplot = ephesus.plot_lcur(pathmoon, dictmodl=dictmodl, strgextn=strgextn+'_moon', \
                                    boolwritover=boolwritover, \
                                    #titl=titlraww, \
                                )
    



globals().get(sys.argv[1])(*sys.argv[2:])
