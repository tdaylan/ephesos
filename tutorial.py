import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy

def make_rflx( \
              # type of orbital architecture
              ## 'psys'
              ## 'psystran'
              ## 'cosc'
              typeobar='cosc', \
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
    
    dictpoplstarcomp, _, _ = retr_dictpoplstarcomp('tessnomi2min', 'psys')
    
    numbtarg = dictpoplstarcomp['radistar'].size
    indxtarg = np.arange(numbtar)

    pathmoon = os.environ['EPHESUS_DATA_PATH'] + '/imag/moon/'
    os.system('mkdir -p %s' % pathmoon)
    
    strgextn = '%s' % (typeobar)
        
    for k in indxtarg:
        
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
    

def test_pbox_psys():
    
    
    
    pass



globals().get(sys.argv[1])(*sys.argv[2:])
