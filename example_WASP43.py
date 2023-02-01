import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy
import pergamon

'''
Make and visualize stars with companions and their relative flux light curves
'''

# fix the seed
np.random.seed(0)

# path of the folder for visuals
pathbase = os.environ['EPHESUS_DATA_PATH'] + '/'
pathimag = pathbase + 'imag/lightCurve/'
os.system('mkdir -p %s' % pathimag)

dictfact = tdpy.retr_factconv()


# determine the times at which the light curve will be sampled
## cadence
cade = 30. / 24. / 60. / 60. # [days]
## time array
time = np.arange(-0.25, 0.75, cade) # [days]

typetarg = 'WASP-43'

# generate light curves from a grid in parameters
dictvaludefa = dict()
dictvaludefa['radistar'] = 0.67 # [R_S]
dictvaludefa['coeflmdklinr'] = 0.1
dictvaludefa['coeflmdkquad'] = 0.05
dictvaludefa['epocmtracomp'] = 0.
dictvaludefa['pericomp'] = 0.813475
dictvaludefa['offsphascomp'] = 30.
dictvaludefa['cosicomp'] = 0.134
dictvaludefa['rsmacomp'] = 0.21
dictvaludefa['resoplan'] = 0.01
dictvaludefa['radicomp'] = 11. # [R_E]
dictvaludefa['tolerrat'] = 0.005
dictvaludefa['typebrgtcomp'] = 'sinusoidal'
dictvaludefa['offsphas'] = 0.

numbcomp = 1
        
dicttemptemp['rratcomp'] = dicttemptemp['radicomp'] / dicttemptemp['radistar'] / dictfact['rsre']
del dicttemptemp['radicomp']
del dicttemptemp['radistar']

strgextn = 'WASP-43'

dicttemptemp['typelmdk'] = 'quad'
dicttemptemp['pathfoldanim'] = pathfoldanim
dicttemptemp['typeverb'] = 1
dicttemptemp['strgextn'] = strgextn
dicttemptemp['titlvisu'] = strgtitl

# generate light curve
dictefes = ephesus.retr_rflxtranmodl(time, typesyst='psyspcur', **dicttemptemp)

# plot the light curve
ephesus.plot_modllcur_phas(pathimag, dictefes)


