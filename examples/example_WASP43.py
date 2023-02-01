import numpy as np

import ephesus

'''
Make and visualize stars with companions and their relative flux light curves
'''

# path of the folder for visuals
pathbase = '/path/to/folder/'

import sys, os
pathbase = os.environ['EPHESUS_DATA_PATH'] + '/'
pathimag = pathbase + 'imag/lightCurve/'
os.system('mkdir -p %s' % pathimag)

# determine the times at which the light curve will be sampled
## cadence
cade = 30. / 24. / 60. / 60. # [days]
## time array
time = np.arange(-0.25, 0.75, cade) # [days]

typetarg = 'WASP-43'

dictefesinpt = dict()
dictefesinpt['rratcomp'] = 0.1615 # (Patel & Espinoza 2022)
dictefesinpt['coeflmdklinr'] = 0.1
dictefesinpt['coeflmdkquad'] = 0.05
dictefesinpt['epocmtracomp'] = 0.
dictefesinpt['pericomp'] = 0.813475
dictefesinpt['offsphascomp'] = 30.
dictefesinpt['cosicomp'] = 0.134
dictefesinpt['rsmacomp'] = 0.21
dictefesinpt['resoplan'] = 0.01
dictefesinpt['tolerrat'] = 0.005
dictefesinpt['typebrgtcomp'] = 'sinusoidal'
dictefesinpt['offsphas'] = 0.
dictefesinpt['typelmdk'] = 'quad'
dictefesinpt['pathfoldanim'] = pathimag
dictefesinpt['typeverb'] = 1
dictefesinpt['strgtitl'] = strgtitl

# generate light curve
dictefes = ephesus.retr_rflxtranmodl(time, typesyst='psyspcur', **dictefesinpt)

# plot the light curve
ephesus.plot_modllcur_phas(pathimag, dictefes)
