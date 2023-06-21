import numpy as np
import os, sys

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import nicomedia
import ephesos
import pergamon
from tdpy import summgene
import miletos

'''
Compute the relative flux light curves of systems of bodies drawn from a population model
'''

# path of the folder for visuals
pathbase = os.environ['EPHESOS_DATA_PATH'] + '/'
pathpopl = pathbase + 'Differences/'

# fix the seed
np.random.seed(2)

# type of the population of systems
## TESS 2-min target list during the nominal mission
#typepoplsyst = 'TESS_PrimaryMission_2min'
typepoplsyst = 'General'

# compound typesyst

pathpoplanls = pathpopl

# number of systems
numbsyst = 1

minmtime = -0.07
maxmtime = 0.07

minmnumbcompstar = 1
maxmnumbcompstar = 1

# number of systems to visualize via ephesos
numbsystvisu = 0

listtypesyst = ['PlanetarySystem', 'PlanetarySystemWithRingsInclinedVertical', 'PlanetarySystemWithRingsInclinedHorizontal']

print('minmnumbcompstar')
print(minmnumbcompstar)
print('maxmnumbcompstar')
print(maxmnumbcompstar)
# get dictionaries for stars, companions, and moons
dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = nicomedia.retr_dictpoplstarcomp( \
                                                                                                                                  'PlanetarySystem', \
                                                                                                                                  
                                                                                                                                  typepoplsyst, \
                                                                                                                                  booltoyysunn=True, \
                                                                                                                                  minmnumbcompstar=minmnumbcompstar, \
                                                                                                                                  maxmnumbcompstar=maxmnumbcompstar, \
                                                                                                                                  epocmtracomp=0., \
                                                                                                                                  #minmradicomp=10., \
                                                                                                                                  minmmasscomp=10., \
                                                                                                                                  minmpericomp=0.4, \
                                                                                                                                  maxmpericomp=2., \
                                                                                                                                  maxmcosicomp=0.1, \
                                                                                                                                  numbsyst=numbsyst, \
                                                                                                                                )


strgpoplstartotl = 'star' + typepoplsyst + 'totl'
strgpoplcomptotl = 'compstar' + typepoplsyst + 'totl'
strgpoplcomptran = 'compstar' + typepoplsyst + 'tran'

if (dictpoplcomp[strgpoplcomptotl]['numbcompstar'] > maxmnumbcompstar).any():
    raise Exception('')

# number of systems
numbsyst = dictpoplstar[strgpoplstartotl]['radistar'].size

# indices of the systems
indxsyst = np.arange(numbsyst)

# visualize individual systems
pathvisu = pathpoplanls + 'ephesos/'
os.system('mkdir -p %s' % pathvisu)

dictefesinpt = dict()
dictefesinpt['typelmdk'] = 'quad'
#dictefesinpt['typesyst'] = typesyst
#dictefesinpt['typenorm'] = 'edgeleft'
dictefesinpt['lablunittime'] = 'days'
dictefesinpt['booltqdm'] = False
dictefesinpt['typeverb'] = 0
#dictefesinpt['typelang'] = typelang
#dictefesinpt['typefileplot'] = typefileplot

#dictefesinpt['booldiag'] = False

#dictefesinpt['boolintp'] = boolintp
#dictefesinpt['boolwritover'] = boolwritover
#dictefesinpt['strgextn'] = strgextn
#dictefesinpt['strgtitl'] = strgtitl
#dictefesinpt['typecoor'] = typecoor

## cadence of simulation
cade = 0.1 / 60. / 24. # days
### duration of simulation
#if typesyst == 'PlanetarySystemWithPhaseCurve':
#    durasimu = dicttemp['pericomp']
#    if namebatc == 'longbase':
#        durasimu *= 3.
#else:
#    durasimu = 6. / 24. # days

#duratrantotl = nicomedia.retr_duratrantotl(dicttemp['pericomp'], dicttemp['rsmacomp'], dicttemp['cosicomp']) / 24. # [days]
#if typesyst == 'PlanetarySystemWithPhaseCurve':
#    # minimum time
#    minmtime = -0.25 * durasimu
#    # maximum time
#    maxmtime = 0.75 * durasimu
#else:
#    # minimum time
#    minmtime = -0.5 * 1.5 * duratrantotl
#    # maximum time
#    maxmtime = 0.5 * 1.5 * duratrantotl
# time axis
time = np.arange(minmtime, maxmtime, cade)

listnamevarb = ['peri', 'epocmtra', 'rsma', 'cosi']
listnamevarb += ['rrat']

from tqdm import tqdm

dictmodl = dict()

for k in range(len(listtypesyst)):
    
    #dictefesinpt['boolmakeimaglfov'] = True
    #dictefesinpt['boolmakeanim'] = True
    #pathvisu = pathpopl + '%s/' % listtypesyst[k]
    #dictefesinpt['pathvisu'] = pathvisu
    #os.system('mkdir -p %s' % pathvisu)

    for namevarb in listnamevarb:
        dictefesinpt['%scomp' % namevarb] = dictpoplcomp[strgpoplcomptotl]['%scomp' % namevarb][indxcompstar[0]]
    
    dictefesinpt['strgextn'] = '%s_%04d' % (listtypesyst[k], 0)
    
    # generate light curve
    dictefesoutp = ephesos.eval_modl(time, listtypesyst[k], **dictefesinpt)
    
    dictmodl[listtypesyst[k]] = dict()
    dictmodl[listtypesyst[k]]['time'] = time
    dictmodl[listtypesyst[k]]['tser'] = 1e6 * (dictefesoutp['rflx'][:, 0] - 1)
    #dictmodl[listtypesyst[k]]['labl'] =         

strgextn = 'RingComparison'
pathplot = miletos.plot_tser(pathvisu, \
                             dictmodl=dictmodl, \
                             #listxdatvert=listxdatvert, \
                             strgextn=strgextn, \
                             #typesigncode='ephesos', \
                            )


