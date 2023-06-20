import numpy as np
import os, sys

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import nicomedia
import ephesos
import pergamon
from tdpy import summgene

'''
Compute the relative flux light curves of systems of bodies drawn from a population model
'''

# path of the folder for visuals
pathbase = os.environ['EPHESOS_DATA_PATH'] + '/'
pathpopl = pathbase + 'Population/'

# fix the seed
np.random.seed(2)

# type of the population of systems
## TESS 2-min target list during the nominal mission
#typepoplsyst = 'TESS_PrimaryMission_2min'
typepoplsyst = 'General'

# compound typesyst
typesystCompound = sys.argv[1]
typesyst = typesystCompound.split('_')[0]

pathpoplanls = pathpopl + typesystCompound + '/'
print('typesyst')
print(typesyst)

# number of systems
numbsyst = 1000

boolsystpsys = typesyst.startswith('PlanetarySystem')
boolsystpsysring = typesyst.startswith('PlanetarySystemWithRings')

if boolsystpsys:
    if typesyst== 'PlanetarySystem' or typesyst == 'PlanetarySystemWithNonKeplerianObjects':
        if typesystCompound == 'PlanetarySystem_Single':
            minmnumbcompstar = 1
            maxmnumbcompstar = 1
            typesamporbtcomp = 'peri'
        elif typesystCompound == 'PlanetarySystemWithNonKeplerianObjects':
            minmnumbcompstar = 2
            maxmnumbcompstar = 2
            typesamporbtcomp = 'smax'
        else:
            minmnumbcompstar = 8
            maxmnumbcompstar = 10
            typesamporbtcomp = 'peri'
    elif boolsystpsysring:
        minmnumbcompstar = 1
        maxmnumbcompstar = 1
        typesamporbtcomp = 'peri'
    else:
        print('typesyst')
        print(typesyst)
        raise Exception('')
else:
    print('typesyst')
    print(typesyst)
    raise Exception('')

# get dictionaries for stars, companions, and moons
dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = nicomedia.retr_dictpoplstarcomp( \
                                                                                                                                  typesyst, \
                                                                                                                                  
                                                                                                                                  typepoplsyst, \
                                                                                                                                  booltoyysunn=True, \
                                                                                                                                  typesamporbtcomp=typesamporbtcomp, \
                                                                                                                                  minmnumbcompstar=minmnumbcompstar, \
                                                                                                                                  maxmnumbcompstar=maxmnumbcompstar, \
                                                                                                                                  #minmradicomp=10., \
                                                                                                                                  minmmasscomp=10., \
                                                                                                                                  minmpericomp=0.5, \
                                                                                                                                  maxmpericomp=2., \
                                                                                                                                  maxmcosicomp=0.1, \
                                                                                                                                  numbsyst=numbsyst, \
                                                                                                                                )

print('Visualizing the simulated population...')

strgpoplstartotl = 'star' + typepoplsyst + 'totl'
strgpoplcomptotl = 'compstar' + typepoplsyst + 'totl'
strgpoplcomptran = 'compstar' + typepoplsyst + 'tran'
  
liststrgtitlcomp = []
listboolcompexcl = []
if typesyst == 'CompactObjectStellarCompanion':
    liststrgtitlcomp.append('Compact Objects with a stellar companion')
    lablsampgene = 'COSC'
elif boolsystpsys:
    liststrgtitlcomp.append('Planets')
    lablsampgene = 'planet'
elif typesyst == 'psysmoon':
    liststrgtitlcomp.append('Exomoons')
    lablsampgene = 'exomoon'
else:
    print('')
    print('')
    print('')
    print('typesyst')
    print(typesyst)
    raise Exception('typesyst undefined.')

listdictlablcolrpopl = []
listdictlablcolrpopl.append(dict())
listdictlablcolrpopl[-1][strgpoplcomptotl] = ['All', 'black']
if typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemWithPhaseCurve' or typesyst == 'CompactObjectStellarCompanion':
    listdictlablcolrpopl[-1][strgpoplcomptran] = ['Transiting', 'blue']
listboolcompexcl.append(False)

typeanls = '%s' % (typesyst)
pergamon.init( \
              typeanls, \
              pathbase=pathpoplanls, \
              dictpopl=dictpoplcomp, \
              
              listdictlablcolrpopl=listdictlablcolrpopl, \
              listboolcompexcl=listboolcompexcl, \
              listtitlcomp=liststrgtitlcomp, \
              
              lablsampgene=lablsampgene, \

              #boolsortpoplsize=False, \
             )

# dictionary of features for stars
dictpoplstar = dictpoplstar[strgpoplstartotl]

# dictionary of features for companions
dictpoplcomp = dictpoplcomp[strgpoplcomptotl]

# number of systems
numbsyst = dictpoplstar['radistar'].size

# indices of the systems
indxsyst = np.arange(numbsyst)

dictefesinpt = dict()
dictefesinpt['typelmdk'] = 'quad'
#dictefesinpt['typesyst'] = typesyst
#dictefesinpt['typenorm'] = 'edgeleft'
dictefesinpt['lablunittime'] = 'days'
dictefesinpt['booltqdm'] = True
#dictefesinpt['typelang'] = typelang
#dictefesinpt['typefileplot'] = typefileplot

#dictefesinpt['booldiag'] = False

dictefesinpt['boolmakeanim'] = True
dictefesinpt['pathvisu'] = pathpoplanls
#dictefesinpt['typeverb'] = 2

#dictefesinpt['boolintp'] = boolintp
#dictefesinpt['boolwritover'] = boolwritover
#dictefesinpt['strgextn'] = strgextn
#dictefesinpt['strgtitl'] = strgtitl
#dictefesinpt['typecoor'] = typecoor

## cadence of simulation
cade = 30. / 24. / 60. / 60. # days
### duration of simulation
#if typesyst == 'PlanetarySystemWithPhaseCurve':
#    durasimu = dicttemp['pericomp']
#    if namebatc == 'longbase':
#        durasimu *= 3.
#else:
#    durasimu = 6. / 24. # days

minmtime = 0.
maxmtime = 1.

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
if boolsystpsys:
    listnamevarb += ['rrat']

for k in range(numbsyst):
    
    if indxcompstar[k].size == 0:
        print('')
        print('')
        print('')
        raise Exception('indxcompstar[k].size == 0')
    
    if not np.isfinite(dictpoplcomp['rratcomp']).all():
        print('')
        print('')
        print('')
        print('dictpoplcomp[rratcomp]')
        summgene(dictpoplcomp['rratcomp'])
        raise Exception('not np.isfinite(dictpoplcomp[rratcomp]).all()')

    for namevarb in listnamevarb:
        dictefesinpt['%scomp' % namevarb] = dictpoplcomp['%scomp' % namevarb][indxcompstar[k]]
    
    dictefesinpt['strgextn'] = '%s_%04d' % (typesyst, k)
    
    # generate light curve
    dictefesoutp = ephesos.eval_modl(time, typesyst, **dictefesinpt)

