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
pathpopl = pathbase + 'Populations/'

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

minmtime = 0.
maxmtime = 2.

boolcalcdistcomp = False
# number of systems to visualize via ephesos
numbsystvisu = 2
if boolsystpsys:
    if typesyst== 'PlanetarySystem' or typesyst == 'PlanetarySystemWithNonKeplerianObjects':
        if typesystCompound == 'PlanetarySystem_Single':
            minmnumbcompstar = 1
            maxmnumbcompstar = 1
            typesamporbtcomp = 'peri'
        elif typesystCompound == 'PlanetarySystemWithNonKeplerianObjects':
            minmnumbcompstar = 1
            maxmnumbcompstar = 1
            typesamporbtcomp = 'smax'
            numbsystvisu = 0
        elif typesystCompound == 'PlanetarySystem_Multiple':
            boolcalcdistcomp = True
            numbsystvisu = 0
            minmnumbcompstar = 8
            maxmnumbcompstar = 10
            typesamporbtcomp = 'peri'
        else:
            raise Exception('')
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

print('minmnumbcompstar')
print(minmnumbcompstar)
print('maxmnumbcompstar')
print(maxmnumbcompstar)
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

dictefesinpt['boolcalcdistcomp'] = boolcalcdistcomp

#dictefesinpt['boolintp'] = boolintp
#dictefesinpt['boolwritover'] = boolwritover
#dictefesinpt['strgextn'] = strgextn
#dictefesinpt['strgtitl'] = strgtitl
#dictefesinpt['typecoor'] = typecoor

## cadence of simulation
cade = 4. / 60. / 24. # days
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
if boolsystpsys:
    listnamevarb += ['rrat']

from tqdm import tqdm

if boolcalcdistcomp:
    listnamefeatmult = ['rateppcr', 'numbppcr', 'minmcompdist']
    for name in listnamefeatmult:
        dictpoplstar[strgpoplstartotl][name] = np.empty(dictpoplstar[strgpoplstartotl]['radistar'].size)

print('Running ephesos on each system...')
for k in tqdm(range(numbsyst)):
    
    if typesystCompound == 'PlanetarySystem_Multiple':
        dictefesinpt['boolmakeimaglfov'] = False
        dictefesinpt['boolplotdistcomp'] = True
        dictefesinpt['boolmakeanim'] = False
        dictefesinpt['pathvisu'] = pathvisu
    elif k < numbsystvisu:
        dictefesinpt['boolmakeimaglfov'] = True
        dictefesinpt['boolmakeanim'] = True
        dictefesinpt['pathvisu'] = pathvisu
    else:
        dictefesinpt['boolmakeimaglfov'] = False
        dictefesinpt['boolmakeanim'] = False
        dictefesinpt['pathvisu'] = None
    
    if indxcompstar[k].size == 0:
        print('')
        print('')
        print('')
        raise Exception('indxcompstar[k].size == 0')
    
    if not np.isfinite(dictpoplcomp[strgpoplcomptotl]['rratcomp']).all():
        print('')
        print('')
        print('')
        print('dictpoplcomp[strgpoplcomptotl][rratcomp]')
        summgene(dictpoplcomp[strgpoplcomptotl]['rratcomp'])
        raise Exception('not np.isfinite(dictpoplcomp[strgpoplcomptotl][rratcomp]).all()')

    for namevarb in listnamevarb:
        dictefesinpt['%scomp' % namevarb] = dictpoplcomp[strgpoplcomptotl]['%scomp' % namevarb][indxcompstar[k]]
    
    dictefesinpt['strgextn'] = '%s_%04d' % (typesyst, k)
    
    # generate light curve
    dictefesoutp = ephesos.eval_modl(time, typesyst, **dictefesinpt)

    if boolcalcdistcomp:
        for name in listnamefeatmult:
            if not np.isscalar(dictefesoutp[name]):
                print('name')
                print(name)
                print( dictefesoutp[name].shape)
                raise Exception('')
    
            dictpoplstar[strgpoplstartotl][name][k] = dictefesoutp[name]

print('Visualizing the simulated population...')
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

pathbase = pathpoplanls + 'pergamon/'
os.system('mkdir -p %s' % pathbase)
typeanls = '%s' % (typesyst)

listdictlablcolrpopl = []

if typesystCompound == 'PlanetarySystem_Multiple':
    dictpopl = dictpoplstar
    listdictlablcolrpopl.append(dict())
    listdictlablcolrpopl[-1][strgpoplstartotl] = ['All', 'black']
    listboolcompexcl.append(True)

else:
    dictpopl = dictpoplcomp
    listdictlablcolrpopl.append(dict())
    listdictlablcolrpopl[-1][strgpoplcomptotl] = ['All', 'black']
    if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
        listdictlablcolrpopl[-1]['AnomalousNKF'] = ['Anomalous NKF', 'blue']
        indx = np.where(dictpoplcomp[strgpoplcomptotl]['factnonkcomp'] < 0.3)[0]
        pergamon.retr_subp(dictpoplcomp, None, None, strgpoplcomptotl, 'AnomalousNKF', indx)
    listboolcompexcl.append(True)

pergamon.init( \
             typeanls, \
             pathbase=pathbase, \
             dictpopl=dictpopl, \
             
             listdictlablcolrpopl=listdictlablcolrpopl, \
             listboolcompexcl=listboolcompexcl, \
             listtitlcomp=liststrgtitlcomp, \
             
             lablsampgene=lablsampgene, \

             #boolsortpoplsize=False, \
            )



