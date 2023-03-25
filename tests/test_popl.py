import numpy as np
import os

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import nicomedia
import ephesos
import pergamon

'''
Compute the relative flux light curves of systems of bodies drawn from a population model
'''

# type of the population of systems
## TESS 2-min target list during the nominal mission
#typepoplsyst = 'tessprms2min'
typepoplsyst = 'gene'
typesyst = 'psys'

# path of the folder for visuals
pathbase = os.environ['EPHESOS_DATA_PATH'] + '/'
pathpopl = pathbase + 'Population/'
pathvisupopl = pathbase + 'visuals/'
pathdatapopl = pathbase + 'data/'

# number of systems
numbsyst = 30

# get dictionaries for stars, companions, and moons
dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = nicomedia.retr_dictpoplstarcomp( \
                                                                                                                                  typesyst, \
                                                                                                                                  
                                                                                                                                  typepoplsyst, \
                                                                                                                                  epocmtracomp=0., \
                                                                                                                                  booltoyysunn=True, \
                                                                                                                                  typesamporbtcomp='peri', \
                                                                                                                                  #minmradicomp=10., \
                                                                                                                                  minmmasscomp=300., \
                                                                                                                                  minmpericomp=1., \
                                                                                                                                  maxmpericomp=2., \
                                                                                                                                  numbsyst=numbsyst, \
                                                                                                                                 )
print('Visualizing the simulated population...')

strgpoplstartotl = 'star' + typepoplsyst + 'totl'
strgpoplcomptotl = 'compstar' + typepoplsyst + 'totl'
strgpoplcomptran = 'compstar' + typepoplsyst + 'tran'
  
#del dictpoplcomp[strgpoplcomptotl]['idenstar']
#del dictpoplcomp[strgpoplcomptran]['idenstar']
liststrgtitlcomp = []
listboolcompexcl = []
if typesyst == 'cosc':
    liststrgtitlcomp.append('Compact Objects with a stellar companion')
    lablsampgene = 'COSC'
if typesyst == 'psys' or typesyst == 'psyspcur':
    liststrgtitlcomp.append('Planets')
    lablsampgene = 'planet'
if typesyst == 'psysmoon':
    liststrgtitlcomp.append('Exomoons')
    lablsampgene = 'exomoon'

listdictlablcolrpopl = []
listdictlablcolrpopl.append(dict())
listdictlablcolrpopl[-1][strgpoplcomptotl] = ['All', 'black']
if typesyst == 'psys' or typesyst == 'psyspcur' or typesyst == 'cosc':
    listdictlablcolrpopl[-1][strgpoplcomptran] = ['Transiting', 'blue']
listboolcompexcl.append(False)

typeanls = '%s' % (typesyst)
pergamon.init( \
              typeanls, \
              dictpopl=dictpoplcomp, \
              
              listdictlablcolrpopl=listdictlablcolrpopl, \
              listboolcompexcl=listboolcompexcl, \
              listtitlcomp=liststrgtitlcomp, \
              
              lablsampgene=lablsampgene, \

              pathvisu=pathvisupopl, \
              pathdata=pathdatapopl, \
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
dictefesinpt['pathvisu'] = pathvisupopl
#dictefesinpt['typelang'] = typelang
#dictefesinpt['typefileplot'] = typefileplot
#dictefesinpt['booldiag'] = booldiag
dictefesinpt['typeverb'] = 1
#dictefesinpt['boolintp'] = boolintp
#dictefesinpt['boolwritover'] = boolwritover
#dictefesinpt['strgextn'] = strgextn
#dictefesinpt['strgtitl'] = strgtitl
#dictefesinpt['typecoor'] = typecoor

print('dictefesinpt')
print(dictefesinpt)


## cadence of simulation
cade = 30. / 24. / 60. / 60. # days
### duration of simulation
#if typesyst == 'psyspcur':
#    durasimu = dicttemp['pericomp']
#    if namebatc == 'longbase':
#        durasimu *= 3.
#else:
#    durasimu = 6. / 24. # days

minmtime = 0.
maxmtime = 10.
#duratrantotl = nicomedia.retr_duratrantotl(dicttemp['pericomp'], dicttemp['rsmacomp'], dicttemp['cosicomp']) / 24. # [days]
#if typesyst == 'psyspcur':
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
for k in range(numbsyst):
    
    for namevarb in listnamevarb:
        dictefesinpt['%scomp' % namevarb] = dictpoplcomp['%scomp' % namevarb][indxcompstar[k]]
    
    # generate light curve
    dictefesoutp = ephesos.eval_modl(time, typesyst, **dictefesinpt)
