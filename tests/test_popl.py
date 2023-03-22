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

# generate light curve
dictefesoutp = ephesos.eval_modl(time, typesyst, **dictefesinpt)

# dictionary for the configuration
dictmodl[strgextn] = dict()
dictmodl[strgextn]['time'] = time * 24. # [hours]
if dictlistvalubatc[namebatc]['vari'][nameparavari].size > 1:
    if not isinstance(dictlistvalubatc[namebatc]['vari'][nameparavari][k], str):
        dictmodl[strgextn]['labl'] = '%s = %.3g %s' % (dictlabl['root'][nameparavari], \
                            dictlistvalubatc[namebatc]['vari'][nameparavari][k], dictlabl['unit'][nameparavari])
    else:
        dictmodl[strgextn]['labl'] = '%s' % (dictlistvalubatc[namebatc]['vari'][nameparavari][k])
dictmodl[strgextn]['lcur'] = 1e6 * (dictefesoutp['rflx'] - 1)

