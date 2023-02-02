import ephesos

# type of the population of systems
## TESS 2-min target list during the nominal mission
#typepoplsyst = 'tessprms2min'
typepoplsyst = 'gene'
typesyst = 'psys'

# path of the folder for visuals
pathbase = os.environ['EPHESUS_DATA_PATH'] + '/'
pathvisupopl = pathbase + 'imag/Population/'
pathdatapopl = pathbase + 'data/Population/'

# number of systems
numbsyst = 30

# get dictionaries for stars, companions, and moons
dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = retr_dictpoplstarcomp( \
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
              #namesamp=namesamp, \

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




