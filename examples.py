import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy
import pergamon

def retr_strgtitl(dictstrgtitl):
    
    strgtitl = '$R_* = %.1f R _\odot' % dictstrgtitl['radistar']
    
    listnamecomp = []
    for name in dictstrgtitl.keys():
        if name.endswith('comp') and dictstrgtitl[name] is not None:
            listnamecomp.append(name)
            numbcomp = len(dictstrgtitl[name])

    for name in dictstrgtitl.keys():
        
        if name in listnamecomp:
            for j, valu in enumerate(dictstrgtitl[name]):
                if name == 'radicomp':
                    strgtitl += ', R_{c%d} = ' % j
                if name == 'pericomp':
                    strgtitl += ', P_{c%d} = ' % j
            
                strgtitl += '%.3g' % (valu)
                if j != numbcomp - 1:
                    strgtitl += ', '
            
            if name == 'radicomp':
                strgtitl += ' R_{\oplus}'
            if name == 'pericomp':
                strgtitl += '$ days$'
    
    if strgtitl[-1] == '$':
        strgtitl = strgtitl[:-1]
    else:
        strgtitl += '$'

    return strgtitl


def make_rflx( \
             ):
    '''
    Make relative flux light curves of stars with companions
    '''
    
    # fix the seed
    np.random.seed(0)

    # time axis
    time = np.linspace(0.4, 0.6, 1000)
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
     
    # type of systems to be illustrated
    listtypesyst = [ \
                    'cosc', \
                    'psys', \
                    'psyspcur', \
                    'psysmoon', \
                    ]
    
    # Boolean flag to overwrite visuals
    boolwritover = True

    # path of the folder for visuals
    pathbase = os.environ['EPHESUS_DATA_PATH'] + '/'
    pathimag = pathbase + 'imag/lightCurve/'
    pathdata = pathbase + 'data/LightCurve/'
    os.system('mkdir -p %s' % pathimag)
    
    # number of systems
    numbsyst = 10

    # type of the population of systems
    ## TESS 2-min target list during the nominal mission
    #typepoplsyst = 'tessprms2min'
    typepoplsyst = 'gene'
    
    for typesyst in listtypesyst:
        
        print('typesyst')
        print(typesyst)

        pathimagpopl = pathimag + typesyst + '/'
        pathdatapopl = pathdata + typesyst + '/'
        
        # get dictionaries for stars, companions, and moons
        dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = ephesus.retr_dictpoplstarcomp( \
                                                                                                                                             typesyst, \
                                                                                                                                             
                                                                                                                                             typepoplsyst, \
                                                                                                                                             epocmtracomp=0.5, \
                                                                                                                                             booltoyysunn=True, \
                                                                                                                                             typesamporbtcomp='peri', \
                                                                                                                                             minmpericomp=1., \
                                                                                                                                             maxmpericomp=2., \
                                                                                                                                             numbsyst=numbsyst, \
                                                                                                                                            )
          
        print('Visualizing the simulated population...')

        listdictlablcolrpopl = []
        #for k in range(len(listnametypetrue)):
        #    listdictlablcolrpopl[0][listnametypetrue[k]] = [listlabltypetrue[k], listcolrtypetrue[k]]
        
        strgpoplstartotl = 'star' + typepoplsyst + 'totl'
        strgpoplcomptotl = 'compstar' + typepoplsyst + 'totl'
        strgpoplcomptran = 'compstar' + typepoplsyst + 'tran'
          
        del dictpoplcomp[strgpoplcomptotl]['idenstar']
        del dictpoplcomp[strgpoplcomptran]['idenstar']
        
        if typesyst == 'psyspcur':
            typeplanbrgt = 'term'
        else:
            typeplanbrgt = 'dark'
        
        listdictlablcolrpopl.append(dict())
        liststrgtitlcomp = []
        listboolcompexcl = []
        if typesyst == 'cosc':
            liststrgtitlcomp.append('Compact Objects with a stellar companion')
            namesamp = 'COSC'
        if typesyst == 'psys' or typesyst == 'psyspcur':
            liststrgtitlcomp.append('Planets')
            namesamp = 'planet'
        if typesyst == 'psysmoon':
            liststrgtitlcomp.append('Exomoons')
            namesamp = 'exomoon'
        
        listdictlablcolrpopl[-1][strgpoplcomptotl] = ['All', 'black']
        if typesyst == 'psys' or typesyst == 'psyspcur' or typesyst == 'cosc':
            listdictlablcolrpopl[-1][strgpoplcomptran] = ['Transiting', 'blue']
        listboolcompexcl.append(False)

        print('dictpoplstar[strgpoplstartotl][distsyst]')
        summgene(dictpoplstar[strgpoplstartotl]['distsyst'])
        print('dictpoplcomp[strgpoplcomptotl][distsyst]')
        summgene(dictpoplcomp[strgpoplcomptotl]['distsyst'])
        print('dictpoplcomp[strgpoplcomptran][distsyst]')
        summgene(dictpoplcomp[strgpoplcomptran]['distsyst'])
        
        typeanls = '%s' % (typesyst)
        pergamon.init( \
                      typeanls + 'tuto', \
                      dictpopl=dictpoplcomp, \
                      
                      listdictlablcolrpopl=listdictlablcolrpopl, \
                      listboolcompexcl=listboolcompexcl, \
                      listtitlcomp=liststrgtitlcomp, \
                      namesamp=namesamp, \

                      pathimag=pathimagpopl, \
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

        # load the optional moon features
        if listtypesyst == 'psysmoontran':
            perimoon = dictpoplmoon['perimoon']
            epocmtramoon = dictpoplmoon['epocmtramoon']
            radimoon = dictpoplmoon['radimoon']
        else:
            perimoon = None
            epocmtramoon = None
            radimoon = None

        # number of iterations for the type of the system
        if typesyst == 'psys':
            numbiterintp = 3
        else:
            numbiterintp = 1

        for k in indxsyst:
            
            # dictionary for the plotting function holding various light curves
            dictmodl = dict()

            if not np.any(dictpoplcomp['booltran'][indxcompstar[k]]):
                continue
            
            # radii of the companions
            if typesyst == 'cosc':
                radicomp = None
            else:
                radicomp = dictpoplcomp['radicomp'][indxcompstar[k]]
            
            # masses of the companions
            if typesyst == 'psys' or typesyst == 'psysmoon':
                masscomp = None
            else:
                masscomp = dictpoplcomp['masscomp'][indxcompstar[k]]
            
            # title for the plots
            dictstrgtitl = dict()
            dictstrgtitl['radistar'] = dictpoplstar['radistar'][k]
            dictstrgtitl['pericomp'] = dictpoplcomp['pericomp'][indxcompstar[k]]
            dictstrgtitl['radicomp'] = radicomp
            strgtitl = retr_strgtitl(dictstrgtitl)
                
            # string for the configuration
            strgthissyst = '%s_%04d' % (typesyst, k)
            
            for a in range(numbiterintp):
                
                if a > 0:
                    continue

                if a == 0:
                    strgextn = strgthissyst + '_intpplan'
                if a == 1:
                    strgextn = strgthissyst + '_intpstar'
                if a == 2:
                    strgextn = strgthissyst
                
                # dictionary for the configuration
                dictmodl[strgextn] = dict()
                
                if a == 0:
                    boolintp = True
                    strgextn = strgthissyst + '_intpplan'
                    typecoor = 'plan'
                    dictmodl[strgextn]['lsty'] = '-.'
                    dictmodl[strgextn]['labl'] = 'Interpolation, Planet grid'
                if a == 1:
                    boolintp = True
                    strgextn = strgthissyst + '_intpstar'
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '--'
                    dictmodl[strgextn]['labl'] = 'Interpolation, Star grid'
                if a == 2:
                    boolintp = False
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '-'
                    dictmodl[strgextn]['labl'] = 'Full Evaluation'
                
                # generate light curve
                dictoutp = ephesus.retr_rflxtranmodl(time, \
                                                     radistar=dictpoplstar['radistar'][k], \
                                                     massstar=dictpoplstar['massstar'][k], \
                                                     
                                                     rsmacomp=dictpoplcomp['rsmacomp'][indxcompstar[k]], \
                                                     pericomp=dictpoplcomp['pericomp'][indxcompstar[k]], \
                                                     epocmtracomp=dictpoplcomp['epocmtracomp'][indxcompstar[k]], \
                                                     inclcomp=dictpoplcomp['inclcomp'][indxcompstar[k]], \
                                                     radicomp=radicomp, \
                                                     masscomp=masscomp, \
                                                     
                                                     perimoon=perimoon, \
                                                     epocmtramoon=epocmtramoon, \
                                                     radimoon=radimoon, \
                                                     
                                                     typesyst=typesyst, \

                                                     typeplanbrgt=typeplanbrgt, \

                                                     typecoor=typecoor, \

                                                     boolintp=boolintp, \
                                                     
                                                     pathfoldanim=pathimag, \
                                                     boolwritover=boolwritover, \
                                                     strgextn=strgextn, \
                                                     titlvisu=strgtitl, \
                                                    )
                
                print('Making a light curve plot...')
                
                dictmodl[strgextn]['time'] = time
                dictmodl[strgextn]['lcur'] = dictoutp['rflx']
            
            lablyaxi = 'Relative flux - 1 [ppm]'
            pathplot = ephesus.plot_lcur(pathimag, \
                                         dictmodl=dictmodl, \
                                         
                                         boolwritover=boolwritover, \
                                         strgextn=strgthissyst, \
                                         lablyaxi=lablyaxi, \
                                         strgtitl=strgtitl, \
                                        )
            
            if typesyst == 'psysmoontran':
                strgextn = strgthissyst + '_comp'
                dictmodl[strgextn]['lcur'] = dictoutp['rflxcomp']
                
                strgextn = strgthissyst + '_moon'
                dictmodl[strgextn]['lcur'] = dictoutp['rflxmoon']
                pathplot = ephesus.plot_lcur(pathimag, \
                                             dictmodl=dictmodl, \
                                             strgextn='cmpt', \
                                             boolwritover=boolwritover, \
                                             strgtitl=strgtitl, \
                                            )
    
    raise Exception('')

    dictlistvalu = dict()
    
    typesyst = 'psys'
    radistar = 1.
    massstar = None
    epocmtracomp = np.array([0.5])
    dictlistvalu['pericomp'] = np.array([1., 3., 5.])
    dictlistvalu['rratcomp'] = np.array([0.003, 0.01, 0.05, 0.1, 0.3])
    dictlistvalu['radicomp'] = dictlistvalu['rratcomp'] / radistar
    dictlistvalu['cosicomp'] = np.array([0., 0.05, 0.1])
    dictlistvalu['rsmacomp'] = np.array([0.001, 0.1, 0.2])
    dictlistvalu['tolerrat'] = np.array([0.001, 0.01, 0.1])
    dictlistvalu['typeplanbrgt'] = ['term', 'dark']
    
    listname = dictlistvalu.keys()
    dicttemp = dict()
    for name in listname:
        
        dictmodl = dict()
        for nametemp in listname:
            if nametemp == name:
                continue
            dicttemp[nametemp] = np.array([dictlistvalu[nametemp][int(len(dictlistvalu[nametemp]) / 2.)]])
        for k in range(len(dictlistvalu[name])):
            
            dicttemp[name] = np.array([dictlistvalu[name][k]])
            
            # title for the plots
            dictstrgtitl = dict()
            dictstrgtitl['radistar'] = radistar
            for name in ['radicomp', 'pericomp']:
                dictstrgtitl[name] = dicttemp[name]
            strgtitl = retr_strgtitl(dictstrgtitl)
                
            # generate light curve
            dictoutp = ephesus.retr_rflxtranmodl(time, \
                                                 radistar=radistar, \
                                                 massstar=massstar, \
                                         
                                                 epocmtracomp=epocmtracomp, \
                                                 pericomp=dicttemp['pericomp'], \
                                                 rratcomp=dicttemp['rratcomp'], \
                                                 cosicomp=dicttemp['cosicomp'], \
                                                 rsmacomp=dicttemp['rsmacomp'], \
                                                 tolerrat=dicttemp['tolerrat'], \
                                                 
                                                 typeplanbrgt=dicttemp['typeplanbrgt'], \
                                                 
                                                 #perimoon=perimoon, \
                                                 #epocmtramoon=epocmtramoon, \
                                                 #radimoon=radimoon, \

                                                 typecoor=typecoor, \
                                                 typesyst=typesyst, \
                                                 boolintp=boolintp, \
                                                 
                                                 #pathfoldanim=pathimag, \
                                                 boolwritover=boolwritover, \
                                                 strgextn=strgextn, \
                                                 titlvisu=strgtitl, \
                                                )
            
            strgextn = name + '_%02d' % k
            
            # dictionary for the configuration
            dictmodl[strgextn] = dict()
            dictmodl[strgextn]['time'] = time
            dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1)
        
        print('Making a light curve plot...')
        
        strgextn = 'psys_cmps_%s' % name
        lablyaxi = 'Relative flux - 1 [ppm]'
        pathplot = ephesus.plot_lcur(pathimag, \
                                     dictmodl=dictmodl, \
                                     
                                     boolwritover=boolwritover, \
                                     strgextn=strgextn, \
                                     lablyaxi=lablyaxi, \
                                     strgtitl=strgtitl, \
                                    )
    
    

    



def test_pbox_psys():
    
    
    
    pass



globals().get(sys.argv[1])(*sys.argv[2:])
