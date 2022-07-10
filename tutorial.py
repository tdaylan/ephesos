import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy
import pergamon

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
                    #'psys', \
                    'psyspcur', \
                    'psysmoon', \
                    'psystranmoon', \
                    'cosc', \
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
        listtitlcomp = []
        listboolcompexcl = []
        if typesyst == 'cosc':
            listtitlcomp.append('Compact Objects with a stellar companion')
            namesamp = 'COSC'
        if typesyst == 'psys' or typesyst == 'psyspcur':
            listtitlcomp.append('Planets')
            namesamp = 'planet'
        
        listdictlablcolrpopl[-1][strgpoplcomptotl] = ['All', 'black']
        listdictlablcolrpopl[-1][strgpoplcomptran] = ['Transiting', 'blue']
        listboolcompexcl.append(False)

        typeanls = '%s' % (typesyst)
        pergamon.init( \
                      typeanls + 'tuto', \
                      dictpopl=dictpoplcomp, \
                      
                      listdictlablcolrpopl=listdictlablcolrpopl, \
                      listboolcompexcl=listboolcompexcl, \
                      listtitlcomp=listtitlcomp, \
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
            
            for a in range(numbiterintp):
                
                #if a < 2:
                #    continue

                # string for the configuration
                strgthissyst = '%s_%04d' % (typesyst, k)
                
                if a == 0:
                    strgextn = strgthissyst
                if a == 1:
                    strgextn = strgthissyst + '_intpstar'
                if a == 2:
                    strgextn = strgthissyst + '_intpplan'
                
                # dictionary for the configuration
                dictmodl[strgextn] = dict()
                
                if a == 0:
                    boolintp = False
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '-'
                    dictmodl[strgextn]['labl'] = 'Full Evaluation'
                if a == 1:
                    boolintp = True
                    strgextn = strgthissyst + '_intpstar'
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '--'
                    dictmodl[strgextn]['labl'] = 'Interpolation, Star grid'
                if a == 2:
                    boolintp = True
                    strgextn = strgthissyst + '_intpplan'
                    typecoor = 'plan'
                    dictmodl[strgextn]['lsty'] = '-.'
                    dictmodl[strgextn]['labl'] = 'Interpolation, Planet grid'
                
                # title for the plots
                titl = '$R_* = %.3g R_\odot, R_{p} = ' % (dictpoplstar['radistar'][k])
                for u, radi in enumerate(dictpoplcomp['radicomp'][indxcompstar[k]]):
                    titl += '%.3g' % (radi)
                    if u != len(dictpoplcomp['radicomp'][indxcompstar[k]]) - 1:
                        titl += ', '
                titl += ' R_{\oplus}, P = $'
                for u, radi in enumerate(dictpoplcomp['pericomp'][indxcompstar[k]]):
                    titl += '%.3g' % (radi)
                    if u != len(dictpoplcomp['pericomp'][indxcompstar[k]]) - 1:
                        titl += ', '
                titl += ' days'

                # generate light curve
                dictoutp = ephesus.retr_rflxtranmodl(time, \
                                                     radistar=dictpoplstar['radistar'][k], \
                                                     massstar=dictpoplstar['massstar'][k], \
                                                     
                                                     pericomp=dictpoplcomp['pericomp'][indxcompstar[k]], \
                                                     epocmtracomp=dictpoplcomp['epocmtracomp'][indxcompstar[k]], \
                                                     inclcomp=dictpoplcomp['inclcomp'][indxcompstar[k]], \
                                                     radicomp=dictpoplcomp['radicomp'][indxcompstar[k]], \
                                                     masscomp=dictpoplcomp['masscomp'][indxcompstar[k]], \
                                                     
                                                     perimoon=perimoon, \
                                                     epocmtramoon=epocmtramoon, \
                                                     radimoon=radimoon, \
                                                     
                                                     typeplanbrgt=typeplanbrgt, \

                                                     typecoor=typecoor, \

                                                     boolintp=boolintp, \
                                                     
                                                     pathfoldanim=pathimag, \
                                                     boolwritover=boolwritover, \
                                                     strgextn=strgextn, \
                                                     titlvisu=titl, \
                                                    )
                
                print('Making a light curve plot...')
                
                dictmodl[strgextn]['time'] = time
                dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1)
            
            lablyaxi = 'Relative flux - 1 [ppm]'
            pathplot = ephesus.plot_lcur(pathimag, \
                                         dictmodl=dictmodl, \
                                         
                                         boolwritover=boolwritover, \
                                         strgextn=strgthissyst, \
                                         lablyaxi=lablyaxi, \
                                         titl=titl, \
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
                                             titl=titl, \
                                            )
    

def test_pbox_psys():
    
    
    
    pass



globals().get(sys.argv[1])(*sys.argv[2:])
