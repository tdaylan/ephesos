import sys, os

import numpy as np

import ephesus
from tdpy.util import summgene
import tdpy
import pergamon

def make_popl():
    
    # type of the population of systems
    ## TESS 2-min target list during the nominal mission
    #typepoplsyst = 'tessprms2min'
    typepoplsyst = 'gene'
    
    # get dictionaries for stars, companions, and moons
    dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = ephesus.retr_dictpoplstarcomp( \
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
      
    del dictpoplcomp[strgpoplcomptotl]['idenstar']
    del dictpoplcomp[strgpoplcomptran]['idenstar']
    
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


def make_rflx( \
             ):
    '''
    Make and visualize stars with companions and their relative flux light curves
    '''
    
    # fix the seed
    np.random.seed(0)

    # time axis
    time = 0.1 * np.arange(-1., 1., 1. / 24. / 60.) # [1-min]
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    
    #typelang = 'Turkish'
    typelang = 'English'

    lablxaxi = 'Time from mid-transit [hours]'
     
    # type of systems to be illustrated
    listtypesyst = [ \
                    'psys', \
                    #'turkey', \
                    #'psysmoon', \
                    #'psyspcur', \
                    #'cosc', \
                    ]
    
    # Boolean flag to overwrite visuals
    boolwritover = True

    # Boolean flag to diagnose
    booldiag = True

    # path of the folder for visuals
    pathbase = os.environ['EPHESUS_DATA_PATH'] + '/'
    pathimag = pathbase + 'imag/lightCurve/'
    pathdata = pathbase + 'data/LightCurve/'
    os.system('mkdir -p %s' % pathimag)
    
    typefileplot = 'png'

    listcolr = ['g', 'b', 'firebrick', 'orange', 'olive']
    
    # number of systems
    numbsyst = 30

    def retr_strgtitl(dictstrgtitl):
        
        strgtitl = ''
        if 'radistar' in dictstrgtitl:
            strgtitl += '$R_*$ = %.1f $R_\odot$' % dictstrgtitl['radistar']
        if typesyst == 'cosc' and 'massstar' in dictstrgtitl:
            if len(strgtitl) > 0 and strgtitl[-2:] != ', ':
                strgtitl += ', '
            strgtitl += '$M_*$ = %.1f $M_\odot$' % dictstrgtitl['massstar']
            
        cntr = 0
        for kk, name in enumerate(listnamevarbcomp):
            
            if name == 'epocmtracomp' or not name in dictstrgtitl:
                continue
                
            for j, valu in enumerate(dictstrgtitl[name]):
                
                if len(strgtitl) > 0 and strgtitl[-2:] != ', ':
                    strgtitl += ', '
                
                strgtitl += '%s = ' % dictlabl['root'][name]
                
                if name == 'typebrgtcomp':
                    strgtitl += '%s' % (valu)
                else:
                    strgtitl += '%.3g' % (valu)
                
                if name in dictlabl['unit'] and dictlabl['unit'][name] != '':
                    strgtitl += ' %s' % dictlabl['unit'][name]
        
                cntr += 1

        return strgtitl


    dictfact = tdpy.retr_factconv()

    dictsimu = dict()

    dictlistvalu = dict()
    
    for typesyst in listtypesyst:
        
        print('typesyst')
        print(typesyst)
        
        pathimagpopl = pathimag + typesyst + '/'
        pathdatapopl = pathdata + typesyst + '/'
        
        dictlabl = dict()
        dictlabl['root'] = dict()
        dictlabl['unit'] = dict()
        dictlabl['totl'] = dict()
        
        listnamevarbcomp = ['pericomp', 'epocmtracomp', 'cosicomp', 'rsmacomp'] 
        listnamevarbstar = ['radistar', 'coeflmdklinr', 'coeflmdkquad']
        
        # model resolution
        listnamevarbsimu = ['tolerrat', 'diffphas']
        #listnamevarbsimu += ['typecoor']
        
        if typesyst == 'cosc':
            listnamevarbcomp += ['masscomp']
            listnamevarbstar += ['massstar']
        else:
            listnamevarbsimu += ['resoplan']
            listnamevarbcomp += ['radicomp']
            #listnamevarbcomp += ['typebrgtcomp']
        
        listnamevarbsyst = listnamevarbstar + listnamevarbcomp
        listnamevarbtotl = listnamevarbsyst + listnamevarbsimu
        
        dictdefa = dict()
        dictdefa['cosicomp'] = dict()
        dictdefa['cosicomp']['labl'] = ['$\cos i$', '']
        dictdefa['cosicomp']['scal'] = 'self'
        if typesyst != 'cosc':
            dictdefa['radicomp'] = dict()
            dictdefa['radicomp']['labl'] = ['$R_p$', '$R_\oplus$']
            dictdefa['radicomp']['scal'] = 'self'

        listlablpara, listscalpara, listlablroot, listlablunit, listlabltotl = tdpy.retr_listlablscalpara(listnamevarbtotl, \
                                                                                            dictdefa=dictdefa, typelang=typelang, boolmath=True)
        
        # turn lists of labels into dictionaries
        for k, strgvarb in enumerate(listnamevarbtotl):
            dictlabl['root'][strgvarb] = listlablroot[k]
            dictlabl['unit'][strgvarb] = listlablunit[k]
            dictlabl['totl'][strgvarb] = listlabltotl[k]
            

        # generate light curves from a grid in parameters
        dictvaludefa = dict()
        dictvaludefa['radistar'] = 1. # [R_S]
        if typesyst == 'cosc':
            dictvaludefa['massstar'] = 1. # [M_S]
        dictvaludefa['coeflmdklinr'] = 0.4
        dictvaludefa['coeflmdkquad'] = 0.25
        
        dictvaludefa['epocmtracomp'] = 0.
        dictvaludefa['pericomp'] = 3.
        dictvaludefa['cosicomp'] = 0.
        dictvaludefa['rsmacomp'] = 0.1
        if typesyst == 'cosc':
            dictvaludefa['masscomp'] = 1e-1 # [M_S]
        else:
            dictvaludefa['resoplan'] = 0.01
            dictvaludefa['radicomp'] = 10. # [R_E]
            #dictvaludefa['typebrgtcomp'] = np.array(['dark'])
        dictvaludefa['tolerrat'] = 0.005
        dictvaludefa['diffphas'] = 1e-4
        if typesyst == 'psysmoon':
            dictvaludefa['typecoor'] = 'star'
        else:
            dictvaludefa['typecoor'] = 'comp'
        
        # list of names for runs
        listnamevarbuber = []
        for name in listnamevarbtotl:
            if name != 'epocmtracomp' and name != 'radicomp':
                listnamevarbuber.append(name)
        listnamevarbuber += ['visu']
        if typesyst != 'cosc':
            if typesyst != 'turkey':
                listnamevarbuber += ['radicompsmal', 'radicomplarg']

        for name in listnamevarbuber:
            dictlistvalu[name] = dict()
            dictlistvalu[name]['defa'] = dict()
            dictlistvalu[name]['vari'] = dict()
        
        dictlistvalu['radistar']['vari']['radistar'] = np.array([0.7, 1., 1.2])
        if typesyst == 'cosc':
            dictlistvalu['massstar']['vari']['massstar'] = np.array([0.3, 1., 10.])
        dictlistvalu['coeflmdklinr']['vari']['coeflmdklinr'] = np.array([0.1, 0.4, 0.7])
        dictlistvalu['coeflmdkquad']['vari']['coeflmdkquad'] = np.array([0.1, 0.25, 0.4])
        if typesyst == 'cosc':
            dictlistvalu['masscomp']['vari']['masscomp'] = np.array([1., 5., 100.])
        else:
            pass
            #dictlistvalu['typebrgtcomp']['vari']['typebrgtcomp'] = np.array(['term', 'dark'])
        if typesyst != 'cosc':
            if typesyst != 'turkey':
                dictlistvalu['radicompsmal']['vari']['radicomp'] = np.array([0.5, 1., 1.5]) # [R_E]
                dictlistvalu['radicomplarg']['vari']['radicomp'] = np.array([8., 10., 12.]) # [R_E]
                
        #dictlistvalu['visu']['defa']['diffphas'] = 0.00003
        dictlistvalu['visu']['defa']['typecoor'] = 'star'
        if typesyst != 'cosc':
            #dictlistvalu['visu']['defa']['resoplan'] = 0.01
            if typesyst == 'turkey':
                dictlistvalu['visu']['vari']['radicomp'] = np.array([30.]) # [R_E]
            else:
                dictlistvalu['visu']['vari']['radicomp'] = np.array([10.]) # [R_E]
        
        #dictlistvalu['typecoor']['vari']['typecoor'] = ['star', 'comp']
        dictlistvalu['pericomp']['vari']['pericomp'] = np.array([1., 3., 5.])
        dictlistvalu['cosicomp']['vari']['cosicomp'] = np.array([0., 0.06, 0.09])
        dictlistvalu['rsmacomp']['vari']['rsmacomp'] = np.array([0.05, 0.1, 0.15])
        dictlistvalu['tolerrat']['vari']['tolerrat'] = np.array([0.001, 0.01, 0.1])
        if typesyst != 'cosc':
            dictlistvalu['resoplan']['vari']['resoplan'] = np.array([0.01, 0.03, 0.08])
        dictlistvalu['diffphas']['vari']['diffphas'] = np.array([0.00001, 0.00003, 0.0001, 0.0003, 0.001])
        
        numbcomp = 1
                
        dicttemp = dict()
        for nameruns in dictlistvalu.keys():
            
            print('nameruns')
            print(nameruns)
            print('dictlistvalu[nameruns]')
            print(dictlistvalu[nameruns])
            print('dictlistvalu[nameruns][vari]')
            print(dictlistvalu[nameruns]['vari'])
            print('dictlistvalu[nameruns][vari].keys()')
            print(dictlistvalu[nameruns]['vari'].keys())
            nameparavari = list(dictlistvalu[nameruns]['vari'].keys())[0]
            
            # for all other parameters, set the central value
            for nametemp in listnamevarbtotl:
                if nametemp == nameparavari:
                    continue
                
                if nametemp in dictlistvalu[nameruns]['defa'].keys():
                    valu = dictlistvalu[nameruns]['defa'][nametemp]
                else:
                    valu = dictvaludefa[nametemp]
                
                if nametemp in listnamevarbcomp:
                    dicttemp[nametemp] = np.array([valu])
                else:
                    dicttemp[nametemp] = valu
            
            dictmodl = dict()
            
            if nameruns == 'visu':
                
                pathfoldanim = pathimag
                
                print('temp')
                pathfoldanim = None

            else:
                pathfoldanim = None
                
            for k in range(len(dictlistvalu[nameruns]['vari'][nameparavari])):
                
                valu = dictlistvalu[nameruns]['vari'][nameparavari][k]
                if nameparavari in listnamevarbcomp:
                    dicttemp[nameparavari] = np.array([valu])
                else:
                    dicttemp[nameparavari] = valu
                
                # title for the plots
                dictstrgtitl = dict()
                for namevarbtotl in listnamevarbtotl:
                    dictstrgtitl[namevarbtotl] = dicttemp[namevarbtotl]
                strgtitl = retr_strgtitl(dictstrgtitl)
                
                dicttemp['coeflmdk'] = np.array([dicttemp['coeflmdklinr'], dicttemp['coeflmdkquad']])
                dicttemptemp = dict()
                for nametemp in dicttemp:
                    if nametemp != 'coeflmdklinr' and nametemp != 'coeflmdkquad':
                        dicttemptemp[nametemp] = dicttemp[nametemp]
                
                if typesyst != 'cosc':
                    dicttemptemp['rratcomp'] = dicttemptemp['radicomp'] / dicttemptemp['radistar'] / dictfact['rsre']
                    del dicttemptemp['radicomp']
                    del dicttemptemp['radistar']

                boolintp = True
                #if dicttemptemp['typecoor'] == 'star':
                #    boolintp = False
                #else:
                #    boolintp = True

                strgextn = typesyst + '_' + nameruns + '%02d' % k
                
                dicttemptemp['typelmdk'] = 'quad'
                dicttemptemp['typesyst'] = typesyst
                dicttemptemp['typenorm'] = 'edgeleft'
                dicttemptemp['pathfoldanim'] = pathfoldanim
                dicttemptemp['typelang'] = typelang
                dicttemptemp['typefileplot'] = typefileplot
                dicttemptemp['booldiag'] = booldiag
                dicttemptemp['typeverb'] = 1
                dicttemptemp['boolintp'] = boolintp
                dicttemptemp['boolwritover'] = boolwritover
                dicttemptemp['strgextn'] = strgextn
                dicttemptemp['titlvisu'] = strgtitl

                print('dicttemptemp')
                print(dicttemptemp)
                
                # generate light curve
                dictoutp = ephesus.retr_rflxtranmodl(time, **dicttemptemp)
               
                # dictionary for the configuration
                dictmodl[strgextn] = dict()
                dictmodl[strgextn]['time'] = time * 24. # [hours]
                if dictlistvalu[nameruns]['vari'][nameparavari].size > 1:
                    if not isinstance(dictlistvalu[nameruns]['vari'][nameparavari][k], str):
                        dictmodl[strgextn]['labl'] = '%s = %.3g %s' % (dictlabl['root'][nameparavari], \
                                            dictlistvalu[nameruns]['vari'][nameparavari][k], dictlabl['unit'][nameparavari])
                    else:
                        dictmodl[strgextn]['labl'] = '%s' % (dictlistvalu[nameruns]['vari'][nameparavari][k])
                dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1)
                
                dictmodl[strgextn]['colr'] = listcolr[k]

            print('Making a light curve plot...')
            
            if len(dictlistvalu[nameruns]['vari'][nameparavari]) == 1:
                listxdatvert = [-0.5 * 24. * dictoutp['duratrantotl'], 0.5 * 24. * dictoutp['duratrantotl']] 
                if 'duratranfull' in dictoutp:
                    listxdatvert += [-0.5 * 24. * dictoutp['duratranfull'], 0.5 * 24. * dictoutp['duratranfull']]
                listxdatvert = np.array(listxdatvert)
            else:
                listxdatvert = None

            # title for the plots
            dictstrgtitl = dict()
            for namevarbtotl in listnamevarbtotl:
                if namevarbtotl != nameparavari or dictlistvalu[nameruns]['vari'][nameparavari].size == 1:
                    dictstrgtitl[namevarbtotl] = dicttemp[namevarbtotl]
            strgtitl = retr_strgtitl(dictstrgtitl)
                
            strgextn = '%s_cmps_%s' % (typesyst, nameruns)
            
            lablyaxi = 'Relative flux - 1 [ppm]'
            pathplot = ephesus.plot_lcur(pathimag, \
                                         dictmodl=dictmodl, \
                                         
                                         typefileplot=typefileplot, \
                                         boolwritover=boolwritover, \
                                         listxdatvert=listxdatvert, \
                                         strgextn=strgextn, \
                                         lablxaxi=lablxaxi, \
                                         lablyaxi=lablyaxi, \
                                         strgtitl=strgtitl, \
                                         typesigncode='ephesus', \
                                        )
            print('')
            print('')
            print('')
    

globals().get(sys.argv[1])(*sys.argv[2:])
