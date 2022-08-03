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
    time = 0.1 * np.linspace(-1., 1., 10000)
    minmtime = np.amin(time)
    maxmtime = np.amax(time)
    
    lablxaxi = 'Time from mid-transit [hours]'
     
    # type of systems to be illustrated
    listtypesyst = [ \
                    'psys', \
                    'cosc', \
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
    
    typefileplot = 'png'

    listcolr = ['g', 'b', 'r', 'orange', 'olive']
    
    # number of systems
    numbsyst = 30

    #dictlabl['labl']['tolerrat'] = '$\delta_r$'
    #dictlabl['unit']['radicomp'] = '$R_{\oplus}$'
    #dictlabl['unit']['pericomp'] = ' days'
    
    # labels for parameters

    # type of the population of systems
    ## TESS 2-min target list during the nominal mission
    #typepoplsyst = 'tessprms2min'
    typepoplsyst = 'gene'
    
    def retr_strgtitl(dictstrgtitl):
        
        cntr = 0
        strgtitl = '$R_*$ = %.1f $R_\odot$' % dictstrgtitl['radistar']
        
        listnamecomp = []
        for name in dictstrgtitl.keys():
            if name.endswith('comp') and dictstrgtitl[name] is not None:
                listnamecomp.append(name)
                numbcomp = len(dictstrgtitl[name])
        
        for name in dictstrgtitl.keys():
            
            if name in listnamecomp:
                
                for j, valu in enumerate(dictstrgtitl[name]):
                    strgtitl += ', %s = ' % dictlabl['root'][name]
                    
                    strgtitl += '%.3g' % (valu)
                    if j != numbcomp - 1:
                        strgtitl += ', '
                    
                    #if cntr % 3 == 0:
                    #    strgtitl += '$\n$'
                    
                    cntr += 1

                if name in dictlabl['unit'] and dictlabl['unit'][name] != '':
                    strgtitl += ' %s' % dictlabl['unit'][name]
        
        return strgtitl


    for typesyst in listtypesyst:
        
        print('typesyst')
        print(typesyst)
        
        pathimagpopl = pathimag + typesyst + '/'
        pathdatapopl = pathdata + typesyst + '/'
        
        # get dictionaries for stars, companions, and moons
        dictpoplstar, dictpoplcomp, dictpoplmoon, dictcompnumb, dictcompindx, indxcompstar, indxmooncompstar = ephesus.retr_dictpoplstarcomp( \
                                                                                                                                             typesyst, \
                                                                                                                                             
                                                                                                                                             typepoplsyst, \
                                                                                                                                             epocmtracomp=0., \
                                                                                                                                             booltoyysunn=True, \
                                                                                                                                             typesamporbtcomp='peri', \
                                                                                                                                             minmmasscomp=100., \
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
            #numbiterintp = 3
            numbiterintp = 1
        else:
            numbiterintp = 1
        
        dictlabl = dict()
        dictlabl['root'] = dict()
        dictlabl['unit'] = dict()
        dictlabl['totl'] = dict()
        liststrgvarb = ['radistar', 'pericomp', 'cosicomp', 'rsmacomp', 'radicomp', 'rratcomp', 'coeflmdklinr', 'coeflmdkquad', 'tolerrat', 'resoplan', 'diffphas']
        
        dictdefa = dict()
        dictdefa['cosicomp'] = dict()
        dictdefa['cosicomp']['labl'] = ['$\cos i$', '']
        dictdefa['cosicomp']['scal'] = 'self'
        dictdefa['radicomp'] = dict()
        dictdefa['radicomp']['labl'] = ['$R_p$', '$R_\oplus$']
        dictdefa['radicomp']['scal'] = 'self'

        listlablpara, listscalpara, listlablroot, listlablunit, listlabltotl = tdpy.retr_listlablscalpara(liststrgvarb, dictdefa)
        
        # turn lists of labels into dictionaries
        for k, strgvarb in enumerate(liststrgvarb):
            dictlabl['root'][strgvarb] = listlablroot[k]
            dictlabl['unit'][strgvarb] = listlablunit[k]
            dictlabl['totl'][strgvarb] = listlabltotl[k]

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
            
            # number of components
            numbcomp = len(dictpoplcomp['rsmacomp'][indxcompstar[k]])
            
            # title for the plots
            dictstrgtitl = dict()
            dictstrgtitl['radistar'] = dictpoplstar['radistar'][k]
            for namevarbcomp in ['pericomp', 'cosicomp', 'rsmacomp']:
                dictstrgtitl[namevarbcomp] = dictpoplcomp[namevarbcomp][indxcompstar[k]]
            dictstrgtitl['radicomp'] = radicomp
            strgtitl = retr_strgtitl(dictstrgtitl)
                
            # string for the configuration
            strgthissyst = '%s_%04d' % (typesyst, k)
            
            for a in range(numbiterintp):
                
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
                    if numbiterintp > 1:
                        dictmodl[strgextn]['lsty'] = '-.'
                        dictmodl[strgextn]['labl'] = 'Interpolation, Planet grid'
                    else:
                        dictmodl[strgextn]['lsty'] = '-'
                    pathfoldanim = pathimag
                if a == 1:
                    boolintp = True
                    strgextn = strgthissyst + '_intpstar'
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '--'
                    dictmodl[strgextn]['labl'] = 'Interpolation, Star grid'
                    pathfoldanim = pathimag
                if a == 2:
                    boolintp = False
                    typecoor = 'star'
                    dictmodl[strgextn]['lsty'] = '-'
                    dictmodl[strgextn]['labl'] = 'Full Evaluation'
                    pathfoldanim = None
                
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
                                                     
                                                     typefileplot=typefileplot, \

                                                     pathfoldanim=pathfoldanim, \
                                                     
                                                     boolwritover=boolwritover, \
                                                     strgextn=strgextn, \
                                                     titlvisu=strgtitl, \
                                                    )
                
                print('Making a light curve plot...')
                
                dictmodl[strgextn]['time'] = time * 24. # [hours]
                dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1.)
            
            listxdatvert = [-0.5 * 24. * dictoutp['duratrantotl'], 0.5 * 24. * dictoutp['duratrantotl']] 
            if 'duratranfull' in dictoutp:
                listxdatvert += [-0.5 * 24. * dictoutp['duratranfull'], 0.5 * 24. * dictoutp['duratranfull']]
            listxdatvert = np.array(listxdatvert)
            #listcolrvert = 4 * [listcolr[k]]

            lablyaxi = 'Relative flux - 1 [ppm]'
            pathplot = ephesus.plot_lcur(pathimag, \
                                         dictmodl=dictmodl, \
                                         
                                         typefileplot=typefileplot, \
                                         boolwritover=boolwritover, \
                                         strgextn=strgthissyst, \
                                         listxdatvert=listxdatvert, \
                                         #listcolrvert=listcolrvert, \

                                         lablxaxi=lablxaxi, \
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
                                             typefileplot=typefileplot, \
                                             boolwritover=boolwritover, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                            )
    
    dictlistvalu = dict()
    
    typesyst = 'psys'
    radistar = 1.
    massstar = None
    epocmtracomp = np.array([0.])
    


    dictlistvalu['pericomp'] = np.array([1., 3., 5.])
    dictlistvalu['rratcomp'] = np.array([0.05, 0.1, 0.15])
    dictlistvalu['coeflmdklinr'] = np.array([0.1, 0.4, 0.7])
    dictlistvalu['coeflmdkquad'] = np.array([0.1, 0.25, 0.4])
    dictlistvalu['cosicomp'] = np.array([0., 0.05, 0.09])
    dictlistvalu['rsmacomp'] = np.array([0.05, 0.1, 0.15])
    dictlistvalu['tolerrat'] = np.array([0.001, 0.01, 0.1])
    dictlistvalu['resoplan'] = np.array([0.005, 0.01, 0.05, 0.1, 0.2])
    dictlistvalu['diffphas'] = np.array([0.00001, 0.00003, 0.0001, 0.0003, 0.001])
    dictlistvalu['typeplanbrgt'] = ['term', 'dark']
    
    numbcomp = 1
            
    listname = dictlistvalu.keys()
    dicttemp = dict()
    for name in listname:
        
        dictmodl = dict()
        for nametemp in listname:
            if nametemp == name:
                continue
            dicttemp[nametemp] = np.array([dictlistvalu[nametemp][int(len(dictlistvalu[nametemp]) / 2.)]])
            
        listxdatvert = []
        listcolrvert = []
        for k in range(len(dictlistvalu[name])):
            
            dicttemp[name] = np.array([dictlistvalu[name][k]])
            
            # title for the plots
            dictstrgtitl = dict()
            dictstrgtitl['radistar'] = radistar
            for namevarbcomp in ['rratcomp', 'pericomp', 'rsmacomp', 'cosicomp']:
                if namevarbcomp != name:
                    dictstrgtitl[namevarbcomp] = dicttemp[namevarbcomp]
            strgtitl = retr_strgtitl(dictstrgtitl)
            
            coeflmdk = np.array([dicttemp['coeflmdklinr'], dicttemp['coeflmdkquad']])
            
            if dicttemp['resoplan'][0] == np.amin(dictlistvalu['resoplan']):
                pathfoldanim = pathimag
            else:
                pathfoldanim = None
            
            # generate light curve
            dictoutp = ephesus.retr_rflxtranmodl(time, \
                                                 massstar=massstar, \
                                         
                                                 epocmtracomp=epocmtracomp, \
                                                 pericomp=dicttemp['pericomp'], \
                                                 rratcomp=dicttemp['rratcomp'], \
                                                 cosicomp=dicttemp['cosicomp'], \
                                                 rsmacomp=dicttemp['rsmacomp'], \
                                                 tolerrat=dicttemp['tolerrat'][0], \
                                                 resoplan=dicttemp['resoplan'][0], \
                                                 diffphas=dicttemp['diffphas'][0], \
                                                 coeflmdk=coeflmdk, \
                                                 
                                                 typeplanbrgt=dicttemp['typeplanbrgt'], \
                                                 
                                                 #perimoon=perimoon, \
                                                 #epocmtramoon=epocmtramoon, \
                                                 #radimoon=radimoon, \

                                                 typelmdk='quad', \
                                                 
                                                 typecoor=typecoor, \
                                                 typesyst=typesyst, \
                                
                                                 #booldiagtran=True, \
                                                 
                                                 typefileplot=typefileplot, \

                                                 boolintp=boolintp, \
                                                 
                                                 pathfoldanim=pathfoldanim, \
                                                 boolwritover=boolwritover, \
                                                 strgextn=strgextn, \
                                                 titlvisu=strgtitl, \

                                                )
           

            dictfact = ephesus.retr_factconv()

            #import allesfitter
            #params = dict()
            #params['b_rr'] = dicttemp['rratcomp'][0]
            #model_i = allesfitter.calculate_model(params, 'TESS','flux')

            #import ellc
            #a = radistar * (1. + dicttemp['rratcomp'][0])/ dicttemp['rsmacomp'][0] / dictfact['aurs']
            #model_flux1, model_flux2 = ellc.fluxes(
            #                    t_obs =       time, \
            #                    sbratio =    0., \
            #                    radius_1 =    radistar / a / dictfact['aurs'], \
            #                    radius_2 =    dicttemp['rratcomp'][0], \
            #                    #sbratio =     params[companion+'_sbratio_'+inst], 
            #                    incl =        np.arccos(dicttemp['cosicomp'][0]) * 180. / np.pi, \
            #                    t_zero =      epocmtracomp, \
            #                    period =      dicttemp['pericomp'][0], \
            #                    a =            a, \
            #                    #q =           params[companion+'_q'],
            #                    #f_c =         params[companion+'_f_c'],
            #                    #f_s =         params[companion+'_f_s'],
            #                    
            #                    #ldc_1 =       coeflmdk, \
            #                    
            #                    #ldc_2 =       params[companion+'_ldc_'+inst],
            #                    #gdc_1 =       params['host_gdc_'+inst],
            #                    #gdc_2 =       params[companion+'_gdc_'+inst],
            #                    #didt =        params['didt_'+inst], 
            #                    #domdt =       params['domdt_'+inst], 
            #                    #rotfac_1 =    params['host_rotfac_'+inst], 
            #                    #rotfac_2 =    params[companion+'_rotfac_'+inst], 
            #                    #hf_1 =        params['host_hf_'+inst], #1.5, 
            #                    #hf_2 =        params[companion+'_hf_'+inst], #1.5,
            #                    #bfac_1 =      params['host_bfac_'+inst],
            #                    #bfac_2 =      params[companion+'_bfac_'+inst], 
            #                    #heat_1 =      divide(params['host_heat_'+inst],2.),
            #                    #heat_2 =      divide(params[companion+'_heat_'+inst],2.),
            #                    #lambda_1 =    params['host_lambda'], 
            #                    #lambda_2 =    params[companion+'_lambda'], 
            #                    #vsini_1 =     params['host_vsini'],
            #                    #vsini_2 =     params[companion+'_vsini'], 
            #                    #t_exp =       t_exp,
            #                    #n_int =       n_int,
            #                    #grid_1 =      settings['host_grid_'+inst],
            #                    #grid_2 =      settings[companion+'_grid_'+inst],
            #                    ld_1 =        'quad', \
            #                    #ld_2 =        settings[companion+'_ld_law_'+inst],
            #                    #shape_1 =     settings['host_shape_'+inst],
            #                    #shape_2 =     settings[companion+'_shape_'+inst],
            #                    #spots_1 =     params['host_spots_'+inst], 
            #                    #spots_2 =     params[companion+'_spots_'+inst], 
            #                    )

            #print('model_flux1')
            #summgene(model_flux1)
            #print('model_flux2')
            #summgene(model_flux2)



            strgextn = name + '_%02d' % k
            
            # dictionary for the configuration
            dictmodl[strgextn] = dict()
            dictmodl[strgextn]['time'] = time * 24. # [hours]
            if not isinstance(dictlistvalu[name][k], str):
                dictmodl[strgextn]['labl'] = '%s = %.3g %s' % (dictlabl['root'][name], dictlistvalu[name][k], dictlabl['unit'][name])
            else:
                dictmodl[strgextn]['labl'] = '%s' % (dictlistvalu[name][k])
            dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1)
            
            dictmodl[strgextn]['colr'] = listcolr[k]

            #listxdatvert.extend([-0.5 * dictoutp['duratrantotl'], 0.5 * dictoutp['duratrantotl']])
            #if 'duratranfull' in dictoutp:
            #    listxdatvert.extend([-0.5 * dictoutp['duratranfull'], 0.5 * dictoutp['duratranfull']])
            #listcolrvert.extend(4 * [listcolr[k]])
        
        print('Making a light curve plot...')
        
        strgextn = 'psys_cmps_%s' % name
        
        listxdatvert = np.array(listxdatvert)
            
        lablyaxi = 'Relative flux - 1 [ppm]'
        pathplot = ephesus.plot_lcur(pathimag, \
                                     dictmodl=dictmodl, \
                                     
                                     #listxdatvert=listxdatvert, \
                                     #listcolrvert=listcolrvert, \
                                     typefileplot=typefileplot, \
                                     boolwritover=boolwritover, \
                                     strgextn=strgextn, \
                                     lablxaxi=lablxaxi, \
                                     lablyaxi=lablyaxi, \
                                     strgtitl=strgtitl, \
                                    )
    
    

    



def test_pbox_psys():
    
    
    
    pass



globals().get(sys.argv[1])(*sys.argv[2:])
