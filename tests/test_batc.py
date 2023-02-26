import numpy as np
import os
import tdpy
import ephesos

'''
Make and visualize stars with companions and their relative flux light curves
'''

# fix the seed
np.random.seed(0)

#typelang = 'Turkish'
typelang = 'English'

lablxaxi = 'Time from mid-transit [hours]'
 
# type of systems to be illustrated
listtypesyst = [ \
                'psys', \
                #'turkey', \
                'psysdiskedgehori', \
                #'psysmoon', \
                'psyspcur', \
                'cosc', \
                ]

# Boolean flag to overwrite visuals
boolwritover = False#True

# Boolean flag to diagnose
booldiag = True

# path of the folder for visuals
pathvisu = os.environ['EPHESOS_PATH'] + '/visuals/'
os.system('mkdir -p %s' % pathvisu)

typecoor = 'comp'
typefileplot = 'png'

listcolr = ['g', 'b', 'firebrick', 'orange', 'olive']

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
        
        if name == 'typebrgtcomp':
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

dicttdpy = tdpy.retr_dictstrg()

dictsimu = dict()

for typesyst in listtypesyst:
    
    print('typesyst')
    print(typesyst)

    boolsystpsys = typesyst.startswith('psys')
    
    pathvisupopl = pathvisu + dicttdpy[typesyst] + '/'
    
    listtypetarg = []
    if typesyst.startswith('psysdisk'):
        listtypetarg += ['ExoSaturn']
    if typesyst == 'psys':
        listtypetarg += ['HotJupiter']
    if typesyst == 'psyspcur':
        listtypetarg += ['WASP-43']
    
    for typetarg in listtypetarg:
        
        dictlistvalubatc = dict()

        dictlabl = dict()
        dictlabl['root'] = dict()
        dictlabl['unit'] = dict()
        dictlabl['totl'] = dict()
        
        listnamevarbcomp = ['pericomp', 'epocmtracomp', 'cosicomp', 'rsmacomp'] 
        if typesyst == 'psyspcur':
            listnamevarbcomp += ['offsphascomp']

        listnamevarbstar = ['radistar', 'coeflmdklinr', 'coeflmdkquad']
        
        # model resolution
        listnamevarbsimu = ['tolerrat']#, 'diffphas']
        #listnamevarbsimu += ['typecoor']
        
        if typesyst == 'cosc':
            listnamevarbcomp += ['masscomp']
            listnamevarbstar += ['massstar']
        else:
            listnamevarbsimu += ['resoplan']
            listnamevarbcomp += ['radicomp']
        if typesyst == 'psyspcur':
            listnamevarbcomp += ['typebrgtcomp']
        
        print('listnamevarbcomp')
        print(listnamevarbcomp)
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
        if typetarg == 'WASP-43':
            dictvaludefa['radistar'] = 0.67 # [R_S]
        else:
            dictvaludefa['radistar'] = 1. # [R_S]
        
        if typesyst == 'cosc':
            dictvaludefa['massstar'] = 1. # [M_S]
        
        if typetarg == 'WASP-43':
            dictvaludefa['coeflmdklinr'] = 0.1
            dictvaludefa['coeflmdkquad'] = 0.05
        else:
            dictvaludefa['coeflmdklinr'] = 0.4
            dictvaludefa['coeflmdkquad'] = 0.25
        
        dictvaludefa['epocmtracomp'] = 0.
        
        if typetarg == 'WASP-43':
            dictvaludefa['pericomp'] = 0.813475
        else:
            dictvaludefa['pericomp'] = 3.
        
        if typetarg == 'WASP-43':
            dictvaludefa['offsphascomp'] = 30.
        else:
            dictvaludefa['offsphascomp'] = 0.
        
        if typetarg == 'WASP-43':
            dictvaludefa['cosicomp'] = 0.134
        else:
            dictvaludefa['cosicomp'] = 0.
        
        if boolsystpsys:
            if typetarg == 'WASP-43':
                dictvaludefa['rsmacomp'] = 0.21
            else:
                dictvaludefa['rsmacomp'] = 0.1
        
        if typesyst == 'cosc':
            dictvaludefa['masscomp'] = 1e-1 # [M_S]
        else:
            dictvaludefa['resoplan'] = 0.01
            if typesyst == 'psys':
                dictvaludefa['radicomp'] = 10. # [R_E]
            if typesyst == 'psyspcur':
                if typetarg == 'WASP-43':
                    dictvaludefa['radicomp'] = 11. # [R_E]
                else:
                    dictvaludefa['radicomp'] = 20. # [R_E]
            #dictvaludefa['typebrgtcomp'] = np.array(['dark'])
        dictvaludefa['tolerrat'] = 0.005
        
        if typesyst == 'psyspcur':
            dictvaludefa['typebrgtcomp'] = 'sinusoidal'
            dictvaludefa['offsphas'] = 0.

        if typesyst == 'psysmoon':
            dictvaludefa['typecoor'] = 'star'
        else:
            dictvaludefa['typecoor'] = 'comp'
        
        # list of names of batches
        ## default batch with single element
        listnamebatc = ['defa']
        ## other batches that iterate over the model parameters
        print('listnamevarbtotl')
        print(listnamevarbtotl)
        print('typesyst')
        print(typesyst)
        print('typetarg')
        print(typetarg)
        for name in listnamevarbtotl:
            if typesyst.startswith('psysdisk'): 
                continue
            if typetarg != 'HotJupiter' and name != 'typebrgtcomp':
                continue
            if name == 'epocmtracomp' or name == 'radicomp':
                continue
            if typesyst == 'psyspcur' and name == 'pericomp':
                continue
            
            listnamebatc.append(name)
        
        if typesyst == 'psys':
            listnamebatc += ['radicompsmal', 'radicomplarg']
        
        print('listnamebatc')
        print(listnamebatc)
        for name in listnamebatc:
            dictlistvalubatc[name] = dict()
            dictlistvalubatc[name]['defa'] = dict()
            dictlistvalubatc[name]['vari'] = dict()
        
        if typetarg == 'HotJupiter':
            dictlistvalubatc['radistar']['vari']['radistar'] = np.array([0.7, 1., 1.2])
            if typesyst == 'cosc':
                dictlistvalubatc['massstar']['vari']['massstar'] = np.array([0.3, 1., 10.])
            dictlistvalubatc['coeflmdklinr']['vari']['coeflmdklinr'] = np.array([0.1, 0.4, 0.7])
            dictlistvalubatc['coeflmdkquad']['vari']['coeflmdkquad'] = np.array([0.1, 0.25, 0.4])
            if typesyst == 'cosc':
                dictlistvalubatc['masscomp']['vari']['masscomp'] = np.array([1., 5., 100.])
            else:
                pass
                #dictlistvalubatc['typebrgtcomp']['vari']['typebrgtcomp'] = np.array(['term', 'dark'])
            if typesyst == 'psys':
                dictlistvalubatc['radicompsmal']['vari']['radicomp'] = np.array([0.5, 1., 1.5]) # [R_E]
                dictlistvalubatc['radicomplarg']['vari']['radicomp'] = np.array([8., 10., 12.]) # [R_E]
                    
            #dictlistvalubatc['defa']['defa']['diffphas'] = 0.00003
        
        if typesyst == 'psyspcur':
            dictlistvalubatc['typebrgtcomp']['vari']['typebrgtcomp'] = np.array(['sinusoidal', 'sliced'])
        
        if typesyst == 'psyspcur':
            dictlistvalubatc['defa']['defa']['typecoor'] = 'comp'
        else:
            dictlistvalubatc['defa']['defa']['typecoor'] = 'star'

        if typesyst != 'cosc':
            #dictlistvalubatc['defa']['defa']['resoplan'] = 0.01
            if typesyst == 'turkey':
                dictlistvalubatc['defa']['vari']['radicomp'] = np.array([30.]) # [R_E]
            else:
                dictlistvalubatc['defa']['vari']['radicomp'] = np.array([10.]) # [R_E]
        
        if typetarg == 'HotJupiter':
            #dictlistvalubatc['typecoor']['vari']['typecoor'] = ['star', 'comp']
            if typesyst != 'psyspcur':
                dictlistvalubatc['pericomp']['vari']['pericomp'] = np.array([1., 3., 5.])
            dictlistvalubatc['cosicomp']['vari']['cosicomp'] = np.array([0., 0.07, 0.09])
            if typesyst == 'psyspcur':
                dictlistvalubatc['offsphascomp']['vari']['offsphascomp'] = np.array([0., -30., 40., 140.])
            dictlistvalubatc['rsmacomp']['vari']['rsmacomp'] = np.array([0.05, 0.1, 0.15])
            dictlistvalubatc['tolerrat']['vari']['tolerrat'] = np.array([0.001, 0.01, 0.1])
            if typesyst != 'cosc':
                dictlistvalubatc['resoplan']['vari']['resoplan'] = np.array([0.01, 0.03, 0.08])
            
            #dictlistvalubatc['diffphas']['vari']['diffphas'] = np.array([0.00001, 0.00003, 0.0001, 0.0003, 0.001])
        
        numbcomp = 1
                
        dicttemp = dict()
        for namebatc in dictlistvalubatc.keys():
            
            print('typesyst')
            print(typesyst)
            print('namebatc')
            print(namebatc)
            print('dictlistvalubatc[namebatc]')
            print(dictlistvalubatc[namebatc])
            print('dictlistvalubatc[namebatc][vari]')
            print(dictlistvalubatc[namebatc]['vari'])
            print('dictlistvalubatc[namebatc][vari].keys()')
            print(dictlistvalubatc[namebatc]['vari'].keys())
            nameparavari = list(dictlistvalubatc[namebatc]['vari'].keys())[0]
            print('nameparavari')
            print(nameparavari)

            # for all other parameters, set the central value
            print('listnamevarbtotl')
            print(listnamevarbtotl)
            for nametemp in listnamevarbtotl:
                if nametemp == nameparavari:
                    continue
                
                if nametemp in dictlistvalubatc[namebatc]['defa'].keys():
                    valu = dictlistvalubatc[namebatc]['defa'][nametemp]
                else:
                    valu = dictvaludefa[nametemp]
                
                if nametemp in listnamevarbcomp:
                    dicttemp[nametemp] = np.array([valu])
                else:
                    dicttemp[nametemp] = valu
            
            #if dicttemp['typebrgtcomp'] == 'sliced':
            #    raise Exception('')
            
            dictmodl = dict()
            
            print('temp')
            if namebatc == 'defa':
                pathfoldanim = pathvisu
                pathfoldanim = None
            else:
                pathfoldanim = None
                
            ## cadence of simulation
            cade = 30. / 24. / 60. / 60. # days
            ## duration of simulation
            if typesyst == 'psyspcur':
                durasimu = dicttemp['pericomp']
                if namebatc == 'longbase':
                    durasimu *= 3.
            else:
                durasimu = 6. / 24. # days

            duratrantotl = ephesos.retr_duratrantotl(dicttemp['pericomp'], dicttemp['rsmacomp'], dicttemp['cosicomp']) / 24. # [days]
            if typesyst == 'psyspcur':
                # minimum time
                minmtime = -0.25 * durasimu
                # maximum time
                maxmtime = 0.75 * durasimu
            else:
                # minimum time
                minmtime = -0.5 * 1.5 * duratrantotl
                # maximum time
                maxmtime = 0.5 * 1.5 * duratrantotl
            # time axis
            time = np.arange(minmtime, maxmtime, cade)
            
            for k in range(len(dictlistvalubatc[namebatc]['vari'][nameparavari])):
                
                valu = dictlistvalubatc[namebatc]['vari'][nameparavari][k]
                print('listnamevarbcomp')
                print(listnamevarbcomp)
                if nameparavari in listnamevarbcomp:
                    dicttemp[nameparavari] = np.array([valu])
                else:
                    dicttemp[nameparavari] = valu
                
                print('dicttemp')
                print(dicttemp)
                
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

                if typecoor == 'star':
                    boolintp = False
                else:
                    boolintp = True

                strgextn = '%s_%s' % (dicttdpy[typesyst], namebatc)
                if namebatc != 'defa':
                    strgextn += '%02d' % k

                dicttemptemp['typelmdk'] = 'quad'
                #dicttemptemp['typesyst'] = typesyst
                #dicttemptemp['typenorm'] = 'edgeleft'
                dicttemptemp['pathfoldanim'] = pathfoldanim
                dicttemptemp['typelang'] = typelang
                dicttemptemp['typefileplot'] = typefileplot
                dicttemptemp['booldiag'] = booldiag
                dicttemptemp['typeverb'] = 1
                dicttemptemp['boolintp'] = boolintp
                dicttemptemp['boolwritover'] = boolwritover
                dicttemptemp['strgextn'] = strgextn
                dicttemptemp['strgtitl'] = strgtitl
                dicttemptemp['typecoor'] = typecoor

                print('dicttemptemp')
                print(dicttemptemp)
                
                # generate light curve
                dictoutp = ephesos.eval_modl(time, typesyst, **dicttemptemp)
               
                # dictionary for the configuration
                dictmodl[strgextn] = dict()
                dictmodl[strgextn]['time'] = time * 24. # [hours]
                if dictlistvalubatc[namebatc]['vari'][nameparavari].size > 1:
                    if not isinstance(dictlistvalubatc[namebatc]['vari'][nameparavari][k], str):
                        dictmodl[strgextn]['labl'] = '%s = %.3g %s' % (dictlabl['root'][nameparavari], \
                                            dictlistvalubatc[namebatc]['vari'][nameparavari][k], dictlabl['unit'][nameparavari])
                    else:
                        dictmodl[strgextn]['labl'] = '%s' % (dictlistvalubatc[namebatc]['vari'][nameparavari][k])
                dictmodl[strgextn]['lcur'] = 1e6 * (dictoutp['rflx'] - 1)
                
                dictmodl[strgextn]['colr'] = listcolr[k]

            print('Making a light curve plot...')
            
            if len(dictlistvalubatc[namebatc]['vari'][nameparavari]) == 1:
                listxdatvert = [-0.5 * 24. * dictoutp['duratrantotl'], 0.5 * 24. * dictoutp['duratrantotl']] 
                if 'duratranfull' in dictoutp:
                    listxdatvert += [-0.5 * 24. * dictoutp['duratranfull'], 0.5 * 24. * dictoutp['duratranfull']]
                listxdatvert = np.array(listxdatvert)
            else:
                listxdatvert = None

            # title for the plots
            dictstrgtitl = dict()
            for namevarbtotl in listnamevarbtotl:
                if namevarbtotl != nameparavari or dictlistvalubatc[namebatc]['vari'][nameparavari].size == 1:
                    dictstrgtitl[namevarbtotl] = dicttemp[namevarbtotl]
            strgtitl = retr_strgtitl(dictstrgtitl)
                
            
            lablyaxi = 'Relative flux - 1 [ppm]'
            
            strgextn = 'Simulated_%s_%s_%s' % (dicttdpy[typesyst], typetarg, namebatc)
            pathplot = ephesos.plot_lcur(pathvisu, \
                                         dictmodl=dictmodl, \
                                         typefileplot=typefileplot, \
                                         boolwritover=boolwritover, \
                                         listxdatvert=listxdatvert, \
                                         strgextn=strgextn, \
                                         lablxaxi=lablxaxi, \
                                         lablyaxi=lablyaxi, \
                                         strgtitl=strgtitl, \
                                         typesigncode='ephesos', \
                                        )
            
            if typesyst == 'psyspcur':
                # vertical zoom onto the phase curve
                strgextn = '%s_%s_%s_pcur' % (dicttdpy[typesyst], typetarg, namebatc)
                pathplot = ephesos.plot_lcur(pathvisu, \
                                             dictmodl=dictmodl, \
                                             typefileplot=typefileplot, \
                                             boolwritover=boolwritover, \
                                             listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtyaxi=[-500, None], \
                                             typesigncode='ephesos', \
                                            )
                
                # horizontal zoom around the primary
                strgextn = '%s_%s_%s_prim' % (dicttdpy[typesyst], typetarg, namebatc)
                #limtxaxi = np.array([-24. * 0.7 * dictoutp['duratrantotl'], 24. * 0.7 * dictoutp['duratrantotl']])
                limtxaxi = np.array([-2, 2.])
                pathplot = ephesos.plot_lcur(pathvisu, \
                                             dictmodl=dictmodl, \
                                             typefileplot=typefileplot, \
                                             boolwritover=boolwritover, \
                                             listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             typesigncode='ephesos', \
                                            )
                
                # horizontal zoom around the secondary
                strgextn = '%s_%s_%s_seco' % (dicttdpy[typesyst], typetarg, namebatc)
                limtxaxi += 0.5 * dicttemptemp['pericomp'] * 24.
                pathplot = ephesos.plot_lcur(pathvisu, \
                                             dictmodl=dictmodl, \
                                             typefileplot=typefileplot, \
                                             boolwritover=boolwritover, \
                                             listxdatvert=listxdatvert, \
                                             strgextn=strgextn, \
                                             lablxaxi=lablxaxi, \
                                             lablyaxi=lablyaxi, \
                                             strgtitl=strgtitl, \
                                             limtxaxi=limtxaxi, \
                                             limtyaxi=[-500, None], \
                                             typesigncode='ephesos', \
                                            )
                


            print('')
            print('')
            print('')
        


