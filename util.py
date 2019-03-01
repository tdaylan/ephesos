from __init__ import *
import os
import astroquery
import astroquery.mast
#import astroquery.mast.Observations
import allesfitter
import sys, os
from allesfitter.priors.estimate_noise_wrap import estimate_noise_wrap
from astropy.io import fits
import fnmatch, numpy as np
from tdpy.util import summgene

def init_alle(tici, dictsett=None, dictpara=None, strgtarg=None, pathfold=None, pathtmpt=None):
    
    if pathfold is None:
        pathfold = os.environ['TESSTARG_DATA_PATH'] + '/'
    if pathtmpt is None:
        pathtmpt = os.environ['TESSTARG_DATA_PATH'] + '/tmpt/'
    if strgtarg is None:
        strgtarg = 'tici_%d' % tici

    pathtarg = pathfold + strgtarg + '/'
    # download data
    obsTable = astroquery.mast.Observations.query_criteria(target_name=tici, \
                                                           obs_collection='TESS', \
                                                           dataproduct_type='timeseries', \
                                           )
    listpath = []
    #print 'tici'
    #print tici
    for k in range(len(obsTable)):
        dataProducts = astroquery.mast.Observations.get_product_list(obsTable[k])
        want = (dataProducts['obs_collection'] == 'TESS') * (dataProducts['dataproduct_type'] == 'timeseries')
        for k in range(len(dataProducts['productFilename'])):
            if not dataProducts['productFilename'][k].endswith('_lc.fits'):
                want[k] = 0
        listpath.append(pathtarg + dataProducts[want]['productFilename'].data[0]) 
        manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathtarg)
        print 'manifest'
        print manifest
    
    if len(obsTable) == 0:
        #print 'No TESS light curve has been found.'
        return
    else:
        print 'Found TESS data'

    # construct target folder structure
    pathalle = pathtarg + 'allesfit/'
    cmnd = 'mkdir -p %s %s' % (pathtarg, pathalle)
    print cmnd
    os.system(cmnd)
    
    # make the changes to params.csv and settings.csv
    for a in range(2):
        if a == 0:
            strgfile = 'settings.csv'
        if a == 1:
            strgfile = 'params.csv'
        
        pathfile = pathalle + strgfile
        
        print 'Working on %s...' % strgfile
        
        if not os.path.exists(pathfile):
            print '%s does not exist at %s%s.csv.' % (strgfile, pathalle, strgfile)
            cmnd = 'cp %s%s %s' % (pathtmpt, strgfile, pathalle)
            print cmnd
            os.system(cmnd)
            if a == 0:
                dicttemp = dictsett
            if a == 1:
                dicttemp = dictpara
            booldoit = dicttemp is not None
        else:
            print 'Found previously existing %s%s...' % (pathalle, strgfile)
            if a == 0:
                booldoit = False 
            if a == 1:
                dicttemp = {}
                pathfilepost = pathalle + 'results/mcmc_table.csv'
                objtfilepost = open(pathfilepost, 'r')
                listlinepost = []
                for linepost in objtfilepost:
                    listlinepost.append(linepost)
                objtfilepost.close()
                print 'listlinepost'
                print listlinepost
                objtfile = open(pathfile, 'r')
                for line in objtfile:
                    linesplt = line.split(',')
                    #print 'linesplt[0]'
                    #print linesplt[0]
                    #print 'objtfilepost'
                    #print objtfilepost
                    for linepost in listlinepost:
                        linespltpost = linepost.split(',')
                        #print 'linespltpost[0]'
                        #print linespltpost[0]
                        if linespltpost[0] == linesplt[0]:
                            dicttemp[linespltpost[0]] = [linespltpost[1]] + linesplt[2:]
                    #print
                objtfile.close()
                print 'dicttemp'
                print dicttemp
                booldoit = True
        
        print 'booldoit'
        print booldoit
        #if a == 1:
        #    raise Exception('')
        if booldoit:
            objtfile = open(pathfile, 'r')
            listline = []
            for line in objtfile:
                listline.append(line)
            objtfile.close()
            os.system('rm %s' % pathfile)
            objtfile = open(pathfile, 'w')
            numbstrg = len(dicttemp)
            numbline = len(listline)
            numbfond = np.zeros((numbline, numbstrg))
            for m, line in enumerate(listline):
                linesplt = line.split(',')
                for k, strgtemp in enumerate(dicttemp):
                    if linesplt[0] == strgtemp:
                        numbfond[m, k] += 1
                indx = np.where(numbfond[m, :] > 0)[0]
                if indx.size > 1:
                    raise Exception('')
                if indx.size > 0:
                    strgtemp = dicttemp.keys()[np.asscalar(indx)]
                    lineneww = lineneww = ','.join([strgtemp] + dicttemp[strgtemp]) + '\n'
                else:
                    lineneww = line
                objtfile.write(lineneww)
                    
            for k, strgtemp in enumerate(dicttemp):
                if np.sum(numbfond[:, k], 0) > 1:
                    raise Exception('')
                if np.sum(numbfond[:, k], 0) == 0:
                    objtfile.write(','.join([strgtemp] + dicttemp[strgtemp]) + '\n')
            objtfile.close()
    
    # read the files to make the CSV file
    pathdown = pathtarg + 'mastDownload/TESS/'
    arry = read_tesskplr_fold(pathdown, pathalle)
    pathoutp = '%sTESS.csv' % pathalle
    np.savetxt(pathoutp, arry, delimiter=',')
    
    print 'pathalle'
    print pathalle
    allesfitter.show_initial_guess(pathalle)
    allesfitter.mcmc_fit(pathalle)
    allesfitter.mcmc_output(pathalle)


def call_alle():
    '''
    Convenience routine for calling allesfitter
    argument 1:
        plotnest -- calls allesfitter.ns_output()
        sampnest -- calls allesfitter.ns_fit()
        plotmcmc -- calls allesfitter.mcmc_output()
        sampmcmc -- calls allesfitter.mcmc_fit()
    argument 2: name of the planet
    argument 3: kind of the run
        global
        occultation
        eccentricity
    argument 4: type of the run
        wouttess
        withtess
        onlytess
        
    If only the first two arguments are provided, remakes all plots of the given planet
    '''

    if sys.argv[1] == 'plotnest':
        func = allesfitter.ns_output
    if sys.argv[1] == 'plotmcmc':
        func = allesfitter.mcmc_output
    if sys.argv[1] == 'sampprio':
        func = estimate_noise_wrap
    if sys.argv[1] == 'sampnest':
        func = allesfitter.ns_fit
    if sys.argv[1] == 'sampmcmc':
        func = allesfitter.mcmc_fit
    
    pathbase = os.environ['TESSTARG_DATA_PATH'] + '/'
    
    if sys.argv[1] == 'plot' and len(sys.argv) == 3:
        # make plotting automatic
        for strgkind in ['global', 'occulation', 'eccentricity']:
            for strgtype in ['wouttess', 'withtess', 'onlytes']:
                path = pathbase + sys.argv[2] + '/allesfit_' + strgkind + '/allesfit_' + strgtype
                if sys.argv[1].endswith('nest'):
                    path += '_ns/'
                else:
                    path += '_mc/'
                func(path)
    elif sys.argv[1] == 'viol':
        path = pathbase + sys.argv[2] + '/allesfit_' + sys.argv[3] + '/'
        allesfitter.postprocessing.plot_viol.proc_post(path)
    else:
        path = pathbase + sys.argv[2] + '/' + sys.argv[3] + '/'
        #path = pathbase + sys.argv[2] + '/allesfit_' + sys.argv[3] + '/allesfit_' + sys.argv[4]
        #if sys.argv[1].endswith('nest'):
        #    path += '_ns/'
        #else:
        #    path += '_mc/'
        func(path)


def read_tesskplr_fold(pathfold, pathwrit, typeinst='tess'):
    
    '''
    Reads all TESS or Kepler light curves in a folder and returns a data cube with time, flux and flux error
    '''

    print 'pathfold'
    print pathfold
    listpath = fnmatch.filter(os.listdir(pathfold), '%s*' % typeinst)
    print 'listpath'
    print listpath
    listarry = []
    for path in listpath:
        arry = read_tesskplr_file(pathfold + path + '/' + path + '_lc.fits', typeinst=typeinst)
        listarry.append(arry)
    
    # merge sectors
    arry = np.concatenate(listarry, axis=0)
    
    # sort in time
    indxsort = np.argsort(arry[:, 0])
    arry = arry[indxsort, :]
    
    # save
    pathoutp = '%s/%s.csv' % (pathwrit, typeinst)
    np.savetxt(pathoutp, arry, delimiter=',')

    return arry 


def read_tesskplr_file(path, typeinst='tess'):
    
    '''
    Reads a TESS or Kepler light curve file and returns a data cube with time, flux and flux error
    '''
    
    print 'path'
    print path
    listhdun = fits.open(path)
    time = listhdun[1].data['TIME'] + 2457000
    if typeinst == 'TESS':
        time += 2457000
    if typeinst == 'kplr':
        time += 2454833
    flux = listhdun[1].data['PDCSAP_FLUX']
    stdv = listhdun[1].data['PDCSAP_FLUX_ERR']
    indx = listhdun[1].data['QUALITY'] == 0
    
    # filtering
    time = time[indx]
    flux = flux[indx]
    stdv = stdv[indx]
    
    numbtime = time.size
    arry = np.empty((numbtime, 3))
    arry[:, 0] = time
    arry[:, 1] = flux
    arry[:, 2] = stdv
    arry = arry[~np.any(np.isnan(arry), axis=1)]
    arry[:, 2] /= np.mean(arry[:, 1])
    arry[:, 1] /= np.mean(arry[:, 1])
    
    return arry

