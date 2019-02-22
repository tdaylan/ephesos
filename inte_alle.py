import allesfitter
import sys, os
from allesfitter.priors.estimate_noise_wrap import estimate_noise_wrap

def summgene(arry):
    try:
        print np.amin(arry)
        print np.amax(arry)
        print np.mean(arry)
        print arry.shape
    except:
        print arry

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
if sys.argv[1] != 'maketessdata':

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

else: 
    from astropy.io import fits
    import fnmatch, numpy as np

    pathdata = os.environ['TESSTARG_DATA_PATH'] + '/%s/' % sys.argv[2]
    pathdataorig = pathdata + 'data_preparation/original_data/'
    
    listpath = fnmatch.filter(os.listdir(pathdataorig), 'tess*')
    
    listarry = []
    for path in listpath:
        listhdun = fits.open(pathdataorig + path)
        #time = listhdun[1].data['TIME'] + 2454833
        time = listhdun[1].data['TIME'] + 2457000
        flux = listhdun[1].data['PDCSAP_FLUX']
        stdv = listhdun[1].data['PDCSAP_FLUX_ERR']
        indx = listhdun[1].data['QUALITY'] == 0
        
        print 'path'
        print path
        
        print 'time'
        summgene(time)
        
        print 'Good indices:'
        summgene(indx)
        
        # filtering
        time = time[indx]
        flux = flux[indx]
        stdv = stdv[indx]
        
        print 'time'
        summgene(time)
        print

        numbtime = time.size
        arry = np.empty((numbtime, 3))
        arry[:, 0] = time
        arry[:, 1] = flux
        arry[:, 2] = stdv
        arry = arry[~np.any(np.isnan(arry), axis=1)]
        arry[:, 2] /= np.mean(arry[:, 1])
        arry[:, 1] /= np.mean(arry[:, 1])
        
        listarry.append(arry)
    # merge sectors
    arry = np.concatenate(listarry, axis=0)
    
    # sort in time
    indxsort = np.argsort(arry[:, 0])
    arry = arry[indxsort, :]
    
    # save
    pathoutp = pathdata + '%s/TESS.csv' % sys.argv[3]
    np.savetxt(pathoutp, arry, delimiter=',')

