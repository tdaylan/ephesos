from __init__ import *

def init_alle(tici, strgfold):
    pathdata = os.environ['WORK'] + '/data/tesstarg/'
    pathtarg = pathdata + sys.argv[2] + '/'
    pathtmpt = pathdata + 'tmpt/'
    
    # construct target folder structure
    pathalle = pathtarg + 'allesfit/'
    cmnd = 'mkdir -p %s %s' % (pathtarg, pathalle)
    print cmnd
    os.system(cmnd)
    
    # copy template params.csv and settings.csv to the target folder
    pathsett = pathalle + 'settings.csv'
    if not os.path.exists(pathsett):
        cmnd = 'cp %ssettings.csv %s' % (pathtmpt, pathalle)
        print cmnd
        os.system(cmnd)
    
    pathpara = pathalle + 'params.csv'
    if not os.path.exists(pathpara):
        cmnd = 'cp %sparams.csv %s' % (pathtmpt, pathalle)
        print cmnd
        os.system(cmnd)
    
    # download data
    obsTable = astroquery.mast.Observations.query_criteria(target_name=tici, \
                                                           obs_collection='TESS', \
                                                           dataproduct_type='timeseries', \
                                           )
    listpath = []
    for k in range(len(obsTable)):
        dataProducts = astroquery.mast.Observations.get_product_list(obsTable[k])
        want = (dataProducts['obs_collection'] == 'TESS') * (dataProducts['dataproduct_type'] == 'timeseries')
        for k in range(len(dataProducts['productFilename'])):
            if not dataProducts['productFilename'][k].endswith('_lc.fits'):
                want[k] = 0
        listpath.append(pathtarg + dataProducts[want]['productFilename'].data[0]) 
        manifest = astroquery.mast.Observations.download_products(dataProducts[want], download_dir=pathtarg)
    
    # read the files to make the CSV file
    arry = read_tess(pathtarg + 'mastDownload/TESS/')
    pathoutp = '%sTESS.csv' % pathalle
    np.savetxt(pathoutp, arry, delimiter=',')
    
    allesfitter.show_initial_guess(pathalle)
    allesfitter.mcmc_fit(pathalle)


