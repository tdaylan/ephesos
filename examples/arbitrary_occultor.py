            
# arbitrary occultors
if gdat.numbcomp != 1:
    raise Exception('')
path = os.environ['EPHESOS_DATA_PATH'] + '/data/LightCurve/turkey.csv'
print('Reading from %s...' % path)
gdat.positurk = np.loadtxt(path, delimiter=',')

print('Scaling and centering the template coordinates...')
for a in range(2):
    gdat.positurk[:, a] -= np.amin(gdat.positurk[:, a])
# half of the diagonal of the rectangle
halfdiag = 0.5 * np.sqrt((np.amax(gdat.positurk[:, 0]))**2 + (np.amax(gdat.positurk[:, 1]))**2)
# normalize
gdat.positurk *= gdat.rratcomp[0] / halfdiag
# center
gdat.positurk -= 0.5 * np.amax(gdat.positurk, 0)[None, :]

diffturk = 1. * gdat.diffgrid
diffturksmth = 2. * diffturk
gdat.xposturk = np.arange(-5 * diffturk + np.amin(gdat.positurk[:, 0]), np.amax(gdat.positurk[:, 0]) + 5. * diffturk, diffturk)
gdat.yposturk = np.arange(-5 * diffturk + np.amin(gdat.positurk[:, 1]), np.amax(gdat.positurk[:, 1]) + 5. * diffturk, diffturk)
gdat.maxmxposturkmesh = np.amax(gdat.xposturk)
gdat.maxmyposturkmesh = np.amax(gdat.yposturk)

gdat.xposturkmesh, gdat.yposturkmesh = np.meshgrid(gdat.xposturk, gdat.yposturk)
gdat.xposturkmeshflat = gdat.xposturkmesh.flatten()
gdat.yposturkmeshflat = gdat.yposturkmesh.flatten()
gdat.positurkmesh = np.vstack([gdat.xposturkmeshflat, gdat.yposturkmeshflat]).T

gdat.valuturkmesh = np.exp(-((gdat.positurk[:, 0, None] - gdat.xposturkmeshflat[None, :]) / diffturksmth)**2 \
                          - ((gdat.positurk[:, 1, None] - gdat.yposturkmeshflat[None, :]) / diffturksmth)**2)
gdat.valuturkmesh = np.sum(gdat.valuturkmesh, 0)

if not np.isfinite(gdat.valuturkmesh).all():
    print('gdat.xposturkmeshflat')
    summgene(gdat.xposturkmeshflat)
    print('gdat.yposturkmeshflat')
    summgene(gdat.yposturkmeshflat)
    print('')
    raise Exception('')

