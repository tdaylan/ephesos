from util import *
from tdpy.util import summgene
import numpy as np

np.random.seed(0)

factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = retr_factconv()
    
numbtime = 1000
epoc = 2457001.3
time = np.linspace(0., 10., numbtime) + 2457000
#      retr_rflxtranmodl(time, peri, epoc, radiplan, radistar, rsma, cosi, ecce=0., sinw=0.):

rsma = 0.3
cosi = 0
peri = 2.
radiplan = 1.
radistar = 1.
rrat = radiplan / radistar / factrsre
dept = rrat**2
dura = retr_dura(peri, rsma, cosi)
rs2a = rsma - rsma * radiplan / (radiplan + radistar * factrsre)
durafull = retr_durafull(peri, rs2a, 1., rrat, 0)
print('peri')
print(peri)
print('epoc')
print(epoc)
print('dura')
print(dura)
print('durafull')
print(durafull)
print('dept')
print(dept)
rflx = retr_rflxtranmodl(time, [peri], [epoc], [1.], 1., [rsma], [cosi])
stdv = 1e-6
rflx += np.random.randn(numbtime) * stdv
#print('rflx')
#for rflxtemp in rflx:
#    print(rflxtemp)

arrytser = np.zeros((numbtime, 3))
arrytser[:, 0] = time
arrytser[:, 1] = rflx
arrytser[:, 2] = stdv
find_tran(arrytser)
pathimag = os.environ['EPHESUS_DATA_PATH'] + '/imag/test/'
os.system('mkdir -p %s' % pathimag)
exec_blsq(arrytser, pathimag)
#, numbplan=None, maxmnumbplantlsq=None, strgextn='', thrssdeetlsq=7.1, boolpuls=False, \
#                                 ticitarg=None, dicttlsqinpt=None, booltlsq=False, \
#                                 strgplotextn='pdf', figrsize=(4., 3.), figrsizeydobskin=(8, 2.5), alphraww=0.2, \
#                                 ):

