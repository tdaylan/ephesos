import numpy as np

import ephesus
from tdpy.util import summgene

'''
Using the transit model
'''

# time axis
time = np.linspace(0., 30., 10000)

# orbital periods
peri = np.array([3., 5.])
# mid-transit epochs
epoc = np.array([200., 600.])
# radii of the planets [R_E]
radiplan = np.array([1.2, 3.4])
# radius of the star [R_S]
radistar = 1.0
# total mass [M_S]
masstotl = 1.0
# semi-major axes  [AU] 
smax = ephesus.retr_smaxkepl(peri, masstotl)
# sum of radii divided by the semi-major axis
rsma = ephesus.retr_rsma(radiplan, radistar, smax)
# cosine of inclination
cosi = np.array([0.03, 0.05])
rfxl = ephesus.retr_rflxtranmodl(time, peri, epoc, radiplan, radistar, rsma, cosi)


print('rfxl')
summgene(rfxl)

