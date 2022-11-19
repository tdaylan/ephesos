import numpy as np

import ephesus
from tdpy.util import summgene

epoc = 0.5
peri = 3.
time = np.arange(0., 10., 1. / 3600.)
listepoc = ephesus.retr_listepoctran(time, epoc, peri)
print('listepoc')
print(listepoc)
