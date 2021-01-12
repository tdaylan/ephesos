import os
import tesstarg.util
import allesfitter
pathalle = os.getcwd() + '/'

alles = allesfitter.allesclass(pathalle)
allesfitter.config.init(pathalle)

pathfilepara = pathalle + 'params.csv'
objtfilepara = open(pathfilepara, 'r')

# read params.csv
listlinepara = []
for linepara in objtfilepara:
    listlinepara.append(linepara)
objtfilepara.close()

numbsamp = alles.posterior_params[list(alles.posterior_params.keys())[0]].size
#liststrg = list(alles.posterior_params.keys())
#for k, strg in enumerate(liststrg):
    #post = alles.posterior_params[strg]

listlineneww = []
for k, line in enumerate(listlinepara):
    print('line')
    print(line)
    linesplt = line.split(',')
    for strg in alles.posterior_params_at_maximum_likelihood:
        #print('strg')
        #print(strg)
        #print('alles.posterior_params_at_maximum_likelihood[strg]')
        #print(alles.posterior_params_at_maximum_likelihood[strg])
        if linesplt[0] == strg:
            print('linesplt')
            print(linesplt)
            linesplt[1] = '%s' % alles.posterior_params_at_maximum_likelihood[strg][0]
    listlineneww.append(','.join(linesplt))
    print('listlineneww[k]')
    print(listlineneww[k])
    print

# rewrite
pathfilepara = pathalle + 'params.csv'
objtfilepara = open(pathfilepara, 'w')
for lineparaneww in listlineneww:
    objtfilepara.write("%s" % lineparaneww)
objtfilepara.close()

