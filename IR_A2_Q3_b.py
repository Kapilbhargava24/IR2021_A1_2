import pandas as pd
import numpy as np
import math
from itertools import groupby


file = open('/home/kaushal/Desktop/IR_A2_MD.txt', encoding='utf8', errors='ignore') 
read = file.read() 
file.seek(0)
data = read.split()
relevance = []
idx = 0
for i in range(103):
    relevance.append(int(data[idx*138]))
    idx+=1

fileid = [i for i in range(103)]
rel_idx = zip(relevance,fileid)
rel = []
for t in rel_idx:
    rel.append(t)
rel_du = rel[:]    
rel.sort(key=lambda x:x[0],reverse=True)

dcg = rel[0][0]
del rel[0]
for i, f in enumerate(rel):
   dcg = dcg + f[0]/math.log((i+2),2)
print('Max DCG is {}\n'.format(dcg))
print("Possible order of query-url pair with MAX DCG")

permut = []
rel = rel_du[:]
rel.sort(key=lambda x:x[0],reverse=True)
for i in rel:
  print(i[1]+1)
  permut.append(i[0])
    
poss_perm = [len(list(group)) for key, group in groupby(permut)]  
ansr = 1
for i in range(len(poss_perm)):
    ansr=ansr*math.factorial(poss_perm[i])
print('\n\nTotal no of files possible with max DCG are {}'.format(ansr))





         