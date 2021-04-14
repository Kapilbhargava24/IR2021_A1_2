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

mdcg = rel[0][0]
del rel[0]
for i, f in enumerate(rel):
   mdcg = mdcg + f[0]/math.log((i+2),2)

rel = rel_du[:50]
dcg = rel[0][0]
del rel[0]
for i, f in enumerate(rel):
   dcg = dcg + f[0]/math.log((i+2),2)
ndcg = dcg/mdcg   
print("Normalized DCG for first 50 docs is {}\n".format(ndcg))   

rel = rel_du[:]
dcg = rel[0][0]
del rel[0]
for i, f in enumerate(rel):
   dcg = dcg + f[0]/math.log((i+2),2)
ndcg = dcg/mdcg   
print("Normalized DCG for whole dataset is {}".format(ndcg)) 




         