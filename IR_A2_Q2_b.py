import re
import nltk
import string
import glob        
import pickle
import mimetypes
import math
import numpy as np

from string import digits
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
import pickle
ps = PorterStemmer()

sw = stopwords.words('english')
pos_index = dict()
file_map = {}
fileno = 0

def preprocessing(read):
    remove_digits = str.maketrans('', '', digits)
    read = read.translate(remove_digits)
    
    read = read.replace("'","")
    
    res = re.findall(r'\w+', read) 
 
    words = list(map(lambda x:x.lower(), res))
    data = []
    for w in words: 
      if not w in sw:   
        data.append(ps.stem(w)) 
    
    
    data = " ".join(data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = word_tokenize(data)
    return data

index = pickle.load(open('/home/kaushal/Desktop/INDEX.pkl','rb'))
mapp = pickle.load(open('/home/kaushal/Desktop/MAP.pkl','rb'))

'''
v= list(index.keys())
vocab = []
for w in v:
    if not w in sw:  
        vocab.append(w)
'''

vocab = pickle.load(open('/home/kaushal/Desktop/vocab.pkl','rb'))
S1, S2, S3, S4, S5 = [],[],[],[],[]  
idf = []
for w in vocab:
    df = len(index[w][1].keys())
    idf.append(math.log(472/(df+1),10))

#S1
for i in range(472):
    temp = []
    for pos,w in enumerate(vocab):
        docid = index[w][1].keys()
        if i in docid:
            temp.append(idf[pos])
        else:
            temp.append(0)
    S1.append(temp)      
            
#S2
for i in range(472):
    temp = []
    for pos,w in enumerate(vocab):
        docid = index[w][1].keys()
        j = 0
        if i in docid:
            freq = len(index[w][1][i])     
            temp.append(idf[pos]*freq)
        else:
            temp.append(0)
    S2.append(temp) 
      
'''    
#S3
for i in range(472):
    temp = []
    for pos,w in enumerate(vocab):
        docid = index[w][1].keys()
        j = 0
        if i in docid:
            freq = len(index[w][1][i])
            for path in glob.glob("/home/kaushal/Desktop/stories/*"):
              if i==j:
                  break
              else:
                  j+=1
            file = open(path, encoding='utf8', errors='ignore') 
            read = file.read() 
            file.seek(0)    
            data = preprocessing(read)
            freq = freq/len(data) 
            temp.append(idf[pos]*freq)
        else:
            temp.append(0)
    S3.append(temp)     
pickle.dump(S3,(open('/home/kaushal/Desktop/S3.pkl','wb')))
'''

S3 = pickle.load(open('/home/kaushal/Desktop/S3.pkl','rb'))
    
#S4
for i in range(472):
    temp = []
    for pos,w in enumerate(vocab):
        docid = index[w][1].keys()
        j = 0
        if i in docid:
            freq = len(index[w][1][i])     
            temp.append(idf[pos]*math.log10(1+freq))
        else:
            temp.append(0)
    S4.append(temp)

'''
#S5
for i in range(472):
    temp = []
    for pos,w in enumerate(vocab):
        docid = index[w][1].keys()
        j = 0
        if i in docid:
            freq = len(index[w][1][i])
            for path in glob.glob("/home/kaushal/Desktop/stories/*"):
              if i==j:
                  break
              else:
                  j+=1
            file = open(path, encoding='utf8', errors='ignore') 
            read = file.read() 
            file.seek(0)    
            data = preprocessing(read)
            freq2 = []
            for wo in data: 
              if wo in index.keys():
               if i in index[wo][1].keys():   
                freq2.append(len(index[wo][1][i]))
            freq2 =  max(freq2)     
            temp.append(idf[pos]*(0.5+0.5*freq/freq2))
        else:
            temp.append(idf[pos]*0.5)
    S5.append(temp)              
pickle.dump(S5,(open('/home/kaushal/Desktop/S5.pkl','wb')))
'''    
S5 = pickle.load(open('/home/kaushal/Desktop/S5.pkl','rb'))


query = input("Please enter query\n")
query = preprocessing(query)

#S1
tfidfscore, fid = [], [i for i in range(472)]
for i in range(472):
    score = 0
    for term in query:
        if term in index.keys():
            idx = vocab.index(term)
            score+=S1[i][idx]
    tfidfscore.append(score)
        
r = zip(fid,tfidfscore)
res = []
for i in r:
    res.append(i)
res.sort(key=lambda x:x[1],reverse=True)
print("Using Scheme 1")
for idx in res[:5]:
    print(mapp[idx[0]])
    
    
#S2
tfidfscore, fid = [], [i for i in range(472)]
for i in range(472):
    score = 0
    for term in query:
        if term in index.keys():
            idx = vocab.index(term)
            score+=S2[i][idx]
    tfidfscore.append(score)
        
r = zip(fid,tfidfscore)
res = []
for i in r:
    res.append(i)
res.sort(key=lambda x:x[1],reverse=True)
print("\n\nUsing Scheme 2")
for idx in res[:5]:
    print(mapp[idx[0]])
        
#S3
tfidfscore, fid = [], [i for i in range(472)]
for i in range(472):
    score = 0
    for term in query:
        if term in index.keys():
            idx = vocab.index(term)
            score+=S3[i][idx]
    tfidfscore.append(score)
        
r = zip(fid,tfidfscore)
res = []
for i in r:
    res.append(i)
res.sort(key=lambda x:x[1],reverse=True)
print("\n\nUsing Scheme 3")
for idx in res[:5]:
    print(mapp[idx[0]])
        
#S4
tfidfscore, fid = [], [i for i in range(472)]
for i in range(472):
    score = 0
    for term in query:
        if term in index.keys():
            idx = vocab.index(term)
            score+=S4[i][idx]
    tfidfscore.append(score)
        
r = zip(fid,tfidfscore)
res = []
for i in r:
    res.append(i)
res.sort(key=lambda x:x[1],reverse=True)
print("\n\nUsing Scheme 4")
for idx in res[:5]:
    print(mapp[idx[0]])
        
    
#S5
tfidfscore, fid = [], [i for i in range(472)]
for i in range(472):
    score = 0
    for term in query:
        if term in index.keys():
            idx = vocab.index(term)
            score+=S5[i][idx]
    tfidfscore.append(score)
        
r = zip(fid,tfidfscore)
res = []
for i in r:
    res.append(i)
res.sort(key=lambda x:x[1],reverse=True)
print("\n\nUsing Scheme 5")
for idx in res[:5]:
    print(mapp[idx[0]])
    

