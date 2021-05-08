from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import nltk
import string
import glob        
import pickle
from string import digits
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

model = KeyedVectors.load_word2vec_format('/home/kaushal/Desktop/GoogleNews-vectors-negative300.bin.gz',binary=True,limit=500000)    

def preprocessing(read,flag):
    remove_digits = str.maketrans('', '', digits)
    read = read.translate(remove_digits)
    
    read = read.replace("'","")
    
    res = re.findall(r'\w+', read) 
 
    words = list(map(lambda x:x.lower(), res))
    if flag:    
        data = []
        for w in words: 
          if not w in stop_words:   
            data.append(ps.stem(w)) 
    else:   
        data = words         
    
    data = " ".join(data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = word_tokenize(data)
    return data

doc_emb = []
for path in glob.glob("/home/kaushal/Desktop/stories/*"):
    file = open(path, encoding='utf8', errors='ignore') 
    read = file.read() 
    file.seek(0) 
    data = preprocessing(read,1)
    emb = []
    for tok in data:
        if tok in model:
            emb.append(model.get_vector(tok))
        else:
            emb.append(np.random.rand(300))
    doc_emb.append(np.mean(emb,axis=0)) 

    
file = open("/home/kaushal/Desktop/QUERY", encoding='utf8', errors='ignore')
file.seek(0) 
query = file.read() 
query = query.split('\n')
query = query[:-1]
query_emb = []
for q in query:
    data = preprocessing(q,1)
    emb = []
    for tok in data:
        if tok in model:
            emb.append(model.get_vector(tok))
        else:
            emb.append(np.random.rand(300))
    query_emb.append(np.mean(emb,axis=0)) 


file = open("/home/kaushal/Desktop/DOC_ID", encoding='utf8', errors='ignore')
file.seek(0) 
doc_id = file.read() 
doc_id= doc_id.split('\n')
doc_id = doc_id[:-1]
for i in range(len(doc_id)):
  doc_id[i] = list(map(int, doc_id[i].split()))
did = [i for i in range(472)]

mean_avg_prec = 0
idx = 0
for query_vector in query_emb:
    
    similarity_measure = []
    for d in doc_emb:
        similarity_measure.append(cosine_similarity([d],[query_vector]))
    
    rel_idx = zip(similarity_measure,did)
    rel = []
    for t in rel_idx:
        rel.append(t)
    rel.sort(key=lambda x:x[0],reverse=True)

    avg = 0
    relevant = 0
    
    for j in range(len(doc_id[0])):
        if rel[j][1] == doc_id[idx][j]:
            relevant+=1
            avg+=relevant/(j+1)
    if relevant:         
      avg = avg/(0.056*relevant)    
    mean_avg_prec+=avg
    idx+=1

print(mean_avg_prec/100)
    
    
    
    


