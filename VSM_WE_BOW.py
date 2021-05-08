from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import nltk
import string
import glob        
import pickle
from string import digits
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
import pickle
ps = PorterStemmer()

def preprocessing(read):
    remove_digits = str.maketrans('', '', digits)
    read = read.translate(remove_digits)
    
    read = read.replace("'","")
    
    res = re.findall(r'\w+', read) 
 
    words = list(map(lambda x:x.lower(), res))
    data = []
    for w in words: 
      if not w in stop_words:   
        data.append(ps.stem(w)) 
    
    
    data = " ".join(data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = word_tokenize(data)
    return data


cleaned_corpus = []
for path in glob.glob("/home/kaushal/Desktop/stories/*"):
    file = open(path, encoding='utf8', errors='ignore') 
    read = file.read() 
    file.seek(0) 
    data = preprocessing(read)
    data = ' '.join(data)
    cleaned_corpus.append(data)

vectorizer =  CountVectorizer()
vectorizer.fit(cleaned_corpus) 
corpus_vector = vectorizer.transform(cleaned_corpus)

 
file = open("/home/kaushal/Desktop/QUERY", encoding='utf8', errors='ignore')
file.seek(0) 
query = file.read() 
query = query.split('\n')
query = query[:-1]

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
for q in query:
    q = preprocessing(q)
    q = ' '.join(q)
    query_vector = vectorizer.transform([q])
    similarity_measure = cosine_similarity(corpus_vector,query_vector).flatten()

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
      avg = avg/relevant    
    mean_avg_prec+=avg
    idx+=1

print(mean_avg_prec/100)
    
    
    
    

