import re
import nltk
import string
import glob        
import pickle
import mimetypes

from string import digits
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
import pickle
ps = PorterStemmer()

index = dict()
mapp = {}
fileid = 0

def preprocessing(read):
    remove_digits = str.maketrans('', '', digits)
    read = read.translate(remove_digits)
    
    read = read.replace("'","")
    
    res = re.findall(r'\w+', read) 
 
    words = list(map(lambda x:x.lower(), res))
    data = []
    for w in words: 
      if not w in stopwords.words():   
        data.append(ps.stem(w)) 
    
    
    data = " ".join(data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = word_tokenize(data)
    return data
'''
for path in glob.glob("/home/kaushal/Desktop/stories/*"):
    #print(path[30:])
    file = open(path, encoding='utf8', errors='ignore') 
    read = file.read() 
    file.seek(0)
    
    data = preprocessing(read)
    
    for p, term in enumerate(data):
  
                    # If token exists in the index.
                    if term in index:
                          
                          # Incrementing total term freq.
                          index[term][0] = index[term][0] + 1
                          # whether term existed in document fileid before.
                          if fileid in index[term][1]:
                              index[term][1][fileid].append(p)
                          else:
                              index[term][1][fileid] = [p]
  
                    #Token doesn't exist in index.
                    else:
                        #assigning empty list.
                        index[term] = []
                        #total term frequency is 1.
                        index[term].append(1)
                        #posting list is null .
                        index[term].append({})      
                        #append fileid in posting list.
                        index[term][1][fileid] = [p]
  
        # Map the file no. to the file name.
    mapp[fileid] = path[30:]
  
        # Increment the file no. counter for document ID mapping              
    fileid += 1
        
pickle.dump(index,open('INDEX.pkl','wb'))        
pickle.dump(mapp,open('MAP.pkl','wb'))
'''

index = pickle.load(open('/home/kaushal/Desktop/INDEX.pkl','rb'))
mapp = pickle.load(open('/home/kaushal/Desktop/MAP.pkl','rb'))

ip = input('Enter query\n')
ip = preprocessing(ip)

l = [[],[],[],[],[]]
for i in range(len(ip)):
    l[i] = list(index[ip[i]][1].keys())
    
common = l[0]    
for i in range(len(ip)):
    common = list(set(l[i]) & set(common))
common.sort()    

answer = []
for docid in common:
 for i in range(len(ip)):
    l[i] = list(index[ip[i]][1][docid])    
 for i in range(1,len(ip)):
    for j in range(len(l[i])):
        l[i][j] = l[i][j] - i

 temp = l[0]    
 for i in range(len(ip)):
    temp = list(set(l[i]) & set(temp))
 if temp != []:
     answer.append(docid)
     
print(len(answer))    
for name in answer:
    print(mapp[name])
