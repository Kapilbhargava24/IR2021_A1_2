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


print('Jaccard Coefficient')    
ip = input("Enter query\n")
ip = preprocessing(ip) 
ip = set(ip)    
mapp = pickle.load(open('/home/kaushal/Desktop/MAP.pkl','rb'))
coeff, fid = [], [i for i in range(472)] 
for path in glob.glob("/home/kaushal/Desktop/stories/*"):
    file = open(path, encoding='utf8', errors='ignore') 
    read = file.read() 
    file.seek(0) 
    data = preprocessing(read)
    
    data = set(data)
    size_s1 = len(ip); 
    size_s2 = len(data); 
   
    intersect = ip & data;  
    size_in = len(intersect); 
    jaccard_in = size_in/(size_s1+size_s2-size_in);
    coeff.append(jaccard_in)

fid_coeff = []
for c in zip(fid,coeff):
    fid_coeff.append(c)

fid_coeff.sort(key=lambda x:x[1],reverse=True)

for idx in fid_coeff[:5]:
    print(mapp[idx[0]])


    