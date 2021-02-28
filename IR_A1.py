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
ps = PorterStemmer()

'''
INDEX = dict()
MAP = dict()
id = 0

for path in glob.glob("/home/kaushal/Desktop/stories/*"):
    MAP[id] = path[30:]
    print(path[30:])
    file = open(path, encoding='utf8', errors='ignore') 
    read = file.read() 
    file.seek(0)
    
    remove_digits = str.maketrans('', '', digits)
    read = read.translate(remove_digits)
    
    read = read.replace("'","")
    
    res = re.findall(r'\w+', read) 
    res = " ".join(res)
 
    read = []
    words = word_tokenize(res)    
    for w in words: 
       if not w in stopwords.words():  
        read.append(ps.stem(w)) 
    
    data = list(map(lambda x:x.lower(), read))
    
    data = " ".join(list(set(data)))
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = word_tokenize(data)
    for word in data:
        if word in INDEX.keys():
            INDEX[word].append(id)
        else:
            INDEX[word] = []
            INDEX[word].append(id)

    id = id + 1
        
pickle.dump(INDEX,open('INDEX.pkl','wb'))        
pickle.dump(MAP,open('MAP.pkl','wb'))
'''


INDEX = pickle.load(open('INDEX.pkl','rb'))
MAP= pickle.load(open('MAP.pkl','rb'))
file_id = [i for i in range(len(MAP))]


def min_no_of_comparisons(l1,l2):
         l1, l2 = list(l1), list(l2)
         i = j = 0
         l1.sort()
         l2.sort()
         c = 0
         while(i<len(l1) and j<len(l2)):
              if l1[i] == l2[j]:
                i+=1
                j+=1
                c+=1
              elif l1[i] < l2[j]:
                i+=1
                c+=1
              else:
                j+=1  
         return c


N = int(input('Enter no of queries\n'))

while(N!=0):
    print("Query {}".format(N))
    N-=1
    ip1 = input('Sentence : ')
    
    remove_digits = str.maketrans('', '', digits)
    read = ip1.translate(remove_digits)
        
    read = read.replace("'","")
        
    res = re.findall(r'\w+', read) 
    res = " ".join(res)
     
    read = []
    words = word_tokenize(res)    
    for w in words: 
       if not w in stopwords.words():  
        read.append(ps.stem(w)) 
        
    data = list(map(lambda x:x.lower(), read))
        
    data = " ".join(data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    ip1 = word_tokenize(data)
    
    if len(ip1) > 1:
        print("Please enter {} operations".format(len(ip1)-1))
        ip2 = input('Operation : ')
        
        data = "".join(list(map(lambda x:x.lower(), ip2)))
        ip2 = data.split(',')
        
        if len(ip1) == (len(ip2)+1):
            i=1
            while(i<=len(ip2)):
                ip1.insert(2*i-1,ip2[i-1])
                i+=1
            
            counter = 0
            while(1):
                if 'and' in ip1:
                    index = ip1.index('and')
                    left, right = index-1, index+1 
                    key1, key2 = ip1[left], ip1[right]
                    ip1.pop(left)
                    ip1.pop(left)
                    
                    if type(key1) != type([]):
                       if key1 in INDEX.keys():
                           key1 = INDEX[key1]
                       else:
                           key1 = []
                           
                    if type(key2) != type([]):
                       if key2 in INDEX.keys():
                           key2 = INDEX[key2]
                       else:
                           key2 = []  
                         
                    result = list(set(key1) & set(key2))
                    ip1[left] = result
                    counter+=min_no_of_comparisons(set(key1),set(key2))
            
                    
                elif 'and not' in ip1:
                    index = ip1.index('and not')
                    left, right = index-1, index+1 
                    key1, key2 = ip1[left], ip1[right]
                    ip1.pop(left)
                    ip1.pop(left)
                    
                    if type(key1) != type([]):
                       if key1 in INDEX.keys():
                           key1 = INDEX[key1]
                       else:
                           key1 = []
                           
                    if type(key2) != type([]):
                       if key2 in INDEX.keys():
                           key2 = INDEX[key2]
                       else:
                           key2 = [] 
                           
                    key2 = set(file_id).symmetric_difference(set(key2)) 
                    result = list(set(key1) & set(key2))
                    ip1[left] = result 
                    counter+=min_no_of_comparisons(set(key1),set(key2))
                    
                           
                    
                elif 'or not' in ip1:
                    index = ip1.index('or not')
                    left, right = index-1, index+1 
                    key1, key2 = ip1[left], ip1[right]
                    ip1.pop(left)
                    ip1.pop(left)
                    
                    if type(key1) != type([]):
                       if key1 in INDEX.keys():
                           key1 = INDEX[key1]
                       else:
                           key1 = []
                           
                    if type(key2) != type([]):
                       if key2 in INDEX.keys():
                           key2 = INDEX[key2]
                       else:
                           key2 = [] 
                    
                    key2 = list(set(file_id).symmetric_difference(set(key2))) 
                    result = list(set(key1) | set(key2))
                    ip1[left] = result                              
                    
                elif 'or' in ip1:
                    index = ip1.index('or')
                    left, right = index-1, index+1 
                    key1, key2 = ip1[left], ip1[right]
                    ip1.pop(left)
                    ip1.pop(left)
                    
                    if type(key1) != type([]):
                       if key1 in INDEX.keys():
                           key1 = INDEX[key1]
                       else:
                           key1 = []
                           
                    if type(key2) != type([]):
                       if key2 in INDEX.keys():
                           key2 = INDEX[key2]
                       else:
                           key2 = []  
                     
                    result = list(set(key1) | set(key2))
                    ip1[left] = result                
                    
                else:
                    break
                
            print("No of documents : {}".format(len(result)))
            print("Documents list :")
            for id in result:
                print(MAP[id])
            print("Minimum no of comparisons required are {}".format(counter))    
                
        
        else:
            print('Invalid number of operations')
    else:
      if len(ip1) == 1:  
        print("No operations are performed cause only single word remains after preprocessing they query.")  
        if ip1[0] in INDEX.keys():        
           print("No of documents : {}".format(len(INDEX[ip1[0]])))
           print("Documents list :")
           for id in INDEX[ip1[0]]:
                print(MAP[id])
           print("Minimum no of comparisons required are 0.")    
                
        else:
           print("Word is not present in our inverted index data structure.") 
           print("No of documents : 0") 
      else:
        print("No operations are performed cause no word remains after preprocessing they query.")   
            
         
