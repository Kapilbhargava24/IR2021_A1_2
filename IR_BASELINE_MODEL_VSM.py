from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import dask.dataframe as dd
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


queries=pd.read_table('msmarco-doctrain-queries.tsv',header=None)
queries.columns=['qid','query']
queries=queries.sample(n=2000,random_state=42).reset_index(drop=True)

top_100_doc=pd.read_table('msmarco-doctrain-top100',delimiter=' ',header=None)
top_100_doc.columns=['qid','Q0','docid','rank','score','runstring']
top_100_doc=top_100_doc[top_100_doc['qid'].isin(queries['qid'].unique())].reset_index(drop=True)

relevant=list(range(1,11))
nonrelevant=list(range(91,101))
top_100_doc['relevant']=top_100_doc['rank'].apply(lambda x: 1 if x in relevant else ( 0 if x in nonrelevant else np.nan))
top_100_doc=top_100_doc.dropna()
top_100_doc['relevant']=top_100_doc['relevant'].astype(int)

data=dd.read_table('msmarco-docs.tsv',blocksize=100e6,header=None)
data.columns=['docid','url','title','body']

def CORPUS(result):
  unique_docid=result['docid'].unique()
  condition=data['docid'].isin(unique_docid)
  corpus=data[condition].reset_index(drop=True)
  corpus=corpus.drop(columns='url')
  return corpus

corpus=CORPUS(top_100_doc)

corpus['cleaned']=corpus['body'].apply(lambda x:x.lower())

def CLEANING(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text
 

corpus['cleaned']=corpus['cleaned'].apply(lambda x: CLEANING(x))
corpus['cleaned']=corpus['cleaned'].apply(lambda x: re.sub(' +',' ',x))


def GET_TOKENS(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens

def STEMMER(token_list):
    ps = nltk.stem.PorterStemmer()
    ste = []
    for words in token_list:
        ste.append(ps.stem(words))
    return ste

def STOPWORDS_REMOVAL(doc_text):
    text = []
    for words in doc_text:
        if words not in stop_words:
            text.append(words)
        return text


cleaned_corpus = []
for doc in corpus:
   tokens = GET_TOKENS(doc)
   doc_text = STOPWORDS_REMOVAL(tokens)
   doc_text = ' '.join(doc_text)
   cleaned_corpus.append(doc_text)

vectorizer =  TfidfVectorizer()
vectorizer.fit(cleaned_corpus) 
corpus_vector = vectorizer.transform(cleaned_corpus)

q = []
idx = 0
final_prec = 0

for query in queries['query']
  query = GET_TOKENS(query)
  query = STOPWORDS_REMOVAL(query)
  query = ' '.join(query)

  query_vector = vectorizer.transform([q])
  similarity_measure = cosine_similarity(corpus_vector,query_vector).flatten()
  indices = similarity_measure.argsort()[:-10:-1]
  indices = indices[:20]
  
  result = top_100_doc[idx:idx+20]
  idx+=20
  c = 0 
  prec = 0
  for i in indices:
      if i == result[c]
         prec++
      c+=1

  prec = prec/20
  final_prec+=prec

final_prec = final_prec/len(queries)
print("Mean Average Precision is {}".format(final_prec))   


