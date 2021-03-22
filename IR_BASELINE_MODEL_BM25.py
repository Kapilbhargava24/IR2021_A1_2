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


WCD = {}
for text in cleaned_corpus:
    for token in text:
        count = WCD.get(token, 0) + 1
        WCD[token] = count

texts = [[token for token in text if WCD[token] > 1] for text in cleaned_corpus]


class Best_Match_25:
    
    def __init__(self, k=1.5, b=0.75):
        self.b = b
        self.k = k

    def fit(self, corpus):
       
        tf = []
        df = {}
        idf = {}
        doc_length = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_length.append(len(document))

            frequency = {}
            for term in document:
                term_count = frequency.get(term, 0) + 1
                frequency[term] = term_count

            tf.append(frequency)

            for term, _ in frequency.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            var = (corpus_size-freq+0.5)/(freq+0.5)
            idf[term] = math.log(1+var)

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_length_ = doc_length
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_length_ = sum(doc_length)/corpus_size
        return self

    def RESULT(self, query):
        scores = [self.RELEVANCE_SCORE(query, index) for index in range(self.corpus_size_)]
        return scores

    def RELEVANCE_SCORE(self, query, index):
        score = 0.0
        doc_length = self.doc_length_[index]
        frequency = self.tf_[index]
        for term in query:
            if term not in frequency:
                continue

            term_fre = frequency[term]
            num = self.idf_[term]*term_fre*(self.k + 1)
            denom = term_fre+ self.k*(1-self.b+self.b*doc_length/self.avg_doc_length_)
            score+=(num/denom)
        return score
    
model = Best_Match_25()
model.fit(texts)

for query in queries['query']
  
  scores = model.RESULT(query)
  indices = scores.argsort()[:-10:-1]
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



    
