import matplotlib.pyplot as plt

file = open('/home/kaushal/Desktop/IR_A2_MD.txt', encoding='utf8', errors='ignore') 
read = file.read() 
file.seek(0)
data = read.split()
relevance = []
idx = 0
for i in range(103):
    relevance.append(int(data[idx*138]))
    idx+=1

precision, recall = [], []
rel = 0
for i,e in enumerate(relevance):
    if e:
        rel+=1
        precision.append(rel/(i+1))
    else:
        precision.append(rel/(i+1))
        
ret = 0        
for e in relevance:
    if e:
        ret+=1
        recall.append(ret/rel)
    else:
        recall.append(ret/rel)

plt.plot(recall,precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title('Precision vs Recall for qid:4')
plt.show()        

         