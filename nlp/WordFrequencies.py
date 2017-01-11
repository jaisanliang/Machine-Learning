'''
Module that handles background work such as dividing
data into training, cross-validation, and test sets
'''

import time
import matplotlib.pyplot as plt
from pattern.en import ngrams
from pattern.en import lemma

start=time.clock()
#text='Hi my name is be Jason'
textFile=open('text.txt','r')
text=''
for line in textFile:
        text+=line
words=ngrams(text,n=1)
words=[str(lemma(word[0])) for word in words]
length=len(words)
words.sort()
wordFreq=dict()
count=0
'''
for i in range(len(words)):
	if i==len(words)-1:
		continue
	elif words[i]!=words[i+1]:
		wordFreq[words[i]]=count+1
		count=0
	else:
		count+=1
'''
for word in words:
        if word not in wordFreq.keys():
                wordFreq[word]=1.0/length
                count=0
        else:
                wordFreq[word]+=1.0/length
print sorted(wordFreq.items(),key=lambda x:x[1],reverse=True)[:20]
elapsed=time.clock()-start
print elapsed
