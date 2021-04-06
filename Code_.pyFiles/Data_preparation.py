#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
import codecs


# In[2]:


tokenizer = RegexpTokenizer(r'\w+')
detokenizer = TreebankWordDetokenizer()


# In[3]:


colnames=['Snippets', 'Targets'] 
df = pd.read_csv('dataset.csv', names=colnames, header=None)
embedding_file = './crawl-300d-2M.vec'


# In[4]:


def loadEmbed():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(embedding_file, encoding='utf-8')
    flag = True
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float64')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index


# In[5]:


#Loading pre-trained embeddings
model=loadEmbed()


# In[6]:


all_words=[]
for i in range(len(df['Snippets'])):
    all_words.extend(df['Snippets'][i].split())
all_words=list(dict.fromkeys(all_words))
all_words=[x.lower() for x in all_words]


# In[7]:


embeddings={}
for each in all_words:
    if each.lower() not in model.keys():
        embeddings[each]=model['unk']
    else:
        embeddings[each]=model[each.lower()]


# In[8]:


ppd=pd.read_csv('pre_processed_dataset.csv',encoding='utf-8')
length_l=[]
for i in range(len(ppd['left_context'])):
    length_l.append(len(tokenizer.tokenize(ppd['left_context'][i])))
length_r=[]
for i in range(len(ppd['right_context'])):
    length_r.append(len(tokenizer.tokenize(ppd['right_context'][i])))


# In[9]:


embeddings['<pad>']= [0]*300
# Embedding Left Context 
keras_left_context=[]
for i in range(len(ppd['left_context'])):
    one_vector=[]
    temp=tokenizer.tokenize(ppd['left_context'][i])
    one_vector.append(model['start'])
    for m in temp[1:]:
        a = embeddings[m.lower()]
        one_vector.append(a)
    one_vector.extend([embeddings['<pad>'] for x in range(78-length_l[i])])
    keras_left_context.append(one_vector)


# In[10]:


# Embedding Right Context 
keras_right_context=[]
for i in range(len(ppd['right_context'])):
    one_vector=[]
    temp=tokenizer.tokenize(ppd['right_context'][i])
    for m in temp[:-1]:
        one_vector.append(embeddings[m.lower()])
    one_vector.append(model['end'])
    one_vector.extend([embeddings['<pad>'] for x in range(78-length_r[i])])
    keras_right_context.append(one_vector)


# In[11]:


# Embedding Candidate Word
keras_middle=[]
for i in range(len(ppd['Candidate_words'])):
    keras_middle.append(embeddings[ppd['Candidate_words'][i].lower()])


# In[12]:


#Saving the processed dataset in a pickle file
f = open(b"Data_fast.pkl","wb")
pickle.dump(zip(keras_left_context,keras_right_context,keras_middle,ppd['target_status']),f)

