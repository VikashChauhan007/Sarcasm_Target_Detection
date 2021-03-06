#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
from empath import Empath


# In[2]:


colnames=['Snippets', 'Targets'] 
df = pd.read_csv('dataset.csv', names=colnames, header=None)
ppd=pd.read_csv('pre_processed_dataset.csv',encoding='utf-8')

ohe=OneHotEncoder()
lb=LabelEncoder()

# Using Stanford NER Tagger API
jar_n = 'stanford-ner-2018-10-16/stanford-ner-3.9.2.jar'
model_n = 'stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
ner_tagger = StanfordNERTagger(model_n, jar_n, encoding='utf8')

# Using Stanford POS Tagger API
jar = 'stanford-postagger-full-2018-10-16/stanford-postagger-3.9.2.jar'
model = 'stanford-postagger-full-2018-10-16/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')


# In[3]:


#Setting
import os
java_path = "C:/Program Files/Java/jdk-15.0.2/bin/java.exe"
os.environ['JAVAHOME'] = java_path


# In[ ]:


# Extracting POS Features
POS_snippets=[]
for i in range(len(df['Snippets'])):
    POS_snippets.extend(pos_tagger.tag(word_tokenize(df['Snippets'][i])))
POS_snippets_type=[x[1] for x in POS_snippets]
POS_snippets_type=lb.fit_transform(POS_snippets_type)
pos_vec=ohe.fit_transform(np.reshape(POS_snippets_type,(-1, 1)))
pos_vec=pos_vec.todense()


# In[4]:


# Extracting NER Features
ner_snippets=[]
for i in range(len(df['Snippets'])):
    ner_snippets.extend(ner_tagger.tag(word_tokenize(df['Snippets'][i])))
ner_snippets_type=[x[1] for x in ner_snippets]
ner_snippets_type=lb.fit_transform(ner_snippets_type)
ner_vec=ohe.fit_transform(np.reshape(ner_snippets_type,(-1, 1)))
ner_vec=ner_vec.todense()


# In[ ]:


# Extracting Empath Features
lexicon = Empath()
empath_vec=[]
for text in ppd['Candidate_words']:
    a=lexicon.analyze(text, normalize=True)
    bv=[]
    for i in a.values():
        bv.append(i)
    empath_vec.append(bv)


# In[5]:


#Dumping extracted features in a pickle file 
f = open(b"Data_aug.pkl","wb")
pickle.dump(zip(pos_vec, ner_vec, empath_vec),f,protocol = 2)

