# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:22:17 2019

@author: Arjun
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:32:12 2019

@author: Arjun
"""

import nltk
import xlrd
import numpy as np
import re
import string
import tensorflow
#import gensim
from numpy import asarray
import keras
from numpy import zeros
from gensim import corpora, models, similarities
from keras.utils import np_utils
from keras.models import Sequential
from nltk.util import ngrams
from keras.layers.merge import concatenate
#from keras.layers import Conv2D, MaxPooling2D, concatenate
from textdistance import cosine
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers import Conv1D, MaxPooling1D
from utils.system import parse_params, check_version
from keras.layers.core import Dense, Dropout, Activation, Flatten
from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plot
plot.style.use('ggplot')
#from keras.utils import plot_model
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Input
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.models import load_model
from keras.layers import Permute,Reshape, RepeatVector, Input, Lambda
from keras import regularizers
from keras import backend as k
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_wnl = nltk.WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
model = KeyedVectors.load_word2vec_format('your--path/GoogleNews-vectors-negative300.bin', binary=True)
#model = KeyedVectors.load_word2vec_format('D:/Work/fast_text_wiki/English/bin/wiki.en/wiki.en.vec')

path="your--path/stage2/my_stance_data/"
f1=path+"train_stances.csv"
f2=path+"train_bodies.csv"
f3=path+"competition_test_stances.csv"
f4=path+"competition_test_bodies.csv"

'''

f9=open("D:/Work/FNC/my_stance_data/new_exp/cnn2/sentlexi.txt") 
lines=f9.read().split('\n') 
sentr=[]
for l in lines[1:13428]:
    l=l.replace('[','')
    l=l.replace(']','')
    x=[]
    for j in l.split():
        x.append(float(j))
    sentr.append(x)    
f9=open("D:/Work/FNC/my_stance_data/new_exp/cnn2/sentlexi_bd.txt")
lines=f9.read().split('\n')
sbdtr=[] 
for l in lines[1:13428]:
    l=l.replace('[','')
    l=l.replace(']','')
    for j in l.split():
        x.append(float(j))
    sentr.append(x)     
'''
def split_line(line):
    cols = line.split("\t")
    return cols

def get_words(cols):
    words_ids = cols[4].split(" ")
    words = [w.split("#")[0] for w in words_ids]
    return words

def get_positive(cols):
    return cols[2]

def get_negative(cols):
    return cols[3]

def get_objective(cols):
    return 1 - (float(cols[2]) + float(cols[3]))

def get_gloss(cols):
    return cols[5]

                        
def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def get_bd(s):
    r=''
    s=s.replace('\n\n','.')
    s=s.replace('\n','.')
    x=s.split('.')
    for i in range(len(x)):
        if i<10:
            r=r+str(x[i])+" "
    return r


def senti_analyze(hd,bd):

    #elif l[i][2]=="unrelated":
     #   ytrain.append(3)

    hs=sid.polarity_scores(hd)
    bs=sid.polarity_scores(bd)
    hl=list(hs)
    bl=list(bs)
    hc=[]
    bc=[]
    for v in hl:
        hc.append(hs.get(v))
    for v in bl:
        bc.append(bs.get(v))
    hc=np.array(hc)
    bc=np.array(bc) 
    return hc,bc    

def read_data(st,bds,al,s):
  df1=pd.read_csv(st)
  l=df1.values.tolist()
  dfb=pd.read_csv(bds)
  lb=dfb.values.tolist()
  bdy,hdl,sah,sab=[],[],[],[]
  yt=[]

  for i in range(len(l)):
    if  l[i][2]=="agree" or l[i][2]=="disagree" or l[i][2]=="discuss":   
        line=''
        for j in range(len(lb)):
            if l[i][1] == lb[j][0]:
            
                line=lb[j][1]
    #hd,bd,hc,bc=make_data(l[i][0],line[0])
                   
        if l[i][2]=="agree" :
            y=0
        elif l[i][2]=="disagree":
            y=0
        elif l[i][2]=="discuss":
            y=1
        al.append(l[i][0])
        al.append(line)
    
    #ph,nh,oh= get_scores(hd.split())
    #pb,nb,ob= get_scores(bd.split())
    
        bd=get_bd(line)
        hs,bs=senti_analyze(l[i][0],bd)
        yt.append(y)
        bdy.append(bd)
        hdl.append(l[i][0])
        sah.append(hs)
        sab.append(bs)
        '''
        if s=='tr' and l[i][2]=='disagree':
             yt.append(y)
             bdy.append(bd)
             hdl.append(l[i][0])
             sah.append(hs)
             sab.append(bs)
         '''   
    #hnet.append([ph,nh,oh])
    #bnet.append([pb,nb,ob])
 
  return bdy,hdl,np.array(sab),np.array(sah),yt,al


vocab=[]
train_bdy,train_hd,sb_tr,sh_tr,ytrain,vocab=read_data(f1,f2,vocab,'tr')
test_bdy,test_hd,sb_ts,sh_ts,ytest,vocab=read_data(f3,f4,vocab,'ts')

tok_bdy=Tokenizer()
tok_bdy.fit_on_texts(vocab)
text_vocab_size=len(tok_bdy.word_index)+1

encode_text=tok_bdy.texts_to_sequences(train_bdy)
max_len_text=200
trainbd=pad_sequences(encode_text, maxlen=max_len_text, padding='post')


print (text_vocab_size)

encode_text2=tok_bdy.texts_to_sequences(train_hd)
max_len_hd=40
trainhd=pad_sequences(encode_text2, maxlen=max_len_hd, padding='post')

embed_mat_bdy=np.zeros((text_vocab_size,300))
#embed_mat_sent=np.zeros((text_vocab_size,3))

for word, i in tok_bdy.word_index.items():
	if word in model:
		embedding_vector = model[word]
	if embedding_vector is not None:
		embed_mat_bdy[i] = embedding_vector
        


topics_train = np.array(ytrain, dtype=int)
y_train = np_utils.to_categorical(topics_train,2)
y_train=np.array(y_train)



rel=[]
for i in range(len(predic)):
    if predic[i]=='related':
        rel.append(i)
        
        
def read_data(st,bds,al,s):
  df1=pd.read_csv(st)
  l=df1.values.tolist()
  dfb=pd.read_csv(bds)
  lb=dfb.values.tolist()
  bdy,hdl,sah,sab=[],[],[],[]
  yt=[]

  for i in range(len(l)):
    if i in rel:   
        line=''
        for j in range(len(lb)):
            if l[i][1] == lb[j][0]:
            
              line=lb[j][1]
        bd=get_bd(line)
        hs,bs=senti_analyze(l[i][0],bd)       
        bdy.append(bd)
        hdl.append(l[i][0])
        sah.append(hs)
        sab.append(bs)
        yt.append(l[i][2])
 
  return bdy,hdl,np.array(sab),np.array(sah),yt

test_bdy,test_hd,sb_ts,sh_ts,yt=read_data("your--path/main_data/competition_test_stances.csv","your--path/main_data/competition_test_bodies.csv",vocab,'ts')

encode_text3=tok_bdy.texts_to_sequences(test_bdy)
testbd=pad_sequences(encode_text3, maxlen=max_len_text, padding='post')



encode_text4=tok_bdy.texts_to_sequences(test_hd)
testhd=pad_sequences(encode_text4, maxlen=max_len_hd, padding='post')



input2=Input(shape=(max_len_hd,))              
input1=Input(shape=(max_len_text,))              
input3=Input(shape=(4,))
input4=Input(shape=(4,)) 

embed1=Embedding(output_dim=300,input_dim=text_vocab_size, input_length=max_len_text, weights=[embed_mat_bdy], trainable=False)(input1)
print (embed1.shape)
embed2=Embedding(output_dim=300,input_dim=text_vocab_size, input_length=max_len_hd, weights=[embed_mat_bdy], trainable=False)(input2)
conv1 = Conv1D(65, kernel_size=(1), activation='relu')(embed1)
pool1=GlobalMaxPooling1D(data_format='channels_last')(conv1)
conv2 = Conv1D(65, kernel_size=(1), activation='relu')(embed2)
pool2=GlobalMaxPooling1D(data_format='channels_last')(conv2)
merb=concatenate([pool1,input3])
merh=concatenate([pool2,input4])
#den0=Dense(32,activation="relu")(mer)
den1=Dense(32,activation="relu",kernel_regularizer=regularizers.l1(0.01),bias_regularizer=regularizers.l1(0.01))(merb)
den2=Dense(32,activation="relu",kernel_regularizer=regularizers.l1(0.01),bias_regularizer=regularizers.l1(0.01))(merh)
mer=concatenate([den1,den2])
output=Dense(2,activation=(lambda x: k.tf.nn.softmax(x)))(mer)
clf2=Model(inputs=[input1,input2,input3,input4],outputs=output)
clf2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
filepath="your--path/stage2/weights_embed_SA_3.hdf5"
clf2.load_weights(filepath)

score=clf2.predict([testbd,testhd,sb_ts,sh_ts])
sc=np.argmax(score,axis=1)


'''
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
for i in range(1):
    clf.fit([trainbd,trainhd,sb_tr,sh_tr], y_train, batch_size = 64, epochs =20,   callbacks=callbacks_list, validation_data=([valbd,valhd,sb_val,sh_val],y_val))


'''
