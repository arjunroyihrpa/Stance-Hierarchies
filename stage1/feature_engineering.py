# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:38:34 2018

@author: Arjun
"""
from keras.models import Sequential
import os
import math
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from numpy import asarray
#from nltk.stem.wordnet import WordNetLemmatizer
_wnl = nltk.WordNetLemmatizer()
import gensim.models.keyedvectors as word2vec
from nltk import pos_tag
import pandas as pd
from nltk import word_tokenize  
from scipy import spatial
import retinasdk
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from nltk.util import ngrams
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input
from sklearn.feature_extraction.text import TfidfVectorizer
from math import isnan
from textdistance import cosine
pth="your--path/my_stance_data/"

vectorizer = TfidfVectorizer(max_features=2000)
dfb=pd.read_csv(pth+"train_bodies.csv")
dfh=pd.read_csv(pth+"train_stances.csv")

lb=dfb.values.tolist()
lh=dfh.values.tolist()
voch,vocb=[],[]



liteClient = retinasdk.LiteClient("d2690680-f10c-11e8-bb65-69ed2d3c7927")
def lem(l):
    a=l.split()
    b=[]
    for w in a:
        b.append(_wnl.lemmatize(w))
    return " ".join(b)    

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies,kh,kb, feature_file):
#def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):    
    #print ("etaiiiiiiiiiiiiiiii DEKHOOOOOOOOOOOOOOOOO REEEEEEEEEEEEEEE")    
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies,kh,kb)
        np.save(feature_file, feats)

    return np.load(feature_file)


def texdis_st(headline,body):
    mx=0
    x=[]
    b = clean(body)
    h = clean(headline)
    for li in b.split('.'):
        sc=cosine(h,li)
        if sc>mx:
            mx=sc
    x.append(mx) 
    mxp=0
    for li in b.split('\n'):
        scp=cosine(h,li)
        if scp>mx:
            mxp=scp
    x.append(mxp)    
    scd=cosine(h,b)
    x.append(scd)      
    return  x   
 

def word_overlap_features(headlines, bodies,kh,kb):
    X = []
  #  print ("etaiiiiiiiiiiiiiiii DEKHOOOOOOOOOOOOOOOOO REEEEEEEEEEEEEEE")
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies,kh,kb):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies,kh,kb):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngram(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngram(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies,kh,kb):
#def hand_features(headlines, bodies):    
    #print("#################################-----ARJUN------##############")

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph
        #i=0
        

        clean_body = clean(body)
        clean_headline = clean(headline)
        #print(i,clean_headline)
        #lem_bdy=lem(clean_body)
        lem_hd=lem(clean_headline)
        features = []
        features = append_chargrams(features, lem_hd, clean_body, 2)
        features = append_chargrams(features, lem_hd, clean_body, 3)
        features = append_chargrams(features, lem_hd, clean_body, 4)
        #features = append_chargrams(features, lem_hd, clean_body, 5)
       # print("#################################-----ARJUN------##############")
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        #features = append_ngrams(features, clean_headline, clean_body, 5)
        #features = append_ngrams(features, clean_headline, clean_body, 6)
       # print("#################################-----ARJUN------##############")
        return features


    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        #X.append(count_grams(headline, body))
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)+count_grams(headline, body))#+texdis_st(headline, body))

    return X

def score_feature(headlines, bodies,kh,kb):
    nop=['s','’']
    def sim(h,b):
        hd=word_tokenize(h)
        bd=word_tokenize(b)
        poh=pos_tag(hd)
        pob=pos_tag(bd)
        h=[]
        b=[]
        n=0
        for i in range(len(poh)):
       # if poh[i][1].startswith('NN') and poh[i][0] not in nop :
            if (poh[i][1]==('NNP') or poh[i][1]==('NNPS')) and poh[i][0] not in nop :
                h.append(poh[i][0])
        for i in range(len(pob)):
                    # if pob[i][1].startswith('NN') and pob[i][0] not in nop :
            if (pob[i][1]==('NNP') or pob[i][1]==('NNPS')) and pob[i][0] not in nop:
                b.append(pob[i][0])
            
        for i in range(len(h)):
            if h[i] in b:
                n+=1
        if len(h)==0:
             s=-1
        else:        
             s=n/len(h)
       
        return s      

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(sim(headline, body))
    return X        
       
'''
embed_index=dict()
f = open('./glove.6B/glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embed_index[word] = coefs
f.close()
'''

def word_features(h,b):  ## before running this please download GoogleNews-vectors-negative300.bin
    model = word2vec.KeyedVectors.load_word2vec_format('your--path/GoogleNews-vectors-negative300.bin', binary=True) 
    def vecs(e,n):
        m=0
        h=clean(e)        
        #k=0
        c=0
        if n==1:
           k=35 
        if n==2:
            k=200
        '''
        sen=[]
        for w in h.split():           
            if c<k:
                m+=1
                if w is embed_index:
                    ws=embed_index.get(w)
                    if ws is not 'nan':
                        senv=np.array(ws)
                        sen.append(senv)
                        c+=1
        while c<k:
            senp=np.zeros((100))
            if c<k:
                sen.append(senp)
                c+=1
        sen=np.array(sen)
        sen=sen.reshape(k,100,1).astype('float64')
        return sen         
        '''
        h=clean(e)
        #b=clean(bod)
        senp=np.zeros((300))
        for w in h.split():
            #if w in b.split():
            m+=1
            if w in model:
                ws=model[w]
                #if ws is not 'nan':
                senv=np.array(ws)
                senp=senp+senv
        if m!=0:
            senp=senp/m
        return senp
               
             
    X=[]
    H=[]
    for i, (hd,bd) in tqdm(enumerate(zip(h,b))):
        a=vecs(hd,1)
        b=vecs(bd,2)
        H.append(a) 
        X.append(b)
    X=np.array(X)
    H=np.array(H)
    #print (X.shape)    
    #print ("m is------",b)  
    return X,H  

def word_vec_sim(headlines, bodies,kh,kb):
    
    model = Sequential()

    model.add(Embedding(1000, 64, input_length=100))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta')
    '''
    #vmodel = word2vec.KeyedVectors.load_word2vec_format('./Google_word_vec/GoogleNews-vectors-negative300.bin', binary=True)
    #embed_index=dict()
    f = open('./glove.6B/glove.6B.100d.txt', encoding="utf8")
    model = KeyedVectors.load_word2vec_format('D:/Work/Google_word_vec/GoogleNews-vectors-negative300.bin', binary=True)
    voc=[]
    voc.append(headlines)
    voc.append(bodies)
    '''
    f = open('./glove.6B/glove.6B.100d.txt', encoding="utf8")
    embed_index=dict()
    for line in f:
	     values = line.split()
	     word = values[0]
	     coefs = asarray(values[1:], dtype='float32')
	     embed_index[word] = coefs
    f.close() 
    '''
    max_len_text=60
    tok_bdy=Tokenizer()
    tok_bdy.fit_on_texts(voc)
    text_vocab_size=len(tok_bdy.word_index)+1
    
    embed_mat_bdy=np.zeros((text_vocab_size,300))
    print(text_vocab_size)
    for word, i in tok_bdy.word_index.items():
	       if word in model:
		        embedding_vector = model[word]
		        if embedding_vector is not None: embed_mat_bdy[i] = embedding_vector  
                                            
    input1=Input(shape=(max_len_text,)) 
    embed1=Embedding(output_dim=300,input_dim=text_vocab_size, input_length=max_len_text, weights=[embed_mat_bdy], trainable=False)(input1)
    clf=Model(inputs=[input1],outputs=embed1)
    clf.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    '''
    def w_sim(h,b):
        hd=clean(h)
        bd=clean(b)
        th=[]
        tb=[]
        c=0
        k=0
        #print (hd)
        
        for w in hd.split():
             if w in embed_index:
                 ws=embed_index.get(w)
                 vec=np.array(ws)
                 th.append(vec)
                 c+=1
        if c==0:
            senp=np.zeros((100))
            th.append(senp)
            c+=1
        for w in bd.split():
             if k<(4*c):
                 if w in embed_index:
                    ws=embed_index.get(w)
                    vec=np.array(ws)
                    tb.append(vec)
                    k+=1 
        while k<4*c:
            senp=np.zeros((100))
            tb.append(senp)
            k+=1
        th=np.array(th)
        tb=np.array(tb)
        #print (tb.shape)
        #print (th.shape)
        '''
        encode_text=tok_bdy.texts_to_sequences([hd])
        th=pad_sequences(encode_text, maxlen=max_len_text, padding='post')
        
        #th=th.reshape(1,max_len_text)
        encode_text2=tok_bdy.texts_to_sequences([bd])
        #print(len(encode_text),len(encode_text2))
        tb=pad_sequences(encode_text2, maxlen=max_len_text, padding='post')
        #print(th.shape,tb.shape)
        
        #tb=tb.reshape(1,max_len_text)
        out1=clf.predict(th)
        out2=clf.predict(tb)
        np.array(out1).shape
        #o1=out1.reshape(c*100*64)
        o1=out1.reshape(60*300)
        o2=out2.reshape(60*300)
        rs=1 - spatial.distance.cosine(o1, o2)
        if float('-inf') < float(rs) < float('inf'):
             return rs
        else:
            return 0.001
        
        '''
        out1=model.predict(th)
        o1=out1.reshape(c*100*64)
        mx=0
        for i in range(4):
            inp2=tb[i*c:(i+1)*c]
            out2=model.predict(inp2)
            o2=out2.reshape(c*100*64)
            rs=1 - spatial.distance.cosine(o1, o2)
            if rs>mx:
                mx=rs
        #print (mx)        
        return mx  
        
        
            
                 
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(w_sim(headline, body))
    return X        
    



def features_sim(headlines, bodies,keyh,keyb):
    def overlap_similar(headline, body,k1,k2):

        clean_hd = clean(headline)
        clean_bd = clean(body)
        headline_tok = get_tokenized_lemmas(clean_hd)
        body_tok = get_tokenized_lemmas(clean_bd)
        kh=get_tokenized_lemmas(clean(k1))
        kb=get_tokenized_lemmas(clean(k2))
        bigrm_hd=[]
        for w in ngrams(headline_tok[0:],2):
            bigrm_hd.append(w)

        key=len(set(kh).intersection(set(kb)))

        c=cosine(clean_hd,clean_bd)
        uni=len(set(headline_tok).intersection(set(body_tok)))
        bigrm_bd=[]
        for w in ngrams(body_tok[0:],2):
            bigrm_bd.append(w)
        bi=len(set(bigrm_hd).intersection(set(bigrm_bd))) 
        #scr=uni+2*bi+5*c+2*key
        '''
        tgrm_hd=[]
        for w in ngrams(headline_tok[0:],3):
            tgrm_hd.append(w)
        tgrm_bd=[]
        for w in ngrams(body_tok[0:],3):
            tgrm_bd.append(w)
        ti=len(set(tgrm_hd).intersection(set(tgrm_bd)))    
        '''
        #t=texdis_st(headline, body)
                  #s=' '.join(lt)
        #h=' '.join(headline_tok)    
        #bigrm_hd=nltk.bigrams(clean_headline)
        #bigrm_bd=nltk.bigrams(clean_body)
        #return [key]
        return [key]#,c,uni,bi]
    X = []
    for i, (headline, body,k1,k2) in tqdm(enumerate(zip(headlines, bodies,keyh,keyb))):
        X.append(overlap_similar(headline, body,k1,k2))
    return X    
for v in lb:
    vocb.append(clean(v[1]))
for v in lh:
    voch.append(clean(v[0]))  
XH=vectorizer.fit(voch)
XB=vectorizer.fit(vocb)    

def tfidf(headlines, bodies):
    X=[]
    X1=XH.transform(headlines).toarray()
    X2=XB.transform(bodies).toarray()
    
    for i in range(len(X1)):
        #z.append(X1[i])
        #z.append(X2[i])
        x1=list(X1[i])
        x2=list(X2[i])
        rs= 1- spatial.distance.cosine(x1,x2)
        if math.isnan(rs)==True:
            rs=0.0
        X.append(list(X1[i])+list(X2[i])+[rs])
        #else:
        #    X.append(list(X1[i])+list(X2[i])+[0.0])
        #z=np.reshape(np.array(z).astype(float),4001)
        #X.append(z)
    #print(R)
    #rs= cosine_similarity(X1,X2)    
    print(np.array(X).shape)    
    return np.c_[X1,X2]  

    
    
    