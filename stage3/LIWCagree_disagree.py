# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:17:14 2019

@author: Arjun
"""

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from utils.score import report_score
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textdistance import cosine
from nltk.util import ngrams
from sklearn.metrics import classification_report,confusion_matrix
from gensim.models import KeyedVectors

sid = SentimentIntensityAnalyzer()
path="your--path/my_stance_data/"
model = KeyedVectors.load_word2vec_format('your--path/GoogleNews-vectors-negative300.bin', binary=True)
def vec(lin):    
    vec=[]
    for l in lin:
             c=0.00001
             senp=np.zeros((300))
             for w in l.split():                       
               
                 if w in model:
                    ws=model[w]
                    senp+=ws							
                    c+=1			

             lvec=senp/c
             vec.append(lvec)
    return np.array(vec)

def read_data(st,bds,al):
  df1=pd.read_csv(path+st)
  l=df1.values.tolist()
  dfb=pd.read_csv(path+bds)
  lb=dfb.values.tolist()
  bdy,hdl=[],[]
  yt=[]

  for i in range(len(l)):
   if l[i][2]=="agree" or l[i][2]=="disagree":   
    line=''
    for j in range(len(lb)):
        if l[i][1] == lb[j][0]:
            
            line=lb[j][1]
    #hd,bd,hc,bc=make_data(l[i][0],line[0])
                   
    if l[i][2]=="agree" :
        y=0
    elif l[i][2]=="disagree":
        y=1
    al.append(l[i][0])
    al.append(line)
    bd=get_bd(line)
    
        
    yt.append(y)
    bdy.append(bd)
    hdl.append(l[i][0])
    
    #hnet.append([ph,nh,oh])
    #bnet.append([pb,nb,ob])
 
  return bdy,hdl,al

def get_bd(s):
    r=''
    s=s.replace('\n\n','.')
    s=s.replace('\n','.')
    x=s.split('.')
    for i in range(len(x)):
        if i<10:
            r=r+str(x[i])+" "
    return r
def overlap_similar(headline, body):

        clean_hd = clean(headline)
        headline_tok = clean_hd.split()
        
        bigrm_hd=[]
        for w in ngrams(headline_tok[0:],2):
            bigrm_hd.append(w)
        al=body.split('\n')
        mx=0
        s=''
        for l in al:
              c=cosine(clean_hd,clean(l))
              lt=clean(l).split()
              uni=len(set(headline_tok).intersection(set(lt)))
              blg=[]
              for w in ngrams(lt[0:],2):
                  blg.append(w)
              bi=len(set(bigrm_hd).intersection(set(blg)))    
              scr=uni+2*bi+5*c
                 
              if scr>mx:
                  mx=scr
                  s=' '.join(lt)
        h=' '.join(headline_tok)    
        #bigrm_hd=nltk.bigrams(clean_headline)
        #bigrm_bd=nltk.bigrams(clean_body)
       
        return h,s
    
def get_senti(h,b):
    hd,bd = overlap_similar(h,b)
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
def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def senti_features(fb,fh):
    X1,X2=[],[]
    b="your--path/main_data/"+fb
    h="your--path/main_data/"+fh
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    for i in range(len(l)): 
        lab=l[i][2]
        if lab=='agree' or lab=='disagree':
            bid=l[i][1]
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                  
                    sh,sb=get_senti(l[i][0],lb[j][1])
                    X1.append(sh)
                    X2.append(sb)
                
    return X1,X2               
def refuting_features(fb,fh):
    dic = ['not','deny','denies','refuses','fake','fraud','hoax','false', 
                       'despite','nope','doubt', 'doubts','bogus',
                       'debunk','prank','pranks',
        'retract',"t", "not", "never", 
           "couldn", "didn","don", "has", "have", "isn",
            "nope", "nowhere",
           "shouldn", "rarely", "seldom",
            "denied", 'reject',"rejected"]
    _refuting_words=list(set(dic))
    
    X = []
    b="your--path/my_stance_data/"+fb
    h="your--path/my_stance_data/"+fh
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    for i in range(len(l)): 
        lab=l[i][2]
        if lab=='agree' or lab=='disagree':
            bid=l[i][1]
            f1,f2=[],[]
            #clean_headline = clean(l[i][0])
            clean_headline = l[i][0].split()
            for word in _refuting_words:
                if word in clean_headline:
                    f1.append(1)
                else:
                    f1.append(0)
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                    clean_bod = clean(lb[j][1])
                    clean_bod = clean_bod.split()
                    for word in _refuting_words:
                        if word in clean_headline:
                            f2.append(1)
                        else:
                            f2.append(0)
            X.append(np.array(f1)+np.array(f2))         
    return X

sf=[1, 2, 3, 4, 21, 22, 25, 29, 30, 31, 32, 33, 34, 46, 57, 58]

def load_LIWC_features(fb,fh):
    b="your--path/stage3/my_stance_data/LIWC2015_Annotations/"+str(fb)
    h="your--path/stage3/my_stance_data/LIWC2015_Annotations/"+str(fh)
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    X,y=[],[]
    for i in range(len(l)):
      if i!=0: 
        lab=l[i][2]
        #print(lab)
        if lab=='agree' or lab=='disagree':
            #print ('inside')
            '''
            if lab=='agree':
                y.append(0)
            else:
                y.append(1)
            '''
            y.append(lab)
            bid=l[i][1]
            f=[]
            for v in range(len(l[i])):
                if v-3 in sf:
                    f.append(l[i][v])
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                    for v in range(len(lb[j])):
                        if v-1 in sf:
                            f.append(lb[j][v])
            X.append(f)    
    return X,y
    
    
f1="train_stances.csv"
f2="train_bodies.csv"


Lwtr,ytrain=load_LIWC_features(f2,f1)


Reftr=refuting_features(f2,f1)

SentiHtr,SentiBtr=senti_features(f2,f1)

Xtrain=np.c_[Lwtr,Reftr,SentiHtr,SentiBtr]

#clf = GradientBoostingClassifier(n_estimators=300, random_state=14128,verbose=True)
#clf=SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0)
#clf=LinearSVC(class_weight='balanced',random_state=14128)
clf3=LinearSVC(C=1.5,class_weight={'agree':0.15,'disagree':3.57},loss='hinge',random_state=15,max_iter=16505)
#clf=RandomForestClassifier(n_estimators=100,criterion='entropy',class_weight={'agree':1.0,'disagree':4.5},random_state=5)
clf3.fit(Xtrain, ytrain)

op,dis=[],[]
for i in range(len(rel)):
    if sc[i]==0:
        op.append(rel[i])
    elif sc[i]==1:    
        dis.append(rel[i])
        
for y in dis:
    predic[y]='discuss' 

f3="competition_test_stances.csv"
f4="competition_test_bodies.csv"

sf=[1, 2, 3, 4, 21, 22, 25, 29, 30, 31, 32, 33, 34, 46, 57, 58]

def load_LIWC_features_test(fb,fh):
    b="your--path/stage3/my_stance_data/LIWC2015_Annotations/"+str(fb)
    h="your--path/stage3/my_stance_data/LIWC2015_Annotations/"+str(fh)
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    X,y=[],[]
    for i in range(len(l)):
      if i!=0: 
        if (i-1) in op:    ####### if only predicted as opinion by the  stage 2 
            
            y.append(l[i][2])
            bid=l[i][1]
            f=[]
            for v in range(len(l[i])):
                if v-3 in sf:
                    f.append(l[i][v])
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                    for v in range(len(lb[j])):
                        if v-1 in sf:
                            f.append(lb[j][v])
            X.append(f)    
    return X,y      



def senti_features_test(fb,fh):
    X1,X2=[],[]
    b="D:/Work/"+fb
    h="D:/Work/"+fh
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    for i in range(len(l)): 
        if i in op:
            bid=l[i][1]
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                  
                    sh,sb=get_senti(l[i][0],lb[j][1])
                    X1.append(sh)
                    X2.append(sb)
                
    return X1,X2

def refuting_features_test(fb,fh):
    dic = ['not','deny','denies','refuses','fake','fraud','hoax','false', 
                       'despite','nope','doubt', 'doubts','bogus',
                       'debunk','prank','pranks',
        'retract',"t", "not", "never", 
           "couldn", "didn","don", "has", "have", "isn",
            "nope", "nowhere",
           "shouldn", "rarely", "seldom",
            "denied", 'reject',"rejected"]
    _refuting_words=list(set(dic))
    
    X = []
    b="your--path/main_data/"+fb
    h="your--path/main_data/"+fh
    df1=pd.read_csv(h)
    l=df1.values.tolist()
    dfb=pd.read_csv(b)
    lb=dfb.values.tolist()
    for i in range(len(l)): 
        if i in op:
            bid=l[i][1]
            f1,f2=[],[]
            #clean_headline = clean(l[i][0])
            clean_headline = l[i][0].split()
            for word in _refuting_words:
                if word in clean_headline:
                    f1.append(1)
                else:
                    f1.append(0)
            for j in range(len(lb)):
                if int(lb[j][0])==int(bid):
                    clean_bod = clean(lb[j][1])
                    clean_bod = clean_bod.split()
                    for word in _refuting_words:
                        if word in clean_headline:
                            f2.append(1)
                        else:
                            f2.append(0)
            X.append(np.array(f1)+np.array(f2))         
    return X

Lwts,ytest=load_LIWC_features_test(f4,f3)
Refts=refuting_features_test(f4,f3)
SentiHts,SentiBts=senti_features_test(f4,f3) 
Xtest=np.c_[Lwts,Refts,SentiHts,SentiBts] 
agdispred=clf3.predict(Xtest)  
len(agdispred),len(op)  


a,d=0,0
for i in range(len(agdispred)):
    predic[op[i]]=agdispred[i]
    if agdispred[i]=='agree':
        a+=1
    else:    
        d+=1
 
df=pd.read_csv('your--path/main_data/competition_test_stances.csv')
l=df.values.tolist()


act=[]
for i in range(len(l)):
    act.append(l[i][2])


print("Scores on the test set")
#report_score(ytest,pred)
target_name=['agree','disagree','discuss','unrelated']
print(classification_report(act,predic,target_names=target_name))
print(confusion_matrix(act,predic))

'''
                   
'''        