# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:51:17 2019

@author: Arjun
"""
import pandas as pd
df1=pd.read_csv("D:/Work/train_bodies.csv")

l=df1.values.tolist()

ide=[]
for i in range(len(l)):
    ide.append(l[i][0])





fn='D:/Work/FNC/my_stance_data/LIWC2015_Annotations/LIWC2015_Results (train_bodies).txt'
f=open(fn,"r",encoding="utf-8")
k=f.read().split('\n')
line,d=[],[]
for l in k[2:]:
    x=(l.split('\t'))
    if x[0].isnumeric() and int(x[0]) in ide:
            if len(d)==0:
               d.append(int(x[0]))
            if len(x)>90:
                for v in x[2:]:
                    d.append(v)
                line.append(d)
                d=[]    
    else:
        if len(x)>90 and len(d)==1:
            for v in x[1:]:
                d.append(v)
            line.append(d)        
            d=[]

hd=[]
hd.append('BodyID')
for v in k[0].split('\t')[2:]:
    hd.append(v)           
            
df = pd.DataFrame(line,columns=hd)

df.to_csv('D:/Work/FNC/my_stance_data/LIWC2015_Annotations/train_bodies.csv', index=False) 

['Analytic',2,3,4,'conj','negate','compare','affect','posemo','negemo','anx','anger','sad','differ','affiliation','achieve']           