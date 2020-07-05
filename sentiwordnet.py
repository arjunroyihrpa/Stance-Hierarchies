# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:58:01 2019

@author: Arjun
"""

from nltk.corpus import sentiwordnet as swn


list(swn.senti_synsets('breakdown'))[0].pos_score()
list(swn.senti_synsets('breakdown'))[0].neg_score()
list(swn.senti_synsets('breakdown'))[0].obj_score() 
'''#objective score= 1-(pos+neg)'''



from nltk.corpus import wordnet as wn

z=wn.synsets('dog', pos=wn.VERB)
q=str(z[0]).split('\'')[1]
print(swn.senti_synset(q))