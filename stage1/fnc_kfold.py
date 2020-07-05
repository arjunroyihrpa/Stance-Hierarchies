# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:39:46 2018

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
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features,score_feature,word_vec_sim,features_sim
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.metrics import classification_report,confusion_matrix
from utils.system import parse_params, check_version

#LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
def generate_features(stances,dataset,name):
    h, b,kh,kb, y = [],[],[],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        kh.append(stance['Key'])
        kb.append(dataset.keys[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b,kh,kb, "your--path/features/overlap."+name+".npy")
    #X_refuting = gen_or_load_feats(refuting_features, h, b,kh,kb, "your--path/features/refuting."+name+".npy")
    #X_polarity = gen_or_load_feats(polarity_features, h, b,kh,kb, "your--path/features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b,kh,kb, "your--path/stage1/features/hand."+name+".npy")
    X_sc = gen_or_load_feats(score_feature, h, b,kh,kb, "your--path/stage1/features/score."+name+".npy")
    X_wvs = gen_or_load_feats(word_vec_sim, h, b,kh,kb, "your--path/stage1/features/wv_sim."+name+".npy")
    X_fs = gen_or_load_feats(features_sim, h, b,kh,kb, "your--path/stage1/features/feat_sim."+name+".npy")
    #X= np.c_[ X_hand,X_overlap,X_polarity,X_sc,X_wvs]
    #X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    #X = np.c_[X_hand,X_fs, X_refuting, X_overlap, X_sc]
    X= np.c_[ X_hand, X_fs,X_overlap,X_sc, X_wvs] 
    #X= np.c_[X_hand, X_fs,X_overlap,X_sc]
    #X= np.c_[ X_hand]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    #print (len(fold_stances[2]))
    #print (hold_out_stances[1])

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    #print (X_holdout)
    
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]
        
        #clf=SVC(kernel='rbf',verbose=True,random_state=14128)
        #clf=LinearSVC(verbose=True,random_state=14128,class_weight={0: 0.74535, 1: 1.7549},loss='hinge', max_iter=10000)
        #clf = AdaBoostClassifier(n_estimators=200, random_state=14128)
        #clf=SGDClassifier(loss='squared_hinge',verbose=0, random_state=14128)
        #clf = AdaBoostClassifier(SGDClassifier(loss='perceptron',max_iter=2000,verbose=0), algorithm='SAMME',n_estimators=200, random_state=14128)
        #clf = AdaBoostClassifier(SVC(kernel='linear'),algorithm='SAMME',n_estimators=200, random_state=14128)
        #clf = AdaBoostClassifier(LinearSVC(class_weight={0: 0.52, 1: 1.37},loss='hinge', max_iter=10000),algorithm='SAMME',n_estimators=200, random_state=14128)
        #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128,verbose=True)
        #clf = RandomForestClassifier(n_estimators=200, class_weight={0:0.074535,1:2.7549},random_state=53)
        #clf = AdaBoostClassifier(clf1,algorithm='SAMME',n_estimators=200, random_state=14128)
        #clf=LinearSVC(verbose=True,random_state=14128,class_weight={0: 0.52, 1: 1.37},loss='hinge', max_iter=10000)
        clf=LinearSVC(verbose=True,random_state=14128,class_weight={0: 0.74535, 1: 2.7549},loss='hinge', max_iter=15000)

    
        print (X_train.shape)
        clf.fit(X_train, y_train)

        predic = [LABELS[int(a)] for a in clf.predict(X_test)]
        act = [LABELS[int(a)] for a in y_test]

        fold_score, _,fa = score_submission(act, predic)
        max_fold_score, _,bfa = score_submission(act, act)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predic = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    act = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(act,predic)
    
    target_name=['Related','Unrelated']
    print(classification_report(act,predic,target_names=target_name))
    print("")
    print("")

    #Run on competition dataset
    predic = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    act = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(act,predic)
    print(classification_report(act,predic,target_names=target_name))
'''
for i in range(len(l)):
    if i in misr:
        if l[i][2]=='agree':
            a+=1
        elif l[i][2]=='disagree':    
            da+=1
        elif l[i][2]=='discuss':    
            ds+=1
            

a
Out[22]: 168

da
Out[23]: 158

ds
Out[24]: 399

j=0
for i in range(len(l)):
  if l[i][2]== 'agree' or  l[i][2]=='disagree' or l[i][2]=='discuss':
      
    if i in misr:
        if l[i][2]=='agree':
            a.append(j)
        elif l[i][2]=='disagree':    
            da.append(j)
        elif l[i][2]=='discuss':    
            ds.append(j)
    j+=1  
    
$$$$$$$$$$$$
a,da,ds,misop=[],[],[],[]

j=0
for i in range(len(l)):
  if l[i][2]== 'agree' or  l[i][2]=='disagree' or l[i][2]=='discuss':
      
    if i in misr:
        misop.append(j)
        if l[i][2]=='agree':
            a.append(j)
        elif l[i][2]=='disagree':    
            da.append(j)
        elif l[i][2]=='discuss':    
            ds.append(j)
    j+=1
    

len(misop)
Out[22]: 588

len(ds)
Out[23]: 310

len(da)
Out[24]: 143    
'''
#################################################################################
'''




'''
