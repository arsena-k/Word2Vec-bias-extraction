# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:03:17 2019

@author: Alina Arseniev
"""
import numpy as np
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec, KeyedVectors
from statistics import mean, stdev
from sklearn import svm

class semantic_svm:
    def __init__(self, semantic_direction):
        self.semantic_direction= semantic_direction #this is another class, made with code in build_lexicon.py
        self.method= 'SVM Linear Classifer' 
        self.w2vmodel= semantic_direction.w2vmodel
        if len(self.semantic_direction.pos_train) < 50 or len(self.semantic_direction.pos_train) < 50:
            print("Warning, SVM Classifier may overfit with few training words")
        if self.w2vmodel.vector_size > 300:
            print("Warning, SVM Classifier may overfit with high dimensional word-vectors")
        
        train_vecs= [self.w2vmodel.wv[i] for i in self.semantic_direction.pos_train]
        train_classes = [1 for i in self.semantic_direction.pos_train]
    
        for i in self.semantic_direction.neg_train:
            train_vecs.append(self.w2vmodel.wv[i])
            train_classes.append(0)
        test_vecs= [self.w2vmodel.wv[i] for i in semantic_direction.pos_test]
        test_classes = [1 for i in semantic_direction.pos_test]
        for i in semantic_direction.neg_test:
            test_vecs.append(self.w2vmodel.wv[i])
            test_classes.append(0)
        self.train_vecs=train_vecs
        self.test_vecs=test_vecs
        self.train_classes=train_classes
        self.test_classes= test_classes
        clf=svm.SVC(kernel='linear', C=1, random_state=123) #set up classifier hyperparameters #add random state for data shuffle
        self.clf = clf.fit(self.train_vecs, self.train_classes) #train classifier on all training data
        
    def trainaccuracy(self):
        self.train_preds = self.clf.predict(self.train_vecs) #make predictions on training words, not doing probabilities here since lwo sample sizes of words to train on, probabilities aren't likely reliable
        accuracy_prop= accuracy_score(self.train_classes, self.train_preds) 
        accuracy_N= accuracy_score(self.train_classes, self.train_preds, normalize=False) 
        return(accuracy_prop, accuracy_N, self.train_preds) #note: this will not, by default, include any training words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)

    def testaccuracy(self):
        self.test_preds= self.clf.predict(self.test_vecs) #note: this will not, by default, include any testing words not in the vocabulary
        accuracy_prop= accuracy_score(self.test_classes, self.test_preds)
        accuracy_N= accuracy_score(self.test_classes, self.test_preds, normalize=False)
        return(accuracy_prop, accuracy_N, self.test_preds)  #note: this will not, by default, include any training words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)

    def pred(self,inputwords,returnNAs):
         assert type(inputwords)==list, "Enter word(s) as a list, e.g., ['word']"
         preds=[]
         #inputwords_clean=[]
         for i in inputwords:
            try:
                self.w2vmodel.wv[i]
                preds.append(self.clf.predict(np.asarray(self.w2vmodel.wv[i]).reshape(1,-1))[0])
            except KeyError:
                if returnNAs==True:
                    preds.append(np.nan)
                continue
         return(preds)
         
    def pred_bodyterms(self): 
        bodyterms= ['obese','morbidly_obese',  'obesity',  'being_overweight', 'overweight','gained_weight','excess_weight', 'weight_gain',
                    'thin', 'skinny', 'slender', 'slim', 'lithe',
                    'burly','muscular', 'lean', 'toned',
                    'anorexic', 'anorexia', 'bulimia','bulimic', 'obesity_epidemic', 
                    'diet', 'dieting', 
                    'health', 'healthy', 'unhealthy',
                    'overeating', 'sedentary','inactive',
                    'exercise', 'active', 'physically_fit',
                    'flab', 'flabby', 'fat']
        
        return(bodyterms, self.pred(bodyterms, returnNAs=True)) #note this returns words and then predicted class
        
        