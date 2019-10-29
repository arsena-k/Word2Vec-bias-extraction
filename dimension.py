# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:41:20 2019

@author: Alina Arseniev
"""
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec, KeyedVectors
from statistics import mode, mean, stdev, median
from sklearn.model_selection import KFold 
from sklearn.metrics.pairwise import cosine_similarity

#import build_lexicon #Alina wrote, get it here: https://github.com/arsena-k/Word2Vec-bias-extraction
#note that build_lexicon depends on word_lists, also get it here: https://github.com/arsena-k/Word2Vec-bias-extraction

def calc_wordlist_mean(wordlist, w2vmodel): #calcuate the mean word-vector for a few words
    wordlist= [w2vmodel.wv[i] for i in wordlist]
    meanvec = np.mean(wordlist,0) 
    meanvec= preprocessing.normalize(meanvec.reshape(1,-1), norm='l2') #ensure normalized to len 1
    meanvec= meanvec.reshape(w2vmodel.vector_size,)  #now will work with gensim similarity fcns
    return(meanvec)
   
#main class which holds a dimension, extracted using either larsen or bolukbasi geometric approaches    
class dimension: 
    def __init__(self, semantic_direction, method):
        self.semantic_direction= semantic_direction #this is another class, made with code in build_lexicon.py
        self.method= method #larsen or  bolubasi
        self.w2vmodel= semantic_direction.w2vmodel
        assert self.method in ('larsen', 'bolukbasi'), "Select one method: 'larsen' or 'bolukbasi'"

    def calc_dim_boluk(self): #https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py
        #the original vec found using the pca method may be pointing either pos to neg or neg to pos, may need to flip it to match training data, below. This is only flipped based on the TRAINING words, not testing or outcome words. 
        assert len(self.semantic_direction.pos_train)==len(self.semantic_direction.neg_train), "The Bolukbasi method is not applicable to non-paired data, it requires equal number of positive and negative training words that are paired" 
        assert self.semantic_direction.direction_of_interest not in ('purity', 'moral_mfd'), "The Bolukbasi method is not applicable to non-paired data, it requires equal number of positive and negative training words that are paired"
        pairs= zip(self.semantic_direction.pos_train, self.semantic_direction.neg_train)
        num_components = len(self.semantic_direction.pos_train)+len(self.semantic_direction.pos_train)
        matrix = []
        for a, b in pairs:
            center = (self.w2vmodel.wv[a] + self.w2vmodel.wv[b])/2
            matrix.append(self.w2vmodel.wv[a] - center)
            matrix.append(self.w2vmodel.wv[b] - center)
        matrix = np.array(matrix)
        pca = PCA(n_components = num_components)
        pca.fit(matrix)
    
        return pca  #this is actually just returning a pca object from sklearn
        #bar(range(num_components), pca.explained_variance_ratio_)
        #gender_direction = doPCA(pairsboluk, model).components_[0] #already normalized
        #doPCA(pairsboluk, model).explained_variance_ratio_ #
        
    def boluk_evals(self):
        return(self.calc_dim_boluk().explained_variance_ratio_)
        
    def pca_direction_matcher(self):
        #the original vec found using the pca method may be pointing either pos to neg or neg to pos, may need to flip it to match training data, below. This is only flipped based on the TRAINING words, not testing or outcome words. 
        training_pos_sims=[]
        initialvec= self.calc_dim_boluk().components_[0] #this is the original vec found using pca method
        for i in self.semantic_direction.pos_train:
            training_pos_sims.append(cosine_similarity(initialvec.reshape(1,-1), self.w2vmodel.wv[i].reshape(1,-1))[0][0]) #just get the number and append it, rather than append it as a np array
        training_pos_sims= [float(i) for i in training_pos_sims]
        if mean(training_pos_sims)<0:
            return(-initialvec)
        else:
            return(initialvec)
            
    def dimensionvec(self):  
        if self.method =='bolukbasi':
            return self.pca_direction_matcher() #this is the dimension vector according to the bolukbasi method
        elif self.method =='larsen':
            return self.calc_dim_larsen() #this is the dimension according to the larsen method
        
    #get difference between two sets vectors
    def calc_dim_larsen(self): 
        diffvec= calc_wordlist_mean(self.semantic_direction.pos_train, self.w2vmodel) - calc_wordlist_mean(self.semantic_direction.neg_train, self.w2vmodel)
        diffvec= preprocessing.normalize(diffvec.reshape(1,-1), norm='l2')
        diffvec= diffvec.reshape(self.w2vmodel.vector_size,) #now will work with gensim similarity fcns
        return(diffvec)
        #return cossims between the found vector and some new word(s), and choose returnNAs if you still want to return words even if NAs 
    
    def cos_sim(self,inputwords, returnNAs): 
        assert type(inputwords)==list, "Enter word(s) as a list, e.g., ['word']"
        interesting_dim=self.dimensionvec().reshape(1,-1) 
        cossims= []
        for i in np.array(inputwords):
            if i=='nan' and returnNAs==True:
                cossims.append(np.nan)
            elif i!='nan':
                try:
                    cossims.append(cosine_similarity(self.w2vmodel.wv[i].reshape(1,-1),interesting_dim)[0][0])
                except KeyError:
                    if returnNAs==True:
                        cossims.append(np.nan)
                    continue
        return(cossims)
          
    def cos_sim_bodyterms(self): 
        bodyterms= ['obese','morbidly_obese',  'obesity',  'being_overweight', 'overweight','gained_weight','excess_weight', 'weight_gain',
                    'thin', 'skinny', 'slender', 'slim', 'lithe',
                    'burly','muscular', 'lean', 'toned',
                    'anorexic', 'anorexia', 'bulimia','bulimic', 'obesity_epidemic', 
                    'diet', 'dieting', 
                    'health', 'healthy', 'unhealthy',
                    'overeating', 'sedentary','inactive',
                    'exercise', 'active', 'physically_fit',
                    'flab', 'flabby', 'fat']
        
        return(bodyterms, self.cos_sim(bodyterms, returnNAs=True)) #note this returns words and then cossims
    
    def trainaccuracy(self): 
        true_class=[]
        cossim_vec= []
        predicted_class=[]
        cossim_vec.extend(self.cos_sim(self.semantic_direction.pos_train, returnNAs= False))
        cossim_vec.extend(self.cos_sim(self.semantic_direction.neg_train, returnNAs=False))
        
        for i in self.semantic_direction.pos_train:
            true_class.append(1)
        for i in self.semantic_direction.neg_train:
            true_class.append(0)
        for i in cossim_vec:
            if float(i) > 0:
                predicted_class.append(1)
            if float(i) <0:
                predicted_class.append(0)
        accuracy= accuracy_score(true_class, predicted_class)  
        return(accuracy, true_class, predicted_class, cossim_vec) #note: this will not, by default, include any training words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)
    

    def testaccuracy(self):
        true_class=[]
        cossim_vec= []
        predicted_class=[]
        cossim_vec.extend(self.cos_sim(self.semantic_direction.pos_test, returnNAs=False))
        cossim_vec.extend(self.cos_sim(self.semantic_direction.neg_test, returnNAs=False))
        
        for i in self.semantic_direction.pos_test:
            true_class.append(1)
        for i in self.semantic_direction.neg_test:
            true_class.append(0)
        for i in cossim_vec:
            if float(i) > 0:
                predicted_class.append(1)
            if float(i) <0:
                predicted_class.append(0)
        
        accuracy= accuracy_score(true_class, predicted_class)  
        return(accuracy, true_class, predicted_class, cossim_vec) #note: this will not, by default, include any testing words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)

 
#enter in the words for the semantic direction, using build_lexicon class    
def kfold_dim(semantic_direction,method='larsen', splits= 10): #splits will be among the smaller of pos and neg train #only works for larsen method here, since bolukbasi method requires paired data
    min_group_n= len(min([semantic_direction.pos_train, semantic_direction.neg_train], key=len))
    min_group= semantic_direction.pos_train if len(semantic_direction.pos_train)== min_group_n else semantic_direction.neg_train
    kf= KFold(n_splits=splits, shuffle=True)  
    testaccy=[]
    trainaccy=[]
    for train_index, test_index in kf.split(np.array(min_group)): #kf.split(np.array(semantic_direction.pos_train)):
        
        semantic_directionk = copy.deepcopy(semantic_direction)
        
        semantic_directionk.pos_test = list(np.array(semantic_directionk.pos_train)[test_index]) #must do test indices first since modifying training words next 
        semantic_directionk.neg_test = list(np.array(semantic_directionk.neg_train)[test_index])
   
        semantic_directionk.pos_train = list(np.array(semantic_directionk.pos_train)[train_index]) #kfold in sklearn only works for arrays
        semantic_directionk.neg_train = list(np.array(semantic_directionk.neg_train)[train_index])
   
        larsen_k= dimension(semantic_directionk, method='larsen') 
        print("Train Accuracy:" + str(larsen_k.trainaccuracy()[0]), "Test Accuracy:" + str(larsen_k.testaccuracy()[0]))
        trainaccy.append(larsen_k.trainaccuracy()[0])
        testaccy.append(larsen_k.testaccuracy()[0])
    print("With default 10 splits, each training subset size: " + str(len(train_index)), "Each hold-out subset size: " + str(len(test_index)))
    print('\033[1m' +'Mean, SD, Min Accuracy across Training Subsets: '  + '\033[0m'+ str(round(mean(trainaccy), 2)), str(round(stdev(trainaccy),2)),str(round(min(trainaccy),2)) )
    print('\033[1m' +  'Mean, SD, Min Accuracy across Held-Out Subsets: ' + '\033[0m'+ str(round(mean(testaccy),2)), str(round(stdev(testaccy),2)), str(round(min(testaccy),2)))
   

