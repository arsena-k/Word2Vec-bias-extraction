# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:23:22 2019

@author: Alina Arseniev
"""

import word_lists #Alina wrote, get it here: https://github.com/arsena-k/Word2Vec-bias-extraction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from itertools import combinations
from random import seed, sample


#this cleans up a word list (or single word), so that only words in the w2v model are included
def clean_words(word_list, w2vmodel, returnNA, min_count=1): #by default, doesn't return words not in the vocab
    assert type(word_list)== list, "Enter words as a list"
    cleaned_list= []
    for i in word_list:
        if returnNA==False:
            try:
                w2vmodel.wv[i]
                if w2vmodel.wv.vocab[i].count >= min_count: #skip this word if it is not in the model at least min count times
                    cleaned_list.append(i)
            except KeyError: #skip this word if it is not in the model
                continue
        elif returnNA==True:
            try:
                w2vmodel.wv[i]
                if w2vmodel.wv.vocab[i].count >= min_count:
                    cleaned_list.append(i)
                else:
                    cleaned_list.append(np.nan)
            except KeyError:
                cleaned_list.append(np.nan) #add nan if this word if it is not in the model 
                continue
    return cleaned_list
    
    
    
#this cleans up two word lists that are paired (e.g., for the Bolukbasi method to find a dimension), so that only a word pair will be included if both words in the pair are in the w2v model
def clean_words_pairwise(pair1, pair2, w2vmodel, returnNA, min_count=1):  
    assert len(pair1)>0, "Requires at least two pairs"  
    assert len(pair1)==len(pair2), "Cleaning a pairwise wordlist requires the same number of positive and negative words, which are intentionally paired up"
    zipped= zip(pair1, pair2)
    pair1_inmodel= []
    pair2_inmodel= []
    for a,b in zipped:
        if returnNA==False:
            try:
                w2vmodel.wv[a]
                w2vmodel.wv[b]
            except KeyError:
                continue #skip this word pair if either word in the pair is not in the model
            if (w2vmodel.wv.vocab[a].count >= min_count and w2vmodel.wv.vocab[b].count >= min_count):
                pair1_inmodel.append(a)
                pair2_inmodel.append(b)
        elif returnNA==True:
            try:
                w2vmodel.wv[a]
                w2vmodel.wv[b]
                if (w2vmodel.wv.vocab[a].count >= min_count and w2vmodel.wv.vocab[b].count >= min_count):
                    pair1_inmodel.append(a)
                    pair2_inmodel.append(b)
                else:
                    pair1_inmodel.append(np.nan)
                    pair2_inmodel.append(np.nan)
            except KeyError:
                pair1_inmodel.append(np.nan)
                pair2_inmodel.append(np.nan)
                continue #skip this word pair if either word in the pair is not in the model
    return (pair1_inmodel, pair2_inmodel)

#put in a word list (e.g., all "immoral" wordvecs) to see the mean and SD of the simiarlities beween each set of two in the group (gives an idea of how coherent the word list is)
def sim_wordlist(words, w2vmodel):
    cossim_tracker=[]
    for i,j in combinations(range(0,len(words)), 2):
        if i!= j:
            cossim_tracker.append(cosine_similarity(w2vmodel.wv[words[i]].reshape(1,-1), w2vmodel.wv[words[j]].reshape(1,-1))[0][0])
    print('Mean Cosine Similarity:', np.mean(cossim_tracker), 'SD:', np.std(cossim_tracker))
    return(cossim_tracker)

    
#This just just keeps (cleaned) vocabulary to build a dimension, using the PRESET word lists from word_list.py. 
#It cleans the vocab we want to use to build a dimension, only vocabulary that is actually in the data. 
class dimension_lexicon_builtin:
    def __init__(self, direction_of_interest, w2vmodel, min_count=1):
        self.direction_of_interest= direction_of_interest #gender, health, ses, genderboluk, purity, or moral_mfd
        self.w2vmodel= w2vmodel
        
        if self.direction_of_interest =='gender':
            self.pos_label="feminine"
            self.neg_label="masculine"
            self.all_pos_train = word_lists.gender_train['pos']
            self.all_neg_train = word_lists.gender_train['neg']
            self.all_pos_test= word_lists.gender_test['pos'] 
            self.all_neg_test= word_lists.gender_test['neg']
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.gender_train['pos'], word_lists.gender_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.gender_test['pos'] , self.w2vmodel, returnNA=False)
            self.neg_test= clean_words(word_lists.gender_test['neg'], self.w2vmodel, returnNA=False)
        elif self.direction_of_interest=='health':
            self.pos_label="healthy"
            self.neg_label="unhealthy"
            self.all_pos_train = word_lists.health_train['pos']
            self.all_neg_train = word_lists.health_train['neg']
            self.all_pos_test= word_lists.health_test['pos'] 
            self.all_neg_test= word_lists.health_test['neg']
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.health_train['pos'], word_lists.health_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.health_test['pos'] , self.w2vmodel, returnNA=False)
            self.neg_test= clean_words(word_lists.health_test['neg'], self.w2vmodel, returnNA=False)
        elif self.direction_of_interest=='ses':
            self.pos_label="high class"
            self.neg_label="low class"
            self.all_pos_train = word_lists.ses_train['pos']
            self.all_neg_train = word_lists.ses_train['neg']
            self.all_pos_test= word_lists.ses_test['pos'] 
            self.all_neg_test= word_lists.ses_test['neg']
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.ses_train['pos'], word_lists.ses_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.ses_test['pos'] , self.w2vmodel,returnNA=False)
            self.neg_test= clean_words(word_lists.ses_test['neg'], self.w2vmodel,returnNA=False)
        elif self.direction_of_interest=='genderboluk': #bolukbasi's original gender words
            self.pos_label="feminine"
            self.neg_label="masculine"
            self.all_pos_train = word_lists.genderboluk_train['pos']
            self.all_neg_train = word_lists.genderboluk_train['neg']
            self.all_pos_test= word_lists.gender_test['pos'] 
            self.all_neg_test= word_lists.gender_test['neg']
            
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.genderboluk_train['pos'], word_lists.genderboluk_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.gender_test['pos'] , self.w2vmodel,returnNA=False)
            self.neg_test= clean_words(word_lists.gender_test['neg'], self.w2vmodel,returnNA=False)
        
        elif self.direction_of_interest=='gender2': #used for SVM to prevent overfitting
            self.pos_label="feminine"
            self.neg_label="masculine"
            self.all_pos_train = word_lists.gender2_train['pos']
            self.all_neg_train = word_lists.gender2_train['neg']
            self.all_pos_test= word_lists.gender_test['pos'] 
            self.all_neg_test= word_lists.gender_test['neg']
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.gender2_train['pos'], word_lists.gender2_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.gender_test['pos'] , self.w2vmodel,returnNA=False)
            self.neg_test= clean_words(word_lists.gender_test['neg'], self.w2vmodel,returnNA=False)

        elif self.direction_of_interest=='gender3': #used for SVM to prevent overfitting
            self.pos_label="feminine"
            self.neg_label="masculine"
            self.all_pos_train = word_lists.gender3_train['pos']
            self.all_neg_train = word_lists.gender3_train['neg']
            self.all_pos_test= word_lists.gender_test['pos'] 
            self.all_neg_test= word_lists.gender_test['neg']
            
            self.pos_train, self.neg_train= clean_words_pairwise(word_lists.gender3_train['pos'], word_lists.gender3_train['neg'], self.w2vmodel, returnNA=False, min_count=min_count) 
            self.pos_test= clean_words(word_lists.gender_test['pos'] , self.w2vmodel,returnNA=False)
            self.neg_test= clean_words(word_lists.gender_test['neg'], self.w2vmodel,returnNA=False)

        elif self.direction_of_interest=='purity':  #this is different than gender/health/ses since drawn from moral foundations theory lexicon, and more purity words than impurity words
            self.pos_label="pure (moral)"
            self.neg_label="impure  (immoral)"
            seed(123)
            samped_neg= sample(word_lists.purity['neg'], len(word_lists.purity['pos'])+10) #sample neg words. Adding 10 to this neg set, since adding 10 more to neg testing than the pos testing sets, below
            
            self.all_pos_train, self.all_pos_test = train_test_split(word_lists.purity['pos'] , test_size=10, random_state=42) #there are few purity words
            self.all_neg_train, self.all_neg_test = train_test_split(samped_neg , test_size=20, random_state=42) #there are few purity words, but more on the neg side
            
            self.pos_train= clean_words(self.all_pos_train, self.w2vmodel, returnNA=False) #among sampled words, clean these. This is done after the sampling so that, as much as possible, across different models the same training/testing words will be used. 
            self.neg_train= clean_words(self.all_neg_train, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
            self.pos_test= clean_words(self.all_pos_test, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
            self.neg_test= clean_words(self.all_neg_test, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
                  
        elif self.direction_of_interest=='moral_mfd':  #this is different than gender/health/ses since drawn from moral foundations theory lexicon
            self.pos_label="moral"
            self.neg_label="immoral"
            self.all_pos_train, self.all_pos_test = train_test_split(word_lists.moral_mfd['pos'] , test_size=30, random_state=42) 
            self.all_neg_train, self.all_neg_test = train_test_split(word_lists.moral_mfd['neg'] , test_size=30, random_state=42) 
            
            self.pos_train= clean_words(self.all_pos_train, self.w2vmodel, returnNA=False) #among sampled words, clean these. This is done after the sampling so that, as much as possible, across different models the same training/testing words will be used. 
            self.neg_train= clean_words(self.all_neg_train, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
            
            self.pos_test= clean_words(self.all_pos_test, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
            self.neg_test= clean_words(self.all_neg_test, self.w2vmodel, returnNA=False) #there are fewer pos words than neg words
            

