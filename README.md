# Word2Vec-bias-extraction

All code written with Python 3 on Windows. Some changes to Gensim Word2Vec since then that need to be updated. IN PROGRESS PLEASE DO NOT CITE. 

## Part A:  Train Word2Vec Model, Evaluate Performance and Explore
Files needed from this repo:
* Word2Vec_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

**Description:** This code is for training a Word2Vec model using Gensim, including suggested hyperparamters. Code is also included for evaluating model quality on the Google Analogy Test. Some suggested ways to explore the model are also included. 

## Part B: Try a Geometric Approach to measure bias inspired by Bolukbasi et al 2016
Files needed from this repo:
* Geometrically_Extract_Dimensions_with_Word2Vec.ipynb

**Description:** ""Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a bipolar concept such as gender which is often portrayed as ranging continuosly from hypermasculine to hyperfeminine, or socioeconomic status (SES) which ranges from poor to rich. 

This notebook explores how a language model (Word2Vec) **learns words with respect to these dimensions.** Word2Vec models words as numeric vectors, for a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We classify a word-vector by first finding the dimension (like gender), and then projecting the word-vector onto the vector representing the dimension. We get a scalar that corresponds to how far the word-vector lies at one end of the dimension or another (e.g., how masculine or feminine the word-vector is).
* To use gender as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative projection of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec learned this word as gender-neutral. 
* Dimensions in this juypter notebook that are ready for extraction are **gender, morality, health,** and **ses**. Code is modifiable to extract other dimensions that may be interesting: you will need to adjust the training/testing words and corresponding labels. 
* This code is written for use with Word2Vec models, modifiable for other word-vector models as well. 

For the original application of these methods to gender biases see [Bolukbasi et. al. 2016](https://arxiv.org/abs/1607.06520).



## Part C: Try another Geometric Approach 
Files needed from this repo:

* Part_C_Larsen_W2V_Dimension_Extraction.ipynb

**Description:** This modifies the methods used in Part B (which were proposed by Bolukbasi et al. 2016) to extract dimensions a slightly different way. This method is more flexible to a variety of training-words to extract directions, than that used in Part B. Please see Part B for motivation and explanation. 


## Part D: Try a Machine-Learning Classifier instead of a Geometic approch to measure bias
Files needed from this repo:

* SVM, Dec Tree ExperimentClassification.ipynb
* Classification Alternate Strategy- Extract WordVectors for ML and Neural Net Experimenting.ipynp

**Description:** This is code for a totally different way to measure the biases learned by a Word2Vec model, now using a machine-learning classifier rather than a geometric approach. The disadvantage to this appraoch compared to those in Part B and Part C is that machine-learning models tend to be way overparametrized, since there are few training examples (words-vectors) compared to the number of features for each word-vector (dimensions in a Word2Vec models tend to range from 50-500). Still, it is a way to check biases results in Part B and C. Please see Part B for motivation and explanation. 



