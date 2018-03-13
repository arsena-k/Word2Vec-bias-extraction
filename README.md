# Word2Vec-bias-extraction

All code written with Python 3 on Windows Machine, and checked as of 3/13/2018. In progress. 


## Part 1:  Train Word2Vec Model, Select Hyperparameters, Evaluate Performance on Google Analogy Test, and Explore Models
Files needed from this repo:
* Word2Vec_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

## Part 2A: Try a Geometric Approach to measure inspired by Bolukbasi et al 2016
Files needed from this repo:
* Geometrically_Extract_Dimensions_with_Word2Vec.ipynb

"Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a bipolar concept such as "gender" which ranges continously from hypermasculine to hyperfeminine, or "socioeconomic status" which ranges from poor to rich. 

We can **see how "biased" a word is learned by Word2Vec by** projecting it onto an extracted dimension. We'll get a scalar that corresponds to the learned bias. 
* To use gender bias as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative proejction of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec learned this word as gender-neutral. 

* Dimensions in this juypter notebook that are ready for extraction are **gender, morality, health,** and **ses**. Code is easily modifiable to extract other dimensions that may be interesting, you will need to adjust the training/testing words and corresponding labels. 
* This code is written for use with Word2Vec models, easily modifiable for other word-vector models as well. 

Methods are inspired by Bolukbasi et. al. 2016 (https://arxiv.org/abs/1607.06520) which quantifies gender biases in Word2Vec models. 



## Part 2B: Try another Geometric Approach (a slight twist)  to measure bias
Files needed from this repo:
* 




## Part 2C: Try a Machine-Learning Classifier instead of a Geometic approch to measure bias
Files needed from this repo:
* 
