# Word2Vec-bias-extraction

*Notebook to accompany paper in preparation by Arseniev-Koehler and Foster. Code written in Python 3 in Windows. Please do not cite or reuse this code yet. This code is still in preparation and may contain errors.*

This project explores **how language in the news is loaded with meanings of gender, morality, healthiness, and socio-economic status (SES)**. For example, which words are more masculine or feminine? Are certain words loaded with meanings of immorality or morality? 

We develop and then train models to classify words with respect to each of these four dimensions (gender, morality, healthiness, and SES) on a set of training words. Then, we test model performances on a fresh set of testing words. 

Finally, we look at how language about body weight, such as "obese" and "slender,"  to see how these words are connoted with gender, morality, health, and social class. You might use this code to look at meanings of language in other arenas too - such as occupations, academic disciplines, or food. You might also extend this code to other types of meaning, or to other data sources. 

We include three possible modeling frameworks in Part B, C, and D. Developing three unique models enables us to check to robustness of our assumptions and empirical findings. We also include multiple other robustness checks along the way. 

Whichever model framework you choose, we begin by feeding in a trained Word2Vec Model in the dataset we want to explore. In the Jupyter notebook for each of Part B, C and D, we suggest a pre-trained model on GoogleNews if you don't have one. OR, see [Part A](https://github.com/arsena-k/Word2Vec-bias-extraction) of this project for a tutorial on training your own Word2Vec model. If you are unfamiliar with text analysis or Word2Vec, start with Part A! 

## Part A:  Modeling Language with Word2Vec: Train a Word2Vec Model
Files needed from this repo:
* Part_A_W2V_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

**Description:** This code is for training a Word2Vec model using Gensim, including suggested hyperparameters. Code is also included for evaluating model quality on the Google Analogy Test. Some suggested ways to explore the model are also included. Word2Vec models words in a text dataset as numeric vectors. For a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). 

## Part B: Try a Geometric Approach to measure bias inspired by Bolukbasi et al 2016
Files needed from this repo:
* Part_B_Bolukbasi_W2V_Dimension_Extraction.ipynb

**Description:** ""Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a bipolar concept such as gender which is often portrayed as ranging continuosly from hypermasculine to hyperfeminine, or socioeconomic status (SES) which ranges from poor to rich. 

This notebook explores how Word2Vec model of language **learns words with respect to these dimensions.** We classify a word-vector by first finding the dimension (like gender), and then projecting the word-vector onto the vector representing the dimension. We get a scalar that corresponds to how far the word-vector lies at one end of the dimension or another (e.g., how masculine or feminine the word-vector is).
* To use gender as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative projection of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec learned this word as gender-neutral. 
* Dimensions in this juypter notebook that are ready for extraction are **gender, morality, health,** and **ses**. Code is modifiable to extract other dimensions that may be interesting: you will need to adjust the training/testing words and corresponding labels. 
* This code is written for use with Word2Vec models, modifiable for other word-vector models as well. 

Using methods inspired to detect gender biases by [Bolukbasi et. al. 2016](https://arxiv.org/abs/1607.06520).


## Part C: Try another Geometric Approach 
Files needed from this repo:

* Part_C_Larsen_W2V_Dimension_Extraction.ipynb

**Description:** This modifies the methods used in Part B (which were proposed by Bolukbasi et al. 2016) to extract dimensions a slightly different way. This method is more flexible to a variety of training-words to extract directions, than that used in Part B. Please see Part B for motivation and explanation. 

Using methods inspired by [Larsen et. al. 2016](https://arxiv.org/abs/1512.09300?context=cs)

## Part D: Try a Machine-Learning Classifier instead of a Geometic Approch
Files needed from this repo:

* SVM, Dec Tree ExperimentClassification.ipynb
* Classification Alternate Strategy- Extract WordVectors for ML and Neural Net Experimenting.ipynp

**Description:** This is code for a totally different way to extract dimensions and measure the biases learned by a Word2Vec model, now using a machine-learning classifier rather than a geometric approach. The disadvantage to this approach compared to those in Part B and Part C is that machine-learning models tend to be way overparametrized for this task, since there are few training examples (word-vectors) compared to the number of features for each word-vector (dimensions in a Word2Vec models tend to range from 50-500). Still, it is a way to check biases results in Part B and C. Please see Part B for additional motivation and explanation. 



