# Word2Vec-bias-extraction

*GitHub repository to accompany paper in preparation by Arseniev-Koehler and Foster, "Teaching an algorithm what it means to be fat: machine-learning as a model for cultural learning." Please cite this repository or paper if reused. Code written in Python 3 in Windows. 

This project explores **how language in the news is loaded with meanings of gender, morality, healthiness, and socio-economic status (SES)**. For example, which words are more masculine or feminine? Are certain words loaded with meanings of immorality or morality? 

We develop and then train models to classify words with respect to each of these four dimensions (gender, morality, healthiness, and SES) on a set of training words. Then, we test model performances on a fresh set of testing words. 

Finally, we examine language about body weight, such as "obese" and "slender,"  to see how obesity-related lexicon is connoted with gender, morality, health, and social class. 

We include three possible modeling frameworks for modeling these dimensions (in parts B, C, and D). Developing three unique models enables us to check to robustness of our assumptions and empirical findings. We also include multiple other robustness checks along the way. 

For all modeling frameworks, we feed in a trained Word2Vec model on New York Times (in the code, you be pointed to where to download). In the Jupyter notebook for each of Part B, C and D, we also suggest a pre-trained model on GoogleNews. See [Part A](https://github.com/arsena-k/Word2Vec-bias-extraction) of this project for a tutorial on training and exploring your own Word2Vec model. If you are unfamiliar with text analysis or Word2Vec, start with Part A. 

## Part A:  Modeling Language with Word2Vec: Train a Word2Vec Model
Files needed from this repo:
* Part_A_W2V_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

Files needed from [OSF](https://osf.io/jvarx/files/):
* modelA_ALLYEARS_500dim_10CW

**Description:** In Part A, we train a Word2Vec model of text data using Gensim.  Word2Vec models words in a text dataset as numeric vectors. For a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We also include code in this juputer notebook to evaluate model quality on the Google Analogy Test on any trained Word2Vec model, and explore any trained Word2Vec model.

## Part B: Try an approach to measure bias inspired by Bolukbasi et al 2016
Files needed from this repo:
* Part_B_Bolukbasi_W2V_Dimension_Extraction.ipynb
* helpers_partB.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_500dim_10CW


**Description:** ""Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a bipolar concept such as gender which is often portrayed as ranging continuosly from hypermasculine to hyperfeminine, or socioeconomic status (SES) which ranges from poor to rich. 

This notebook explores how Word2Vec model of language **learns words with respect to these dimensions.** We classify a word-vector by first finding the dimension (like gender), and then projecting the word-vector onto the vector representing the dimension. We get a scalar that corresponds to how far the word-vector lies at one end of the dimension or another (e.g., how masculine or feminine the word-vector is).
* To use gender as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative projection of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec learned this word as gender-neutral. 
* Dimensions in this juypter notebook that are ready for extraction are **gender, morality, health,** and **ses**. Code is modifiable to extract other dimensions that may be interesting: you will need to adjust the training/testing words and corresponding labels. 
* This code is written for use with Word2Vec models, modifiable for other word-vector models as well. 

Using methods inspired to detect gender biases by [Bolukbasi et. al. 2016](https://arxiv.org/abs/1607.06520).

## Part C: Try another approach to measure bias inspired by Larsen et al 2016
Files needed from this repo:
* Part_C_Larsen_W2V_Dimension_Extraction.ipynb
* helpers_partC.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_500dim_10CW

**Description:** This modifies the methods used in Part B (which were proposed by Bolukbasi et al. 2016) to extract dimensions a slightly different way. This method is more flexible to a variety of training-words to extract directions, than that used in Part B. Please see Part B for motivation and explanation. 

Using methods inspired by [Larsen et. al. 2016](https://arxiv.org/abs/1512.09300?context=cs)

## Part D: Try a Machine-Learning Classifier Approach (a Support Vector Machine)
Files needed from this repo:
* Part_D_MachineLearning_W2V_Dimension_Extraction.ipynb
* helpers_partD.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_300dim_10CW     *note this model is different (lower-dimensional word-vectors) from the one used on Part A, B and C

**Description:** This is code for a totally different way to extract dimensions and measure the biases learned by a Word2Vec model, now using a machine-learning classifier rather than a geometric approach. The disadvantage to this approach compared to those in Part B and Part C is that machine-learning models tend to be way overparametrized for this task, since there are few training examples (word-vectors) compared to the number of features for each word-vector (dimensions in a Word2Vec models tend to range from 50-500). Still, it is a way to check biases results in Part B and C. Please see Part B for additional motivation and explanation. 



