# Word2Vec-bias-extraction

*GitHub repository to accompany research article in preparation by Arseniev-Koehler and Foster, "Teaching an algorithm what it means to be fat: machine-learning as a model for cultural learning." Please cite this repository or paper if reused. Code written in Python 3 in Windows.* 

This research examines how **news reporting on body weight and health is loaded with meanings of gender, morality, health, and socio-economic status (SES)**. For example, is "overweight" more masculine or feminine? What about "obese," and "obese"? As described in our paper, qualtiative work suggests that obesity connotes immorality, unhealthiness, low SES, and is considered more problematic for women than for men. 

Code in this repository shows our methods to classify words with respect to each of these four dimensions (gender, morality, healthiness, and SES). We do so first on a training set, and then, we upon refining our methods do so on a fresh set of testing words. 
Finally, we examine language about body weight, such as "obese" and "slender,"  to see how obesity-related lexicon is connoted with gender, morality, health, and social class. 

We include three possible methods for classifying words. Two of these methods, Bolukbasi and Larsen methods, are both geometric approaches and are covered in Part B. The third method uses a machine-learning classifier (Support Vector Machine) and is covered in Part C. Developing three unique methods enables us to check to robustness of our assumptions and empirical findings. We also include multiple other robustness checks along the way. 

For all three methods, we feed in a Word2Vec model trained New York Times articles about health and body weight. In the Jupyter notebooks, we also suggest a pre-trained model on GoogleNews. We find that the conclcusions in our research are largely the same whether we use a model trained on GoogleNews or the New York Times. See [Part A](https://github.com/arsena-k/Word2Vec-bias-extraction) of this project for a tutorial on training and exploring your own Word2Vec model. If you are unfamiliar with text analysis or Word2Vec, start with Part A. 

## Part A:  Modeling Language with Word2Vec: Train and Explore a Word2Vec Model
Files needed from this repo:
* Part_A_W2V_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

Files needed from [OSF](https://osf.io/jvarx/files/):
* modelA_ALLYEARS_500dim_10CW

**Description:** In Part A, we train a Word2Vec model of text data using Gensim.  Word2Vec takes words (or phrases) in a text data set and models these as n-dimensional numeric vectors. With enough data, these vectors are meaningful; for example, words taht are mroe similar in meaning (like man and boy) will have similar vectors. For a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We also include code in this juputer notebook to evaluate model quality on the Google Analogy Test on any trained Word2Vec model, and explore any trained Word2Vec model.

## Part B: Try two geometric approaches to classify words, inspired by [Bolukbasi et. al. 2016](https://arxiv.org/abs/1607.06520) and [Larsen et. al. 2016](https://arxiv.org/abs/1512.09300?context=cs)

Files needed from this repo:
* Part_B_BolukbasiLarsen_W2V_Dimension_Extraction.ipynb
* helpers_partB.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_500dim_10CW


**Description:** "Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a binary construct such as gender which is often portrayed as ranging continuously from hypermasculine to hyperfeminine, or socioeconomic status (SES) which ranges from poor to rich.

This notebook explores how a Word2Vec model of language **learns words with respect to these dimensions.** To classify a word-vector according to a binary construct, we first find a dimension in a trained Word2Vec model that correponds to the binary construct (e.g., finding a dimension corresponding to gender). Then, we project the word-vector onto this dimension. This projection gives a scalar corresponding to how far the word-vector lies at one end of the dimension or another (e.g., how masculine or feminine the word-vector is). The Bolukbasi and Larsen methods are two different ways to extract a dimension more robustly, as explained in the Jupyter Notebook and in our paper.
* To use gender as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative projection of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec learned this word as gender-neutral. 
* Dimensions in this juypter notebook that are ready for extraction are **gender, morality, health,** and **ses**. 


## Part C: Try a Machine-Learning Classifier Approach (a Support Vector Machine)
Files needed from this repo:
* Part_C_MachineLearning_W2V_Dimension_Extraction.ipynb
* helpers_partC.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_300dim_10CW     *note this model is different (lower-dimensional word-vectors) from the one used on Part A, B and C

**Description:** This is code for a quite different way to classify words which were learned by a Word2Vec model. In this approach, we use a machine-learning classifier (a Support Vector Machine) rather than a geometric approach. The disadvantage to this approach compared to those in Part B is that machine-learning classifiers tend to be way overparametrized for this task, since there are few training examples (word-vectors) compared to the number of features for each word-vector (dimensions in a Word2Vec models tend to range from 50-500). Still, it is a way to check biases results in Part B. Please see Part B for additional motivation and explanations. 



