# Word2Vec-bias-extraction

*GitHub repository to accompany research paper in preparation by Alina Arseniev-Koehler and Jacob G. Foster, "Machine learning as a model for cultural learning: teaching an algorithm what it means to be fat." Preprint available at: https://osf.io/preprints/socarxiv/c9yj3/ Please cite this repository or paper if reused. Code written in Python 3 in Windows.* 

This research examines how **news reporting on body weight and health is loaded with meanings of gender, morality, health, and socio-economic status (SES)**. For example, is "overweight" more masculine or feminine? What about "slender," and "obese"? As described in our paper, qualtiative work suggests that obesity connotes immorality, unhealthiness, low SES, and is considered more problematic for women than for men. 

**Paper Abstract:** Overweight individuals in the U.S. are disparaged as effeminate, immoral, and unhealthy. These negative conceptions are not intrinsic to obesity; rather, they are the tainted fruit of cultural learning. Scholars often cite media consumption as a key mechanism for learning cultural biases, but have not offered a formal theory of how this public culture becomes private culture. Here we provide a computational account of this learning mechanism, showing that cultural schemata can be learned from news reporting. We extract schemata about obesity from *New York Times* articles with Word2Vec, a neural language model inspired by human cognition. We identify several cultural schemata around obesity, linking it to femininity, immorality, poor health, and low socioeconomic class. Such schemata may be subtly but pervasively activated by our language; thus, language can chronically reproduce biases (e.g., about body weight and health). Our findings also reinforce ongoing concerns that machine learning can encode, and reproduce, harmful human biases.

**Code in this repository shows our methods** to train Word2Vec models (Part A) and then classify words (Part B) with respect to each of these four cultural dimensions (gender, morality, health, and SES), and check robustness of results. 

## Part A:  Modeling Language with Word2Vec: Train and Explore a Word2Vec Model

Files needed from this repo:
* Part_A_W2V_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

Files needed from [OSF](https://osf.io/jvarx/files/):
* modelA_ALLYEARS_500dim_10CW

**Description:** In Part A, we train a Word2Vec model of text data using Gensim in Python 3.  Word2Vec takes words (or phrases) in a text data set and models these as n-dimensional numeric vectors. With enough data, these vectors are meaningful; for example, words taht are mroe similar in meaning (like man and boy) will have similar vectors. For a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). While illustrate in this notebook how to train a Word2Vec model on public text data, and show how to use Gensim to work with pretrained Word2Vec models and evaluate their quality (such as our models and a publicly available model trained on GoogleNews). 

## Part B: Measuring the connotations of obesity in news discourse, with respect to gender, morality, health and social class

Files needed from this repo: 
* Part_B_Connotations
* build_lexicon.py
* word_lists.py
* dimension.py
* semantic_svm.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_500dim_10CW
 * modelA_ALLYEARS_300dim_10CW

**Description:**  This notebook illustrates the cultural connotations of words about obesity, using Word2Vec models trained on New York Times. In brief, we use geometric methods to first find a vector correspodning to each sematnic direction (gender, morality, health, and social class). We operationalize morality as "purity," in the code.  Upon extracting a dimension, we then project a target word-vector (e.g., "obese") onto the dimension. Then, we take the cosine similarity between our words about obesity and a given dimension (e.g., the cosine similarity beween the "gender" direction and the word-vector for "overweight."). This yields a scalar which tells us the direction and magnitude of the connotation - e.g., whether "overweight" connotes masculinity or femininity, and how much masculinity or femininity. Since one Word2Vec model may be instable, in our paper, we report results based on findings from multiple Word2Vec models trained on the same data. See our paper for details on thse methods. 

We also include code to classify words' connotations by using a machine-learning classifier (a Support Vector Machine). The "features" of a word in this classifcation task are simply the loading on each of the dimensions. The disadvantage to this approach compared to the geometric approaches in Part B is that machine-learning classifiers tend to be way overparametrized for this task, since there are few training examples (e.g., words that correspond to femininity or masculinity) compared to the number of features (dimensions) for each corresponding word-vector (dimensions in a Word2Vec models tend to range from 25-500). We use a Word2Vec model trained with 300 dimensions in this part of our methods, to help address this. 



