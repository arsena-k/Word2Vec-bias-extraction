# Word2Vec-bias-extraction

*GitHub repository to accompany research paper in preparation by Alina Arseniev-Koehler and Jacob Foster, "Teaching an algorithm what it means to be fat: machine-learning as a model for cultural learning." Please cite this repository or paper if reused. Code written in Python 3 in Windows.* 

This research examines how **news reporting on body weight and health is loaded with meanings of gender, morality, health, and socio-economic status (SES)**. For example, is "overweight" more masculine or feminine? What about "obese," and "obese"? As described in our paper, qualtiative work suggests that obesity connotes immorality, unhealthiness, low SES, and is considered more problematic for women than for men. 

**Paper Abstract:** Overweight individuals in the U.S. are disparaged as effeminate, immoral, and unhealthy. These negative conceptions are not intrinsic to obesity; rather, they are the tainted fruit of cultural learning. Scholars often cite media consumption as a key mechanism for learning cultural biases, but have not offered a formal theory of how this public culture becomes private culture. Here we provide a computational account of this learning mechanism, showing that cultural schemata can be learned from news reporting. We extract schemata about obesity from \textit{New York Times} articles with Word2Vec, a neural language model inspired by human cognition. We identify several cultural schemata around obesity, linking it to femininity, immorality, poor health, and low socioeconomic class. Such schemata may be subtly but pervasively activated by our language; thus, language can chronically reproduce biases (e.g., about body weight and health). Our findings also reinforce ongoing concerns that machine learning can encode, and reproduce, harmful human biases.

**Code in this repository shows our methods** to train Word2Vec models and then classify words with respect to each of these four cultural dimensions (gender, morality, health, and SES). For each dimension, we do this classification first on a training set of words. Then, we upon checking the robustness of our methods, we classify a fresh set of testing words. Finally, we examine language about body weight, such as "obese" and "slender,"  to see how obesity-related lexicon is connoted with gender, morality, health, and SES. 

We include three possible methods for classifying words. Two of these methods, Bolukbasi and Larsen methods, are both geometric approaches and are covered in Part B. The third method trains a machine-learning classifier (Support Vector Machine) and is covered in Part C. Developing three unique methods enables us to check the robustness of our assumptions and empirical findings. We also include multiple other robustness checks along the way. 

For all three methods, we start from a Word2Vec model trained New York Times articles about health and body weight. In the Jupyter notebooks, we also suggest a pre-trained model on GoogleNews. We find that the conclcusions in our research are largely the same whether we use a model trained on GoogleNews or the New York Times. See [Part A](https://github.com/arsena-k/Word2Vec-bias-extraction) of this project for a tutorial on training and exploring your own Word2Vec model. If you are unfamiliar with text analysis or Word2Vec, start with Part A. 

## Part A:  Modeling Language with Word2Vec: Train and Explore a Word2Vec Model
Files needed from this repo:
* Part_A_W2V_training_performance_exploring.ipynb
* testing.py
* questions_words_pasted.txt
* questions_words.txt

Files needed from [OSF](https://osf.io/jvarx/files/):
* modelA_ALLYEARS_500dim_10CW

**Description:** In Part A, we train a Word2Vec model of text data using Gensim in Python 3.  Word2Vec takes words (or phrases) in a text data set and models these as n-dimensional numeric vectors. With enough data, these vectors are meaningful; for example, words taht are mroe similar in meaning (like man and boy) will have similar vectors. For a review of Word2Vec check out this [blog post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). We also include code in this juputer notebook to evaluate model quality on the Google Analogy Test on any trained Word2Vec model, and explore any trained Word2Vec model.

## Part B: Two geometric approaches to Classify Words, inspired by [Bolukbasi et. al. 2016](https://arxiv.org/abs/1607.06520) and [Larsen et. al. 2016](https://arxiv.org/abs/1512.09300?context=cs)

Files needed from this repo:
* Part_B_BolukbasiLarsen_W2V_Dimension_Extraction.ipynb
* helpers_partB.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_500dim_10CW


**Description:** "Dimension," "direction" and "subspace" are used in this notebook interchangeably to refer to a vector that captures a binary construct such as gender which is often portrayed as ranging continuously from hypermasculine to hyperfeminine, or socioeconomic status (SES) which ranges from poor to rich.

This notebook explores how a Word2Vec model of language **learns words with respect to these dimensions.** To classify a word-vector according to a binary construct, we first find a dimension in a trained Word2Vec model that correponds to the binary construct (e.g., finding a dimension corresponding to gender). Then, we project the word-vector onto this dimension. This projection gives a scalar corresponding to how far the word-vector lies at one end of the dimension or another (e.g., how masculine or feminine the word-vector is). The Bolukbasi and Larsen methods use two different ways to extract a dimension more robustly, as explained in the Jupyter Notebook and in our paper.
* To use gender as an example - a larger, positive projection of a word onto the gender dimension suggests this word is learned as highly feminine, while a larger, negative projection of a word onto the gender dimension suggests that this word is learned as highly masculine. A word with a projection near zero suggests that Word2Vec did not learn that this word carries gendered information.
* In this Jupyter Notebook, we extract dimensions corresponding to **gender, morality, health,** and **ses**. 


## Part C: Machine-Learning Classifier Approach to Classify Words (a Support Vector Machine)
Files needed from this repo:
* Part_C_MachineLearning_W2V_Dimension_Extraction.ipynb
* helpers_partC.py

Files needed from [OSF](https://osf.io/jvarx/files/):
 * modelA_ALLYEARS_300dim_10CW     *note this model is different (lower-dimensional word-vectors) from the one used on Parts A and B

**Description:** This is code for a quite different way to classify words which were learned by a Word2Vec model. In this approach, we train a machine-learning classifier (a Support Vector Machine) to classify word-vectors directly, rather than use a geometric approach to first extract a dimension. The "features" of a word in this classifcation task are simply the loading on each of the dimensions. The disadvantage to this approach compared to the geometric approaches in Part B is that machine-learning classifiers tend to be way overparametrized for this task, since there are few training examples (e.g., words that correspond to femininity or masculinity) compared to the number of features (dimensions) for each corresponding word-vector (dimensions in a Word2Vec models tend to range from 50-500). We use a Word2Vec model trained with 300 dimensions in this part of our methods, to help address this. See Part B for additional motivation and explanations. 



