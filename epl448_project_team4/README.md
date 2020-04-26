# Real or Not? NLP with Disaster Tweets
Related code and data for [this Kaggle competition](https://www.kaggle.com/c/nlp-getting-started).

## Competition Description
Twitter has become an important communication channel in times of emergency.  

The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

### Challenge
Build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

## Objectives
+ apply several data analysis methodologies to understand well the given data
+ apply various NLP preprcessing techniques and examine how they affect model performance
+ experiment with several machine learning models and improve overall score

## Data analysis
Some results (plots) of the data analysis process can be found in directory [graphs](https://github.com/constandinos/disaster-tweets/tree/master/graphs).

## Preprocessing
Details related to the applied preprocessing can be found [here](./dataset/README.md).

We have also tested applying preprocessing using only the library [ekphrasis](https://github.com/cbaziotis/ekphrasis), which is a text processing tool, geared towards text from social networks and compare our results.

## Text Data Vectorization
Methods tested for data vectorization:
+ DistilBERT
+ gensim doc2vec
+ TF-IDF
+ Bag of words

## ML
Machine learning models tested:
+ Logistic Regression
+ K-nearest neighbors
+ SVC
+ Random Forest
+ Decision Trees

## Requirements
In **Anaconda enviroment**

### 1. Data preprocessing `tweet_preprocessing.py`
```
pip install emot nltk requests inflect pyspellchecker ekphrasis
```

### 3. Feature creation `feature_creation.py`
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge transformers
conda install -c anaconda gensim
```

### 4. Keras `keras_nn.py`
To install the current release of GPU TensorFlow
```
conda create -n tf-gpu tensorflow-gpu
conda activate tf-gpu
conda install --name tf-gpu pylint -y
```
To install the current release of CPU-only TensorFlow
```
conda create -n tf tensorflow
conda activate tf
conda install --name tf pylint -y
```
To install tensorflow this package with conda run one of the following:
```
conda install -c conda-forge tensorflow-hub
conda install -c conda-forge/label/gcc7 tensorflow-hub
conda install -c conda-forge/label/cf201901 tensorflow-hub
conda install -c conda-forge/label/cf202003 tensorflow-hub
```
To install tokenization package with conda run: (only for Linux :/)
```
conda install -c powerai sentencepiece
```
To install pandas package with conda run:
```
conda install -c anaconda pandas
```
## Run (in the following order)
1. `visualization.py`
2. `tweet_preprocessing.py` and `dropduplicates.py`
3. `experiments.py`
4. `final_model_score.py`
5. `keras_nn.py` (optional for Kaggle's results)

## Authors
Andreas Tsouloupas  
Constandinos Demetriou    
George Hadjiantonis
