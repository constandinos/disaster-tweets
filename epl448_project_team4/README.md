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

## Authors
Andreas Tsouloupas  
Constandinos Demetriou    
George Hadjiantonis