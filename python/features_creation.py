import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

##### FUNCTIONS SECTION #####

def tfidf_words(train_data):
    """
    This function
    Returns:
        transformed_data:
        vectorizer:
    """
    # initialize tf-idf vectorizer with the following parameters
    vectorizer = TfidfVectorizer(analyzer = 'word',\
                                    max_df = 1.0,\
                                    min_df = 1,\
                                    ngram_range = (1,1),\
                                    max_features = None,\
                                    norm = 'l2')

    transformed_data = vectorizer.fit_transform(train_data)
    return transformed_data, vectorizer


def tfidf_chars(train_data, ngram_min, ngram_max):
    """
    This function
    Returns:
        transformed_data:
        vectorizer:
        ngram_min:
        ngram_max:
    """
    vectorizer = TfidfVectorizer(analyzer = 'char',\
                                    max_df = 1.0,\
                                    min_df = 1,\
                                    ngram_range = (ngram_min,ngram_max),\
                                    max_features = None,\
                                    norm = 'l2')

    transformed_data = vectorizer.fit_transform(train_data)
    return transformed_data, vectorizer


def bag_of_words(train_data):
    """
    This function
    Returns:
        transformed_data:
        vectorizer:
    """
    vectorizer = CountVectorizer(analyzer = 'word',\
                                    max_df = 1.0,\
                                    min_df = 1,\
                                    ngram_range = (1,1),\
                                    max_features = None)

    transformed_data = vectorizer.fit_transform(train_data)
    return transformed_data, vectorizer

##### END OF FUNCTIONS SECTION #####


tweet_df = pd.read_csv('../dataset/train.csv')
test_df =pd.read_csv('../dataset/test.csv')
print("Number of tweets, features: ",tweet_df.shape)

tweets_text = list(tweet_df['text'])

"""  TF-IDF """
# TF-IDF words
transformed_data_tfidf_words, vectorizer_tfidf_words = tfidf_words(tweets_text)
transformed_data_tfidf_words = transformed_data_tfidf_words.toarray()
print(transformed_data_tfidf_words)
print(transformed_data_tfidf_words.shape)
# Vocabulary words
vocabulary_tfidf_words = vectorizer_tfidf_words.get_feature_names()

# TF-IDF chars
transformed_data_tfidf_chars, vectorizer_tfidf_chars = tfidf_chars(tweets_text,5,5)
transformed_data_tfidf_chars = transformed_data_tfidf_chars.toarray()
print(transformed_data_tfidf_chars)
print(transformed_data_tfidf_chars.shape)
# Vocabulary chars
vocabulary_tfidf_chars = vectorizer_tfidf_chars.get_feature_names()


""" BAG OF WORDS """
transformed_data_bow, vectorizer_bow = bag_of_words(tweets_text)
transformed_data_bow = transformed_data_bow.toarray()
print(transformed_data_bow)
print(transformed_data_bow.shape)
# Vocabulary words
vocabulary_bow = vectorizer_bow.get_feature_names()

