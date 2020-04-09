import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

##### FUNCTIONS SECTION #####
def get_vectorizer(train_data, tf_idf = True, analyzer = 'word', ngram_range = (1,1),\
                    max_df = 1.0, min_df = 1, max_features = None, norm = 'l2'):
    """
    This function trains a TfidfVectorizer or CountVectorizer with the given
    training data.
    
    Parameters
    ----------
    train_data : List of strings 
        Content of the docuemnts
    tf_idf: bool
        If true then it will return TfidfVectorizer otherwise CountVectorizer
    For the remaining arguments you can find more information in the documentations
    of sklearn for TfidfVectorizer and CountVectorizer

    Returns
    -------
    transformed_data : numpy array
        This array contains the created features for each document for the training set
    vectorizer : TfidVectorizer or CountVectorizer
    """
    if (tf_idf):
        # initialize tf-idf vectorizer
        vectorizer = TfidfVectorizer(analyzer = 'word',\
                                        max_df = max_df,\
                                        min_df = min_df,\
                                        ngram_range = ngram_range,\
                                        max_features = max_features,\
                                        norm = norm)

        # train vectorizer and transform the given data
        transformed_data = vectorizer.fit_transform(train_data)
    else:
        # initialize bag of words vectorizer
        vectorizer = CountVectorizer(analyzer = analyzer,\
                                    max_df = max_df,\
                                    min_df = min_df,\
                                    ngram_range = ngram_range,\
                                    max_features = max_features)

        transformed_data = vectorizer.fit_transform(train_data)
    
    transformed_data.toarray()
    return transformed_data, vectorizer

##### END OF FUNCTIONS SECTION #####


tweet_df = pd.read_csv('../dataset/train.csv')
print("Number of tweets, features: ",tweet_df.shape)

tweets_text = list(tweet_df['text'])
labels = tweet_df['target']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
def lr(emb_features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(emb_features, labels)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    #print(lr_clf.score(test_features, test_labels))

    # Get the predictions for the test data
    y_pred = lr_clf.predict(test_features)

    # Print classification reports for Linear Regression
    
    print("Linear Regression Classifier:")
    print(classification_report(test_labels, y_pred))

"""  TF-IDF """
# TF-IDF words
features, vectorizer_tfidf_words = get_vectorizer(tweets_text)
#print(features)
#print(features.shape)
# Vocabulary words
#vocabulary_tfidf_words = vectorizer_tfidf_words.get_feature_names()
lr(features,labels)

# TF-IDF chars
features, vectorizer_tfidf_chars = get_vectorizer(tweets_text,True,'char',(5,5))
#print(features)
#print(features.shape)
# Vocabulary chars
#vocabulary_tfidf_chars = vectorizer_tfidf_chars.get_feature_names()
lr(features,labels)

""" BAG OF WORDS """
features, vectorizer_bow = get_vectorizer(tweets_text,False)
#print(features)
#print(features.shape)
# Vocabulary words
#vocabulary_bow = vectorizer_bow.get_feature_names()
lr(features,labels)
