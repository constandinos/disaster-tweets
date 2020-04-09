"""
reference: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
"""

# To install gensim for anaconda run:
# conda install -c anaconda gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd

##### FUNCTIONS SECTION #####
def train_doc2vec_model(df, text, is_text_column = True):
    """
    This function will train a Doc2vec model with the given document's content
    
    Parameters
    ----------
    df : DataFrame
        Documents' dataframe which contains the text and the target for each
        document.
    text : List of strings or string
        If is_text_column is True then the argument must be a string which
        inticates which column of the dataframe contains the text of the docs.
        If is_text_column is False then the argument must be a list of strings
        which are the content of each document.
    is_text_column: bool
        See description for parameter text.

    Returns
    -------
    d2v_model : Gensim Doc2vec model
        This model is trained with the given documents' content.
    """
    ## Tag corpus data for training - doc2vec format
    if (is_text_column):
        corpus_tuple = tuple(df[text])
    else:
        corpus_tuple = tuple(text)
    #print(corpus_tuple)

    tagged_corpus_list = list()
    for i, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ') #gensim.utils.simple_preprocess(corpus)
        
        # Add tags for training data
        tagged_corpus_list.append(TaggedDocument(tokens, [i]))


    ## Train doc2vec model
    vector_size = 50 #  Dimensionality of the feature vectors.
    min_count = 2 # Ignores all words with total frequency lower than this.
    dm = 0  # Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. 
            # Otherwise, distributed bag of words (PV-DBOW) is employed.
    epochs = 100 # Number of iterations (epochs) over the corpus.
    alpha = 0.025 # The initial learning rate.

    d2v_model = Doc2Vec(vector_size = vector_size,\
                        min_count = min_count,\
                        dm = dm,\
                        epochs = epochs,\
                        alpha = alpha,\
                        min_alpha=0.0025) # use fixed learning rate
    
    d2v_model.build_vocab(tagged_corpus_list)

    # Produces better results if you repeat the procedure and
    # each time decreasing the learning rate
 
    d2v_model.train(tagged_corpus_list,\
                    total_examples = d2v_model.corpus_count,\
                    epochs = d2v_model.epochs)

    return d2v_model


def doc2vec_embedding_testset(model, df, text, is_text_column = True):
    """
    This function will create features for the given documents by using
    the given doc2vec pre-trained model
    
    Parameters
    ----------
    model : Gensim Doc2vec model
        Pre-trained Doc2vec model
    df : DataFrame
        Documents' dataframe which contains the text and the target for each
        document.
    text : List of strings or string
        If is_text_column is True then the argument must be a string which
        inticates which column of the dataframe contains the text of the docs.
        If is_text_column is False then the argument must be a list of strings
        which are the content of each document.
    is_text_column: bool
        See description for parameter text.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    """
    ## doc2vec format - to infer features vector
    if (is_text_column):
        corpus_tuple = tuple(df[text])
    else:
        corpus_tuple = tuple(text)
    #print(corpus_tuple)

    corpus_d2v_format = list()
    for i, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ') #gensim.utils.simple_preprocess(corpus)
        corpus_d2v_format.append(tokens)


    features = [model.infer_vector(corpus_d2v_format[i]) for i in range(len(corpus_d2v_format))]
    features = np.array([np.array(x) for x in features])

    return features


def doc2vec_embedding_trainset(model, df, text, target, is_text_column = True):
    """
    This function will create features for the given documents by using
    the given doc2vec pre-trained model
    
    Parameters
    ----------
    model : Gensim Doc2vec model
        Pre-trained Doc2vec model
    df : DataFrame
        Documents' dataframe which contains the text and the target for each
        document.
    text : List of strings or string
        If is_text_column is True then the argument must be a string which
        inticates which column of the dataframe contains the text of the docs.
        If is_text_column is False then the argument must be a list of strings
        which are the content of each document.
    target: string
        Inticates which column of the dataframe contains the label of the docs.
    is_text_column: bool
        See description for parameter text.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    labels : pandas Series
        Labels for each document
    """
    features = doc2vec_embedding_testset(model, df, text, is_text_column)

    # Labels of train dataset
    labels = df[target]


    return features, labels


def doc2vec_evaluate_success_on_train_data(transformed_data, model):
    """
    This function will evaluate a doc2vec model. The evaluation
    will be done with the vectors of the training data that 
    were produced from the given model.
    """
    ranks = []
    for doc_id in range(len(transformed_data)):
        inferred_vector = transformed_data[doc_id]

        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    counter = 0
    for x in ranks:
        if x == 0:
            counter += 1

    print('Documents most similar to themselfs',str(counter),'out of',str(len(transformed_data)))


##### END OF FUNCTIONS SECTION #####


## Import dataset
tweet_df = pd.read_csv('../dataset/train.csv')
print("Number of tweets, features:",tweet_df.shape)

model = train_doc2vec_model(tweet_df, 'text')
emb_features, labels = doc2vec_embedding_trainset(model, tweet_df, 'text', 'target')


## Test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train_features, test_features, train_labels, test_labels = train_test_split(emb_features, labels)


## Searching for the best value of the C parameter, which determines regularization strength
#from sklearn.model_selection import GridSearchCV
#parameters = {'C': np.linspace(0.0001, 100, 20)}
#grid_search = GridSearchCV(LogisticRegression(), parameters)
#grid_search.fit(train_features, train_labels)

#print('best parameters: ', grid_search.best_params_)
#print('best scrores: ', grid_search.best_score_)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

#print(lr_clf.score(test_features, test_labels))

# Get the predictions for the test data
y_pred = lr_clf.predict(test_features)

# Print classification reports for Linear Regression
from sklearn.metrics import classification_report
print("Linear Regression Classifier:")
print(classification_report(test_labels, y_pred))


#doc2vec_evaluate_success_on_train_data(emb_features, model)