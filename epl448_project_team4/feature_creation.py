import numpy as np
import pandas as pd
import time


# =============================================================================#
#                              FEATURE CREATION                                #
# =============================================================================#

# =============================================================================#
#                              DistilBert / Bert                               #
# =============================================================================#

# To install this package with conda run:
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
import torch
# To install this package with conda run one of the following:
# conda install -c conda-forge transformers
# conda install -c conda-forge/label/cf202003 transformers
import transformers as tr


def bert_feature_creation(docs, distilbert=True):
    """
    Create features for each given document. This function will use one of the
    pre-trained BERT or DistilBert models. 
    Number of produced features will be equal to 768 for each document.

    Parameters
    ----------
    docs : List of strings
        Documents' content.
    distilbert: bool (default true)
        If it is True then distilbert pre-trained model will be used otherwise
        BERT pre-trained model will be used.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    """
    ## Loading a pre-trained BERT model
    # Importing pre-trained DistilBert model and tokenizer
    # DistilBERT is a small, fast, cheap and light Transformer model trained by
    # distilling Bert base. It has 40% less parameters than bert-base-uncased,
    # runs 60% faster while preserving over 95% of Bert’s performances.
    if (distilbert):
        model_class, tokenizer_class, weights = (tr.DistilBertModel, \
                                                 tr.DistilBertTokenizer, \
                                                 'distilbert-base-uncased')
    else:
        # Use BERT instead of distilBERT
        # It’s a bidirectional transformer pre-trained using a combination of masked
        # language modeling objective and next sentence prediction on a large corpus
        # comprising the Toronto Book Corpus and Wikipedia.
        model_class, tokenizer_class, weights = (tr.BertModel, \
                                                 tr.BertTokenizer, \
                                                 'bert-base-uncased')

    # Load pretrained model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(weights)
    bert_model = model_class.from_pretrained(weights)

    ## Tokenization
    # Tokenize every sentece - BERT format (list of lists)
    docs_df = pd.DataFrame(docs)
    tokenized = docs_df[0].apply((lambda x: tokenizer.encode(str(x), \
                                                             add_special_tokens=True)))
    # print(tokenized.head())

    ## Padding
    # Find the length of the longer sentence of tokens
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    # Add padding to the end of each sentence of tokens. As a result we will
    # have equal length sentences of tokens and transform them to numpy array.
    # BERT processing is faster with padding
    padded_tokens = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # print(padded_tokens,'\nPadded tokens shape:',np.array(padded_tokens).shape)

    ## Masking
    # Create a mask that tells BERT to ignore the padding we have previously
    # added to the sentences of tokens.
    # Zero(0) means ignore.
    bert_mask = np.where(padded_tokens != 0, 1, 0)
    # print('Bert mask shape:',bert_mask.shape)

    ## Running BERT model - feature creation
    padded_tokens_torch = torch.tensor(padded_tokens, dtype=torch.int64)
    bert_mask_torch = torch.tensor(bert_mask)

    # print(padded_tokens_torch)
    # print(bert_mask_torch)

    with torch.no_grad():
        hidden_states = bert_model(input_ids=padded_tokens_torch, \
                                   attention_mask=bert_mask_torch)

    # The reason we are getting only the first element that bert returns is
    # because bert adds a classification token at the first element of each
    # sentence and this is the value that we need from all the hidden layers to
    # form the embedding.
    features = hidden_states[0][:, 0, :].numpy()

    return features


# ==============================================================================#
#                                   Doc2Vec                                    #
# ==============================================================================#

# To install gensim for anaconda run:
# conda install -c anaconda gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_doc2vec_model(docs, num_features=100):
    """
    Train a gensim Doc2vec model with the content of the given documents.

    Parameters
    ----------
    docs : List of strings
        Documents' content.
    num_features: int (default 50)
        This number determines the dimensionality of the feature vectors.

    Returns
    -------
    model : Gensim Doc2vec model
        This model is trained with the content of the given documents.
    """
    ## Add Tags for each given document - required doc2vec format for training
    corpus_tuple = (str(x) for x in docs)
    # print(corpus_tuple)

    tagged_corpus_list = list()
    for i, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ')  # gensim.utils.simple_preprocess(corpus)

        # Add tags for training data
        tagged_corpus_list.append(TaggedDocument(tokens, [i]))

    ## Train doc2vec model
    vector_size = num_features  # Dimensionality of the feature vectors.
    min_count = 2  # Ignores all words with total frequency lower than this.
    dm = 0  # Defines the training algorithm. If dm=1, ‘distributed memory’
    # (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is
    # employed.
    epochs = 100  # Number of iterations (epochs) over the corpus.
    alpha = 0.025  # The initial learning rate.

    model = Doc2Vec(vector_size=vector_size, \
                    min_count=min_count, \
                    dm=dm, \
                    epochs=epochs, \
                    alpha=alpha, \
                    min_alpha=0.0025)

    model.build_vocab(tagged_corpus_list)

    model.train(tagged_corpus_list, \
                total_examples=model.corpus_count, \
                epochs=model.epochs)

    return model


def doc2vec_feature_creation(model, docs):
    """
    Create features for each given document. This function will use the given
    pre-trained Doc2Vec model.

    Parameters
    ----------
    model : Gensim Doc2vec model
        Any pre-trained Doc2vec model.
    docs : List of strings
        Documents' content.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    """
    ## doc2vec format
    corpus_tuple = (str(x) for x in docs)
    # print(corpus_tuple)

    corpus_d2v_format = list()
    for _, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ')  # gensim.utils.simple_preprocess(corpus)
        corpus_d2v_format.append(tokens)

    # Get a feature vector for each document.
    features = [model.infer_vector(corpus_d2v_format[i])
                for i in range(len(corpus_d2v_format))]
    features = np.array([np.array(x) for x in features])

    return features


def doc2vec_evaluate_success_on_train_data(model, feature_vectors):
    """
    Use the feature vectors of the data that were used to train the given
    doc2vec model.

    Parameters
    ----------
    model : Gensim Doc2vec model
        Any pre-trained Doc2vec model.
    feature_vectors : numpy array
        Feature vectors of the data which were used to train the given model.

    """
    ranks = []
    for doc_id in range(len(feature_vectors)):
        inferred_vector = feature_vectors[doc_id]

        sims = model.docvecs.most_similar([inferred_vector], \
                                          topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    counter = 0
    for x in ranks:
        if x == 0:
            counter += 1

    print('Documents most similar to themselfs', str(counter), 'out of', \
          str(len(feature_vectors)))


# ==============================================================================#
#                        Vectorizers(TF-IDF/Bag of words)                      #
# ==============================================================================#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def train_vectorizer(docs, tf_idf=True, analyzer='word', \
                     ngram_range=(1, 1), max_df=1.0, min_df=1, \
                     max_features=None, norm='l2'):
    """
    Train a TfidfVectorizer or CountVectorizer with the given training data.

    Parameters
    ----------
    docs : List of strings
        Documents' content.
    tf_idf: bool
        If true then it will return TfidfVectorizer otherwise CountVectorizer

    See sklearn TfidfVectorizer and CountVectorizer for more information about
    the remaining parameters.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    vectorizer : sklearn TfidVectorizer or CountVectorizer
        See tf_idf parameter for more information.
    """
    train_data = [str(x) for x in docs]
    if (tf_idf):
        # initialize tf-idf vectorizer
        vectorizer = TfidfVectorizer(analyzer=analyzer, \
                                     max_df=max_df, \
                                     min_df=min_df, \
                                     ngram_range=ngram_range, \
                                     max_features=max_features, \
                                     norm=norm)

        # train vectorizer and transform the given data
        features = vectorizer.fit_transform(train_data)
    else:
        # initialize bag of words vectorizer
        vectorizer = CountVectorizer(analyzer=analyzer, \
                                     max_df=max_df, \
                                     min_df=min_df, \
                                     ngram_range=ngram_range, \
                                     max_features=max_features)

        features = vectorizer.fit_transform(train_data)

    features.toarray()
    return features, vectorizer


# =============================================================================#
# =============================================================================#



# =============================================================================#
#                             FEATURE EXTRACTION                               #
# =============================================================================#

from IPython.display import display

# =============================================================================#
#                Dimensionality Reduction (PCA/TruncatedSVD)                   #
# =============================================================================#

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


def reduction_components_evaluation_report(features, max_components=200, \
                                           pca=True):
    """
    Print an evaluation report about the variance for different values of
    components(dimensions).
    Components will start from 1,2,3,4,5 to max_components with 10 components
    step on every round after 5.

    Parameters
    ----------
    features : numpy array
        Features to which dimensionality reduction will be applied.
    max_components: int (default 200)
        Determines the number of components that we want on our report
    pca : bool (default True)
        Determines the dimensionality reduction algorithm. True means PCA
        otherwise TruncatedSVD.
    """
    if (pca):
        columns = ["Num Components", "PCA Time", "Variance explained"]
    else:
        columns = ["Num Components", "TSVD Time", "Variance explained"]

    dimensions_to_test = [i for i in range(1, 6, 1) if i <= features.shape[1]]
    dimensions_to_test += [i for i in (range(10, max_components + 1, 10))
                           if i <= features.shape[1]]
    dimensions_to_test.reverse()

    model_results_df = pd.DataFrame(0, index=dimensions_to_test, \
                                    columns=columns)

    if (pca):
        print("PCA feature extraction report:")
    else:
        print("TSVD feature extraction report:")

    for k in dimensions_to_test:
        print("Handling num dimensions = ", k)
        start_time = time.time()

        # Instantiate TruncatedSVD or PCA object with k components
        if (pca):
            model = PCA(n_components=k)
        else:
            model = TruncatedSVD(n_components=k)

        model.fit(features)

        model_time = time.time() - start_time

        variance = np.sum(model.explained_variance_ratio_)

        # Update df
        model_results_df.loc[k] = [k, model_time, variance]

    # Display evaluation report
    display(model_results_df)


def dimensionality_reduction(n_components, train_features, test_features=None, \
                             pca=True):
    """
    Perfome a dimensionality reduction with TruncatedSVD or PCA on both train
    and test features. train_features will train the model.
    The dimensions will be reduced to n_components.

    Parameters
    ----------
    n_components : int
        New number of dimensions.
    train_features: numpy array
    test_features: numpy array
    pca : bool (default True)
        Determines the dimensionality reduction algorithm. True means PCA
        otherwise TruncatedSVD.

    Returns
    -------
    train_features_reduced : numpy array
        Features after dimensionality reduction.
    test_features_reduced : numpy array
        Features after dimensionality reduction.
    """
    if (pca):
        model = PCA(n_components=n_components)
    else:
        model = TruncatedSVD(n_components=n_components)
    model.fit(train_features)

    # Transform the training and test class data with a dim reduction algorithm.
    train_features_reduced = model.transform(train_features)
    if test_features is not None:
        test_features_reduced = model.transform(test_features)
    else:
        test_features_reduced = None

    variance = np.sum(model.explained_variance_ratio_)
    print('Variance explained with '+str(n_components)+' components: '+ str(variance))

    return train_features_reduced, test_features_reduced


# =============================================================================#
#                 Dimensionality Reduction (LDA) - Supervised                  #
# =============================================================================#

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def reduction_components_evaluation_report_lda(features, labels, max_components=200):
    """
    Print an evaluation report about the variance for different values of
    components(dimensions).
    Components will start from 1,2,3,4,5 to max_components with 10 components
    step on every round after 5.

    Parameters
    ----------
    features : numpy array
        Features to which dimensionality reduction will be applied.
    labels : numpy array
        Classification of the given data.
    max_components: int (default 200)
        Determines the number of components that we want on our report
    """

    columns = ["Num Components", "LDA Time", "Variance explained"]

    dimensions_to_test = [i for i in range(1, 6, 1) if i <= features.shape[1]]
    dimensions_to_test += [i for i in (range(10, max_components + 1, 10))
                           if i <= features.shape[1]]
    dimensions_to_test.reverse()

    model_results_df = pd.DataFrame(0, index=dimensions_to_test, \
                                    columns=columns)

    print("LDA feature extraction report:")
    for k in dimensions_to_test:
        print("Handling num dimensions = ", k)
        start_time = time.time()

        # Instantiate LDA object with k components
        model = LinearDiscriminantAnalysis(n_components=k)

        model.fit(features, labels)

        model_time = time.time() - start_time

        variance = np.sum(model.explained_variance_ratio_)

        # Update df
        model_results_df.loc[k] = [k, model_time, variance]

    # Display evaluation report
    display(model_results_df)


def dimensionality_reduction_lda(n_components, train_features, labels, test_features=None):
    """
    Perfome a dimensionality reduction with TruncatedSVD or PCA on both train
    and test features. train_features will train the model.
    The dimensions will be reduced to n_components.

    Parameters
    ----------
    n_components : int
        New number of dimensions.
    train_features: numpy array
    labels : numpy array
        Classification of the given train data.
    test_features: numpy array


    Returns
    -------
    train_features_reduced : numpy array
        Features after dimensionality reduction.
    test_features_reduced : numpy array
        Features after dimensionality reduction.
    """

    model = LinearDiscriminantAnalysis(n_components=n_components)
    model.fit(train_features, labels)

    # Transform the training and test class data with a dim reduction algorithm.
    train_features_reduced = model.transform(train_features)
    if test_features is not None:
        test_features_reduced = model.transform(test_features)
    else:
        test_features_reduced = None

    variance = np.sum(model.explained_variance_ratio_)
    print('Variance explained with '+str(n_components)+' components: '+ str(variance))

    return train_features_reduced, test_features_reduced


# =============================================================================#
#                     Number of components based on threshold                  #
# =============================================================================#

def select_n_components(goal_var, X, Y=None, algorithm='PCA'):
    """
    Selecting The Best Number Of Components For Dimensionality reduction based
    on the explained variance.

    Parameters
    ----------
    goal_var : float
        Goal threshold.
    X : numpy array
        Data to fit the model for dimensionality reduction
    Y : numpy array(default None)
        Labels in case the choosen algorithm is LDA
    algorithm: string(default PCA)
        Options, 'PCA', 'LDA' otherwise 'TSVD'

    Returns
    -------
    n_components: int
        The number of components that you need to overtake the goal threshold.
    """
    model = None
    if (algorithm == 'PCA'):
        model = PCA(n_components=None)
    elif (algorithm == 'LDA'):
        model = LinearDiscriminantAnalysis(n_components=None)
    else:
        model = TruncatedSVD(n_components=None)


    if (algorithm == 'LDA'):
        model.fit(X,Y)
    else:
        model.fit(X)

    var_ratio = model.explained_variance_ratio_

    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components


# =============================================================================#
#                                      Test                                    #
# =============================================================================#

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def reports(features, labels, model):
    print('Reports for', model)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2, shuffle=True)

    train_features,test_features = dimensionality_reduction_lda(1,train_features,labels,test_labels)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    y_pred = lr_clf.predict(test_features)

    print("Linear Regression Classifier:")
    print(classification_report(test_labels, y_pred))


if __name__ == "__main__":
# this won't be run when imported
    ## Read datasets
    tweet_df = pd.read_csv('dataset/train_dropduplicates.csv')
    test_df = pd.read_csv('dataset/test_processed.csv')
    print("Number of tweets, features:", tweet_df.shape)
    print("Number of test, features:", test_df.shape)


    print("Running bert...")
    features = bert_feature_creation(list(tweet_df['processed']))
    print("Features created")

    reports(features,tweet_df['target'],'BERT')

    reduction_components_evaluation_report_lda(features, tweet_df['target'])
    reduction_components_evaluation_report(features, pca=True)
    reduction_components_evaluation_report(features, pca=False)