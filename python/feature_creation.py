import numpy as np
import pandas as pd
import time

from IPython.display import display

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

#==============================================================================#
#                              DistilBert / Bert                               #
#==============================================================================#

# To install this package with conda run:
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
import torch
# To install this package with conda run one of the following:
# conda install -c conda-forge transformers
# conda install -c conda-forge/label/cf202003 transformers
import transformers as tr

def bert_feature_creation(docs, distilbert = True):
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
        model_class, tokenizer_class, weights = (tr.DistilBertModel,\
                                                    tr.DistilBertTokenizer,\
                                                    'distilbert-base-uncased')
    else:
    # Use BERT instead of distilBERT
    # It’s a bidirectional transformer pre-trained using a combination of masked
    # language modeling objective and next sentence prediction on a large corpus 
    # comprising the Toronto Book Corpus and Wikipedia.
        model_class, tokenizer_class, weights = (tr.BertModel,\
                                                    tr.BertTokenizer,\
                                                    'bert-base-uncased')
    
    # Load pretrained model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(weights)
    bert_model = model_class.from_pretrained(weights)
    

    ## Tokenization
    # Tokenize every sentece - BERT format (list of lists)
    docs_df = pd.DataFrame(docs)
    tokenized = docs_df[0].apply((lambda x: tokenizer.encode(str(x),\
                                    add_special_tokens=True)))
    #print(tokenized.head())
    
    
    ## Padding
    # Find the length of the longer sentence of tokens
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    # Add padding to the end of each sentence of tokens. As a result we will 
    # have equal length sentences of tokens and transform them to numpy array.
    # BERT processing is faster with padding
    padded_tokens = np.array([i+[0]*(max_len-len(i)) for i in tokenized.values])
    #print(padded_tokens,'\nPadded tokens shape:',np.array(padded_tokens).shape)
    
    
    ## Masking
    # Create a mask that tells BERT to ignore the padding we have previously 
    # added to the sentences of tokens.
    # Zero(0) means ignore.
    bert_mask = np.where(padded_tokens != 0, 1, 0)
    #print('Bert mask shape:',bert_mask.shape)
    
    
    ## Running BERT model - feature creation
    padded_tokens_torch = torch.tensor(padded_tokens, dtype=torch.int64)  
    bert_mask_torch = torch.tensor(bert_mask)
    
    #print(padded_tokens_torch)
    #print(bert_mask_torch)
    
    with torch.no_grad():
        hidden_states = bert_model(input_ids = padded_tokens_torch, \
                                    attention_mask = bert_mask_torch)
    
    # The reason we are getting only the first element that bert returns is 
    # because bert adds a classification token at the first element of each 
    # sentence and this is the value that we need from all the hidden layers to
    # form the embedding.
    features = hidden_states[0][:,0,:].numpy()

    return features




#==============================================================================#
#                                   Doc2Vec                                    #
#==============================================================================#

# To install gensim for anaconda run:
# conda install -c anaconda gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def train_doc2vec_model(docs, num_features = 100):
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
    #print(corpus_tuple)

    tagged_corpus_list = list()
    for i, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ') #gensim.utils.simple_preprocess(corpus)
        
        # Add tags for training data
        tagged_corpus_list.append(TaggedDocument(tokens, [i]))


    ## Train doc2vec model
    vector_size = num_features #  Dimensionality of the feature vectors.
    min_count = 2 # Ignores all words with total frequency lower than this.
    dm = 0  # Defines the training algorithm. If dm=1, ‘distributed memory’ 
            #(PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is 
            # employed.
    epochs = 100    # Number of iterations (epochs) over the corpus.
    alpha = 0.025   # The initial learning rate.

    model = Doc2Vec(vector_size = vector_size,\
                        min_count = min_count,\
                        dm = dm,\
                        epochs = epochs,\
                        alpha = alpha,\
                        min_alpha=0.0025)
    
    model.build_vocab(tagged_corpus_list)
 
    model.train(tagged_corpus_list,\
                    total_examples = model.corpus_count,\
                    epochs = model.epochs)

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
    #print(corpus_tuple)

    corpus_d2v_format = list()
    for _, corpus in enumerate(corpus_tuple):
        tokens = corpus.split(' ') #gensim.utils.simple_preprocess(corpus)
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

    print('Documents most similar to themselfs',str(counter),'out of',\
            str(len(feature_vectors)))




#==============================================================================#
#                        Vectorizers(TF-IDF/Bag of words)                      #
#==============================================================================#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def train_vectorizer(docs, tf_idf = True, analyzer = 'word',\
                        ngram_range = (1,1), max_df = 1.0, min_df = 1, \
                        max_features = None, norm = 'l2'):
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
        vectorizer = TfidfVectorizer(analyzer = 'word',\
                                        max_df = max_df,\
                                        min_df = min_df,\
                                        ngram_range = ngram_range,\
                                        max_features = max_features,\
                                        norm = norm)

        # train vectorizer and transform the given data
        features = vectorizer.fit_transform(train_data)
    else:
        # initialize bag of words vectorizer
        vectorizer = CountVectorizer(analyzer = analyzer,\
                                    max_df = max_df,\
                                    min_df = min_df,\
                                    ngram_range = ngram_range,\
                                    max_features = max_features)

        features = vectorizer.fit_transform(train_data)
    
    features.toarray()
    return features, vectorizer




#==============================================================================#
#                Dimensionality Reduction (PCA/TruncatedSVD)                   #
#==============================================================================#

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def reduction_components_evaluation_report(features, max_components = 200,\
                                            pca = True):
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
        Determines the number of components that we wont on our report
    pca : bool (default True)
        Determines the dimensionality reduction algorithm. True means PCA
        otherwise TruncatedSVD.
    """
    if (pca):
        columns = ["Num Components", "PCA Time", "Variance explained"]
    else:
        columns = ["Num Components", "TSVD Time", "Variance explained"]
    
    dimensions_to_test = [i for i in range(1,6,1) if i <= features.shape[1]]
    dimensions_to_test += [i for i in (range(10,max_components+1,10)) 
                            if i <= features.shape[1]]
    dimensions_to_test.reverse()

    model_results_df = pd.DataFrame(0, index = dimensions_to_test, \
                                    columns = columns)

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

def dimensionality_reduction(n_components, train_features, test_features,\
                                pca = True):
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
        model = PCA(n_components = n_components)
    else:
        model = TruncatedSVD(n_components = n_components)
    model.fit(train_features)

    #Transform the training and test class data with a dim reduction algorithm.
    train_features_reduced = model.transform(train_features)
    test_features_reduced = model.transform(test_features)

    return train_features_reduced, test_features_reduced



#==============================================================================#


#==============================================================================#
#                     Grid Search and Cross Validation                         #
#==============================================================================#
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def grid_search_cross_validation(clf_list, x_train, y_train, k_folds=10, score_type='f1_macro'):
    """
	This function will apply grid search to search over specified parameter values
	for an estimator to find the optimal parameters for a machine learning algorithm.
	Also, this function will apply k-folds cross validation to calculate the average
	f1_macro score in order to select the machine learning algorithm with highest
	score.

	Parameters
	----------
	clf_list: list of tuples with name of
		Each tuple contains the name of machine learning algorithm, the initialization
		estimator and a set with the parameters
	x_train: numpy array
		The train data
	y_train: numpy array
		The labels of train data
	k_folds: integer
		The number of folds
	score_type: string
		The name of score type

	Returns
	-------
	model_names: list of strings
		This list contains the names of machine learning algorithms
	model_scores: list of floats
		This list contains the best cross validation f1 scores of machine learning
		algorithms
	model_std: list of floats
		This list contains the cross validation standard deviations of machine learning
		algorithms
   """

    model_names, model_scores, model_std = [], [], []  # return list

    for name, model, parameters in clf_list:
        # grid search
        search = GridSearchCV(model, parameters, scoring=score_type)
        search.fit(x_train, y_train)
        # print(search.best_params_)
        best_est = search.best_estimator_  # estimator with the best parameters
        # k-fold cross validation
        kfold = model_selection.KFold(n_splits=k_folds)
        f1_score = model_selection.cross_val_score(best_est, x_train, y_train, cv=kfold, scoring=score_type)
        # append results to the return lists
        model_names.append(name)
        model_scores.append(f1_score.mean())
        model_std.append(f1_score.std())

    return model_names, model_scores, model_std


#==============================================================================#
#                     Visualize the cross validation results                   #
#==============================================================================#
import matplotlib.pyplot as plt

def plot_graphs(title_name, labels, f1_score, std):
    """
    This function will plot the results of cross validation for each machine
    learning algorithm

    Parameters
    ----------
	title_name: string
		Method that extracts features
    labels: list of strings
        This list contains the names of machine learning algorithms
    f1_score: list of floats
		This list contains the best cross validation f1 scores of machine learning
		algorithms
	std: list of floats
		his list contains the cross validation standard deviations of machine learning
		algorithms
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(labels, f1_score, color='tab:green')
    axes[0].set_ylabel('f1 score')
    axes[1].bar(labels, std, color='tab:orange')
    axes[1].set_ylabel('Standard Deviation')
    fig.suptitle(title_name)
    fig.savefig(title_name+'.png')
    plt.show()
    plt.close(fig)


#==============================================================================#
#                             Execution on data                                #
#==============================================================================#

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def reports(features,labels,model,pca):
    print('Reports for',model)
    train_features, test_features, train_labels, test_labels = \
                                            train_test_split(features, labels)

    
    reduction_components_evaluation_report(features, pca=pca)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    y_pred = lr_clf.predict(test_features)

    print("Linear Regression Classifier:")
    print(classification_report(test_labels, y_pred))


def execute(df, bert = False, doc2vec = False, tfidf = False, bow = False):
    docs = list(df['processed_text'])
    labels = df['target']

    clf_list = [("Logistic Regression", LogisticRegression(), {}),
                ("K-Neighbors Classifier", KNeighborsClassifier(), {'n_neighbors': [4, 5]}),
                ("Multi-layer Perceptron", MLPClassifier(), {}),
                ("Random Forest", RandomForestClassifier(), {}),
                ("SVM", SVC(), {})]
    
    if (bert):
        features = bert_feature_creation(docs)
        #reports(features, labels, 'BERT',pca=True)
        model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
        plot_graphs('BERT', model_names, model_scores, model_std)
    
    if (doc2vec):
        model = train_doc2vec_model(docs)
        features = doc2vec_feature_creation(model,docs)
        #reports(features, labels, 'DOC2VEC',pca=True)
        model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
        plot_graphs('DOC2VEC', model_names, model_scores, model_std)

    if (tfidf):
        features, _ = train_vectorizer(docs,tf_idf=True)
        #reports(features, labels, 'TFIDF',pca=False)
        model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
        plot_graphs('TFIDF', model_names, model_scores, model_std)
    
    if (bow):
        features, _ = train_vectorizer(docs,tf_idf=False)
        #reports(features, labels, 'Bag of words',pca=False)
        model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
        plot_graphs('Bag of words', model_names, model_scores, model_std)



## Read datasets
tweet_df = pd.read_csv('../dataset/train_processed_lem.csv')
test_df = pd.read_csv('../dataset/test.csv')
print("Number of tweets, features:",tweet_df.shape)
print("Number of test, features:",test_df.shape)

execute(tweet_df, bert = True, doc2vec = True, tfidf = True, bow = True)