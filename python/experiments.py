import numpy as np
import pandas as pd
import feature_creation as fc

# =============================================================================#
#                     Grid Search and Cross Validation                         #
# =============================================================================#
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

def grid_search_cross_validation(clf_list, x_train, y_train, k_folds=10, score_type='f1_macro'):
    """
	This function will apply grid search to search over specified parameter 
    values for an estimator to find the optimal parameters for a machine 
    learning algorithm.
	Also, this function will apply k-folds cross validation to calculate the 
    average f1_macro score in order to select the machine learning algorithm 
    with highest score.

	Parameters
	----------
	clf_list: list of tuples with name of
		Each tuple contains the name of machine learning algorithm, the 
        initialization estimator and a set with the parameters
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
		This list contains the best cross validation f1 scores of machine 
        learning algorithms
	model_std: list of floats
		This list contains the cross validation standard deviations of machine 
        learning algorithms
   """

    model_names, model_scores, model_std = [], [], []  # return list
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)
    #kfold = model_selection.StratifiedKFold(n_splits=k_folds, shuffle=True)
    for name, model, parameters in clf_list:
        # grid search
        print("Grid search for " + name)
        search = GridSearchCV(model, parameters, scoring=score_type, cv=kfold, n_jobs=-1)
        search.fit(x_train, y_train)
        print("Best parameters: " + str(search.best_params_))
        best_est = search.best_estimator_  # estimator with the best parameters
        
        # k-fold cross validation
        f1_mean, f1_std = cross_validation(best_est, x_train, y_train, k_folds, score_type)
        # append results to the return lists
        model_names.append(name)
        model_scores.append(f1_mean)
        model_std.append(f1_std)

    return model_names, model_scores, model_std


def cross_validation(estimator, x_train, y_train, k_folds=10, score_type='f1_weighted'):
    """
	This function will apply k-folds cross validation to calculate the average
	f1_macro score in order to select the machine learning algorithm with 
    highest score.

	Parameters
	----------
	clf_list: list of estimators
		Estimator (ml or nn) algorithm
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
	estimator_score: list of floats
		This list contains the best cross validation f1 scores of machine 
        learning algorithms
	estimator_std: list of floats
		This list contains the cross validation standard deviations of machine 
        learning algorithms
   """

    estimator_score, estimator_std = None, None  # return results
    kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)
    
    # k-fold cross validation
    #print("Start "+str(k_folds)+"-folds cross validation...")
    f1_score = model_selection.cross_val_score(estimator, x_train, y_train, cv=kfold, scoring=score_type, n_jobs=-1)
    # append results to the return lists
    estimator_score = f1_score.mean()
    estimator_std = f1_score.std()
    #print("End cross validation")

    return estimator_score, estimator_std


# =============================================================================#
#                     Visualize the cross validation results                   #
# =============================================================================#
import matplotlib.pyplot as plt


def plot_graphs(title_name, column_name, labels, f1_score, std):
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
		This list contains the best cross validation f1 scores of machine 
        learning algorithms
	std: list of floats
		his list contains the cross validation standard deviations of machine 
        learning algorithms
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(labels, f1_score, color='tab:green')
    axes[0].set_ylabel('f1 score')
    axes[1].bar(labels, std, color='tab:orange')
    axes[1].set_ylabel('Standard Deviation')
    fig.suptitle(title_name)
    fig.savefig(column_name + title_name + '.png')
    plt.show()
    plt.close(fig)



# =============================================================================#
#                             Execution on data                                #
# =============================================================================#

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""
def reports(features, labels, model, pca):
    print('Reports for', model)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.1, shuffle=True)

    reduction_components_evaluation_report(features, pca=pca)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    y_pred = lr_clf.predict(test_features)

    print("Linear Regression Classifier:")
    print(classification_report(test_labels, y_pred))
"""

def execute(df, bert=False, doc2vec=False, tfidf=False, bow=False):
    clf_list = [("logistic_regression", LogisticRegression(), {'C': np.logspace(-4, 4, 20),\
                                                               'max_iter': [100, 200, 300, 400, 500]}),
                ("k-nn", KNeighborsClassifier(), {'n_neighbors': np.arange(1, 25),  \
                                                  'metric': ['euclidean', 'minkowski']}),
                ("mlp", MLPClassifier(), {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], \
                                          'alpha': [0.0001, 0.05],\
                                          'learning_rate': ['constant','adaptive'],\
                                          'max_iter': [300, 500, 800, 1000, 2000]}),
                ("random_forest", RandomForestClassifier(), {'n_estimators': [200, 500, 1000], \
                                                             'max_features': ['sqrt', 'log2'], \
                                                             'max_depth' : [50, 100, 200, 300]}),
                ("svc", SVC(), {'C': [0.1, 1, 10, 100], \
                                'gamma': [0.01, 0.1, 1],\
                                'kernel': ['rbf', 'linear', 'sigmoid']})]

    column_list = ['text', 'processed_lem', 'processed_stem', 'processed_lem_key', 'processed_stem_key', \
                   'processed_text_deep_without_url', 'processed_text_deep_without_url_key']

    for column in column_list:
        print("*** Column is "+column)
        docs = list(df[column])
        labels = df['target']

        if (bert):
            print("Running bert...")
            features = fc.bert_feature_creation(docs)
            print("Features created")
            #reports(features, labels, 'BERT',pca=True)
            model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
            plot_graphs('BERT', column, model_names, model_scores, model_std)
            print("----------------------------------------------------\n")

        if (doc2vec):
            print("Running doc2vec...")
            model = fc.train_doc2vec_model(docs)
            features = fc.doc2vec_feature_creation(model, docs)
            print("Features created")
            #reports(features, labels, 'DOC2VEC',pca=True)
            model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
            plot_graphs('DOC2VEC', column, model_names, model_scores, model_std)
            print("----------------------------------------------------\n")

        if (tfidf):
            print("Running tfidf...")
            features, _ = fc.train_vectorizer(docs, tf_idf=True)
            print("Features created")
            #reports(features, labels, 'TFIDF',pca=False)
            model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
            plot_graphs('TFIDF', column, model_names, model_scores, model_std)
            print("----------------------------------------------------\n")

        if (bow):
            print("Running bow...")
            features, _ = fc.train_vectorizer(docs, tf_idf=False)
            print("Features created")
            #reports(features, labels, 'Bag of words',pca=False)
            model_names, model_scores, model_std = grid_search_cross_validation(clf_list, features, labels)
            plot_graphs('Bag of words', column, model_names, model_scores, model_std)
            print("----------------------------------------------------\n")

        print("****************************************************\n")

def predict_results(train_df, test_df,text_col):
    x_train = list(train_df[text_col])
    y_train = train_df['target']
    x_test = list([str(x) for x in test_df[text_col]])

    #features_train = fc.bert_feature_creation(x_train)
    #features_test = fc.bert_feature_creation(x_test)

    features_train, vectorizer = fc.train_vectorizer(x_train,tf_idf=True)
    features_test = vectorizer.transform(x_test)
    #svc_clf = SVC(C=10, gamma=0.01, kernel='rbf')
    #cross_validation(svc_clf, features_train, y_train)
    #svc_clf.fit(features_train, y_train)

    features_train = features_train.toarray()
    features_test = features_test.toarray()

    print(features_train)
    lr_clf = LogisticRegression()
    features_train, features_test = fc.dimensionality_reduction_lda(2000,features_train,list(y_train),features_test)
    
    lr_clf.fit(features_train, list(y_train))

    y_pred = lr_clf.predict(features_test)
    submission = pd.read_csv("dataset/sample_submission.csv")
    submission['target'] = y_pred
    submission.to_csv('submission.csv', index=False)


def choose_best_vectorizer(df):
    process_columns = ['text','processed','lemmatization','stemming']
    estimators = [LogisticRegression(),
                  KNeighborsClassifier(),
                  #MLPClassifier(),
                  RandomForestClassifier(),
                  SVC()]
    name_estimators = ["logistic_regression",
                       "k-nn",
                       #"mlp",
                       "random_forest",
                       "svc"]


    Y = list(df['target'])

    # Check with the defaul parameters for all columns
    for col in process_columns:
        tfidf_scores = []
        bow_scores = []
        print("-------------")
        print("Start column: "+col)
        X = list(df[col].astype(str))

        print("Running tf-idf(default)...")
        features_tfidf, _ = fc.train_vectorizer(X,tf_idf=True)
        print("Features created")
        #features_reduced_tfidf, _ = fc.dimensionality_reduction_lda(2000,features_tfidf.toarray(),Y)
        # performance of tfidf with default parameters
        for estimator,name in zip(estimators,name_estimators):
            print("(default)Cross validation for "+name)
            score,_=cross_validation(estimator, features_tfidf, Y)
            tfidf_scores.append(score)
        print("------------")

        print("Running bow(default)...")
        features_bow, _ = fc.train_vectorizer(X,tf_idf=False)
        print("Features created")
        #features_reduced_bow, _ = fc.dimensionality_reduction_lda(2000,features_bow.toarray(),Y)
        # performance of bow with default parameters
        for estimator,name in zip(estimators,name_estimators):
            print("(default)Cross validation for "+name)
            score,_=cross_validation(estimator, features_bow, Y)
            bow_scores.append(score)

        print("End column: "+col)
        print("-----------")
        max_index = np.argmax(tfidf_scores)
        print("Report for column: "+col)
        print("Best of tf-idf: "+str(name_estimators[max_index])+" with score "+str(tfidf_scores[max_index]))
        max_index = np.argmax(bow_scores)
        print("Best of bow:"+str(name_estimators[max_index])+" with score "+str(bow_scores[max_index]))
        print("-----------")


def get_scores(X_train, Y_train, X_test, Y_test, vectorizer_name):
    estimators = [LogisticRegression(),
                  KNeighborsClassifier(),
                  DecisionTreeClassifier(),
                  RandomForestClassifier(),
                  SVC()]
    name_estimators = ["logistic_regression",
                       "k-nn",
                       "DecisionTreeClassifier",
                       "random_forest",
                       "svc"]

    #print("Vectorizer"+str(vectorizer_name))
    #print("Estimator\tMean f1-score\tStd f1-score\tTest-score")
    print(str(vectorizer_name), end = '\t')
    for estimator, estimator_name in zip(estimators, name_estimators):
        mean_score, std_score = cross_validation(estimator, X_train, Y_train)

        estimator.fit(X_train,Y_train)
        Y_pred = estimator.predict(X_test)
        test_score = f1_score(Y_test,Y_pred, average='weighted')

        print(str(mean_score)+"\t"+str(std_score)+"\t"+str(test_score), end = '\t')
    print()

def find_best_for_columns(columns, X_train, Y_train, X_test, Y_test):
    for column in columns:
        print("Column: " + column + '\n')
        train_text = X_train[column].to_list()
        test_text = X_test[column].to_list()

        #Bert
        bert_train_features = fc.bert_feature_creation(train_text)
        bert_test_features = fc.bert_feature_creation(test_text)
        get_scores(bert_train_features, Y_train, bert_test_features, Y_test, 'bert')

        #Doc2Vec
        doc2vec_train_model = fc.train_doc2vec_model(train_text)
        doc2vec_train_features = fc.doc2vec_feature_creation(doc2vec_train_model, train_text)
        doc2vec_test_features = fc.doc2vec_feature_creation(doc2vec_train_model, test_text)
        get_scores(doc2vec_train_features, Y_train, doc2vec_test_features, Y_test, 'doc2vec')

        #TFIDF
        tfidf_train_features, tfidf_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=True)
        tfidf_test_features = tfidf_train_vectorizer.transform(test_text)
        get_scores(tfidf_train_features, Y_train, tfidf_test_features, Y_test, 'tfidf')

        tfidf_train_features, tfidf_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=True, max_features=5000)
        tfidf_test_features = tfidf_train_vectorizer.transform(test_text)
        get_scores(tfidf_train_features, Y_train, tfidf_test_features, Y_test, 'tfidf, features=5000')

        tfidf_train_features, tfidf_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=True, ngram_range=(1,2),\
                                                                           max_features=5000)
        tfidf_test_features = tfidf_train_vectorizer.transform(test_text)
        get_scores(tfidf_train_features, Y_train, tfidf_test_features, Y_test, 'tfidf, ngrams=(1,2), features=5000')

        tfidf_train_features, tfidf_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=True, ngram_range=(2,2),\
                                                                           max_features=5000)
        tfidf_test_features = tfidf_train_vectorizer.transform(test_text)
        get_scores(tfidf_train_features, Y_train, tfidf_test_features, Y_test, 'tfidf, ngrams=(2,2), features=5000')

        tfidf_train_features, tfidf_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=True, min_df=0.05)
        tfidf_test_features = tfidf_train_vectorizer.transform(test_text)
        get_scores(tfidf_train_features, Y_train, tfidf_test_features, Y_test, 'tfidf, min_df=0.05')

        #Bow
        bow_train_features, bow_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=False)
        bow_test_features = bow_train_vectorizer.transform(test_text)
        get_scores(bow_train_features, Y_train, bow_test_features, Y_test, 'bow')

        bow_train_features, bow_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=False, max_features=5000)
        bow_test_features = bow_train_vectorizer.transform(test_text)
        get_scores(bow_train_features, Y_train, bow_test_features, Y_test, 'bow, features=5000')

        bow_train_features, bow_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=False, ngram_range=(1, 2), \
                                                                           max_features=5000)
        bow_test_features = bow_train_vectorizer.transform(test_text)
        get_scores(bow_train_features, Y_train, bow_test_features, Y_test, 'bow, ngrams=(1,2), features=5000')

        bow_train_features, bow_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=False, ngram_range=(2, 2), \
                                                                           max_features=5000)
        bow_test_features = bow_train_vectorizer.transform(test_text)
        get_scores(bow_train_features, Y_train, bow_test_features, Y_test, 'bow, ngrams=(2,2), features=5000')

        bow_train_features, bow_train_vectorizer = fc.train_vectorizer(train_text, tf_idf=False, min_df=0.05)
        bow_test_features = bow_train_vectorizer.transform(test_text)
        get_scores(bow_train_features, Y_train, bow_test_features, Y_test, 'bow, min_df=0.05')

if __name__ == "__main__":
# this won't be run when imported
    ## Read datasets
    tweet_df = pd.read_csv('dataset/train_dropduplicates.csv')
    #test_df = pd.read_csv('dataset/test_processed.csv')
    print("Number of tweets, features:", tweet_df.shape)
    #print("Number of test, features:", test_df.shape)
    
    columns = ['text', 'processed', 'lemmatization', 'stemming']
    X_train, X_test, Y_train, Y_test = train_test_split(tweet_df, tweet_df['target'], test_size=0.2, shuffle=True)

    find_best_for_columns(columns, X_train, Y_train, X_test, Y_test)











# =============================================================================#
#                          Choose Best Vectorizer                              #
# =============================================================================#
# At this point of our experiments we decided that we have to choose the best
# vectorizer among the TFIDF and COUNT vectorizers (a.k.a bag of words).
# For choosing the best vectorizer we have tried to find the best hyper params
# for these vectorizers and choose one of the TFIDF and COUNT along side with
# their best hyper parameters.
# The hyper parameters that we have explored are:
# 1. ngram_range : (1,1) , (1,2) , (2,2)
#   Reason: From our EDA analysis we have seen that tweeter's most common length
#   is between 10 to 20, so trying to increase ngrams will not help at all.
# 2. max_df : 0.80 , 0.90, 1.0
#   Reason: If there is a word that appears in more than the choosen percent of 
#   the documents then this word might confuse our estimator. To reduce this 
#   confusion we have decided to check for some reasonable percentages.
    
    #choose_best_vectorizer(tweet_df)
    





    #execute(tweet_df, bert=True, doc2vec=False, tfidf=False, bow=False)
    #predict_results(tweet_df, test_df,'processed')

    print("End execution")