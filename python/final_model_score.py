import experiments as exp
import feature_creation as fc

import pandas as pd
import numpy as np
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def our_test_score():
    # Pre-processing step is omitted. If you want to see our job on the
    # preprocessing step then check tweet_preprocessing.py

    # Read datasets
    train_df = pd.read_csv('dataset/our_train.csv')
    print("Number of train, features:", train_df.shape)

    # This is the split that we used to evaluate our models
    test_df = pd.read_csv('dataset/our_test.csv')
    print("Number of test, features:", test_df.shape)

    X_train = train_df.astype(str)
    X_test = test_df.astype(str)

    # Choose our best preprocessed column
    X_train = X_train['ekphrasis'].to_list()
    X_test = X_test['ekphrasis'].to_list()
    Y_train = train_df['target'].astype(str)
    Y_test = test_df['target'].astype(str)

    # Run our best vectorizer (BERT)
    print("\nStart features creation with BERT")
    X_train_features = fc.bert_feature_creation(X_train)
    X_test_features = fc.bert_feature_creation(X_test)
    print("End of features creation\n")

    # Best hyper-parameters for logistic regression after gridsearch
    best_c = 0.14384498882876628
    best_max_iter = 50
    best_solver = 'liblinear'

    # Our best estimator is logistic regression
    estimator = LogisticRegression(C=best_c, max_iter=best_max_iter, solver=best_solver)

    # Cross validation on the train features
    print("\nStart cross validation for Logistic regression")
    mean_score, std_score = exp.cross_validation(estimator, X_train_features, Y_train)
    print("End of cross validation for Logistic regression")

    print("\nLogistic Regression (Cross validation results):")
    print("Best parameters: C = "+str(best_c)+" Max_iter = "+str(best_max_iter)+" Solver = "+best_solver)
    print('(Cross validation) Average f1-weighted score: %.2f%%(+/- %.2f%%)' % (mean_score*100,std_score*100))

    # Train logistic regression model for evaluation on the test data
    estimator.fit(X_train_features,Y_train)
    Y_pred = estimator.predict(X_test_features)
    test_score = f1_score(Y_test,Y_pred, average='weighted')

    print('(Test) F1-weighted score: %.2f%%' % (test_score*100))


def kaggle_test_submition():
    ## Create kaggle sumbission on their test data
    train_df_kaggle = pd.read_csv('dataset/train_dropduplicates_all.csv')
    test_df_kaggle = pd.read_csv('dataset/kaggle_test/test_ekphrasis.csv')

    print("\nKaggle submition")

    X_train_kaggle = train_df_kaggle.astype(str)
    X_test_kaggle = test_df_kaggle.astype(str)

    X_train_kaggle = X_train_kaggle['ekphrasis'].to_list()
    X_test_kaggle = X_test_kaggle['ekphrasis'].to_list()
    Y_train_kaggle = train_df_kaggle['target'].astype(str)

    print("Start features creation with BERT for Kaggle")
    X_train_features_kaggle = fc.bert_feature_creation(X_train_kaggle)
    X_test_features_kaggle = fc.bert_feature_creation(X_test_kaggle)
    print("End of features creation\n")


    best_c = 0.14384498882876628
    best_max_iter = 50
    best_solver = 'liblinear'
    estimator_kaggle = LogisticRegression(C=best_c, max_iter=best_max_iter, solver=best_solver)

    print("\nStart training Logistic Regression model")
    estimator_kaggle.fit(X_train_features_kaggle,Y_train_kaggle)
    print("End training\n")
    print("\nStart predictions for Kaggle")
    Y_pred_kaggle = estimator_kaggle.predict(X_test_features_kaggle)
    print("End predictions\n")

    submission = pd.read_csv("dataset/sample_submission.csv")
    submission['target'] = Y_pred_kaggle
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
# this won't be run when imported

    # Our test score
    our_test_score()
    
    ############################################################################

    # Prepare Kaggle submission
    #kaggle_test_submition()