import numpy as np
import pandas as pd
# To install this package with conda run:
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
import torch
# To install this package with conda run one of the following:
# conda install -c conda-forge transformers
# conda install -c conda-forge/label/cf202003 transformers
import transformers as tr # transformers

##### FUNCTIONS SECTION #####
def bert_embedding_testset(df,text,is_text_column = True):
    """
    This function will apply to the given documents a pre-trained bert model and 
    it will create their features.
    It produce 768 features by default.
    
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
    features : numpy array
        This array contains the created features for each document
    """
    ## Loading a pre-trained BERT model
    # Importing pre-trained DistilBert model and tokenizer
    # DistilBert is smaller than BERT but much faster and it requires less memory
    # DistilBERT is a small, fast, cheap and light Transformer model trained by 
    # distilling Bert base. It has 40% less parameters than bert-base-uncased, 
    # runs 60% faster while preserving over 95% of Bert’s performances.
    model_class, tokenizer_class, pretrained_weights = (tr.DistilBertModel, tr.DistilBertTokenizer, 'distilbert-base-uncased')
    
    # For BERT instead of distilBERT Uncomment the following line:
    # It’s a bidirectional transformer pre-trained using a combination of masked
    # language modeling objective and next sentence prediction on a large corpus 
    # comprising the Toronto Book Corpus and Wikipedia.
    #model_class, tokenizer_class, pretrained_weights = (tr.BertModel, tr.BertTokenizer, 'bert-base-uncased')
    
    # Load pretrained model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)
    
    
    ## Tokenization
    # Tokenize every sentece - BERT format (list of lists)
    if (is_text_column) :
        tokenized = df[text].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    else:
        text_df = pd.DataFrame(text)
        tokenized = text_df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    #print(tokenized.head())
    
    
    ## Padding
    # Find the length of the longer sentence of tokens
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    # Add padding to the end of each sentence of tokens. As a result we will have equal length
    # sentences of tokens and transform it to numpy array.
    # BERT processing is faster in that way
    padded_tokens = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    #print(padded_tokens,'\nPadded tokens shape:',np.array(padded_tokens).shape)
    
    
    ## Masking
    # Create a mask that tells BERT to ignore the padding we have previously added
    # to the sentences of tokens.
    # Zero(0) means ignore.
    bert_mask = np.where(padded_tokens != 0, 1, 0)
    #print('Bert mask shape:',bert_mask.shape)
    
    
    ## Running BERT model - feature creation
    padded_tokens_torch = torch.tensor(padded_tokens, dtype=torch.int64)  
    bert_mask_torch = torch.tensor(bert_mask)
    
    #print(padded_tokens_torch)
    #print(bert_mask_torch)
    
    with torch.no_grad():
        hidden_states = bert_model(padded_tokens_torch, attention_mask=bert_mask_torch)
    
    # The reason we are getting only the first element that bert returns is because
    # bert adds a classification token at the first element of each sentence and this
    # is the value that we need from all the hidden layers to form the embedding.
    features = hidden_states[0][:,0,:].numpy()
    

    return features


def bert_embedding_trainset(df, text, target, is_text_column = True):
    """
    This function will apply to the given documents a pre-trained bert model and 
    it will create their features alongside their labels.
    It produce 768 features by default.
    
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
    target: string
        Inticates which column of the dataframe contains the label of the docs.
    is_text_column: bool
        See description for parameter text.

    Returns
    -------
    features : numpy array
        This array contains the created features for each document
    labels : pandas Series
        Labels for each document.
    """
    features = bert_embedding_testset(df, text, is_text_column)
    
    
    # Labels of train dataset
    labels = df[target]


    return features, labels
   
##### END OF FUNCTIONS SECTION #####


## Import dataset
tweet_df = pd.read_csv('../dataset/train.csv')
print("Number of tweets, features:",tweet_df.shape)

emb_features, labels = bert_embedding_trainset(tweet_df,'text','target')


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
