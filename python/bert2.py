# To install the current release of CPU-only TensorFlow
# conda create -n tf tensorflow
# conda activate tf
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
# To install this package with conda run one of the following:
# conda install -c conda-forge tensorflow-hub
# conda install -c conda-forge/label/gcc7 tensorflow-hub
# conda install -c conda-forge/label/cf201901 tensorflow-hub
# conda install -c conda-forge/label/cf202003 tensorflow-hub
import tensorflow_hub as hub
# Official tokenization script created by the Google team
# !wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# To install this package with conda run:
# conda install -c powerai sentencepiece
import tokenization

# conda install --name tf pylint -y

# To install this package with conda run:
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
import torch
# To install this package with conda run one of the following:
# conda install -c conda-forge transformers
# conda install -c conda-forge/label/cf202003 transformers
import transformers as tr

import numpy as np
# To install this package with conda run:
# conda install -c anaconda pandas
import pandas as pd

# To install this package with conda run:
# conda install -c anaconda scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score





class DisasterDetector:
    
    def __init__(self, bert_layer, lr=0.001, epochs=10, batch_size=32):
        
        # BERT and Tokenization params
        self.bert_layer = bert_layer
        
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        
        # Learning control params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.nn_model = None
        self.method = None
        
        
    def encode(self, texts):
        """ 
        Encode every text to its BERT format.

        Format: [101, ..., 102]
        """
        bert_tokens = []
        bert_masks = []
        bert_segments = []

        for text in texts:
            # Split text to array of tokens (words)
            text = self.tokenizer.tokenize(text)
            
            converter_input = ['[CLS]'] + text + ['[SEP]']

            tokens = self.tokenizer.convert_tokens_to_ids(converter_input)

            bert_tokens.append(tokens)

        ## Padding
        # Find the length of the longer sentence of tokens
        max_len = 0
        for i in bert_tokens:
            if len(i) > max_len:
                max_len = len(i)

        # Add padding to the end of each sentence of tokens. As a result we will 
        # have equal length sentences of tokens and transform them to numpy 
        # array.
        padded_tokens = np.array([i+[0]*(max_len-len(i)) for i in bert_tokens])
        ## Masking
        # Zero(0) means ignore.
        bert_masks = np.where(padded_tokens != 0, 1, 0)
        ## Segments
        bert_segments = np.where(padded_tokens != 0, 0, 0)

        return padded_tokens, bert_masks, bert_segments
    

    def bert_feature_creation(self, docs, distilbert = True):
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
        # Importing pre-trained DistilBert model and tokenizer
        if (distilbert):
            model_class, tokenizer_class, weights = (tr.DistilBertModel,\
                                                        tr.DistilBertTokenizer,\
                                                        'distilbert-base-uncased')
        else:
        # Use BERT instead of distilBERT
            model_class, tokenizer_class, weights = (tr.BertModel,\
                                                        tr.BertTokenizer,\
                                                        'bert-base-uncased')
        
        # Load pretrained model and tokenizer
        tokenizer = tokenizer_class.from_pretrained(weights)
        bert_model = model_class.from_pretrained(weights)
        

        ## Tokenization
        docs_df = pd.DataFrame(docs)
        tokenized = docs_df[0].apply((lambda x: tokenizer.encode(str(x),\
                                        add_special_tokens=True)))
        
        
        ## Padding
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded_tokens = np.array([i+[0]*(max_len-len(i)) for i in tokenized.values])
        
        ## Masking
        bert_mask = np.where(padded_tokens != 0, 1, 0)
        
        ## Running BERT model - feature creation
        padded_tokens_torch = torch.tensor(padded_tokens, dtype=torch.int64)  
        bert_mask_torch = torch.tensor(bert_mask)
        
        with torch.no_grad():
            hidden_states = bert_model(input_ids = padded_tokens_torch, \
                                        attention_mask = bert_mask_torch)
        
        features = hidden_states[0][:,0,:].numpy()

        return features


    def create_model(self):
        """
        Build a binary classification NN model
        Use sigmoid activation function
        output in {0,1}
        Loss function : Binary crossentropy
        """
        # Instantiate Keras tensors.
        # shape: A shape tuple (integers), not including the batch size.
        # For instance, shape=(32,) indicates that the expected input will be 
        # batches of 32-dimensional vectors. Elements of this tuple can be 
        # None; 'None' elements represent dimensions where the shape is not 
        # known.
        input_ids = Input(shape=(None,), dtype=tf.int32)
        input_mask = Input(shape=(None,), dtype=tf.int32)
        input_segments = Input(shape=(None,), dtype=tf.int32)    

        _, output_tensor = self.bert_layer([input_ids, input_mask, input_segments]) 
        # We get only the first element [CLS] because bert adds this 
        # classification token at the first element of each sentence.
        cls_output = output_tensor[:, 0, :]
        # Just your regular densely-connected NN layer.
        # Activation is the element-wise activation function
        # output arrays of shape (*, 1)
        # 2D input with shape (batch_size, input_dim) - in our case cls_output
        # cls_output (batch_size = None, input_dim = 768)
        out = Dense(1, activation='sigmoid')(cls_output)
        # Model groups layers into an object with training and inference
        # features.
        model = Model(inputs=[input_ids, input_mask, input_segments], outputs=out)
        # Adamax is sometimes superior to adam, specially in models with 
        # embeddings
        optimizer = Adamax(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Binary Cross-Entropy Loss is intended for use with binary 
        # classification where the target values are in the set {0, 1}.
        # The function requires that the output layer is configured with a 
        # single node and a ‘sigmoid‘ activation in order to predict the 
        # probability for class 1.
        return model
    

    def create_model2(self):
        """
        Build a binary classification NN model
        Use sigmoid activation function
        output in {0,1}
        Loss function : Binary crossentropy
        """
        # Instantiate Keras tensors.
        input_ids = Input(shape=(768,), dtype=tf.int32)
  
        out = Dense(1, activation='sigmoid')(input_ids)

        model = Model(inputs=input_ids, outputs=out)
        optimizer = Adamax(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model


    def train(self, X, Y, tokens=True):
        """
        Train the nn model with the given X and its labels Y.
        """
        if (tokens):
            X_encoded = self.encode(X)
        else:
            X_encoded = self.bert_feature_creation(X)
        Y = np.array(Y)

        if (tokens):
            self.nn_model = self.create_model().fit(X_encoded, Y, epochs=self.epochs, batch_size=self.batch_size)
            self.method = 0
        else:
            self.nn_model = self.create_model2().fit(X_encoded, Y, epochs=self.epochs, batch_size=self.batch_size)
            self.method = 1


    def cross_validation(self, X, Y):
        """
        Cross validation for the model.

        Works only with features method not the tokens one.
        """
        X_encoded = self.bert_feature_creation(X)
        Y = np.array(Y)

        ## Cross validation
        # create model
        model = KerasClassifier(build_fn=self.create_model2, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        # evaluate using 10-fold cross validation
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(model, X_encoded, Y, cv=kfold, scoring='f1_weighted')
        print('Average weighted f1-score:',results.mean())
        print('Average weighted f1-score std:',results.std())


    def predict(self, X):
        """
        Predict the class of the given examples.
        Prediction can be in the set {0,1}

        It will apply  the same method (tokens or features) as the train did
        the last time it was called.
        """
        if self.method is None:
            return np.zeros((len(X), 1))

        if (self.method==0):
            X_encoded = self.encode(X)
        else:
            X_encoded = self.bert_feature_creation(X)

        y_pred = np.zeros((len(X), 1))

        y_pred = self.nn_model.predict(X_encoded)

        return y_pred
    

    def load_model(self, tokens, file):
        if (tokens == True):
            self.nn_model = load_model(file)
            self.method = 0
        else:
            self.nn_model = load_model(file)
            self.method = 1

    def save_model(self,file):
        if self.method is None:
            return
        else:
            name = file+'.h5'
            self.nn_model.save(name)

# Load Bert
# Wraps a SavedModel (or a legacy Hub.Module) as a Keras Layer.
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)

clf = DisasterDetector(bert_layer, lr=0.001, epochs=10, batch_size=32)


df_train = pd.read_csv('../dataset/train_processed_lem.csv')
df_test = pd.read_csv('../dataset/test.csv')
print("Number of tweets, features:",df_train.shape)
print("Number of test, features:",df_test.shape)

clf.cross_validation(list(df_train['text']),list(df_train['target']))


#clf.train(list(df_train['text']),list(df_train['target']), tokens=True)
#y_pred = clf.predict(df_test)
#df = pd.DataFrame(y_pred)
#df.to_csv('letssee', sep='\t', encoding='utf-8')


