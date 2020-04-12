# Use tensorflow for GPU !!!
# To install the current release of GPU TensorFlow
# conda create -n tf-gpu tensorflow-gpu
# conda activate tf-gpu
# conda install --name tf-gpu pylint -y
# To install the current release of CPU-only TensorFlow
# conda create -n tf tensorflow
# conda activate tf
# conda install --name tf pylint -y
import tensorflow as tf
# Uncomment to check if tensorflow is using your GPU
#tf.debugging.set_log_device_placement(True)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
# To install this package with conda run one of the following:
# conda install -c conda-forge tensorflow-hub
# conda install -c conda-forge/label/gcc7 tensorflow-hub
# conda install -c conda-forge/label/cf201901 tensorflow-hub
# conda install -c conda-forge/label/cf202003 tensorflow-hub
import tensorflow_hub as hub
# Official tokenization script created by the Google team:
# https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# To install this package with conda run:
# conda install -c powerai sentencepiece
import tokenization


import numpy as np
# To install this package with conda run:
# conda install -c anaconda pandas
import pandas as pd

# To install this package with conda run:
# conda install -c anaconda scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score



# =============================================================================#
#                              Disaster Classifier                             #
# =============================================================================#

class DisasterClassifier:
    
    def __init__(self, bert_layer = None, lr=0.00001, epochs=4, batch_size=32):
        # Load Bert
        # Wraps a SavedModel (or a legacy Hub.Module) as a Keras Layer.
        if bert_layer is None:
            module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
            bert_layer = hub.KerasLayer(module_url, trainable=True)

        # BERT and Tokenization params
        self.bert_layer = bert_layer
        
        vocab_file=self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        
        # Learning control params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.nn_model = None
        
        
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
        # have equal length sentences of tokens and rhen transform them to numpy 
        # array.
        padded_tokens = np.array([i+[0]*(max_len-len(i)) for i in bert_tokens])
        ## Masking
        # Zero(0) means ignore.
        bert_masks = np.where(padded_tokens != 0, 1, 0)
        ## Segments
        bert_segments = np.where(padded_tokens != 0, 0, 0)

        return padded_tokens, bert_masks, bert_segments


    def create_model(self):
        """
        Build a binary classification NN model
        Use sigmoid activation function
        output in {0,1}
        Loss function : Binary crossentropy
        """
        # Instantiate Keras tensors.
        input_word_ids = Input(shape=(None,), dtype=tf.int32, 
                               name="input_word_ids")
        input_mask = Input(shape=(None,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(None,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, 
                                             segment_ids])
        # We get only the first element [CLS] because bert adds this 
        # classification token at the first element of each sentence.
        cls_output = sequence_output[:, 0, :]
        

        # Connect the previous layer with a Dense layer with only one node.
        # This layer will be the output layer.
        out = Dense(1, activation='sigmoid')(cls_output)
        # Build our model
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], 
                      outputs=out)
        # Choose an optimizer to adjust weights.
        optimizer = Adam(learning_rate=self.lr)
        # Compile our model.
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                      metrics=['accuracy',Precision(),Recall()])
        
        # Binary Cross-Entropy Loss is intended for use with binary 
        # classification where the target values are in the set {0, 1}.
        # The function requires that the output layer is configured with a 
        # single node and a ‘sigmoid‘ activation, as in our case.
        return model


    def train(self, X, Y):
        """
        Train the nn model with the given X and its labels Y.
        You can not continue training, you are re-training from scratch with
        this function.
        """
        X = [str(x) for x in X]

        X_encoded = self.encode(X)

        self.nn_model = self.create_model()
        self.nn_model.fit(X_encoded, np.array(Y), epochs=self.epochs, 
                          batch_size=self.batch_size)


    def predict(self, X):
        """
        Predict the class of the given examples.
        Prediction can be in the set {0,1}
        You have to train the model at least once otherwise it will classify
        everything in the class 0.
        """
        if self.nn_model is None:
            return np.zeros((len(X), 1))

        X = [str(x) for x in X]

        X_encoded = self.encode(X)

        y_pred = np.zeros((len(X), 1))

        y_pred = self.nn_model.predict(X_encoded)
        y_pred = y_pred.round().astype(int)
        return y_pred
    
    def get_model_summary(self):
        if self.nn_model is None:
            return
        
        # summarize the model
        self.nn_model.summary()

        # summarize the model with a plot
        plot_model(self.nn_model, 'nn_model.png', show_shapes=True)


    def load_model(self, file):
        """
        Load a nn model.
        """
        self.nn_model = load_model(file)


    def save_model(self,file):
        """
        Save the current model.
        """
        name = file+'.h5'
        self.nn_model.save(name)

# =============================================================================#
#                         End of Disaster Classifier                           #
# =============================================================================#



clf = DisasterClassifier(lr=0.00001, epochs=4, batch_size=32)


df_train = pd.read_csv('dataset/train_dropduplicates.csv')
df_test = pd.read_csv('dataset/test_processed.csv')
print("Number of tweets, features:",df_train.shape)
print("Number of test, features:",df_test.shape)

clf.train([str(x) for x in list(df_train['processed_lem'])],
                                list(df_train['target']))
y_pred = clf.predict([str(x) for x in list(df_test['processed_lem'])])
submission = pd.read_csv("dataset/sample_submission.csv")
submission['target'] = y_pred
submission.to_csv('submission.csv', index=False)