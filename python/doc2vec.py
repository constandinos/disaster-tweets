"""
reference: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
"""

import gensim
# install gensim for anaconda as follows:
# conda install -c anaconda gensim
import numpy as np
import pandas as pd

##### FUNCTIONS SECTION #####
def doc2vec_create_corpus(lines, tokens_only=False):
    """
    This function will transform a list of tweets to
    a list of lists of the words in each tweet.
    If tokens_only is set to True it will not append
    and it will not prepend the tags that are necessary
    during the training process. 
    """
    transformed_corpus = list()
    for i, line in enumerate(lines):
        tokens = line.split(' ') #gensim.utils.simple_preprocess(line)
        if tokens_only:
            transformed_corpus.append(tokens)
        else:
            # Add tags for training data
            transformed_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
    return transformed_corpus

def doc2vec_train_model(raw_lines):
    """
    This function will train a doc2vec model with
    the given tweets, which must be a list of
    tweet strings.
    """
    train_corpus = doc2vec_create_corpus(raw_lines)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    transformed_data = \
        [model.infer_vector(train_corpus[i].words) for i in range(len(train_corpus))]
    transformed_data = np.array([np.array(x) for x in transformed_data])
    return  transformed_data, model

def doc2vec_count_success_on_train_data(transformed_data, model):
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


tweet_df = pd.read_csv('../dataset/train.csv')
test_df =pd.read_csv('../dataset/test.csv')
print("Number of tweets, features:",tweet_df.shape)

tweets_text = list(tweet_df['text'])

transformed_data, model = doc2vec_train_model(tweets_text)

#test_corpus = doc2vec_corpus(tweets, tokens_only=True)
# print(transformed_data)

doc2vec_count_success_on_train_data(transformed_data, model)
# print(model.infer_vector(test_corpus[0]))
# print(test_corpus)

