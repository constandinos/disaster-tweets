import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import string


def create_all_words(target):
    """
    This function will create a list with all words in tweets with target=target

    Parameters
    ----------
    target: integer
        0 for not disaster tweets
        1 for disaster tweets
    Returns
	-------
	words: list of strings
		This list contains all words in tweets with target=target
    """
    words = []
    for x in train_data[train_data['target'] == target]['text'].str.split():
        for i in x:
            words.append(i)
    return words


def create_dictionary(words):
    """
    This function will create a dictionary with the frequency of appearing each word

    Parameters
    ----------
    words: list of strings
        This list contains all words in tweets with a specific target
    Returns
    -------
    words: dictionary
        This dictionary contains the frequency of appearing each word
    """
    dic = defaultdict(int)
    for word in words:
        if word in stop_words:
            dic[word] += 1
    return dic


def create_punctuation_dictionary(words):
    """
    This function will create a dictionary with the frequency of appearing each punctuation

    Parameters
    ----------
    words: list of strings
        This list contains all punctuation in tweets with a specific target
    Returns
    -------
    words: dictionary
        This dictionary contains the frequency of appearing each punctuation
    """
    dic = defaultdict(int)
    special = string.punctuation
    for i in words:
        if i in special:
            dic[i] += 1
    return dic


def find_most_freq_word(words):
    """
    This function will create a dictionary with the frequency of appearing each punctuation

    Parameters
    ----------
    words: list of strings
        This list contains all punctuation in tweets with a specific target
    Returns
    -------
    x: list of string
        This list contains the most frequent not stopwords
    y: list of integer
        This list contains the frequency of the most frequent not stopwords
    """
    counter = Counter(words)
    most = counter.most_common()
    x = []
    y = []
    for word, count in most[:40]:
        if (word not in stop_words):
            x.append(word)
            y.append(count)
    return x, y

#Read the data
train_data = pd.read_csv('../dataset/train.csv')
test_data = pd.read_csv('../dataset/test.csv')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
train_data.set_index('id', inplace=True)

count_row = train_data.shape[0]  # gives number of row count
count_col = train_data.shape[1]  # gives number of col count
print("Total number of rows = " + str(count_row))
print("Total number of columns = " + str(count_col))


#Graph1: Tweets
count_disaster = train_data[train_data['target'] == 1].shape[0]
count_not_disaster = train_data[train_data['target'] == 0].shape[0]

plt.bar("Disaster tweets", count_disaster, color='tab:blue')
plt.bar("Not disaster tweets", count_not_disaster, color='tab:red')
plt.ylabel('Number of samples')
plt.title('Tweets')
plt.savefig('figure1_disasterOrNot.png')
plt.show()


#Graph2: Characters in tweets
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.len()
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.len()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].hist(disaster_tweet_chars, color='tab:blue')
axes[0].set_xlabel('Number of characters in a tweet')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Disaster tweets')
axes[1].hist(not_disaster_tweet_chars, color='tab:red')
axes[1].set_xlabel('Number of characters in a tweet')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
fig.savefig('figure2_chars.png')
plt.show()
plt.close(fig)


#Graph3: Words in tweets
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.split().map(lambda x: len(x))
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.split().map(lambda x: len(x))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].hist(disaster_tweet_chars, color='tab:blue')
axes[0].set_xlabel('Number of words in a tweet')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Disaster tweets')
axes[1].hist(not_disaster_tweet_chars, color='tab:red')
axes[1].set_xlabel('Number of words in a tweet')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Not disaster tweets')
fig.suptitle('Words in tweets')
fig.savefig('figure3_words.png')
plt.show()
plt.close(fig)


#Graph4: Average word lenght in each tweets
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.split().\
                                                        apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.split().\
                                                        apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].hist(disaster_tweet_chars, color='tab:blue')
axes[0].set_xlabel('Average word length')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Disaster tweets')
axes[1].hist(not_disaster_tweet_chars, color='tab:red')
axes[1].set_xlabel('Average word length')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Not disaster tweets')
fig.suptitle('Average word length in each tweets')
fig.savefig('figure4_wordLenght.png')
plt.show()
plt.close(fig)


#Graph 5: Most frequent stopwords
disaster_all_words = create_all_words(1)
not_disaster_all_words = create_all_words(0)
disaster_most_freq_words = sorted(create_dictionary(disaster_all_words).\
                                                                        items(), key=lambda x: x[1], reverse=True)[:15]
x0, y0 = zip(*disaster_most_freq_words)
not_disaster_most_freq_words = sorted(create_dictionary(not_disaster_all_words).\
                                                                        items(), key=lambda x: x[1], reverse=True)[:15]
x1, y1 = zip(*not_disaster_most_freq_words)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
axes[0].bar(x0, y0, color='tab:blue')
axes[0].set_title('Disaster tweets')
axes[0].set_xlabel('Most frequent stopwords')
axes[0].set_ylabel('Frequency')
axes[1].bar(x1,y1, color='tab:red')
axes[1].set_title('Not disaster tweets')
axes[1].set_xlabel('Most frequent stopwords')
axes[1].set_ylabel('Frequency')
fig.savefig('figure5_freqStopwords.png')
plt.show()
plt.close(fig)


#Graph 6: Most frequent punctuations
disaster_most_freq_words = sorted(create_punctuation_dictionary(disaster_all_words).\
                                                                        items(), key=lambda x: x[1], reverse=True)[:15]
x0, y0 = zip(*disaster_most_freq_words)
not_disaster_most_freq_words = sorted(create_punctuation_dictionary(not_disaster_all_words).\
                                                                        items(), key=lambda x: x[1], reverse=True)[:15]
x1, y1 = zip(*not_disaster_most_freq_words)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
axes[0].bar(x0, y0, color='tab:blue')
axes[0].set_title('Disaster tweets')
axes[0].set_xlabel('Most frequent punctuations')
axes[0].set_ylabel('Frequency')
axes[1].bar(x1, y1, color='tab:red')
axes[1].set_title('Not disaster tweets')
axes[1].set_xlabel('Most frequent punctuations')
axes[1].set_ylabel('Frequency')
fig.savefig('figure6_freqPunctuations.png')
plt.show()
plt.close(fig)


#Graph 7: Most frequent words
x0, y0 = find_most_freq_word(disaster_all_words)
x1, y1 = find_most_freq_word(not_disaster_all_words)

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
axes[0].bar(x0, y0, color='tab:blue')
axes[0].set_title('Disaster tweets')
axes[0].set_xlabel('Most frequent words')
axes[0].set_ylabel('Frequency')
axes[1].bar(x1, y1, color='tab:red')
axes[1].set_title('Not disaster tweets')
axes[1].set_xlabel('Most frequent words')
axes[1].set_ylabel('Frequency')
fig.savefig('figure7_freqWords.png')
plt.show()
plt.close(fig)


#Graph 8: Missing values
disaster_count_of_missing_keyword = train_data[train_data['target'] == 1]['keyword'].isnull().sum();
not_disaster_count_of_missing_keyword = train_data[train_data['target'] == 0]['keyword'].isnull().sum();
disaster_count_of_missing_location = train_data[train_data['target'] == 1]['location'].isnull().sum();
not_disaster_count_of_missing_location = train_data[train_data['target'] == 0]['location'].isnull().sum();

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].bar('keyword', disaster_count_of_missing_keyword, color='tab:blue')
axes[0].bar('location', disaster_count_of_missing_location, color='tab:blue')
axes[0].set_title('Disaster tweets')
axes[0].set_ylabel('Count of missing values')
axes[1].bar('keyword', not_disaster_count_of_missing_keyword, color='tab:red')
axes[1].bar('location', not_disaster_count_of_missing_location, color='tab:red')
axes[1].set_title('Not disaster tweets')
axes[1].set_ylabel('Count of missing values')
fig.suptitle('Missing values')
fig.savefig('figure8_missingValues.png')
plt.show()
plt.close(fig)

print("End visualization")