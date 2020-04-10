import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import string


def create_all_words(target):
    words = []
    for x in train_data[train_data['target'] == target]['text'].str.split():
        for i in x:
            words.append(i)
    return words


def create_dictionary(words):
    dic = defaultdict(int)
    for word in words:
        if word in stop_words:
            dic[word] += 1
    return dic


def create_punctuation_dictionary(words):
    dic = defaultdict(int)
    special = string.punctuation
    for i in words:
        if i in special:
            dic[i] += 1
    return dic


train_data = pd.read_csv('../dataset/train.csv')
test_data = pd.read_csv('../dataset/test.csv')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
train_data.set_index('id', inplace=True)
print(train_data['location'])
'''
count_row = train_data.shape[0]  # gives number of row count
count_col = train_data.shape[1]  # gives number of col count
print("Total number of rows = " + str(count_row))
print("Total number of columns = " + str(count_col))

count_disaster = train_data[train_data['target'] == 1].shape[0]
count_not_disaster = train_data[train_data['target'] == 0].shape[0]

#Graph1: Tweets
plt.bar("Disaster tweets", count_disaster, color='tab:blue')
plt.bar("Not disaster tweets", count_not_disaster, color='tab:red')
plt.ylabel('Number of samples')
plt.title('Tweets')
plt.show()

#Graph2: Characters in tweets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.len()
ax1.hist(disaster_tweet_chars, color='tab:blue')
ax1.set_xlabel('Number of characters in a tweet')
ax1.set_ylabel('Frequency')
ax1.set_title('Disaster tweets')
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.len()
ax2.hist(not_disaster_tweet_chars, color='tab:red')
ax2.set_xlabel('Number of characters in a tweet')
ax2.set_ylabel('Frequency')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()

#Graph3: Words in tweets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(disaster_tweet_chars, color='tab:blue')
ax1.set_xlabel('Number of words in a tweet')
ax1.set_ylabel('Frequency')
ax1.set_title('Disaster tweets')
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.split().map(lambda x: len(x))
ax2.hist(not_disaster_tweet_chars, color='tab:red')
ax2.set_xlabel('Number of words in a tweet')
ax2.set_ylabel('Frequency')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in tweets')
plt.show()

#Graph4: Average word lenght in each tweets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))
ax1.hist(disaster_tweet_chars, color='tab:blue')
ax1.set_xlabel('Average word length')
ax1.set_ylabel('Frequency')
ax1.set_title('Disaster tweets')
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))
ax2.hist(not_disaster_tweet_chars, color='tab:red')
ax2.set_xlabel('Average word length')
ax2.set_ylabel('Frequency')
ax2.set_title('Not disaster tweets')
fig.suptitle('Average word length in each tweets')
plt.show()

disaster_all_words = create_all_words(1)
not_disaster_all_words = create_all_words(0)

#Graph 5: Most frequent stopwords
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
disaster_most_freq_words = sorted(create_dictionary(disaster_all_words).items(), key=lambda x: x[1], reverse=True)[:15]
x, y = zip(*disaster_most_freq_words)
ax1.bar(x, y, color='tab:blue')
ax1.set_title('Disaster tweets')
ax1.set_xlabel('Most frequent stopwords')
ax1.set_ylabel('Frequency')
not_disaster_most_freq_words = sorted(create_dictionary(not_disaster_all_words).items(), key=lambda x: x[1], reverse=True)[:15]
x, y = zip(*not_disaster_most_freq_words)
ax2.bar(x,y, color='tab:red')
ax2.set_title('Not disaster tweets')
ax2.set_xlabel('Most frequent stopwords')
ax2.set_ylabel('Frequency')
plt.show()

#Graph 6: Most frequent punctuations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
disaster_most_freq_words = sorted(create_punctuation_dictionary(disaster_all_words).items(), key=lambda x: x[1], reverse=True)[:15]
x, y = zip(*disaster_most_freq_words)
ax1.bar(x, y, color='tab:blue')
ax1.set_title('Disaster tweets')
ax1.set_xlabel('Most frequent punctuations')
ax1.set_ylabel('Frequency')
not_disaster_most_freq_words = sorted(create_punctuation_dictionary(not_disaster_all_words).items(), key=lambda x: x[1], reverse=True)[:15]
x, y = zip(*not_disaster_most_freq_words)
ax2.bar(x,y, color='tab:red')
ax2.set_title('Not disaster tweets')
ax2.set_xlabel('Most frequent punctuations')
ax2.set_ylabel('Frequency')
plt.show()

#Graph 7: Most frequent words
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
counter = Counter(disaster_all_words)
most = counter.most_common()
x = []
y = []
for word, count in most[:40]:
    if (word not in stop_words):
        x.append(word)
        y.append(count)
ax1.bar(x, y, color='tab:blue')
ax1.set_title('Disaster tweets')
ax1.set_xlabel('Most frequent words')
ax1.set_ylabel('Frequency')
counter = Counter(not_disaster_all_words)
most = counter.most_common()
x = []
y = []
for word, count in most[:40]:
    if (word not in stop_words):
        x.append(word)
        y.append(count)
ax2.bar(x, y, color='tab:red')
ax2.set_title('Not disaster tweets')
ax2.set_xlabel('Most frequent words')
ax2.set_ylabel('Frequency')
plt.show()

#Graph 7: Missing values
disaster_count_of_missing_keyword = train_data[train_data['target'] == 1]['keyword'].isnull().sum();
not_disaster_count_of_missing_keyword = train_data[train_data['target'] == 0]['keyword'].isnull().sum();
disaster_count_of_missing_location = train_data[train_data['target'] == 1]['location'].isnull().sum();
not_disaster_count_of_missing_location = train_data[train_data['target'] == 0]['location'].isnull().sum();
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar('keyword', disaster_count_of_missing_keyword, color='tab:blue')
ax1.bar('location', disaster_count_of_missing_location, color='tab:blue')
ax1.set_title('Disaster tweets')
ax1.set_ylabel('Count of missing values')
ax2.bar('keyword', not_disaster_count_of_missing_keyword, color='tab:red')
ax2.bar('location', not_disaster_count_of_missing_location, color='tab:red')
ax2.set_title('Not disaster tweets')
ax2.set_ylabel('Count of missing values')
fig.suptitle('Missing values')
plt.show()

'''