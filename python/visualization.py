import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
import string

categories = ['Disaster', 'Not disaster']
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation


def create_all_words(target, col_name):
    """
    This function will create a list with all words in tweets with target=target

    Parameters
    ----------
    target: integer
        0 for not disaster tweets
        1 for disaster tweets
    col_name: string
        The name of the column
    Returns
	-------
	words: list of strings
		This list contains all words in tweets with target=target
    """
    words = []
    for x in train_data[train_data['target'] == target][col_name].str.split():
        if str(x) != 'nan':
            for i in x:
                words.append(i)
    return words


def create_all_fields(target, col_name):
    """
    This function will create a list with all tweets rows in tweets with target=target

    Parameters
    ----------
    target: integer
        0 for not disaster tweets
        1 for disaster tweets
    col_name: string
        The name of the column
    Returns
	-------
	words: list of strings
		This list contains all rows in tweets with target=target
    """
    words = []
    for x in train_data[train_data['target'] == target][col_name].to_list():
        if str(x) != 'nan':
            words.append(x)
    return words

def count_urls(target, col_name):
    """
    This function will count urls in tweets with target=target

    Parameters
    ----------
    target: integer
        0 for not disaster tweets
        1 for disaster tweets
    col_name: string
        The name of the column
    Returns
	-------
	count: integer
		The count of urls
    """
    count = 0
    for x in train_data[train_data['target'] == target][col_name].to_list():
        if "http" in str(x):
            count += 1
    return count


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
    cnt = Counter()
    for i in words:
        if i is not None:
            cnt[i] += 1
    return cnt


def create_stopwords_dictionary(words):
    """
    This function will create a dictionary with the frequency of appearing each stopword

    Parameters
    ----------
    words: list of strings
        This list contains all words in tweets with a specific target
    Returns
    -------
    cnt: dictionary
        This dictionary contains the frequency of appearing each word
    """
    cnt = Counter()
    for i in words:
        if i in stop_words:
            cnt[i] += 1
    return cnt

def create_punctuation_dictionary(words):
    """
    This function will create a dictionary with the frequency of appearing each punctuation

    Parameters
    ----------
    words: list of strings
        This list contains all punctuation in tweets with a specific target
    Returns
    -------
    cnt: dictionary
        This dictionary contains the frequency of appearing each punctuation
    """
    cnt = Counter()
    for i in words:
        if i in punctuations:
            cnt[i] += 1
    return cnt

def find_most_freq_nonstopwords_nonpunc(words):
    """
    This function will create a dictionary with the frequency of appearing each non-stopwords and non-punctuations

    Parameters
    ----------
    words: list of strings
        This list contains all punctuation in tweets with a specific target
    Returns
    -------
    cnt: dictionary
        This dictionary contains the frequency of non-stopwords and non-punctuations
    """
    cnt = Counter()
    for i in words:
        if (i not in stop_words) and (i not in punctuations):
            cnt[i] += 1
    return cnt


def plot_barchart(y, title, y_label, out_name):
    """
    This function will plot a bar chart

    Parameters
    ----------
    y: list of numbers
        The data to plot
    title: string
        The title of the graph
    y_label: string
        The label of y axis
    out_name: string
        The name of output file
    """
    plt.bar(0, y[0], color='tab:red')
    plt.bar(1, y[1], color='tab:green')
    plt.ylabel(y_label, fontweight='bold')
    plt.xticks(np.arange(2), categories, fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend(categories)
    plt.savefig(out_name)
    plt.show()


def plot_double_histogram(set0, set1, x_label, title, out_name, y_label='Frequency'):
    """
    This function will plot a histogram chart

    Parameters
    ----------
    set0: list of numbers
        The data to plot in first subfigure
    set1: list of numbers
        The data to plot in second subfigure
    x_label: string
        The label of x axis
    title: string
        The title of the graph
    out_name: string
        The name of output file
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].hist(set0, color='tab:red')
    axes[0].set_xlabel(x_label, fontweight='bold')
    axes[0].set_ylabel(y_label, fontweight='bold')
    axes[0].legend([categories[0]])
    axes[1].hist(set1, color='tab:green')
    axes[1].set_xlabel(x_label, fontweight='bold')
    #axes[1].set_ylabel(y_label, fontweight='bold')
    axes[1].legend([categories[1]])
    fig.suptitle(title, fontweight='bold')
    fig.savefig(out_name)
    plt.show()
    plt.close(fig)


def plot_group_barchart(x, y_disaster, y_notdisaster, title, x_label, out_name, rotate=False, degree=0,\
                        y_label='Frequency'):
    """
    This function will plot a group bar chart

    Parameters
    ----------
    x: list of string
        The x axis values
    y_disaster: list of numbers
        The data to plot for disaster tweets
    y_notdisaster: list of numbers
        The data to plot for not disaster tweets
    title: string
        The title of the graph
    x_label: string
        The label of x axis
    out_name: string
        The name of output file
    """
    pos = np.arange(len(x))
    bar_width = 0.35
    plt.bar(pos, y_disaster, bar_width, color='tab:red', edgecolor='white')
    plt.bar(pos + bar_width, y_notdisaster, bar_width, color='tab:green', edgecolor='white')
    plt.xticks(pos + bar_width / 2, list(x))
    plt.title(title, fontweight='bold')
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.legend(categories)
    if rotate:
        plt.xticks(rotation=degree)
    plt.savefig(out_name)
    plt.show()

def plot_group_barh(y, x_disaster, x_notdisaster, title, y_label, out_name, rotate=False, degree=0,\
                    x_label='Frequency'):
    """
    This function will plot a group barh chart

    Parameters
    ----------
    y: list of string
        The y axis values
    x_disaster: list of numbers
        The data to plot for disaster tweets
    x_notdisaster: list of numbers
        The data to plot for not disaster tweets
    title: string
        The title of the graph
    y_label: string
        The label of y axis
    out_name: string
        The name of output file
    """
    pos = np.arange(len(y))
    bar_width = 0.35
    plt.figure(figsize=(15, 30))
    plt.barh(pos, x_disaster, bar_width, color='tab:red', edgecolor='white')
    plt.barh(pos + bar_width, x_notdisaster, bar_width, color='tab:green', edgecolor='white')
    plt.yticks(pos + bar_width / 2, list(y))
    plt.title(title, fontweight='bold')
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.legend(categories)
    if rotate:
        plt.xticks(rotation=degree)
    plt.savefig(out_name)
    plt.show()


def find_correspond_values(most_freq_x, dict_y):
    """
    This function will find corresponding values between disaster and not sets

    Parameters
    ----------
    most_freq_x: list of strings
        This list contains the most frequent words in x set
    dict_y: counter
        This contains the dictionary for y set
    Returns
    -------
    results: list of strings
        This list contains the corresponding frequency
    """
    results = []
    for i in most_freq_x:
        results.append(dict_y[i[0]])
    return results


#MAIN
#Read the data
train_data = pd.read_csv('dataset/train_processed_all.csv')
train_data.set_index('id', inplace=True)

#Graph1: Tweets
count_disaster = train_data[train_data['target'] == 1].shape[0]
count_not_disaster = train_data[train_data['target'] == 0].shape[0]
#print(count_disaster)
#print(count_not_disaster)
plot_barchart([count_disaster, count_not_disaster], 'Tweets', 'Number of samples', 'fig01_tweets')

#Graph 2: Missing values
disaster_count_of_missing_keyword = train_data[train_data['target'] == 1]['keyword'].isnull().sum();
not_disaster_count_of_missing_keyword = train_data[train_data['target'] == 0]['keyword'].isnull().sum();
disaster_count_of_missing_location = train_data[train_data['target'] == 1]['location'].isnull().sum();
not_disaster_count_of_missing_location = train_data[train_data['target'] == 0]['location'].isnull().sum();

plot_group_barchart(['keyword', 'location'], [disaster_count_of_missing_keyword, disaster_count_of_missing_location],\
                    [not_disaster_count_of_missing_keyword, not_disaster_count_of_missing_location], 'Missing values',\
                    '', 'fig02_missing_values')

#Graph3: Characters in tweets
disaster_tweet_chars = train_data[train_data['target'] == 1]['text'].str.len()
not_disaster_tweet_chars = train_data[train_data['target'] == 0]['text'].str.len()

plot_double_histogram(disaster_tweet_chars, not_disaster_tweet_chars, 'Number of characters in a tweet',\
                      'Characters in tweets (Histogram)', 'fig03_chars.png')

#Graph4: Words in tweets
disaster_tweet_words = train_data[train_data['target'] == 1]['text'].str.split().map(lambda x: len(x))
not_disaster_tweet_words = train_data[train_data['target'] == 0]['text'].str.split().map(lambda x: len(x))
#print(sum(disaster_tweet_words.to_list())/len(disaster_tweet_words.to_list()))
#print(sum(not_disaster_tweet_words.to_list())/len(not_disaster_tweet_words.to_list()))

plot_double_histogram(disaster_tweet_words, not_disaster_tweet_words, 'Number of words in a tweet',\
                      'Words in tweets (Histogram)', 'fig04_words.png')

#Graph5: Average word lenght in each tweets
disaster_tweet_word_len = train_data[train_data['target'] == 1]['text'].str.split().\
                                                        apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))
not_disaster_tweet_word_len = train_data[train_data['target'] == 0]['text'].str.split().\
                                                        apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x))

plot_double_histogram(disaster_tweet_word_len, not_disaster_tweet_word_len, 'Average word length',\
                      'Average word length in each tweets (Histogram)', 'fig05_word_len.png')

#Graph 6 & 7: Most frequent stopwords

all_words_processed_disaster = create_all_words(1, 'processed')
all_words_processed_not_disaster = create_all_words(0, 'processed')
dict_all_words_disaster = create_stopwords_dictionary(all_words_processed_disaster)
dict_all_words_not_disaster = create_stopwords_dictionary(all_words_processed_not_disaster)
most_freq_words_disaster = dict_all_words_disaster.most_common(15)
most_freq_words_not_disaster = dict_all_words_not_disaster.most_common(15)
disaster_x0, disaster_y0 = zip(*most_freq_words_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_words_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_words_disaster, dict_all_words_not_disaster)
disaster_y1 = find_correspond_values(most_freq_words_not_disaster, dict_all_words_disaster)

plot_group_barchart(disaster_x0, disaster_y0, not_disaster_y0, 'Stopwords',\
                    'Most frequent stopwords in disaster tweets', 'fig06_stopwords_disasters')

plot_group_barchart(not_disaster_x1, disaster_y1, not_disaster_y1, 'Stopwords',\
                    'Most frequent stopwords in not disaster tweets', 'fig07_stopwords_not_disasters')


#Graph 8 & 9: Most frequent punctuations
all_words_text_disaster = create_all_words(1, 'text')
all_words_text_not_disaster = create_all_words(0, 'text')
dict_punctuation_disaster = create_punctuation_dictionary(all_words_text_disaster)
dict_punctuation_not_disaster = create_punctuation_dictionary(all_words_text_not_disaster)
most_freq_punctuation_disaster = dict_punctuation_disaster.most_common(15)
most_freq_punctuation_not_disaster = dict_punctuation_not_disaster.most_common(15)
disaster_x0, disaster_y0 = zip(*most_freq_punctuation_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_punctuation_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_punctuation_disaster, dict_punctuation_not_disaster)
disaster_y1 = find_correspond_values(most_freq_punctuation_not_disaster, dict_punctuation_disaster)

plot_group_barchart(disaster_x0, disaster_y0, not_disaster_y0, 'Punctuations',\
                    'Most frequent punctuations in disaster tweets', 'fig08_punctuations_disasters')

plot_group_barchart(not_disaster_x1, disaster_y1, not_disaster_y1, 'Punctuations',\
                    'Most frequent punctuations in not disaster tweets', 'fig09_punctuations_not_disasters')

#Graph 10 & 11: Most frequent non-stopwaords and non-punctuations

dict_words_disaster = find_most_freq_nonstopwords_nonpunc(all_words_processed_disaster)
dict_words_not_disaster = find_most_freq_nonstopwords_nonpunc(all_words_processed_not_disaster)
most_freq_words_disaster = dict_words_disaster.most_common(16)[1:]
most_freq_words_not_disaster = dict_words_not_disaster.most_common(16)[1:]
disaster_x0, disaster_y0 = zip(*most_freq_words_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_words_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_words_disaster, dict_words_not_disaster)
disaster_y1 = find_correspond_values(most_freq_words_not_disaster, dict_words_disaster)

plot_group_barchart(disaster_x0, disaster_y0, not_disaster_y0, 'Non-stopwords and non-punctuations words',\
                    'Most frequent words in disaster tweets', 'fig10_words_disasters', rotate=True, degree=30)

plot_group_barchart(not_disaster_x1, disaster_y1, not_disaster_y1, 'Non-stopwords and non-punctuations words',\
                    'Most frequent words in not disaster tweets', 'fig11_words_not_disasters', rotate=True, degree=30)


#Graph 12 & 13: Most frequent keywords
all_words_keyword_disaster = create_all_words(1, 'keyword')
all_words_keyword_not_disaster = create_all_words(0, 'keyword')
dict_keyword_disaster = create_dictionary(all_words_keyword_disaster)
dict_keyword_not_disaster = create_dictionary(all_words_keyword_not_disaster)
most_freq_keyword_disaster = dict_keyword_disaster.most_common(15)
most_freq_keyword_not_disaster = dict_keyword_not_disaster.most_common(15)
disaster_x0, disaster_y0 = zip(*most_freq_keyword_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_keyword_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_keyword_disaster, dict_keyword_not_disaster)
disaster_y1 = find_correspond_values(most_freq_keyword_not_disaster, dict_keyword_disaster)

plot_group_barchart(disaster_x0, disaster_y0, not_disaster_y0, 'Keywords',\
                    'Most frequent keywords in disaster tweets', 'fig12_keywords_disasters', rotate=True, degree=90)

plot_group_barchart(not_disaster_x1, disaster_y1, not_disaster_y1, 'Keywords',\
                    'Most frequent keywords in not disaster tweets', 'fig13_keywords_not_disasters', rotate=True,\
                    degree=90)

#Graph 12b & 13b: Most frequent keywords
most_freq_keyword_disaster = dict_keyword_disaster.most_common(90)
most_freq_keyword_not_disaster = dict_keyword_not_disaster.most_common(90)
disaster_x0, disaster_y0 = zip(*most_freq_keyword_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_keyword_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_keyword_disaster, dict_keyword_not_disaster)
disaster_y1 = find_correspond_values(most_freq_keyword_not_disaster, dict_keyword_disaster)
plot_group_barh(disaster_x0, disaster_y0, not_disaster_y0, 'Keywords',\
                    'Most frequent keywords in disaster tweets', 'fig12b_keywords_disasters', rotate=True, degree=90)

plot_group_barh(not_disaster_x1, disaster_y1, not_disaster_y1, 'Keywords',\
                    'Most frequent keywords in not disaster tweets', 'fig13b_keywords_not_disasters', rotate=True,\
                    degree=90)

#Graph 14 & 15: Most frequent locations
all_words_locations_disaster = create_all_fields(1, 'location_processed')
all_words_locations_not_disaster = create_all_fields(0, 'location_processed')
dict_locations_disaster = create_dictionary(all_words_locations_disaster)
dict_locations_not_disaster = create_dictionary(all_words_locations_not_disaster)
most_freq_locations_disaster = dict_locations_disaster.most_common(15)
most_freq_locations_not_disaster = dict_locations_not_disaster.most_common(15)
disaster_x0, disaster_y0 = zip(*most_freq_locations_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_locations_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_locations_disaster, dict_locations_not_disaster)
disaster_y1 = find_correspond_values(most_freq_locations_not_disaster, dict_locations_disaster)

plot_group_barchart(disaster_x0, disaster_y0, not_disaster_y0, 'Locations',\
                    'Most frequent locations in disaster tweets', 'fig14_locations_disasters', rotate=True, degree=90)

plot_group_barchart(not_disaster_x1, disaster_y1, not_disaster_y1, 'Locations',\
                    'Most frequent locations in not disaster tweets', 'fig15_locations_not_disasters', rotate=True,\
                    degree=90)

#Graph 14b & 15b: Most frequent locations
most_freq_locations_disaster = dict_locations_disaster.most_common(100)
most_freq_locations_not_disaster = dict_locations_not_disaster.most_common(100)
disaster_x0, disaster_y0 = zip(*most_freq_locations_disaster)
not_disaster_x1, not_disaster_y1 = zip(*most_freq_locations_not_disaster)
not_disaster_y0 = find_correspond_values(most_freq_locations_disaster, dict_locations_not_disaster)
disaster_y1 = find_correspond_values(most_freq_locations_not_disaster, dict_locations_disaster)

plot_group_barh(disaster_x0, disaster_y0, not_disaster_y0, 'Locations',\
                    'Most frequent locations in disaster tweets', 'fig14b_locations_disasters', rotate=True, degree=90)

plot_group_barh(not_disaster_x1, disaster_y1, not_disaster_y1, 'Locations',\
                    'Most frequent locations in not disaster tweets', 'fig15b_locations_not_disasters', rotate=True,\
                    degree=90)

#Graph 16: URLs
count_url_disaster = count_urls(1, 'text')
count_url_not_disaster = count_urls(0, 'text')
#print(count_url_disaster)
#print(count_url_not_disaster)
plot_barchart([count_url_disaster, count_url_not_disaster], 'URLs', 'Number of tweets with URL', 'fig16_url')

#Graph 17 & 18: Our dataset
our_train_data = pd.read_csv('dataset/our_train.csv')
count_disaster = our_train_data[our_train_data['target'] == 1].shape[0]
count_not_disaster = our_train_data[our_train_data['target'] == 0].shape[0]
plot_barchart([count_disaster, count_not_disaster], 'Our train set', 'Number of samples', 'fig17_our_train')

our_test_data = pd.read_csv('dataset/our_test.csv')
count_disaster = our_test_data[our_test_data['target'] == 1].shape[0]
count_not_disaster = our_test_data[our_test_data['target'] == 0].shape[0]
plot_barchart([count_disaster, count_not_disaster], 'Our test set', 'Number of samples', 'fig18_our_test')