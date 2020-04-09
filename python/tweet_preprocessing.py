# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

import string
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS # reference https://github.com/NeelShah18/emot

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import requests
from bs4 import BeautifulSoup
from urlmarker import URL_REGEX # reference https://gist.github.com/gruber/8891611

import inflect

from spellchecker import SpellChecker


# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# %%
train_df = pd.read_csv('../dataset/train.csv')
train_df.set_index('id', inplace=True)
train_df


# %%
abbreviation_dict = {}
with open("../dictionaries/abbreviations.txt") as f:
    for line in f:
       (key, val) = line.split('\t')
       abbreviation_dict[(key)] = val.replace('\n', '')

contraction_dict = {}
with open("../dictionaries/contractions.txt") as f:
    for line in f:
       (key, val) = line.split(':')
       contraction_dict[(key)] = val.replace('\n', '')

# %% [markdown]
# # URL related functions

# %%
def removeURLs(tweet):
    """
    Replaces URLs in the tweet given with the string 'URL'.
    
    Parameters:
        tweet (string): tweet to be processed.

    Returns:
        string: given tweet with the URLs removed.
    """
    tweet = re.sub(URL_REGEX, 'URL', tweet)
    return tweet

def listURLs(tweet):
    """
    Returns a list of URLs contained in the given tweet.
            
    Parameters:
        tweet (string): tweet to be processed.

    Returns: 
        list: a list of URLs.
    """
    return re.findall(URL_REGEX, tweet)

def extractTextFromURLs(urls):
    """
    Returns text from the given list of URL filtering out some HTML tags.
        
    Parameters:
        url (list): list of URL to be processed.

    Returns: 
        string: text extracted from the given URLs.
    """
    extracted = ''
    for url in urls:
        res = requests.get(url)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)
        
        undesired = ['[document]', 'noscript',
	                'header', 'html',
	                'meta', 'head', 
                    'input', 'script',
                    'style', 'title']
        for t in text:
	        if t.parent.name not in undesired:
		        extracted += '{} '.format(t)

    return extracted

# %% [markdown]
# # Remove unwanted elements

# %%
def removeNonAscii(tweet):
    """
    Removes non ascii characters from given string.

    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with non ascii characters removed.    
    """
    return tweet.encode('ascii', 'ignore').decode('ascii')

def removeNonPrintable(tweet):
    """
    Removes non printable characters from given string.

    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with non printable removed.    
    """
    return ''.join(filter(lambda x: x in string.printable, tweet))

def removePunctuation(tweet):
    """
    Removes punctuations (removes # as well).

    Parameters:
        tweet (string): tweet to be processed.
    
    Returns:
        string: given tweet with punctuations removed.
    """
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return tweet.translate(translator)

def removeNums(tweet):
    """
    Removes numeric values from the given string.
    
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with numeric values removed.    
    """
    return ''.join([char for char in tweet if not char.isdigit()])

def removeUsernames(tweet):
    """
    Removes usernames from given tweet.
    
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with usernames removed.   
    """
    return re.sub('@[^\s]+', '', tweet)

# %% [markdown]
# # Format related functions

# %%
def toLowerCase(tweet):
    """
    Separate camelCase to space delimited and convert tweet to lower-case.
    
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet to lower case.
    """
    tweet = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', tweet)
    tweet = tweet.lower()
    return tweet

# %% [markdown]
# # Meaning related functions

# %%
def replaceEmojis(tweet):
    """
    Replace emojis in the text with their correspinding meaning.
    
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with emojis replaced.  
    """
    for emot in UNICODE_EMO:
        tweet = tweet.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return tweet

def replaceEmoticons(tweet):
    """
    Replace emoticons in the text with their correspinding meaning.
    
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with emoticons replaced.  
    """
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), tweet)
    return tweet

def replaceNums(tweet):
    """
    Replace numerical values with their textual representation.
        
    Parameters:
        tweet (string): tweet to be processed.
    
    Returns: 
        string: given tweet with numerical values replaced.
    """
    infeng = inflect.engine()
    processed_tweet = []
    for word in tweet.split():
         processed_tweet.append(infeng.number_to_words(word) if word.isdigit() else word)
    return ' '.join(processed_tweet)          

def correctSpelling(tweet_list):
    """
    Corrects spelling in the given string.
    
    Parameters:
        tweet_list (list): list of string-words to be processed.
    
    Returns: 
        list: given tweet-list spelling-corrected.
    """
    spell = SpellChecker()
    spell.word_frequency.load_words(['url']) # add url to the dictionary
    # find those words that may be misspelled
    misspelled = spell.unknown(tweet_list)
    processed_tweet = []
    for word in tweet_list:
        # Replaced misspelled with the one most likely answer
        processed_tweet.append(spell.correction(word) if word in misspelled else word)
    return processed_tweet

def replaceAbbreviations(tweet_list, abbreviation_dict):
    """
    Replaces abbreviation with the corresponding full text from dictionary.
    
    Parameters:
        tweet_list (list): list of string-words to be processed.
        abbreviation_dict (dictionary): dictionary of abbreviation.
    
    Returns: 
        list: given tweet-list with the abbreviations replaced.
    """
    processed_list = []
    for word in tweet_list:
        if word in abbreviation_dict:
            if len(abbreviation_dict.get(word).split()) > 1: # in case of multiple words
                processed_list.extend(abbreviation_dict.get(word).split())
            else:
                processed_list.append(abbreviation_dict.get(word))
        else:
            processed_list.append(word)
    return processed_list   

def replaceContractions(tweet_list, contraction_dict):
    """
    Replaces contractions with the corresponding full text from dictionary.
        
    Parameters:
        tweet_list (list): list of string-words to be processed.
        contraction_dict (dictionary): dictionary of contractions.
    
    Returns: 
        list: given tweet-list with the contractions replaced.
    """
    processed_list = []
    for word in tweet_list:
        if word in contraction_dict:
            if len(contraction_dict.get(word).split()) > 1: # in case of multiple words
                processed_list.extend(contraction_dict.get(word).split())
            else:
                processed_list.append(contraction_dict.get(word))
        else:
            processed_list.append(word)
    return processed_list 

def removeStopWords(tweet_list):
    """
    Removes stop-words from the given tweet.
        
    Parameters:
        tweet_list (list): list of string-words to be processed.
    
    Returns: 
        list: given tweet with stop-words removed.
    """
    return [word for word in tweet_list if word not in stopwords.words('english')]

def stemming(tweet_list):
    """
    Stemming - reduces the word-forms by removing suffixes.

    Parameters:
        tweet_list (list): list of string-words to be processed.

    Returns: 
        list: given tweet stemmed.
    """
    return [PorterStemmer().stem(word) for word in tweet_list]

def lemmatization(tweet_list):
    """
    Lemmatization - reduces the word-forms to linguistically valid lemmas.

    Parameters:
        tweet_list (list): list of string-words to be processed.

    Returns: 
        list: given tweet lemmatized.
    """
    return [WordNetLemmatizer().lemmatize(word) for word in tweet_list]


# %%
def preprocess_tweet(tweet, abbreviation_dict, contraction_dict):
    tweet = replaceEmojis(tweet)
    tweet = removeNonAscii(tweet)
    tweet = removeURLs(tweet)
    tweet = removeUsernames(tweet)
    tweet = removeNonPrintable(tweet)
    
    tweet = toLowerCase(tweet)

    tweet_list = tweet.split()
    tweet_list = replaceAbbreviations(tweet_list, abbreviation_dict)
    tweet_list = replaceContractions(tweet_list, contraction_dict)

    tweet_list = (removeNums(' '.join(tweet_list))).split()
    tweet_list = (removePunctuation(' '.join(tweet_list))).split()
    
    tweet_list = correctSpelling(tweet_list)
    
    tweet_list = removeStopWords(tweet_list)
    tweet_list = lemmatization(tweet_list)
    tweet_list = stemming(tweet_list)
    return tweet_list


# %%
for index, row in train_df.iterrows():
    train_df.at[index, 'processed_text'] = ' '.join(preprocess_tweet(row['text'], abbreviation_dict, contraction_dict))
    try:
        train_df.at[index, 'processed_URLs'] = ' '.join(preprocess_tweet(' '.join([word for word in extractTextFromURLs(listURLs(row['text'])).split() if all(c in string.printable and not c.isdigit() for c in word) and len(word)>3]), abbreviation_dict, contraction_dict))
    except:
        print('Error reaching URL for record #{}'.format(index))    
    print("record #{} processing finished".format(index))

train_df.to_csv('../dataset/train_processed.csv')

