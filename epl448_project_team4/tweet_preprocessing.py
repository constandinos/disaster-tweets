import pandas as pd

import string
import re
import html

# reference: https://github.com/NeelShah18/emot 
# pip install emot --upgrade
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# reference: https://github.com/nltk/nltk
# pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# reference: https://github.com/psf/requests
# pip install requests
import requests
# reference: https://pypi.org/project/beautifulsoup4/
# pip install beautifulsoup4 
from bs4 import BeautifulSoup

# reference https://gist.github.com/gruber/8891611
from urlmarker import URL_REGEX 

# reference: https://pypi.org/project/inflect/
# pip install inflect
import inflect

# reference: https://pypi.org/project/pyspellchecker/
# pip install pyspellchecker
from spellchecker import SpellChecker

# reference: https://github.com/cbaziotis/ekphrasis
# pip install ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict


# ============================================================================ #
#                          URL related functions                               #
# ============================================================================ #

def removeURLs(tweet):
    """
    Removes URLs in the tweet given.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    -------
    str
        given tweet with the URLs removed.
    """
    tweet = re.sub(URL_REGEX, '', tweet)
    return tweet


def listURLs(tweet):
    """
    Returns a list of URLs contained in the given tweet.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
    list of str
        a list of URLs.
    """
    return re.findall(URL_REGEX, tweet)


def extractTextFromURLs(urls):
    """
    Returns text from the given list of URL filtering out some HTML tags.

    Parameters
    ----------
    url : list of str
        list of URL to be processed.

    Returns 
    -------
    str
        text extracted from the given URLs.
    """
    extracted = ''
    for url in urls:
        try:
            res = requests.get(url)
        except Exception as e:
            print(e)
            continue

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


# ============================================================================ #
#                        Remove unwanted elements                              #
# ============================================================================ #

def replaceHTMLChar(tweet):
    """
    Convert all named and numeric character references 
    (e.g. &gt;, &#62;, &#x3e;) in the string s to the 
    corresponding Unicode characters.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    -------
    str
        given tweet with non html characters replaced. 
    """
    return html.unescape(tweet)


def removeNonAscii(tweet):
    """
    Removes non ascii characters from given string.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    -------
    str
        given tweet with non ascii characters removed.    
    """
    return tweet.encode('ascii', 'ignore').decode('ascii')


def removeNonPrintable(tweet):
    """
    Removes non printable characters from given string.

    Parameters
    ----------
     tweet : str
        tweet to be processed.

    Returns
    ------- 
    str
        given tweet with non printable removed.    
    """
    return ''.join(filter(lambda x: x in string.printable, tweet))


def removePunctuation(tweet):
    """
    Removes punctuations (removes # as well).

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    -------
    str
        given tweet with punctuations removed.
    """
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return tweet.translate(translator)


def removeNums(tweet):
    """
    Removes numeric values from the given string.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
     str
        given tweet with numeric values removed.    
    """
    return ''.join([char for char in tweet if not char.isdigit()])


def removeUsernames(tweet):
    """
    Removes usernames from given tweet.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
    str
        given tweet with usernames removed.   
    """
    return re.sub('@[^\s]+', '', tweet)


def removeRepeatedChars(tweet):
    """
    Reduces repeated consecutive characters from given tweet to only two.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns: 
        string: given tweet with repeated characters removed.   
    """
    return re.sub(r'(.)\1+', r'\1\1', tweet)


# ============================================================================ #
#                           Format related functions                           #
# ============================================================================ #

def toLowerCase(tweet):
    """
    Separate camelCase to space delimited and convert tweet to lower-case.

    Parameters
    ----------
    tweet : str 
        tweet to be processed.

    Returns
    ------- 
    str
        given tweet to lower case.
    """
    tweet = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', tweet)
    tweet = tweet.lower()
    return tweet


# ============================================================================ #
#                           Meaning related functions                          #
# ============================================================================ #

def replaceEmojis(tweet):
    """
    Replace emojis in the text with their correspinding meaning.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
    str
        given tweet with emojis replaced.  
    """
    for emot in UNICODE_EMO:
        tweet = tweet.replace(emot, "_".join(
            UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return tweet


def replaceEmoticons(tweet):
    """
    Replace emoticons in the text with their correspinding meaning.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
     str
        given tweet with emoticons replaced.  
    """
    for emot in EMOTICONS:
        tweet = re.sub(
            u'('+emot+')', "_".join(EMOTICONS[emot].replace(",", "").split()), tweet)
    return tweet


def replaceNums(tweet):
    """
    Replace numerical values with their textual representation.

    Parameters
    ----------
    tweet : str
        tweet to be processed.

    Returns
    ------- 
    str
        given tweet with numerical values replaced.
    """
    infeng = inflect.engine()
    processed_tweet = []
    for word in tweet.split():
        processed_tweet.append(infeng.number_to_words(
            word) if word.isdigit() else word)
    return ' '.join(processed_tweet)


def correctSpelling(tweet_list):
    """
    Corrects spelling in the given string.

    Parameters
    ----------
    tweet_list : list of str
        list of string-words to be processed.

    Returns
    -------
    list of str
        given tweet-list spelling-corrected.
    """
    spell = SpellChecker()
    spell.word_frequency.load_words(['url'])  # add url to the dictionary
    # find those words that may be misspelled
    misspelled = spell.unknown(tweet_list)
    processed_tweet = []
    for word in tweet_list:
        # Replaced misspelled with the one most likely answer
        processed_tweet.append(spell.correction(
            word) if word in misspelled else word)
    return processed_tweet


def replaceFromDictionary(tweet_list, dictionary):
    """
    Replaces words included in the dictionary given with their corresponding value.

    Parameters
    ----------
    tweet_list : list of str
        list of string-words to be processed.
    dictionary : dictionary
        dictionary of words to be replaced with the correspinding value.

    Returns: 
    --------
    list of str
        given tweet-list with the words inlcuded in the dictionary provided replaced.
    """
    processed_list = []
    for word in tweet_list:
        if word in dictionary:
            if len(dictionary.get(word).split()) > 1:  # in case of multiple words in value
                processed_list.extend(dictionary.get(word).split())
            else:
                processed_list.append(dictionary.get(word))
        else:
            processed_list.append(word)
    return processed_list

def removeStopWords(tweet_list):
    """
    Removes stop-words from the given tweet.

    Parameters
    ----------
    tweet_list : list of str
        list of string-words to be processed.

    Returns
    -------
    list of str
        given tweet with stop-words removed.
    """
    return [word for word in tweet_list if word not in stopwords.words('english')]


def stemming(tweet_list):
    """
    Stemming - reduces the word-forms by removing suffixes.

    Parameters
    ----------
    tweet_list : list of str
        list of string-words to be processed.

    Returns
    -------
    list of str
        given tweet stemmed.
    """
    return [PorterStemmer().stem(word) for word in tweet_list]


def lemmatization(tweet_list):
    """
    Lemmatization - reduces the word-forms to linguistically valid lemmas.

    Parameters
    ----------
    tweet_list : list of str
        list of string-words to be processed.

    Returns
    ------- 
    list of str
        given tweet lemmatized.
    """
    return [WordNetLemmatizer().lemmatize(word) for word in tweet_list]


# ============================================================================ #
#                        Main Preprocessing Function                           #
# ============================================================================ #

def preprocess_tweet(tweet, remove_url = True, remove_usernames = True, 
                     replace_emojis = True, replace_emoticons = True,
                     remove_non_ascii = True, remove_non_printable = True,
                     remove_repeated = True, replace_html_char = True,
                     to_lower_case = True, text_processor = None, 
                     dictionaries = None, # list of dictionaries
                     remove_nums = True, remove_pun = True,
                     spelling = True, remove_stopwords = True,
                     lemmatization = False, stemming = False):
    """
    Apply preprocessing on the given tweet based on the parameters 
    choosen.

    Parameters
    ----------
    tweet : str
        tweet to be preocessed.

    Returns
    -------
    list of str
        list of words after preprocessing.
    """
    if remove_url:          tweet = removeURLs(tweet)
    if replace_html_char:   tweet = replaceHTMLChar(tweet)

    if remove_usernames:    tweet = removeUsernames(tweet)

    if replaceEmojis:       tweet = replaceEmojis(tweet)
    if replaceEmoticons:    tweet = replaceEmoticons(tweet)

    if remove_non_ascii:    tweet = removeNonAscii(tweet)
    if remove_non_printable:tweet = removeNonPrintable(tweet)
    if remove_repeated:     tweet = removeRepeatedChars(tweet)
    if to_lower_case:       tweet = toLowerCase(tweet)
    
    if text_processor:      tweet = text_processor.pre_process_doc(tweet)

    tweet_list = tweet.split()

    if dictionaries:   
        for d in dictionaries:
            tweet_list = replaceFromDictionary(tweet_list, d)

    if remove_nums:         tweet_list = (removeNums(' '.join(tweet_list))).split()
    if remove_pun:          tweet_list = (removePunctuation(' '.join(tweet_list))).split()
    if spelling:            tweet_list = correctSpelling(tweet_list)

    if remove_stopwords:    tweet_list = removeStopWords(tweet_list)

    if lemmatization:       tweet_list = lemmatization(tweet_list)
    elif stemming:          tweet_list = stemming(tweet_list)

    return tweet_list


if __name__ == "__main__":
    # Load data
    abbreviation_dict = {}
    with open("dictionaries/abbreviations.txt") as f:
        for line in f:
            (key, val) = line.split('\t')
            abbreviation_dict[(key)] = val.replace('\n', '')

    contraction_dict = {}
    with open("dictionaries/contractions.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            contraction_dict[(key)] = val.replace('\n', '')

    states_dict = {}
    with open("dictionaries/states.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            states_dict[(key)] = val.replace('\n', '')

    location_fixes_dict = {}
    with open("dictionaries/location_fixes.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            location_fixes_dict[(key)] = val.replace('\n', '')

    train_df = pd.read_csv('dataset/train.csv')
    train_df.set_index('id', inplace=True)

    # set ekphrasis text preprocessor object
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        fix_html=True,  # fix HTML tokens
    
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
    
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
    
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words
    
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons, ] # slangdict
    )
    # tags used in ekphrasis preprocessing
    tags = ['<url>', '<email>', '<percent>', '<money>', '<phone>', '<user>', '<time>', '<date>', '<number>']

    # preprocessor to expand hashtags
    hashtag_preprocessor = TextPreProcessor(unpack_hashtags=True)
    # Process tweets
    train_df['keyword'] = train_df['keyword'].astype(str)
    train_df['location'] = train_df['location'].astype(str)

    for index, row in train_df.iterrows():
        # keyword preprocesing
        train_df.at[index, 'keyword'] = row['keyword'].replace('%20', ' ')

        # location preprocessing
        if not row['location'] == row['location']: 
            train_df.at[index, 'location_processed'] = 'nan'
        else:
            train_df.at[index, 'location_processed'] = ' '.join(preprocess_tweet(row['location'],
                                                                            dictionaries=[states_dict, location_fixes_dict],
                                                                            text_processor=hashtag_preprocessor,
                                                                            remove_stopwords=False))
        # text preprocessing
        processed = preprocess_tweet(row['text'],
                                     dictionaries=[abbreviation_dict, contraction_dict],
                                     text_processor=hashtag_preprocessor,
                                     remove_stopwords=False)

        train_df.at[index, 'processed'] = ' '.join(processed)
        processed = removeStopWords(processed)
        train_df.at[index, 'lemmatization'] = ' '.join(lemmatization(processed))
        train_df.at[index, 'stemming'] = ' '.join(stemming(processed))

        train_df.at[index, 'no_punc_no_abb'] = ' '.join(preprocess_tweet(row['text'],
                                                                         dictionaries=[contraction_dict],
                                                                         text_processor=hashtag_preprocessor,
                                                                         remove_pun=False,
                                                                         spelling=False,
                                                                         remove_stopwords=False))

        # ekphrasis text preprocessing
        train_df.at[index, 'ekphrasis'] = ' '.join(text_processor.pre_process_doc(row['text']))
        train_df.at[index, 'ekphrasis_no_symtags'] = train_df.at[index, 'ekphrasis'].replace('>', '').replace('<', '')
        text = train_df.at[index, 'ekphrasis']
        for t in tags:
            text  = text.replace(t, '')
        train_df.at[index, 'ekphrasis_rm'] = removeNums(removePunctuation(text))

        text = removeStopWords(train_df.at[index, 'ekphrasis_rm'].split())
        train_df.at[index, 'ekphrasis_lemmatization'] = ' '.join(lemmatization(text))
        train_df.at[index, 'ekphrasis_stemming'] = ' '.join(stemming(text))

        # append keyword/location on ekphrasis column
        keyword = ''
        location = ''
        if not train_df.at[index, 'keyword'] == 'nan':
            keyword = str(train_df.at[index, 'keyword']) + ' '

        if not train_df.at[index, 'location_processed'] == 'nan':
            location = str(train_df.at[index, 'location_processed']) + ' '

        train_df.at[index, 'keyword_ekphrasis'] = keyword + str(train_df.at[index, 'ekphrasis'])
        train_df.at[index, 'location_ekphrasis'] = location + str(train_df.at[index, 'ekphrasis'])
        train_df.at[index, 'keyword_location_ekphrasis'] = keyword + location + str(train_df.at[index, 'ekphrasis'])
        print("record #{} processing finished".format(index))

    # save processed dataframe to csv
    train_df.to_csv('dataset/train_processed.csv')