# Dataset files description
## About the files:
1. `test.csv`: original  
2. `train.csv`: original  
3. `train_processed_all.csv`: see description of columns below  
4. `train_dropduplicates_all.csv`: remove duplicates from train_processed.csv, and correct target value of the duplicates  
5. `our_train.csv`: train dataset as a result of split from train_dropduplicates_all.csv
6. `our_test.csv`: test dataset as a result of split from train_dropduplicates_all.csv

Folder `kaggle_test` contains test and processed dataset for submission to Kaggle's contest. 

## About the columns:
1. id: no change
2. keyword: replace %20 with space character
3. location: no change
4. text: no change
5. target: no change
6. location_processed:
    + remove URLs
    + replace named and numeric html characters to the Unicode characters
    + remove usernames
    + replace emojis & emoticons with there meaning
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + to lowercase, split camelCase words
    + expand hashtags
    + replace abbreviations & contractions (use of related dictionaries)
    + remove numbers
    + remove punctuations
    + correct spelling
    + ~~remove stopwords~~
    + ~~lemmatization/stemming~~
7. processed:
    + remove URLs
    + replace named and numeric html characters to the Unicode characters
    + remove usernames
    + replace emojis & emoticons with there meaning
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + to lowercase, split camelCase words
    + expand hashtags
    + replace abbreviations & contractions
    + remove numbers
    + remove punctuations
    + correct spelling
    + ~~remove stopwords~~
    + ~~lemmatization/stemming~~*
8. lemmatization:
    + use text from 'processed' column
    + remove stopwords
    + lemmatization
9. stemming:
    + use text from 'processed' column
    + remove stopwords
    + stemming
10. no_punc_no_abb:
    + remove URLs
    + replace named and numeric html characters to the Unicode characters
    + remove usernames
    + replace emojis & emoticons with there meaning
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + to lowercase, split camelCase words
    + expand hashtags
    + replace ~~abbreviations &~~ contractions
    + remove numbers
    + ~~remove punctuations~~
    + ~~correct spelling~~
    + ~~remove stopwords~~
    + ~~lemmatization/stemming~~
11. ekphrasis (use of library Ekphrasis for the following):
    + ~~remove URLs~~
    + replace named and numeric html characters to the Unicode characters
    + remove usernames
    + replace emojis & emoticons with there meaning
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + to lowercase, split camelCase words
    + expand hashtags
    + replace ~~abbreviations &~~ contractions
    + ~~remove numbers~~
    + ~~remove punctuations~~
    + correct spelling
    + ~~remove stopwords~~
    + ~~lemmatization/stemming~~
    + extra
        + normalization (URL, email, percent, money, phone, user, time, date, number)
        + fix html
        + use social tokenizer
12. ekphrasis_no_symtags:
    + use text from ekphrasis
    + remove '<>' from tags
13. ekphrasis_rm
    + use text from ekphrasis
    + remove puntuations
    + remove numbers
    + remove stopwords
14. ekphrasis_lemmatization:
    + use text from ekphrasis_rm
    + lemmatization
15. ekphrasis_stemming:
    + use text from ekphrasis_rm
    + stemming
16. keyword_ekphrasis:
    + append keyword to text from ekphrasis
17. location_ekphrasis:
    + append location_processed to text from ekphrasis
18. keyword_location_processed:
    + append keyword and location_processed to text from ekphrasis