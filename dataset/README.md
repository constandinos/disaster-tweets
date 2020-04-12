# Dataset files description
## About the files:
1. test.csv: original  
2. train.csv: original  
3. train_processed.csv: see description of columns below  
4. train_dropduplicates.csv: remove duplicates from train_processed.csv, and correct target value of the duplicates  
5. test_processed.csv: see description of columns below  

## About the columns:
+ id:  
    + no change
+ keyword:
    + replace %20 to space, lowercase
+ location:
    + no change
+ text:
    + no change
+ target
    + no change 
+ processed_lem:
    + replace URLs with 'url'
    + remove usernames
    + replace emojis & emoticons
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + replace abbreviations & contractions
    + remove numbers
    + remove punctuations
    + correct spelling
    + remove stop-words
    + **lemmatization**
+ processed_stem:
    + replace URLs with 'url'
    + remove usernames
    + replace emojis & emoticons
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + replace abbreviations & contractions
    + remove numbers
    + remove punctuations
    + correct spelling
    + remove stop-words
    + **stemming**
+ processed_lem_key:
    + as processed_lem but with keyword concatenated at the beginning
+ processed_stem_key:
    + as processed_stem but with keyword concatenated at the beginning
+ processed_text_deep:
    + replace URLs with 'url'
    + remove usernames
    + replace emojis & emoticons
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + replace abbreviations & contractions
    + remove numbers
    + remove punctuations
    + correct spelling
    + **NOT remove stop-words**
    + **NO lemmatization/stemming**
+ processed_text_deep_without_url:
    + **remove URLs**
    + remove usernames
    + replace emojis & emoticons
    + remove non ascii
    + remove non printable characters
    + remove repeated consecutive characters
    + replace abbreviations & contractions
    + remove numbers
    + remove punctuations
    + correct spelling
    + **NOT remove stop-words**
    + **NO lemmatization/stemming**
+ processed_text_deep_without_url_key:
    + as processed_text_deep_without_url but with keyword concatenated at the beginning 