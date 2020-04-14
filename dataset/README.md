# Dataset files description
## About the files:
1. test.csv: original  
2. train.csv: original  
3. train_processed.csv: see description of columns below  
4. train_dropduplicates.csv: remove duplicates from train_processed.csv, and correct target value of the duplicates  
5. test_processed.csv: see description of columns below  

## About the columns:
+ id: no change
+ keyword: no change
+ location: no change
+ text: no change
+ target: no change 
+ processed:
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
+ lemmatization:
    + use text from 'processed' column
    + remove stop-words
    + lemmatization
+ stemming:
    + use text from 'processed' column
    + remove stop-words
    + stemming