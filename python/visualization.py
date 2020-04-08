import pandas as pd

tweet= pd.read_csv('../dataset/train.csv')
test=pd.read_csv('../dataset/test.csv')
print(tweet.head(3))