# Merge csv files of lemmatization and stemming
# Remove %20 character from keyword column

import pandas as pd

lem_df = pd.read_csv('../dataset/train_processed_lem.csv')
lem_df.set_index('id', inplace=True)

stem_df = pd.read_csv('../dataset/train_processed_stem.csv')
stem_df.set_index('id', inplace=True)

lem_df['keyword'] = lem_df['keyword'].astype(str)
for index, row in lem_df.iterrows():
    lem_df.at[index, 'keyword'] = row['keyword'].replace('%20', ' ')

lem_df.rename(columns={'processed_text':'processed_lem'}, inplace = True)
lem_df['stem'] = stem_df['processed_stem']

for index, row in lem_df.iterrows():
    if not (row['keyword'] == 'nan'):
        key = row['keyword']
    else:
        key = ''

    lem_df.at[index, 'processed_lem_key'] = key + ' ' + row['processed_lem']
    lem_df.at[index, 'processed_stem_key'] = key + ' ' + row['processed_stem']

lem_df.to_csv('../dataset/train_processed.csv')