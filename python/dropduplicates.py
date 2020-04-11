import pandas as pd

df = pd.read_csv('../dataset/train_processed.csv')
group = df.groupby('processed_lem')['id'].unique()

drop_id = []
count_different_target = 0 # number of tuples, its elements have different target value
count_tuples = 0 # number of different tuples
df.set_index('id', inplace=True)
for key, items in group.items():
    if len(items)>1:
        count_tuples += 1
        count0 = 0
        count1 = 0
        for item in items:
            if df.loc[item, 'target'] == 0:
                count0 += 1
            else:
                count1 += 1
        # set value of target that appears more in the tuple
        if count0 > count1:
            newtarget = 0 
        else:
            newtarget = 1

        if count0>0 and count1>0:
            count_different_target += 1

        drop_id += items[1:].tolist() # id to be dropped
        df.loc[items[0], 'target'] = newtarget # set the target of the record left

print("number of tubles: {}".format(count_tuples)) #355
print("number of tubles with different target: {}".format(count_different_target)) #87

# drop duplicates
df.drop(drop_id , inplace=True)
# save to csv
df.to_csv('../dataset/train_dropduplicates.csv')
