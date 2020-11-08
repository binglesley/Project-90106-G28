import pandas as pd
import os
from sklearn.cluster import KMeans
import collections
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filename):
    filepath = os.path.join(os.getcwd(), filename)
    data = pd.read_csv(filepath)
    data['Is_External__c'] = data['Is_External__c'].fillna('Internal')
    data = data[data['StageName'].isin(['Closed Lost', 'Closed Won'])]
    data = data.drop('Id', axis=1)

    return data

def deal_data(data):
    names = []
    no_missing_1 = {}
    no_missing = {}
    missing = {}
    drop_name = []
    raw_data = data
    for index, col in data.iteritems():
        percentage = len(col[col.isnull()]) / len(col)
        if percentage >= 0.4:
            print(index, 'features missing more than 40%')
            data = data.drop(index, axis=1)  # drop it
            drop_name.append(index)
        if percentage > 0 and percentage < 0.4:
            names.append(index)
            data = data.drop(index, axis=1)

    for item in names:
        no_missing_1[item] = get_nomissing_set(data,raw_data,item)
        data = data.drop(item,axis = 1)
        no_missing[item] = get_set(data,raw_data,item)
        data = data.drop(item, axis=1)
        missing[item] = get_missing_set(data,raw_data,item)
        data = data.drop(item, axis=1)

    for key,value in no_missing_1.items():
        no_missing_1[key] = factorize(value)


    for key,value in no_missing.items():
        no_missing[key] = factorize(value)


    for key,value in missing.items():
        missing[key] = factorize(value)

    for i in drop_name:
        raw_data = raw_data.drop(i,axis = 1)

    return no_missing,missing,names,no_missing_1,raw_data

def factorize(dataset):

    for index,col in dataset.iteritems():
        types = str(col[col.notnull()].dtype)
        if types != 'int64':
            dataset[index] = pd.factorize(dataset[index])[0].astype(np.uint16)

    return dataset

def get_nomissing_set(x,raw_data,name):
    y = raw_data[name]
    y_1 = x
    y_1[name] = y
    y_1 = y_1.dropna(subset=[name], how='any', axis=0)

    return y_1

def get_missing_set(x,raw_data,name):
    y = raw_data[name]
    y_1 = x
    y_1[name] = y
    y_1 = y_1[y_1.isnull().values]
    y_1 = y_1.drop(name,axis = 1)

    return y_1

def get_set(x,raw_data,name):
    y = raw_data[name]
    y_1 = x
    y_1[name] = y
    y_1 = y_1.dropna(subset=[name], how='any', axis=0)
    y_1 = y_1.drop(name,axis = 1)

    return y_1

def k_imputation(no_missing,missing,no_missing_1,names):
    clf = KMeans()
    result = {}
    sum_dict = {}
    count_dict = {}
    for name in names:
        clf.fit(no_missing[name])
        predictedtime = clf.predict(missing[name])
        label = clf.labels_
        sums = collections.defaultdict(int)
        count = collections.defaultdict(int)
        means = collections.defaultdict(int)
        lists = []
        i = 0
        for item in label:
            sums[str(item)] += no_missing_1[name][name].values[i]
            count[str(item)] += 1
            i += 1
        sum_dict[name] = sums
        count_dict[name] = count_dict
        for key in count.keys():
            means[str(key)] = sums[str(key)] / count[str(key)]
        for item in predictedtime:
            lists.append(means[str(item)])
        result[name] = lists
    return result

def missing_indexs(missing):
    missing_index = {}
    for key,item in missing.items():
        missing_index[key] = list(item.index)
    return missing_index

def sub_data():
    for key,item in missing_index.items():
        c = 0
        for indexs in item:
            raw_data.loc[indexs, key] = result[key][c]
            c += 1
    train = raw_data.drop('StageName',axis = 1)
    label = raw_data['StageName']
    return train,label

data = load_data('research_joint_data.csv')
no_missing,missing,names,no_missing_1,raw_data = deal_data(data)
result = k_imputation(no_missing,missing,no_missing_1,names)
missing_index = missing_indexs(missing)
train,y = sub_data()
X_train, X_test, y_train, y_test = train_test_split(train, y, train_size=0.8, random_state=1)


