import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_data(filename):
    filepath = os.path.join(os.getcwd(), filename)
    data = pd.read_csv(filepath)
    data_all = data # adding cross and special features
    data_inter = data_all.drop('RN',axis = 1) #adding cross features
    data_inter = data_inter.drop('days',axis = 1)
    data_raw = data_inter.drop('f_won_before',axis = 1) # raw data
    data_raw = data_raw.drop('fac_oppo',axis = 1)
    data_raw = data_raw.drop('oppo_busi',axis = 1)
    data_rn = data.drop('f_won_before',axis = 1)
    data_rn = data_rn.drop('fac_oppo',axis = 1)
    data_rn = data_rn.drop('oppo_busi',axis = 1)

    return data_raw,data_inter,data_rn,data_all

def deal_data(raw_data):
    missing_set = {}
    no_missing = {}
    index_missing = {}
    index_no_missing = {}
    labels = {}
    names = []
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    for index, col in raw_data.iteritems():
        if raw_data[index].isnull().any():
            if len(col[col.isnull()]) / len(col) >= 0.4:  # if missing more than 40% ,imputed by most frequent
                print(index,'feature missing more than 40%')
                raw_data[index] = imp.fit_transform(raw_data[index].values.reshape(-1, 1))

            else:

                index_missing[index] = col[col.isnull()].index
                index_no_missing[index] = col[col.notnull()].index
                names.append(index)
                print(index,'missing less than 40%')

    for index,col in raw_data.iteritems():
        if str(col[col.notnull()][0]).isdigit() == False:
            if index == 'StageName':
                labels_corres = pd.factorize(raw_data[index])[1]
            raw_data[index] = pd.factorize(raw_data[index])[0].astype(np.uint16)

    for i in names:
        labels[i] = raw_data.loc[index_no_missing[i],i]


    raw_data = raw_data.dropna(axis = 1)

    for key,value in index_missing.items():
        missing_set[key] = raw_data.loc[value]
        no_values = index_no_missing[key]
        no_missing[key] = raw_data.loc[no_values]

    return no_missing,missing_set,index_no_missing,index_missing,labels,names


def impute(no_missing,missing_set,index_missing,labels,names,raw_data):
    predict = {}

    for feature in names:
        label = labels[feature]
        train_set = no_missing[feature]
        missing = missing_set[feature]
        predict[feature] = random_forest(train_set,label,missing)

    for key,items in predict.items():
        c = 0
        for i in index_missing[key]:
            raw_data.loc[i,key] = items[c]
            c += 1
    train = raw_data.drop('StageName',axis=1)
    y_label = raw_data['StageName']

    return train,y_label


def random_forest(train,label,missing):
    random_f = RandomForestRegressor()
    random_f.fit(train,label)
    y = random_f.predict(missing)
    return y





