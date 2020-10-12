import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import SVM
from sklearn.preprocessing import Normalizer

def Load_data():
    filepath = os.path.join(os.getcwd(),'research_joint_data.csv')
    data = pd.read_csv(filepath)

    data = data.dropna(how='all',axis=1)
    data = data.drop('Id',axis = 1)
    data = data.drop('Close_year',axis = 1)
    data = data.drop('Close_month',axis = 1)
    data['Is_External__c'] = data['Is_External__c'].fillna('Internal')
    data = data[data['StageName'].isin(['Closed Lost', 'Closed Won'])]
    #print(data.info())


    """for index,col in data.iteritems(): #判断空值占比
        null_proportion = cal_null(col)

        if null_proportion >= 0.4:
            data = data.drop(index, axis=1)
    data = data.dropna()"""

    for index, col in data.iteritems():#判断是不是数字
        types = type_(col)
        if types == 'catogerical':
           #if index == 'StageName':
                #labels_corres = pd.factorize(data[index])[1]
                #data[index] = pd.factorize(data[index])[0].astype(np.uint16)
            #else:
                data[index] = pd.factorize(data[index])[0].astype(np.uint16)
    print(data.info())
    x = data
    Labels = x['StageName']
    Trains = x.drop('StageName',axis=1)
    #Trains = Trains.drop('Status_Reason__c',axis = 1)
    names = Trains.columns


    return Trains,Labels,names


def cal_null(col):
    is_null = col[col.isnull()]
    percentage = len(is_null)/len(col)
    return percentage

def type_(col):
    element = str(col[col.notnull()].dtype)
    if element == 'int64':
        return 'number'
    else:
        return 'catogerical'

def random_forest(train_set,label_set,test_set,ground_truth):
    forest_clf = RandomForestRegressor()
    forest_clf.fit(train_set,label_set)
    y = forest_clf.predict(test_set)
    print(sorted(zip(map(lambda x: round(x, 4), forest_clf.feature_importances_),names),
             reverse=True))


Trains,Labels,names = Load_data()

X_train,X_test,y_train,y_test = train_test_split(Trains,Labels,train_size=0.8,random_state=1)
"""svm_precision,svm_recall,svm_f1 = SVM.svc(X_train,y_train,X_test,y_test)
print('svm_precision for each labels:',svm_precision,'\n')
print('svm_recall for each labels:',svm_recall,'\n')
print('svm_f1 for each labels:',svm_f1,'\n')"""
svc_precision,svc_recall,svc_f1 = SVM.svm(X_train,y_train,X_test,y_test)
print('svc_precision for each labels:',svc_precision,'\n')
print('svc_recall for each labels:',svc_recall,'\n')
print('svc_f1 for each labels:',svc_f1,'\n')

#SVM.svm(X_strain,y_strain,X_stest,y_stest)
tree_precision,tree_recall,tree_f1 = SVM.decision_tree(X_train,y_train,X_test,y_test,Trains)
print('tree_precision for each labels:',tree_precision,'\n')
print('tree_recall for each labels:',tree_recall,'\n')
print('tree_f1 for each labels:',tree_f1,'\n')
#random_forest(X_train,y_train,X_test,y_test)
rf_precision,rf_recall,rf_f1 = SVM.rf(X_train,y_train,X_test,y_test,Trains)
print('forest_precision for each labels:',rf_precision,'\n')
print('forest_recall for each labels:',rf_recall,'\n')
print('forset_f1 for each labels:',rf_f1,'\n')


