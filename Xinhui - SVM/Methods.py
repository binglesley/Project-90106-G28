from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
import datetime as dt
import pandas as pd

def method_logit(train_set,label_set,test_set,ground_truth):
    # Baseline for begining analysis
    logit_clf = LogisticRegression()
    logit_clf.fit(train_set, label_set)
    y = logit_clf.predict(test_set)
    print('logisitic_precision for each labels:', precision_score(y, ground_truth), '\n')
    print('logistic_recall for each labels:', recall_score(y, ground_truth), '\n')
    print('logistic_f1 for each labels:', f1_score(y, ground_truth), '\n')
    print('logistic accuracy', logit_clf.score(test_set, ground_truth))

def method_svm(train_set,label_set,test_set,ground_truth):

    train_set = Normalizer().fit_transform(train_set)
    test_set = Normalizer().fit_transform(test_set)
    svm_clf = SVC(C=0.2, kernel='linear')

    # For parameters tunning
    # svm_clf = SVC()
    # s = cross_validate(svm_clf,train_set,label_set)
    # grid = GridSearchCV(svm_clf,param_grid={"C":[0.1,0.2,0.5,1,2,3,5,8,10],"kernel":['linear','rbf']},cv=10)
    # grid.fit(train_set,label_set)
    # print(grid.score(test_set,ground_truth))
    # print(grid.best_params_)

    start = dt.datetime.now()
    svm_clf.fit(train_set, label_set)
    end = dt.datetime.now()
    y = svm_clf.predict(test_set)

    print(end - start) # calculates the run time
    print('svm_precision for each labels:', precision_score(y, ground_truth), '\n')
    print('svm_recall for each labels:', recall_score(y, ground_truth), '\n')
    print('svm_f1 for each labels:', f1_score(y, ground_truth), '\n')
    print('svm accuracy', svm_clf.score(test_set, ground_truth))


def method_rf(train_set,label_set,test_set,ground_truth,Trains):
    ground_truth = np.array(ground_truth.values)
    start = dt.datetime.now()
    forest_clf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=115)

    # For parameter tunning
    # grid = GridSearchCV(forest_clf,param_grid={"n_estimators":[100,200,500,1500,2000,2500,3000]},cv=10)
    # grid.fit(train_set,label_set)
    # y = grid.predict(test_set)
    forest_clf.fit(train_set, label_set)
    end = dt.datetime.now()
    y = forest_clf.predict(test_set)

    # Predict
    """predict_result = []
    Prob_result = forest_clf.predict_proba(test_set)

    for item in y:
        if item == 0:
            predict_result.append('Lose')
        else:
            predict_result.append('Win')

    predict_result = pd.DataFrame(predict_result,columns=['Predict classisication'])
    ground_truth = pd.DataFrame(ground_truth,columns=['Opportunity Id'])
    Prob_result = pd.DataFrame(Prob_result,columns=['Lose Probability','Win Probability'])
    temp = [ground_truth,predict_result,Prob_result]
    Result = pd.concat(temp,axis=1)

    save_path = os.path.join(os.getcwd(),'Result.csv')
    Result.to_csv(save_path)
    print(Result)"""

    print(end - start) # calculates the runtime
    print('rf_precision for each labels:', precision_score(y, ground_truth), '\n')
    print('rf_recall for each labels:', recall_score(y, ground_truth), '\n')
    print('rf_f1 for each labels:', f1_score(y, ground_truth), '\n')
    print('rf accuracy', forest_clf.score(test_set, ground_truth))

    feature_importance = forest_clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, Trains.columns[sorted_idx])

    plt.title('Variable Importance')
    plt.show()

