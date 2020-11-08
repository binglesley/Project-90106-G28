from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.preprocessing import Normalizer


def svc(train_set,label_set,test_set,ground_truth):
    svm_clf = LinearSVC(max_iter=10000)
    svm_clf.fit(train_set, label_set)
    y = svm_clf.predict(test_set)

    print(svm_clf.score(test_set,ground_truth))
    print(svm_clf.score(train_set,label_set))
    print(accuracy_score(y,ground_truth))
    p = precision_score(y,ground_truth)
    r = recall_score(y,ground_truth)
    f = f1_score(y,ground_truth)

    return p,r,f

def svm(train_set,label_set,test_set,ground_truth):
    train_set = Normalizer().fit_transform(train_set)
    test_set = Normalizer().fit_transform(test_set)
    svm_clf = SVC(C=0.2,kernel='linear')
    #svm_clf = SVC()
    #s = cross_validate(svm_clf,train_set,label_set)
    #print(s)
    #grid = GridSearchCV(svm_clf,param_grid={"C":[0.2,0.5,1.0,1.2,1.5,3,10],"kernel":['linear','rbf']},cv=10)
    #grid.fit(train_set,label_set)
    rfe = RFE(estimator=svm_clf, n_features_to_select =2, step=1)
    # n=5,0.6497 n=8，0.66358  n=12，0.66728  n=15，0.66635
    """"[True  True False  True False  True  True False  True  True  True  True
     True  True False  True]
    [1 1 5 1 2 1 1 3 1 1 1 1 1 1 4 1]"""

    rfe.fit(train_set,label_set)

    #print(rfe.support_)
    #print(rfe.ranking_)
    #svm_clf.fit(train_set,label_set)
    #y_score = svm_clf.decision_function(test_set)
    #y = svm_clf.predict(test_set)
    y = rfe.predict(test_set)
    #fpr, tpr, threshold = roc_curve(test_set, y_score)
    #roc_auc = auc(fpr, tpr)
    #y = grid.predict(test_set)
    print(rfe.score(test_set,ground_truth))

    #print(svm_clf.score(test_set,ground_truth))
    #print(svm_clf.score(train_set,label_set))
    #print(grid.score(test_set,ground_truth))
    #print(grid.score(train_set,label_set))
    #print(grid.best_params_)


    p = precision_score(y, ground_truth)
    r = recall_score(y, ground_truth)
    f = f1_score(y, ground_truth)

    return p, r, f

def decision_tree(train_set,label_set,test_set,ground_truth,Trains):
    """train_set = train_set.drop('Is_External__c',axis = 1)
    train_set = train_set.drop('RecordType.Name.1',axis = 1)
    test_set = test_set.drop('Is_External__c', axis=1)
    test_set = test_set.drop('RecordType.Name.1', axis=1)
    Trains = Trains.drop('Is_External__c', axis=1)
    Trains = Trains.drop('RecordType.Name.1', axis=1)"""

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(train_set,label_set)
    y = tree_clf.predict(test_set)

    print(tree_clf.score(test_set,ground_truth))
    print(roc_auc_score(ground_truth, y))
    feature_importance = tree_clf.feature_importances_
    #feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, Trains.columns[sorted_idx])

    plt.title('Variable Importance')
    plt.show()

    p = precision_score(y, ground_truth)
    r = recall_score(y, ground_truth)
    f = f1_score(y, ground_truth)

    return p,r,f

def rf(train_set,label_set,test_set,ground_truth,Trains):

    train_set = train_set.drop('Is_External__c', axis=1)
    train_set = train_set.drop('RecordType.Name.1', axis=1)
    test_set = test_set.drop('Is_External__c', axis=1)
    test_set = test_set.drop('RecordType.Name.1', axis=1)
    Trains = Trains.drop('Is_External__c', axis=1)
    Trains = Trains.drop('RecordType.Name.1',axis = 1)
    ground_truth = np.array(ground_truth.values)
    forest_clf = RandomForestClassifier(random_state=0, n_jobs=-1,n_estimators=115)
    #forest_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    #grid = GridSearchCV(forest_clf,param_grid={"n_estimators":[100,200,500,1500,2000,2500,3000]},cv=10)
    #grid.fit(train_set,label_set)
    forest_clf.fit(train_set,label_set)
    y = forest_clf.predict(test_set)
    #y = grid.predict(test_set)

    p = precision_score(y, ground_truth)
    r = recall_score(y, ground_truth)
    f = f1_score(y, ground_truth)

    print(forest_clf.score(test_set,ground_truth))
    """print(forest_clf.score(train_set,label_set))
    mse = metrics.mean_squared_error(ground_truth, y)
    print("MSE: %.4f" % mse)

    mae = metrics.mean_absolute_error(ground_truth, y)
    print("MAE: %.4f" % mae)

    R2 = metrics.r2_score(ground_truth, y)
    print("R2: %.4f" % R2)"""
    print(roc_auc_score(ground_truth, y))
    feature_importance = forest_clf.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, Trains.columns[sorted_idx])

    plt.title('Variable Importance')
    plt.show()

    return p, r, f
