import SVM as training
import imputation as imp
import K_means_imp as kimp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer,MinMaxScaler,StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import scale
import opti_svm as ops
import opti_forest as opf

if __name__ == '__main__':
    raw_data = imp.load_data('cleaned_1-OCT-modelling.csv')
    #nom  = Normalizer(norm='l2')

    #m = MinMaxScaler()
    #s = StandardScaler()
    no_missing, missing_set, index_no_missing, index_missing, labels, names = imp.deal_data(raw_data)
    X_set, y = imp.impute(no_missing, missing_set, index_missing, labels, names, raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X_set, y, train_size=0.8, random_state=1)
    #X_strain,X_stest,y_strain,y_stest = train_test_split(X_norm, y, train_size=0.8, random_state=1)
    svc_precision, svc_recall, svc_f1 = training.svm(X_train, y_train, X_test, y_test)
    #training.svm(X_strain, y_strain, X_stest, y_stest)
    print('svc_precision for each labels:', svc_precision, '\n')
    print('svc_recall for each labels:', svc_recall, '\n')
    print('svc_f1 for each labels:', svc_f1, '\n')
    """tree_precision, tree_recall, tree_f1 = training.decision_tree(X_train, y_train, X_test, y_test,X_set)
    print('tree_precision for each labels:', tree_precision, '\n')
    print('tree_recall for each labels:', tree_recall, '\n')
    print('tree_f1 for each labels:', tree_f1, '\n')"""

    rf_precision, rf_recall, rf_f1 = training.rf(X_train, y_train, X_test, y_test,X_set)

    print('forest_precision for each labels:', rf_precision, '\n')
    print('forest_recall for each labels:', rf_recall, '\n')
    print('forset_f1 for each labels:', rf_f1, '\n')
    #res.random_forest(X_train,y_train,X_test,y_test)
    #print(p,r,f)
    #res.random_forest(kimp.X_train, kimp.y_train, kimp.X_test, kimp.y_test)
    #ops.Svm(X_train, y_train, X_test, y_test,X_set)
    #opf.Forest(X_train, y_train, X_test, y_test,X_set)
    opf.max_deeps(X_train, y_train, X_test, y_test,X_set)