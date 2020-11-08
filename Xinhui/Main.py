import Methods as training
import imputation_rf as imp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import Model_optimize as opf

if __name__ == '__main__':
    raw_data,inter_data,rn_data,all_data = imp.load_data('cleaned-16-Oct.csv')

    no_missing, missing_set, index_no_missing, index_missing, labels, names = imp.deal_data(all_data)
    X_set, y = imp.impute(no_missing, missing_set, index_missing, labels, names, all_data)
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X_set, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, random_state=1)

    # Model training
    training.logit(X_train, y_train, X_test, y_test)
    training.svm(X_train, y_train, X_test, y_test)
    training.rf(X_train, y_train, X_test, y_test,X_set)

    # Model selection
    #opf.Forest(X_train, y_train)
    #opf.max_deeps(X_train, y_train, X_test, y_test)