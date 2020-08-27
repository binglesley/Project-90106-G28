import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn import metrics, neighbors
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import preprocessing, linear_model, svm


features = ['StageName', 'Status_Reason__c', 'RecordType.Name',
            "Final_Record_Type__c", "RICE_Supported__c", "CloseDate", "Actual_Close_Date__c", "Amount",
            "Estimated_Project_Total_Value__c", "Booked_Revenue__c", 'Actual_Project_Total_Value__c', 'BD_Cluster__c',
            'BD_Division__c', 'AccountId', 'Customer_Contact__c', 'Lead_Academic_contact__c', 'Lead_Faculty__c',
            'Lead_School__c', 'Lead_Department__c', 'OwnerId', 'account_type', 'Industry', 'Industry_Sub_Type__c',
            'Business_Type__c', 'Country__c', 'Is_External__c']

columns=['StageName', 'Status_Reason__c', 'RecordType.Name',
            "Final_Record_Type__c", "RICE_Supported__c", "CloseDate", "Actual_Close_Date__c", "Amount",
            "Estimated_Project_Total_Value__c", "Booked_Revenue__c", 'Actual_Project_Total_Value__c', 'BD_Cluster__c',
            'BD_Division__c', 'AccountId', 'Customer_Contact__c', 'Lead_Academic_contact__c', 'Lead_Faculty__c',
            'Lead_School__c', 'Lead_Department__c', 'OwnerId', 'account_type', 'Industry', 'Industry_Sub_Type__c',
            'Business_Type__c', 'Country__c', 'Is_External__c', 'Converted']


def import_data(file):
    data = pd.read_csv(file, header=0)
    le = preprocessing.LabelEncoder()
    processed_data = data.apply(le.fit_transform)
    return processed_data


# print(data.info()) # get a summary of the data


def logistic():
    lm = LogisticRegression()
    return lm


# Return data set with missing value imputed
def impute_missing_value(data, column_name, function):
    imp = SimpleImputer(strategy="most_frequent")
    data_imputed = pd.DataFrame(imp.fit_transform(data), columns=column_name)
    return data_imputed


# Split data into X_train, X_test, y_train, y_test
def train_test(data_imputed, feature, predictor, test_size, random_state=0):
    X = data_imputed[feature]
    y = data_imputed[predictor]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# fit in logistic model and show test results
def test_Logistic_regression(data_imputed, feature, predictor, test_size):
    X_train, X_test ,y_train , y_test = train_test(data_imputed, feature, predictor, test_size, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred=logreg.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)

    print("Coefficients:%s, intercept %s"%(logreg.coef_, logreg.intercept_))
    print("score:%.2f"% logreg.score(X_test,y_test))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    plt.show()


if __name__ == '__main__':

    file1 = "imputed_data.csv"
    file2 = "clean_data.csv"
    file3 = "cleaned_23:8_updated.csv"
    data = import_data( file3 )

    feature3 = ["RecordType.Name", "RICE_Supported__c", "CreatedDate", "CloseDate",
                "Parent_Opportunity__c", "RecordType.Name.1", "Industry", "Business_Type__c", "Is_External__c",
                "ParentId"]

    # cross feature "RecordType.Name" and "RecordType.Name.1" (combine together)

    predictor = "Converted"

    X = data[feature3]
    y = data[predictor]

    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    lm = LogisticRegression()

    # print(data.info())
    # cv_result_lm = cross_val_score(lm, X, y, cv=10)
    # print(cv_result_lm)
    # print("mean score of lm:%.2f" % cv_result_lm.mean())
    #
    # cv_result_knn = cross_val_score(knn,  X, y, cv=10)
    # print(cv_result_knn)
    # print("mean score of knn:%.2f" % cv_result_knn.mean())

    # print( "Coefficients:%s, intercept %s" % (lm.coef_, lm.intercept_) )