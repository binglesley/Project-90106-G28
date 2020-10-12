from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def Svm(train_set,label_set,test_set,ground_truth,Train):
    Train =  np.arange(Train.shape[-1])
    selector = SelectKBest(f_classif, k=4)
    selector.fit(train_set, label_set)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(Train - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)')
    clf = make_pipeline(MinMaxScaler(), SVC(kernel='linear'))
    clf.fit(train_set, label_set)
    print('Classification accuracy without selecting features: {:.3f}'
          .format(clf.score(test_set, ground_truth)))

    svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()

    plt.bar(Train - .25, svm_weights, width=.2, label='SVM weight')

    clf_selected = make_pipeline(
        SelectKBest(f_classif, k=4), MinMaxScaler(), SVC(kernel='linear')
    )
    clf_selected.fit(train_set, label_set)
    print('Classification accuracy after univariate feature selection: {:.3f}'
          .format(clf_selected.score(test_set, ground_truth)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(Train[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

    """Classification accuracy without selecting features: 0.677
Classification accuracy after univariate feature selection: 0.677"""