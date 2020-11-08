from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def Forest(train_set,label_set):
    RANDOM_STATE = 123
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='log2'",
         RandomForestClassifier(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
         RandomForestClassifier(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE))
    ]
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    min_estimators = 15
    max_estimators = 400

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(train_set, label_set)
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def max_deeps(train_set,label_set,test_set,ground_truth):
    clf = RandomForestClassifier()
    min_depths = 2
    max_depths = 50
    acc_x = []
    y = []

    for i in range(min_depths, max_depths + 1):
        clf.set_params(max_depth=i)
        clf.fit(train_set, label_set)
        acc = clf.score(test_set,ground_truth)
        acc_x.append(acc)
        y.append(i)

    plt.title("Depth and Accuracy of Random Forest")
    plt.xlabel("Tree's Depth")
    plt.ylabel("Accuracy")
    plt.plot(y,acc_x)
    plt.show()

