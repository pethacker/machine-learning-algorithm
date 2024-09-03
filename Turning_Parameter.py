# Grid search: a technique that can help us adjust multiple parameters simultaneously, enumeration technique
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def turning_parameter():

    gini_thresholds = np.linspace(0, 0.5, 20)
    # entropy_thresholds = np.linespace(0, 1, 50)

    parameters = {
        'splitter': ('best', 'random'),
        'criterion': ('gini', 'entropy'),
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [1, 2, 5],
        'min_impurity_decrease': gini_thresholds
    }

    clf = DecisionTreeClassifier(random_state=25)  # decision tree
    GS = GridSearchCV(clf, parameters, cv=5, n_jobs=-1, verbose=2)
    return GS
