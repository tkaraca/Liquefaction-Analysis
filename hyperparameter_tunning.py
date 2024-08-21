import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NeighbourhoodCleaningRule
from collections import Counter
from imblearn.pipeline import make_pipeline
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score,roc_curve, RocCurveDisplay
from yellowbrick.classifier import ROCAUC
from sklearn.preprocessing import LabelBinarizer
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score,
    auc,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
from yellowbrick.classifier import (
    ClassificationReport,
    DiscriminationThreshold,
)

from sklearn.metrics import (
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

from imblearn.metrics import (
    geometric_mean_score,
    make_index_balanced_accuracy,
)
import warnings

data = pd.read_csv('adapvtest.csv',sep=';')

data.head()

# imbalanced target

data.dver.value_counts() / len(data)
features = data[["pga","H","B","q","depth", "thickness"]]
X = np.array(features).reshape(-1,6)
y = np.array(data['dver'])

X=preprocessing.MinMaxScaler().fit_transform(X)
#y = preprocessing.label_binarize(y, classes=[0, 1, 2, 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# separate dataset into train and test

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(labels=['dver'], axis=1),  # drop the target
#     data['dver'],  # just the target
#     test_size=0.2,
#     random_state=0)

# X_train.shape, X_test.shape

# set up initial random forest
weight_0= 1/45*184/4
weight_1= 1/46*184/4
weight_2= 1/24*184/4
weight_3= 1/69*184/4
class_weight={0:weight_0, 1:weight_1, 2:weight_2, 3:weight_3}

rf = RandomForestClassifier(n_estimators=50,
                            random_state=39,
                            max_depth=2,
                            n_jobs=4,
                            class_weight=None)



param_grid = {
  'n_estimators': [10, 50, 100],
  'max_depth': [None, 2,4],
  'class_weight': [None,'balanced', 'balanced_subsample', class_weight],
  'bootstrap': [True, False],
  #'max_features': ['auto', 'sqrt'],
  #'min_samples_leaf': [1, 2, 4],
  #'min_samples_split': [2, 5, 10],
}


multi_roc = make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True)
search = GridSearchCV(estimator=rf,
                      scoring=multi_roc,
                      param_grid= param_grid,
                      cv=5,
                     ).fit(X_train, y_train)

print(search.best_score_)

print(search.best_params_)

print(search.best_estimator_)

print(search.score(X_test, y_test))


svc = svm.SVC(probability=True)

param_grid2 = {'C': [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'class_weight': [None, class_weight],
              }

search1 = GridSearchCV(estimator= svc,
                      scoring=multi_roc,
                      param_grid= param_grid2,
                      cv=5,
                     ).fit(X_train, y_train)

print(search1.best_score_)

print(search1.best_params_)

print(search1.best_estimator_)

print(search1.score(X_test, y_test))



logit = LogisticRegression(multi_class='ovr')
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

param_grid3 = {'solver':solvers,
               'penalty': ['l1', 'l2', 'elasticnet'],
               'class_weight': [ class_weight,None],
               'C':c_values}

search2 = GridSearchCV(estimator= logit,
                      scoring=multi_roc,
                      param_grid= param_grid3,
                      cv=5,
                     ).fit(X_train, y_train)

print(search2.best_score_)

print(search2.best_params_)

print(search2.best_estimator_)

print(search2.score(X_test, y_test))





cross_valid_scores =[]


for k in range(2,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, scoring=multi_roc, cv=5)
    cross_valid_scores.append(scores.mean())
    #y_test_knn = knn.predict_proba(X_test)
    #y_test_rp_knn=knn.predict(X_test)
print("optimal", np.argmax(cross_valid_scores))

