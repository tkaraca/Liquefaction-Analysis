
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score,roc_curve, RocCurveDisplay

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
warnings.filterwarnings("ignore")


data = pd.read_csv('adapvtest.csv',sep=';')

features = data[["pga","H","B","q","depth", "thickness"]]
X = np.array(features).reshape(-1,6)
y = np.array(data['dver'])
X=preprocessing.MinMaxScaler().fit_transform(X)

sm = KMeansSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=2,
    n_jobs=None,
    kmeans_estimator=KMeans(n_clusters=4, random_state=0),
    cluster_balance_threshold=0.1,
    density_exponent='auto'
)

X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)
clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))


def run_randomForests(X_train, X_test, y_train, y_test, class_weight):
    
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))
    
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred,multi_class='ovr')))

    visualizer = ClassificationReport(rf)

    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()

    print(balanced_accuracy_score(y_test, pred))
    print(geometric_mean_score(y_test,pred))

run_randomForests(X_train,
          X_test,
          y_train,
          y_test,
          class_weight=None)  



# X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.20)

# def run_randomForest2(X_train2, X_test2, y_train2, y_test2, class_weigth):
    

    
#     rf2=RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)


#     rf2.fit(X_train2,y_train2)

#     print('Train set2')
#     pred2 = rf2.predict_proba(X_train2)
#     print('Random Forest roc_auc2:{}'.format(roc_auc_score(y_train2,pred2,multi_class='ovr')))


#     print('Test set2')
#     pred2 = rf2.predict_proba(X_test2)
#     print('Random Forest roc_auc2:{}'.format(roc_auc_score(y_test2,pred2, multi_class='ovr')))
#     visualizer = ClassificationReport(clf)

#     visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
#     visualizer.score(X_test, y_test)        # Evaluate the model on the test data
#     visualizer.show()

#     print(balanced_accuracy_score(y_test, predictions))
#     print(geometric_mean_score(y_test,predictions))



# run_randomForests(X_train2,
#         X_test2,
#         y_train2,
#         y_test2,
#         class_weight=None)  





                     
