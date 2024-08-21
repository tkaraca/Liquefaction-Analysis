import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
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

data = pd.read_csv('adapvtest.csv',sep=';')

features = data[["pga","H","B","q","depth", "thickness"]]
X = np.array(features).reshape(-1,6)
y = np.array(data['dver'])

X=preprocessing.MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# separate dataset into train and test

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(labels=['dver'], axis=1),  # drop the target
#     data['dver'],  # just the target
#     test_size=0.2,
#     random_state=39)

# X_train.shape, X_test.shape

# Logistic Regression with class_weight

# we initialize the cost / weights when we set up the transformer
weight_0= 1/45*184/4
weight_1= 1/46*184/4
weight_2= 1/24*184/4
weight_3= 1/69*184/4
class_weight={0:weight_0, 1:weight_1, 2:weight_2, 3:weight_3}

def run_rf(X_train, X_test, y_train, y_test, class_weight):
    
    # weights introduced here
    rf = RandomForestClassifier(
        n_estimators=100,
        #penalty='l2',
        #solver='newton-cg',
        random_state=39,
        #max_iter=10,
        n_jobs=4,
        class_weight=class_weight, # weights / cost
        max_depth=4,
    )
    
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = rf.predict_proba(X_test)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred,multi_class='ovr')))
    y_test_rp_rf=rf.predict(X_test)
    print('Accuracy Score Random Forest:{}'.format(accuracy_score(y_test, rf.predict(X_test))))

    print('Precision Random Forest test:{}'.format(precision_score(y_test, y_test_rp_rf, average='macro')))
    print('Recall Random Forest test:{}'.format(recall_score(y_test, y_test_rp_rf, average='macro')))
    print('F-measure Random Forest test:{}'.format(f1_score(y_test, y_test_rp_rf, average='macro')))

    visualizer = PrecisionRecallCurve(rf, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(rf, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  

    visualizer = ClassificationReport(rf)

    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()

    micro_precision, micro_recall, micro_fscore, _ = score(y_test, y_test_rp_rf, average='micro')    
    print("Micro avg precision: ", micro_precision)
    print("Micro avg recall: ", micro_recall)
    print("Micro avg F1 score: ", micro_fscore) 

    
    
run_rf(X_train,
        X_test,
        y_train,
        y_test,
        class_weight=class_weight)

# evaluate performance of algorithm built
# cost estimated as imbalance ratio

# 'balanced' indicates that we want same amount of 
# each observation, thus, imbalance ratio

run_rf(X_train,
          X_test,
          y_train,
          y_test,
          class_weight='balanced_subsample')


def run_Logit(X_train, X_test, y_train, y_test, class_weight):
    
    # weights introduced here
    logit = LogisticRegression(
        penalty='l2',
        solver='newton-cg',
        random_state=0,
        max_iter=10,
        n_jobs=4,
        class_weight=class_weight # weights / cost
    )

    logit.fit(X_train, y_train)

    print('Train set')
    pred = logit.predict_proba(X_train)
    print(
        'Logistic roc-auc: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = logit.predict_proba(X_test)
    print(
        'Logistic roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
    visualizer = PrecisionRecallCurve(logit, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(logit, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  
        
    
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          class_weight=class_weight)
    # evaluate performance of algorithm built
# cost estimated as imbalance ratio

# with numpy.where, we introduce a cost of 99 to
# each observation of the minority class, and 1
# otherwise.

def run_svc(X_train, X_test, y_train, y_test, class_weight):
    
    # weights introduced here
    svc = svm.SVC(class_weight=class_weight,probability=True # weights / cost
    )

    svc.fit(X_train, y_train)

    print('Train set')
    pred = svc.predict_proba(X_train)
    print(
        'Support Vector Machine: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = svc.predict_proba(X_test)
    print(
        'Support Vector Machine: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
    visualizer = PrecisionRecallCurve(svc, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(svc, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  

    
run_svc(X_train,
          X_test,
          y_train,
          y_test,
          class_weight=class_weight)


def run_knn(X_train, X_test, y_train,y_test, weights):
    
    # weights introduced here
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')# weights / cost
    

    knn.fit(X_train, y_train)

    print('Train set')
    pred = knn.predict_proba(X_train)
    print(
        'KNeighborsClassifier: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = knn.predict_proba(X_test)
    print(
        'KNeighborsClassifier: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
    visualizer = PrecisionRecallCurve(knn, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(knn, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  
        
    
run_knn(X_train,
          X_test,
          y_train,
          y_test,
          weights='distance'
         )