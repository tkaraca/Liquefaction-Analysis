from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)

from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from imblearn.datasets import fetch_datasets

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    RUSBoostClassifier,
    EasyEnsembleClassifier,
)
import warnings
warnings.filterwarnings("ignore")
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
# just re-sampling methods (no classifier)
strategy = {0:250, 1:250, 2:250, 3:250}
resampling_dict = {
    
    'random': RandomUnderSampler(
        sampling_strategy='auto',
        random_state=0,
        replacement=True,
    ),

    'smote': SMOTE(
        sampling_strategy=strategy,
        random_state=0,
        k_neighbors=5,
        #n_jobs=4,
    ),
}

# ensemble methods (with or without resampling)

ensemble_dict = {

    # balanced random forests (bagging)
    'balancedRF': BalancedRandomForestClassifier(
        n_estimators=20,
        criterion='gini',
        max_depth=3,
        sampling_strategy='auto',
        n_jobs=4,
        random_state=2909,
    ),

    # bagging of Logistic regression, no resampling
    'bagging': BaggingClassifier(
        estimator=LogisticRegression(random_state=2909, class_weight='balanced'),
        n_estimators=20,
        n_jobs=4,
        random_state=2909,
    ),

    #bagging of Logistic regression, with resampling
    'balancedbagging': BalancedBaggingClassifier(
        estimator=LogisticRegression(random_state=2909, class_weight='balanced'),
        n_estimators=20,
        max_samples=1.0,  # The number of samples to draw from X to train each base estimator
        max_features=1.0,  # The number of features to draw from X to train each base estimator
        bootstrap=True,
        bootstrap_features=False,
        sampling_strategy='auto',
        n_jobs=4,
        random_state=2909,
    ),

    # boosting + undersampling
    # 'rusboost': RUSBoostClassifier(
    #     estimator='RandomForestClassifier',
    #     n_estimators=20,
    #     learning_rate=1.0,
    #     sampling_strategy='auto',
    #     random_state=2909,
    # ),

    # bagging + boosting + under-sammpling
    'easyEnsemble': EasyEnsembleClassifier(
        n_estimators=20,
        sampling_strategy='auto',
        n_jobs=4,
        random_state=2909,
    ),
}

datasets_ls = ['adapvtest.csv']

# function to train random forests and evaluate the performance

def run_randomForests(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(
        n_estimators=20, random_state=39, max_depth=2, n_jobs=4, class_weight='balanced')
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred,multi_class='ovr')))

    print('Test set')
    pred = rf.predict_proba(X_test)
    print(
        'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
    visualizer = PrecisionRecallCurve(rf, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(rf, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  
    

    return roc_auc_score(y_test, pred,multi_class='ovr')

# function to train random forests and evaluate the performance

# def run_randomForests(X_train, X_test, y_train, y_test):

#     rf = RandomForestClassifier(
#         n_estimators=20, random_state=39, max_depth=2, n_jobs=4)
#     rf.fit(X_train, y_train)

#     print('Train set')
#     pred = rf.predict_proba(X_train)
    # print(
    #     'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    # print('Test set')
    # pred = rf.predict_proba(X_test)
    # print(
    #     'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

    # return roc_auc_score(y_test, pred[:, 1])

# function to train random forests and evaluate the peadaormance

def run_adaboost(X_train, X_test, y_train, y_test):

    ada = AdaBoostClassifier(n_estimators=20, random_state=2909)
    
    ada.fit(X_train, y_train)

    print('Train set')
    pred = ada.predict_proba(X_train)
    print(
        'AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = ada.predict_proba(X_test)
    print(
        'AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    visualizer = PrecisionRecallCurve(ada, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(ada, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  

    return roc_auc_score(y_test, pred, multi_class='ovr')

# function to train random forests and evaluate the peensembleormance

def run_ensemble(ensemble, X_train, X_test, y_train, y_test):
    
    ensemble.fit(X_train, y_train)

    print('Train set')
    pred = ensemble.predict_proba(X_train)
    print(
        'ensembleBoost roc-auc: {}'.format(roc_auc_score(y_train, pred, multi_class='ovr')))

    print('Test set')
    pred = ensemble.predict_proba(X_test)
    print(
        'ensembleBoost roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    visualizer = PrecisionRecallCurve(ensemble, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 

    visualizer = ROCAUC(ensemble, classes=[0, 1, 2, 3])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()  

    return roc_auc_score

# to save the results
results_dict = {}

for dataset in datasets_ls:
    
    results_dict[dataset] = {}    
    print(dataset)
    data = pd.read_csv('adapvtest.csv',sep=';')

    features = data[["pga","H","B","q","depth", "thickness"]]
    X = np.array(features).reshape(-1,6)
    y = np.array(data['dver'])
    X=preprocessing.MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        
    
    # train model and store result
    roc = run_randomForests(X_train, X_test, y_train, y_test)
    results_dict[dataset]['full_data'] = roc
    print()
    
    # train model and store result
    roc = run_adaboost(X_train, X_test, y_train, y_test)
    results_dict[dataset]['full_data_adaboost'] = roc
    print()
    
    for sampler in resampling_dict.keys():
        
        print(sampler)
        
        # resample
        X_resampled, y_resampled = resampling_dict[sampler].fit_resample(X_train, y_train)
        
        # train model and store result
        roc = run_randomForests(X_resampled, X_test, y_resampled, y_test)
        results_dict[dataset][sampler] = roc
        print()
    
    for ensemble in ensemble_dict.keys():
        
        print(ensemble)
        
        # train model and store result
        roc = run_ensemble(ensemble_dict[ensemble], X_train, X_test, y_train, y_test)
        results_dict[dataset][ensemble] = roc
        print()
        


# data = pd.read_csv('adapvtest.csv',sep=';')

# features = data[["pga","H","B","q","depth", "thickness"]]
# X = np.array(features).reshape(-1,6)
# y = np.array(data['dver'])
# X=preprocessing.MinMaxScaler().fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

