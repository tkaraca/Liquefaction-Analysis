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
warnings.filterwarnings("ignore")


from imblearn.under_sampling import (RandomUnderSampler,CondensedNearestNeighbour, 
                                     OneSidedSelection,
                                     TomekLinks,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                      AllKNN,
                                      NeighbourhoodCleaningRule,NearMiss,
                                      InstanceHardnessThreshold)

from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN, ADASYN, 
                                    BorderlineSMOTE, SVMSMOTE, KMeansSMOTE)

from imblearn.combine import SMOTEENN, SMOTETomek



data = pd.read_csv('adapvtest.csv',sep=';')

features = data[["pga","H","B","q","depth", "thickness"]]

print (data.dver.value_counts() / len(data))
print (data.dver.value_counts())
# plt.bar([3,1,0,2],data.dver.value_counts()/ len(data))
# plt.show()

X = np.array(features).reshape(-1,6)
y = np.array(data['dver'])
X=preprocessing.MinMaxScaler().fit_transform(X)

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X)

X_pca = pd.DataFrame(X_pca, columns =['X_pca[:, 0]', 'X_pca[:, 1]'])
y = pd.Series(y)


sns.scatterplot(
    data=X_pca, x="X_pca[:, 0]", y="X_pca[:, 1]", hue=y,alpha=0.5
)


strategy = {0:250, 1:250, 2:250, 3:250}

plt.title('Original dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

#####undersampling####
# rus = RandomUnderSampler(
#     sampling_strategy='auto',  # samples only from majority c
#     random_state=0,  # for reproducibility
#     replacement=True # if it should resample with replacement
#  )  
# X_resampled, y_resampled = rus.fit_resample(X, y)
# cnn = CondensedNearestNeighbour(
#     sampling_strategy='auto',  # undersamples only the majority class
#     random_state=0,  # for reproducibility
#     n_neighbors=1,# default
#     n_jobs=4)  # I have 4 cores in my laptop

# X_resampled, y_resampled = cnn.fit_resample(X, y)
oss = OneSidedSelection(
    sampling_strategy='auto',  # undersamples only the majority class
    random_state=0,  # for reproducibility
    n_neighbors=1,# default, algo to find the hardest instances.
    n_jobs=4)  # I have 4 cores in my laptop

# X_resampled, y_resampled = oss.fit_resample(X, y)
tl = TomekLinks(
    sampling_strategy='all',  # undersamples only the major
    n_jobs=4)  # I have 4 cores in my laptop

#X_resampled, y_resampled = tl.fit_resample(X, y)
enn = EditedNearestNeighbours(
    sampling_strategy='auto',  # undersamples only t
    n_neighbors=3, # the number of neighbours to exa
    kind_sel='all',  # all neighbours need to have t
    n_jobs=4)  # I have 4 cores in my laptop

# X_resampled, y_resampled = enn.fit_resample(X, y)

# renn = RepeatedEditedNearestNeighbours(
#     sampling_strategy='auto',# removes only the majority c
#     n_neighbors=3, # the number of neighbours to examine
#     kind_sel='all', # all neighbouring observations should
#     n_jobs=4, # 4 processors in my laptop
#     max_iter=100) # maximum number of iterations
# X_resampled, y_resampled = renn.fit_resample(X, y)
allknn = AllKNN(
    sampling_strategy='auto',  # undersamples only the majority class
    n_neighbors=5, # the maximum size of the neighbourhood to examine
    kind_sel='all',  # all neighbours need to have the same label as the observation examined
    n_jobs=4)  # I have 4 cores in my laptop
# X_resampled, y_resampled = allknn.fit_resample(X, y)
# ncr = NeighbourhoodCleaningRule(
#     sampling_strategy='auto',# undersamples from all classes except minority
#     n_neighbors=3, # explores 3 neighbours per observation
#     kind_sel='all', # all neighbouring need to disagree, only applies to cleaning step
#                     # alternatively, we can se this to mode, and then most neighbours
#                     # need to disagree to be removed.
#     n_jobs=4, # 4 processors in my laptop
#     threshold_cleaning=0.5, # the threshold to evaluate a class for cleaning (used only for clearning step)
# ) 
# X_resampled, y_resampled = ncr.fit_resample(X, y)
nm2 = NearMiss(
    sampling_strategy='auto',  # undersamples only the
    version=2,
    n_neighbors=3,
    n_jobs=4)  # I have 4 cores in my laptop

# X_resampled, y_resampled = nm2.fit_resample(X, y)

####oversampling###
ros = RandomOverSampler(
    sampling_strategy='auto', # samples all but majority class
    random_state=0,  # for reproducibility
) 
sm = SMOTE(
    sampling_strategy=strategy,  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4
)
sampler = SMOTEN(
    sampling_strategy='auto', # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4,
)
ada = ADASYN(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    n_neighbors=5,
    n_jobs=4
)
sm_b1 = BorderlineSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=3, # the neighbours to crete the new examples
    m_neighbors=5, # the neiighbours to find the DANGER group
    kind='borderline-1',
    n_jobs=4
)
sm_b2 = BorderlineSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=3, # the neighbours to crete the new examples
    m_neighbors=5, # the neiighbours to find the DANGER group
    kind='borderline-2',
    n_jobs=4
)
# sm = SVMSMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=5, # neighbours to create the synthetic examples
#     m_neighbors=10, # neighbours to determine if minority class is in "danger"
#     n_jobs=4,
#     svm_estimator = svm.SVC(kernel='linear')
# )

ksm = KMeansSMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=3,
    n_jobs=None,
    kmeans_estimator=KMeans(n_clusters=3, random_state=0),
    cluster_balance_threshold=0.1,
    density_exponent='auto'
)

###########oversampling bitti#######

#############over and under sampling#####

smenn = SMOTEENN(
    sampling_strategy=strategy,  # samples only the minority class
    random_state=0,  # for reproducibility
    smote=sm,
    enn=enn,
    n_jobs=4
)

smtomek = SMOTETomek(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    smote=sm,
    tomek=tl,
    n_jobs=4
)


X_res, y_res = smenn.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

# X_train=X_res
# y_train=y_res



###görselleştirme
print (y_train.value_counts() / len(y_train))
print (y_train.value_counts())
plt.bar([3,1,0,2],y_train.value_counts()/ len(y_train))
plt.show()

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_train)

X_pca = pd.DataFrame(X_pca, columns =['X_pca[:, 0]', 'X_pca[:, 1]'])
y_train = pd.Series(y_train)


sns.scatterplot(
    data=X_pca, x="X_pca[:, 0]", y="X_pca[:, 1]", hue=y_train,alpha=0.5
)
##################
plt.title('Oversampled and Udersampled dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

rf = OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=100, random_state=39, max_depth=4, n_jobs=4,
    )
)

#rf = RandomForestClassifier(n_estimators=100, random_state=39, max_depth=4)
rf.fit(X_train, y_train)
y_train_rf = rf.predict_proba(X_train)
y_test_rf = rf.predict_proba(X_test)
y_test_rp_rf=rf.predict(X_test)
scores_rf = cross_val_score(rf, X, y, scoring='accuracy', cv=5)


logit = LogisticRegression(
    random_state=0, multi_class='ovr', max_iter=10,
)
# train
logit.fit(X_train, y_train)
# obtain the probabilities
y_train_logit = logit.predict_proba(X_train)
y_test_logit = logit.predict_proba(X_test)
y_test_rp_logit=logit.predict(X_test)
scores_logit = cross_val_score(logit, X, y, scoring='accuracy', cv=5)


sv = svm.LinearSVC()
sv.fit(X_train, y_train)
y_train_sv = sv._predict_proba_lr(X_train)
y_test_sv = sv._predict_proba_lr(X_test)
y_test_rp_sv=sv.predict(X_test)
scores_sv = cross_val_score(sv, X, y, scoring='accuracy', cv=5)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_train_nb = nb.predict_proba(X_train)
y_test_nb = nb.predict_proba(X_test)
y_test_rp_nb=nb.predict(X_test)
scores_nb = cross_val_score(nb, X, y, scoring='accuracy', cv=5)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_train_knn = knn.predict_proba(X_train)
y_test_knn = knn.predict_proba(X_test)
y_test_rp_knn=knn.predict(X_test)
scores_knn = cross_val_score(knn, X, y, scoring='accuracy', cv=5)


print('Accuracy Score Random Forest:{}'.format(accuracy_score(y_test, rf.predict(X_test))))
print('Mean score for cross validation Random Forest:{}'.format(scores_rf.mean()))
print('Balalanced Accuracy Random Forest:{}'.format(balanced_accuracy_score(y_test, rf.predict(X_test))))
#print('Geometric Mean Random Forest:{}'.format(geometric_mean_score(y_test,rf.predict(X_test))))
#print('Precision Random Forest test:{}'.format(precision_score(y_test, rf.predict(X_test), average='macro')))
#print('Recall Random Forest test:{}'.format(recall_score(y_test, rf.predict(X_test), average='macro')))
#print('F-measure Random Forest test:{}'.format(f1_score(y_test, rf.predict(X_test), average='macro')))

print('Accuracy Score Logistic Regression:{}'.format(accuracy_score(y_test, logit.predict(X_test))))
print('Mean score for cross validation R Logistic Regression:{}'.format(scores_logit.mean()))
print('Balalanced Accuracy  Logistic Regression:{}'.format(balanced_accuracy_score(y_test, logit.predict(X_test))))

print('Accuracy Score SVM:{}'.format(accuracy_score(y_test, y_test_rp_sv)))
print('Mean score for cross validation SVM:{}'.format(scores_sv.mean()))
print('Balanced Accuracy SVM:{}'.format(balanced_accuracy_score(y_test, y_test_rp_sv)))
#print('Geometric Mean SVM:{}'.format(geometric_mean_score(y_test,sv.predict(X_test),)))
#print('Precision SVM test:{}'.format(precision_score(y_test, sv.predict(X_test), average='macro')))
#print('Recall SVM test:{}'.format(recall_score(y_test, sv.predict(X_test), average='macro')))
#print('F-measure SVM test:{}'.format(f1_score(y_test, sv.predict(X_test), average='macro')))

print('Accuracy Score GaussianNB:{}'.format(accuracy_score(y_test, y_test_rp_nb)))
print('Mean score for cross validation GaussianNB:{}'.format(scores_nb.mean()))
print('Balanced Accuracy GaussianNB:{}'.format(balanced_accuracy_score(y_test, y_test_rp_nb)))
#print('Geometric Mean GaussianNB:{}'.format(geometric_mean_score(y_test,nb.predict(X_test),)))
#print('Precision GaussianNB  test:{}'.format(precision_score(y_test, nb.predict(X_test), average='macro')))
#print('Recall GaussianNB test:{}'.format(recall_score(y_test, nb.predict(X_test), average='macro')))
#print('F-measure GaussianNB test:{}'.format(f1_score(y_test, nb.predict(X_test), average='macro')))

print('Accuracy Score KNeighbors:{}'.format(accuracy_score(y_test, knn.predict(X_test))))
print('Mean score for cross validation KNeighbors:{}'.format(scores_knn.mean()))
print('Balanced Accuracy KNeighbors:{}'.format(balanced_accuracy_score(y_test, knn.predict(X_test))))
#print('Geometric Mean KNeighbors:{}'.format(geometric_mean_score(y_test,knn.predict(X_test),)))
#print('Precision KNeighbors test:{}'.format(precision_score(y_test, knn.predict(X_test), average='macro')))
#print('Recall KNeighbors test:{}'.format(recall_score(y_test, knn.predict(X_test), average='macro')))
#print('F-measure KNeighbors test:{}'.format(f1_score(y_test, knn.predict(X_test), average='macro')))

print('Train set')
print('ROC-AUC Random Forest test:{}'.format(roc_auc_score(y_train, y_train_rf ,multi_class='ovr')))
print('ROC-AUC Logistic Regression:{}'.format(roc_auc_score(y_train, y_train_logit,multi_class='ovr')))
print('ROC-AUC Support Vector Machine:{}'.format(roc_auc_score(y_train, y_train_sv, multi_class='ovr')))
print('ROC-AUC GaussianNB:{}'.format(roc_auc_score(y_train, y_train_nb, multi_class='ovr')))
print('ROC-AUC K NeighborsClassifier:{}'.format( roc_auc_score(y_train, y_train_knn, multi_class='ovr')))     

print('Test set')
print('Precision Random Forest test:{}'.format(precision_score(y_test, y_test_rp_rf, average='macro')))
print('Recall Random Forest test:{}'.format(recall_score(y_test, y_test_rp_rf, average='macro')))
print('F-measure Random Forest test:{}'.format(f1_score(y_test, y_test_rp_rf, average='macro')))

print('Test set')
print('Precision Random Logistic Regression:{}'.format(precision_score(y_test, y_test_rp_logit, average='macro')))
print('Recall Random Logistic Regression:{}'.format(recall_score(y_test, y_test_rp_logit, average='macro')))
print('F-measure Random Logistic Regression:{}'.format(f1_score(y_test, y_test_rp_logit, average='macro')))


print('Precision SVM test:{}'.format(precision_score(y_test, y_test_rp_sv, average='macro')))
print('Recall SVM test:{}'.format(recall_score(y_test, y_test_rp_sv, average='macro')))
print('F-measure SVM test:{}'.format(f1_score(y_test, y_test_rp_sv, average='macro')))

print('Precision GaussianNB test:{}'.format(precision_score(y_test, y_test_rp_nb, average='macro')))
print('Recall GaussianNB test:{}'.format(recall_score(y_test, y_test_rp_nb, average='macro')))
print('F-measure GaussianNB test:{}'.format(f1_score(y_test, y_test_rp_nb, average='macro')))

print('Precision K NeighborsClassifier test:{}'.format(precision_score(y_test, y_test_rp_knn, average='macro')))
print('Recall K NeighborsClassifier test:{}'.format(recall_score(y_test, y_test_rp_knn, average='macro')))
print('F-measure K NeighborsClassifier test:{}'.format(f1_score(y_test, y_test_rp_knn, average='macro')))

print('ROC-AUC Random Forest test:{}'.format(roc_auc_score(y_test, y_test_rf,multi_class='ovr')))
print('ROC-AUC Logistic Regression:{}'.format(roc_auc_score(y_test, y_test_logit ,multi_class='ovr')))
print('ROC-AUC Support Vector Machine:{}'.format(roc_auc_score(y_test, y_test_sv,multi_class='ovr')))
print('ROC-AUC GaussianNB:{}'.format(roc_auc_score(y_test,y_test_nb, multi_class='ovr')))
print('ROC-AUC K NeighborsClassifier:{}'.format( roc_auc_score(y_test, y_test_knn, multi_class='ovr')))     


visualizer = ClassificationReport(rf)

visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()


# visualizer = ClassificationReport(sv)

# visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
# visualizer.score(X_test, y_test)        # Evaluate the model on the test data
# visualizer.show()


# visualizer = ClassificationReport(nb)

# visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
# visualizer.score(X_test, y_test)        # Evaluate the model on the test data
# visualizer.show()


# visualizer = ClassificationReport(knn)

# visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
# visualizer.score(X_test, y_test)        # Evaluate the model on the test data
# visualizer.show()

visualizer = ROCAUC(rf, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = ROCAUC(logit, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = ROCAUC(sv, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = ROCAUC(nb, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = ROCAUC(knn, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  



visualizer = PrecisionRecallCurve(rf, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()      

visualizer = PrecisionRecallCurve(logit, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()   

visualizer = PrecisionRecallCurve(sv, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = PrecisionRecallCurve(nb, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

visualizer = PrecisionRecallCurve(knn, classes=[0, 1, 2, 3])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show() 







# svm = OneVsRestClassifier(
#     KNeighborsClassifier()
# )

# # train the model
# rff.fit(X_train, y_train)

# # produce the predictions (as probabilities)
# y_train_rff = rf.predict_proba(X_train)
# y_test_rff = rf.predict_proba(X_test)


y_test = label_binarize(y_test, classes=[0, 1, 2, 3])



fpr, tpr, thresholds = roc_curve(y_test[:, 3], y_test_rf[:, 3])
fpr_rf = dict()
tpr_rf = dict()

# for each class
for i in range(4):
    
    # determine tpr and fpr at various thresholds
    # in a 1 vs all fashion
    fpr_rf[i], tpr_rf[i], _ = roc_curve(
        y_test[:, i], y_test_rf[:, i])
    


fpr, tpr, thresholds = roc_curve(y_test[:, 3], y_test_logit[:, 3])
fpr_lg = dict()
tpr_lg = dict()
# for each class
for i in range(4):
    
    # determine precision and recall at various thresholds
    # in a 1 vs all fashion
    fpr_lg[i], tpr_lg[i], _ = roc_curve(
        y_test[:, i], y_test_logit[:, i])
    

fpr, tpr, thresholds = roc_curve(y_test[:, 3], y_test_sv[:, 3])
fpr_sv = dict()
tpr_sv = dict()
# for each class
for i in range(4):
    
    # determine precision and recall at various thresholds
    # in a 1 vs all fashion
    fpr_lg[i], tpr_lg[i], _ = roc_curve(
        y_test[:, i], y_test_logit[:, i])
    

fpr, tpr, thresholds = roc_curve(y_test[:, 3], y_test_nb[:, 3])
fpr_nb = dict()
tpr_nb = dict()
for i in range(4):
    
    # determine precision and recall at various thresholds
    # in a 1 vs all fashion
    fpr_nb[i], tpr_nb[i], _ = roc_curve(
        y_test[:, i], y_test_nb[:, i])

fpr, tpr, thresholds = roc_curve(y_test[:, 3], y_test_knn[:, 3])
fpr_knn = dict()
tpr_knn = dict()
for i in range(4):
    
    # determine precision and recall at various thresholds
    # in a 1 vs all fashion
    fpr_knn[i], tpr_knn[i], _ = roc_curve(
        y_test[:, i], y_test_knn[:, i])




# Compute micro-average ROC curve and ROC area
fpr_rf["micro"], tpr_rf["micro"], _ = roc_curve(
    y_test.ravel(), y_test_rf.ravel(),
)

# for logistic regression
# Compute micro-average ROC curve and ROC area
fpr_lg["micro"], tpr_lg["micro"], _ = roc_curve(
    y_test.ravel(), y_test_logit.ravel(),
)

# for logistic regression
# Compute micro-average ROC curve and ROC area
fpr_sv["micro"], tpr_sv["micro"], _ = roc_curve(
    y_test.ravel(), y_test_sv.ravel(),
)

fpr_nb["micro"], tpr_nb["micro"], _ = roc_curve(
    y_test.ravel(), y_test_nb.ravel(),
)

fpr_knn["micro"], tpr_knn["micro"], _ = roc_curve(
    y_test.ravel(), y_test_knn.ravel(),
)

i = "micro"

plt.plot(fpr_lg[i], tpr_lg[i], label='logit micro {}')
plt.plot(fpr_rf[i], tpr_rf[i], label='rf micro {}')
plt.plot(fpr_sv[i], tpr_sv[i], label='sv micro {}')
plt.plot(fpr_nb[i], tpr_nb[i], label='nb micro {}')
plt.plot(fpr_knn[i], tpr_knn[i], label='knn micro {}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


print('Random Forest ovr,ovo')
micro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_rf, multi_class="ovo", average="micro")
macro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_rf, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_test_rf, multi_class="ovo", average="weighted")
micro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_rf, multi_class="ovr", average="micro")
macro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_rf, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_test_rf, multi_class="ovr", average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovr, weighted_roc_auc_ovr))


print('Logistic regression ovr,ovo')
micro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_logit, multi_class="ovo", average="micro")
macro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_logit, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_test_logit, multi_class="ovo", average="weighted")
micro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_logit, multi_class="ovr", average="micro")
macro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_logit, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_test_logit, multi_class="ovr", average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovr, weighted_roc_auc_ovr))



print('SVM ovr,ovo')
micro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_sv, multi_class="ovo", average="micro")
macro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_sv, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_test_sv, multi_class="ovo", average="weighted")

micro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_sv, multi_class="ovr", average="micro")
macro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_sv, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_test_sv, multi_class="ovr", average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovr, weighted_roc_auc_ovr))


print('NB ovr,ovo')
micro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_nb, multi_class="ovo", average="micro")
macro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_nb, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_test_nb, multi_class="ovo", average="weighted")
micro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_nb, multi_class="ovr", average="micro")
macro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_nb, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_test_nb, multi_class="ovr", average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovr, weighted_roc_auc_ovr))


print('KNN ovr,ovo')
micro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_knn, multi_class="ovo", average="micro")
macro_roc_auc_ovo = roc_auc_score(
    y_test, y_test_knn, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_test_knn, multi_class="ovo", average="weighted")
micro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_knn, multi_class="ovr", average="micro")
macro_roc_auc_ovr = roc_auc_score(
    y_test, y_test_knn, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_test_knn, multi_class="ovr", average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (micro),\n{:.6f} (macro),\n{:.6f} "
      "(weighted)"
      .format(micro_roc_auc_ovr, macro_roc_auc_ovr, weighted_roc_auc_ovr))




#####undersampling bölümü

# undersampler_dict = {

#     'random': RandomUnderSampler(
#         sampling_strategy='auto',
#         random_state=0,
#         replacement=False),

#     'cnn': CondensedNearestNeighbour(
#         sampling_strategy='auto',
#         random_state=0,
#         n_neighbors=1,
#         n_jobs=4),

#     'tomek': TomekLinks(
#         sampling_strategy='auto',
#         n_jobs=4),

#     'oss': OneSidedSelection(
#         sampling_strategy='auto',
#         random_state=0,
#         n_neighbors=1,
#         n_jobs=4),

#     'enn': EditedNearestNeighbours(
#         sampling_strategy='auto',
#         n_neighbors=3,
#         kind_sel='all',
#         n_jobs=4),

#     'renn': RepeatedEditedNearestNeighbours(
#         sampling_strategy='auto',
#         n_neighbors=3,
#         kind_sel='all',
#         n_jobs=4,
#         max_iter=100),

#     'allknn': AllKNN(
#         sampling_strategy='auto',
#         n_neighbors=3,
#         kind_sel='all',
#         n_jobs=4),

#     'ncr': NeighbourhoodCleaningRule(
#         sampling_strategy='auto',
#         n_neighbors=3,
#         kind_sel='all',
#         n_jobs=4,
#         threshold_cleaning=0.5),

#     'nm1': NearMiss(
#         sampling_strategy='auto',
#         version=1,
#         n_neighbors=3,
#         n_jobs=4),


#     'iht': InstanceHardnessThreshold(
#         estimator=LogisticRegression(random_state=0),
#         sampling_strategy='auto',
#         random_state=0,
#         n_jobs=4,
#         cv=3)}

# def run_randomForests(X_train, X_test, y_train, y_test):

#     # rf = RandomForestClassifier(
#     #     n_estimators=200, random_state=39, max_depth=4, n_jobs=4,
#     # )
#     # rf = LogisticRegression(
#     #     random_state=0, multi_class='ovr', max_iter=10,
#     #     )
#     # rf = GaussianNB()

#     # rf = KNeighborsClassifier(n_neighbors=2)
#     rf = svm.LinearSVC()
    
#     rf.fit(X_train, y_train)

#     print('Train set')
#     pred = rf._predict_proba_lr(X_train)
#     print(
#         'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred,multi_class='ovr')))

#     print('Test set')
#     pred = rf._predict_proba_lr(X_test)
#     print(
#         'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
#     # NOTE: that this function returns the ROC over the test set
#     # which is the portion of the data that would not be under-sampled
#     return roc_auc_score(y_test, pred ,multi_class='ovr')



# # for each dataset

    
# # start a new dictionary per dataset
# results_dict = {}



# # roc = run_randomForests(X_train, X_test, y_train, y_test)

# # # store results
# # #results_dict = roc
# # shapes_dict = len(X_train)

# # print()

# # # now, we test the different under-samplers, 1 at a time
# for undersampler in undersampler_dict.keys():
    
#     print(undersampler)
    
#     # resample the train set only
#     X_resampled, y_resampled = undersampler_dict[undersampler].fit_resample(X_train, y_train)
    
#     # train model and evaluate performance
    
#     # Note the performance returned is using the
#     # test set, which was not under-sampled
    
#     roc = run_randomForests(X_resampled, X_test, y_resampled, y_test)
    
#     # store results
#     results_dict[undersampler]= roc
#     #shapes_dict = len(X_resampled)
# print()
    


# for undersampler in undersampler_dict.keys():
#     pd.Series(results_dict).plot.bar()
#     plt.ylabel('roc-auc')
#     #plt.ylim(0.55, 0.9)
#     plt.axhline(results_dict[undersampler], color='r')
#     plt.show()
#     ###undersampling bitti



# ####OVERSAMPLİNG
#  ##oversampling dictionary 
# undersampler_dict = {

#     'random': RandomOverSampler(
#         sampling_strategy='auto',
#         random_state=0),

#     'smote': SMOTE(
#         sampling_strategy='auto',  # samples only the minority class
#         random_state=0,  # for reproducibility
#         k_neighbors=5,
#         n_jobs=4),

#     'adasyn': ADASYN(
#         sampling_strategy='auto',  # samples only the minority class
#         random_state=0,  # for reproducibility
#         n_neighbors=5,
#         n_jobs=4),

#     'border1': BorderlineSMOTE(
#         sampling_strategy='auto',  # samples only the minority class
#         random_state=0,  # for reproducibility
#         k_neighbors=5,
#         m_neighbors=10,
#         kind='borderline-1',
#         n_jobs=4),

#     'border2': BorderlineSMOTE(
#         sampling_strategy='auto',  # samples only the minority class
#         random_state=0,  # for reproducibility
#         k_neighbors=5,
#         m_neighbors=10,
#         kind='borderline-2',
#         n_jobs=4),

#     'svm': SVMSMOTE(
#         sampling_strategy='auto',  # samples only the minority class
#         random_state=0,  # for reproducibility
#         k_neighbors=5,
#         m_neighbors=10,
#         n_jobs=4,
#         svm_estimator=svm.SVC(kernel='linear')),
# }
# ##oversampling dictionary 



# def run_randomForests(X_train, X_test, y_train, y_test):

#     # rf = RandomForestClassifier(
#     #     n_estimators=200, random_state=39, max_depth=4, n_jobs=4,
#     # )
#     # rf = LogisticRegression(
#     # #     random_state=0, multi_class='ovr', max_iter=10,
#     #      )
#     # rf = GaussianNB()

#     # rf = KNeighborsClassifier(n_neighbors=5)
#     rf = svm.LinearSVC()
    
#     rf.fit(X_train, y_train)

#     print('Train set')
#     pred = rf._predict_proba_lr(X_train)
#     print(
#         'Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred,multi_class='ovr')))

#     print('Test set')
#     pred = rf._predict_proba_lr(X_test)
#     print(
#         'Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred, multi_class='ovr')))
    
#     # NOTE: that this function returns the ROC over the test set
#     # which is the portion of the data that would not be under-sampled
#     return roc_auc_score(y_test, pred ,multi_class='ovr')


# results_dict = {}



# for undersampler in undersampler_dict.keys():
    
#     print(undersampler)
    
#     # resample the train set only
#     X_resampled, y_resampled = undersampler_dict[undersampler].fit_resample(X_train, y_train)
    
  
    
#     roc = run_randomForests(X_resampled, X_test, y_resampled, y_test)
    
#     # store results
#     results_dict[undersampler]= roc
    
# print()
    


# for undersampler in undersampler_dict.keys():
#     pd.Series(results_dict).plot.bar()
#     plt.title('Results')
#     plt.ylabel('roc-auc')
#     #plt.ylim(0.6, 1)
#     plt.axhline(results_dict[undersampler], color='r')
#     plt.show()
# ##oversampling bitti



# function to train random forests and evaluate the performance

# ***with cross-validation başladı***

# def run_model(X_train, y_train, oversampler):
    
#     # set up the classifier
#     rf = RandomForestClassifier(
#             n_estimators=100, random_state=39, max_depth=3, n_jobs=4
#         )
    
#     # set up a scaler 
#     # (as the oversampling techniques use KNN
#     # we put the variables in the same scale)
#     #scaler = MinMaxScaler()
    
#     # without sampling:
#     if not oversampler:

#         model = rf
    
#     # set up a pipeline with sampling:
#     else:
        
#         # important to scale before the re-sampler
#         # as the many of methods require the variables in 
#         # a similar scale
#         model = make_pipeline(
#             #scaler,
#             oversampler,
#             rf,
#         )
        
        
#     # When we make a pipeline and then run the training of the model
#     # with cross-validation, the procedure works as follows:
    
#     # 1) take 2 of the 3 fold as train set
#     # 2) resample the 2 fold (aka, the train set)
#     # 3) train the model on the resampled data from point 2
#     # 4) evaluate performance on the 3rd fold, that was not resampled
    
#     # this way, we make sure that we are not evaluating the performance
#     # of our classifier on the over-sampled data
    
#     cv_results = cross_validate(
#         model, # the random forest or the pipeline
#         X_train, # the data that will be used in the cross-validation
#         y_train, # the target
#         scoring="average_precision", # the metric that we want to evaluate
#         cv=4, # the cross-validation fold
#     )

#     print(
#         'Random Forests average precision: {0} +/- {1}'.format(
#         cv_results['test_score'].mean(), cv_results['test_score'].std()
#         )
#     )

#     return cv_results['test_score'].mean(), cv_results['test_score'].std()


# pr_mean_dict = {}
# pr_std_dict = {}
    

    

    
# for oversampler in undersampler_dict.keys():
        
#         print(oversampler)
               
#         # resample, train and evaluate performance
#         # with cross-validation
#         aps_mean, aps_std = run_model(X_train, y_train, undersampler_dict[oversampler])
        
#         #store results
#         pr_mean_dict[oversampler] = aps_mean
#         pr_std_dict[oversampler] = aps_std
#         print()
        
# print()


    
# pr_mean_s = pd.Series(pr_mean_dict)
# pr_std_s = pd.Series(pr_std_dict)
# pr_mean_s.plot.bar(yerr=[pr_std_s, pr_std_s])
# plt.ylabel('Average Precision')
# plt.show()
# ###cros validation bitti






























