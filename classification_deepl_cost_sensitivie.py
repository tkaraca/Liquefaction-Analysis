import tensorflow 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score,roc_curve, RocCurveDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)

strategy = {0:500, 1:500, 2:500, 3:500}

data = pd.read_csv('adapvtest.csv', sep=';')

data.dver.value_counts()

X = data[ ["pga","H","B","q","depth", "thickness"]]
X = preprocessing.MinMaxScaler().fit_transform(X)
labels = np.array(data.dver).reshape(-1,1)
encoder = OneHotEncoder()
y = encoder.fit_transform(labels).toarray()

sm = SMOTE(
    sampling_strategy=strategy,  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=1
)

X, y = sm.fit_resample(X, y)

enn = EditedNearestNeighbours(
    sampling_strategy='all',
    n_neighbors=3,
    kind_sel='all',
    n_jobs=1)


smenn = SMOTEENN(
    sampling_strategy=strategy,  # samples only the minority class
    random_state=0,  # for reproducibility
    smote=sm,
    enn=enn,
    n_jobs=1
)




weight_0= 1/45*184/4
weight_1= 1/46*184/4
weight_2= 1/24*184/4
weight_3= 1/69*184/4
class_weight={0:weight_0, 1:weight_1, 2:weight_2, 3:weight_3}
# sampler = SMOTEN(
#     sampling_strategy='auto', # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=5,
#     n_jobs=1,
# )

# X_res, y_res = sampler.fit_resample(X,y)

# ada = ADASYN(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     n_neighbors=3,
#     n_jobs=1
# )

# X_res, y_res = ada.fit_resample(X, y)

# sm_b1 = BorderlineSMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=4, # the neighbours to crete the new examples
#     m_neighbors=2, # the neiighbours to find the DANGER group
#     kind='borderline-1',
#     n_jobs=1
# )

# sm_b2 = BorderlineSMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=2, # the neighbours to crete the new examples
#     m_neighbors=3, # the neiighbours to find the DANGER group
#     kind='borderline-2',
#     n_jobs=4
# )
# X_res, y_res = sm_b2.fit_resample(X, y)

# sm = SVMSMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=2, # neighbours to create the synthetic examples
#     m_neighbors=3, # neighbours to determine if minority class is in "danger"
#     n_jobs=1,
#     svm_estimator = svm.SVC(kernel='linear')
# )

# X_res, y_res = sm.fit_resample(X, y)

# sm = KMeansSMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=2,
#     n_jobs=None,
#     kmeans_estimator=KMeans(n_clusters=4, random_state=0),
#     cluster_balance_threshold=0.1,
#     density_exponent='auto'
# )

# X_res, y_res = sm.fit_resample(X, y)






# tl = TomekLinks(
#     sampling_strategy='auto',  # undersamples only the majority class
#     n_jobs=4)  # I have 4 cores in my laptop

# X_resampled, y_resampled = tl.fit_resample(X, y)

# ros = RandomOverSampler(
#     sampling_strategy='auto', # samples only the minority class
#     random_state=0,  # for reproducibility
# )  

# X_res, y_res = ros.fit_resample(X, y)

# ros = RandomOverSampler(
#         sampling_strategy='auto', # samples only the minority class
#         random_state=50,  # for reproducibility
#         shrinkage = 10,
#  ) 
# X_res, y_res = ros.fit_resample(X, y)

# sm = SMOTE(
#     sampling_strategy='auto',  # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=5,
#     n_jobs=4
# )

#X_res, y_res = sm.fit_resample(X, y)
# smnc = SMOTENC(
#     sampling_strategy='auto', # samples only the minority class
#     random_state=0,  # for reproducibility
#     k_neighbors=5,
#     n_jobs=4,
#     categorical_features=[2,3] # indeces of the columns of categorical variables
# )  

#X_res, y_res = smnc.fit_resample(X, y)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.20)
X_smenn, y_smenn = smenn.fit_resample(trainX, trainY)

trainX=X_smenn
trainY=y_smenn

model = Sequential()
# first parameter is output dimension
model.add(Dense(200, input_dim=6, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(4, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...
optimizer = Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(trainX, trainY, validation_split = 0.1, epochs=10000, batch_size=200, verbose=2, class_weight=class_weight)

results = model.evaluate(testX, testY)
y_test_dl = model.predict(testX)
#scores = cross_val_score(trainX, testX, scoring='accuracy', cv=10)
f=np.array([[0.5,15,14,50,1,12.3]])
sonuc = model.predict(f)

print("Accuracy on the test dataset: %.2f" % results[1])
print(sonuc)

print('ROC-AUC Random Forest test:{}'.format(roc_auc_score(testY, y_test_dl,multi_class='ovr')))
#print('Precision Random Logistic Regression:{}'.format(precision_score(testY, y_test_dl)))
#print('Recall Random Logistic Regression:{}'.format(recall_score(testY, y_test_dl)))
#print('F-measure Random Logistic Regression:{}'.format(f1_score(testY, y_test_dl)))