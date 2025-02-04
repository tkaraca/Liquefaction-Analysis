import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

###################################Featureleri optimize et ve scatterplot çiz################
# Veri setini yükleme
file_path = 'G:\\My Drive\\deprem makale\\pythonProject2\\adapvtest.csv'
data = pd.read_csv('adapvtest.csv',sep=';')

# Özellikler ve hedef değişkeni ayırma
X = data[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
y = data['dver']

# Random Forest ile özellik önem sıralaması
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importances = rf.feature_importances_

# En etkili iki özelliği seçme
important_features = X.columns[np.argsort(feature_importances)[-2:]]  # Son iki en önemli özellik
print(f"En etkili iki özellik: {important_features[0]} ve {important_features[1]}")

# Scatter plot
plt.figure(figsize=(10, 6))
for class_label in y.unique():
    plt.scatter(X[important_features[0]][y == class_label], X[important_features[1]][y == class_label],
                label=f"Class {class_label}", alpha=0.7)
plt.title(f"Scatter Plot of {important_features[0]} vs {important_features[1]}")
plt.xlabel(important_features[0])
plt.ylabel(important_features[1])
plt.legend(title="Classes", fontsize=10)
plt.grid(alpha=0.3)
plt.show()
#####################################################################

# Load the uploaded dataset
# file_path = 'G:\My Drive\deprem makale\pythonProject2\adapvtest.csv'
# data = pd.read_csv('adapvtest.csv',sep=';')

# Display the first few rows to understand the structure of the dataset
data.head(), data.info(), data.describe()

# Veriyi kontrol et
print(data.head())

# Sınıf dağılımını kontrol et
print(data['dver'].value_counts())

# Sınıf dağılımını kontrol et
class_distribution = data['dver'].value_counts(normalize=True) * 100
print(class_distribution)

plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution for dver (Damage Group)', fontsize=14)
plt.xlabel('Damage Group', fontsize=12)
plt.ylabel('Percentage of Samples (%)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Statistical summary of numerical features
feature_stats = data.describe()

# Display the summary statistics to the user

# Statistical summary of numerical features
feature_stats = data.describe()

# Print the summary statistics to the console
print("Feature Statistical Summary:")
print(feature_stats)

# Plotting feature distributions by class
import matplotlib.pyplot as plt

features = ['pga', 'H', 'B', 'q', 'depth', 'thickness']
plt.figure(figsize=(16, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    for cls in sorted(data['dver'].unique()):
        subset = data[data['dver'] == cls]
        subset[feature].plot(kind='kde', label=f"Class {cls}", alpha=0.7)
    plt.title(f"Distribution of {feature} by Class")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout()
plt.show()


# Mean and standard deviation for each feature grouped by class
class_stats = data.groupby('dver').agg(['mean', 'std'])

# Flattening MultiIndex columns for better readability
class_stats.columns = ['_'.join(col) for col in class_stats.columns]

# Print the statistics by class
print("Class-wise Mean and Standard Deviation:")
print(class_stats)

# Correlation analysis
correlation_matrix = data.corr()

# Plot the correlation matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm', alpha=0.9)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix", pad=20)
plt.show()


# Boxplot for each feature to identify outliers
# Boxplot for each feature to identify outliers
import matplotlib.pyplot as plt

features = ['pga', 'H', 'B', 'q', 'depth', 'thickness']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Alt grafikler için yerleşim
axes = axes.flatten()

for i, feature in enumerate(features):
    data.boxplot(column=feature, by='dver', grid=False, showfliers=True, ax=axes[i])
    axes[i].set_title(f"{feature} Distribution by Class")
    axes[i].set_xlabel("Damage Group (dver)")
    axes[i].set_ylabel(feature)

# Fazladan üst başlığı kaldır
plt.suptitle("")  # Üst başlık tamamen kaldırılıyor
plt.tight_layout()  # Alt grafiklerin düzenlenmesi
plt.show()


# 

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# Veri kümenizi yükleyin
# data = pd.read_csv('path_to_your_dataset.csv')  # Veri kümesini tekrar yükleyin

# 1. Winsorization
winsorized_data = data.copy()
for feature in ['pga', 'H', 'B', 'q', 'depth', 'thickness']:
    winsorized_data[feature] = winsorize(winsorized_data[feature], limits=(0.05, 0.05))

print("Winsorized Data Description:")
print(winsorized_data.describe())

# 2. Z-Skor Filtreleme
def remove_outliers_zscore(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    z_scores = (data[column] - mean) / std
    return data[np.abs(z_scores) <= threshold]

zscore_cleaned_data = data.copy()
for feature in ['pga', 'H', 'B', 'q', 'depth', 'thickness']:
    zscore_cleaned_data = remove_outliers_zscore(zscore_cleaned_data, feature)

print(f"\nNumber of rows removed due to Z-Score filtering: {len(data) - len(zscore_cleaned_data)}")
print("Z-Score Filtered Data Description:")
print(zscore_cleaned_data.describe())

# 3. IQR (Interquartile Range) Filtreleme
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

iqr_cleaned_data = data.copy()
for feature in ['pga', 'H', 'B', 'q', 'depth', 'thickness']:
    iqr_cleaned_data = remove_outliers_iqr(iqr_cleaned_data, feature)

print(f"\nNumber of rows removed due to IQR filtering: {len(data) - len(iqr_cleaned_data)}")
print("IQR Filtered Data Description:")
print(iqr_cleaned_data.describe())

# 4. Karşılaştırma
print("\nOriginal Data Size:", len(data))
print("Winsorized Data Size:", len(winsorized_data))
print("Z-Score Filtered Data Size:", len(zscore_cleaned_data))
print("IQR Filtered Data Size:", len(iqr_cleaned_data))


#####################################modeller ve analizleri 3 veri seti ve hyperparametre tunning################

# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# import numpy as np
# from sklearn.preprocessing import label_binarize

# # Veri setlerini ayrı ayrı analiz etmek için listeler halinde düzenleme
# datasets = [
#     ("Winsorized", winsorized_data),
#     ("Z-Score Filtered", zscore_cleaned_data),
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # Neural Network hiperparametre kombinasyonları
# nn_params = {
#     "learning_rates": [0.001, 0.01],
#     "batch_sizes": [16, 32],
#     "epochs": [50, 100]
# }

# # GradientBoostingClassifier için hiperparametre aralığı (örnek)
# gb_params = {
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5],
#     'n_estimators': [100, 200],
#     'subsample': [0.8, 1],
#     # Dilerseniz 'min_samples_split', 'min_samples_leaf' vb. ekleyebilirsiniz.
# }

# def compute_macro_roc(y_test_bin, y_prob):
#     """
#     Çoklu sınıfta (One-vs-Rest) her sınıfın ROC eğrisini hesaplayıp
#     bunların ortalamasını (macro-average) döndürür.
#     """
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_test_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc

# for dataset_name, dataset in datasets:
#     print(f"\n### {dataset_name} Dataset ###")
    
#     # Özellikler (X) ve hedef değişken (y) ayırma
#     X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#     y = dataset['dver']
    
#     # Veriyi eğitim ve test olarak ayırma
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # Çoklu sınıf (ör. 0,1,2...) senaryoda test seti etiketlerini binarize ederek
#     # One-vs-Rest ROC hesaplaması yapacağız.
#     unique_classes = np.unique(y_train)
#     y_test_binarized = label_binarize(y_test, classes=unique_classes)
    
#     # Model sonuçlarını saklama
#     models = {}
#     classification_reports = {}
    
#     # ----------------------------
#     # 1. Random Forest
#     # ----------------------------
#     print("\nOptimizasyon: Random Forest")
#     rf_params = {
#         'n_estimators': [100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
#     rf_grid = GridSearchCV(
#         RandomForestClassifier(random_state=42),
#         rf_params,
#         cv=3,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     rf_grid.fit(X_train, y_train)
#     best_rf = rf_grid.best_estimator_
#     y_pred_rf = best_rf.predict(X_test)
#     y_pred_prob_rf = best_rf.predict_proba(X_test)
    
#     models["Random Forest"] = y_pred_prob_rf
#     classification_reports["Random Forest"] = classification_report(y_test, y_pred_rf)
#     print("Best Random Forest Parameters:", rf_grid.best_params_)
    
#     # ----------------------------
#     # 2. XGBoost
#     # ----------------------------
#     print("\nOptimizasyon: XGBoost")
#     xgb_params = {
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5],
#         'n_estimators': [100, 200],
#         'subsample': [0.8, 1],
#         'colsample_bytree': [0.8, 1]
#     }
#     xgb_grid = GridSearchCV(
#         XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#         xgb_params,
#         cv=3,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     xgb_grid.fit(X_train, y_train)
#     best_xgb = xgb_grid.best_estimator_
#     y_pred_xgb = best_xgb.predict(X_test)
#     y_pred_prob_xgb = best_xgb.predict_proba(X_test)
    
#     models["XGBoost"] = y_pred_prob_xgb
#     classification_reports["XGBoost"] = classification_report(y_test, y_pred_xgb)
#     print("Best XGBoost Parameters:", xgb_grid.best_params_)
    
#     # ----------------------------
#     # 3. Neural Network
#     # ----------------------------
#     print("\nOptimizasyon: Neural Network")
#     best_accuracy = 0
#     best_nn_params = {}
#     best_nn_model = None
    
#     for lr in nn_params["learning_rates"]:
#         for bs in nn_params["batch_sizes"]:
#             for ep in nn_params["epochs"]:
#                 model = Sequential([
#                     Dense(64, input_dim=X_train.shape[1], activation='relu'),
#                     Dense(32, activation='relu'),
#                     Dense(len(unique_classes), activation='softmax')
#                 ])
#                 model.compile(
#                     optimizer=Adam(learning_rate=lr),
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy']
#                 )
#                 model.fit(X_train, y_train, epochs=ep, batch_size=bs, verbose=0)
#                 nn_eval = model.evaluate(X_test, y_test, verbose=0)
                
#                 if nn_eval[1] > best_accuracy:
#                     best_accuracy = nn_eval[1]
#                     best_nn_params = {"learning_rate": lr, "batch_size": bs, "epochs": ep}
#                     best_nn_model = model
    
#     print("Best Neural Network Parameters:", best_nn_params)
#     y_pred_prob_nn = best_nn_model.predict(X_test)
#     y_pred_nn_classes = y_pred_prob_nn.argmax(axis=1)
#     models["Neural Network"] = y_pred_prob_nn
#     classification_reports["Neural Network"] = classification_report(y_test, y_pred_nn_classes)
    
#     # ----------------------------
#     # 4. Gradient Boosting
#     # ----------------------------
#     print("\nOptimizasyon: Gradient Boosting")
#     gb_grid = GridSearchCV(
#         GradientBoostingClassifier(random_state=42),
#         gb_params,
#         cv=3,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     y_pred_gb = best_gb.predict(X_test)
#     y_pred_prob_gb = best_gb.predict_proba(X_test)
    
#     models["Gradient Boosting"] = y_pred_prob_gb
#     classification_reports["Gradient Boosting"] = classification_report(y_test, y_pred_gb)
#     print("Best Gradient Boosting Parameters:", gb_grid.best_params_)
    
#     # ========================
#     # ROC-AUC Eğrileri
#     # (5 Subplot: 4 model + 1 macro-average)
#     # ========================
#    # ROC Eğrileri ve Macro-Average ROC (2 Satır x 3 Sütun Düzeni)
#     plt.figure(figsize=(24, 16))  # Büyük boyut

#     # ROC Subplots (2 Satır x 3 Sütun Düzeni)
#     for idx, (model_name, probs) in enumerate([
#         ("RandomForest", y_pred_prob_rf),
#         ("XGBoost", y_pred_prob_xgb),
#         ("NeuralNetwork", y_pred_prob_nn),
#         ("GradientBoosting", y_pred_prob_gb)
#     ], start=1):
#         plt.subplot(2, 3, idx)
#         for class_label in unique_classes:
#             class_idx = np.where(unique_classes == class_label)[0][0]
#             fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx], probs[:, class_idx])
#             auc_val = auc(fpr, tpr)
#             plt.plot(
#                 fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})", linewidth=2
#             )
#         plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)  # Diyagonal çizgi
#         plt.title(f"{model_name} ROC", fontsize=16, weight='bold')  # Bold başlık
#         plt.xlabel("False Positive Rate", fontsize=14, weight='bold')  # X ekseni bold
#         plt.ylabel("True Positive Rate", fontsize=14, weight='bold')  # Y ekseni bold
#         plt.legend(loc="lower right", fontsize=14, title="Classes", title_fontsize=14, frameon=True, prop={'size': 16,'weight': 'bold'})  # Legend

#     # Macro-Average ROC Curve (Alt Sağ Grafik)
#     plt.subplot(2, 3, 5)
#     for model_name, probs in {
#         "RF": y_pred_prob_rf,
#         "XGB": y_pred_prob_xgb,
#         "NN": y_pred_prob_nn,
#         "GB": y_pred_prob_gb
#     }.items():
#         mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_binarized, probs)
#         plt.plot(
#             mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})", linewidth=2.5
#         )
#     plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)  # Diyagonal çizgi
#     plt.title("Macro-Average ROC Curve", fontsize=16, weight='bold')  # Bold başlık
#     plt.xlabel("False Positive Rate", fontsize=14, weight='bold')  # X ekseni bold
#     plt.ylabel("True Positive Rate", fontsize=14, weight='bold')  # Y ekseni bold
#     plt.legend(loc="lower right", fontsize=14, title="Models", title_fontsize=14, frameon=True, prop={'size': 16,'weight': 'bold'})  # Legend

#     # Genel Başlık
#     plt.suptitle(f"ROC Curves for {dataset_name} Dataset", fontsize=22, weight='bold')  # Genel başlık
#     plt.tight_layout(rect=[0, 0, 1, 0.92])  # Genel başlık ve alt yazılar için düzenleme
#     #plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)  # Alt yazılar ve subplot arası alan
#     plt.show()

    
#     # Sınıflandırma raporlarını yazdırma
#     print(f"\nClassification Reports for {dataset_name}:")
#     for model_name, report in classification_reports.items():
#         print(f"\n{model_name}:\n{report}")







       #########################################hyperparameter tunning for SMOTE and ROS ###############
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report, roc_curve, auc
# import matplotlib.pyplot as plt
# import numpy as np

# # Keras/TensorFlow kütüphaneleri (Neural Network için):
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# # label_binarize: Çoklu sınıf ROC çizimi için gerekli
# from sklearn.preprocessing import label_binarize

# # ==== Veri setleri ====
# datasets = [
#     ("Winsorized", winsorized_data),
#     ("Z-Score Filtered", zscore_cleaned_data),
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemleri (SMOTE veya ROS) ====
# oversampling_methods = {
#     "SMOTE": SMOTE(random_state=42),
#     "ROS": RandomOverSampler(random_state=42)
# }

# # Sonuçları saklamak için sözlük
# hyperparameter_results = {}

# # Neural Network (NN) hiperparametre kombinasyonları
# nn_params = {
#     "learning_rates": [0.001, 0.01],
#     "batch_sizes": [16, 32],
#     "epochs": [50, 100]
# }

# # GradientBoostingClassifier için parametre aralığı (örnek)
# gb_params = {
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5],
#     'n_estimators': [100, 200],
#     'subsample': [0.8, 1]
#     # Dilerseniz min_samples_split vb. de ekleyebilirsiniz
# }

# # ---- Fonksiyon: Macro-Average ROC hesaplama ----
# def compute_macro_roc(y_true_bin, y_prob):
#     """
#     Çoklu sınıfta her sınıfın ROC eğrisini hesaplayıp
#     bunları ortalayarak (macro-average) tek bir eğri olarak döndürür.
#     """
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)  # 0..1 arası 100 nokta
#     for i in range(y_true_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         # fpr_i -> tpr_i değerlerini mean_fpr üzerinde interpolate ediyoruz.
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc
# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 4) Çoklu sınıf ROC için test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)
#         y_test_bin = label_binarize(y_test, classes=unique_classes)

#         # ============================
#         # A) Random Forest
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         y_pred_rf = best_rf.predict(X_test)
#         y_pred_prob_rf = best_rf.predict_proba(X_test)
#         print("Best Random Forest Parameters:", rf_grid.best_params_)
#         print(classification_report(y_test, y_pred_rf))

#         # ============================
#         # B) XGBoost
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         y_pred_xgb = best_xgb.predict(X_test)
#         y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#         print("Best XGBoost Parameters:", xgb_grid.best_params_)
#         print(classification_report(y_test, y_pred_xgb))

#         # ============================
#         # C) Neural Network
#         # ============================
#         print("\nOptimizasyon: Neural Network")
#         best_nn_model = None
#         best_nn_acc = 0.0
#         best_nn_params = {}

#         for lr in nn_params["learning_rates"]:
#             for bs in nn_params["batch_sizes"]:
#                 for ep in nn_params["epochs"]:

#                 # ---- NN Model Oluşturma ----
#                 # (64 -> 32 -> sınıf_sayısı)
#                     nn_model = Sequential([
#                         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#                         Dense(32, activation='relu'),
#                         Dense(len(unique_classes), activation='softmax')  
#                     ])
#                     nn_model.compile(
#                         optimizer=Adam(learning_rate=lr),
#                         loss='sparse_categorical_crossentropy',
#                         metrics=['accuracy']
#                     )
#                     nn_model.fit(X_train, y_train, epochs=ep, batch_size=bs, verbose=0)
#                     eval_res = nn_model.evaluate(X_test, y_test, verbose=0)
#                     # eval_res[0] = loss, eval_res[1] = accuracy
#                     if eval_res[1] > best_nn_acc:
#                         best_nn_acc = eval_res[1]
#                         best_nn_params = {
#                             "learning_rate": lr,
#                             "batch_size": bs,
#                             "epochs": ep
#                         }
#                         best_nn_model = nn_model

#         print("Best Neural Network Parameters:", best_nn_params)
#         # NN tahminleri
#         y_pred_prob_nn = best_nn_model.predict(X_test)
#         y_pred_nn = np.argmax(y_pred_prob_nn, axis=1)

#         print(classification_report(y_test, y_pred_nn))

#         # ============================
#         # D) Gradient Boosting
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         y_pred_gb = best_gb.predict(X_test)
#         y_pred_prob_gb = best_gb.predict_proba(X_test)
#         print("Best Gradient Boosting Parameters:", gb_grid.best_params_)
#         print(classification_report(y_test, y_pred_gb))

#         # ============================
#         # ROC Eğrileri: 5 subplot
#         #  1) RF
#         #  2) XGB
#         #  3) NN
#         #  4) GB
#         #  5) Macro-Average (4 modeli tek ROC üzerinde)
#         # ============================
#         # ROC Eğrileri ve Macro-Average ROC (2 Satır x 3 Sütun Düzeni)
#         plt.figure(figsize=(24, 16))  # Büyük boyut

#         # ROC Subplots (2 Satır x 3 Sütun Düzeni)
#         for idx, (model_name, y_pred_probs) in enumerate([
#             ("RandomForest", y_pred_prob_rf),
#             ("XGBoost", y_pred_prob_xgb),
#             ("NeuralNetwork", y_pred_prob_nn),
#             ("GradientBoosting", y_pred_prob_gb)
#         ], start=1):
#             plt.subplot(2, 3, idx)  # 2 Satır x 3 Sütun
#             for class_label in unique_classes:
#                 class_idx = np.where(unique_classes == class_label)[0][0]
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_probs[:, class_idx])
#                 auc_val = auc(fpr, tpr)
#                 plt.plot(
#                     fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})", linewidth=2
#                 )
#             plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)  # Diyagonal çizgi
#             plt.title(f"{model_name} ROC\n({oversampling_name} + {dataset_name})", fontsize=16, weight='bold')  # Başlık bold
#             plt.xlabel("False Positive Rate", fontsize=14, weight='bold')  # Ekseni bold
#             plt.ylabel("True Positive Rate", fontsize=14, weight='bold')  # Ekseni bold
#             plt.legend(loc="lower right", fontsize=12, title="Classes", title_fontsize=14, frameon=True, prop={'size': 16,'weight': 'bold'})  # Legend ayarları

#         # Macro-Average ROC Curve (Alt Sağ Grafik)
#         plt.subplot(2, 3, 5)
#         for model_name, probs in {
#             "RandomForest": y_pred_prob_rf,
#             "XGBoost": y_pred_prob_xgb,
#             "NeuralNetwork": y_pred_prob_nn,
#             "GradientBoosting": y_pred_prob_gb
#         }.items():
#             mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probs)
#             plt.plot(
#                 mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})", linewidth=2.5
#             )
#         plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)  # Diyagonal çizgi
#         plt.title("Macro-Average ROC Curve", fontsize=16, weight='bold')  # Başlık bold
#         plt.xlabel("False Positive Rate", fontsize=14, weight='bold')  # Ekseni bold
#         plt.ylabel("True Positive Rate", fontsize=14, weight='bold')  # Ekseni bold
#         plt.legend(loc="lower right", fontsize=12, title="Models", title_fontsize=14, frameon=True, prop={'size': 16,'weight': 'bold'})  # Legend ayarları

#         # Genel Başlık
#         plt.suptitle(f"ROC Curves - {oversampling_name} + {dataset_name}", fontsize=22, weight='bold')  # Genel başlık yukarıda
#         plt.tight_layout(rect=[0, 0, 1, 0.92])  # Genel başlık ve alt yazılar için düzenleme
#         plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)  # Alt yazılar ve subplot arası alan
#         plt.show()


#         # Sonuçları saklama (Classification Report'lar, en iyi parametreler vb.)
#         hyperparameter_results[f"{oversampling_name} + {dataset_name}"] = {
#             "Random Forest Best Params": rf_grid.best_params_,
#             "Random Forest Report": classification_report(y_test, y_pred_rf, output_dict=True),
#             "XGBoost Best Params": xgb_grid.best_params_,
#             "XGBoost Report": classification_report(y_test, y_pred_xgb, output_dict=True),
#             "Neural Network Best Params": best_nn_params,
#             "Gradient Boosting Best Params": gb_grid.best_params_,
#             "Gradient Boosting Report": classification_report(y_test, y_pred_gb, output_dict=True)
#         }







##########hyper parameter tunning with wighted methods
####################################################################################
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report, roc_curve, auc
# from sklearn.model_selection import train_test_split, GridSearchCV

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize

# # Veri setleri: Winsorized, Z-Score Filtered, IQR Filtered
# datasets = [
#     ("Winsorized", winsorized_data),
#     ("Z-Score Filtered", zscore_cleaned_data),
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # Sonuçları saklamak için bir sözlük
# weighted_results = {}

# # Gradient Boosting parametre aralığı (örnek)
# gb_params = {
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5],
#     'n_estimators': [100, 200],
#     'subsample': [0.8, 1]
#     # Dilerseniz min_samples_split, min_samples_leaf vb. de ekleyebilirsiniz
# }

# # Ek: Macro-average ROC fonksiyonu (çoklu sınıfta One-vs-Rest)
# def compute_macro_roc(y_test_bin, y_prob):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_test_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc

# for dataset_name, dataset in datasets:
#     print(f"\n### Sınıf Ağırlıkları Kullanımı + GB (GridSearch) + ROC => {dataset_name} Dataset ###")

#     # 1) Özellikler ve hedef değişkeni ayırma
#     X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#     y = dataset['dver']
    
#     # 2) Eğitim ve test verisi ayırma
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # 3) Sınıf ağırlıklarını hesaplama (balanced)
#     classes_ = np.unique(y_train)
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=classes_,
#         y=y_train
#     )
#     # Sınıf -> ağırlık map'i (Keras'ta class_weight: {class_label: weight})
#     class_weights_dict = {}
#     for i, cls in enumerate(classes_):
#         class_weights_dict[cls] = class_weights[i]
    
#     print(f"Sınıf Ağırlıkları (dict): {class_weights_dict}")

#     # ROC hesaplaması için (çoklu sınıf ROC => binarize)
#     y_test_bin = label_binarize(y_test, classes=classes_)

#     # =============================================
#     # 1) Random Forest
#     # =============================================
#     print("\nModel: Random Forest (class_weight)")
#     rf = RandomForestClassifier(
#         class_weight=class_weights_dict,
#         random_state=42
#     )
#     rf.fit(X_train, y_train)
#     y_pred_rf = rf.predict(X_test)
#     print(f"Classification Report for Random Forest ({dataset_name}):")
#     print(classification_report(y_test, y_pred_rf))
    
#     # ROC proba
#     y_pred_prob_rf = rf.predict_proba(X_test)

#     # =============================================
#     # 2) XGBoost
#     # =============================================
#     print("\nModel: XGBoost (scale_pos_weight basit yaklaşım)")
#     xgb = XGBClassifier(
#         scale_pos_weight=1,  # Basit, genelde ikili senaryoda negative/positive ratio
#         use_label_encoder=False,
#         eval_metric='mlogloss',
#         random_state=42
#     )
#     xgb.fit(X_train, y_train)
#     y_pred_xgb = xgb.predict(X_test)
#     print(f"Classification Report for XGBoost ({dataset_name}):")
#     print(classification_report(y_test, y_pred_xgb))

#     # ROC proba
#     y_pred_prob_xgb = xgb.predict_proba(X_test)
    
#     # =============================================
#     # 3) Neural Network (Keras, class_weight)
#     # =============================================
#     print("\nModel: Neural Network (Keras) with class_weight")
#     model = Sequential([
#         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(classes_), activation='softmax')  # Çok sınıflı
#     ])
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=16,
#         verbose=0,
#         class_weight=class_weights_dict
#     )
#     nn_eval = model.evaluate(X_test, y_test, verbose=0)
#     y_pred_nn = np.argmax(model.predict(X_test), axis=1)
#     print(f"Neural Network Accuracy: {nn_eval[1]}")
#     print(f"Classification Report for Neural Network ({dataset_name}):")
#     print(classification_report(y_test, y_pred_nn))

#     # ROC proba
#     y_pred_prob_nn = model.predict(X_test)

#     # =============================================
#     # 4) Gradient Boosting (GridSearchCV)
#     # =============================================
#     print("\nModel: Gradient Boosting (GridSearchCV)")
#     gb_params = {
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5],
#         'n_estimators': [100, 200],
#         'subsample': [0.8, 1]
#     }
#     gb_grid = GridSearchCV(
#         estimator=GradientBoostingClassifier(random_state=42),
#         param_grid=gb_params,
#         cv=3,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     print("Best Gradient Boosting Parameters:", gb_grid.best_params_)
#     y_pred_gb = best_gb.predict(X_test)
#     print(f"Classification Report for Gradient Boosting ({dataset_name}):")
#     print(classification_report(y_test, y_pred_gb))

#     # ROC proba
#     y_pred_prob_gb = best_gb.predict_proba(X_test)

#     # =============================================
#     # ROC Çizimi (5 Subplot: RF, XGB, NN, GB, macro-average)
#     # =============================================
#     import matplotlib.pyplot as plt

# # =============================================
# # ROC Curve Plotting (2 Rows × 3 Columns)
# # =============================================
#     plt.figure(figsize=(18, 12))  # Adjust the figure size

#     # Define model names for plotting
#     model_names = ["Random Forest", "XGBoost", "Neural Network", "Gradient Boosting", "Macro-Average ROC"]

#     # Define the probabilities for each model
#     models_with_probs = {
#         "RandomForest": y_pred_prob_rf,
#         "XGBoost": y_pred_prob_xgb,
#         "NeuralNetwork": y_pred_prob_nn,
#         "GradientBoosting": y_pred_prob_gb
#     }

#     # ==== Subplots 1-4: Individual Model ROC Curves ====
#     for idx, (model_name, y_probs) in enumerate(models_with_probs.items(), start=1):
#         plt.subplot(2, 3, idx)  # 2 rows, 3 columns

#         for class_label in classes_:
#             idx_class = np.where(classes_ == class_label)[0][0]
#             fpr, tpr, _ = roc_curve(y_test_bin[:, idx_class], y_probs[:, idx_class])
#             auc_value = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label=f"Class {class_label} (AUC={auc_value:.2f})", linewidth=2)

#         plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
#         plt.title(f"**{model_names[idx-1]}**", fontsize=15, fontweight='bold')
#         plt.xlabel("**False Positive Rate**", fontsize=14, fontweight='bold')
#         plt.ylabel("**True Positive Rate**", fontsize=14, fontweight='bold')
#         plt.legend(loc="lower right", fontsize=20, prop={'size': 14,'weight': 'bold'})  # Increase legend size

#     # ==== Subplot 5: Macro-Average ROC (All Models in One) ====
#     plt.subplot(2, 3, 5)  # Position for macro-average ROC

#     for model_name, y_probs in models_with_probs.items():
#         mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, y_probs)
#         plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})", linewidth=2)

#     plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
#     plt.title("**Macro-Average ROC (All Models)**", fontsize=15, fontweight='bold')
#     plt.xlabel("**False Positive Rate**", fontsize=14, fontweight='bold')
#     plt.ylabel("**True Positive Rate**", fontsize=14, fontweight='bold')
#     plt.legend(loc="lower right", fontsize=14, prop={'size': 14,'weight': 'bold'})  # Increase legend size

#     # Adjust layout and show the figure
#     plt.suptitle("**ROC Curves for Cost Sensitive**", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()


#     # =============================================
#     # Sonuçları saklama
#     # =============================================
#     weighted_results[dataset_name] = {
#         "Random Forest Report": classification_report(y_test, y_pred_rf, output_dict=True),
#         "XGBoost Report": classification_report(y_test, y_pred_xgb, output_dict=True),
#         "Neural Network Accuracy": nn_eval[1],
#         "Neural Network Report": classification_report(y_test, y_pred_nn, output_dict=True),
#         "Gradient Boosting Best Params": gb_grid.best_params_,
#         "Gradient Boosting Report": classification_report(y_test, y_pred_gb, output_dict=True)
#     }






##########################################tümü birarada###################### Threshold Tuning Eklenmiş Versiyon#############
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, roc_curve, auc, f1_score
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from sklearn.preprocessing import label_binarize

# # Veri kümesi listesi: Winsorized, Z-Score Filtered, IQR Filtered
# datasets = {
#     "Winsorized Data": winsorized_data,
#     "Z-Score Filtered Data": zscore_cleaned_data,
#     "IQR Filtered Data": iqr_cleaned_data
# }

# # Her veri kümesi üzerinde işlemler
# for dataset_name, dataset in datasets.items():
#     print(f"\n=== {dataset_name} ===")
    
#     # Bağımsız ve bağımlı değişkenleri ayırma
#     X = dataset.drop("dver", axis=1)
#     y = dataset["dver"]
    
#     # 1) SMOTE ile Veri Dengeleme
#     smote = SMOTE(random_state=42)
#     X_res, y_res = smote.fit_resample(X, y)

#     # 2) Eğitim ve Test Verisi Ayrımı
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
#     )

#     # 3) Sınıf Ağırlıkları Hesaplama
#     classes_ = np.unique(y_train)
#     cw_values = compute_class_weight(class_weight='balanced', classes=classes_, y=y_train)
#     class_weight_dict = {cls: w for cls, w in zip(classes_, cw_values)}

#     # 4) Modellerin Eğitimi
#     # A) RandomForest
#     rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
#     rf_clf.fit(X_train, y_train)

#     # B) XGBoost
#     xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight=1)
#     xgb_clf.fit(X_train, y_train)

#     # C) Neural Network
#     nn_model = Sequential([
#         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(classes_), activation='softmax')
#     ])
#     nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, class_weight=class_weight_dict)

#     # D) Gradient Boosting
#     gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#     gb_clf.fit(X_train, y_train)

#     # E) Ensemble (VotingClassifier)
#     ensemble_clf = VotingClassifier(
#         estimators=[("rf", rf_clf), ("xgb", xgb_clf), ("gb", gb_clf)],
#         voting="soft"
#     )
#     ensemble_clf.fit(X_train, y_train)

#     # 5) ROC Eğrileri ve Tahmin Olasılıkları
#     rf_probs = rf_clf.predict_proba(X_test)
#     xgb_probs = xgb_clf.predict_proba(X_test)
#     nn_probs = nn_model.predict(X_test)
#     gb_probs = gb_clf.predict_proba(X_test)
#     ens_probs = ensemble_clf.predict_proba(X_test)

#     y_test_bin = label_binarize(y_test, classes=classes_)

#     # 6) Threshold Tuning
#     models_with_probs = {
#         "RandomForest": rf_probs,
#         "XGBoost": xgb_probs,
#         "GradientBoosting": gb_probs,
#         "Ensemble": ens_probs
#     }

#     optimal_thresholds = {}

#     print("\n=== Threshold Tuning Sonuçları ===")
#     for model_name, probs in models_with_probs.items():
#         thresholds = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
#         best_threshold = 0.5
#         best_f1 = 0

#         for threshold in thresholds:
#             preds = np.argmax(probs >= threshold, axis=1)
#             f1 = f1_score(y_test, preds, average='macro')  # Makro F1-Score
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         # En iyi threshold ve F1-score'u kaydet
#         optimal_thresholds[model_name] = (best_threshold, best_f1)
#         print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best F1-Score = {best_f1:.3f}")

#     # Nihai Tahminler ve Raporlama
#     print("\n=== Nihai Tahminler ve Classification Report ===")
#     for model_name, (threshold, _) in optimal_thresholds.items():
#         probs = models_with_probs[model_name]
#         final_preds = np.argmax(probs >= threshold, axis=1)
#         print(f"\n{model_name} (Threshold = {threshold:.2f}):")
#         print(classification_report(y_test, final_preds))

#     # 7) Macro-Average ROC Eğrileri
#     plt.figure(figsize=(18, 12))

#     def plot_roc_with_auc(probs, model_name, subplot_index):
#         plt.subplot(2, 3, subplot_index)
#         for cls in classes_:
#             idx = np.where(classes_ == cls)[0][0]
#             fpr, tpr, _ = roc_curve(y_test_bin[:, idx], probs[:, idx])
#             auc_val = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_val:.2f})")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title(model_name)
#         plt.xlabel("FPR")
#         plt.ylabel("TPR")
#         plt.legend(loc="lower right")
    
#     plot_roc_with_auc(rf_probs, "RandomForest", 1)
#     plot_roc_with_auc(xgb_probs, "XGBoost", 2)
#     plot_roc_with_auc(nn_probs, "NeuralNet", 3)
#     plot_roc_with_auc(gb_probs, "GradientBoosting", 4)
#     plot_roc_with_auc(ens_probs, "Ensemble", 5)

#     # Macro-Average ROC
#     plt.subplot(2, 3, 6)
#     for model_name, probs in models_with_probs.items():
#         tprs, aucs = [], []
#         mean_fpr = np.linspace(0, 1, 100)
#         for i in range(y_test_bin.shape[1]):
#             fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
#             aucs.append(auc(fpr, tpr))
#             interp_tpr = np.interp(mean_fpr, fpr, tpr)
#             tprs.append(interp_tpr)
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         macro_auc = np.mean(aucs)
#         plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
    
#     plt.plot([0, 1], [0, 1], '--', color='gray')
#     plt.title("Macro-Average ROC")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend(loc="lower right")
#     plt.suptitle(f"ROC Curves for {dataset_name}", fontsize=16)
#     plt.tight_layout()
#     plt.show()

##########################################kırmızı işaretli threshold optimization SMOTE ile#############

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, roc_curve, auc, f1_score
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from sklearn.preprocessing import label_binarize

# # Veri kümesi listesi: Winsorized, Z-Score Filtered, IQR Filtered
# datasets = {
#     "Winsorized Data": winsorized_data,
#     "Z-Score Filtered Data": zscore_cleaned_data,
#     "IQR Filtered Data": iqr_cleaned_data
# }

# # Her veri kümesi üzerinde işlemler
# for dataset_name, dataset in datasets.items():
#     print(f"\n=== {dataset_name} ===")
    
#     # Bağımsız ve bağımlı değişkenleri ayırma
#     X = dataset.drop("dver", axis=1)
#     y = dataset["dver"]
    
#     # 1) SMOTE ile Veri Dengeleme
#     smote = SMOTE(random_state=42)
#     X_res, y_res = smote.fit_resample(X, y)

#     # 2) Eğitim ve Test Verisi Ayrımı
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
#     )

#     # 3) Sınıf Ağırlıkları Hesaplama
#     classes_ = np.unique(y_train)
#     cw_values = compute_class_weight(class_weight='balanced', classes=classes_, y=y_train)
#     class_weight_dict = {cls: w for cls, w in zip(classes_, cw_values)}

#     # 4) Modellerin Eğitimi
#     # A) RandomForest
#     rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
#     rf_clf.fit(X_train, y_train)

#     # B) XGBoost
#     xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight=1)
#     xgb_clf.fit(X_train, y_train)

#     # C) Neural Network
#     nn_model = Sequential([
#         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(classes_), activation='softmax')
#     ])
#     nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, class_weight=class_weight_dict)

#     # D) Gradient Boosting
#     gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#     gb_clf.fit(X_train, y_train)

#     # E) Ensemble (VotingClassifier)
#     ensemble_clf = VotingClassifier(
#         estimators=[("rf", rf_clf), ("xgb", xgb_clf), ("gb", gb_clf)],
#         voting="soft"
#     )
#     ensemble_clf.fit(X_train, y_train)

#     # 5) Tahmin Olasılıkları
#     rf_probs = rf_clf.predict_proba(X_test)
#     xgb_probs = xgb_clf.predict_proba(X_test)
#     nn_probs = nn_model.predict(X_test)
#     gb_probs = gb_clf.predict_proba(X_test)
#     ens_probs = ensemble_clf.predict_proba(X_test)

#     y_test_bin = label_binarize(y_test, classes=classes_)

#     # 6) Threshold Tuning
#     models_with_probs = {
#         "RandomForest": rf_probs,
#         "XGBoost": xgb_probs,
#         "GradientBoosting": gb_probs,
#         "Ensemble": ens_probs
#     }

#     optimal_thresholds = {}

#     print("\n=== Threshold Tuning Sonuçları ===")
#     for model_name, probs in models_with_probs.items():
#         thresholds = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
#         best_threshold = 0.5
#         best_f1 = 0

#         for threshold in thresholds:
#             preds = np.argmax(probs >= threshold, axis=1)
#             f1 = f1_score(y_test, preds, average='macro')
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         # En iyi threshold ve F1-score'u tuple olarak kaydet
#         optimal_thresholds[model_name] = (best_threshold, best_f1)
#         print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best F1-Score = {best_f1:.3f}")

#     # Nihai Tahminler ve Raporlama
#     print("\n=== Nihai Tahminler ve Classification Report ===")
#     for model_name, (threshold, _) in optimal_thresholds.items():
#         probs = models_with_probs[model_name]
#         final_preds = np.argmax(probs >= threshold, axis=1)
#         print(f"\n{model_name} (Threshold = {threshold:.2f}):")
#         print(classification_report(y_test, final_preds))

#     # 7) ROC Eğrileri ve En İyi Threshold İşaretleme
#     plt.figure(figsize=(18, 12))

#     def plot_roc_with_best_threshold(probs, model_name, best_threshold, subplot_index):
#         plt.subplot(2, 3, subplot_index)
#         for cls in classes_:
#             idx = np.where(classes_ == cls)[0][0]
#             fpr, tpr, thresholds = roc_curve(y_test_bin[:, idx], probs[:, idx])
#             auc_val = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_val:.2f})")
#             # En iyi threshold'u işaretle
#             best_idx = np.argmin(np.abs(thresholds - best_threshold))
#             plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label=f"Best Thresh: {best_threshold:.2f}")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title(model_name)
#         plt.xlabel("FPR")
#         plt.ylabel("TPR")
#         plt.legend(loc="lower right")
    
#     plot_roc_with_best_threshold(rf_probs, "RandomForest", optimal_thresholds["RandomForest"][0], 1)
#     plot_roc_with_best_threshold(xgb_probs, "XGBoost", optimal_thresholds["XGBoost"][0], 2)
#     plot_roc_with_best_threshold(nn_probs, "NeuralNet", 0.5, 3)  # NeuralNet için threshold tuning yapılmadı
#     plot_roc_with_best_threshold(gb_probs, "GradientBoosting", optimal_thresholds["GradientBoosting"][0], 4)
#     plot_roc_with_best_threshold(ens_probs, "Ensemble", optimal_thresholds["Ensemble"][0], 5)

#     # Macro-Average ROC
#     plt.subplot(2, 3, 6)
#     for model_name, probs in models_with_probs.items():
#         tprs, aucs = [], []
#         mean_fpr = np.linspace(0, 1, 100)
#         for i in range(y_test_bin.shape[1]):
#             fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
#             aucs.append(auc(fpr, tpr))
#             interp_tpr = np.interp(mean_fpr, fpr, tpr)
#             tprs.append(interp_tpr)
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         macro_auc = np.mean(aucs)
#         plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
    
#     plt.plot([0, 1], [0, 1], '--', color='gray')
#     plt.title("Macro-Average ROC")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend(loc="lower right")
#     plt.suptitle(f"ROC Curves for {dataset_name} (with Thresholds)", fontsize=16)
#     plt.tight_layout()
#     plt.show()

# ##########################################kırmızı işaretli threshold optimization ROS ile+ HYPERPARAMETRE TUNNİNG#############
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, roc_curve, auc, f1_score
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from sklearn.preprocessing import label_binarize

# # Veri kümesi listesi: Winsorized, Z-Score Filtered, IQR Filtered
# datasets = {
#     "Winsorized Data": winsorized_data,
#     "Z-Score Filtered Data": zscore_cleaned_data,
#     "IQR Filtered Data": iqr_cleaned_data
# }

# # Her veri kümesi üzerinde işlemler
# for dataset_name, dataset in datasets.items():
#     print(f"\n=== {dataset_name} ===")
    
#     # Bağımsız ve bağımlı değişkenleri ayırma
#     X = dataset.drop("dver", axis=1)
#     y = dataset["dver"]
    
#     # 1) ROS ile Veri Dengeleme
#     ros = RandomOverSampler(random_state=42)
#     X_res, y_res = ros.fit_resample(X, y)

#     # 2) Eğitim ve Test Verisi Ayrımı
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
#     )

#     # 3) Sınıf Ağırlıkları Hesaplama
#     classes_ = np.unique(y_train)
#     cw_values = compute_class_weight(class_weight='balanced', classes=classes_, y=y_train)
#     class_weight_dict = {cls: w for cls, w in zip(classes_, cw_values)}

#     # 4) Hyperparameter Tuning ve Model Eğitimi
#     # A) RandomForest
#     rf_params = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     rf_grid = GridSearchCV(
#         RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
#         param_grid=rf_params,
#         cv=3,
#         scoring='f1_macro',
#         n_jobs=-1
#     )
#     rf_grid.fit(X_train, y_train)
#     best_rf = rf_grid.best_estimator_
#     print(f"Best RandomForest Params: {rf_grid.best_params_}")

#     # B) XGBoost
#     xgb_params = {
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'n_estimators': [50, 100, 200],
#         'subsample': [0.8, 1],
#         'colsample_bytree': [0.8, 1]
#     }
#     xgb_grid = GridSearchCV(
#         XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#         param_grid=xgb_params,
#         cv=3,
#         scoring='f1_macro',
#         n_jobs=-1
#     )
#     xgb_grid.fit(X_train, y_train)
#     best_xgb = xgb_grid.best_estimator_
#     print(f"Best XGBoost Params: {xgb_grid.best_params_}")

#     # C) Gradient Boosting
#     gb_params = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 1]
#     }
#     gb_grid = GridSearchCV(
#         GradientBoostingClassifier(random_state=42),
#         param_grid=gb_params,
#         cv=3,
#         scoring='f1_macro',
#         n_jobs=-1
#     )
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     print(f"Best GradientBoosting Params: {gb_grid.best_params_}")

#     # D) Neural Network
#     nn_model = Sequential([
#         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(classes_), activation='softmax')
#     ])
#     nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, class_weight=class_weight_dict)

#     # E) Ensemble (VotingClassifier)
#     ensemble_clf = VotingClassifier(
#         estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#         voting="soft"
#     )
#     ensemble_clf.fit(X_train, y_train)

#     # 5) Tahmin Olasılıkları
#     rf_probs = best_rf.predict_proba(X_test)
#     xgb_probs = best_xgb.predict_proba(X_test)
#     nn_probs = nn_model.predict(X_test)
#     gb_probs = best_gb.predict_proba(X_test)
#     ens_probs = ensemble_clf.predict_proba(X_test)

#     y_test_bin = label_binarize(y_test, classes=classes_)

#     # Threshold Tuning
#     optimal_thresholds = {}
#     models_with_probs = {
#         "RandomForest": rf_probs,
#         "XGBoost": xgb_probs,
#         "NeuralNet": nn_probs,
#         "GradientBoosting": gb_probs,
#         "Ensemble": ens_probs
#     }
#     for model_name, probs in models_with_probs.items():
#         thresholds = np.linspace(0.1, 0.9, 9)
#         best_threshold = 0.5
#         best_f1 = 0

#         for threshold in thresholds:
#             preds = np.argmax(probs >= threshold, axis=1)
#             f1 = f1_score(y_test, preds, average='macro')
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         optimal_thresholds[model_name] = (best_threshold, best_f1)
#         print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best F1-Score = {best_f1:.3f}")

#     # ROC Eğrileri ve Macro-Average ROC
#     plt.figure(figsize=(18, 12))

#     def plot_roc_with_best_threshold(probs, model_name, best_threshold, subplot_index):
#         plt.subplot(2, 3, subplot_index)
#         for cls in classes_:
#             idx = np.where(classes_ == cls)[0][0]
#             fpr, tpr, thresholds = roc_curve(y_test_bin[:, idx], probs[:, idx])
#             auc_val = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_val:.2f})")
#             # En iyi threshold'u işaretle
#             best_idx = np.argmin(np.abs(thresholds - best_threshold))
#             plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label=f"Best Thresh: {best_threshold:.2f}")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title(model_name)
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.legend(loc="lower right")

#     # ROC Eğrileri
#     plot_roc_with_best_threshold(rf_probs, "RandomForest", optimal_thresholds["RandomForest"][0], 1)
#     plot_roc_with_best_threshold(xgb_probs, "XGBoost", optimal_thresholds["XGBoost"][0], 2)
#     plot_roc_with_best_threshold(nn_probs, "NeuralNet", 0.5, 3)
#     plot_roc_with_best_threshold(gb_probs, "GradientBoosting", optimal_thresholds["GradientBoosting"][0], 4)
#     plot_roc_with_best_threshold(ens_probs, "Ensemble", optimal_thresholds["Ensemble"][0], 5)

#     # Macro-Average ROC
#     plt.subplot(2, 3, 6)
#     mean_fpr = np.linspace(0, 1, 100)
#     for model_name, probs in models_with_probs.items():
#         tprs, aucs = [], []
#         for i in range(y_test_bin.shape[1]):
#             fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], probs[:, i])
#             interp_tpr = np.interp(mean_fpr, fpr, tpr)
#             tprs.append(interp_tpr)
#             auc_val = auc(fpr, tpr)
#             aucs.append(auc_val)
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         macro_auc = np.mean(aucs)
#         plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
#     plt.plot([0, 1], [0, 1], '--', color='gray')
#     plt.title("Macro-Average ROC Curve")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.show()




#############################recalll odaklı yani thresholdun düşük olduğu durumlar ROS#############

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, roc_curve, auc, recall_score
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from sklearn.preprocessing import label_binarize

# # Veri kümesi listesi: Winsorized, Z-Score Filtered, IQR Filtered
# datasets = {
#     "Winsorized Data": winsorized_data,
#     "Z-Score Filtered Data": zscore_cleaned_data,
#     "IQR Filtered Data": iqr_cleaned_data
# }

# # Her veri kümesi üzerinde işlemler
# for dataset_name, dataset in datasets.items():
#     print(f"\n=== {dataset_name} ===")
    
#     # Bağımsız ve bağımlı değişkenleri ayırma
#     X = dataset.drop("dver", axis=1)
#     y = dataset["dver"]
    
#     # 1) ROS ile Veri Dengeleme
#     ros = RandomOverSampler(random_state=42)
#     X_res, y_res = ros.fit_resample(X, y)

#     # 2) Eğitim ve Test Verisi Ayrımı
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
#     )

#     # 3) Sınıf Ağırlıkları Hesaplama
#     classes_ = np.unique(y_train)
#     cw_values = compute_class_weight(class_weight='balanced', classes=classes_, y=y_train)
#     class_weight_dict = {cls: w for cls, w in zip(classes_, cw_values)}

#     # 4) Hyperparameter Tuning ve Model Eğitimi
#     # A) RandomForest
#     rf_params = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     rf_grid = GridSearchCV(
#         RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
#         param_grid=rf_params,
#         cv=3,
#         scoring='recall_macro',
#         n_jobs=-1
#     )
#     rf_grid.fit(X_train, y_train)
#     best_rf = rf_grid.best_estimator_
#     print(f"Best RandomForest Params: {rf_grid.best_params_}")

#     # B) XGBoost
#     xgb_params = {
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'n_estimators': [50, 100, 200],
#         'subsample': [0.8, 1],
#         'colsample_bytree': [0.8, 1]
#     }
#     xgb_grid = GridSearchCV(
#         XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#         param_grid=xgb_params,
#         cv=3,
#         scoring='recall_macro',
#         n_jobs=-1
#     )
#     xgb_grid.fit(X_train, y_train)
#     best_xgb = xgb_grid.best_estimator_
#     print(f"Best XGBoost Params: {xgb_grid.best_params_}")

#     # C) Gradient Boosting
#     gb_params = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 1]
#     }
#     gb_grid = GridSearchCV(
#         GradientBoostingClassifier(random_state=42),
#         param_grid=gb_params,
#         cv=3,
#         scoring='recall_macro',
#         n_jobs=-1
#     )
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     print(f"Best GradientBoosting Params: {gb_grid.best_params_}")

#     # D) Neural Network
#     nn_model = Sequential([
#         Dense(64, input_dim=X_train.shape[1], activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(classes_), activation='softmax')
#     ])
#     nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, class_weight=class_weight_dict)

#     # E) Ensemble Hyperparameter Tuning
#     ensemble_weights = {
#         'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]  # Ağırlık kombinasyonları
#     }
#     ensemble_grid = GridSearchCV(
#         VotingClassifier(estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)], voting="soft"),
#         param_grid=ensemble_weights,
#         cv=3,
#         scoring='recall_macro',
#         n_jobs=-1
#     )
#     ensemble_grid.fit(X_train, y_train)
#     best_ensemble = ensemble_grid.best_estimator_
#     print(f"Best Ensemble Weights: {ensemble_grid.best_params_}")

#     # Tahmin Olasılıkları
#     rf_probs = best_rf.predict_proba(X_test)
#     xgb_probs = best_xgb.predict_proba(X_test)
#     nn_probs = nn_model.predict(X_test)
#     gb_probs = best_gb.predict_proba(X_test)
#     ens_probs = best_ensemble.predict_proba(X_test)

#     y_test_bin = label_binarize(y_test, classes=classes_)

#     # 5) Recall Odaklı Threshold Tuning
#     models_with_probs = {
#         "RandomForest": rf_probs,
#         "XGBoost": xgb_probs,
#         "GradientBoosting": gb_probs,
#         "Ensemble": ens_probs
#     }

#     optimal_thresholds = {}

#     print("\n=== Recall Odaklı Threshold Tuning Sonuçları ===")
#     for model_name, probs in models_with_probs.items():
#         thresholds = np.linspace(0.1, 0.5, 9)  # 0.1, 0.2, ..., 0.5 (düşük threshold aralığı)
#         best_threshold = 0.5
#         best_recall = 0

#         for threshold in thresholds:
#             preds = np.argmax(probs >= threshold, axis=1)
#             recall = recall_score(y_test, preds, average='macro')
#             if recall > best_recall:
#                 best_recall = recall
#                 best_threshold = threshold

#         # En iyi threshold ve Recall'u tuple olarak kaydet
#         optimal_thresholds[model_name] = (best_threshold, best_recall)
#         print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best Recall = {best_recall:.3f}")

#     # Nihai Tahminler ve Raporlama
#     print("\n=== Nihai Tahminler ve Recall Odaklı Classification Report ===")
#     for model_name, (threshold, _) in optimal_thresholds.items():
#         probs = models_with_probs[model_name]
#         final_preds = np.argmax(probs >= threshold, axis=1)
#         print(f"\n{model_name} (Threshold = {threshold:.2f}):")
#         print(classification_report(y_test, final_preds))

#     # ROC Eğrileri ve Macro-Average ROC
#     plt.figure(figsize=(18, 12))

#     def plot_roc_with_best_threshold(probs, model_name, best_threshold, subplot_index):
#         plt.subplot(2, 3, subplot_index)
#         for cls in classes_:
#             idx = np.where(classes_ == cls)[0][0]
#             fpr, tpr, thresholds = roc_curve(y_test_bin[:, idx], probs[:, idx])
#             auc_val = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_val:.2f})")
#             # En iyi threshold'u işaretle
#             best_idx = np.argmin(np.abs(thresholds - best_threshold))
#             plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label=f"Best Thresh: {best_threshold:.2f}")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title(model_name)
#         plt.xlabel("FPR")
#         plt.ylabel("TPR")
#         plt.legend(loc="lower right")

#     # ROC Eğrileri
#     plot_roc_with_best_threshold(rf_probs, "RandomForest", optimal_thresholds["RandomForest"][0], 1)
#     plot_roc_with_best_threshold(xgb_probs, "XGBoost", optimal_thresholds["XGBoost"][0], 2)
#     plot_roc_with_best_threshold(nn_probs, "NeuralNet", 0.5, 3)  # NeuralNet için threshold tuning yapılmadı
#     plot_roc_with_best_threshold(gb_probs, "GradientBoosting", optimal_thresholds["GradientBoosting"][0], 4)
#     plot_roc_with_best_threshold(ens_probs, "Ensemble", optimal_thresholds["Ensemble"][0], 5)

#     # Macro-Average ROC
#     plt.subplot(2, 3, 6)
#     mean_fpr = np.linspace(0, 1, 100)
#     for model_name, probs in models_with_probs.items():
#         tprs, aucs = [], []
#         for i in range(y_test_bin.shape[1]):
#             fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
#             aucs.append(auc(fpr, tpr))
#             interp_tpr = np.interp(mean_fpr, fpr, tpr)
#             tprs.append(interp_tpr)
#         mean_tpr = np.mean(tprs, axis=0)
#         mean_tpr[-1] = 1.0
#         macro_auc = np.mean(aucs)
#         plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
#     plt.plot([0, 1], [0, 1], '--', color='gray')
#     plt.title("Macro-Average ROC")
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.show()





#############################ROS+HYPERPARAMETRETUNNİNG##############
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import label_binarize

# # ==== Veri seti ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemi (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # Sonuçları saklamak için sözlük
# hyperparameter_results = {}

# # ---- Fonksiyon: Macro-Average ROC hesaplama ----
# def compute_macro_roc(y_true_bin, y_prob):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)  # 0..1 arası 100 nokta
#     for i in range(y_true_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc
# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 4) Çoklu sınıf ROC için test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)
#         y_test_bin = label_binarize(y_test, classes=unique_classes)

#         # ============================
#         # A) Random Forest
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         y_pred_rf = best_rf.predict(X_test)
#         y_pred_prob_rf = best_rf.predict_proba(X_test)
#         print("Best Random Forest Parameters:", rf_grid.best_params_)
#         print(classification_report(y_test, y_pred_rf))

#         # ============================
#         # B) XGBoost
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         y_pred_xgb = best_xgb.predict(X_test)
#         y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#         print("Best XGBoost Parameters:", xgb_grid.best_params_)
#         print(classification_report(y_test, y_pred_xgb))

#         # ============================
#         # C) Gradient Boosting
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_params = {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1]
#         }
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         y_pred_gb = best_gb.predict(X_test)
#         y_pred_prob_gb = best_gb.predict_proba(X_test)
#         print("Best Gradient Boosting Parameters:", gb_grid.best_params_)
#         print(classification_report(y_test, y_pred_gb))

#         # ============================
#         # D) Ensemble
#         # ============================
#         print("\nOptimizasyon: Ensemble")
#         ensemble_clf = VotingClassifier(
#             estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         y_pred_ensemble = ensemble_clf.predict(X_test)
#         y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)
#         print(classification_report(y_test, y_pred_ensemble))

#         # ============================
#         # ROC Eğrileri
#         # ============================
#         plt.figure(figsize=(24, 6))

#         # ROC Subplots
#         models = {
#             "Random Forest": y_pred_prob_rf,
#             "XGBoost": y_pred_prob_xgb,
#             "Gradient Boosting": y_pred_prob_gb,
#             "Ensemble": y_pred_prob_ensemble
#         }

#         for idx, (model_name, probs) in enumerate(models.items(), start=1):
#             plt.subplot(1, 5, idx)
#             for class_label in unique_classes:
#                 class_idx = np.where(unique_classes == class_label)[0][0]
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
#                 auc_val = auc(fpr, tpr)
#                 plt.plot(fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})")
#             plt.plot([0, 1], [0, 1], '--', color='gray')
#             plt.title(f"{model_name}\n({oversampling_name}+{dataset_name})")
#             plt.xlabel("FPR")
#             plt.ylabel("TPR")
#             plt.legend(loc="lower right")

#         # Macro-Average ROC Curve
#         plt.subplot(1, 5, 5)
#         for model_name, probs in models.items():
#             mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probs)
#             plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title("Macro-Average ROC")
#         plt.xlabel("FPR")
#         plt.ylabel("TPR")
#         plt.legend(loc="lower right")

#         plt.suptitle(f"ROC Curves - {oversampling_name} + {dataset_name}", fontsize=16)
#         plt.tight_layout()
#         plt.show()

        #################################################ROS+HYPERPARAMETRETUNNİNG+threshold TUNNİNG##################

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc, f1_score
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import label_binarize

# # ==== Veri seti ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemi (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ---- Fonksiyon: Macro-Average ROC hesaplama ----
# def compute_macro_roc(y_true_bin, y_prob):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_true_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc
# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 4) Çoklu sınıf ROC için test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)
#         y_test_bin = label_binarize(y_test, classes=unique_classes)

#         # ============================
#         # A) Random Forest
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         y_pred_prob_rf = best_rf.predict_proba(X_test)
#         print("Best Random Forest Parameters:", rf_grid.best_params_)

#         # ============================
#         # B) XGBoost
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#         print("Best XGBoost Parameters:", xgb_grid.best_params_)

#         # ============================
#         # C) Gradient Boosting
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_params = {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1]
#         }
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=3,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         y_pred_prob_gb = best_gb.predict_proba(X_test)
#         print("Best Gradient Boosting Parameters:", gb_grid.best_params_)

#         # ============================
#         # D) Ensemble
#         # ============================
#         print("\nOptimizasyon: Ensemble")
#         ensemble_clf = VotingClassifier(
#             estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)

#         # ============================
#         # Threshold Tuning
#         # ============================
#         models_with_probs = {
#             "RandomForest": y_pred_prob_rf,
#             "XGBoost": y_pred_prob_xgb,
#             "GradientBoosting": y_pred_prob_gb,
#             "Ensemble": y_pred_prob_ensemble
#         }

#         optimal_thresholds = {}
#         for model_name, probs in models_with_probs.items():
#             thresholds = np.linspace(0.1, 0.9, 9)
#             best_threshold = 0.5
#             best_f1 = 0

#             for threshold in thresholds:
#                 preds = np.argmax(probs >= threshold, axis=1)
#                 f1 = f1_score(y_test, preds, average='macro')
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_threshold = threshold

#             optimal_thresholds[model_name] = (best_threshold, best_f1)
#             print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best F1-Score = {best_f1:.3f}")

#         # Nihai Classification Reports
#         print("\n=== Classification Reports ===")
#         for model_name, probs in models_with_probs.items():
#             threshold = optimal_thresholds[model_name][0]
#             preds = np.argmax(probs >= threshold, axis=1)
#             print(f"\nClassification Report for {model_name} (Threshold = {threshold:.2f}):")
#             print(classification_report(y_test, preds))

#         # ============================
#         # ROC Eğrileri ve Macro-Average ROC
#         # ============================
#         plt.figure(figsize=(24, 6))

#         # ROC Subplots
#         for idx, (model_name, probs) in enumerate(models_with_probs.items(), start=1):
#             plt.subplot(1, 5, idx)
#             for class_label in unique_classes:
#                 class_idx = np.where(unique_classes == class_label)[0][0]
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
#                 auc_val = auc(fpr, tpr)
#                 plt.plot(fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})")
#             plt.plot([0, 1], [0, 1], '--', color='gray')
#             plt.title(f"{model_name}\n({oversampling_name}+{dataset_name})")
#             plt.xlabel("FPR")
#             plt.ylabel("TPR")
#             plt.legend(loc="lower right")

#         # Macro-Average ROC Curve
#         plt.subplot(1, 5, 5)
#         for model_name, probs in models_with_probs.items():
#             mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probs)
#             plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title("Macro-Average ROC")
#         plt.xlabel("FPR")
#         plt.ylabel("TPR")
#         plt.legend(loc="lower right")

#         plt.suptitle(f"ROC Curves - {oversampling_name} + {dataset_name}", fontsize=16)
#         plt.tight_layout()
#         plt.show()


##################################################1.MODEL TRAINING PHASE WITH (ROS+HYPERPARAMETRETUNNİNG+threshold TUNNİNG (0.5))#####################

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc, recall_score
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import label_binarize

# # ==== Veri seti ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemi (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ---- Fonksiyon: Macro-Average ROC hesaplama ----
# def compute_macro_roc(y_true_bin, y_prob):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_true_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc
# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 4) Çoklu sınıf ROC için test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)
#         y_test_bin = label_binarize(y_test, classes=unique_classes)

#         # ============================
#         # A) Random Forest
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=3,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         y_pred_prob_rf = best_rf.predict_proba(X_test)
#         print("Best Random Forest Parameters:", rf_grid.best_params_)

#         # ============================
#         # B) XGBoost
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=3,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#         print("Best XGBoost Parameters:", xgb_grid.best_params_)

#         # ============================
#         # C) Gradient Boosting
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_params = {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1]
#         }
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=3,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         y_pred_prob_gb = best_gb.predict_proba(X_test)
#         print("Best Gradient Boosting Parameters:", gb_grid.best_params_)

#         # ============================
#         # D) Ensemble
#         # ============================
#         print("\nOptimizasyon: Ensemble")
#         ensemble_clf = VotingClassifier(
#             estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)

#         # ============================
#         # Recall Odaklı Threshold Tuning
#         # ============================
#         models_with_probs = {
#             "RandomForest": y_pred_prob_rf,
#             "XGBoost": y_pred_prob_xgb,
#             "GradientBoosting": y_pred_prob_gb,
#             "Ensemble": y_pred_prob_ensemble
#         }

#         optimal_thresholds = {}
#         for model_name, probs in models_with_probs.items():
#             thresholds = np.linspace(0.1, 0.5, 5)  # 0.1 ile 0.5 arasında threshold tuning
#             best_threshold = 0.5
#             best_recall = 0

#             for threshold in thresholds:
#                 preds = np.argmax(probs >= threshold, axis=1)
#                 recall = recall_score(y_test, preds, average='macro')
#                 if recall > best_recall:
#                     best_recall = recall
#                     best_threshold = threshold

#             optimal_thresholds[model_name] = (best_threshold, best_recall)
#             print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best Recall = {best_recall:.3f}")

#         # Nihai Classification Reports
#         print("\n=== Classification Reports ===")
#         for model_name, probs in models_with_probs.items():
#             threshold = optimal_thresholds[model_name][0]
#             preds = np.argmax(probs >= threshold, axis=1)
#             print(f"\nClassification Report for {model_name} (Threshold = {threshold:.2f}):")
#             print(classification_report(y_test, preds))

#         # ============================
#         # ROC Eğrileri ve Macro-Average ROC
#         # ============================
#         plt.figure(figsize=(18, 12))

#         # ROC Subplots
#         for idx, (model_name, probs) in enumerate(models_with_probs.items(), start=1):
#             plt.subplot(2, 3, idx)
#             for class_label in unique_classes:
#                 class_idx = np.where(unique_classes == class_label)[0][0]
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
#                 auc_val = auc(fpr, tpr)
#                 plt.plot(fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})")
#             plt.plot([0, 1], [0, 1], '--', color='gray')
#             plt.title(f"**{model_name}**", fontsize=15, fontweight='bold')
#             plt.xlabel("FPR", fontsize=14, fontweight='bold')
#             plt.ylabel("TPR", fontsize=14, fontweight='bold')
#             plt.legend(loc="lower right", fontsize=14, prop={'size': 14,'weight': 'bold'})

#         # Macro-Average ROC Curve
#         plt.subplot(2, 3, 5)
#         for model_name, probs in models_with_probs.items():
#             mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probs)
#             plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})")
#         plt.plot([0, 1], [0, 1], '--', color='gray')
#         plt.title("**Macro-Average ROC (All Models)**", fontsize=15, fontweight='bold')
#         plt.xlabel("FPR", fontsize=14, fontweight='bold')
#         plt.ylabel("TPR", fontsize=14, fontweight='bold')
#         plt.legend(loc="lower right", fontsize=14, prop={'size': 14,'weight': 'bold'})

#         plt.suptitle(f"ROC Curves - {oversampling_name} + {dataset_name}", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

#         #     plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
# #     plt.title("**Macro-Average ROC (All Models)**", fontsize=15, fontweight='bold')
# #     plt.xlabel("**False Positive Rate**", fontsize=14, fontweight='bold')
# #     plt.ylabel("**True Positive Rate**", fontsize=14, fontweight='bold')
# #     plt.legend(loc="lower right", fontsize=14, prop={'size': 14,'weight': 'bold'})  # Increase legend size

# #     # Adjust layout and show the figure
# #     plt.suptitle("**ROC Curves for Cost Sensitive**", fontsize=16, fontweight='bold')
# #     plt.tight_layout(rect=[0, 0, 1, 0.95])
# #     plt.show()

# ########################2.PREDICTION PHASE PREDICTION######################
# # Input Cases for Prediction
# import pandas as pd

# input_cases = pd.DataFrame({
#     "PGA": [0.13, 0.37, 0.12, 0.375],
#     "H": [18, 12, 6, 18],
#     "B": [71, 10, 10, 11],
#     "q": [20, 60, 15, 90],
#     "depth": [1.5, 3.3, 2, 1.2],
#     "thickness": [5, 6, 4, 5]
# })

# # ROS+HYPERPARAMETRETUNNİNG+threshold TUNNİNG (0.3) ile prediction
# # Sütun isimlerini eğitim sırasında kullanılan isimlerle eşleştir
# input_cases.columns = ['pga', 'H', 'B', 'q', 'depth', 'thickness']

# # Optimum threshold değerini ayarla
# optimal_threshold = 0.35

# # Random Forest
# rf_probs = best_rf.predict_proba(input_cases)
# rf_preds = np.argmax(rf_probs >= optimal_threshold, axis=1)

# # XGBoost
# xgb_probs = best_xgb.predict_proba(input_cases)
# xgb_preds = np.argmax(xgb_probs >= optimal_threshold, axis=1)

# # Gradient Boosting
# gb_probs = best_gb.predict_proba(input_cases)
# gb_preds = np.argmax(gb_probs >= optimal_threshold, axis=1)

# # Ensemble
# ensemble_probs = ensemble_clf.predict_proba(input_cases)
# ensemble_preds = np.argmax(ensemble_probs >= optimal_threshold, axis=1)

# # Tahminleri bir DataFrame'de birleştir
# predictions = pd.DataFrame({
#     "Case": [1, 2, 3, 4],
#     "RandomForest_Prediction": rf_preds,
#     "XGBoost_Prediction": xgb_preds,
#     "GradientBoosting_Prediction": gb_preds,
#     "Ensemble_Prediction": ensemble_preds
# })

# print("\n=== Tahminler ===")
# print(predictions)

############################################################### ESAS 1.MODEL TRAINING PHASE WITH (ROS+HYPERPARAMETRETUNNİNG+threshold TUNNİNG (0.5))k-fold cross-validation#############
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc, recall_score
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import label_binarize

# # ==== Veri seti ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemi (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ---- Fonksiyon: Macro-Average ROC hesaplama ----
# def compute_macro_roc(y_true_bin, y_prob):
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_true_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc

# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 4) Çoklu sınıf ROC için test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)
#         y_test_bin = label_binarize(y_test, classes=unique_classes)

#         # Stratified K-Fold Cross Validation
#         stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # ============================
#         # A) Random Forest with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         y_pred_prob_rf = best_rf.predict_proba(X_test)
#         print("Best Random Forest Parameters:", rf_grid.best_params_)

#         # ============================
#         # B) XGBoost with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#         print("Best XGBoost Parameters:", xgb_grid.best_params_)

#         # ============================
#         # C) Gradient Boosting with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_params = {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1]
#         }
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         y_pred_prob_gb = best_gb.predict_proba(X_test)
#         print("Best Gradient Boosting Parameters:", gb_grid.best_params_)

#         # ============================
#         # D) Ensemble
#         # ============================
#         print("\nOptimizasyon: Ensemble")
#         ensemble_clf = VotingClassifier(
#             estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)

#         # ============================
#         # Recall Odaklı Threshold Tuning
#         # ============================
#         models_with_probs = {
#             "RandomForest": y_pred_prob_rf,
#             "XGBoost": y_pred_prob_xgb,
#             "GradientBoosting": y_pred_prob_gb,
#             "Ensemble": y_pred_prob_ensemble
#         }

#         optimal_thresholds = {}
#         for model_name, probs in models_with_probs.items():
#             thresholds = np.linspace(0.1, 0.5, 5)  
#             best_threshold = 0.5
#             best_recall = 0

#             for threshold in thresholds:
#                 preds = np.argmax(probs >= threshold, axis=1)
#                 recall = recall_score(y_test, preds, average='macro')
#                 if recall > best_recall:
#                     best_recall = recall
#                     best_threshold = threshold

#             optimal_thresholds[model_name] = (best_threshold, best_recall)
#             print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best Recall = {best_recall:.3f}")

#         # Final Classification Reports
#         print("\n=== Classification Reports ===")
#         for model_name, probs in models_with_probs.items():
#             threshold = optimal_thresholds[model_name][0]
#             preds = np.argmax(probs >= threshold, axis=1)
#             print(f"\nClassification Report for {model_name} (Threshold = {threshold:.2f}):")
#             print(classification_report(y_test, preds))

#         # ============================
# # ROC Visualization (2x3 Layout)
# # ============================

#         plt.figure(figsize=(18, 12))

#         # ROC Subplots (2 Satır x 3 Sütun Düzeni)
#         for idx, (model_name, probs) in enumerate(models_with_probs.items(), start=1):
#             plt.subplot(2, 3, idx)  # 2 rows, 3 columns
#             for class_label in unique_classes:
#                 class_idx = np.where(unique_classes == class_label)[0][0]
#                 fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
#                 auc_val = auc(fpr, tpr)
#                 plt.plot(fpr, tpr, label=f"Class {class_label} (AUC={auc_val:.2f})", linewidth=2)

#             plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
#             plt.title(f"**{model_name}**", fontsize=15, fontweight='bold')
#             plt.xlabel("**False Positive Rate**", fontsize=14, fontweight='bold')
#             plt.ylabel("**True Positive Rate**", fontsize=14, fontweight='bold')
#             plt.legend(loc="lower right", fontsize=14, prop={'size': 14, 'weight': 'bold'})

#         # Macro-Average ROC Curve
#         plt.subplot(2, 3, 5)  # Position for macro-average ROC
#         for model_name, probs in models_with_probs.items():
#             mean_fpr, mean_tpr, macro_auc = compute_macro_roc(y_test_bin, probs)
#             plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={macro_auc:.2f})", linewidth=2)

#         plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
#         plt.title("**Macro-Average ROC (All Models)**", fontsize=15, fontweight='bold')
#         plt.xlabel("**False Positive Rate**", fontsize=14, fontweight='bold')
#         plt.ylabel("**True Positive Rate**", fontsize=14, fontweight='bold')
#         plt.legend(loc="lower right", fontsize=14, prop={'size': 14, 'weight': 'bold'})

#         plt.suptitle(f"ROC Curves - {oversampling_name} + {dataset_name}", fontsize=16, fontweight='bold')
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()




# ########################2.PREDICTION PHASE PREDICTION######################
# # ============================
# # 2. PREDICTION PHASE
# # ============================

# import pandas as pd

# # Input Cases for Prediction
# input_cases = pd.DataFrame({
#     "PGA": [0.13, 0.37, 0.12, 0.375],
#     "H": [18, 12, 6, 18],
#     "B": [71, 10, 10, 11],
#     "q": [20, 60, 15, 90],
#     "depth": [1.5, 3.3, 2, 1.2],
#     "thickness": [5, 6, 4, 5]
# })

# # Column names should match the training dataset
# input_cases.columns = ['pga', 'H', 'B', 'q', 'depth', 'thickness']

# # Predefined Optimal Thresholds from Training Phase
# optimal_thresholds = {
#     "RandomForest": 0.30,
#     "XGBoost": 0.35, #0.30,
#     "GradientBoosting":0.10,
#     "Ensemble": 0.20
# }

# # ============================
# #  APPLYING THRESHOLD FOR EACH MODEL
# # ============================

# # Random Forest Predictions
# rf_probs = best_rf.predict_proba(input_cases)
# rf_preds = np.argmax(rf_probs >= optimal_thresholds["RandomForest"], axis=1)

# # XGBoost Predictions
# xgb_probs = best_xgb.predict_proba(input_cases)
# xgb_preds = np.argmax(xgb_probs >= optimal_thresholds["XGBoost"], axis=1)

# # Gradient Boosting Predictions
# gb_probs = best_gb.predict_proba(input_cases)
# gb_preds = np.argmax(gb_probs >= optimal_thresholds["GradientBoosting"], axis=1)

# # Ensemble Model Predictions
# ensemble_probs = ensemble_clf.predict_proba(input_cases)
# ensemble_preds = np.argmax(ensemble_probs >= optimal_thresholds["Ensemble"], axis=1)

# # ============================
# #  CREATING A PREDICTION REPORT
# # ============================

# # Compile results into a structured DataFrame
# predictions = pd.DataFrame({
#     "Case": [1, 2, 3, 4],
#     "RandomForest_Prediction": rf_preds,
#     "XGBoost_Prediction": xgb_preds,
#     "GradientBoosting_Prediction": gb_preds,
#     "Ensemble_Prediction": ensemble_preds
# })

# print("\n=== Final Predictions Using Optimized Thresholds ===")
# print(predictions)

# # Optionally, save predictions to a CSV file
# predictions.to_csv("final_predictions.csv", index=False)




###########################################################################

##################################################################
# MODEL TRAINING PHASE WITH (ROS + HYPERPARAMETER TUNING + THRESHOLD TUNING + K-FOLD CROSS VALIDATION) with normalization
##################################################################
##################################################################
# MODEL TRAINING PHASE WITH CLASSIFICATION REPORTS
##################################################################

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.preprocessing import StandardScaler, label_binarize
# from sklearn.metrics import classification_report, recall_score
# import matplotlib.pyplot as plt
# import numpy as np
# from joblib import dump

# # ==== Veri seti ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Yöntemi (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ------------------------------------------------

# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Özellikler ve hedef değişkeni ayırma
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling ile veri dengesi sağlama
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Normalization (StandardScaler)
#         scaler = StandardScaler()
#         X_resampled_scaled = scaler.fit_transform(X_resampled)

#         # 4) Eğitim ve test verisi ayırma
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
#         )

#         # 5) Çoklu sınıf test etiketlerini binarize ediyoruz (One-vs-Rest)
#         unique_classes = np.unique(y_resampled)

#         # Stratified K-Fold Cross Validation
#         stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # ============================
#         # A) Random Forest with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: Random Forest")
#         rf_params = {
#             'n_estimators': [100, 200],
#             'max_depth': [None, 10, 20],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#         rf_grid = GridSearchCV(
#             RandomForestClassifier(random_state=42),
#             rf_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         rf_grid.fit(X_train, y_train)
#         best_rf = rf_grid.best_estimator_
#         rf_preds = best_rf.predict(X_test)

#         print("\nRandom Forest Classification Report:")
#         print(classification_report(y_test, rf_preds))

#         # ============================
#         # B) XGBoost with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: XGBoost")
#         xgb_params = {
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'n_estimators': [100, 200],
#             'subsample': [0.8, 1],
#             'colsample_bytree': [0.8, 1]
#         }
#         xgb_grid = GridSearchCV(
#             XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#             xgb_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         xgb_grid.fit(X_train, y_train)
#         best_xgb = xgb_grid.best_estimator_
#         xgb_preds = best_xgb.predict(X_test)

#         print("\nXGBoost Classification Report:")
#         print(classification_report(y_test, xgb_preds))

#         # ============================
#         # C) Gradient Boosting with Stratified K-Fold
#         # ============================
#         print("\nOptimizasyon: Gradient Boosting")
#         gb_params = {
#             'n_estimators': [100, 200],
#             'learning_rate': [0.01, 0.1],
#             'max_depth': [3, 5],
#             'subsample': [0.8, 1]
#         }
#         gb_grid = GridSearchCV(
#             GradientBoostingClassifier(random_state=42),
#             gb_params,
#             cv=stratified_kfold,
#             scoring='recall_macro',
#             n_jobs=-1
#         )
#         gb_grid.fit(X_train, y_train)
#         best_gb = gb_grid.best_estimator_
#         gb_preds = best_gb.predict(X_test)

#         print("\nGradient Boosting Classification Report:")
#         print(classification_report(y_test, gb_preds))

#         # ============================
#         # D) Ensemble Model
#         # ============================
#         print("\nOptimizasyon: Ensemble")
#         ensemble_clf = VotingClassifier(
#             estimators=[("rf", best_rf), ("xgb", best_xgb), ("gb", best_gb)],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         ensemble_preds = ensemble_clf.predict(X_test)

#         print("\nEnsemble Model Classification Report:")
#         print(classification_report(y_test, ensemble_preds))

#         # ============================
#         # Save Models
#         # ============================
#         dump(best_rf, "best_rf_model.joblib")
#         dump(best_xgb, "best_xgb_model.joblib")
#         dump(best_gb, "best_gb_model.joblib")
#         dump(ensemble_clf, "ensemble_clf_model.joblib")
#         dump(scaler, "scaler.joblib")
# # ============================
# # 2. PREDICTION PHASE WITH ENSEMBLE AND OPTIMIZED THRESHOLDS
# # ============================

# import pandas as pd
# import numpy as np
# from joblib import load

# # Input Cases for Prediction
# input_cases = pd.DataFrame({
#     "PGA": [0.13, 0.37, 0.12, 0.375],
#     "H": [18, 12, 6, 18],
#     "B": [71, 10, 10, 11],
#     "q": [20, 60, 15, 90],
#     "depth": [1.5, 3.3, 2, 1.2],
#     "thickness": [5, 6, 4, 5]
# })

# # Column names should match the training dataset
# input_cases.columns = ['pga', 'H', 'B', 'q', 'depth', 'thickness']

# # ============================
# # RELOAD TRAINED MODELS AND SCALER
# # ============================

# # Load models and scaler
# best_rf = load("best_rf_model.joblib")
# best_xgb = load("best_xgb_model.joblib")
# best_gb = load("best_gb_model.joblib")
# ensemble_clf = load("ensemble_clf_model.joblib")
# scaler = load("scaler.joblib")

# # ============================
# # APPLY STANDARDIZATION TO INPUT CASES
# # ============================

# # Standardize input cases using the previously saved scaler
# input_cases_scaled = scaler.transform(input_cases)

# # ============================
# # APPLYING OPTIMIZED THRESHOLDS FOR EACH MODEL
# # ============================

# # Predefined Optimal Thresholds
# optimal_thresholds = {
#     "RandomForest": 0.30,
#     "XGBoost": 0.35,
#     "GradientBoosting": 0.10,
#     "Ensemble": 0.20
# }

# # Random Forest Predictions
# rf_probs = best_rf.predict_proba(input_cases_scaled)
# rf_preds = np.argmax(rf_probs >= optimal_thresholds["RandomForest"], axis=1)

# # XGBoost Predictions
# xgb_probs = best_xgb.predict_proba(input_cases_scaled)
# xgb_preds = np.argmax(xgb_probs >= optimal_thresholds["XGBoost"], axis=1)

# # Gradient Boosting Predictions
# gb_probs = best_gb.predict_proba(input_cases_scaled)
# gb_preds = np.argmax(gb_probs >= optimal_thresholds["GradientBoosting"], axis=1)

# # Ensemble Model Predictions
# ensemble_probs = ensemble_clf.predict_proba(input_cases_scaled)
# ensemble_preds = np.argmax(ensemble_probs >= optimal_thresholds["Ensemble"], axis=1)

# # ============================
# # CREATE A PREDICTION REPORT
# # ============================

# # Compile results into a structured DataFrame
# predictions = pd.DataFrame({
#     "Case": [1, 2, 3, 4],
#     "RandomForest_Prediction": rf_preds,
#     "XGBoost_Prediction": xgb_preds,
#     "GradientBoosting_Prediction": gb_preds,
#     "Ensemble_Prediction": ensemble_preds
# })

# print("\n=== Final Predictions Using Optimized Thresholds ===")
# print(predictions)

# # Save predictions to a CSV file for further analysis
# predictions.to_csv("final_predictions.csv", index=False)

# # Optionally display the DataFrame for visualization
# predictions



#####################################################################shap_feature_selection Function####################
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, recall_score
import pandas as pd
import numpy as np
import shap

# ==== Dataset Selection ====
datasets = [
    ("IQR Filtered", iqr_cleaned_data)
]

# ==== Oversampling Method ====
oversampling_methods = {
    "ROS": RandomOverSampler(random_state=42)
}

# ==== SHAP Feature Selection ====
def shap_feature_selection(model, X, num_features=5):
    explainer = shap.Explainer(model.predict, X)  
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_type="bar")  
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="Importance", ascending=False)
    selected_features = importance_df["Feature"].head(num_features).tolist()
    return selected_features

# ==== Training and Evaluation ====
for oversampling_name, oversampler in oversampling_methods.items():
    print(f"\n=== Oversampling Method: {oversampling_name} ===")

    for dataset_name, dataset in datasets:
        print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

        # 1) Features and Target Separation
        X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
        y = dataset['dver']

        # 2) Oversampling
        X_resampled, y_resampled = oversampler.fit_resample(X, y)

        # 3) Feature Selection with SHAP
        print("\nPerforming Feature Selection...")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_resampled, y_resampled)
        selected_features = shap_feature_selection(rf_model, X_resampled, num_features=4)
        print(f"Selected Features: {selected_features}")

        # 4) Narrowing Data Based on Selected Features
        X_selected = X_resampled[selected_features]

        # 5) Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_resampled, test_size=0.2, random_state=42
        )

        # Stratified K-Fold Cross Validation
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Models and Parameters
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42)
        }

        params = {
            "RandomForest": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "XGBoost": {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1]
            },
            "GradientBoosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1]
            }
        }

        best_estimators = {}
        optimal_thresholds = {}

        for model_name, model in models.items():
            print(f"\nOptimizing: {model_name}")
            grid = GridSearchCV(
                model, params[model_name], cv=stratified_kfold,
                scoring='recall_macro', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_estimators[model_name] = grid.best_estimator_
            print(f"Best Parameters for {model_name}: {grid.best_params_}")

            # Threshold Tuning (0.1 - 0.5)
            print(f"\nTuning Threshold for {model_name}...")
            best_threshold, best_recall = None, 0
            thresholds = np.linspace(0.1, 0.5, 5)

            for threshold in thresholds:
                y_pred_prob = best_estimators[model_name].predict_proba(X_test)
                preds = np.argmax(y_pred_prob >= threshold, axis=1)
                recall = recall_score(y_test, preds, average='macro')

                print(f"Threshold {threshold:.2f}: Recall = {recall:.3f}")  # Debugging

                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold

            optimal_thresholds[model_name] = best_threshold
            print(f"Optimal Threshold for {model_name}: {best_threshold:.2f} with Recall: {best_recall:.3f}")

        # Ensemble Model
        ensemble_clf = VotingClassifier(
            estimators=[
                ("RandomForest", best_estimators["RandomForest"]),
                ("XGBoost", best_estimators["XGBoost"]),
                ("GradientBoosting", best_estimators["GradientBoosting"])
            ],
            voting="soft"
        )
        ensemble_clf.fit(X_train, y_train)

        # Apply threshold tuning to Ensemble model
        print(f"\nTuning Threshold for Ensemble Model...")
        best_threshold, best_recall = None, 0
        thresholds = np.linspace(0.1, 0.5, 5)

        for threshold in thresholds:
            y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)
            preds = np.argmax(y_pred_prob_ensemble >= threshold, axis=1)
            recall = recall_score(y_test, preds, average='macro')

            print(f"Threshold {threshold:.2f}: Recall = {recall:.3f}")

            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold

        optimal_thresholds["Ensemble"] = best_threshold
        print(f"Optimal Threshold for Ensemble: {best_threshold:.2f} with Recall: {best_recall:.3f}")

        # Apply optimal threshold to predictions
        preds_ensemble = np.argmax(ensemble_clf.predict_proba(X_test) >= best_threshold, axis=1)

        print(f"\n=== Classification Report for Ensemble (Threshold = {best_threshold:.2f}) ===")
        print(classification_report(y_test, preds_ensemble))

        # ============================
        # Prediction Phase with Optimized Thresholds
        # ============================

        input_cases = pd.DataFrame({
            "pga": [0.13, 0.37, 0.12, 0.375],
            "H": [18, 12, 6, 18],
            "B": [71, 10, 10, 11],
            "q": [20, 60, 15, 90],
            "depth": [1.5, 3.3, 2, 1.2],
            "thickness": [5, 6, 4, 5]
        })

        input_selected = input_cases[selected_features]

        predictions = pd.DataFrame({
            "Case": [1, 2, 3, 4],
            "RandomForest": np.argmax(best_estimators["RandomForest"].predict_proba(input_selected) >= optimal_thresholds["RandomForest"], axis=1),
            "XGBoost": np.argmax(best_estimators["XGBoost"].predict_proba(input_selected) >= optimal_thresholds["XGBoost"], axis=1),
            "GradientBoosting": np.argmax(best_estimators["GradientBoosting"].predict_proba(input_selected) >= optimal_thresholds["GradientBoosting"], axis=1),
            "Ensemble": np.argmax(ensemble_clf.predict_proba(input_selected) >= optimal_thresholds["Ensemble"], axis=1)
        })

        print("\n=== Final Predictions Using Optimized Thresholds ===")
        print(predictions)








#####################################################################################
# MODEL TRAINING PHASE WITH FEATURE ENGINEERING + ENSEMBLE WITH DYNAMIC VOTING (FIXED)
#####################################################################################



# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_curve, auc
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures, label_binarize
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from joblib import dump

# # ==== Dataset Selection ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Gradient Boosting Hyperparameter Grid ====
# gb_params = {
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5],
#     'n_estimators': [100, 200],
#     'subsample': [0.8, 1]
# }

# # ==== Compute Macro-Average ROC ====
# def compute_macro_roc(y_test_bin, y_prob):
#     tprs, aucs = [], []
#     mean_fpr = np.linspace(0, 1, 100)
#     for i in range(y_test_bin.shape[1]):
#         fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
#         auc_i = auc(fpr_i, tpr_i)
#         aucs.append(auc_i)
#         tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     macro_auc = np.mean(aucs)
#     return mean_fpr, mean_tpr, macro_auc

# # ==== Feature Engineering Function (Fixed) ====
# def feature_engineering(X, y):
#     """
#     Apply feature selection, transformation, and scaling.
#     Ensures X and y remain consistent in shape.
#     """
#     # Ensure X and y have the same number of samples
#     min_samples = min(X.shape[0], y.shape[0])
#     X, y = X.iloc[:min_samples, :], y.iloc[:min_samples]

#     # 1️⃣ Remove highly correlated features
#     X = X.drop(columns=['B', 'q'])  # Removing 'B' (high correlation with H), 'q' (high correlation with H)

#     # 2️⃣ Create polynomial and interaction features
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_poly = poly.fit_transform(X[['pga', 'H', 'depth']])
#     X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['pga', 'H', 'depth']))

#     # 3️⃣ Ensure all transformations keep the same number of rows
#     X_transformed = pd.concat([X.reset_index(drop=True), X_poly], axis=1)

#     # 4️⃣ Scale the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_transformed)
    
#     return X_scaled, y.reset_index(drop=True), scaler

# # ==== TRAINING LOOP ====
# for dataset_name, dataset in datasets:
#     print(f"\n### FEATURE ENGINEERING + TRAINING: {dataset_name} Dataset ###")

#     # 1) Extract Features and Target
#     X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#     y = dataset['dver']

#     # 2) Feature Engineering
#     X_transformed, y_transformed, scaler = feature_engineering(X, y)

#     # 3) Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_transformed, y_transformed, test_size=0.2, random_state=42
#     )

#     # 4) Compute Class Weights
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

#     # 5) Label Binarization for Multi-Class ROC Analysis
#     y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

#     # =============================================
#     # 1) Random Forest
#     # =============================================
#     rf = RandomForestClassifier(class_weight=class_weights_dict, random_state=42)
#     rf.fit(X_train, y_train)
#     y_pred_rf = rf.predict(X_test)
#     y_pred_prob_rf = rf.predict_proba(X_test)
#     print("\nRandom Forest Classification Report:")
#     print(classification_report(y_test, y_pred_rf))

#     # =============================================
#     # 2) XGBoost
#     # =============================================
#     xgb = XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
#     xgb.fit(X_train, y_train)
#     y_pred_xgb = xgb.predict(X_test)
#     y_pred_prob_xgb = xgb.predict_proba(X_test)
#     print("\nXGBoost Classification Report:")
#     print(classification_report(y_test, y_pred_xgb))

#     # =============================================
#     # 3) Gradient Boosting (Hyperparameter Tuning)
#     # =============================================
#     gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=gb_params, cv=3, scoring='accuracy', n_jobs=-1)
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     gb_preds = best_gb.predict(X_test)
#     y_pred_prob_gb = best_gb.predict_proba(X_test)
#     print("Best Gradient Boosting Parameters:", gb_grid.best_params_)
#     print("\nGradient Boosting Classification Report:")
#     print(classification_report(y_test, gb_preds))

#     # =============================================
#     # 4) Dynamic Voting Ensemble Model
#     # =============================================
#     print("\nTraining Dynamic Voting Ensemble Model...")

#     # Compute Model Confidence Scores for Dynamic Weighting
#     confidence_rf = np.mean(np.max(y_pred_prob_rf, axis=1))
#     confidence_xgb = np.mean(np.max(y_pred_prob_xgb, axis=1))
#     confidence_gb = np.mean(np.max(y_pred_prob_gb, axis=1))

#     # Normalize Weights (Sum to 1)
#     total_confidence = confidence_rf + confidence_xgb + confidence_gb
#     weights = {
#         "rf": confidence_rf / total_confidence,
#         "xgb": confidence_xgb / total_confidence,
#         "gb": confidence_gb / total_confidence
#     }

#     print(f"Dynamic Weights: RF={weights['rf']:.2f}, XGB={weights['xgb']:.2f}, GB={weights['gb']:.2f}")

#     # Train VotingClassifier with Dynamic Weights
#     ensemble_clf = VotingClassifier(
#         estimators=[("rf", rf), ("xgb", xgb), ("gb", best_gb)],
#         voting="soft",
#         weights=[weights["rf"], weights["xgb"], weights["gb"]]
#     )
#     ensemble_clf.fit(X_train, y_train)
#     y_pred_ensemble = ensemble_clf.predict(X_test)
#     y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)
    
#     print("\nEnsemble Classification Report:")
#     print(classification_report(y_test, y_pred_ensemble))

#     # =============================================
#     # Save Models and Scaler
#     # =============================================
#     dump(ensemble_clf, f"{dataset_name}_dynamic_ensemble_model.joblib")
#     dump(scaler, f"{dataset_name}_scaler.joblib")

#     print("\n=== Final Models and Dynamic Ensemble Saved Successfully! ===")



# #########################################Full Code with Threshold Tuning and Focal Loss########################################
# ##############################################################################################################################
# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, recall_score, roc_curve, auc

# # ==== Dataset ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Method ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ==== Custom Focal Loss for XGBoost ====
# def focal_loss_objective(y_true, y_pred, gamma=2.0):
#     """
#     Custom Focal Loss for multi-class classification in XGBoost.
#     """
#     y_true_one_hot = np.eye(y_pred.shape[1])[y_true.astype(int)]  # One-hot encode y_true
#     p = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)  # Softmax probabilities

#     # Gradient
#     grad = (p - y_true_one_hot) * ((1 - p) ** gamma)

#     # Hessian
#     hess = p * (1 - p) * ((1 - p) ** gamma) * (
#         gamma * np.log(np.maximum(p, 1e-15)) * (y_true_one_hot - p) + 1
#     )
#     return grad.ravel(), hess.ravel()

# # ==== Threshold Tuning Function ====
# def tune_threshold(model, X_test, y_test, thresholds=np.linspace(0.1, 0.5, 5)):
#     """
#     Tune the threshold for the model to optimize recall or another metric.
#     """
#     best_threshold = 0.5
#     best_recall = 0
#     y_prob = model.predict_proba(X_test)

#     for threshold in thresholds:
#         y_pred = np.argmax(y_prob >= threshold, axis=1)
#         recall = recall_score(y_test, y_pred, average='macro')
#         if recall > best_recall:
#             best_recall = recall
#             best_threshold = threshold

#     return best_threshold, best_recall

# # ==== Training Pipeline ====
# for oversampling_name, oversampler in oversampling_methods.items():
#     print(f"\n=== Oversampling Method: {oversampling_name} ===")

#     for dataset_name, dataset in datasets:
#         print(f"\n### {oversampling_name} + {dataset_name} Dataset ###")

#         # 1) Features and Target Separation
#         X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#         y = dataset['dver']

#         # 2) Oversampling
#         X_resampled, y_resampled = oversampler.fit_resample(X, y)

#         # 3) Train-Test Split
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
#         )

#         # Stratified K-Fold Cross Validation
#         stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # ============================
#         # A) Random Forest
#         # ============================
#         print("\nTraining: Random Forest")
#         rf_model = RandomForestClassifier(
#             n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
#         )
#         rf_model.fit(X_train, y_train)
#         rf_threshold, rf_recall = tune_threshold(rf_model, X_test, y_test)
#         print(f"\nRandom Forest Optimal Threshold: {rf_threshold:.2f}, Recall: {rf_recall:.3f}")
#         rf_preds = np.argmax(rf_model.predict_proba(X_test) >= rf_threshold, axis=1)
#         print("\nRandom Forest Classification Report:")
#         print(classification_report(y_test, rf_preds))

#         # ============================
#         # B) XGBoost with Focal Loss
#         # ============================
#         print("\nTraining: XGBoost with Focal Loss")
#         xgb_model = XGBClassifier(
#             max_depth=5,
#             learning_rate=0.1,
#             n_estimators=100,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             objective=focal_loss_objective,
#             use_label_encoder=False,
#             eval_metric="mlogloss",
#             random_state=42
#         )
#         xgb_model.fit(X_train, y_train)
#         xgb_threshold, xgb_recall = tune_threshold(xgb_model, X_test, y_test)
#         print(f"\nXGBoost Optimal Threshold: {xgb_threshold:.2f}, Recall: {xgb_recall:.3f}")
#         xgb_preds = np.argmax(xgb_model.predict_proba(X_test) >= xgb_threshold, axis=1)
#         print("\nXGBoost Classification Report:")
#         print(classification_report(y_test, xgb_preds))

#         # ============================
#         # C) Gradient Boosting with Class Weights
#         # ============================
#         print("\nTraining: Gradient Boosting with Class Weights")
#         gb_model = GradientBoostingClassifier(
#             n_estimators=200,
#             learning_rate=0.1,
#             max_depth=3,
#             subsample=0.8,
#             random_state=42
#         )
#         gb_model.fit(X_train, y_train)
#         gb_threshold, gb_recall = tune_threshold(gb_model, X_test, y_test)
#         print(f"\nGradient Boosting Optimal Threshold: {gb_threshold:.2f}, Recall: {gb_recall:.3f}")
#         gb_preds = np.argmax(gb_model.predict_proba(X_test) >= gb_threshold, axis=1)
#         print("\nGradient Boosting Classification Report:")
#         print(classification_report(y_test, gb_preds))

#         # ============================
#         # D) Ensemble Model
#         # ============================
#         ensemble_clf = VotingClassifier(
#             estimators=[
#                 ("RandomForest", rf_model),
#                 ("XGBoost", xgb_model),
#                 ("GradientBoosting", gb_model)
#             ],
#             voting="soft"
#         )
#         ensemble_clf.fit(X_train, y_train)
#         ensemble_threshold, ensemble_recall = tune_threshold(ensemble_clf, X_test, y_test)
#         print(f"\nEnsemble Optimal Threshold: {ensemble_threshold:.2f}, Recall: {ensemble_recall:.3f}")
#         ensemble_preds = np.argmax(ensemble_clf.predict_proba(X_test) >= ensemble_threshold, axis=1)
#         print("\nEnsemble Model Classification Report:")
#         print(classification_report(y_test, ensemble_preds))

#         # ============================
#         # Prediction Phase
#         # ============================
#         input_cases = pd.DataFrame({
#             "pga": [0.13, 0.37, 0.12, 0.375],
#             "H": [18, 12, 6, 18],
#             "B": [71, 10, 10, 11],
#             "q": [20, 60, 15, 90],
#             "depth": [1.5, 3.3, 2, 1.2],
#             "thickness": [5, 6, 4, 5]
#         })

#         predictions = pd.DataFrame({
#             "Case": [1, 2, 3, 4],
#             "RandomForest": np.argmax(rf_model.predict_proba(input_cases) >= rf_threshold, axis=1),
#             "XGBoost": np.argmax(xgb_model.predict_proba(input_cases) >= xgb_threshold, axis=1),
#             "GradientBoosting": np.argmax(gb_model.predict_proba(input_cases) >= gb_threshold, axis=1),
#             "Ensemble": np.argmax(ensemble_clf.predict_proba(input_cases) >= ensemble_threshold, axis=1)
#         })

#         print("\n=== Final Predictions with Threshold Tuning ===")
#         print(predictions)


################Updated Code: Multi-Objective Optimization ######################################

# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, make_scorer
# import numpy as np
# import pandas as pd

# # ==== Dataset Selection ====
# datasets = [
#     ("IQR Filtered", iqr_cleaned_data)
# ]

# # ==== Oversampling Method (ROS) ====
# oversampling_methods = {
#     "ROS": RandomOverSampler(random_state=42)
# }

# # ==== Multi-Objective Scoring Function ====
# def multi_objective_scorer(y_true, y_pred):
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
#     precision = precision_score(y_true, y_pred, average='macro')
    
#     return (0.9* recall) + (0.05 * f1) + (0.05 * precision)

# custom_scorer = make_scorer(multi_objective_scorer)

# # ==== Hyperparameter Grids ====
# xgb_params = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.8, 1],
#     'colsample_bytree': [0.8, 1]
# }

# gb_params = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1]
# }

# # ==== TRAINING PHASE ====
# for dataset_name, dataset in datasets:
#     print(f"\n### Training XGBoost + Gradient Boosting + Ensemble: {dataset_name} ###")

#     # 1) Extract Features and Target
#     X = dataset[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
#     y = dataset['dver']

#     # 2) Oversampling for Class Balance
#     oversampler = oversampling_methods["ROS"]
#     X_resampled, y_resampled = oversampler.fit_resample(X, y)

#     # 3) Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_resampled, y_resampled, test_size=0.2, random_state=42
#     )

#     # 4) Stratified K-Fold Cross Validation
#     stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     # ============================
#     # 1) XGBoost with Multi-Objective Optimization
#     # ============================
#     print("\nOptimizing XGBoost with Multi-Objective Scoring...")
#     xgb_grid = GridSearchCV(
#         XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#         xgb_params,
#         cv=stratified_kfold,
#         scoring=custom_scorer,
#         n_jobs=-1
#     )
#     xgb_grid.fit(X_train, y_train)
#     best_xgb = xgb_grid.best_estimator_
#     y_pred_prob_xgb = best_xgb.predict_proba(X_test)
#     print("\nBest XGBoost Parameters:", xgb_grid.best_params_)

#     # ============================
#     # 2) Gradient Boosting with Multi-Objective Optimization
#     # ============================
#     print("\nOptimizing Gradient Boosting with Multi-Objective Scoring...")
#     gb_grid = GridSearchCV(
#         GradientBoostingClassifier(random_state=42),
#         gb_params,
#         cv=stratified_kfold,
#         scoring=custom_scorer,
#         n_jobs=-1
#     )
#     gb_grid.fit(X_train, y_train)
#     best_gb = gb_grid.best_estimator_
#     y_pred_prob_gb = best_gb.predict_proba(X_test)
#     print("\nBest Gradient Boosting Parameters:", gb_grid.best_params_)

#     # ============================
#     # 3) Dynamic Voting Ensemble Model
#     # ============================
#     print("\nTraining Dynamic Voting Ensemble Model...")
#     confidence_xgb = np.mean(np.max(y_pred_prob_xgb, axis=1))
#     confidence_gb = np.mean(np.max(y_pred_prob_gb, axis=1))

#     total_confidence = confidence_xgb + confidence_gb
#     weights = {
#         "xgb": confidence_xgb / total_confidence,
#         "gb": confidence_gb / total_confidence
#     }

#     ensemble_clf = VotingClassifier(
#         estimators=[("xgb", best_xgb), ("gb", best_gb)],
#         voting="soft",
#         weights=[weights["xgb"], weights["gb"]]
#     )
#     ensemble_clf.fit(X_train, y_train)
#     y_pred_prob_ensemble = ensemble_clf.predict_proba(X_test)
#     print("Dynamic Voting Ensemble Model Trained Successfully!")

#     # ============================
#     # 4) Threshold Tuning
#     # ============================
#     models_with_probs = {
#         "XGBoost": y_pred_prob_xgb,
#         "GradientBoosting": y_pred_prob_gb,
#         "Ensemble": y_pred_prob_ensemble
#     }

#     optimal_thresholds = {}
#     for model_name, probs in models_with_probs.items():
#         thresholds = np.linspace(0.1, 0.5, 5)
#         best_threshold = 0.5
#         best_recall = 0

#         for threshold in thresholds:
#             preds = np.argmax(probs >= threshold, axis=1)
#             recall = recall_score(y_test, preds, average='macro')
#             if recall > best_recall:
#                 best_recall = recall
#                 best_threshold = threshold

#         optimal_thresholds[model_name] = (best_threshold, best_recall)
#         print(f"{model_name}: Best Threshold = {best_threshold:.2f}, Best Recall = {best_recall:.3f}")

# # ============================
# # 2. PREDICTION PHASE
# # ============================

# # Input Cases for Prediction
# input_cases = pd.DataFrame({
#     "pga": [0.13, 0.37, 0.12, 0.375],
#     "H": [18, 12, 6, 18],
#     "B": [71, 10, 10, 11],
#     "q": [20, 60, 15, 90],
#     "depth": [1.5, 3.3, 2, 1.2],
#     "thickness": [5, 6, 4, 5]
# })

# # Applying the optimized threshold to predictions
# xgb_probs = best_xgb.predict_proba(input_cases)
# xgb_preds = np.argmax(xgb_probs >= optimal_thresholds["XGBoost"][0], axis=1)

# gb_probs = best_gb.predict_proba(input_cases)
# gb_preds = np.argmax(gb_probs >= optimal_thresholds["GradientBoosting"][0], axis=1)

# ensemble_probs = ensemble_clf.predict_proba(input_cases)
# ensemble_preds = np.argmax(ensemble_probs >= optimal_thresholds["Ensemble"][0], axis=1)

# # Compile results into a structured DataFrame
# predictions = pd.DataFrame({
#     "Case": [1, 2, 3, 4],
#     "XGBoost_Prediction": xgb_preds,
#     "GradientBoosting_Prediction": gb_preds,
#     "Ensemble_Prediction": ensemble_preds
# })

# print("\n=== Final Predictions Using Optimized Models ===")
# print(predictions)

# # Optionally, save predictions to a CSV file
# predictions.to_csv("final_predictions.csv", index=False)




