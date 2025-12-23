import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# 1) Veriyi oku
df = pd.read_csv("adapvtest.csv", sep=";")  # sizde farklıysa düzeltin

X = df[['pga', 'H', 'B', 'q', 'depth', 'thickness']]
y = df['dver']

# 2) HATALI düzeni simüle et: önce oversample, sonra split
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# ros.sample_indices_: resampled her satırın orijinalde hangi satırdan geldiğini verir (ROS için)
orig_idx = ros.sample_indices_

# 3) Resampled veri üzerinde split
res_ids = np.arange(len(y_res))
train_ids, test_ids = train_test_split(
    res_ids, test_size=0.2, random_state=42, stratify=y_res
)

train_orig = set(orig_idx[train_ids])
test_orig  = set(orig_idx[test_ids])

overlap = train_orig.intersection(test_orig)
print("Train ve test'e ortak düşen orijinal örnek sayısı:", len(overlap))

# Eğer bu sayı 0'dan büyükse:
# -> Aynı orijinal gözlem kopyaları hem train hem test'e girmiştir (leakage göstergesi).
