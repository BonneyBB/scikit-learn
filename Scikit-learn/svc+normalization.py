from sklearn import datasets
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

# 初步展现preprocessing使用效果
input = np.array([[1,2,3],
                  [7,9,3],
                  [-3,-5,-9]], dtype=np.float64)

print(input)
input = preprocessing.scale(input)
# input = preprocessing.minmax_scale(input, feature_range=(0,1))
print(input)

model = SVC()
X,y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2
                          , random_state=22, n_clusters_per_class=1,scale=100)

# 进行标准化前的训练
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
model.fit(train_X, train_y)
print(model.score(test_X, test_y))

# 进行标准化后的训练
X = preprocessing.scale(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
model.fit(train_X, train_y)
print(model.score(test_X, test_y))
