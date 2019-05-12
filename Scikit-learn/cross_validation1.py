from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import  matplotlib.pyplot as plt


# load iris_dataset
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# use cross validation to train
knn = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(knn, iris_X, iris_y, cv=5, scoring="accuracy")
print(score.mean())

k = range(1,31)
k_score = []
k_loss = []
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    loss = -cross_val_score(knn, iris_X, iris_y, cv=5, scoring="neg_mean_squared_error") # for regression
    k_loss.append(loss.mean())
    score = cross_val_score(knn, iris_X, iris_y, cv=5, scoring="accuracy") # for classification
    k_score.append(score.mean())
plt.plot(k, k_loss)
plt.show()
