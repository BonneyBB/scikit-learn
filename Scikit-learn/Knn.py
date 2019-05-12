from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载sklearn中的鸢尾花数据集
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X.shape) 150 * 4
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
test_Y = knn.predict(test_X)
print(test_Y)
print(test_y)
