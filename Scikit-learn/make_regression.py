from sklearn import datasets
import matplotlib.pyplot as plt

data_X, data_y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1,noise=10)
plt.scatter(data_X, data_y)
plt.show()
print(data_X)