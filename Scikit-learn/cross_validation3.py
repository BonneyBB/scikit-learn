# 测试gamma的值不同时的训练效果


from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

loaded_data = load_digits()
X = loaded_data.data
y = loaded_data.target

param_range = np.logspace(-8, -3, 6)
train_loss, test_loss = validation_curve(
    SVC(), X, y, cv=10, param_name="gamma",param_range=param_range, scoring="neg_mean_squared_error")
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, '*-', color="g",
         label="testing")

plt.xlabel("Gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()