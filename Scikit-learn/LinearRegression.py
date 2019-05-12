from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# 加载数据 并且分为输入和输出数据
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

# print(data_X.shape)
# 将数据分为训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3)

# 获得模型对象
model = LinearRegression()
# 训练数据
model.fit(train_X, train_y)
# 预测数据
test_Y = model.predict(test_X)
# test_X进行预测，结果与test_y进行比较然后打分
print(model.score(test_X, test_y))
print(model.intercept_)
print(model.coef_)
# print(test_y)
# print(test_Y)
# print(test_Y.shape)
# print(test_y.shape)

# 进行数据标准化
data_X = preprocessing.scale(data_X)
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3)

# 获得模型对象
model = LinearRegression()
# 训练数据
model.fit(train_X, train_y)
# 预测数据
test_Y = model.predict(test_X)
# test_X进行预测，结果与test_y进行比较然后打分
print(model.score(test_X, test_y))
