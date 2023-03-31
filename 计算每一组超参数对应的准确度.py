import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import csv


#读取数据
# Initialize empty lists to store the labels and features
y_train = []
x_train = []

with open('提取有用的特征.csv', 'r') as file:
    csv_reader = csv.reader(file)
    # skip the header row
    next(csv_reader)
    for row in csv_reader:
        label = row[0] # get the label value from the first column
        features = row[1:] # get the feature values from the remaining columns
        # Append the label and features to the respective lists
        y_train.append(label)
        x_train.append(features)

x_test = []
y_test = []
with open('提取有用的特征test.csv', 'r') as file:
    csv_reader = csv.reader(file)
    # skip the header row
    next(csv_reader)
    for row in csv_reader:
        label = row[0] # get the label value from the first column
        features = row[1:] # get the feature values from the remaining columns
        # Append the label and features to the respective lists
        y_test.append(label)
        x_test.append(features)

# 读取包含超参数的CSV文件
df = pd.read_csv('output1.csv')

# 将超参数值分别存储到两个NumPy数组中
gammas = np.array(df['gamma'])
cs = np.array(df['c'])

# 初始化准确度列表
accuracy_list = []

# 循环遍历每一组超参数值
for i in range(len(df)):
    gamma, C = gammas[i], cs[i]

    # 初始化SVM分类器
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)


    clf.fit(x_train, y_train)

    # 计算分类器的准确度并将其添加到列表中
    accuracy = clf.score(x_test, y_test)
    accuracy_list.append(accuracy)

# 将超参数和准确度转换为NumPy数组
accuracy_array = np.array(accuracy_list)
# 打印每个分类器的准确度和超参数值
for gamma, C, accuracy in accuracy_list:
    print('Gamma: {}, C: {}, Accuracy: {:.2f}'.format(gamma, C, accuracy))

# 绘制超参数和准确度的关系图


fig, ax = plt.subplots()
ax.scatter(gammas, cs, c=accuracy_array, cmap='viridis')
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
plt.show()
