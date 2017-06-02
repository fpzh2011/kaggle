# -*- coding:utf-8 -*-

'''
这个版本得分0.76555
对Age的缺失值只用全部数据的frequent值
'''

import pandas as pd

filename = "train.csv"
my_data = pd.read_csv(filename, header=0)

'''
skip_list = ["PassengerId", "Name", "Ticket", "Cabin"]
for col in my_data.columns :
    if col in skip_list :
        continue
    c = pd.value_counts(my_data[col], sort=False)
    print col, "======================"
    print c.sum()
    print c
'''

# sex。将female/male转换为0/1
my_data['Sex'].replace({'female':0, 'male':1}, inplace=True)
# Embarked的多个值转换为one-hot。缺失值作为一个离散值。
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(my_data['Embarked'].values)
a = le.transform(my_data['Embarked'].values)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
enc.fit(a.reshape(-1, 1))
embarked = enc.transform(a.reshape(-1, 1))
# age缺失值处理，用最常出现的年龄代替
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
a = my_data['Age'].values.reshape(-1, 1)
imp.fit(a)
age = imp.transform(a)
#Fare缺失数据处理
imp_f = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp_f.fit(my_data['Fare'].values.reshape(-1, 1))
# 拼接X, y
y = my_data['Survived']
X = my_data[['Pclass','Sex','SibSp','Parch','Fare']].values
X = np.concatenate((X, age, embarked), axis=1)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
from sys import stdout

# 选择树深度15
def select_depth() :
    for depth in range(5, 31, 2) :
        clf = RandomForestClassifier(n_estimators=50, max_depth=depth, n_jobs=8)
        scores = cross_val_score(clf, X, y, cv=kf, n_jobs=5)
        stdout.write(str(depth))
        stdout.write('\t')
        stdout.write(str(scores.mean()))
        stdout.write('\t')
        print scores

# 选择树个数60
def select_tree_count() :
    for tree_count in range(5, 66, 5) :
        clf = RandomForestClassifier(n_estimators=tree_count, max_depth=15, n_jobs=8)
        scores = cross_val_score(clf, X, y, cv=kf, n_jobs=5)
        stdout.write(str(tree_count))
        stdout.write('\t')
        stdout.write(str(scores.mean()))
        stdout.write('\t')
        print scores

#select_depth()
#select_tree_count()

# 训练
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
clf = RandomForestClassifier(n_estimators=60, max_depth=15, n_jobs=8)
clf = clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
# 打印性能指标
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from numpy import logical_and
from numpy import logical_not
'''
print "sum: " , y_predict.sum(), y_test.sum(), y_train.sum(), logical_and(y_predict, logical_not(y_test)).sum()
print "accruacy: ", str(accuracy_score(y_test, y_predict))
print "f1: ", str(f1_score(y_test, y_predict))
print "precision: ", str(precision_score(y_test, y_predict))
print "recall: ", str(recall_score(y_test, y_predict))
'''

# 加载测试数据
test_data = pd.read_csv('test.csv', header=0)
test_data['Sex'].replace({'female':0, 'male':1}, inplace=True)
# Embarked的多个值转换为one-hot。缺失值作为一个离散值。
a = le.transform(test_data['Embarked'].values)
embarked = enc.transform(a.reshape(-1, 1))
# age缺失值处理，用最常出现的年龄代替
a = test_data['Age'].values.reshape(-1, 1)
age = imp.transform(a)
#Fare缺失值
a = test_data['Fare'].values.reshape(-1, 1)
fare = imp_f.transform(a)

# 拼接X, y
X_test = test_data[['Pclass','Sex','SibSp','Parch']].values
X_test = np.concatenate((X_test, fare, age, embarked), axis=1)
y_test = clf.predict(X_test)

# 输出
for (id, y) in zip(test_data['PassengerId'], y_test):
    print id, y


