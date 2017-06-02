# -*- coding:utf-8 -*-

'''
这个版本得分0.79904。参考 https://www.kaggle.com/fpzh2011/titanic-rf-prediction-0-81818
在版本2的基础上，将FamSize替换为离散的FSizeD；降低决策树深度为5
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import re

# 让DataFrame的row不跨行
pd.set_option('expand_frame_repr', False)

title_pattern = re.compile('[^,]*,\s*([^\.]+)')
def get_title(name):
	'''
	根据name返回title（如Mr, Miss）
	'''
	m = title_pattern.match(name)
	return name if m is None or len(m.group(1)) == 0 else m.group(1)

def merge_title(title):
	'''
	对title进行聚合
	'''
	officer = ('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
	if title in officer:
		return 'Officer'
	royalty = ('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')
	if title in royalty:
		return 'Royalty'
	miss = ('Mlle', 'Ms')
	if title in miss:
		return 'Miss'
	if title == 'Mme':
		return 'Mrs'
	return title

def family_size_d(size):
	if size == 0:
		return 'alone'
	if size > 0 and size < 4:
		return 'small'
	return 'big'

def preprocess(df):
	'''
	预处理数据。补充缺失值，增加新特征字段。
	'''
	df['FamSize'] = df['SibSp'] + df['Parch']
	df['FSizeD'] = df.FamSize.map(family_size_d)
	df['Sex'].replace({'female':0, 'male':1}, inplace=True)
	# title
	df['Title'] = df.Name.map(get_title).map(merge_title)
	#Fare缺失数据处理
	df.Fare.fillna(8.05, inplace=True)
	df['consume'] = (df['Fare'] > 20)
	# age缺失值处理，用title的年龄中位数代替
	title_age = {'Master':4, 'Miss':22, 'Mr':29, 'Mrs':35, 'Officer':49, 'Royalty':39}
	df.Age.fillna(df.Title.map(title_age), inplace=True)
	# 下面3个变量的作用似乎不大
	df['Child'] = (df['Age'] < 18)
	df['orphan'] = ((df['Age'] < 18) & (df['FamSize'] == 0))
	df['HaveFam'] = ((df['Age'] >= 18) & (df['FamSize'] != 0))
	# Embarked
	df.Embarked.fillna('S', inplace=True)
	return df

my_data = pd.read_csv("train.csv", header=0)
my_data = preprocess(my_data)
# Embarked的多个值转换为one-hot。缺失值设为S。
le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(my_data['Embarked'].values)
a = le_embarked.transform(my_data['Embarked'].values)
enc_embarked = OneHotEncoder(sparse=False)
enc_embarked.fit(a.reshape(-1, 1))
embarked = enc_embarked.transform(a.reshape(-1, 1))
# Title的多个值转换为one-hot
le_title = preprocessing.LabelEncoder()
le_title.fit(my_data['Title'].values)
a = le_title.transform(my_data['Title'].values)
enc_title = OneHotEncoder(sparse=False)
enc_title.fit(a.reshape(-1, 1))
title = enc_title.transform(a.reshape(-1, 1))
# FSizeD的多个值转换为one-hot
le_FSizeD = preprocessing.LabelEncoder()
le_FSizeD.fit(my_data['FSizeD'].values)
a = le_FSizeD.transform(my_data['FSizeD'].values)
enc_FSizeD = OneHotEncoder(sparse=False)
enc_FSizeD.fit(a.reshape(-1, 1))
FSizeD = enc_FSizeD.transform(a.reshape(-1, 1))

# 拼接X, y
y = my_data['Survived']
#X = my_data[['Pclass','Sex','Age','SibSp','Parch','Fare', 'FamSize', 'Child', 'orphan', 'HaveFam', 'consume']].values
#X = my_data[['Pclass','Sex','Age','SibSp','Parch','Fare', 'FamSize', 'Child']].values
X = my_data[['Pclass','Sex','Fare', 'Child']].values
X = np.concatenate((X, embarked, title, FSizeD), axis=1)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
kf = KFold(n_splits=5)
from sys import stdout

# 训练
clf = RandomForestClassifier(n_estimators=160, max_depth=5, n_jobs=8)
y_predict = cross_val_predict(clf, X, y, cv=kf, n_jobs=5)
print 'f1', metrics.f1_score(y, y_predict)
print 'accuracy', metrics.accuracy_score(y, y_predict)
clf = clf.fit(X, y)
# y_predict = clf.predict(X)

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
test_data = preprocess(test_data)
a = le_embarked.transform(test_data['Embarked'].values)
embarked = enc_embarked.transform(a.reshape(-1, 1))
a = le_title.transform(test_data['Title'].values)
title = enc_title.transform(a.reshape(-1, 1))
a = le_FSizeD.transform(test_data['FSizeD'].values)
FSizeD = enc_FSizeD.transform(a.reshape(-1, 1))

# 拼接X, y
#X_test = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare', 'FamSize', 'Child', 'orphan', 'HaveFam', 'consume']].values
#X_test = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare', 'FamSize', 'Child']].values
X_test = test_data[['Pclass','Sex','Fare', 'Child']].values
X_test = np.concatenate((X_test, embarked, title, FSizeD), axis=1)
y_test = clf.predict(X_test)

# 输出
for (id, y) in zip(test_data['PassengerId'], y_test):
    print id, y



