# Titanic数据分析

```python
# -*- coding:utf-8 -*-

# ipython --pylab

import pandas as pd
import re

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

train = pd.read_csv('train.csv', header=0)
print train.Survived.value_counts() # 各类别数据基本均衡
test = pd.read_csv('test.csv', header=0)
total = pd.concat([test, 
	train[train.columns.copy().drop('Survived')]
	])

# 统计缺失值
print total.isnull().sum()
# 数值数据的基本描述统计
print total.describe()
# 几个主要类别的数据分布
for col in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:
	print col, ' ===========================\n'
	print total[col].value_counts()
# 统计Cabin类别的分布
first_ch = lambda s: s[:1] if isinstance(s,str) else s
print total.Cabin.groupby(total.Cabin.copy().apply(first_ch)).count()
# 统计年龄分布，每10岁一组
bins = pd.cut(total.Age, xrange(0, 101, 10))
print total.Age.groupby(bins).count()
# 统计Fare分布，没50一组
bins = pd.cut(total.Fare, xrange(0, 550, 25))
print total.Fare.groupby(bins).count()

'''
按不同属性统计train，并用图形显示
'''

# Title
total['Title'] = total.Name.map(get_title).map(merge_title)
total.groupby('Title')['Age'].median()
'''
Master      4.0
Miss       22.0
Mr         29.0
Mrs        35.0
Officer    49.0
Royalty    39.0
title_age = {'Master':4, 'Miss':22, 'Mr':29, 'Mrs':35, 'Officer':49, 'Royalty':39}
'''

# Age
bins = pd.cut(train.Age, xrange(0, 101, 5))
by_age = train.groupby([train['Survived'], bins])['PassengerId'].count().unstack(level='Survived')
by_age['ratio'] = by_age[1] / by_age[0] # 生存/死亡
fig, axes = plt.subplots(2,1)
by_age[[0,1]].plot(kind='bar', ax=axes[0])
by_age['ratio'].plot(kind='bar', ax=axes[1])
# 剔除比例超过100%的。剩余数据的生存概率，有一定差异，但效果还需要和其它属性联合来看
by_age['ratio'][by_age['ratio'] < 1].plot(kind='bar')
'''
10岁以下的生存概率明显较大。可以考虑抽出Child这一中间概念。是否需要壮年、老年概念？
'''

# Sex
# 非常明显。女性生存概率明显高于男性
train.groupby(['Survived', 'Sex'])['PassengerId'].count().unstack(level='Survived').plot(kind='bar')

# Pclass
# 舱位等级越高，生存概率越大
train.groupby(['Survived', 'Pclass'])['PassengerId'].count().unstack(level='Survived').plot(kind='bar')

# SibSp & Parch
# 分别看SibSp和Parch，如果有兄弟或父母在船上，生存概率会高一些。
# 但是，可以将SibSp和Parch加起来，效果更明显。即，如果有亲属（无论是兄弟、夫妻或父母）在船上，生存概率的差异更明显。
print train.SibSp.value_counts()
fig, axes = plt.subplots(2,2)
train.groupby(['Survived', 'SibSp'])['PassengerId'].count().unstack(level='Survived').plot(kind='bar', ax=axes[0,0])
train.groupby(['Survived', 'Parch'])['PassengerId'].count().unstack(level='Survived').plot(kind='bar', ax=axes[0,1])
train.groupby(['Survived', train['Parch']+train['SibSp']])['PassengerId'].count().unstack(level='Survived').plot(kind='bar', ax=axes[1,0])
# 年轻而又没有亲属的，生存概率更低
print train[(train.Age > 10) & (train.Age <= 25) & (train['Parch']+train['SibSp'] == 0)].Survived.value_counts()
# 20到60岁，Parch!=0的人，生存概率更高。Parch的差异比SibSp明显一些。
fig, axes = plt.subplots(1,2)
temp = train[(train.Age >= 20) & (train.Age <= 60) & (train['Parch'] != 0)]
bins = pd.cut(temp.Age, xrange(0, 101, 5))
temp.groupby([temp['Survived'], bins])['PassengerId'].count().unstack(level='Survived').plot(kind='bar', ax=axes[0])
temp = train[(train.Age >= 20) & (train.Age <= 60) & (train['SibSp'] != 0)]
bins = pd.cut(temp.Age, xrange(0, 101, 5))
temp.groupby([temp['Survived'], bins])['PassengerId'].count().unstack(level='Survived').plot(kind='bar', ax=axes[1])

# Fare
# 消费超过20（或40）的人，生存概率明显较高
bins = pd.cut(train.Fare, xrange(0, 600, 20))
by_fare = train.groupby([train['Survived'], bins])['PassengerId'].count().unstack(level='Survived')
by_fare['ratio'] = by_fare[1] / by_fare[0] # 生存/死亡
fig, axes = plt.subplots(2,1)
by_fare[[0,1]].plot(kind='bar', ax=axes[0])
by_fare['ratio'].plot(kind='bar', ax=axes[1])

# Cabin
# Cabin的数据缺失比较严重。从已有数据看，BDE的生存概率明显高于其它仓室。
by_cabin = train.groupby([train['Survived'], train.Cabin.copy().apply(first_ch)])['PassengerId'].count().unstack(level='Survived')
by_cabin['ratio'] = by_cabin[1] / by_cabin[0] # 生存/死亡
fig, axes = plt.subplots(2,1)
by_cabin[[0,1]].plot(kind='bar', ax=axes[0])
by_cabin['ratio'].plot(kind='bar', ax=axes[1])

# Embarked
by_embarked = train.groupby(['Survived', 'Embarked'])['PassengerId'].count().unstack(level='Survived')
by_embarked['ratio'] = by_embarked[1] / by_embarked[0] # 生存/死亡
fig, axes = plt.subplots(2,1)
by_embarked[[0,1]].plot(kind='bar', ax=axes[0])
by_embarked['ratio'].plot(kind='bar', ax=axes[1])

'''
缺失值分析
'''
# Fare 1个
print total[total.Fare.isnull()]
# 设置为7
print total[(total.Pclass == 3) & (total.Sex == 'male') & (total.Age >= 60)]
total.Fare.fillna(7, inplace=True)

# Embarked 只有2条缺失。通过其它信息也不好反推乘客从哪里上船。简单设置为frequrent值S，这也大体符合下面的分析。
print total[total.Embarked.isnull()]
print total[(total.Pclass == 1) & (total.Sex == 'female')]['Embarked'].value_counts()
total.Embarked.fillna('S', inplace=True)

# 77%的数据缺失Cabin，不用这个字段

# Age 综合来看，可以通过Pclass来拟合Age
# 缺失Age的，大部分是3等仓
total.groupby([total['Pclass'], total.Age.isnull()])['PassengerId'].count().unstack().plot(kind='bar')
# 缺失Age的，性别差异不是非常明显
total.groupby([total['Sex'], total.Age.isnull()])['PassengerId'].count().unstack().plot(kind='bar')
# 后登船的，年龄缺失多一些。
total.groupby([total['Embarked'], total.Age.isnull()])['PassengerId'].count().unstack().plot(kind='bar')
# 年龄缺失与消费高度相关，其实也可以通过Pclass来看。
bins = pd.cut(total.Fare, xrange(0, 600, 20))
total.groupby([bins, total.Age.isnull()])['PassengerId'].count().unstack().plot(kind='bar')
# 根据Pclass填充缺失的Age
age = total[total.Age.notnull()]
bins = pd.cut(age.Age, xrange(0, 101, 5))
print age.groupby([bins, age['Pclass']])['PassengerId'].count().unstack()
# 根据上面的输出，一等仓缺失年龄设置为40岁，二等舱设置为30岁，三等仓设置为25岁
age_pclass = {1:40, 2:30, 3:25}
total.Age.fillna(total.Pclass.map(age_pclass), inplace=True)

'''
kernel

https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
对数据还是要观察观察再观察，思考思考再思考。比如，从Name中抽取出Title。
如果信息过于分散，可以尝试适度合并（阈值的选择要结合数据分布、对比）。比如Title、FsizeD、Child。合并相当于创建学习过程中的中间概念，而算法自动学习这些中间概念可能比较困难。
对缺失值可以用拟合的方法，但是要对拟合结果进行检验。
不要以上来就急着做模型、跑结果，还是要先多看看数据。
为什么Pclass的重要性降低了？因为三等舱的Child数量更多。从以下代码可见：
'''
bins = pd.cut(total.Age, xrange(0, 101, 5))
a = total.groupby([total['Pclass'], bins, total['Sex']])['PassengerId'].count()
fig, axes = plt.subplots(1,3,sharey=True)
axes[0].set_ylabel('count')
a[1].unstack().plot(kind='bar', ax=axes[0], title='Pclass 1')
a[2].unstack().plot(kind='bar', ax=axes[1], title='Pclass 2')
a[3].unstack().plot(kind='bar', ax=axes[2], title='Pclass 3')

total.groupby(['Pclass', 'Title'])['PassengerId'].count().unstack().plot(kind='bar')


'''
https://www.kaggle.com/omarelgabry/a-journey-through-titanic
https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
pytanic 有大量seaborn图形操作
https://www.kaggle.com/headsortails/pytanic
https://www.kaggle.com/vincentlugat/titanic/titanic-rf-prediction-0-81818
'''

