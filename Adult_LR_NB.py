import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from itertools import chain
from matplotlib import pyplot as plt

# 导入数据集
train_set = pd.read_csv('Adult数据集/adult.data',header=None)
train_set.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
test_set = pd.read_csv('Adult数据集/adult.test',header=None)
test_set.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

# 去除字符串数值前面的空格
str_cols=[1,3,5,6,7,8,9,13,14]
for col in str_cols:
    train_set.iloc[:,col]=train_set.iloc[:,col].map(lambda x: x.strip())
    test_set.iloc[:,col]=test_set.iloc[:,col].map(lambda x: x.strip())

# 删除缺失值样本
for i in train_set.columns:
    test_set=test_set[test_set[i]!='?']
    train_set=train_set[train_set[i]!='?']

# #删除fnlgwt列
# train_set.drop('fnlwgt',axis=1,inplace=True)
# test_set.drop('fnlwgt',axis=1,inplace=True)

# 对字符数据进行编码
from sklearn import preprocessing
label_encoder=[] # 放置每一列的encoder
train_encoded_set = np.empty(train_set.shape)
test_encoded_set = np.empty(test_set.shape)
for col in range(train_set.shape[1]):
    encoder=None
    if train_set.iloc[:,col].dtype==object: # 字符型数据
        encoder=preprocessing.LabelEncoder()
        train_encoded_set[:,col]=encoder.fit_transform(train_set.iloc[:,col])
    else:  # 数值型数据
        train_encoded_set[:,col]=train_set.iloc[:,col]
    if test_set.iloc[:,col].dtype==object:
        test_encoded_set[:,col]=encoder.fit_transform(test_set.iloc[:,col])
    else:
        test_encoded_set[:,col]=test_set.iloc[:,col]
    label_encoder.append(encoder)

# 看一下各特征与income的相关系数
data=np.array(train_encoded_set)
train_df=pd.DataFrame(data=data[0:,0:],columns=train_set.columns)
temp=train_df.copy()
corr=temp.corr()
score=corr['income'].sort_values()
print('相关系数:')
print(score)

# 对某些列进行范围缩放
cols=[2,10,11]
data_scalers=[] # 专门用来放置scaler
train_ravel_set = np.copy(train_encoded_set)
test_ravel_set = np.copy(test_encoded_set)
for col in cols:
    data_scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)) 
    train_ravel_set[:,col]=np.ravel(data_scaler.fit_transform(train_ravel_set[:,col].reshape(-1,1)))
    test_ravel_set[:,col]=np.ravel(data_scaler.fit_transform(test_ravel_set[:,col].reshape(-1,1)))
    data_scalers.append(data_scaler)

# 将Income分出
X_train,Y_train=train_ravel_set[:,:-1],train_ravel_set[:,-1]
X_test,Y_test=test_ravel_set[:,:-1],test_ravel_set[:,-1]
# X_train,Y_train=train_encoded_set[:,:-1],train_encoded_set[:,-1]
# X_test,Y_test=test_encoded_set[:,:-1],test_encoded_set[:,-1]

# 划分数据
Y = Y_train.reshape(-1,1)  # 训练集Y
X = X_train  
X = np.hstack([np.ones((len(X), 1)), X])  # 训练集X

y = Y_test.reshape(-1,1)  # 测试集y
x = X_test
x = np.hstack([np.ones((len(x), 1)), x])  # 测试集x

# 训练集的行列数
X_row=X.shape[0]
X_col=X.shape[1]
# 测试集行数
x_row=x.shape[0]

#逻辑回归梯度下降法----------------------------------------------------

# sigmoid函数
def sigmoid(X,theta):
    return 1/(1+np.exp(-X.dot(theta)))

# 损失函数
def cost(X,Y,theta):
    H = sigmoid(X,theta)
    return (1-Y).T.dot(np.log(1-H+1e-5)) - Y.T.dot((np.log(H+1e-5)))

# 梯度下降
y_t = []
def Gradient_descent(X,Y,alpha,maxIter):
    #初始化theta
    np.random.seed(42)
    theta = np.mat(np.random.randn(X_col,1))
    loss = cost(X,Y,theta)
    y_t.append(loss)
    #更新theta
    for i in range(maxIter):
        H = sigmoid(X,theta)
        dtheta = X.T.dot((H - Y))/len(Y)
        theta -= alpha*dtheta
        loss = cost(X,Y,theta)
        y_t.append(loss)
    return theta

theta = Gradient_descent(X,Y,0.001,11000)
print('梯度下降theta:')
print(theta)

# 计算准确率
a=sigmoid(x,theta)
correct=0
for i in range(x_row):
    if a[i]<0.5:
        a[i]=0
    else:
        a[i]=1
    if a[i]==y[i]:
        correct+=1
rate=correct/x_row
print('逻辑回归准确率:',rate)

# 朴素贝叶斯------------------------------------------------

# 按类别划分数据
def seprateByClass(dataset):
  seprate_dict = {}
  info_dict = {}
  for vector in dataset:
      if vector[-1] not in seprate_dict:
          seprate_dict[vector[-1]] = []
          info_dict[vector[-1]] = 0
      seprate_dict[vector[-1]].append(vector)
      info_dict[vector[-1]] +=1
  return seprate_dict,info_dict

train_separated,train_info = seprateByClass(train_encoded_set) #划分好的数据

# 计算每个类别的先验概率(P(yi))
def calulateClassPriorProb(dataset,dataset_info):
  dataset_prior_prob = {}
  sample_sum = len(dataset)
  for class_value, sample_nums in dataset_info.items():
      dataset_prior_prob[class_value] = sample_nums/float(sample_sum)
  return dataset_prior_prob

# 每个类别的先验概率(P(yi))
prior_prob = calulateClassPriorProb(train_encoded_set,train_info)

# 均值
def mean(list):
  list = [float(x) for x in list] #字符串转数字
  return sum(list)/float(len(list))

# 方差
def var(list):
  list = [float(x) for x in list]
  avg = mean(list)
  var = sum([math.pow((x-avg),2) for x in list])/float(len(list)-1)
  return var

# 概率密度函数
def calculateProb(x,mean,var):
    exponent = math.exp(math.pow((x-mean),2)/(-2*var))
    p = (1/math.sqrt(2*math.pi*var))*exponent
    return p

# 计算每个属性的均值和方差
def summarizeAttribute(dataset):
    dataset = np.delete(dataset,-1,axis = 1) # delete label
    summaries = [(mean(attr),var(attr)) for attr in zip(*dataset)] #按列提取
    return summaries

# 按类别提取属性特征 会得到 类别数目*属性数目 组
def summarizeByClass(dataset):
  summarize_by_class = {}
  for classValue, vector in train_separated.items():
      summarize_by_class[classValue] = summarizeAttribute(vector)
  return summarize_by_class

# 按类别提取属性特征
train_Summary_by_class = summarizeByClass(train_encoded_set)

# 计算属于某类的类条件概率(P(x|yi))
def calculateClassProb(input_data,train_Summary_by_class):
  prob = {}
  for class_value, summary in train_Summary_by_class.items():
      prob[class_value] = 1
      for i in range(len(summary)):
        mean,var = summary[i]
        x = input_data[i]
        p = calculateProb(x,mean,var)
        prob[class_value] *=p
  return prob

# 朴素贝叶斯分类器
def bayesianPredictOneSample(input_data):
  classprob_dict = calculateClassProb(input_data,train_Summary_by_class) # 计算属于某类的类条件概率(P(x|yi))
  result = {}
  for class_value,class_prob in classprob_dict.items():
    p = class_prob*prior_prob[class_value]
    result[class_value] = p
  return max(result,key=result.get)

# 计算准确率
save = []
def calculateAccByBeyesian(dataset):
  correct = 0
  for vector in dataset:
      input_data = vector[:-1]
      label = vector[-1]
      result = bayesianPredictOneSample(input_data)
      save.append(result)
      if result == label:
          correct+=1
  return correct/len(dataset)

rate = calculateAccByBeyesian(test_encoded_set)
print("朴素贝叶斯准确率：",rate)
